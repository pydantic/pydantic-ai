"""An in-memory SQLite stand-in for an asyncpg pool, for the Postgres store unit tests.

There is no in-process Postgres, so the unit suites run `PostgresMediaStore`,
`PostgresStepStore` and `PostgresSpendStore` against SQLite after rewriting the few
Postgres-only spellings the stores use: `$n` parameters, identity columns,
`= ANY($n::bigint[])` or `::text[]`, `now()`, `clock_timestamp()` and `+ $n::interval`, and the
locking and isolation clauses a single-writer database has no use for. Both clocks read `SqlitePool.now`,
which a test moves by hand, as ISO 8601 text in UTC, which orders the way the instants do.
SQLite shares the semantics the stores rely on (`ON CONFLICT ... DO NOTHING`,
`RETURNING`, partial unique indexes, `EXCLUDED`), so reads, writes, idempotency
and retention run for real. What only a Postgres server shows (identifier
folding, NUL bytes in `TEXT`, racing `CREATE TABLE IF NOT EXISTS` across
sessions) stays in `src/pydantic_ai_harness/integration_tests/postgres`.
"""

from __future__ import annotations

import json
import re
import sqlite3
import zlib
from collections.abc import AsyncGenerator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta

import anyio
import anyio.lowlevel

from pydantic_ai_harness.media import PostgresConnection

_IDENTITY = 'BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY'
_ANY_ARRAY = re.compile(r'(\w+) = ANY\(\$(\d+)::(?:bigint|text)\[\]\)')
_NOW = re.compile(r'(?:now|clock_timestamp)\(\)')
_NOW_PLUS = re.compile(r'(?:now|clock_timestamp)\(\) \+ \$(\d+)::interval')
_PARAMETER = re.compile(r'\$(\d+)')


def _to_sqlite(query: str) -> str:
    query = query.replace(_IDENTITY, 'INTEGER PRIMARY KEY AUTOINCREMENT')
    query = _ANY_ARRAY.sub(r'\1 IN (SELECT value FROM json_each(?\2))', query)
    query = _NOW.sub('pg_now()', _NOW_PLUS.sub(r'pg_now_plus($\1)', query))
    query = query.replace(' FOR UPDATE SKIP LOCKED', '')
    query = query.replace('SET TRANSACTION ISOLATION LEVEL READ COMMITTED', 'SELECT 1')
    return _PARAMETER.sub(r'?\1', query)


def _to_sqlite_value(value: object) -> object:
    # The only lists the stores bind are arrays, which `_to_sqlite` reads with `json_each`.
    if isinstance(value, list):
        return json.dumps(value)
    # The only intervals are added to `now()`, which `pg_now_plus` takes in microseconds.
    if isinstance(value, timedelta):
        return value // timedelta(microseconds=1)
    return value


def _timestamp(instant: datetime) -> str:
    return instant.astimezone(UTC).isoformat(timespec='microseconds')


def _hashtext(text: str) -> int:
    return zlib.crc32(text.encode())


def _advisory_xact_lock(key: int) -> None:
    """SQLite has a single writer, so there is no other session to lock out."""


class SqliteConnection:
    """A `PostgresConnection` over one SQLite connection.

    `statements` records every query as the store wrote it, before rewriting.
    """

    def __init__(self, db: sqlite3.Connection, statements: list[str]) -> None:
        self._db = db
        self._statements = statements

    @asynccontextmanager
    async def _transaction(self) -> AsyncGenerator[object]:
        self._db.execute('BEGIN')
        # The connection's context manager commits on success and rolls back on error.
        with self._db:
            yield None

    def transaction(self) -> AbstractAsyncContextManager[object]:
        return self._transaction()

    async def execute(self, query: str, *args: object) -> object:
        await self.fetch(query, *args)
        return None

    async def fetchval(self, query: str, *args: object) -> object:
        row = await self.fetchrow(query, *args)
        return None if row is None else row[0]

    async def fetchrow(self, query: str, *args: object) -> Sequence[object] | None:
        rows = await self.fetch(query, *args)
        return rows[0] if rows else None

    async def fetch(self, query: str, *args: object) -> Sequence[Sequence[object]]:
        # Yield like a network round trip would, so concurrent callers interleave.
        await anyio.lowlevel.checkpoint()
        self._statements.append(query)
        return self._db.execute(_to_sqlite(query), [_to_sqlite_value(arg) for arg in args]).fetchall()


class SqlitePool:
    """A `PostgresPool` with a single connection, like `asyncpg.create_pool(max_size=1)`.

    `acquire` waits while the connection is out, so a store that held two
    connections at once would hang instead of passing.
    """

    def __init__(self) -> None:
        self.db = sqlite3.connect(':memory:', isolation_level=None)
        self.db.create_function('hashtext', 1, _hashtext)
        self.db.create_function('pg_advisory_xact_lock', 1, _advisory_xact_lock)
        self.db.create_function('pg_now', 0, lambda: _timestamp(self.now))
        self.db.create_function('pg_now_plus', 1, self._now_plus)
        self.db.create_function('to_regclass', 1, self._regclass)
        self.now = datetime(2026, 8, 4, 12, tzinfo=UTC)
        """What `now()` returns. Move it to make a horizon pass."""
        self.statements: list[str] = []
        self._slot = anyio.Lock()

    def _now_plus(self, microseconds: int | None) -> str | None:
        """`now() + interval`, where a `NULL` interval gives `NULL` as it does in Postgres."""
        return None if microseconds is None else _timestamp(self.now + timedelta(microseconds=microseconds))

    def _regclass(self, name: str) -> str | None:
        """The table's name if it exists, as Postgres's `to_regclass`, else `NULL`."""
        found = self.db.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", [name]).fetchone()
        return None if found is None else str(found[0])

    @asynccontextmanager
    async def _acquire(self) -> AsyncGenerator[PostgresConnection]:
        async with self._slot:
            yield SqliteConnection(self.db, self.statements)

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        return self._acquire()

    async def count_rows(self, table: str) -> int:
        async with self.acquire() as connection:
            count = await connection.fetchval(f'SELECT count(*) FROM {table}')
        assert isinstance(count, int)
        return count

    def close(self) -> None:
        self.db.close()

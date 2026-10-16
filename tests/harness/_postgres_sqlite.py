"""An in-memory SQLite stand-in for an asyncpg pool, for the Postgres store unit tests.

There is no in-process Postgres, so the unit suites run `PostgresMediaStore` and
`PostgresStepStore` against SQLite after rewriting the few Postgres-only spellings
the stores use: `$n` parameters, identity columns, and `= ANY($n::bigint[])`.
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

import anyio
import anyio.lowlevel

from pydantic_ai_harness.media import PostgresConnection

_IDENTITY = 'BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY'
_ANY_BIGINT_ARRAY = re.compile(r'(\w+) = ANY\(\$(\d+)::bigint\[\]\)')
_PARAMETER = re.compile(r'\$(\d+)')


def _to_sqlite(query: str) -> str:
    query = query.replace(_IDENTITY, 'INTEGER PRIMARY KEY AUTOINCREMENT')
    query = _ANY_BIGINT_ARRAY.sub(r'\1 IN (SELECT value FROM json_each(?\2))', query)
    return _PARAMETER.sub(r'?\1', query)


def _to_sqlite_value(value: object) -> object:
    # The only list the stores bind is a `bigint[]`, which `_to_sqlite` reads with `json_each`.
    return json.dumps(value) if isinstance(value, list) else value


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
        self.statements: list[str] = []
        self._slot = anyio.Lock()

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

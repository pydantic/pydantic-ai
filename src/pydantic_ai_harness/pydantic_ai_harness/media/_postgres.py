"""PostgreSQL `MediaStore` over an asyncpg-compatible caller-owned pool.

The store never imports `asyncpg`: it depends only on the `PostgresPool` and
`PostgresConnection` protocols below, so the harness carries no database
driver dependency and the application owns the pool's lifecycle.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

import anyio

from pydantic_ai_harness.media._store import (
    _EMPTY_CONTEXT,  # pyright: ignore[reportPrivateUsage]
    MediaContext,
    PublicUrlResolver,
    _coerce_metadata_mapping,  # pyright: ignore[reportPrivateUsage]
    _resolve_public_url,  # pyright: ignore[reportPrivateUsage]
    media_uri_for,
    parse_media_uri,
)

# Postgres truncates identifiers past 63 bytes, which would let two distinct
# names collide on the same table. Lowercase only: the interpolated identifier
# is unquoted, so Postgres folds it to lowercase and `'Media'` would share a
# table with `'media'`.
_TABLE_RE = re.compile(r'[a-z_][a-z0-9_]{0,62}')


@runtime_checkable
class PostgresConnection(Protocol):
    """The acquired asyncpg-compatible connection surface used by the store."""

    def transaction(self) -> AbstractAsyncContextManager[object]:
        """Return an async transaction context manager."""
        ...  # pragma: no cover

    async def execute(self, query: str, *args: object) -> object:
        """Execute a statement."""
        ...  # pragma: no cover

    async def fetchval(self, query: str, *args: object) -> object:
        """Return the first column of the first row, or `None`."""
        ...  # pragma: no cover

    async def fetchrow(self, query: str, *args: object) -> Sequence[object] | None:
        """Return the first row, or `None`."""
        ...  # pragma: no cover

    async def fetch(self, query: str, *args: object) -> Sequence[Sequence[object]]:
        """Return all rows."""
        ...  # pragma: no cover


@runtime_checkable
class PostgresPool(Protocol):
    """The asyncpg-compatible pool surface used by the Postgres stores."""

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire one connection for an operation or transaction."""
        ...  # pragma: no cover


class PostgresMediaStore:
    """PostgreSQL store. One row per blob in a `media` table keyed by sha256 hex.

    Takes a caller-owned asyncpg-compatible pool; the store never closes it.

    The table layout is:

    ```sql
    CREATE TABLE IF NOT EXISTS media (
        sha256 TEXT PRIMARY KEY,
        media_type TEXT,
        bytes BYTEA NOT NULL,
        size_bytes BIGINT NOT NULL,
        metadata TEXT
    );
    ```

    `ON CONFLICT (sha256) DO NOTHING` makes writes idempotent -- the second
    `put` with the same content is a no-op, not an overwrite. `metadata` is
    stored as JSON of `context.metadata` (an empty mapping is stored as
    `'{}'`) and read back via `get_metadata(uri)`.

    A blob is one `BYTEA` value, which Postgres caps at 1 GB.

    There is no `key_strategy=` parameter: the digest is the primary key, so a
    user-chosen storage key would either break dedup or be a no-op. Use
    `table=` if you need a non-default table name.
    """

    def __init__(
        self,
        pool: PostgresPool,
        *,
        table: str = 'media',
        public_url: PublicUrlResolver | None = None,
    ) -> None:
        if not _TABLE_RE.fullmatch(table):
            raise ValueError(f'invalid table name: {table!r}')
        self._pool = pool
        self._table = table
        self._public_url_resolver = public_url
        self._schema_ready = False
        self._schema_lock = anyio.Lock()

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as connection, connection.transaction():
                # `CREATE TABLE IF NOT EXISTS` is not race-free in Postgres: concurrent
                # callers can collide on `pg_type`'s unique index.
                await connection.fetchval('SELECT pg_advisory_xact_lock(hashtext($1))', self._table)
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._table} ('
                    'sha256 TEXT PRIMARY KEY, '
                    'media_type TEXT, '
                    'bytes BYTEA NOT NULL, '
                    'size_bytes BIGINT NOT NULL, '
                    'metadata TEXT)'
                )
            self._schema_ready = True

    async def put(self, data: bytes, *, context: MediaContext = _EMPTY_CONTEXT) -> str:
        uri = media_uri_for(data)
        digest = parse_media_uri(uri)
        await self._ensure_schema()
        async with self._pool.acquire() as connection, connection.transaction():
            await connection.execute(
                f'INSERT INTO {self._table} (sha256, media_type, bytes, size_bytes, metadata) '
                'VALUES ($1, $2, $3, $4, $5) ON CONFLICT (sha256) DO NOTHING',
                digest,
                context.media_type,
                data,
                len(data),
                json.dumps(dict(context.metadata)),
            )
        return uri

    async def get(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bytes:
        digest = parse_media_uri(uri)
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(f'SELECT bytes FROM {self._table} WHERE sha256 = $1', digest)
        if row is None:
            raise FileNotFoundError(f'media not found: {digest}')
        data = row[0]
        if not isinstance(data, bytes):
            raise ValueError(f'media row for {digest} has wrong types')
        return data

    async def exists(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> bool:
        digest = parse_media_uri(uri)
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(f'SELECT 1 FROM {self._table} WHERE sha256 = $1', digest)
        return row is not None

    async def public_url(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> str | None:
        return await _resolve_public_url(self._public_url_resolver, uri, context)

    async def get_metadata(self, uri: str, *, context: MediaContext = _EMPTY_CONTEXT) -> Mapping[str, str]:
        digest = parse_media_uri(uri)
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(f'SELECT metadata FROM {self._table} WHERE sha256 = $1', digest)
        if row is None:
            raise FileNotFoundError(f'media not found: {digest}')
        value = row[0]
        if not isinstance(value, str):
            raise ValueError(f'media row for {digest} has wrong types')
        return _coerce_metadata_mapping(json.loads(value))

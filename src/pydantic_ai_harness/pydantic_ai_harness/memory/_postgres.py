"""PostgreSQL memory store over an asyncpg-compatible caller-owned pool."""

from __future__ import annotations

import re
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from typing import Protocol, runtime_checkable

import anyio

from pydantic_ai_harness.memory._store import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchResult,
    lexical_search,
    validate_store_path,
    validate_store_prefix,
)

_TABLE_RE = re.compile(r'[A-Za-z_][A-Za-z0-9_]{0,51}')


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
    """The asyncpg-compatible pool surface used by `PostgresMemoryStore`."""

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        """Acquire one connection for an operation or transaction."""
        ...  # pragma: no cover


class PostgresMemoryStore:
    """Transactional PostgreSQL memory store with CAS and operation receipts."""

    def __init__(self, pool: PostgresPool, *, table: str = 'agent_memory') -> None:
        if not _TABLE_RE.fullmatch(table):
            raise ValueError(f'invalid table name: {table!r}')
        self._pool = pool
        self._table = table
        self._operations_table = f'{table}_operations'
        self._version_sequence = f'{table}_versions'
        self._metadata_table = f'{table}_metadata'
        self._schema_ready = False
        self._schema_lock = anyio.Lock()

    async def _ensure_schema(self) -> None:
        if self._schema_ready:
            return
        async with self._schema_lock:
            if self._schema_ready:
                return
            async with self._pool.acquire() as connection, connection.transaction():
                await connection.fetchval('SELECT pg_advisory_xact_lock(hashtext($1))', self._metadata_table)
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._table} ('
                    'path TEXT PRIMARY KEY, content TEXT NOT NULL, '
                    'version BIGINT NOT NULL DEFAULT 1, last_operation_id TEXT)'
                )
                await connection.execute(
                    f'ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 1'
                )
                await connection.execute(f'ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS last_operation_id TEXT')
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._operations_table} ('
                    'id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, version TEXT, '
                    'existed BOOLEAN NOT NULL, completed BOOLEAN NOT NULL)'
                )
                await connection.execute(f'CREATE SEQUENCE IF NOT EXISTS {self._version_sequence} MINVALUE 0 START 0')
                await connection.execute(
                    f'CREATE TABLE IF NOT EXISTS {self._metadata_table} ('
                    'id BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id), versions_initialized BOOLEAN NOT NULL)'
                )
                initialized = await connection.fetchval(
                    f'INSERT INTO {self._metadata_table} (id, versions_initialized) VALUES (TRUE, TRUE) '
                    'ON CONFLICT (id) DO NOTHING RETURNING id'
                )
                if initialized is not None:
                    await connection.execute(f"UPDATE {self._table} SET version = nextval('{self._version_sequence}')")
            self._schema_ready = True

    async def _get_operation(self, connection: PostgresConnection, operation: MemoryOperation) -> MemoryMutation | None:
        row = await connection.fetchrow(
            f'SELECT fingerprint, version, existed, completed FROM {self._operations_table} WHERE id = $1',
            operation.id,
        )
        if row is None:
            return None
        if str(row[0]) != operation.fingerprint:
            raise MemoryOperationConflictError(f'operation id {operation.id!r} was reused with different arguments')
        if not bool(row[3]):  # pragma: no cover - uncommitted reservations are invisible
            return None
        return MemoryMutation(
            version=str(row[1]) if row[1] is not None else None,
            replayed=True,
            existed=bool(row[2]),
        )

    async def _reserve_operation(
        self, connection: PostgresConnection, operation: MemoryOperation
    ) -> MemoryMutation | None:
        inserted = await connection.fetchval(
            f'INSERT INTO {self._operations_table} (id, fingerprint, version, existed, completed) '
            'VALUES ($1, $2, NULL, FALSE, FALSE) ON CONFLICT (id) DO NOTHING RETURNING id',
            operation.id,
            operation.fingerprint,
        )
        if inserted is not None:
            return None
        receipt = await self._get_operation(connection, operation)
        if receipt is None:  # pragma: no cover - the conflicting transaction completes before this statement resumes
            raise RuntimeError(f'operation {operation.id!r} did not produce a committed receipt')
        return receipt

    async def _complete_operation(
        self, connection: PostgresConnection, operation: MemoryOperation, mutation: MemoryMutation
    ) -> None:
        await connection.execute(
            f'UPDATE {self._operations_table} SET version = $2, existed = $3, completed = TRUE WHERE id = $1',
            operation.id,
            mutation.version,
            mutation.existed,
        )

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        validate_store_path(path)
        if max_chars <= 0:
            raise ValueError('max_chars must be positive')
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                f'SELECT left(content, $2), version, last_operation_id, length(content) '
                f'FROM {self._table} WHERE path = $1',
                path,
                max_chars,
            )
        if row is None:
            return None
        return MemoryFile(
            content=str(row[0]),
            version=str(row[1]),
            operation_id=str(row[2]) if row[2] is not None else None,
            truncated=int(str(row[3])) > max_chars,
        )

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            return await self._get_operation(connection, operation)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        await self._ensure_schema()
        async with self._pool.acquire() as connection, connection.transaction():
            if operation is not None and (receipt := await self._reserve_operation(connection, operation)) is not None:
                return receipt
            if expected_version is None:
                row = await connection.fetchrow(
                    f'INSERT INTO {self._table} (path, content, version, last_operation_id) '
                    f"VALUES ($1, $2, nextval('{self._version_sequence}'), $3) "
                    'ON CONFLICT (path) DO NOTHING RETURNING version',
                    path,
                    content,
                    operation.id if operation else None,
                )
                existed = False
            else:
                row = await connection.fetchrow(
                    f"UPDATE {self._table} SET content = $2, version = nextval('{self._version_sequence}'), "
                    'last_operation_id = $3 '
                    'WHERE path = $1 AND version::TEXT = $4 RETURNING version',
                    path,
                    content,
                    operation.id if operation else None,
                    expected_version,
                )
                existed = True
            if row is None:
                raise MemoryConflictError(f'memory path {path!r} changed before it could be written')
            mutation = MemoryMutation(version=str(row[0]), replayed=False, existed=existed)
            if operation is not None:
                await self._complete_operation(connection, operation, mutation)
            return mutation

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        await self._ensure_schema()
        async with self._pool.acquire() as connection, connection.transaction():
            if operation is not None and (receipt := await self._reserve_operation(connection, operation)) is not None:
                return receipt
            await connection.fetchval(f"SELECT nextval('{self._version_sequence}')")
            if expected_version is None:
                exists = await connection.fetchval(f'SELECT EXISTS(SELECT 1 FROM {self._table} WHERE path = $1)', path)
                if bool(exists):
                    raise MemoryConflictError(f'memory path {path!r} changed before it could be deleted')
                existed = False
            else:
                row = await connection.fetchrow(
                    f'DELETE FROM {self._table} WHERE path = $1 AND version::TEXT = $2 RETURNING version',
                    path,
                    expected_version,
                )
                if row is None:
                    raise MemoryConflictError(f'memory path {path!r} changed before it could be deleted')
                existed = True
            mutation = MemoryMutation(version=None, replayed=False, existed=existed)
            if operation is not None:
                await self._complete_operation(connection, operation, mutation)
            return mutation

    async def list_paths(self, prefix: str = '', *, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError('limit must be positive')
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f'SELECT path FROM {self._table} WHERE starts_with(path, $1) ORDER BY path LIMIT $2', prefix, limit
            )
        return [str(row[0]) for row in rows]

    async def search(
        self,
        prefix: str,
        query: str,
        *,
        limit: int,
        max_files: int,
        max_chars: int,
        max_file_chars: int,
    ) -> MemorySearchResult:
        validate_store_prefix(prefix)
        if not query.split() or limit <= 0 or max_files <= 0 or max_chars <= 0 or max_file_chars <= 0:
            return MemorySearchResult(matches=[], scanned=0, truncated=False)
        await self._ensure_schema()
        async with self._pool.acquire() as connection:
            rows = await connection.fetch(
                f'SELECT path, left(content, $2), length(content) FROM {self._table} '
                'WHERE starts_with(path, $1) ORDER BY path LIMIT $3',
                prefix,
                max_file_chars,
                max_files + 1,
            )
        result = lexical_search(
            [(str(row[0]), str(row[1])) for row in rows],
            query,
            limit=limit,
            max_files=max_files,
            max_chars=max_chars,
            score_prefix=prefix,
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or any(int(str(row[2])) > max_file_chars for row in rows),
        )

"""SQLite-backed `MemoryStore`."""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

import anyio.to_thread

from pydantic_ai_harness.memory._store import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchResult,
    decode_metadata,
    encode_metadata,
    lexical_search,
    validate_store_path,
    validate_store_prefix,
)

_SQLITE_SETUP_LOCK = threading.RLock()
_T = TypeVar('_T')


def _enable_wal(connection: sqlite3.Connection) -> None:
    with _SQLITE_SETUP_LOCK:
        try:
            connection.execute('PRAGMA journal_mode = WAL')
        except sqlite3.OperationalError as error:  # pragma: no cover - requires a lock held by another process
            if not any(reason in str(error).lower() for reason in ('busy', 'locked')):
                raise
            # Journal mode is an optimization; transactions remain the consistency boundary.


_SQLITE_MEMORY_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS memory_files ('
    'path TEXT PRIMARY KEY, content TEXT NOT NULL, version INTEGER NOT NULL, last_operation_id TEXT, write_metadata TEXT)'
)
_SQLITE_OPERATIONS_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS memory_operations ('
    'id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, version TEXT, existed INTEGER NOT NULL)'
)
_SQLITE_METADATA_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS memory_metadata (id INTEGER PRIMARY KEY CHECK (id = 1), generation INTEGER NOT NULL)'
)


def _prefix_end(prefix: str) -> str:
    """An upper bound for the paths starting with `prefix`, so SQLite can seek the path index for the range.

    Paths are ASCII (`validate_store_path`), so every one starting with `prefix` sorts below `prefix` plus
    the highest code point under SQLite's default binary collation.
    """
    return f'{prefix}\U0010ffff'


class SqliteMemoryStore:
    """SQLite-backed store with transactional CAS, operation receipts, and write metadata."""

    def __init__(
        self,
        *,
        database: str | Path | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> None:
        if (database is None) == (connection is None):
            raise ValueError('provide exactly one of `database=` or `connection=`')
        if database is not None and str(database) in ('', ':memory:'):
            raise ValueError(
                'an in-memory SQLite database does not work with per-call connections -- '
                'use `InMemoryStore`, or pass a caller-owned `connection=`'
            )
        self._database = database
        self._connection = connection
        self._schema_ready = False
        self._thread_lock = threading.RLock()

    def _connect(self) -> tuple[sqlite3.Connection, bool]:
        if self._connection is not None:
            connection = self._connection
            owned = False
        else:
            assert self._database is not None
            connection = sqlite3.connect(self._database, timeout=30, check_same_thread=False)
            owned = True
        if not owned and connection.in_transaction:
            raise RuntimeError('caller-owned SQLite connection must be idle before a memory operation')
        try:
            connection.execute('PRAGMA busy_timeout = 30000')
            if owned:
                _enable_wal(connection)
            if not self._schema_ready:
                connection.execute('BEGIN IMMEDIATE')
                connection.execute(_SQLITE_MEMORY_SCHEMA)
                columns = {str(row[1]) for row in connection.execute('PRAGMA table_info(memory_files)').fetchall()}
                if 'version' not in columns:
                    connection.execute('ALTER TABLE memory_files ADD COLUMN version INTEGER NOT NULL DEFAULT 1')
                if 'last_operation_id' not in columns:
                    connection.execute('ALTER TABLE memory_files ADD COLUMN last_operation_id TEXT')
                if 'write_metadata' not in columns:
                    connection.execute('ALTER TABLE memory_files ADD COLUMN write_metadata TEXT')
                connection.execute(_SQLITE_OPERATIONS_SCHEMA)
                connection.execute(_SQLITE_METADATA_SCHEMA)
                connection.execute(
                    'INSERT OR IGNORE INTO memory_metadata(id, generation) '
                    'SELECT 1, COALESCE(MAX(version), 0) FROM memory_files'
                )
                connection.commit()
                self._schema_ready = True
            return connection, owned
        except BaseException:
            connection.rollback()
            if owned:
                connection.close()
            raise

    def _run(self, operation: Callable[[sqlite3.Connection], _T], *, immediate: bool = False) -> _T:
        with self._thread_lock:
            connection, owned = self._connect()
            try:
                if immediate:
                    connection.execute('BEGIN IMMEDIATE')
                result = operation(connection)
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise
            finally:
                if owned:
                    connection.close()

    def _get_operation(self, connection: sqlite3.Connection, operation: MemoryOperation) -> MemoryMutation | None:
        row = connection.execute(
            'SELECT fingerprint, version, existed FROM memory_operations WHERE id = ?', (operation.id,)
        ).fetchone()
        if row is None:
            return None
        if str(row[0]) != operation.fingerprint:
            raise MemoryOperationConflictError(f'operation id {operation.id!r} was reused with different arguments')
        return MemoryMutation(
            version=str(row[1]) if row[1] is not None else None,
            replayed=True,
            existed=bool(row[2]),
        )

    def _next_generation(self, connection: sqlite3.Connection) -> int:
        row = connection.execute(
            'UPDATE memory_metadata SET generation = generation + 1 WHERE id = 1 RETURNING generation'
        ).fetchone()
        assert row is not None
        return int(row[0])

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        validate_store_path(path)
        if max_chars <= 0:
            raise ValueError('max_chars must be positive')

        def op(connection: sqlite3.Connection) -> MemoryFile | None:
            row = connection.execute(
                'SELECT substr(content, 1, ?), version, last_operation_id, length(content), write_metadata '
                'FROM memory_files WHERE path = ?',
                (max_chars, path),
            ).fetchone()
            if row is None:
                return None
            return MemoryFile(
                content=str(row[0]),
                version=str(row[1]),
                operation_id=str(row[2]) if row[2] is not None else None,
                truncated=int(row[3]) > max_chars,
                metadata=decode_metadata(row[4]),
            )

        return await anyio.to_thread.run_sync(self._run, op)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        def run() -> MemoryMutation | None:
            def op(connection: sqlite3.Connection) -> MemoryMutation | None:
                return self._get_operation(connection, operation)

            return self._run(op)

        return await anyio.to_thread.run_sync(run)

    def _write(
        self,
        connection: sqlite3.Connection,
        path: str,
        content: str,
        expected_version: str | None,
        operation: MemoryOperation | None,
        metadata: str | None,
    ) -> MemoryMutation:
        if operation is not None and (receipt := self._get_operation(connection, operation)) is not None:
            return receipt
        row = connection.execute('SELECT version FROM memory_files WHERE path = ?', (path,)).fetchone()
        current = str(row[0]) if row is not None else None
        if current != expected_version:
            raise MemoryConflictError(f'memory path {path!r} changed before it could be written')
        version = str(self._next_generation(connection))
        operation_id = operation.id if operation else None
        if current is None:
            connection.execute(
                'INSERT INTO memory_files(path, content, version, last_operation_id, write_metadata) VALUES (?, ?, ?, ?, ?)',
                (path, content, int(version), operation_id, metadata),
            )
        else:
            cursor = connection.execute(
                'UPDATE memory_files SET content = ?, version = ?, last_operation_id = ?, write_metadata = ? '
                'WHERE path = ? AND version = ?',
                (content, int(version), operation_id, metadata, path, int(current)),
            )
            if cursor.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE prevents an intervening writer
                raise MemoryConflictError(f'memory path {path!r} changed before it could be written')
        mutation = MemoryMutation(version=version, replayed=False, existed=current is not None)
        if operation is not None:
            connection.execute(
                'INSERT INTO memory_operations(id, fingerprint, version, existed) VALUES (?, ?, ?, ?)',
                (operation.id, operation.fingerprint, version, int(current is not None)),
            )
        return mutation

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        """Create or replace `path` if its version equals `expected_version`, recording no metadata."""
        return await self.write_with_metadata(
            path, content, expected_version=expected_version, metadata=None, operation=operation
        )

    async def write_with_metadata(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        metadata: Mapping[str, str] | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        encoded = encode_metadata(metadata)
        return await anyio.to_thread.run_sync(
            lambda: self._run(
                lambda connection: self._write(connection, path, content, expected_version, operation, encoded),
                immediate=True,
            )
        )

    def _delete(
        self,
        connection: sqlite3.Connection,
        path: str,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        if operation is not None and (receipt := self._get_operation(connection, operation)) is not None:
            return receipt
        row = connection.execute('SELECT version FROM memory_files WHERE path = ?', (path,)).fetchone()
        current = str(row[0]) if row is not None else None
        if current != expected_version:
            raise MemoryConflictError(f'memory path {path!r} changed before it could be deleted')
        existed = current is not None
        self._next_generation(connection)
        if current is not None:
            cursor = connection.execute('DELETE FROM memory_files WHERE path = ? AND version = ?', (path, int(current)))
            if cursor.rowcount != 1:  # pragma: no cover - BEGIN IMMEDIATE prevents an intervening writer
                raise MemoryConflictError(f'memory path {path!r} changed before it could be deleted')
        mutation = MemoryMutation(version=None, replayed=False, existed=existed)
        if operation is not None:
            connection.execute(
                'INSERT INTO memory_operations(id, fingerprint, version, existed) VALUES (?, ?, NULL, ?)',
                (operation.id, operation.fingerprint, int(existed)),
            )
        return mutation

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        return await anyio.to_thread.run_sync(
            lambda: self._run(
                lambda connection: self._delete(connection, path, expected_version, operation),
                immediate=True,
            )
        )

    async def list_paths(self, prefix: str = '', *, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError('limit must be positive')

        def op(connection: sqlite3.Connection) -> list[str]:
            rows = connection.execute(
                'SELECT path FROM memory_files WHERE path >= ? AND path < ? ORDER BY path LIMIT ?',
                (prefix, _prefix_end(prefix), limit),
            ).fetchall()
            return [str(row[0]) for row in rows]

        return await anyio.to_thread.run_sync(self._run, op)

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

        def op(connection: sqlite3.Connection) -> list[tuple[str, str, int]]:
            # Paths in a deeper scope nested under `prefix` belong to another namespace or agent.
            rows = connection.execute(
                'SELECT path, substr(content, 1, ?), length(content) FROM memory_files '
                'WHERE path >= ? AND path < ? AND instr(substr(path, length(?) + 1), ?) = 0 '
                'ORDER BY path LIMIT ?',
                (max_file_chars, prefix, _prefix_end(prefix), prefix, '/', max_files + 1),
            ).fetchall()
            return [(str(row[0]), str(row[1]), int(row[2])) for row in rows]

        rows = await anyio.to_thread.run_sync(self._run, op)
        result = await lexical_search(
            [(path, content) for path, content, _ in rows],
            query,
            limit=limit,
            max_files=max_files,
            max_chars=max_chars,
            score_prefix=prefix,
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or any(length > max_file_chars for _, _, length in rows),
        )

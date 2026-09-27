"""Versioned, path-addressed persistence backends for the `Memory` capability."""

from __future__ import annotations

import bisect
import heapq
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, TypeVar, runtime_checkable

import anyio
import anyio.to_thread

_VALID_SEGMENT_RE = re.compile(r'[A-Za-z0-9_.-]{1,200}')
_JOURNAL_NAME = '.memory-store.sqlite3'
_FILE_RECOVERY_BATCH_SIZE = 256
_SQLITE_SETUP_LOCK = threading.RLock()
_T = TypeVar('_T')


@dataclass(frozen=True)
class MemoryFile:
    """One memory file and its opaque compare-and-set version."""

    content: str
    version: str
    operation_id: str | None
    truncated: bool


@dataclass(frozen=True)
class MemoryOperation:
    """Stable identity and argument fingerprint for an idempotent mutation."""

    id: str
    fingerprint: str


@dataclass(frozen=True)
class MemoryMutation:
    """Result of a memory write or delete."""

    version: str | None
    replayed: bool
    existed: bool


@dataclass(frozen=True)
class MemorySearchMatch:
    """One bounded lexical-search match."""

    path: str
    snippet: str
    score: float


@dataclass(frozen=True)
class MemorySearchResult:
    """Bounded search results and scan metadata."""

    matches: list[MemorySearchMatch]
    scanned: int
    truncated: bool


class MemoryConflictError(RuntimeError):
    """The stored version did not match the mutation's expected version."""


class MemoryOperationConflictError(RuntimeError):
    """An operation id was reused with a different fingerprint."""


def validate_store_path(path: str) -> None:
    r"""Reject path strings that could escape a store's root directory."""
    if not all(_VALID_SEGMENT_RE.fullmatch(segment) and '..' not in segment for segment in path.split('/')):
        raise ValueError(f'invalid memory path: {path!r}')


def validate_store_prefix(prefix: str) -> None:
    if prefix:
        validate_store_path(prefix.removesuffix('/'))


def _enable_wal(connection: sqlite3.Connection) -> None:
    with _SQLITE_SETUP_LOCK:
        try:
            connection.execute('PRAGMA journal_mode = WAL')
        except sqlite3.OperationalError as error:  # pragma: no cover - requires a lock held by another process
            if not any(reason in str(error).lower() for reason in ('busy', 'locked')):
                raise
            # Journal mode is an optimization; transactions remain the consistency boundary.


def _replayed(mutation: MemoryMutation) -> MemoryMutation:
    return MemoryMutation(version=mutation.version, replayed=True, existed=mutation.existed)


def _check_operation(
    receipts: dict[str, tuple[str, MemoryMutation]], operation: MemoryOperation
) -> MemoryMutation | None:
    receipt = receipts.get(operation.id)
    if receipt is None:
        return None
    fingerprint, mutation = receipt
    if fingerprint != operation.fingerprint:
        raise MemoryOperationConflictError(f'operation id {operation.id!r} was reused with different arguments')
    return _replayed(mutation)


def _snippet(content: str, query: str, max_chars: int) -> str:
    if max_chars <= 0:  # pragma: no cover - lexical_search rejects non-positive budgets
        return ''
    lower = content.lower()
    positions = [lower.find(term) for term in query.lower().split()]
    found = [position for position in positions if position >= 0]
    center = min(found) if found else 0
    start = max(0, center - max_chars // 3)
    end = min(len(content), start + max_chars)
    start = max(0, end - max_chars)
    snippet = content[start:end]
    if start:
        snippet = f'...{snippet[3:]}' if len(snippet) >= 3 else '.' * len(snippet)
    if end < len(content):
        snippet = f'{snippet[:-3]}...' if len(snippet) >= 3 else '.' * len(snippet)
    return snippet


def lexical_search(
    files: Iterable[tuple[str, str]],
    query: str,
    *,
    limit: int,
    max_files: int,
    max_chars: int,
    score_prefix: str = '',
) -> MemorySearchResult:
    """Search a sorted file stream with deterministic scan and output bounds."""
    terms = [term for term in query.lower().split() if term]
    if not terms or limit <= 0 or max_files <= 0 or max_chars <= 0:  # pragma: no cover - stores prevalidate
        return MemorySearchResult(matches=[], scanned=0, truncated=False)
    scored: list[tuple[float, str, str]] = []
    scanned = 0
    truncated = False
    for path, content in files:
        if scanned >= max_files:
            truncated = True
            break
        scanned += 1
        lower_path = path.removeprefix(score_prefix).lower()
        lower_content = content.lower()
        score = float(sum(lower_content.count(term) + 2 * lower_path.count(term) for term in terms))
        if score:
            scored.append((score, path, content))
    scored.sort(key=lambda item: (-item[0], item[1]))
    matches: list[MemorySearchMatch] = []
    remaining = max_chars
    for score, path, content in scored[:limit]:
        visible_path = path.removeprefix(score_prefix)
        available = remaining - len(visible_path)
        if available <= 0:
            truncated = True
            break
        snippet = _snippet(content, query, available)
        matches.append(MemorySearchMatch(path=path, snippet=snippet, score=score))
        remaining -= len(visible_path) + len(snippet)
    if len(scored) > len(matches):
        truncated = True
    return MemorySearchResult(matches=matches, scanned=scanned, truncated=truncated)


@runtime_checkable
class MemoryStore(Protocol):
    """Async versioned storage for agent memories."""

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        """Return the file at `path`, or `None` if it does not exist."""
        ...  # pragma: no cover

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        """Return a prior result for `operation`, validating its fingerprint."""
        ...  # pragma: no cover

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        """Create or replace `path` if its version equals `expected_version`."""
        ...  # pragma: no cover

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        """Delete `path` if its version equals `expected_version`."""
        ...  # pragma: no cover

    async def list_paths(self, prefix: str = '', *, limit: int) -> list[str]:
        """Return all stored paths starting with `prefix`, sorted."""
        ...  # pragma: no cover


@runtime_checkable
class SearchableMemoryStore(Protocol):
    """Optional bounded search extension for a `MemoryStore`."""

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
        """Search only paths below the trusted, application-resolved `prefix`."""
        ...  # pragma: no cover


@dataclass(init=False)
class InMemoryStore:
    """Process-lifetime versioned memory store."""

    _files: dict[str, str] = field(default_factory=dict[str, str], repr=False)
    _versions: dict[str, int] = field(default_factory=dict[str, int], init=False, repr=False)
    _operation_ids: dict[str, str | None] = field(default_factory=dict[str, str | None], init=False, repr=False)
    _receipts: dict[str, tuple[str, MemoryMutation]] = field(
        default_factory=dict[str, tuple[str, MemoryMutation]], init=False, repr=False
    )
    _generation: int = field(default=0, init=False, repr=False)
    _paths: list[str] = field(default_factory=list[str], init=False, repr=False)
    _lock: anyio.Lock = field(default_factory=anyio.Lock, init=False, repr=False)

    def __init__(self, files: Mapping[str, str] | None = None) -> None:
        self._files = dict(files or {})
        for path in self._files:
            validate_store_path(path)
        self._versions = {}
        self._operation_ids = {}
        self._receipts = {}
        self._generation = 0
        self._paths = sorted(self._files)
        self._lock = anyio.Lock()

    @property
    def files(self) -> Mapping[str, str]:
        """A read-only view of stored content; mutate through `write` and `delete`."""
        return MappingProxyType(self._files)

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _current(self, path: str, max_chars: int | None = None) -> MemoryFile | None:
        content = self._files.get(path)
        if content is None:
            return None
        version = self._versions.get(path)
        if version is None:
            version = self._next_generation()
            self._versions[path] = version
        truncated = max_chars is not None and len(content) > max_chars
        return MemoryFile(
            content=content[:max_chars] if max_chars is not None else content,
            version=str(version),
            operation_id=self._operation_ids.get(path),
            truncated=truncated,
        )

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        validate_store_path(path)
        if max_chars <= 0:
            raise ValueError('max_chars must be positive')
        async with self._lock:
            return self._current(path, max_chars)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        async with self._lock:
            return _check_operation(self._receipts, operation)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        async with self._lock:
            if operation is not None and (receipt := _check_operation(self._receipts, operation)) is not None:
                return receipt
            current = self._current(path)
            if (current.version if current else None) != expected_version:
                raise MemoryConflictError(f'memory path {path!r} changed before it could be written')
            version = self._next_generation()
            self._files[path] = content
            if current is None:
                bisect.insort(self._paths, path)
            self._versions[path] = version
            self._operation_ids[path] = operation.id if operation else None
            mutation = MemoryMutation(version=str(version), replayed=False, existed=current is not None)
            if operation is not None:
                self._receipts[operation.id] = (operation.fingerprint, mutation)
            return mutation

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        async with self._lock:
            if operation is not None and (receipt := _check_operation(self._receipts, operation)) is not None:
                return receipt
            current = self._current(path)
            if (current.version if current else None) != expected_version:
                raise MemoryConflictError(f'memory path {path!r} changed before it could be deleted')
            existed = current is not None
            self._files.pop(path, None)
            if current is not None:
                self._paths.remove(path)
            self._versions.pop(path, None)
            self._operation_ids.pop(path, None)
            self._next_generation()
            mutation = MemoryMutation(version=None, replayed=False, existed=existed)
            if operation is not None:
                self._receipts[operation.id] = (operation.fingerprint, mutation)
            return mutation

    async def list_paths(self, prefix: str = '', *, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError('limit must be positive')
        async with self._lock:
            start = bisect.bisect_left(self._paths, prefix)
            paths: list[str] = []
            for index in range(start, len(self._paths)):
                path = self._paths[index]
                if not path.startswith(prefix):
                    break
                paths.append(path)
                if len(paths) == limit:
                    break
            return paths

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
        async with self._lock:
            start = bisect.bisect_left(self._paths, prefix)
            selected: list[str] = []
            for index in range(start, len(self._paths)):
                path = self._paths[index]
                if not path.startswith(prefix):
                    break
                selected.append(path)
                if len(selected) > max_files:
                    break
            scanned_paths = selected[:max_files]
            content_truncated = any(len(self._files[path]) > max_file_chars for path in scanned_paths)
            files = [(path, self._files[path][:max_file_chars]) for path in scanned_paths]
        result = lexical_search(
            files, query, limit=limit, max_files=max_files, max_chars=max_chars, score_prefix=prefix
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or content_truncated or len(selected) > max_files,
        )


_FILE_SCHEMA = """
CREATE TABLE IF NOT EXISTS file_state (
    path TEXT PRIMARY KEY,
    last_operation_id TEXT,
    version INTEGER,
    fingerprint TEXT
);
CREATE TABLE IF NOT EXISTS file_metadata (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    generation INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_operations (
    id TEXT PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    expected_version TEXT,
    new_content TEXT,
    result_version TEXT,
    existed INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS one_pending_memory_operation_per_path
ON memory_operations(path) WHERE status = 'prepared';
"""


class FileStore:
    """Plain-Markdown store with atomic replacement and a hidden SQLite journal."""

    def __init__(self, directory: str | Path) -> None:
        self._root = Path(directory)
        self._thread_lock = threading.RLock()

    def _resolve(self, path: str) -> Path:
        validate_store_path(path)
        if path.split('/', 1)[0] == _JOURNAL_NAME:
            raise ValueError(f'{_JOURNAL_NAME!r} is reserved for FileStore bookkeeping')
        real_root = Path(os.path.realpath(self._root))
        resolved = Path(os.path.realpath(real_root / path))
        if not resolved.is_relative_to(real_root):
            raise ValueError(f'memory path {path!r} resolves outside the store directory')
        return resolved

    def _connect(self) -> sqlite3.Connection:
        self._root.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self._root / _JOURNAL_NAME, timeout=30)
        try:
            connection.execute('PRAGMA busy_timeout = 30000')
            _enable_wal(connection)
            with _SQLITE_SETUP_LOCK:
                connection.executescript(_FILE_SCHEMA)
                connection.execute('BEGIN IMMEDIATE')
                columns = {str(row[1]) for row in connection.execute('PRAGMA table_info(file_state)').fetchall()}
                if 'version' not in columns:
                    connection.execute('ALTER TABLE file_state ADD COLUMN version INTEGER')
                if 'fingerprint' not in columns:
                    connection.execute('ALTER TABLE file_state ADD COLUMN fingerprint TEXT')
                connection.execute(
                    'INSERT OR IGNORE INTO file_metadata(id, generation) '
                    'SELECT 1, COALESCE(MAX(version), 0) FROM file_state'
                )
                journal_version_row = connection.execute('PRAGMA user_version').fetchone()
                assert journal_version_row is not None
                if int(journal_version_row[0]) < 1:
                    connection.execute(
                        'UPDATE memory_operations SET expected_version = NULL, new_content = NULL '
                        "WHERE status = 'completed' AND (expected_version IS NOT NULL OR new_content IS NOT NULL)"
                    )
                    connection.execute('PRAGMA user_version = 1')
                connection.commit()
            return connection
        except BaseException:
            connection.rollback()
            connection.close()
            raise

    def _next_generation(self, connection: sqlite3.Connection) -> int:
        row = connection.execute(
            'UPDATE file_metadata SET generation = generation + 1 WHERE id = 1 RETURNING generation'
        ).fetchone()
        assert row is not None
        return int(row[0])

    def _inspect_file(self, target: Path, max_chars: int) -> tuple[str, str, bool]:
        with target.open(encoding='utf-8') as file:
            preview = file.read(max_chars + 1)
            stat = os.fstat(file.fileno())
        fingerprint = f'{stat.st_dev}:{stat.st_ino}:{stat.st_mtime_ns}:{stat.st_size}'
        return preview[:max_chars], fingerprint, len(preview) > max_chars

    def _record_file(
        self,
        connection: sqlite3.Connection,
        path: str,
        *,
        version: str,
        operation_id: str | None,
    ) -> None:
        target = self._resolve(path)
        _, fingerprint, _ = self._inspect_file(target, 0)
        connection.execute(
            'INSERT INTO file_state(path, last_operation_id, version, fingerprint) VALUES (?, ?, ?, ?) '
            'ON CONFLICT(path) DO UPDATE SET last_operation_id = excluded.last_operation_id, '
            'version = excluded.version, fingerprint = excluded.fingerprint',
            (path, operation_id, int(version), fingerprint),
        )

    def _matches_content(self, path: str, content: str) -> bool:
        target = self._resolve(path)
        if not target.is_file() or target.stat().st_size != len(content.encode()):
            return False
        with target.open(encoding='utf-8') as file:
            return file.read(len(content) + 1) == content

    def _atomic_write(self, target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle, tmp_name = tempfile.mkstemp(dir=target.parent, prefix='.memory-tmp-')
        try:
            with os.fdopen(handle, 'w', encoding='utf-8') as tmp_file:
                tmp_file.write(content)
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):  # pragma: no cover
                os.unlink(tmp_name)

    def _current(self, connection: sqlite3.Connection, path: str, max_chars: int = 0) -> MemoryFile | None:
        target = self._resolve(path)
        if not target.is_file():
            return None
        content, fingerprint, truncated = self._inspect_file(target, max_chars)
        row = connection.execute(
            'SELECT last_operation_id, version, fingerprint FROM file_state WHERE path = ?', (path,)
        ).fetchone()
        if row is None or row[1] is None or str(row[2]) != fingerprint:
            version = str(self._next_generation(connection))
            self._record_file(connection, path, version=version, operation_id=None)
            operation_id = None
        else:
            operation_id = str(row[0]) if row[0] is not None else None
            version = str(row[1])
        return MemoryFile(
            content=content,
            version=version,
            operation_id=operation_id,
            truncated=truncated,
        )

    def _recover(self, connection: sqlite3.Connection, path: str) -> None:
        query = (
            'SELECT id, kind, path, expected_version, new_content, result_version '
            "FROM memory_operations WHERE status = 'prepared' AND path = ? ORDER BY rowid"
        )
        for row in connection.execute(query, (path,)).fetchall():
            operation_id, kind, pending_path = str(row[0]), str(row[1]), str(row[2])
            expected = str(row[3]) if row[3] is not None else None
            content = str(row[4]) if row[4] is not None else None
            result_version = str(row[5]) if row[5] is not None else None
            current = self._current(connection, pending_path)
            current_version = current.version if current else None
            if kind == 'write':
                assert content is not None
                if current_version == expected:
                    self._atomic_write(self._resolve(pending_path), content)
                elif current is None or not self._matches_content(pending_path, content):
                    raise MemoryConflictError(f'externally modified memory path {pending_path!r} blocks recovery')
                assert result_version is not None
                self._record_file(connection, pending_path, version=result_version, operation_id=operation_id)
            else:
                if current_version == expected:
                    self._resolve(pending_path).unlink(missing_ok=True)
                elif current_version is not None:
                    raise MemoryConflictError(f'externally modified memory path {pending_path!r} blocks recovery')
                connection.execute('DELETE FROM file_state WHERE path = ?', (pending_path,))
            connection.execute(
                "UPDATE memory_operations SET status = 'completed', expected_version = NULL, new_content = NULL "
                'WHERE id = ?',
                (operation_id,),
            )

    def _get_operation(self, connection: sqlite3.Connection, operation: MemoryOperation) -> MemoryMutation | None:
        row = connection.execute(
            'SELECT fingerprint, status, path, result_version, existed FROM memory_operations WHERE id = ?',
            (operation.id,),
        ).fetchone()
        if row is None:
            return None
        if str(row[0]) != operation.fingerprint:
            raise MemoryOperationConflictError(f'operation id {operation.id!r} was reused with different arguments')
        if str(row[1]) == 'prepared':
            self._recover(connection, str(row[2]))
        return MemoryMutation(
            version=str(row[3]) if row[3] is not None else None,
            replayed=True,
            existed=bool(row[4]),
        )

    def _transaction(self, operation: Callable[[sqlite3.Connection], _T]) -> _T:
        with self._thread_lock:
            connection = self._connect()
            try:
                connection.execute('BEGIN IMMEDIATE')
                result = operation(connection)
                connection.commit()
                return result
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        validate_store_path(path)
        if max_chars <= 0:
            raise ValueError('max_chars must be positive')

        def op(connection: sqlite3.Connection) -> MemoryFile | None:
            self._recover(connection, path)
            return self._current(connection, path, max_chars)

        return await anyio.to_thread.run_sync(self._transaction, op)

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        def run() -> MemoryMutation | None:
            def op(connection: sqlite3.Connection) -> MemoryMutation | None:
                return self._get_operation(connection, operation)

            return self._transaction(op)

        return await anyio.to_thread.run_sync(run)

    def _prepare_write(
        self,
        connection: sqlite3.Connection,
        path: str,
        content: str,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        self._recover(connection, path)
        if operation is not None and (receipt := self._get_operation(connection, operation)) is not None:
            return receipt
        current = self._current(connection, path)
        if (current.version if current else None) != expected_version:
            raise MemoryConflictError(f'memory path {path!r} changed before it could be written')
        version = str(self._next_generation(connection))
        mutation = MemoryMutation(version=version, replayed=False, existed=current is not None)
        if operation is None:
            self._atomic_write(self._resolve(path), content)
            self._record_file(connection, path, version=version, operation_id=None)
            return mutation
        connection.execute(
            'INSERT INTO memory_operations '
            '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
            "VALUES (?, ?, 'prepared', 'write', ?, ?, ?, ?, ?)",
            (operation.id, operation.fingerprint, path, expected_version, content, version, int(current is not None)),
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
        validate_store_path(path)

        def prepare() -> MemoryMutation:
            def op(connection: sqlite3.Connection) -> MemoryMutation:
                return self._prepare_write(connection, path, content, expected_version, operation)

            return self._transaction(op)

        mutation = await anyio.to_thread.run_sync(prepare)
        if operation is not None and not mutation.replayed:

            def recover() -> None:
                def op(connection: sqlite3.Connection) -> None:
                    self._recover(connection, path)

                self._transaction(op)

            await anyio.to_thread.run_sync(recover)
        return mutation

    def _prepare_delete(
        self,
        connection: sqlite3.Connection,
        path: str,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        self._recover(connection, path)
        if operation is not None and (receipt := self._get_operation(connection, operation)) is not None:
            return receipt
        current = self._current(connection, path)
        if (current.version if current else None) != expected_version:
            raise MemoryConflictError(f'memory path {path!r} changed before it could be deleted')
        mutation = MemoryMutation(version=None, replayed=False, existed=current is not None)
        self._next_generation(connection)
        if operation is None:
            self._resolve(path).unlink(missing_ok=True)
            connection.execute('DELETE FROM file_state WHERE path = ?', (path,))
            return mutation
        if current is None:
            connection.execute(
                'INSERT INTO memory_operations '
                '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
                "VALUES (?, ?, 'completed', 'delete', ?, NULL, NULL, NULL, 0)",
                (operation.id, operation.fingerprint, path),
            )
        else:
            connection.execute(
                'INSERT INTO memory_operations '
                '(id, fingerprint, status, kind, path, expected_version, new_content, result_version, existed) '
                "VALUES (?, ?, 'prepared', 'delete', ?, ?, NULL, NULL, 1)",
                (operation.id, operation.fingerprint, path, expected_version),
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

        def prepare() -> MemoryMutation:
            def op(connection: sqlite3.Connection) -> MemoryMutation:
                return self._prepare_delete(connection, path, expected_version, operation)

            return self._transaction(op)

        mutation = await anyio.to_thread.run_sync(prepare)
        if operation is not None and mutation.existed and not mutation.replayed:

            def recover() -> None:
                def op(connection: sqlite3.Connection) -> None:
                    self._recover(connection, path)

                self._transaction(op)

            await anyio.to_thread.run_sync(recover)
        return mutation

    async def list_paths(self, prefix: str = '', *, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError('limit must be positive')
        return await anyio.to_thread.run_sync(self._sync_list_paths, prefix, limit)

    def _sync_list_paths(self, prefix: str, limit: int) -> list[str]:
        def op(connection: sqlite3.Connection) -> list[str]:
            root = Path(os.path.realpath(self._root))
            walk_root = root
            if prefix.endswith('/'):
                walk_root = self._resolve(prefix.removesuffix('/'))

            def paths(directory: Path) -> Iterable[str]:
                if not directory.is_dir():
                    return
                with os.scandir(directory) as entries:
                    for entry in entries:
                        item = Path(entry.path)
                        if entry.is_dir(follow_symlinks=False):
                            yield from paths(item)
                        elif (
                            entry.is_file(follow_symlinks=False)
                            and not entry.name.startswith(_JOURNAL_NAME)
                            and not entry.name.startswith('.memory-tmp-')
                        ):
                            relative = item.relative_to(root).as_posix()
                            if relative.startswith(prefix):
                                yield relative

            while True:
                selected = heapq.nsmallest(limit, paths(walk_root))
                pending = connection.execute(
                    "SELECT DISTINCT path FROM memory_operations WHERE status = 'prepared' "
                    'AND substr(path, 1, length(?)) = ? ORDER BY path LIMIT ?',
                    (prefix, prefix, _FILE_RECOVERY_BATCH_SIZE),
                ).fetchall()
                if not pending:
                    return selected
                if len(selected) == limit and str(pending[0][0]) > selected[-1]:
                    return selected
                for row in pending:
                    self._recover(connection, str(row[0]))

        return self._transaction(op)

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
        if not query.split() or limit <= 0 or max_files <= 0 or max_chars <= 0 or max_file_chars <= 0:
            return MemorySearchResult(matches=[], scanned=0, truncated=False)
        paths = await self.list_paths(prefix, limit=max_files + 1)
        paths_truncated = len(paths) > max_files

        def load() -> tuple[list[tuple[str, str]], bool]:
            files: list[tuple[str, str]] = []
            truncated = False
            for path in paths[:max_files]:
                try:
                    with self._resolve(path).open(encoding='utf-8') as file:
                        content = file.read(max_file_chars + 1)
                except FileNotFoundError:
                    truncated = True
                    continue
                truncated = truncated or len(content) > max_file_chars
                files.append((path, content[:max_file_chars]))
            return files, truncated

        files, content_truncated = await anyio.to_thread.run_sync(load)
        result = lexical_search(
            files, query, limit=limit, max_files=max_files, max_chars=max_chars, score_prefix=prefix
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or content_truncated or paths_truncated,
        )


_SQLITE_MEMORY_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS memory_files ('
    'path TEXT PRIMARY KEY, content TEXT NOT NULL, version INTEGER NOT NULL, last_operation_id TEXT)'
)
_SQLITE_OPERATIONS_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS memory_operations ('
    'id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, version TEXT, existed INTEGER NOT NULL)'
)
_SQLITE_METADATA_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS memory_metadata (id INTEGER PRIMARY KEY CHECK (id = 1), generation INTEGER NOT NULL)'
)


class SqliteMemoryStore:
    """SQLite-backed store with transactional CAS and operation receipts."""

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
                'SELECT substr(content, 1, ?), version, last_operation_id, length(content) '
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
    ) -> MemoryMutation:
        if operation is not None and (receipt := self._get_operation(connection, operation)) is not None:
            return receipt
        row = connection.execute('SELECT version FROM memory_files WHERE path = ?', (path,)).fetchone()
        current = str(row[0]) if row is not None else None
        if current != expected_version:
            raise MemoryConflictError(f'memory path {path!r} changed before it could be written')
        version = str(self._next_generation(connection))
        if current is None:
            connection.execute(
                'INSERT INTO memory_files(path, content, version, last_operation_id) VALUES (?, ?, ?, ?)',
                (path, content, int(version), operation.id if operation else None),
            )
        else:
            cursor = connection.execute(
                'UPDATE memory_files SET content = ?, version = ?, last_operation_id = ? '
                'WHERE path = ? AND version = ?',
                (content, int(version), operation.id if operation else None, path, int(current)),
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
        validate_store_path(path)
        return await anyio.to_thread.run_sync(
            lambda: self._run(
                lambda connection: self._write(connection, path, content, expected_version, operation),
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
                'SELECT path FROM memory_files WHERE substr(path, 1, length(?)) = ? ORDER BY path LIMIT ?',
                (prefix, prefix, limit),
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
            rows = connection.execute(
                'SELECT path, substr(content, 1, ?), length(content) FROM memory_files '
                'WHERE substr(path, 1, length(?)) = ? ORDER BY path LIMIT ?',
                (max_file_chars, prefix, prefix, max_files + 1),
            ).fetchall()
            return [(str(row[0]), str(row[1]), int(row[2])) for row in rows]

        rows = await anyio.to_thread.run_sync(self._run, op)
        result = lexical_search(
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

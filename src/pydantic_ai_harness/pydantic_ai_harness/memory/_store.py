"""Versioned, path-addressed persistence backends for the `Memory` capability."""

from __future__ import annotations

import bisect
import hashlib
import heapq
import json
import logging
import posixpath
import re
import sqlite3
import threading
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping
from copy import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Protocol, TypeVar, runtime_checkable

import anyio
import anyio.to_thread

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import Workspace, WorkspaceBackend, WorkspaceError, WorkspaceUnavailableError
from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness._workspace import secondary_workspace, supports_commands, workspace_path

_VALID_SEGMENT_RE = re.compile(r'[A-Za-z0-9_.-]{1,200}')
_OPERATIONS_NAME = '.memory-operations.json'
_LEGACY_JOURNAL_NAME = '.memory-store.sqlite3'
"""The SQLite journal earlier `FileStore` releases kept beside the files; hidden from listings."""
_TEMP_PREFIX = '.memory-tmp-'
"""Names the file a write is staged in before it replaces its target; hidden from listings."""
_HIDDEN_PREFIXES = (_OPERATIONS_NAME, _LEGACY_JOURNAL_NAME, _TEMP_PREFIX)
_RENAME_TIMEOUT = 30.0
"""Deadline in seconds for the `mv` that replaces a file, and for removing a staged file after a failure."""
_MAX_RECEIPTS = 1024
_SQLITE_SETUP_LOCK = threading.RLock()
_T = TypeVar('_T')
logger = logging.getLogger(__name__)


class MemoryPathEscapeError(ValueError):
    """A memory path resolves outside its store directory."""


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


@dataclass
class _Receipt:
    """One idempotent mutation in `FileStore`'s sidecar, kept so a replayed tool call is not applied twice."""

    id: str
    fingerprint: str
    kind: Literal['write', 'delete']
    path: str
    expected: str | None
    version: str | None
    existed: bool
    done: bool

    def mutation(self) -> MemoryMutation:
        return MemoryMutation(version=self.version, replayed=True, existed=self.existed)


def content_version(content: str) -> str:
    """`FileStore`'s version for `content`: its SHA-256 hex digest."""
    return hashlib.sha256(content.encode()).hexdigest()


async def _replace_text(workspace: Workspace, target: str, content: str) -> None:
    """Replace the file at `target` with `content`, so a failed write leaves the old content whole.

    The content is staged beside the target and moved over it with `mv`, a rename within one
    filesystem. On failure the staged file is removed, best effort.
    """
    if not supports_commands(workspace):
        # A filesystem-only workspace offers no rename, so there the write is in place, not atomic.
        await workspace.write_text(target, content)
        return
    directory, name = posixpath.split(target)
    # The target's name at the end keeps the staged file recognisable in the directory.
    staged = posixpath.join(directory, f'{_TEMP_PREFIX}{uuid.uuid4().hex}-{name}')
    try:
        await workspace.write_text(staged, content)
        try:
            result = await workspace.run(['mv', '-f', staged, target], timeout=_RENAME_TIMEOUT)
        except WorkspaceUnavailableError:
            raise
        except Exception as error:
            raise WorkspaceError(f'Could not replace memory file {target!r}: {error}') from error
        if result.exit_code != 0:
            detail = result.stderr.strip() or f'`mv` exited with {result.exit_code}'
            raise WorkspaceError(f'Could not replace memory file {target!r}: {detail}')
    except BaseException:
        with anyio.move_on_after(_RENAME_TIMEOUT, shield=True):
            try:
                await workspace.remove(staged)
            except Exception:
                pass  # Best effort: the staged file may not exist, and the original error matters more.
        raise


class FileStore:
    """Plain-Markdown memory files in a workspace directory.

    Files live under `directory` in the run's workspace (`ctx.workspace`), or in `workspace` when
    set; a relative `directory` is relative to the workspace's working directory, so the model can
    also open the files with its file tools. A file's version is the SHA-256 of its content, so an
    edit made outside the store is seen as a change. Receipts for the most recent idempotent
    mutations are kept beside the files in `.memory-operations.json`, so a replayed tool call is
    not applied twice.

    One writer per directory: one `FileStore` serializes its own operations, but two stores, or
    two processes, writing the same directory can lose updates. Use `SqliteMemoryStore` or
    `PostgresMemoryStore` for concurrent writers.
    """

    def __init__(self, directory: str | Path, *, workspace: WorkspaceBackend | None = None) -> None:
        """Create a store.

        Args:
            directory: Workspace directory for the memory files, absolute or relative to the
                workspace's working directory.
            workspace: A workspace to keep the files in instead of the run's, such as
                `LocalWorkspaceBackend('/var/memory')` to keep them on this machine while the agent
                works in a sandbox. Required when the store is used outside a `Memory` run.
        """
        self.directory = workspace_path(directory) if isinstance(directory, Path) else directory
        self.workspace = workspace
        if workspace is None and (posixpath.isabs(self.directory) or Path(directory).is_absolute()):
            warnings.warn(
                f"`FileStore({str(directory)!r})` now keeps memory in the run's workspace, at that path inside it. "
                f'To keep it in that directory on this machine, pass '
                f"`FileStore('.', workspace=LocalWorkspaceBackend({str(directory)!r}))`.",
                category=HarnessDeprecationWarning,
                stacklevel=2,
            )
        self._own = secondary_workspace(workspace, 'FileStore')
        self._run: Workspace | None = None
        self._real_root: tuple[str, str] | None = None
        self._lock = anyio.Lock()

    def bind(self, workspace: Workspace) -> FileStore:
        """This store, reading and writing through `workspace` unless it has a `workspace` of its own.

        `Memory` binds the run's workspace this way on every call. The copy shares this store's lock.
        """
        if self._own is not None:
            return self
        if self._run is workspace:
            return self
        bound = copy(self)
        bound._run = workspace
        bound._real_root = None
        return bound

    def _workspace(self) -> Workspace:
        workspace = self._own or self._run
        if workspace is None:
            raise UserError(
                '`FileStore` has no workspace. Use it through `Memory` in a run with a workspace, '
                "or pass `workspace=`, such as `LocalWorkspaceBackend('.')`."
            )
        return workspace

    async def _root(self, workspace: Workspace) -> str:
        return await workspace.resolve(self.directory)

    @staticmethod
    def _target(root: str, path: str) -> str:
        validate_store_path(path)
        if path.split('/', 1)[0] in (_OPERATIONS_NAME, _LEGACY_JOURNAL_NAME):
            raise ValueError(f'{path!r} is reserved for FileStore bookkeeping')
        return posixpath.join(root, path)

    async def _confine(self, workspace: Workspace, root: str, target: str, path: str) -> str:
        """Return `target` with symlinks resolved, refusing one that leaves the store directory."""
        # Keep the original root boundary for this binding: if it is replaced by a
        # symlink later, target realpaths outside it fail closed rather than following it.
        if self._real_root is None or self._real_root[0] != root:
            self._real_root = (root, await workspace.realpath(root))
        return FileStore._confined(self._real_root[1], await workspace.realpath(target), path)

    @staticmethod
    def _confined(real_root: str, real_target: str, path: str) -> str:
        if not real_target.startswith(real_root.rstrip('/') + '/'):
            raise MemoryPathEscapeError(f'memory path {path!r} resolves outside the store directory')
        return real_target

    @staticmethod
    async def _content(workspace: Workspace, target: str) -> str | None:
        try:
            return await workspace.read_text(target)
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    async def _operations(self, workspace: Workspace, root: str) -> str:
        return await self._confine(workspace, root, posixpath.join(root, _OPERATIONS_NAME), _OPERATIONS_NAME)

    @staticmethod
    async def _receipts(workspace: Workspace, operations: str) -> list[_Receipt]:
        raw = await FileStore._content(workspace, operations)
        if raw is None:
            return []
        try:
            return [_Receipt(**item) for item in json.loads(raw)]
        except (json.JSONDecodeError, TypeError):
            # Invalid receipts cannot establish idempotency; start a fresh journal.
            logger.warning('Invalid FileStore operation receipts; starting with an empty journal')
            return []

    @staticmethod
    async def _save(workspace: Workspace, operations: str, receipts: list[_Receipt]) -> None:
        kept = [asdict(receipt) for receipt in receipts[-_MAX_RECEIPTS:]]
        await _replace_text(workspace, operations, json.dumps(kept))

    async def _settle(self, workspace: Workspace, root: str, receipts: list[_Receipt], path: str) -> bool:
        """Finish or drop receipts for `path` left pending by an interrupted mutation; return whether any changed.

        A pending receipt whose result is on disk is marked done. Otherwise the mutation never
        landed, so the receipt is dropped and a retry applies it again.
        """
        changed = False
        for receipt in [receipt for receipt in receipts if not receipt.done and receipt.path == path]:
            content = await self._content(
                workspace, await self._confine(workspace, root, self._target(root, path), path)
            )
            version = None if content is None else content_version(content)
            changed = True
            if version == receipt.version:
                receipt.done = True
            else:
                receipts.remove(receipt)
        return changed

    @staticmethod
    def _find(receipts: list[_Receipt], operation: MemoryOperation) -> _Receipt | None:
        receipt = next((receipt for receipt in receipts if receipt.id == operation.id), None)
        if receipt is not None and receipt.fingerprint != operation.fingerprint:
            raise MemoryOperationConflictError(f'operation id {operation.id!r} was reused with different arguments')
        return receipt

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        validate_store_path(path)
        if max_chars <= 0:
            raise ValueError('max_chars must be positive')
        workspace = self._workspace()
        root = await self._root(workspace)
        target = await self._confine(workspace, root, self._target(root, path), path)
        content = await self._content(workspace, target)
        if content is None:
            return None
        version = content_version(content)
        receipts = await self._receipts(workspace, await self._operations(workspace, root))
        operation_id = next(
            (
                receipt.id
                for receipt in reversed(receipts)
                if receipt.done and receipt.kind == 'write' and receipt.path == path and receipt.version == version
            ),
            None,
        )
        return MemoryFile(
            content=content[:max_chars],
            version=version,
            operation_id=operation_id,
            truncated=len(content) > max_chars,
        )

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        workspace = self._workspace()
        async with self._lock:
            root = await self._root(workspace)
            operations = await self._operations(workspace, root)
            receipts = await self._receipts(workspace, operations)
            receipt = self._find(receipts, operation)
            if receipt is None:
                return None
            if not receipt.done and await self._settle(workspace, root, receipts, receipt.path):
                await self._save(workspace, operations, receipts)
            return receipt.mutation() if receipt.done else None

    async def _mutate(
        self,
        kind: Literal['write', 'delete'],
        path: str,
        content: str | None,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        workspace = self._workspace()
        async with self._lock:
            root = await self._root(workspace)
            target = self._target(root, path)
            real_target = await self._confine(workspace, root, target, path)
            operations = await self._operations(workspace, root)
            receipts = await self._receipts(workspace, operations)
            settled = await self._settle(workspace, root, receipts, path)
            if operation is not None and (receipt := self._find(receipts, operation)) is not None:
                if settled:
                    await self._save(workspace, operations, receipts)
                return receipt.mutation()
            current = await self._content(workspace, target)
            if (None if current is None else content_version(current)) != expected_version:
                if settled:
                    await self._save(workspace, operations, receipts)
                raise MemoryConflictError(
                    f'memory path {path!r} changed before it could be {"written" if kind == "write" else "deleted"}'
                )
            version = None if content is None else content_version(content)
            receipt = None
            if operation is not None:
                receipt = _Receipt(
                    operation.id,
                    operation.fingerprint,
                    kind,
                    path,
                    expected_version,
                    version,
                    current is not None,
                    False,
                )
                receipts.append(receipt)
                await self._save(workspace, operations, receipts)
            elif settled:
                await self._save(workspace, operations, receipts)
            if content is not None:
                # The real path: `mv` onto a symlink would replace the link rather than write through it.
                await _replace_text(workspace, real_target, content)
            elif current is not None:
                await workspace.remove(target)
            if receipt is not None:
                receipt.done = True
                await self._save(workspace, operations, receipts)
            return MemoryMutation(version=version, replayed=False, existed=current is not None)

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        return await self._mutate('write', path, content, expected_version, operation)

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        validate_store_path(path)
        return await self._mutate('delete', path, None, expected_version, operation)

    async def list_paths(self, prefix: str = '', *, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError('limit must be positive')
        workspace = self._workspace()
        root = await workspace.realpath(await self._root(workspace))
        start = posixpath.join(root, prefix.removesuffix('/')) if prefix.endswith('/') else root
        # A scoped prefix can itself be a symlink; check it before listing its first child.
        if start != root and await workspace.realpath(start) != start:
            return []
        paths: list[str] = []
        pending = [start]
        while pending:
            directory = pending.pop()
            try:
                entries = await workspace.list_dir(directory)
            except (FileNotFoundError, NotADirectoryError):
                continue
            for entry in entries:
                if entry.is_dir:
                    # Walking from the real root, a directory whose real path differs is a symlink:
                    # following it could list another scope's files or leave the store.
                    if await workspace.realpath(entry.path) == entry.path:
                        pending.append(entry.path)
                    continue
                relative = posixpath.relpath(entry.path, root)
                if relative.startswith(prefix) and not entry.name.startswith(_HIDDEN_PREFIXES):
                    paths.append(relative)
        return heapq.nsmallest(limit, paths)

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
        workspace = self._workspace()
        # Resolved once here rather than per file.
        root = await workspace.realpath(await self._root(workspace))
        files: list[tuple[str, str]] = []
        content_truncated = False
        for path in paths[:max_files]:
            try:
                target = self._confined(root, await workspace.realpath(self._target(root, path)), path)
            except ValueError:
                # A listed file that links outside the store directory is not read.
                continue
            content = await self._content(workspace, target)
            if content is None:
                content_truncated = True
                continue
            content_truncated = content_truncated or len(content) > max_file_chars
            files.append((path, content[:max_file_chars]))
        result = lexical_search(
            files, query, limit=limit, max_files=max_files, max_chars=max_chars, score_prefix=prefix
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or content_truncated or len(paths) > max_files,
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

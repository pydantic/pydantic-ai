"""`MemoryStore` that keeps plain Markdown files in a workspace."""

from __future__ import annotations

import hashlib
import heapq
import json
import logging
import posixpath
import uuid
import warnings
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import anyio

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import Workspace, WorkspaceBackend, WorkspaceError, WorkspaceUnavailableError
from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness._workspace import secondary_workspace, supports_commands, workspace_path
from pydantic_ai_harness.memory._store import (
    MAX_RECEIPTS,
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemoryPathEscapeError,
    MemorySearchResult,
    is_direct_child,
    lexical_search,
    validate_store_path,
    validate_store_prefix,
)

_OPERATIONS_NAME = '.memory-operations.json'
_LEGACY_JOURNAL_NAME = '.memory-store.sqlite3'
"""The SQLite journal earlier `FileStore` releases kept beside the files; hidden from listings."""
_TEMP_PREFIX = '.memory-tmp-'
"""Names the file a write is staged in before it replaces its target; hidden from listings."""
_HIDDEN_PREFIXES = (_OPERATIONS_NAME, _LEGACY_JOURNAL_NAME, _TEMP_PREFIX)
_RENAME_TIMEOUT = 30.0
"""Deadline in seconds for the `mv` that replaces a file, and for removing a staged file after a failure."""
logger = logging.getLogger(__name__)


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
        kept = [asdict(receipt) for receipt in receipts[-MAX_RECEIPTS:]]
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
        return await self._list_paths(prefix, limit=limit, nested=True)

    async def _list_paths(self, prefix: str, *, limit: int, nested: bool) -> list[str]:
        """List paths below `prefix`; with `nested=False`, only those directly below it."""
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError('limit must be positive')
        workspace = self._workspace()
        root = await workspace.realpath(await self._root(workspace))
        # Every path below `prefix` sits under its directory part, which also bounds a non-nested listing.
        parent = posixpath.dirname(prefix)
        start = posixpath.join(root, parent) if parent else root
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
                    if nested and await workspace.realpath(entry.path) == entry.path:
                        pending.append(entry.path)
                    continue
                relative = posixpath.relpath(entry.path, root)
                if (
                    relative.startswith(prefix)
                    and (nested or is_direct_child(prefix, relative))
                    and not entry.name.startswith(_HIDDEN_PREFIXES)
                ):
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
        # Paths in a deeper scope nested under `prefix` belong to another namespace or agent.
        paths = await self._list_paths(prefix, limit=max_files + 1, nested=False)
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
        result = await lexical_search(
            files, query, limit=limit, max_files=max_files, max_chars=max_chars, score_prefix=prefix
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or content_truncated or len(paths) > max_files,
        )

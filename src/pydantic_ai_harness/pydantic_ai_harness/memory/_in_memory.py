"""Process-lifetime `MemoryStore`."""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from types import MappingProxyType

import anyio

from pydantic_ai_harness.memory._store import (
    MAX_RECEIPTS,
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchResult,
    decode_metadata,
    encode_metadata,
    is_direct_child,
    lexical_search,
    validate_store_path,
    validate_store_prefix,
)


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


class InMemoryStore:
    """Process-lifetime versioned memory store, with write metadata."""

    def __init__(self, files: Mapping[str, str] | None = None) -> None:
        self._files = dict(files or {})
        for path in self._files:
            validate_store_path(path)
        self._versions: dict[str, int] = {}
        self._operation_ids: dict[str, str | None] = {}
        # Kept as JSON, like the database stores keep it, so no caller holds the stored mapping.
        self._metadata: dict[str, str] = {}
        self._receipts: dict[str, tuple[str, MemoryMutation]] = {}
        self._generation = 0
        self._paths = sorted(self._files)
        self._lock = anyio.Lock()

    def __repr__(self) -> str:
        return f'{type(self).__name__}()'

    @property
    def files(self) -> Mapping[str, str]:
        """A read-only view of stored content; mutate through `write` and `delete`."""
        return MappingProxyType(self._files)

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _keep_receipt(self, operation: MemoryOperation, mutation: MemoryMutation) -> None:
        """Record a receipt, keeping only the most recent ones, as `FileStore` does."""
        self._receipts[operation.id] = (operation.fingerprint, mutation)
        if len(self._receipts) > MAX_RECEIPTS:
            del self._receipts[next(iter(self._receipts))]

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
            metadata=decode_metadata(self._metadata.get(path)),
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
            if encoded is None:
                self._metadata.pop(path, None)
            else:
                self._metadata[path] = encoded
            mutation = MemoryMutation(version=str(version), replayed=False, existed=current is not None)
            if operation is not None:
                self._keep_receipt(operation, mutation)
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
            self._metadata.pop(path, None)
            self._next_generation()
            mutation = MemoryMutation(version=None, replayed=False, existed=existed)
            if operation is not None:
                self._keep_receipt(operation, mutation)
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
            index = bisect.bisect_left(self._paths, prefix)
            selected: list[str] = []
            while index < len(self._paths) and len(selected) <= max_files:
                path = self._paths[index]
                if not path.startswith(prefix):
                    break
                if is_direct_child(prefix, path):
                    selected.append(path)
                    index += 1
                else:
                    # Jump past the whole nested scope instead of walking each of its paths.
                    nested_scope = path[: path.index('/', len(prefix)) + 1]
                    index = bisect.bisect_left(self._paths, nested_scope + '\U0010ffff')
            scanned_paths = selected[:max_files]
            content_truncated = any(len(self._files[path]) > max_file_chars for path in scanned_paths)
            files = [(path, self._files[path][:max_file_chars]) for path in scanned_paths]
        result = await lexical_search(
            files, query, limit=limit, max_files=max_files, max_chars=max_chars, score_prefix=prefix
        )
        return MemorySearchResult(
            matches=result.matches,
            scanned=result.scanned,
            truncated=result.truncated or content_truncated or len(selected) > max_files,
        )

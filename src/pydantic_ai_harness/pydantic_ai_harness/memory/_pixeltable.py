"""Pixeltable backend for Harness `Memory`, with file rows and operation receipts.

Pixeltable indexes string primary keys on `left(path, 256)`, so paths are capped at 255
characters. Its `slice(0, n)` translates to SQL `substr`, while `slice(stop=n)` returns
`n + 1` characters; prefix filters use the former. Compare-and-set relies on Pixeltable's
table lock and affected-row counts.
"""

from __future__ import annotations

import functools
import threading
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

import anyio.to_thread

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

try:
    import pixeltable as pxt
    from pixeltable.exprs.expr import Expr
except ImportError as _import_error:
    raise ImportError(
        'pixeltable is required for `PixeltableMemoryStore`. It needs Python 3.11 to 3.14; install it with: pip install "pydantic-ai-harness[pixeltable]"'
    ) from _import_error

_KIND_FILE = 'file'
_KIND_OP = 'op'
_OP_PREFIX = '__op__/'

# The primary-key index covers left(path, 256): two longer paths whose first 256 characters
# match collide on insert. Cap paths below the boundary.
_MAX_PATH_CHARS = 255


def _check_store_path(path: str) -> None:
    validate_store_path(path)
    if path.split('/', 1)[0] == _OP_PREFIX.rstrip('/'):
        raise ValueError(f'memory path {path!r} is reserved for operation receipts')
    if len(path) > _MAX_PATH_CHARS:
        raise ValueError(f'memory path {path!r} exceeds {_MAX_PATH_CHARS} characters')


_MAX_OPERATION_ID_CHARS = _MAX_PATH_CHARS - len(_OP_PREFIX)


def _receipt_path(operation: MemoryOperation) -> str:
    # Receipts share the path key, so a longer id could collide with another id's receipt.
    if len(operation.id) > _MAX_OPERATION_ID_CHARS:
        raise ValueError(f'operation id exceeds {_MAX_OPERATION_ID_CHARS} characters')
    return f'{_OP_PREFIX}{operation.id}'


_T = TypeVar('_T')


def _insert_rows(t: pxt.Table, rows: list[dict[str, object]]) -> None:
    try:
        t.insert(rows)  # pyright: ignore[reportUnknownMemberType]
    except pxt.Error as exc:
        if (
            isinstance(exc, pxt.RequestError)
            and exc.error_code == pxt.ErrorCode.CONSTRAINT_VIOLATION
            and exc.message.startswith('Duplicate primary key')
        ):
            raise MemoryConflictError(str(exc)) from exc
        raise


# Expected type base per column for a memory table; compare-and-set needs all of
# them plus the primary key on `path`.
_SCHEMA_COLUMNS = {
    'path': 'String',
    'kind': 'String',
    'content': 'String',
    'version': 'String',
    'last_operation_id': 'String',
    'fingerprint': 'String',
    'existed': 'Bool',
}

# File rows and `__op__` receipts each write None to some of these, so they must be nullable.
_NULLABLE_COLUMNS = frozenset({'content', 'version', 'last_operation_id', 'fingerprint', 'existed'})

_PATH_INDEX = 'path_lookup_idx'


class PixeltableMemoryStore:
    """Pydantic AI Harness `MemoryStore` persisted in a Pixeltable table.

    Implements `MemoryStore` and `SearchableMemoryStore`. File mutations use
    compare-and-set on the file row. Operation receipts live in the same table
    under `__op__/`, so paths whose first segment is `__op__` are reserved.
    Pixeltable has no multi-statement transaction, so a receipt is written as
    pending before its mutation and completed after it. The next mutation of a
    file settles receipts a crash left pending on it: a write is completed if the
    file holds its version, and anything else is dropped, so the retry runs
    compare-and-set like a fresh call. Compare-and-set holds across processes;
    exact replay after a crash assumes one writing process per table.

    An uninterrupted write with a new operation id creates three table versions
    (pending receipt, apply write, complete receipt). Receipts are retained for
    replay; deleting them would not remove their history from Pixeltable.

    Args:
        table_name: Pixeltable table path (e.g. `'harness.memory'`).
    """

    def __init__(self, *, table_name: str = 'harness.memory') -> None:
        self._table_name = table_name
        self._lock = threading.RLock()
        self._table: pxt.Table | None = None

    @property
    def table(self) -> pxt.Table:
        """Underlying Pixeltable table for computed columns and queries."""
        with self._lock:
            return self._ensure_table()

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        return await anyio.to_thread.run_sync(self._locked, functools.partial(self._read_sync, path, max_chars))

    async def get_operation(self, operation: MemoryOperation) -> MemoryMutation | None:
        return await anyio.to_thread.run_sync(self._locked, functools.partial(self._get_operation_sync, operation))

    async def write(
        self,
        path: str,
        content: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        return await anyio.to_thread.run_sync(
            self._locked, functools.partial(self._mutate_sync, path, content, expected_version, operation)
        )

    async def delete(
        self,
        path: str,
        *,
        expected_version: str | None,
        operation: MemoryOperation | None = None,
    ) -> MemoryMutation:
        return await anyio.to_thread.run_sync(
            self._locked, functools.partial(self._mutate_sync, path, None, expected_version, operation)
        )

    async def list_paths(self, prefix: str = '', *, limit: int) -> list[str]:
        return await anyio.to_thread.run_sync(self._locked, functools.partial(self._list_paths_sync, prefix, limit))

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
        fetched = await anyio.to_thread.run_sync(
            self._locked, functools.partial(self._search_files_sync, prefix, max_files, max_file_chars)
        )
        files = [(str(row['path']), row['content'] or '') for row in fetched]
        result = await lexical_search(
            [(path, content[:max_file_chars]) for path, content in files],
            query,
            limit=limit,
            max_files=max_files,
            max_chars=max_chars,
            score_prefix=prefix,
        )
        content_truncated = any(len(content) > max_file_chars for _, content in files[:max_files])
        return MemorySearchResult(
            matches=result.matches, scanned=result.scanned, truncated=result.truncated or content_truncated
        )

    def _locked(self, fn: Callable[[], _T]) -> _T:
        with self._lock:
            try:
                return fn()
            except pxt.NotFoundError:
                # The cached handle goes stale if the table was dropped; recreate and retry once.
                # The retry is safe: the drop discarded every file and receipt, so the operation runs
                # again against the new, empty table exactly as a fresh call would.
                self._table = None
                return fn()

    def _ensure_table(self) -> pxt.Table:
        if self._table is not None:
            return self._table
        try:
            t = pxt.get_table(self._table_name)
        except pxt.NotFoundError:
            if parent := self._table_name.rpartition('.')[0]:
                pxt.create_dir(parent, parents=True, if_exists='ignore')
            # if_exists='ignore' can also return a table a concurrent writer just created; it is checked below.
            t = pxt.create_table(  # pyright: ignore[reportUnknownMemberType]
                self._table_name,
                # Bare types are non-nullable; `T | None` declares a nullable column.
                {
                    'path': pxt.String,
                    'kind': pxt.String,
                    'content': pxt.String | None,
                    'version': pxt.String | None,
                    'last_operation_id': pxt.String | None,
                    'fingerprint': pxt.String | None,
                    'existed': pxt.Bool | None,
                },
                primary_key='path',
                if_exists='ignore',
            )
        assert t is not None  # get_table's default if_not_exists='error' raises instead of returning None
        has_default_idxs = self._check_compatible_schema(t)
        # Automatic indexes already cover path and reject explicit B-tree indexes.
        # Otherwise add an exact-path index; the primary key only indexes left(path, 256).
        if not has_default_idxs:
            t.add_btree_index('path', idx_name=_PATH_INDEX, if_exists='ignore')
        self._table = t
        return t

    def _check_compatible_schema(self, t: pxt.Table) -> bool:
        """Reject incompatible tables and return whether Pixeltable manages their indexes."""
        metadata = t.get_metadata()
        if metadata['kind'] != 'table':
            raise ValueError(f'{self._table_name!r} is a {metadata["kind"]}; the memory store needs a writable table')
        columns = metadata['columns']
        problems: list[str] = []
        for name, expected in _SCHEMA_COLUMNS.items():
            info = columns.get(name)
            if info is None:
                problems.append(f'column {name!r} is missing')
                continue
            if info['is_computed']:
                problems.append(f'column {name!r} is computed; the store writes it directly')
                continue
            type_ = info['type_']
            if type_.removesuffix(' | None') != expected:
                problems.append(f'column {name!r} has type {type_!r}, expected {expected!r}')
            elif name in _NULLABLE_COLUMNS and not type_.endswith(' | None'):
                problems.append(f'column {name!r} is not nullable; the store writes None to it')
        for name, info in columns.items():
            if name not in _SCHEMA_COLUMNS and not info['is_computed'] and not info['type_'].endswith(' | None'):
                problems.append(f'column {name!r} is not nullable, and inserts never set it')
        if metadata['primary_key'] != ['path']:
            problems.append(f"primary key is {metadata['primary_key']!r}, expected ['path']")
        if problems:
            raise ValueError(
                f'{self._table_name!r} is not a memory table ({"; ".join(problems)}); drop and recreate it'
            )
        return metadata['has_default_idxs']

    def _file_state(self, t: pxt.Table, path: str) -> tuple[str | None, str | None]:
        """The file's version and the operation id that last wrote it, or `(None, None)` if absent."""
        query = t.where((t.path == path) & (t.kind == _KIND_FILE)).select(t.version, t.last_operation_id)
        assert isinstance(query, pxt.Query)
        rows = query.collect()
        if len(rows) == 0:
            return None, None
        return str(rows[0]['version']), rows[0]['last_operation_id']

    def _landed(self, t: pxt.Table, path: str, version: str | None, existed: bool) -> bool:
        """Whether a pending receipt's mutation is visible in the file.

        A write landed if the file holds its version. A delete of an existing file leaves only an
        absence, which another writer's delete also produces, so it never counts as landed: the
        retry finds the file gone and reports that instead of replaying.
        """
        if version is None:
            return not existed
        return self._file_state(t, path)[0] == version

    def _receipt(self, t: pxt.Table, receipt_path: str, file_path: str, version: str | None) -> Expr:
        """Match one pending receipt exactly, so a writer only settles the receipt it wrote."""
        return (t.path == receipt_path) & (t.content == file_path) & (t.version == version)

    def _settle_path(self, t: pxt.Table, path: str) -> None:
        """Settle receipts a crashed writer left pending on `path`, before another mutation moves it.

        A pending receipt names its file in `content`. A receipt whose mutation `_landed` is completed;
        any other is dropped, so a retry applies it under compare-and-set like a fresh call.
        """
        query = t.where((t.kind == _KIND_OP) & (t.content == path)).select(t.path, t.version, t.existed)
        assert isinstance(query, pxt.Query)
        for row in query.collect():
            version = None if row['version'] is None else str(row['version'])
            receipt = self._receipt(t, str(row['path']), path, version)
            if self._landed(t, path, version, bool(row['existed'])):
                t.update({'content': None}, where=receipt)
            else:
                t.delete(where=receipt)

    def _lookup_operation(self, t: pxt.Table, operation: MemoryOperation) -> MemoryMutation | None:
        """Return the receipt for `operation`, or `None` if it is missing or its mutation has not landed.

        A pending receipt is left in place here: its writer may still be running, and only a mutation of
        its file (`_settle_path`) can drop it.
        """
        receipt_path = _receipt_path(operation)
        query = t.where(t.path == receipt_path).select(t.fingerprint, t.version, t.existed, t.content)
        assert isinstance(query, pxt.Query)
        rows = query.collect()
        if len(rows) == 0:
            return None
        row = rows[0]
        if row['fingerprint'] != operation.fingerprint:
            raise MemoryOperationConflictError(f'operation id {operation.id!r} was reused with different arguments')
        version = None if row['version'] is None else str(row['version'])
        pending = row['content']
        if pending is not None:
            if not self._landed(t, pending, version, bool(row['existed'])):
                return None
            t.update({'content': None}, where=self._receipt(t, receipt_path, pending, version))
        return MemoryMutation(version=version, replayed=True, existed=bool(row['existed']))

    def _apply(
        self, t: pxt.Table, path: str, content: str | None, expected: str | None, values: dict[str, object]
    ) -> None:
        if expected is None:
            if content is not None:
                _insert_rows(t, [{'path': path, **values, 'fingerprint': None, 'existed': None}])
            return
        where = (t.path == path) & (t.kind == _KIND_FILE) & (t.version == expected)
        if content is None:
            changed = t.delete(where=where).row_count_stats.del_rows
        else:
            changed = t.update(values, where=where).row_count_stats.upd_rows
        if changed != 1:
            action = 'deleted' if content is None else 'written'
            raise MemoryConflictError(f'memory path {path!r} changed before it could be {action}')

    def _read_sync(self, path: str, max_chars: int) -> MemoryFile | None:
        _check_store_path(path)
        if max_chars <= 0:
            raise ValueError('max_chars must be positive')
        t = self._ensure_table()
        query = t.where((t.path == path) & (t.kind == _KIND_FILE)).select(
            content=t.content.slice(0, max_chars + 1), version=t.version, operation_id=t.last_operation_id
        )
        assert isinstance(query, pxt.Query)
        rows = query.collect()
        if not rows:
            return None
        row = rows[0]
        content = row['content'] or ''
        return MemoryFile(
            content=content[:max_chars],
            version=str(row['version']),
            operation_id=row['operation_id'],
            truncated=len(content) > max_chars,
        )

    def _get_operation_sync(self, operation: MemoryOperation) -> MemoryMutation | None:
        return self._lookup_operation(self._ensure_table(), operation)

    def _mutate_sync(
        self,
        path: str,
        content: str | None,
        expected_version: str | None,
        operation: MemoryOperation | None,
    ) -> MemoryMutation:
        """Apply a write or delete (`content=None`) with optional operation replay."""
        _check_store_path(path)
        t = self._ensure_table()
        self._settle_path(t, path)
        current, last_operation_id = self._file_state(t, path)
        if operation is not None:
            if (receipt := self._lookup_operation(t, operation)) is not None:
                return receipt
            if last_operation_id == operation.id:
                # The write landed but its receipt was lost (a concurrent writer dropped it, then this writer
                # stopped before restoring it). The file row records the operation id with the content.
                return MemoryMutation(version=current, replayed=True, existed=True)
        if current != expected_version:
            action = 'deleted' if content is None else 'written'
            raise MemoryConflictError(f'memory path {path!r} changed before it could be {action}')
        mutation = MemoryMutation(
            version=None if content is None else uuid.uuid4().hex,
            replayed=False,
            existed=expected_version is not None,
        )
        values: dict[str, object] = {
            'kind': _KIND_FILE,
            'content': content,
            'version': mutation.version,
            'last_operation_id': None if operation is None else operation.id,
        }
        if operation is None:
            self._apply(t, path, content, expected_version, values)
            return mutation

        receipt_path = _receipt_path(operation)
        receipt_row: dict[str, object] = {
            'path': receipt_path,
            'kind': _KIND_OP,
            'content': path,
            'version': mutation.version,
            'last_operation_id': None,
            'fingerprint': operation.fingerprint,
            'existed': mutation.existed,
        }
        # Pending first, so a crash before completion is settled by the next mutation of `path`.
        try:
            _insert_rows(t, [receipt_row])
        except MemoryConflictError:
            # A peer with the same operation id recorded its receipt first; replay it if it landed.
            if (receipt := self._lookup_operation(t, operation)) is not None:
                return receipt
            raise
        pending = self._receipt(t, receipt_path, path, mutation.version)
        try:
            self._apply(t, path, content, expected_version, values)
        except MemoryConflictError:
            t.delete(where=pending)
            raise
        if t.update({'content': None}, where=pending).row_count_stats.upd_rows == 0:
            # A peer running the same operation id dropped our pending receipt before we applied; restore it.
            try:
                _insert_rows(t, [{**receipt_row, 'content': None}])
            except MemoryConflictError:  # pragma: lax no cover - a same-id peer recorded a receipt in between
                pass
        return mutation

    def _file_rows(self, t: pxt.Table, where: Expr, limit: int, content_chars: int = 0) -> list[dict[str, Any]]:
        """Return bounded file rows in path order, slicing content in SQL."""
        columns = {'content': t.content.slice(0, content_chars)} if content_chars else {}
        query = t.where((t.kind == _KIND_FILE) & where).order_by(t.path).limit(limit)
        assert isinstance(query, pxt.Query)
        selected = query.select(t.path, **columns)
        assert isinstance(selected, pxt.Query)
        return list(selected.collect())

    def _list_paths_sync(self, prefix: str, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError('limit must be positive')
        t = self._ensure_table()
        return [str(row['path']) for row in self._file_rows(t, t.path.slice(0, len(prefix)) == prefix, limit)]

    def _search_files_sync(self, prefix: str, max_files: int, max_file_chars: int) -> list[dict[str, Any]]:
        t = self._ensure_table()
        where = (t.path.slice(0, len(prefix)) == prefix) & ~t.path.slice(len(prefix)).contains('/')
        # One extra file/character lets search report each truncation bound.
        return self._file_rows(t, where, max_files + 1, content_chars=max_file_chars + 1)

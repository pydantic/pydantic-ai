"""Pixeltable backend for Harness `Memory`, with file rows and operation receipts.

Pixeltable indexes string primary keys on `left(path, 256)`, so paths are capped at 255
characters. Its `slice(0, n)` translates to SQL `substr`, while `slice(stop=n)` returns
`n + 1` characters; prefix filters use the former. Compare-and-set relies on Pixeltable's
table lock and affected-row counts. Explicit `C` collation keeps listing and search in
code point order even when an external Postgres database uses another collation.
"""

from __future__ import annotations

import functools
import json
import threading
import uuid
from collections.abc import Callable
from typing import Any, TypeVar

import anyio.to_thread
import pixeltable as pxt
import sqlalchemy as sa
from typing_extensions import TypedDict

from pydantic_ai_harness.memory import (
    MemoryConflictError,
    MemoryFile,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemorySearchResult,
)
from pydantic_ai_harness.memory._store import lexical_search, validate_store_path, validate_store_prefix
from pydantic_ai_harness.pixeltable._types import column_base, is_nullable_type

_KIND_FILE = 'file'
_KIND_OP = 'op'
_OP_PREFIX = '__op__/'
# Set in a receipt row's (otherwise unused) last_operation_id by whoever rolls its intent forward.
_CLAIMED = 'claimed'
_RESERVED_ROOTS = frozenset({'__meta__', '__op__'})


def _reject_reserved_path(path: str) -> None:
    if path.split('/', 1)[0] in _RESERVED_ROOTS:
        raise ValueError(f'memory path {path!r} is reserved for store bookkeeping')


# The primary-key index covers left(path, 256): two longer paths whose first 256 characters
# match collide on insert. Cap paths below the boundary.
_MAX_PATH_CHARS = 255


def _check_store_path(path: str) -> None:
    validate_store_path(path)
    _reject_reserved_path(path)
    if len(path) > _MAX_PATH_CHARS:
        raise ValueError(f'memory path {path!r} exceeds {_MAX_PATH_CHARS} characters')


_MAX_OPERATION_ID_CHARS = _MAX_PATH_CHARS - len(_OP_PREFIX)


def _receipt_path(operation: MemoryOperation) -> str:
    # Receipts share the path key, so a longer id could collide with another id's receipt.
    if len(operation.id) > _MAX_OPERATION_ID_CHARS:
        raise ValueError(f'operation id exceeds {_MAX_OPERATION_ID_CHARS} characters')
    return f'{_OP_PREFIX}{operation.id}'


@pxt.udf  # pyright: ignore[reportUnknownMemberType]
def _code_point_order(path: str) -> str:
    """Sort key for `path` in code point order; only its SQL translation is used."""
    return path  # pragma: no cover


@_code_point_order.to_sql  # pyright: ignore[reportUnknownMemberType]
def _(path: sa.ColumnElement[str]) -> sa.ColumnElement[str]:
    # UTF-8 byte order under the "C" collation is code point order, the order Python sorts `str` in.
    return sa.collate(path, 'C')


_T = TypeVar('_T')


class _Intent(TypedDict):
    """Journaled mutation payload kept on a prepared `__op__` receipt until it completes."""

    file: str
    op: str
    expected: str | None
    new: str | None


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
    compare-and-set on the file row. Operation receipts are journaled in the same
    table under `__op__/`: the intended mutation is recorded before it is
    applied, so a crash between the two is rolled forward or detected as applied
    on the next lookup instead of double-applying. Paths whose first segment is
    `__meta__` or `__op__` are reserved.

    An uninterrupted write with a new operation id creates three table versions
    (record intent, apply write, clear intent). Receipts are retained for replay;
    deleting them would not remove their history from Pixeltable.

    Args:
        table_name: Pixeltable table path (e.g. `'harness.memory'`).
    """

    def __init__(self, table_name: str = 'harness.memory') -> None:
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
        return await anyio.to_thread.run_sync(
            self._locked,
            functools.partial(self._search_sync, prefix, query, limit, max_files, max_chars, max_file_chars),
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

    def _ensure_dirs(self) -> None:
        if '.' not in self._table_name:
            return
        parts = self._table_name.rsplit('.', 1)[0].split('.')
        acc: list[str] = []
        for part in parts:
            acc.append(part)
            pxt.create_dir('.'.join(acc), if_exists='ignore')

    def _ensure_table(self) -> pxt.Table:
        if self._table is not None:
            return self._table
        try:
            t = pxt.get_table(self._table_name)
        except pxt.NotFoundError:
            self._ensure_dirs()
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
            if column_base(type_) != expected:
                problems.append(f'column {name!r} has type {type_!r}, expected {expected!r}')
            elif name in _NULLABLE_COLUMNS and not is_nullable_type(type_):
                problems.append(f'column {name!r} is not nullable; the store writes None to it')
        for name, info in columns.items():
            if name not in _SCHEMA_COLUMNS and not info['is_computed'] and not is_nullable_type(info['type_']):
                problems.append(f'column {name!r} is not nullable, and inserts never set it')
        if metadata['primary_key'] != ['path']:
            problems.append(f"primary key is {metadata['primary_key']!r}, expected ['path']")
        if problems:
            raise ValueError(
                f'{self._table_name!r} is not a memory table ({"; ".join(problems)}); drop and recreate it'
            )
        return metadata['has_default_idxs']

    def _file_row(self, t: pxt.Table, path: str) -> dict[str, Any] | None:
        query = t.where((t.path == path) & (t.kind == _KIND_FILE)).select(t.content, t.version, t.last_operation_id)
        assert isinstance(query, pxt.Query)
        rows = query.collect()
        if len(rows) == 0:
            return None
        return rows[0]

    def _lookup_operation(
        self, t: pxt.Table, operation: MemoryOperation, *, withdraw_unapplied: bool = False
    ) -> MemoryMutation | None:
        query = t.where(t.path == _receipt_path(operation)).select(t.fingerprint, t.version, t.existed, t.content)
        assert isinstance(query, pxt.Query)
        rows = query.collect()
        if len(rows) == 0:
            return None
        row = rows[0]
        if row['fingerprint'] != operation.fingerprint:
            raise MemoryOperationConflictError(f'operation id {operation.id!r} was reused with different arguments')
        mutation = MemoryMutation(
            version=None if row['version'] is None else str(row['version']),
            replayed=True,
            existed=bool(row['existed']),
        )
        if row['content'] is not None:
            # Prepared but not completed: the mutation may or may not have landed; settle it first.
            if not self._recover_operation(t, operation, row['content'], mutation, withdraw_unapplied):
                # The intent was withdrawn or replaced before we could claim it; look again.
                return self._lookup_operation(t, operation)
        return mutation

    def _recover_operation(
        self, t: pxt.Table, operation: MemoryOperation, intent: str, receipt: MemoryMutation, withdraw: bool
    ) -> bool:
        """Settle a prepared receipt, claiming it before replay to prevent withdrawal.

        A failed writer withdraws its unclaimed intent; a peer can roll it forward.
        Unsettled claimed intents remain for recovery. A missing file settles a
        delete after a crash. Return `False` if the intent vanished before the claim.
        """
        receipt_row = (t.path == _receipt_path(operation)) & (t.kind == _KIND_OP) & (t.content == intent)
        recorded: _Intent = json.loads(intent)
        conflict = MemoryConflictError(f'memory path {recorded["file"]!r} changed during operation {operation.id!r}')
        unclaimed = receipt_row & (t.last_operation_id == None)  # noqa: E711 (SQL IS NULL)
        if withdraw and t.delete(where=unclaimed).row_count_stats.del_rows == 1:
            raise conflict
        row = self._file_row(t, recorded['file'])
        current = None if row is None else str(row['version'])
        applied = (not receipt.existed or current is None) if recorded['op'] == 'delete' else current == receipt.version
        if not applied and current == recorded['expected']:
            # Claim first: a writer withdrawing this intent deletes it only while unclaimed.
            if t.update({'last_operation_id': _CLAIMED}, where=receipt_row).row_count_stats.upd_rows != 1:
                return False
            try:
                self._apply_intent(t, operation, recorded, receipt)
            except MemoryConflictError:
                # A peer may have applied it concurrently; re-check before declaring a conflict.
                row = self._file_row(t, recorded['file'])
                current = None if row is None else str(row['version'])
                applied = (
                    (not receipt.existed or current is None)
                    if recorded['op'] == 'delete'
                    else current == receipt.version
                )
            else:
                applied = True
        if not applied:
            raise conflict
        self._complete_operation(t, operation, intent)
        return True

    def _apply_intent(
        self, t: pxt.Table, operation: MemoryOperation | None, recorded: _Intent, mutation: MemoryMutation
    ) -> None:
        path = recorded['file']
        if recorded['op'] == 'delete':
            if not mutation.existed:
                return
            status = t.delete(where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == recorded['expected']))
            if status.row_count_stats.del_rows != 1:
                raise MemoryConflictError(f'memory path {path!r} changed before it could be deleted')
        elif recorded['expected'] is None:
            _insert_rows(
                t,
                [
                    {
                        'path': path,
                        'kind': _KIND_FILE,
                        'content': recorded['new'],
                        'version': mutation.version,
                        'last_operation_id': operation.id if operation else None,
                        'fingerprint': None,
                        'existed': None,
                    }
                ],
            )
        else:
            status = t.update(
                {
                    'content': recorded['new'],
                    'version': mutation.version,
                    'last_operation_id': operation.id if operation else None,
                },
                where=(t.path == path) & (t.kind == _KIND_FILE) & (t.version == recorded['expected']),
            )
            if status.row_count_stats.upd_rows != 1:
                raise MemoryConflictError(f'memory path {path!r} changed before it could be written')

    def _prepare_operation(
        self,
        t: pxt.Table,
        operation: MemoryOperation,
        intent: str,
        mutation: MemoryMutation,
    ) -> MemoryMutation | None:
        try:
            _insert_rows(
                t,
                [
                    {
                        'path': _receipt_path(operation),
                        'kind': _KIND_OP,
                        'content': intent,
                        'version': mutation.version,
                        'last_operation_id': None,
                        'fingerprint': operation.fingerprint,
                        'existed': mutation.existed,
                    }
                ],
            )
        except MemoryConflictError:
            # Another writer already journaled this operation id; recover or replay its receipt.
            receipt = self._lookup_operation(t, operation)
            if receipt is None:
                raise MemoryConflictError(f'operation {operation.id!r} receipt vanished mid-flight') from None
            return receipt
        return None

    def _complete_operation(self, t: pxt.Table, operation: MemoryOperation, intent: str) -> None:
        """Drop the journaled intent; the mutation is durable, so the payload is no longer needed.

        Matching on `intent` keeps a late peer from completing a newer intent under the same id.
        """
        t.update(
            {'content': None},
            where=(t.path == _receipt_path(operation)) & (t.kind == _KIND_OP) & (t.content == intent),
        )

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
        if operation is not None:
            receipt = self._lookup_operation(t, operation)
            if receipt is not None:
                return receipt
        row = self._file_row(t, path)
        current = None if row is None else str(row['version'])
        if current != expected_version:
            action = 'deleted' if content is None else 'written'
            raise MemoryConflictError(f'memory path {path!r} changed before it could be {action}')
        mutation = MemoryMutation(
            version=None if content is None else uuid.uuid4().hex, replayed=False, existed=row is not None
        )
        recorded = _Intent(
            file=path, op='delete' if content is None else 'write', expected=expected_version, new=content
        )
        intent = json.dumps(recorded)
        if operation is not None:
            receipt = self._prepare_operation(t, operation, intent, mutation)
            if receipt is not None:
                return receipt
        try:
            self._apply_intent(t, operation, recorded, mutation)
        except MemoryConflictError:
            if operation is not None:
                # A peer sharing the operation id may have applied our intent; adopt it rather
                # than let a retry apply it twice. Otherwise the intent is withdrawn.
                receipt = self._lookup_operation(t, operation, withdraw_unapplied=True)
                if receipt is not None:
                    return receipt
            raise
        if operation is not None:
            self._complete_operation(t, operation, intent)
        return mutation

    def _file_rows(self, t: pxt.Table, prefix: str, limit: int, content_chars: int = 0) -> list[dict[str, Any]]:
        """Return bounded file rows under `prefix` in code point order, slicing content in SQL."""
        pred = t.kind == _KIND_FILE
        if prefix:
            pred = pred & (t.path.slice(0, len(prefix)) == prefix)
        columns = {'content': t.content.slice(0, content_chars)} if content_chars else {}
        query = t.where(pred).order_by(_code_point_order(t.path)).limit(limit)
        assert isinstance(query, pxt.Query)
        selected = query.select(t.path, **columns)
        assert isinstance(selected, pxt.Query)
        return list(selected.collect())

    def _list_paths_sync(self, prefix: str, limit: int) -> list[str]:
        validate_store_prefix(prefix)
        if limit <= 0:
            raise ValueError('limit must be positive')
        t = self._ensure_table()
        return [str(row['path']) for row in self._file_rows(t, prefix, limit)]

    def _search_sync(
        self,
        prefix: str,
        query: str,
        limit: int,
        max_files: int,
        max_chars: int,
        max_file_chars: int,
    ) -> MemorySearchResult:
        validate_store_prefix(prefix)
        if not query.split() or limit <= 0 or max_files <= 0 or max_chars <= 0 or max_file_chars <= 0:
            return MemorySearchResult(matches=[], scanned=0, truncated=False)
        t = self._ensure_table()
        # One extra character tells a file cut at max_file_chars from one that fits exactly.
        # One file past max_files lets lexical_search report the scan bound as truncated.
        fetched = self._file_rows(t, prefix, max_files + 1, content_chars=max_file_chars + 1)
        files = [(str(row['path']), row['content'] or '') for row in fetched]
        result = lexical_search(
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

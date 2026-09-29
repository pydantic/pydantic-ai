"""Contract, recovery, and race tests for `PixeltableMemoryStore`.

Races are staged deterministically: a hook on the store's own table handle runs a peer store
(through its public async API, from the store's worker thread) at the exact point the race
needs, then lets the store's call proceed.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Awaitable, Callable, Iterator
from functools import partial
from typing import Literal, NoReturn

import anyio
import anyio.from_thread
import pixeltable as pxt
import pytest
from pixeltable.exprs.expr import Expr
from sqlalchemy import Engine, event

from pydantic_ai_harness.memory import (
    MemoryConflictError,
    MemoryMutation,
    MemoryOperation,
    MemoryOperationConflictError,
    MemoryStore,
    SearchableMemoryStore,
)
from pydantic_ai_harness.pixeltable import PixeltableMemoryStore

from .support import create_table, insert_rows

RECEIPT_PREFIX = '__op__/'
MEMORY_COLUMNS: dict[str, object] = {
    'path': pxt.String,
    'kind': pxt.String,
    'content': pxt.String | None,
    'version': pxt.String | None,
    'last_operation_id': pxt.String | None,
    'fingerprint': pxt.String | None,
    'existed': pxt.Bool | None,
}

Around = Callable[[Callable[[], pxt.UpdateStatus]], pxt.UpdateStatus]
"""Runs around one intercepted table call; receives the real call as a thunk."""


@pytest.fixture
def root() -> Iterator[str]:
    name = f'harness_pxt_mem_{uuid.uuid4().hex[:8]}'
    pxt.create_dir(name)
    yield name
    pxt.drop_dir(name, force=True)


@pytest.fixture
def table_name(root: str) -> str:
    return f'{root}.memory'


@pytest.fixture
def store(table_name: str) -> PixeltableMemoryStore:
    return PixeltableMemoryStore(table_name=table_name)


def _intercept_insert(monkeypatch: pytest.MonkeyPatch, table: pxt.Table, *, receipt: bool, around: Around) -> None:
    """Route the first insert of a receipt row (or of a file row) on `table` through `around`."""
    real = table.insert  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    pending = [True]

    def insert(rows: list[dict[str, object]]) -> pxt.UpdateStatus:
        if pending and str(rows[0]['path']).startswith(RECEIPT_PREFIX) == receipt:
            pending.clear()
            return around(lambda: real(rows))
        return real(rows)

    monkeypatch.setattr(table, 'insert', insert)


def _intercept_update(monkeypatch: pytest.MonkeyPatch, table: pxt.Table, *, columns: set[str], around: Around) -> None:
    """Route the first update on `table` that sets exactly `columns` through `around`."""
    real = table.update
    pending = [True]

    def update(value_spec: dict[str, object], where: Expr | None = None) -> pxt.UpdateStatus:
        if pending and set(value_spec) == columns:
            pending.clear()
            return around(lambda: real(value_spec, where=where))
        return real(value_spec, where=where)

    monkeypatch.setattr(table, 'update', update)


def _intercept_delete(monkeypatch: pytest.MonkeyPatch, table: pxt.Table, *, around: Around) -> None:
    """Route the first delete on `table` through `around`."""
    real = table.delete
    pending = [True]

    def delete(where: Expr | None = None) -> pxt.UpdateStatus:
        if pending:
            pending.clear()
            return around(lambda: real(where=where))
        return real(where=where)

    monkeypatch.setattr(table, 'delete', delete)


def _drop_receipt(table: pxt.Table, operation: MemoryOperation) -> None:
    table.delete(where=table.path == f'{RECEIPT_PREFIX}{operation.id}')


def _insert_prepared_receipt(
    store: PixeltableMemoryStore,
    operation: MemoryOperation,
    *,
    file: str,
    op: Literal['write', 'delete'],
    expected: str | None,
    new: str | None,
    version: str | None,
    existed: bool,
) -> None:
    """Journal an intent in the prepared state, as a writer that crashed before applying it leaves it."""
    intent = json.dumps({'file': file, 'op': op, 'expected': expected, 'new': new})
    insert_rows(
        store.table,
        [
            {
                'path': f'{RECEIPT_PREFIX}{operation.id}',
                'kind': 'op',
                'content': intent,
                'version': version,
                'last_operation_id': None,
                'fingerprint': operation.fingerprint,
                'existed': existed,
            }
        ],
    )


def _synthetic_error() -> NoReturn:
    raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, 'synthetic insert failure')


class TestPixeltableMemoryStore:
    def test_implements_public_protocols(self, store: PixeltableMemoryStore) -> None:
        assert isinstance(store, MemoryStore)
        assert isinstance(store, SearchableMemoryStore)

    async def test_compare_and_set_contract(self, store: PixeltableMemoryStore) -> None:
        created = await store.write('notes/main.md', 'one', expected_version=None)
        assert created.version is not None
        assert not created.replayed
        assert not created.existed
        file = await store.read('notes/main.md', max_chars=1_000)
        assert file is not None
        assert file.content == 'one'
        assert file.version == created.version
        assert file.operation_id is None

        updated = await store.write('notes/main.md', 'two', expected_version=file.version)
        assert updated.version is not None
        assert updated.version != created.version
        assert updated.existed
        with pytest.raises(MemoryConflictError, match='changed before it could be written'):
            await store.write('notes/main.md', 'stale', expected_version=file.version)
        with pytest.raises(MemoryConflictError, match='changed before it could be deleted'):
            await store.delete('notes/main.md', expected_version=file.version)

        deleted = await store.delete('notes/main.md', expected_version=updated.version)
        assert deleted == MemoryMutation(version=None, replayed=False, existed=True)
        assert await store.read('notes/main.md', max_chars=1_000) is None

    async def test_versions_do_not_repeat_after_delete_and_recreate(self, store: PixeltableMemoryStore) -> None:
        first = await store.write('main.md', 'same', expected_version=None)
        await store.delete('main.md', expected_version=first.version)
        recreated = await store.write('main.md', 'same', expected_version=None)
        assert recreated.version != first.version
        with pytest.raises(MemoryConflictError):
            await store.write('main.md', 'stale', expected_version=first.version)
        with pytest.raises(MemoryConflictError):
            await store.delete('main.md', expected_version=first.version)

    async def test_read_and_listing_bounds(self, store: PixeltableMemoryStore) -> None:
        created = await store.write('a.md', '0123456789', expected_version=None)
        await store.write('b.md', 'b', expected_version=None)
        await store.write('c.md', 'c', expected_version=None)

        bounded = await store.read('a.md', max_chars=4)
        assert bounded is not None
        assert bounded.content == '0123'
        assert bounded.version == created.version
        assert bounded.truncated
        complete = await store.read('a.md', max_chars=20)
        assert complete is not None
        assert complete.content == '0123456789'
        assert not complete.truncated
        assert await store.list_paths(limit=2) == ['a.md', 'b.md']

    async def test_list_paths_prefix_isolation(self, store: PixeltableMemoryStore) -> None:
        await store.write('tenant-a/main.md', 'a', expected_version=None)
        await store.write('tenant-b/main.md', 'b', expected_version=None)
        await store.write('tenant-a/other.md', 'c', expected_version=None)
        assert await store.list_paths('tenant-a/', limit=10) == ['tenant-a/main.md', 'tenant-a/other.md']
        assert await store.list_paths('tenant-b/', limit=10) == ['tenant-b/main.md']

    async def test_operation_receipts(self, store: PixeltableMemoryStore) -> None:
        operation = MemoryOperation(id='run-1:call-1', fingerprint='write:notes/main.md:one')
        first = await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
        assert not first.replayed
        replay = await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
        assert replay == MemoryMutation(version=first.version, replayed=True, existed=False)
        assert await store.get_operation(operation) == replay
        file = await store.read('notes/main.md', max_chars=1_000)
        assert file is not None
        assert file.operation_id == operation.id

        with pytest.raises(MemoryOperationConflictError, match='reused with different arguments'):
            await store.get_operation(MemoryOperation(id=operation.id, fingerprint='different'))
        assert await store.get_operation(MemoryOperation(id='run-1:unknown', fingerprint='x')) is None

        update = MemoryOperation(id='run-1:call-2', fingerprint='write:notes/main.md:two')
        updated = await store.write('notes/main.md', 'two', expected_version=first.version, operation=update)
        assert updated.existed
        assert (await store.write('notes/main.md', 'two', expected_version=first.version, operation=update)).replayed

        delete_operation = MemoryOperation(id='run-1:call-3', fingerprint='delete:missing.md')
        deleted = await store.delete('missing.md', expected_version=None, operation=delete_operation)
        assert deleted == MemoryMutation(version=None, replayed=False, existed=False)
        assert (await store.delete('missing.md', expected_version=None, operation=delete_operation)).replayed

        existing = MemoryOperation(id='run-1:call-4', fingerprint='delete:notes/main.md')
        removed = await store.delete('notes/main.md', expected_version=updated.version, operation=existing)
        assert removed.existed
        assert await store.read('notes/main.md', max_chars=10) is None

    async def test_rejects_unsafe_reserved_and_long_paths(self, store: PixeltableMemoryStore) -> None:
        for path in ('../escape.md', '/absolute.md', 'a//b.md', 'a/../../b.md', 'a b.md'):
            with pytest.raises(ValueError, match='invalid memory path'):
                await store.read(path, max_chars=1_000)
        for path in ('__op__/run-1', '__meta__/generation'):
            with pytest.raises(ValueError, match='reserved'):
                await store.read(path, max_chars=10)
            with pytest.raises(ValueError, match='reserved'):
                await store.write(path, 'nope', expected_version=None)
            with pytest.raises(ValueError, match='reserved'):
                await store.delete(path, expected_version=None)
        # The primary-key index covers left(path, 256): longer paths collide once their first
        # 256 characters match, so the store refuses them. Segments stay within the 200-char limit.
        long_path = f'{"a" * 200}/{"b" * 200}'
        with pytest.raises(ValueError, match='255'):
            await store.write(long_path, 'x', expected_version=None)
        with pytest.raises(ValueError, match='255'):
            await store.read(long_path, max_chars=10)
        with pytest.raises(ValueError, match='255'):
            await store.delete(long_path, expected_version=None)
        # Receipts live at `__op__/<id>` under the same key, so ids stop 7 characters sooner.
        longest = MemoryOperation(id='i' * 248, fingerprint='write:a.md:x')
        assert not (await store.write('a.md', 'x', expected_version=None, operation=longest)).replayed
        too_long = MemoryOperation(id='i' * 249, fingerprint='write:b.md:x')
        with pytest.raises(ValueError, match='operation id exceeds 248'):
            await store.write('b.md', 'x', expected_version=None, operation=too_long)
        with pytest.raises(ValueError, match='operation id exceeds 248'):
            await store.delete('a.md', expected_version=None, operation=too_long)
        with pytest.raises(ValueError, match='operation id exceeds 248'):
            await store.get_operation(too_long)

    async def test_argument_validation(self, store: PixeltableMemoryStore) -> None:
        with pytest.raises(ValueError, match='max_chars must be positive'):
            await store.read('a.md', max_chars=0)
        with pytest.raises(ValueError, match='limit must be positive'):
            await store.list_paths(limit=0)
        with pytest.raises(ValueError):
            await store.list_paths('../x', limit=10)
        with pytest.raises(ValueError):
            await store.search('bad prefix/', 'q', limit=1, max_files=1, max_chars=1, max_file_chars=1)

    async def test_table_escape_hatch(self, store: PixeltableMemoryStore) -> None:
        created = await store.write('note.md', 'hello', expected_version=None)
        t = store.table
        query = t.where(t.kind == 'file').select(t.path, t.content)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        assert isinstance(query, pxt.Query)
        rows = query.collect()
        assert [(row['path'], row['content']) for row in rows] == [('note.md', 'hello')]

        # A direct content edit keeps the version, so compare-and-set still accepts it.
        t.update({'content': 'HACKED'}, where=t.path == 'note.md')
        file = await store.read('note.md', max_chars=100)
        assert file is not None
        assert file.content == 'HACKED'
        assert file.version == created.version
        assert (await store.write('note.md', 'restored', expected_version=file.version)).existed

    async def test_list_and_search_omit_bookkeeping(self, store: PixeltableMemoryStore) -> None:
        operation = MemoryOperation(id='run-1:call-1', fingerprint='write:a.md:alpha')
        await store.write('a.md', 'alpha', expected_version=None, operation=operation)
        assert await store.list_paths('', limit=50) == ['a.md']
        found = await store.search('', 'alpha', limit=10, max_files=10, max_chars=400, max_file_chars=1_000)
        assert [match.path for match in found.matches] == ['a.md']

    async def test_table_path_directories(self, root: str) -> None:
        nested = PixeltableMemoryStore(table_name=f'{root}.a.b.memory')
        await nested.write('x.md', 'x', expected_version=None)
        assert pxt.get_table(f'{root}.a.b.memory', if_not_exists='ignore') is not None

        top_level = f'harness_pxt_top_{uuid.uuid4().hex[:8]}'
        try:
            await PixeltableMemoryStore(table_name=top_level).write('x.md', 'x', expected_version=None)
            assert pxt.get_table(top_level, if_not_exists='ignore') is not None
        finally:
            pxt.drop_table(top_level, force=True, if_not_exists='ignore')

    async def test_dropped_table_is_recreated(self, store: PixeltableMemoryStore, table_name: str) -> None:
        await store.write('a.md', 'old', expected_version=None)
        pxt.drop_table(table_name)
        # The cached handle is stale; the store reopens (here: recreates) the table and retries.
        await store.write('a.md', 'new', expected_version=None)
        file = await store.read('a.md', max_chars=10)
        assert file is not None
        assert file.content == 'new'


class TestPixeltableMemoryStoreSchema:
    def test_existing_memory_table_is_reused(self, table_name: str) -> None:
        first = PixeltableMemoryStore(table_name=table_name).table
        second = PixeltableMemoryStore(table_name=table_name).table
        assert second.get_metadata()['columns'].keys() == first.get_metadata()['columns'].keys()

    def test_hand_built_table_is_accepted_and_indexed(self, root: str) -> None:
        name = f'{root}.manual'
        t = create_table(name, {**MEMORY_COLUMNS, 'note': pxt.String | None}, primary_key='path')
        t.add_computed_column(path_copy=t.path)
        # Tables created without the path index (e.g. by older releases) get it on first open.
        assert 'path_lookup_idx' in PixeltableMemoryStore(table_name=name).table.get_metadata()['indexes']

    @pytest.mark.parametrize(
        ('schema', 'primary_key', 'message'),
        [
            # Pre-release revisions used an Int version column.
            ({'path': pxt.String, 'version': pxt.Int | None}, 'path', "column 'version' has type"),
            # Correct column types but no primary key: compare-and-set would append duplicates.
            (MEMORY_COLUMNS, None, "primary key is None, expected \\['path'\\]"),
            ({'path': pxt.String, 'kind': pxt.String}, 'path', "column 'content' is missing"),
            # `__op__` receipts write None to `content`, so a non-nullable column rejects them.
            ({**MEMORY_COLUMNS, 'content': pxt.String}, 'path', "column 'content' is not nullable"),
            # Inserts never set an extra column, so a non-nullable one rejects them.
            ({**MEMORY_COLUMNS, 'extra': pxt.String}, 'path', 'inserts never set'),
        ],
    )
    def test_incompatible_existing_table_rejected(
        self, root: str, schema: dict[str, object], primary_key: str | None, message: str
    ) -> None:
        name = f'{root}.legacy'
        create_table(name, schema, primary_key=primary_key)
        with pytest.raises(ValueError, match='not a memory table') as error:
            _ = PixeltableMemoryStore(table_name=name).table
        error.match(message)

    def test_computed_store_column_rejected(self, root: str) -> None:
        # Inserts write `content`, so a computed column there rejects them.
        name = f'{root}.computed'
        columns = {key: value for key, value in MEMORY_COLUMNS.items() if key != 'content'}
        t = create_table(name, columns, primary_key='path')
        t.add_computed_column(content=t.version)
        with pytest.raises(ValueError, match="column 'content' is computed"):
            _ = PixeltableMemoryStore(table_name=name).table

    def test_view_rejected(self, root: str) -> None:
        # A view reports the same columns and primary key but rejects writes.
        base = create_table(f'{root}.base', MEMORY_COLUMNS, primary_key='path')
        pxt.create_view(f'{root}.view', base)
        with pytest.raises(ValueError, match='is a view; the memory store needs a writable table'):
            _ = PixeltableMemoryStore(table_name=f'{root}.view').table

    def test_create_path_still_validates(self, root: str, monkeypatch: pytest.MonkeyPatch) -> None:
        # `create_table(if_exists='ignore')` can return a concurrently created table; it is checked too.
        name = f'{root}.raced'
        create_table(name, {'a': pxt.Int})
        real_get_table = pxt.get_table

        def racing_get_table(path: str, if_not_exists: Literal['error', 'ignore'] = 'error') -> pxt.Table | None:
            # At this point in the race the table did not exist yet.
            return real_get_table(f'{path}_missing' if path == name else path, if_not_exists)

        monkeypatch.setattr(pxt, 'get_table', racing_get_table)
        with pytest.raises(ValueError, match="column 'path' is missing"):
            _ = PixeltableMemoryStore(table_name=name).table


class TestPixeltableMemoryStoreSearch:
    async def test_scoped_bounded_search(self, store: PixeltableMemoryStore) -> None:
        for path, content in (
            ('tenant-a/main/alpha.md', 'alpha alpha'),
            ('tenant-a/main/beta.md', 'alpha'),
            ('tenant-a/main/other.md', 'unrelated'),
            ('tenant-b/main/private.md', 'alpha alpha alpha'),
        ):
            await store.write(path, content, expected_version=None)

        result = await store.search(
            'tenant-a/main/', 'alpha', limit=10, max_files=10, max_chars=80, max_file_chars=1_000
        )
        assert [match.path for match in result.matches] == ['tenant-a/main/alpha.md', 'tenant-a/main/beta.md']
        assert result.scanned == 3
        assert not result.truncated
        assert sum(len(match.path) + len(match.snippet) for match in result.matches) <= 80

        bounded = await store.search(
            'tenant-a/main/', 'alpha', limit=10, max_files=1, max_chars=80, max_file_chars=1_000
        )
        assert bounded.scanned == 1
        assert bounded.truncated
        tiny = await store.search('tenant-a/main/', 'alpha', limit=10, max_files=10, max_chars=1, max_file_chars=1_000)
        assert tiny.matches == []
        assert tiny.truncated

    async def test_search_empty_and_invalid_bounds(self, store: PixeltableMemoryStore) -> None:
        await store.write('a.md', 'alpha', expected_version=None)
        for query, limit, max_files, max_chars, max_file_chars in (
            ('', 10, 10, 80, 1_000),
            ('alpha', 0, 10, 80, 1_000),
            ('alpha', 10, 0, 80, 1_000),
            ('alpha', 10, 10, 0, 1_000),
            ('alpha', 10, 10, 80, 0),
        ):
            empty = await store.search(
                '', query, limit=limit, max_files=max_files, max_chars=max_chars, max_file_chars=max_file_chars
            )
            assert empty.matches == []
            assert empty.scanned == 0
            assert not empty.truncated

    async def test_search_snippets_cover_tiny_and_offset_windows(self, store: PixeltableMemoryStore) -> None:
        await store.write('a', '012345alpha-tail', expected_version=None)
        tiny = await store.search('', 'alpha', limit=1, max_files=1, max_chars=3, max_file_chars=1_000)
        assert len(tiny.matches) == 1
        assert len(tiny.matches[0].snippet) == 2
        offset = await store.search('', 'alpha', limit=1, max_files=1, max_chars=12, max_file_chars=1_000)
        assert len(offset.matches) == 1
        assert offset.matches[0].snippet.startswith('...')

    async def test_list_paths_and_search_bound_prefix_in_table(self, store: PixeltableMemoryStore) -> None:
        for path, content in (
            ('tenant-a/main/a.md', 'alpha'),
            ('tenant-a/main/b.md', 'alpha'),
            ('tenant-a/main/c.md', 'alpha'),
            ('tenant-b/main/private.md', 'alpha alpha alpha'),
        ):
            await store.write(path, content, expected_version=None)

        assert await store.list_paths('tenant-a/main/', limit=2) == ['tenant-a/main/a.md', 'tenant-a/main/b.md']
        result = await store.search(
            'tenant-a/main/', 'alpha', limit=10, max_files=2, max_chars=200, max_file_chars=1_000
        )
        assert [match.path for match in result.matches] == ['tenant-a/main/a.md', 'tenant-a/main/b.md']
        assert result.scanned == 2
        assert result.truncated
        outsider = await store.search(
            'tenant-b/main/', 'alpha', limit=10, max_files=10, max_chars=200, max_file_chars=1_000
        )
        assert [match.path for match in outsider.matches] == ['tenant-b/main/private.md']

    async def test_list_paths_and_search_keep_code_point_order(self, store: PixeltableMemoryStore) -> None:
        paths = ['b.md', 'a.md', 'B.md', '_x.md', '-y.md']
        for path in paths:
            await store.write(path, 'alpha', expected_version=None)
        # Ordered under the "C" collation in SQL, so the order and the files a bound keeps do not depend on the
        # database collation.
        assert await store.list_paths(limit=10) == sorted(paths)
        assert await store.list_paths(limit=2) == sorted(paths)[:2]
        found = await store.search('', 'alpha', limit=10, max_files=2, max_chars=200, max_file_chars=100)
        assert [match.path for match in found.matches] == sorted(paths)[:2]
        assert found.truncated

    async def test_list_paths_and_search_push_order_and_bound_into_sql(self, store: PixeltableMemoryStore) -> None:
        # The bound must limit the rows the database returns, not trim a full scan in Python,
        # and the ordering must not depend on the database's default collation.
        for path in ('a.md', 'b.md', 'c.md'):
            await store.write(path, 'alpha', expected_version=None)
        statements: list[str] = []

        def record(*args: object) -> None:
            statement = args[2]
            if isinstance(statement, str) and 'ORDER BY' in statement:
                statements.append(statement)

        event.listen(Engine, 'before_cursor_execute', record)
        try:
            assert await store.list_paths(limit=1) == ['a.md']
            found = await store.search('', 'alpha', limit=10, max_files=1, max_chars=200, max_file_chars=100)
        finally:
            event.remove(Engine, 'before_cursor_execute', record)
        assert [match.path for match in found.matches] == ['a.md']
        assert len(statements) == 2
        for statement in statements:
            assert 'COLLATE "C"' in statement
            assert 'LIMIT' in statement

    async def test_search_bounds_each_file_and_ignores_namespace_prefix(self, store: PixeltableMemoryStore) -> None:
        namespace = 'n' * 180
        prefix = f'{namespace}/main/'
        await store.write(f'{prefix}note.md', 'prefix TARGET', expected_version=None)

        # Only the first 6 characters are searched, and the cut is reported.
        bounded = await store.search(prefix, 'target', limit=10, max_files=10, max_chars=100, max_file_chars=6)
        assert bounded.matches == []
        assert bounded.truncated
        cut_hit = await store.search(prefix, 'prefix', limit=10, max_files=10, max_chars=100, max_file_chars=6)
        assert [match.path for match in cut_hit.matches] == [f'{prefix}note.md']
        assert cut_hit.truncated
        namespace_result = await store.search(
            prefix, namespace, limit=10, max_files=10, max_chars=100, max_file_chars=100
        )
        assert namespace_result.matches == []
        visible = await store.search(prefix, 'target', limit=10, max_files=10, max_chars=20, max_file_chars=100)
        assert [match.path for match in visible.matches] == [f'{prefix}note.md']
        assert not visible.truncated


class TestPixeltableMemoryStoreRecovery:
    """Prepared receipts left by a writer that stopped between journaling and completing."""

    async def test_prepared_create_rolls_forward(self, store: PixeltableMemoryStore) -> None:
        operation = MemoryOperation(id='run-1:call-9', fingerprint='write:notes/a.md:hello')
        _insert_prepared_receipt(
            store, operation, file='notes/a.md', op='write', expected=None, new='hello', version='v1', existed=False
        )
        replay = await store.get_operation(operation)
        assert replay == MemoryMutation(version='v1', replayed=True, existed=False)
        file = await store.read('notes/a.md', max_chars=100)
        assert file is not None
        assert (file.content, file.version, file.operation_id) == ('hello', 'v1', operation.id)
        # The receipt is now complete; a second lookup is a plain replay.
        assert await store.get_operation(operation) == replay

    async def test_prepared_update_rolls_forward(self, store: PixeltableMemoryStore) -> None:
        base = await store.write('u.md', 'base', expected_version=None)
        operation = MemoryOperation(id='run-1:call-10', fingerprint='write:u.md:next')
        _insert_prepared_receipt(
            store, operation, file='u.md', op='write', expected=base.version, new='next', version='v2', existed=True
        )
        replay = await store.write('u.md', 'next', expected_version=base.version, operation=operation)
        assert replay == MemoryMutation(version='v2', replayed=True, existed=True)
        file = await store.read('u.md', max_chars=100)
        assert file is not None
        assert (file.content, file.version) == ('next', 'v2')

    async def test_prepared_delete_rolls_forward(self, store: PixeltableMemoryStore) -> None:
        created = await store.write('d.md', 'x', expected_version=None)
        operation = MemoryOperation(id='run-1:call-12', fingerprint='delete:d.md')
        _insert_prepared_receipt(
            store, operation, file='d.md', op='delete', expected=created.version, new=None, version=None, existed=True
        )
        result = await store.delete('d.md', expected_version=created.version, operation=operation)
        assert result == MemoryMutation(version=None, replayed=True, existed=True)
        assert await store.read('d.md', max_chars=10) is None

    async def test_prepared_receipt_after_apply_does_not_double_write(self, store: PixeltableMemoryStore) -> None:
        # The file mutation landed but the receipt never completed.
        operation = MemoryOperation(id='run-1:call-11', fingerprint='write:b.md:one')
        first = await store.write('b.md', 'one', expected_version=None, operation=operation)
        t = store.table
        intent = json.dumps({'file': 'b.md', 'op': 'write', 'expected': None, 'new': 'one'})
        t.update({'content': intent}, where=t.path == f'{RECEIPT_PREFIX}{operation.id}')
        replay = await store.write('b.md', 'one', expected_version=None, operation=operation)
        assert replay == MemoryMutation(version=first.version, replayed=True, existed=False)
        file = await store.read('b.md', max_chars=100)
        assert file is not None
        assert (file.content, file.version) == ('one', first.version)

    async def test_prepared_receipt_conflicts_when_path_moved(self, store: PixeltableMemoryStore) -> None:
        # The journaled expectation is stale: the path changed after the receipt was written.
        await store.write('c.md', 'v1', expected_version=None)
        operation = MemoryOperation(id='run-1:call-13', fingerprint='write:c.md:x')
        _insert_prepared_receipt(
            store, operation, file='c.md', op='write', expected=None, new='x', version='vx', existed=False
        )
        with pytest.raises(MemoryConflictError, match="changed during operation 'run-1:call-13'"):
            await store.write('c.md', 'x', expected_version=None, operation=operation)
        # A crash-recovered intent that cannot be settled stays and keeps blocking the id.
        with pytest.raises(MemoryConflictError):
            await store.get_operation(operation)

    async def test_intent_withdrawn_before_claim_is_looked_up_again(
        self, store: PixeltableMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='run-1:call-14', fingerprint='write:e.md:x')
        _insert_prepared_receipt(
            store, operation, file='e.md', op='write', expected=None, new='x', version='ve', existed=False
        )
        table = store.table

        def withdraw_first(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            _drop_receipt(table, operation)
            return call()

        _intercept_update(monkeypatch, table, columns={'last_operation_id'}, around=withdraw_first)
        assert await store.get_operation(operation) is None
        assert await store.read('e.md', max_chars=10) is None

    @pytest.mark.parametrize('kind', ['create', 'update', 'delete'])
    async def test_roll_forward_adopts_a_concurrent_recovery(
        self,
        store: PixeltableMemoryStore,
        table_name: str,
        monkeypatch: pytest.MonkeyPatch,
        kind: Literal['create', 'update', 'delete'],
    ) -> None:
        # Two recoverers settle the same intent: ours claims it, the peer applies it first, and
        # our apply's conflict is re-checked and recognized as the intent having landed.
        expected = None
        if kind != 'create':
            expected = (await store.write('r.md', 'base', expected_version=None)).version
        operation = MemoryOperation(id=f'run-1:call-15-{kind}', fingerprint=f'{kind}:r.md')
        op: Literal['write', 'delete'] = 'delete' if kind == 'delete' else 'write'
        version = None if kind == 'delete' else 'vr'
        new = None if kind == 'delete' else 'next'
        _insert_prepared_receipt(
            store, operation, file='r.md', op=op, expected=expected, new=new, version=version, existed=kind != 'create'
        )
        peer = PixeltableMemoryStore(table_name=table_name)

        def peer_applies_after_claim(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            status = call()
            assert anyio.from_thread.run(peer.get_operation, operation) is not None
            return status

        _intercept_update(monkeypatch, store.table, columns={'last_operation_id'}, around=peer_applies_after_claim)
        replay = await store.get_operation(operation)
        assert replay == MemoryMutation(version=version, replayed=True, existed=kind != 'create')
        file = await store.read('r.md', max_chars=10)
        if kind == 'delete':
            assert file is None
        else:
            assert file is not None
            assert (file.content, file.version) == ('next', 'vr')

    async def test_roll_forward_conflicts_when_a_writer_moves_the_path(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='run-1:call-16', fingerprint='write:m.md:x')
        _insert_prepared_receipt(
            store, operation, file='m.md', op='write', expected=None, new='x', version='vm', existed=False
        )
        other = PixeltableMemoryStore(table_name=table_name)

        def other_writes_after_claim(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            status = call()
            anyio.from_thread.run(partial(other.write, 'm.md', 'other', expected_version=None))
            return status

        _intercept_update(monkeypatch, store.table, columns={'last_operation_id'}, around=other_writes_after_claim)
        with pytest.raises(MemoryConflictError, match='changed during operation'):
            await store.get_operation(operation)
        file = await store.read('m.md', max_chars=10)
        assert file is not None
        assert file.content == 'other'


class TestPixeltableMemoryStoreRaces:
    async def test_concurrent_create_across_instances_is_conflict(
        self, store: PixeltableMemoryStore, table_name: str
    ) -> None:
        other = PixeltableMemoryStore(table_name=table_name)
        outcomes: list[str] = []

        async def create(target: PixeltableMemoryStore, content: str) -> None:
            try:
                await target.write('race.md', content, expected_version=None)
            except MemoryConflictError:
                outcomes.append('conflict')
            else:
                outcomes.append('ok')

        async with anyio.create_task_group() as tg:
            tg.start_soon(create, store, 'A')
            tg.start_soon(create, other, 'B')
        assert sorted(outcomes) == ['conflict', 'ok']
        file = await store.read('race.md', max_chars=10)
        assert file is not None
        assert file.content in {'A', 'B'}

    async def test_concurrent_same_operation_id_replays(self, store: PixeltableMemoryStore, table_name: str) -> None:
        other = PixeltableMemoryStore(table_name=table_name)
        operation = MemoryOperation(id='run-1:call-x', fingerprint='delete:missing.md')
        results: list[MemoryMutation] = []

        async def delete(target: PixeltableMemoryStore) -> None:
            results.append(await target.delete('missing.md', expected_version=None, operation=operation))

        async with anyio.create_task_group() as tg:
            tg.start_soon(delete, store)
            tg.start_soon(delete, other)
        assert sorted(result.replayed for result in results) == [False, True]
        assert all(not result.existed for result in results)

    async def test_cas_contention_has_one_winner_per_path(self, store: PixeltableMemoryStore) -> None:
        paths = [f'c{i}.md' for i in range(4)]
        versions = {path: (await store.write(path, 'init', expected_version=None)).version for path in paths}
        winners: list[str] = []

        async def attempt(path: str, i: int) -> None:
            try:
                await store.write(path, f'w{i}', expected_version=versions[path])
            except MemoryConflictError:
                return
            winners.append(path)

        async with anyio.create_task_group() as tg:
            for i in range(40):
                tg.start_soon(attempt, paths[i % 4], i)
        assert sorted(winners) == paths

    async def test_create_race_without_operation_is_conflict(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = PixeltableMemoryStore(table_name=table_name)

        def other_creates_first(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            anyio.from_thread.run(partial(other.write, 'race.md', 'B', expected_version=None))
            return call()

        _intercept_insert(monkeypatch, store.table, receipt=False, around=other_creates_first)
        with pytest.raises(MemoryConflictError, match='Duplicate primary key'):
            await store.write('race.md', 'A', expected_version=None)
        file = await store.read('race.md', max_chars=10)
        assert file is not None
        assert file.content == 'B'

    async def test_update_race_without_operation_is_conflict(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = await store.write('u.md', 'base', expected_version=None)
        other = PixeltableMemoryStore(table_name=table_name)

        def other_updates_first(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            anyio.from_thread.run(partial(other.write, 'u.md', 'B', expected_version=base.version))
            return call()

        columns = {'content', 'version', 'last_operation_id'}
        _intercept_update(monkeypatch, store.table, columns=columns, around=other_updates_first)
        with pytest.raises(MemoryConflictError, match='changed before it could be written'):
            await store.write('u.md', 'A', expected_version=base.version)
        file = await store.read('u.md', max_chars=10)
        assert file is not None
        assert file.content == 'B'

    async def test_delete_race_without_operation_is_conflict(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = await store.write('d.md', 'base', expected_version=None)
        other = PixeltableMemoryStore(table_name=table_name)

        def other_updates_first(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            anyio.from_thread.run(partial(other.write, 'd.md', 'B', expected_version=base.version))
            return call()

        _intercept_delete(monkeypatch, store.table, around=other_updates_first)
        with pytest.raises(MemoryConflictError, match='changed before it could be deleted'):
            await store.delete('d.md', expected_version=base.version)
        current = await store.read('d.md', max_chars=10)
        assert current is not None
        assert current.content == 'B'
        assert (await store.delete('d.md', expected_version=current.version)).existed

    async def test_insert_errors_other_than_duplicates_propagate(
        self, store: PixeltableMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _intercept_insert(monkeypatch, store.table, receipt=False, around=lambda call: _synthetic_error())
        with pytest.raises(pxt.RequestError, match='synthetic insert failure'):
            await store.write('a.md', 'x', expected_version=None)

    @pytest.mark.parametrize('mutation', ['write', 'delete'])
    async def test_journal_conflict_adopts_peer_receipt(
        self,
        store: PixeltableMemoryStore,
        table_name: str,
        monkeypatch: pytest.MonkeyPatch,
        mutation: Literal['write', 'delete'],
    ) -> None:
        # A peer sharing the operation id journals and completes first; our journal insert hits
        # its receipt and replays it instead of applying the mutation a second time.
        base = await store.write('j.md', 'base', expected_version=None)
        operation = MemoryOperation(id=f'run-1:call-17-{mutation}', fingerprint=f'{mutation}:j.md')
        peer = PixeltableMemoryStore(table_name=table_name)

        def mutate(target: PixeltableMemoryStore) -> Callable[[], Awaitable[MemoryMutation]]:
            if mutation == 'write':
                return partial(target.write, 'j.md', 'next', expected_version=base.version, operation=operation)
            return partial(target.delete, 'j.md', expected_version=base.version, operation=operation)

        async def peer_mutation() -> MemoryMutation:
            return await mutate(peer)()

        def peer_completes_first(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            assert not anyio.from_thread.run(peer_mutation).replayed
            return call()

        _intercept_insert(monkeypatch, store.table, receipt=True, around=peer_completes_first)
        outcome = await mutate(store)()
        assert outcome.replayed
        assert outcome.existed
        file = await store.read('j.md', max_chars=10)
        if mutation == 'write':
            assert file is not None
            assert file.content == 'next'
        else:
            assert file is None

    async def test_journal_conflict_with_vanished_receipt(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='run-1:call-18', fingerprint='write:v.md:x')
        peer = PixeltableMemoryStore(table_name=table_name)
        table = store.table

        def peer_journals_then_receipt_vanishes(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            anyio.from_thread.run(partial(peer.write, 'v.md', 'x', expected_version=None, operation=operation))
            try:
                return call()
            finally:
                _drop_receipt(table, operation)

        _intercept_insert(monkeypatch, table, receipt=True, around=peer_journals_then_receipt_vanishes)
        with pytest.raises(MemoryConflictError, match='receipt vanished mid-flight'):
            await store.write('v.md', 'x', expected_version=None, operation=operation)

    async def test_conflicted_create_adopts_peer_receipt(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # We journal the intent, a peer sharing the operation id applies and completes it, then
        # our own insert conflicts. We adopt the peer's receipt rather than retry into a double apply.
        operation = MemoryOperation(id='run-1:call-20', fingerprint='write:p.md:c2')
        peer = PixeltableMemoryStore(table_name=table_name)

        def peer_applies_our_intent(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            status = call()
            replay = anyio.from_thread.run(
                partial(peer.write, 'p.md', 'c2', expected_version=None, operation=operation)
            )
            assert replay.replayed
            return status

        _intercept_insert(monkeypatch, store.table, receipt=True, around=peer_applies_our_intent)
        outcome = await store.write('p.md', 'c2', expected_version=None, operation=operation)
        assert outcome.replayed
        file = await store.read('p.md', max_chars=100)
        assert file is not None
        assert file.content == 'c2'

    async def test_conflicted_delete_adopts_peer_receipt(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = await store.write('p.md', 'base', expected_version=None)
        operation = MemoryOperation(id='run-1:call-22', fingerprint='delete:p.md')
        peer = PixeltableMemoryStore(table_name=table_name)

        def peer_applies_our_intent(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            status = call()
            replay = anyio.from_thread.run(
                partial(peer.delete, 'p.md', expected_version=base.version, operation=operation)
            )
            assert replay.replayed
            return status

        _intercept_insert(monkeypatch, store.table, receipt=True, around=peer_applies_our_intent)
        outcome = await store.delete('p.md', expected_version=base.version, operation=operation)
        assert outcome == MemoryMutation(version=None, replayed=True, existed=True)
        assert await store.read('p.md', max_chars=100) is None

    @pytest.mark.parametrize('mutation', ['write', 'delete'])
    async def test_conflicted_mutation_withdraws_prepared_receipt(
        self,
        store: PixeltableMemoryStore,
        table_name: str,
        monkeypatch: pytest.MonkeyPatch,
        mutation: Literal['write', 'delete'],
    ) -> None:
        # An unrelated writer moves the path after we journal. Our mutation never landed, so the
        # intent is withdrawn and a retry with the same operation id applies cleanly.
        base = await store.write('q.md', 'base', expected_version=None)
        operation = MemoryOperation(id=f'run-1:call-21-{mutation}', fingerprint=f'{mutation}:q.md')
        other = PixeltableMemoryStore(table_name=table_name)

        def other_moves_the_path(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            status = call()
            anyio.from_thread.run(partial(other.write, 'q.md', 'peer', expected_version=base.version))
            return status

        _intercept_insert(monkeypatch, store.table, receipt=True, around=other_moves_the_path)
        with pytest.raises(MemoryConflictError, match='changed during operation'):
            if mutation == 'write':
                await store.write('q.md', 'ours', expected_version=base.version, operation=operation)
            else:
                await store.delete('q.md', expected_version=base.version, operation=operation)
        assert await store.get_operation(operation) is None

        current = await store.read('q.md', max_chars=100)
        assert current is not None
        assert current.content == 'peer'
        if mutation == 'write':
            retried = await store.write('q.md', 'ours', expected_version=current.version, operation=operation)
            assert not retried.replayed
            after = await store.read('q.md', max_chars=100)
            assert after is not None
            assert after.content == 'ours'
        else:
            retried = await store.delete('q.md', expected_version=current.version, operation=operation)
            assert not retried.replayed
            assert await store.read('q.md', max_chars=100) is None

    async def test_conflicted_delete_is_not_credited_with_another_delete(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An unrelated writer deletes the path after we journal. The file is gone, but our delete
        # removed nothing and no peer claimed the intent, so only one of the two deletes succeeds.
        base = await store.write('g.md', 'base', expected_version=None)
        operation = MemoryOperation(id='run-1:call-32', fingerprint='delete:g.md')
        other = PixeltableMemoryStore(table_name=table_name)

        def other_deletes_the_path(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            status = call()
            assert anyio.from_thread.run(partial(other.delete, 'g.md', expected_version=base.version)).existed
            return status

        _intercept_insert(monkeypatch, store.table, receipt=True, around=other_deletes_the_path)
        with pytest.raises(MemoryConflictError, match='changed during operation'):
            await store.delete('g.md', expected_version=base.version, operation=operation)
        assert await store.get_operation(operation) is None

    @pytest.mark.parametrize('mutation', ['write', 'delete'])
    async def test_withdraw_never_drops_an_intent_a_peer_applied(
        self,
        store: PixeltableMemoryStore,
        table_name: str,
        monkeypatch: pytest.MonkeyPatch,
        mutation: Literal['write', 'delete'],
    ) -> None:
        # A peer replaying our operation id applies the intent but has not completed it, then an
        # unrelated writer lands on top. Our compare-and-set fails and the file no longer shows our
        # version, yet the intent did land: the peer's claim must stop the withdraw, or a retry
        # would apply it twice.
        base = await store.write('w.md', 'base\n', expected_version=None)
        operation = MemoryOperation(id=f'run-1:call-30-{mutation}', fingerprint=f'{mutation}:w.md')
        peer = PixeltableMemoryStore(table_name=table_name)
        other = PixeltableMemoryStore(table_name=table_name)
        deferred: list[Callable[[], pxt.UpdateStatus]] = []

        def defer_completion(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            deferred.append(call)
            return pxt.UpdateStatus()

        def peer_rolls_forward_then_other_writes(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            status = call()
            assert anyio.from_thread.run(peer.get_operation, operation) is not None
            current = anyio.from_thread.run(partial(other.read, 'w.md', max_chars=100))
            if mutation == 'write':
                assert current is not None
                anyio.from_thread.run(partial(other.write, 'w.md', 'other\n', expected_version=current.version))
            else:
                assert current is None
                anyio.from_thread.run(partial(other.write, 'w.md', 'recreated\n', expected_version=None))
            return status

        _intercept_update(monkeypatch, peer.table, columns={'content'}, around=defer_completion)
        _intercept_insert(monkeypatch, store.table, receipt=True, around=peer_rolls_forward_then_other_writes)
        with pytest.raises(MemoryConflictError, match='changed during operation'):
            if mutation == 'write':
                await store.write('w.md', 'base\nfact\n', expected_version=base.version, operation=operation)
            else:
                await store.delete('w.md', expected_version=base.version, operation=operation)
        (complete,) = deferred
        complete()

        replay = await store.get_operation(operation)
        assert replay is not None
        assert replay.replayed  # the retry replays; it does not re-apply
        current = await store.read('w.md', max_chars=100)
        assert current is not None
        assert current.content == ('other\n' if mutation == 'write' else 'recreated\n')

    @pytest.mark.parametrize('mutation', ['write', 'delete'])
    async def test_conflicted_mutation_with_vanished_receipt(
        self,
        store: PixeltableMemoryStore,
        table_name: str,
        monkeypatch: pytest.MonkeyPatch,
        mutation: Literal['write', 'delete'],
    ) -> None:
        # Our receipt is gone by the time our compare-and-set fails: there is nothing to adopt, so
        # the conflict itself surfaces.
        base = await store.write('x.md', 'base', expected_version=None)
        operation = MemoryOperation(id=f'run-1:call-31-{mutation}', fingerprint=f'{mutation}:x.md')
        other = PixeltableMemoryStore(table_name=table_name)
        table = store.table

        def receipt_vanishes_and_other_writes(call: Callable[[], pxt.UpdateStatus]) -> pxt.UpdateStatus:
            status = call()
            _drop_receipt(table, operation)
            anyio.from_thread.run(partial(other.write, 'x.md', 'other', expected_version=base.version))
            return status

        _intercept_insert(monkeypatch, table, receipt=True, around=receipt_vanishes_and_other_writes)
        with pytest.raises(MemoryConflictError, match='changed before it could be'):
            if mutation == 'write':
                await store.write('x.md', 'ours', expected_version=base.version, operation=operation)
            else:
                await store.delete('x.md', expected_version=base.version, operation=operation)
        assert await store.get_operation(operation) is None

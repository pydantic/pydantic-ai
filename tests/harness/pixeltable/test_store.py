"""Pixeltable-specific tests for `PixeltableMemoryStore`; the shared contract is in `tests/harness/memory/test_stores.py`.

Crashes and races are staged deterministically: a hook on the store's own table handle fails a call
or runs a peer store (through its public async API, from the store's worker thread) at the exact
point the scenario needs.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from functools import partial
from typing import Literal, NoReturn

import anyio.from_thread
import pytest

from tests.conftest import try_import

with try_import() as imports_successful:
    import pixeltable as pxt

    from pydantic_ai_harness.memory import (
        MemoryConflictError,
        MemoryMutation,
        MemoryOperation,
        PixeltableMemoryStore,
    )

    from .support import create_table

    MEMORY_COLUMNS: dict[str, object] = {
        'path': pxt.String,
        'kind': pxt.String,
        'content': pxt.String | None,
        'version': pxt.String | None,
        'last_operation_id': pxt.String | None,
        'fingerprint': pxt.String | None,
        'existed': pxt.Bool | None,
    }

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pixeltable not installed')

Around = Callable[[Callable[[], object]], object]
"""Runs around one intercepted table call; receives the real call as a thunk."""


@pytest.fixture
def table_name(root: str) -> str:
    return f'{root}.memory'


@pytest.fixture
def store(table_name: str) -> PixeltableMemoryStore:
    return PixeltableMemoryStore(table_name=table_name)


def _intercept(
    monkeypatch: pytest.MonkeyPatch,
    table: pxt.Table,
    method: Literal['insert', 'update', 'delete'],
    *,
    receipt: bool,
    around: Around,
) -> None:
    """Route the first `method` call on `table` that targets a receipt row (or a file row) through `around`."""
    real = getattr(table, method)
    pending = [True]

    def call(*args: object, **kwargs: object) -> object:
        where = kwargs.get('where')
        if method == 'insert':
            targets_receipt = str(args[0][0]['path']).startswith('__op__/')  # pyright: ignore[reportIndexIssue, reportUnknownArgumentType]
        else:
            targets_receipt = '__op__/' in str(where)
        if pending and targets_receipt == receipt:
            pending.clear()
            return around(lambda: real(*args, **kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(table, method, call)


def _crash(call: Callable[[], object]) -> NoReturn:
    raise RuntimeError('crashed')


def _synthetic_error(call: Callable[[], object]) -> NoReturn:
    raise pxt.RequestError(pxt.ErrorCode.INVALID_ARGUMENT, 'synthetic insert failure')


class TestPixeltableMemoryStore:
    async def test_rejects_reserved_and_long_paths(self, store: PixeltableMemoryStore) -> None:
        with pytest.raises(ValueError, match='reserved'):
            await store.read('__op__/run-1', max_chars=10)
        with pytest.raises(ValueError, match='reserved'):
            await store.write('__op__/run-1', 'nope', expected_version=None)
        with pytest.raises(ValueError, match='reserved'):
            await store.delete('__op__/run-1', expected_version=None)
        # The primary-key index covers left(path, 256): longer paths collide once their first
        # 256 characters match, so the store refuses them. Segments stay within the 200-char limit.
        long_path = f'{"a" * 200}/{"b" * 200}'
        with pytest.raises(ValueError, match='255'):
            await store.write(long_path, 'x', expected_version=None)
        with pytest.raises(ValueError, match='255'):
            await store.read(long_path, max_chars=10)
        with pytest.raises(ValueError, match='255'):
            await store.delete(long_path, expected_version=None)

    async def test_rejects_long_operation_ids(self, store: PixeltableMemoryStore) -> None:
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

    async def test_table_is_queryable(self, store: PixeltableMemoryStore) -> None:
        await store.write('note.md', 'hello', expected_version=None)
        t = store.table
        query = t.where(t.kind == 'file').select(t.path, t.content)
        assert isinstance(query, pxt.Query)
        assert [(row['path'], row['content']) for row in query.collect()] == [('note.md', 'hello')]

    async def test_list_and_search_omit_receipts(self, store: PixeltableMemoryStore) -> None:
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
    def test_hand_built_table_is_accepted_and_indexed(self, root: str) -> None:
        name = f'{root}.manual'
        t = create_table(name, {**MEMORY_COLUMNS, 'note': pxt.String | None}, primary_key='path')
        t.add_computed_column(path_copy=t.path)
        # A compatible table created without the path index gets it on first open.
        assert 'path_lookup_idx' in PixeltableMemoryStore(table_name=name).table.get_metadata()['indexes']

    async def test_hand_built_table_with_default_indexes(self, root: str) -> None:
        name = f'{root}.indexed'
        create_table(name, MEMORY_COLUMNS, primary_key='path', has_default_idxs=True)
        store = PixeltableMemoryStore(table_name=name)

        created = await store.write('a.md', 'hello', expected_version=None)
        file = await store.read('a.md', max_chars=10)
        assert file is not None
        assert (file.content, file.version) == ('hello', created.version)

    @pytest.mark.parametrize(
        ('schema', 'primary_key', 'message'),
        [
            # Versions are opaque strings, so another type is rejected.
            ({'path': pxt.String, 'version': pxt.Int | None}, 'path', "column 'version' has type"),
            # Correct column types but no primary key: compare-and-set would append duplicates.
            (MEMORY_COLUMNS, None, "primary key is None, expected \\['path'\\]"),
            ({'path': pxt.String, 'kind': pxt.String}, 'path', "column 'content' is missing"),
            # Completed `__op__` receipts write None to `content`, so a non-nullable column rejects them.
            ({**MEMORY_COLUMNS, 'content': pxt.String}, 'path', "column 'content' is not nullable"),
            # Inserts never set an extra column, so a non-nullable one rejects them.
            ({**MEMORY_COLUMNS, 'extra': pxt.String}, 'path', 'inserts never set'),
        ]
        if imports_successful()
        else [],
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


class TestPixeltableMemoryStoreRecovery:
    """A writer that crashed between its pending receipt and completing it."""

    async def test_write_that_landed_is_replayed(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = await store.write('a.md', 'base', expected_version=None)
        operation = MemoryOperation(id='crash-after-write', fingerprint='write:a.md:next')
        _intercept(monkeypatch, store.table, 'update', receipt=True, around=_crash)
        with pytest.raises(RuntimeError, match='crashed'):
            await store.write('a.md', 'next', expected_version=base.version, operation=operation)

        restarted = PixeltableMemoryStore(table_name=table_name)
        replay = await restarted.get_operation(operation)
        file = await restarted.read('a.md', max_chars=10)
        assert file is not None
        assert replay == MemoryMutation(version=file.version, replayed=True, existed=True)
        assert file.content == 'next'

    async def test_delete_that_landed_is_not_replayed(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Absence cannot tell this delete from another writer's, so the retry finds the file gone instead.
        base = await store.write('a.md', 'base', expected_version=None)
        operation = MemoryOperation(id='crash-after-delete', fingerprint='delete:a.md')
        _intercept(monkeypatch, store.table, 'update', receipt=True, around=_crash)
        with pytest.raises(RuntimeError, match='crashed'):
            await store.delete('a.md', expected_version=base.version, operation=operation)

        restarted = PixeltableMemoryStore(table_name=table_name)
        assert await restarted.get_operation(operation) is None
        assert await restarted.read('a.md', max_chars=10) is None

    async def test_mutation_that_never_landed_is_applied_by_the_retry(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='crash-before-write', fingerprint='write:a.md:hello')
        _intercept(monkeypatch, store.table, 'insert', receipt=False, around=_crash)
        with pytest.raises(RuntimeError, match='crashed'):
            await store.write('a.md', 'hello', expected_version=None, operation=operation)

        restarted = PixeltableMemoryStore(table_name=table_name)
        retried = await restarted.write('a.md', 'hello', expected_version=None, operation=operation)
        assert not retried.replayed
        file = await restarted.read('a.md', max_chars=10)
        assert file is not None
        assert (file.content, file.version) == ('hello', retried.version)
        assert await restarted.get_operation(operation) == MemoryMutation(
            version=retried.version, replayed=True, existed=False
        )

    async def test_pending_receipt_is_dropped_once_another_writer_moves_the_path(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='crash-then-other-writer', fingerprint='write:a.md:hello')
        _intercept(monkeypatch, store.table, 'insert', receipt=False, around=_crash)
        with pytest.raises(RuntimeError, match='crashed'):
            await store.write('a.md', 'hello', expected_version=None, operation=operation)
        other = PixeltableMemoryStore(table_name=table_name)
        await other.write('a.md', 'other', expected_version=None)

        assert await other.get_operation(operation) is None
        with pytest.raises(MemoryConflictError, match='changed before it could be written'):
            await other.write('a.md', 'hello', expected_version=None, operation=operation)
        file = await other.read('a.md', max_chars=10)
        assert file is not None
        assert file.content == 'other'

    async def test_write_with_a_lost_receipt_is_replayed_from_the_file_row(self, store: PixeltableMemoryStore) -> None:
        operation = MemoryOperation(id='run-1:call-1', fingerprint='write:notes/main.md:one')
        first = await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
        table = store.table
        table.delete(where=table.path == f'__op__/{operation.id}')
        # The retry repeats the original call; the file row records the operation id, so it is not applied again.
        replay = await store.write('notes/main.md', 'one', expected_version=None, operation=operation)
        assert replay == MemoryMutation(version=first.version, replayed=True, existed=False)

    async def test_landed_mutation_is_replayed_after_another_writer(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='crash-then-append', fingerprint='write:a.md:x')
        _intercept(monkeypatch, store.table, 'update', receipt=True, around=_crash)
        with pytest.raises(RuntimeError, match='crashed'):
            await store.write('a.md', 'x', expected_version=None, operation=operation)
        other = PixeltableMemoryStore(table_name=table_name)
        landed = await other.read('a.md', max_chars=10)
        assert landed is not None
        await other.write('a.md', 'x+y', expected_version=landed.version)

        # The other writer's mutation completed the pending receipt, so the retry replays it.
        assert await other.get_operation(operation) == MemoryMutation(
            version=landed.version, replayed=True, existed=False
        )


class TestPixeltableMemoryStoreRaces:
    @pytest.mark.parametrize(
        ('mutation', 'method', 'error'),
        [
            ('create', 'insert', 'Duplicate primary key'),
            ('write', 'update', 'changed before it could be written'),
            ('delete', 'delete', 'changed before it could be deleted'),
        ],
    )
    async def test_mutation_race_without_operation_is_conflict(
        self,
        store: PixeltableMemoryStore,
        table_name: str,
        monkeypatch: pytest.MonkeyPatch,
        mutation: str,
        method: Literal['insert', 'update', 'delete'],
        error: str,
    ) -> None:
        base = await store.write('race.md', 'base', expected_version=None) if mutation != 'create' else None
        expected = base.version if base else None
        other = PixeltableMemoryStore(table_name=table_name)

        def other_writes_first(call: Callable[[], object]) -> object:
            anyio.from_thread.run(partial(other.write, 'race.md', 'B', expected_version=expected))
            return call()

        _intercept(monkeypatch, store.table, method, receipt=False, around=other_writes_first)
        with pytest.raises(MemoryConflictError, match=error):
            if mutation == 'delete':
                await store.delete('race.md', expected_version=expected)
            else:
                await store.write('race.md', 'A', expected_version=expected)
        file = await store.read('race.md', max_chars=10)
        assert file is not None
        assert file.content == 'B'

    async def test_insert_errors_other_than_duplicates_propagate(
        self, store: PixeltableMemoryStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _intercept(monkeypatch, store.table, 'insert', receipt=False, around=_synthetic_error)
        with pytest.raises(pxt.RequestError, match='synthetic insert failure'):
            await store.write('a.md', 'x', expected_version=None)

    async def test_peer_with_the_same_operation_id_is_replayed(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='shared-op', fingerprint='write:a.md:hello')
        peer = PixeltableMemoryStore(table_name=table_name)
        landed: list[MemoryMutation] = []

        def peer_completes_first(call: Callable[[], object]) -> object:
            landed.append(
                anyio.from_thread.run(partial(peer.write, 'a.md', 'hello', expected_version=None, operation=operation))
            )
            return call()

        _intercept(monkeypatch, store.table, 'insert', receipt=True, around=peer_completes_first)
        replay = await store.write('a.md', 'hello', expected_version=None, operation=operation)
        assert replay == MemoryMutation(version=landed[0].version, replayed=True, existed=False)

    async def test_peer_that_has_not_landed_is_a_conflict(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='shared-op', fingerprint='write:a.md:hello')
        peer = PixeltableMemoryStore(table_name=table_name)

        def peer_crashes_after_its_receipt(call: Callable[[], object]) -> object:
            _intercept(monkeypatch, peer.table, 'insert', receipt=False, around=_crash)
            with pytest.raises(RuntimeError, match='crashed'):
                anyio.from_thread.run(partial(peer.write, 'a.md', 'hello', expected_version=None, operation=operation))
            # Our receipt insert now collides with the peer's; the peer's write never landed, so it is dropped.
            return call()

        _intercept(monkeypatch, store.table, 'insert', receipt=True, around=peer_crashes_after_its_receipt)
        with pytest.raises(MemoryConflictError, match='Duplicate primary key'):
            await store.write('a.md', 'hello', expected_version=None, operation=operation)
        assert await store.get_operation(operation) is None
        assert not (await store.write('a.md', 'hello', expected_version=None, operation=operation)).replayed

    async def test_conflict_withdraws_the_pending_receipt(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='conflicted', fingerprint='write:a.md:hello')
        other = PixeltableMemoryStore(table_name=table_name)

        def other_writes_first(call: Callable[[], object]) -> object:
            anyio.from_thread.run(partial(other.write, 'a.md', 'other', expected_version=None))
            return call()

        _intercept(monkeypatch, store.table, 'insert', receipt=False, around=other_writes_first)
        with pytest.raises(MemoryConflictError, match='Duplicate primary key'):
            await store.write('a.md', 'hello', expected_version=None, operation=operation)
        assert await store.get_operation(operation) is None

    async def test_lookup_leaves_an_in_flight_receipt(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        operation = MemoryOperation(id='in-flight', fingerprint='write:a.md:x')
        peer = PixeltableMemoryStore(table_name=table_name)
        seen: list[MemoryMutation | None] = []

        def peer_looks_up_first(call: Callable[[], object]) -> object:
            seen.append(anyio.from_thread.run(peer.get_operation, operation))
            return call()

        _intercept(monkeypatch, store.table, 'insert', receipt=False, around=peer_looks_up_first)
        written = await store.write('a.md', 'x', expected_version=None, operation=operation)
        assert seen == [None]
        assert await peer.get_operation(operation) == MemoryMutation(
            version=written.version, replayed=True, existed=False
        )

    async def test_receipt_dropped_by_another_writer_is_restored(
        self, store: PixeltableMemoryStore, table_name: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = await store.write('a.md', 'base', expected_version=None)
        operation = MemoryOperation(id='restored', fingerprint='write:a.md:x')
        other = PixeltableMemoryStore(table_name=table_name)

        def stale_writer_settles_first(call: Callable[[], object]) -> object:
            # Its mutation drops our pending receipt, then fails compare-and-set on a stale version.
            with pytest.raises(MemoryConflictError):
                anyio.from_thread.run(partial(other.write, 'a.md', 'other', expected_version='stale'))
            return call()

        _intercept(monkeypatch, store.table, 'update', receipt=False, around=stale_writer_settles_first)
        written = await store.write('a.md', 'x', expected_version=base.version, operation=operation)
        assert await store.get_operation(operation) == MemoryMutation(
            version=written.version, replayed=True, existed=True
        )

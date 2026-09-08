"""Tests for `PostgresPlanStore` against an in-memory fake asyncpg-style pool."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Any

import anyio
import pytest

from pydantic_ai_harness.planning import (
    PlanEvent,
    PlanEventEmitter,
    PlanEventType,
    PlanItem,
    PostgresConnection,
    PostgresPlanStore,
    PostgresPool,
    TaskStatus,
)
from pydantic_ai_harness.planning._postgres import _deleted_count

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.filterwarnings('ignore::pydantic_ai_harness.HarnessDeprecationWarning'),
]


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


Row = tuple[object, ...]


class FakeConnection:
    """A tiny asyncpg-compatible connection backed by an in-memory row list."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.create_calls = 0
        self.advisory_locks = 0

    def transaction(self) -> AbstractAsyncContextManager[object]:
        conn = self

        class _Txn:
            async def __aenter__(self) -> object:
                return conn

            async def __aexit__(self, *exc: object) -> bool:
                return False

        return _Txn()

    def _select_columns(self, row: dict[str, Any]) -> Row:
        return (row['id'], row['content'], row['status'], row['active_form'], row['parent_id'], row['depends_on'])

    async def execute(self, query: str, *args: object) -> object:
        # A real driver yields to the event loop here; without that, a missing schema
        # guard would look correct because nothing else can interleave.
        await anyio.sleep(0)
        if query.startswith('CREATE TABLE'):
            self.create_calls += 1
            return 'CREATE TABLE'
        if query.startswith('INSERT INTO'):
            session, seq, id_, content, status, active_form, parent_id, depends_on = args
            self._rows.append(
                {
                    'session': session,
                    'seq': seq,
                    'id': id_,
                    'content': content,
                    'status': status,
                    'active_form': active_form,
                    'parent_id': parent_id,
                    'depends_on': depends_on,
                }
            )
            return 'INSERT 0 1'
        if query.startswith('DELETE FROM') and 'AND id' in query:
            session, id_ = args
            before = len(self._rows)
            self._rows[:] = [r for r in self._rows if not (r['session'] == session and r['id'] == id_)]
            return f'DELETE {before - len(self._rows)}'
        if query.startswith('DELETE FROM'):
            (session,) = args
            self._rows[:] = [r for r in self._rows if r['session'] != session]
            return 'DELETE 0'
        if query.startswith('UPDATE'):
            content, status, active_form, parent_id, depends_on, session, id_ = args
            for row in self._rows:
                if row['session'] == session and row['id'] == id_:
                    row.update(
                        content=content,
                        status=status,
                        active_form=active_form,
                        parent_id=parent_id,
                        depends_on=depends_on,
                    )
            return 'UPDATE 1'
        raise AssertionError(f'unexpected execute query: {query}')  # pragma: no cover

    async def fetchval(self, query: str, *args: object) -> object:
        await anyio.sleep(0)
        if 'pg_advisory_xact_lock' in query:
            self.advisory_locks += 1
            return None
        if 'MAX(seq)' in query:
            (session,) = args
            seqs = [int(r['seq']) for r in self._rows if r['session'] == session]
            return max(seqs) + 1 if seqs else 0
        raise AssertionError(f'unexpected fetchval query: {query}')  # pragma: no cover

    async def fetchrow(self, query: str, *args: object) -> Row | None:
        session, id_ = args
        for row in self._rows:
            if row['session'] == session and row['id'] == id_:
                return self._select_columns(row)
        return None

    async def fetch(self, query: str, *args: object) -> list[Row]:
        (session,) = args
        rows = sorted((r for r in self._rows if r['session'] == session), key=lambda r: r['seq'])
        return [self._select_columns(r) for r in rows]


class FakePool:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.connection = FakeConnection(self.rows)

    def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
        conn = self.connection

        class _Acquire:
            async def __aenter__(self) -> PostgresConnection:
                return conn

            async def __aexit__(self, *exc: object) -> bool:
                return False

        return _Acquire()


def _item(content: str, **kwargs: object) -> PlanItem:
    return PlanItem(content=content, **kwargs)  # type: ignore[arg-type]


class TestProtocols:
    def test_fake_satisfies_protocols(self) -> None:
        pool = FakePool()
        assert isinstance(pool, PostgresPool)
        assert isinstance(pool.connection, PostgresConnection)


class TestPostgresStore:
    def test_invalid_table_rejected(self) -> None:
        with pytest.raises(ValueError, match='invalid table name'):
            PostgresPlanStore(FakePool(), table='bad;name')

    async def test_crud_round_trip(self) -> None:
        store = PostgresPlanStore(FakePool())
        assert await store.get_items() == []
        a = await store.add_item(_item('A'))
        await store.add_item(_item('B'))
        items = await store.get_items()
        assert [i.content for i in items] == ['A', 'B']
        assert (await store.get_item(a.id)) is not None
        assert (await store.get_item('missing')) is None

    async def test_schema_created_once(self) -> None:
        pool = FakePool()
        store = PostgresPlanStore(pool)
        await store.get_items()
        await store.get_items()
        assert pool.connection.create_calls == 1
        assert pool.connection.advisory_locks == 1

    async def test_schema_created_once_under_concurrency(self) -> None:
        """`CREATE TABLE IF NOT EXISTS` is not race-free in Postgres, so it must not run twice."""
        pool = FakePool()
        store = PostgresPlanStore(pool)
        async with anyio.create_task_group() as tg:
            for _ in range(4):
                tg.start_soon(store.get_items)
        assert pool.connection.create_calls == 1

    async def test_remove_creates_the_schema_without_an_emitter(self) -> None:
        """`remove_item` skips its read when no emitter is attached, so it cannot rely on one."""
        pool = FakePool()
        store = PostgresPlanStore(pool)
        assert await store.remove_item('missing') is False
        assert pool.connection.create_calls == 1

    async def test_set_items_replaces_in_transaction(self) -> None:
        store = PostgresPlanStore(FakePool())
        await store.set_items([_item('one'), _item('two')])
        await store.set_items([_item('only')])
        assert [i.content for i in await store.get_items()] == ['only']

    async def test_update_found_and_missing(self) -> None:
        store = PostgresPlanStore(FakePool())
        await store.add_item(_item('first'))
        item = await store.add_item(_item('x', depends_on=['d']))
        updated = await store.update_item(item.id, status=TaskStatus.completed, parent_id='p')
        assert updated is not None and updated.status is TaskStatus.completed and updated.parent_id == 'p'
        assert updated.depends_on == ['d']
        assert await store.update_item('missing', status=TaskStatus.completed) is None

    async def test_remove_existing_and_missing(self) -> None:
        store = PostgresPlanStore(FakePool())
        item = await store.add_item(_item('x'))
        assert await store.remove_item(item.id) is True
        assert await store.remove_item(item.id) is False

    async def test_events(self) -> None:
        events: list[PlanEvent] = []
        emitter = PlanEventEmitter()
        for kind in PlanEventType:
            emitter.on(kind, events.append)
        store = PostgresPlanStore(FakePool(), event_emitter=emitter)
        item = await store.add_item(_item('t'))
        await store.update_item(item.id, status=TaskStatus.completed)
        await store.remove_item(item.id)
        kinds = [e.event_type for e in events]
        assert PlanEventType.created in kinds
        assert PlanEventType.completed in kinds
        assert kinds[-1] is PlanEventType.deleted


class TestDeletedCount:
    def test_parses_count(self) -> None:
        assert _deleted_count('DELETE 1') == 1
        assert _deleted_count('DELETE 0') == 0

    def test_non_standard_tag_is_zero(self) -> None:
        assert _deleted_count('WEIRD') == 0
        assert _deleted_count('') == 0

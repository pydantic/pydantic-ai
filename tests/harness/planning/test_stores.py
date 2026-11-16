"""Tests for planning storage backends (in-memory and SQLite)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from pydantic_ai_harness.planning import (
    InMemoryPlanStore,
    PlanItem,
    PlanStore,
    SqlitePlanStore,
    TaskStatus,
)
from pydantic_ai_harness.planning._store import validate_table_name

StoreFactory = Callable[[], PlanStore]


@pytest.fixture(params=['memory', 'sqlite'])
def store_factory(request: pytest.FixtureRequest, tmp_path: Path) -> StoreFactory:
    if request.param == 'memory':
        return InMemoryPlanStore
    database = str(tmp_path / 'plan.db')
    return lambda: SqlitePlanStore(database)


def _item(content: str, **kwargs: object) -> PlanItem:
    return PlanItem(content=content, **kwargs)  # type: ignore[arg-type]


class TestCrudAcrossBackends:
    async def test_empty(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        assert await store.get_items() == []
        assert await store.get_item('missing') is None
        assert await store.remove_item('missing') is False
        assert await store.update_item('missing', status=TaskStatus.completed) is None

    async def test_set_and_get_preserves_order(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        a, b, c = _item('A'), _item('B'), _item('C')
        await store.set_items([a, b, c])
        assert [i.content for i in await store.get_items()] == ['A', 'B', 'C']
        assert (await store.get_item(b.id)) is not None

    async def test_add_appends(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        first = await store.add_item(_item('first'))
        await store.add_item(_item('second'))
        items = await store.get_items()
        assert [i.content for i in items] == ['first', 'second']
        assert items[0].id == first.id

    async def test_update_all_fields(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        item = await store.add_item(_item('old', active_form='Doing old'))
        updated = await store.update_item(
            item.id,
            content='new',
            status=TaskStatus.in_progress,
            active_form='Doing new',
            parent_id='p1',
            depends_on=['d1', 'd2'],
        )
        assert updated is not None
        assert (updated.content, updated.status, updated.active_form) == ('new', TaskStatus.in_progress, 'Doing new')
        assert (updated.parent_id, updated.depends_on) == ('p1', ['d1', 'd2'])
        # persisted
        fetched = await store.get_item(item.id)
        assert fetched is not None and fetched.content == 'new' and fetched.depends_on == ['d1', 'd2']

    async def test_remove(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        item = await store.add_item(_item('gone'))
        assert await store.remove_item(item.id) is True
        assert await store.get_item(item.id) is None
        assert await store.remove_item(item.id) is False

    async def test_remove_non_first_item(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        await store.add_item(_item('first'))
        second = await store.add_item(_item('second'))
        assert await store.remove_item(second.id) is True
        assert [i.content for i in await store.get_items()] == ['first']

    async def test_public_plan_items_do_not_alias_store_state(self, store_factory: StoreFactory) -> None:
        store = store_factory()
        supplied = _item('original', depends_on=['first'])
        await store.set_items([supplied])
        supplied.content = 'caller mutation'
        supplied.depends_on.append('caller')

        stored = await store.get_item(supplied.id)
        assert stored is not None
        assert (stored.content, stored.depends_on) == ('original', ['first'])
        stored.content = 'get_item mutation'
        stored.depends_on.append('get_item')

        [listed] = await store.get_items()
        assert (listed.content, listed.depends_on) == ('original', ['first'])
        listed.content = 'get_items mutation'
        listed.depends_on.append('get_items')

        added = await store.add_item(_item('added', depends_on=['first']))
        added.content = 'add_item mutation'
        added.depends_on.append('add_item')
        updated = await store.update_item(added.id, content='updated', depends_on=['second'])
        assert updated is not None
        updated.content = 'update_item mutation'
        updated.depends_on.append('update_item')

        items = await store.get_items()
        assert [(item.content, item.depends_on) for item in items] == [
            ('original', ['first']),
            ('updated', ['second']),
        ]


class TestSqliteSpecifics:
    def test_memory_database_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not support ':memory:'"):
            SqlitePlanStore(':memory:')

    async def test_persists_across_instances(self, tmp_path: Path) -> None:
        database = str(tmp_path / 'p.db')
        store = SqlitePlanStore(database, session='s1')
        await store.set_items([_item('kept', status=TaskStatus.in_progress)])
        reopened = SqlitePlanStore(database, session='s1')
        items = await reopened.get_items()
        assert [i.content for i in items] == ['kept']
        assert items[0].status is TaskStatus.in_progress

    async def test_sessions_are_isolated(self, tmp_path: Path) -> None:
        database = str(tmp_path / 'p.db')
        s1 = SqlitePlanStore(database, session='s1')
        s2 = SqlitePlanStore(database, session='s2')
        await s1.add_item(_item('only-s1'))
        assert [i.content for i in await s1.get_items()] == ['only-s1']
        assert await s2.get_items() == []

    async def test_parent_and_depends_round_trip(self, tmp_path: Path) -> None:
        store = SqlitePlanStore(str(tmp_path / 'p.db'))
        parent = await store.add_item(_item('parent'))
        await store.add_item(_item('child', parent_id=parent.id, depends_on=[parent.id]))
        child = (await store.get_items())[1]
        assert child.parent_id == parent.id
        assert child.depends_on == [parent.id]

    def test_invalid_table_name_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='invalid table name'):
            SqlitePlanStore(str(tmp_path / 'p.db'), table='bad name;drop')


class TestValidateTableName:
    def test_accepts_valid(self) -> None:
        validate_table_name('plan_items')

    def test_rejects_invalid(self) -> None:
        with pytest.raises(ValueError, match='invalid table name'):
            validate_table_name('1bad')

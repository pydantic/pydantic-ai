"""Tests for the claimable plan stores, run against every backend.

The concurrency tests race many claimers through separate store instances on one
backend wherever the backend can be shared (SQLite file, JSON file, Postgres pool,
Redis client), so the atomicity under test is the backend's, not one instance's
in-process lock.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Coroutine
from pathlib import Path

import anyio
import pytest

from pydantic_ai_harness.planning import (
    BoardChange,
    ClaimablePlanStore,
    ClaimableRedisClient,
    FileClaimablePlanStore,
    InMemoryClaimablePlanStore,
    OwnedPlanItem,
    PlanItem,
    PlanStore,
    PostgresClaimablePlanStore,
    RedisClaimablePlanStore,
    SqliteClaimablePlanStore,
    TaskStatus,
)
from tests.harness._postgres_sqlite import SqlitePool

BoardFactory = Callable[[], ClaimablePlanStore]
"""Returns a new store instance on one shared backend."""

CLAIMERS = 20


class FakeRedis:
    """A redis.asyncio-compatible client that reproduces the store's compare-and-set script.

    It yields before every command, like a network round trip, so concurrent writers
    interleave and some compare-and-sets fail; `conflicts` counts them. The script's
    comparison and write happen without a yield, as on a server.
    """

    def __init__(self, *, decode: bool) -> None:
        self.data: dict[str, str] = {}
        self.expiries: dict[str, str] = {}
        self.conflicts = 0
        self._decode = decode

    async def get(self, name: str, /) -> object:
        await anyio.sleep(0)
        value = self.data.get(name)
        if value is None or self._decode:
            return value
        return value.encode()

    async def eval(self, script: str, numkeys: int, /, *keys_and_args: str) -> object:
        await anyio.sleep(0)
        assert 'redis.call' in script and numkeys == 1
        key, expected, document, expire = keys_and_args
        if self.data.get(key, '') != expected:
            self.conflicts += 1
            return 0
        self.data[key] = document
        self.expiries[key] = expire
        return 1


@pytest.fixture(params=['memory', 'sqlite', 'file', 'postgres', 'redis'])
async def boards(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[BoardFactory]:
    if request.param == 'memory':
        store = InMemoryClaimablePlanStore()
        yield lambda: store
    elif request.param == 'sqlite':
        yield lambda: SqliteClaimablePlanStore(tmp_path / 'board.db')
    elif request.param == 'file':
        yield lambda: FileClaimablePlanStore(tmp_path / 'team' / 'board.json')
    elif request.param == 'postgres':
        pool = SqlitePool()
        yield lambda: PostgresClaimablePlanStore(pool)
        pool.close()
    else:
        client = FakeRedis(decode=False)
        yield lambda: RedisClaimablePlanStore(client)


def _item(content: str, **kwargs: object) -> PlanItem:
    return PlanItem(content=content, **kwargs)  # type: ignore[arg-type]


async def _race(*calls: Callable[[], Coroutine[object, object, object]]) -> None:
    async with anyio.create_task_group() as group:
        for call in calls:
            group.start_soon(call)


class TestClaimablePlanStore:
    async def test_satisfies_both_protocols(self, boards: BoardFactory) -> None:
        store = boards()
        assert isinstance(store, ClaimablePlanStore)
        assert isinstance(store, PlanStore)

    async def test_plan_store_reads_return_plain_items(self, boards: BoardFactory) -> None:
        store = boards()
        assert await store.get_items() == []
        assert await store.get_item('missing') is None
        assert await store.get_owned_item('missing') is None
        added = await store.add_item(_item('A', id='a', depends_on=['x']))
        assert type(added) is PlanItem
        await store.claim('a', owner='alice')

        [item] = await store.get_items()
        assert type(item) is PlanItem and item.id == 'a' and item.depends_on == ['x']
        fetched = await store.get_item('a')
        assert type(fetched) is PlanItem and fetched.status is TaskStatus.in_progress
        assert await store.get_owned_items() == [
            OwnedPlanItem(id='a', content='A', status=TaskStatus.in_progress, depends_on=['x'], owner='alice')
        ]

    async def test_inherited_writes_keep_owners(self, boards: BoardFactory) -> None:
        store = boards()
        await store.set_items([_item('A', id='a'), _item('B', id='b')])
        await store.claim('a', owner='alice')

        updated = await store.update_item('a', content='A, renamed', active_form='Renaming')
        assert updated is not None and updated.content == 'A, renamed'
        assert await store.update_item('missing', content='x') is None
        await store.set_items(
            [
                OwnedPlanItem(id='a', content='A again', status=TaskStatus.in_progress, owner='mallory'),
                _item('C', id='c'),
            ]
        )

        owners = {item.id: item.owner for item in await store.get_owned_items()}
        assert owners == {'a': 'alice', 'c': None}
        assert await store.add_item(OwnedPlanItem(id='d', content='D', owner='mallory')) == _item('D', id='d')
        assert (await store.get_owned_item('d')) == OwnedPlanItem(id='d', content='D')

    async def test_duplicate_ids_are_refused_and_nothing_is_written(self, boards: BoardFactory) -> None:
        store = boards()
        await store.add_item(_item('A', id='a'))

        with pytest.raises(ValueError, match='already'):
            await store.add_item(_item('A again', id='a'))
        with pytest.raises(ValueError, match="Duplicate step ids: 'b'"):
            await store.set_items([_item('B', id='b'), _item('B again', id='b')])
        assert [item.id for item in await store.get_items()] == ['a']

    async def test_remove_item(self, boards: BoardFactory) -> None:
        store = boards()
        await store.set_items([_item('A', id='a'), _item('B', id='b')])

        assert await store.remove_item('a') is True
        assert await store.remove_item('a') is False
        assert [item.id for item in await store.get_items()] == ['b']

    async def test_claim_takes_only_available_steps(self, boards: BoardFactory) -> None:
        store = boards()
        await store.set_items(
            [
                _item('Done', id='done', status=TaskStatus.completed),
                _item('Waits', id='waits', depends_on=['first']),
                _item('First', id='first'),
                _item('Second', id='second'),
            ]
        )

        assert await store.claim('missing', owner='alice') is None
        assert await store.claim('done', owner='alice') is None
        assert await store.claim('waits', owner='alice') is None, 'an unresolved dependency blocks a claim'
        first = await store.claim(None, owner='alice')
        assert first == BoardChange(
            item=OwnedPlanItem(id='first', content='First', status=TaskStatus.in_progress, owner='alice'),
            previous=OwnedPlanItem(id='first', content='First'),
        )
        assert await store.claim('first', owner='bob') is None
        second = await store.claim(None, owner='bob')
        assert second is not None and second.item.id == 'second'
        assert await store.claim(None, owner='carol') is None

    async def test_release_moves_the_owner_and_the_dependents(self, boards: BoardFactory) -> None:
        store = boards()
        await store.set_items(
            [
                _item('Base', id='base'),
                _item('Dependent', id='dep', status=TaskStatus.blocked, depends_on=['base']),
                _item('Unrelated', id='other', status=TaskStatus.blocked, depends_on=['missing-but-blocked']),
            ]
        )
        await store.claim('base', owner='alice')

        with pytest.raises(ValueError, match='claim it instead'):
            await store.release('base', owner='alice', status=TaskStatus.in_progress)
        assert await store.release('base', owner='bob', status=TaskStatus.completed) == []
        assert await store.release('missing', owner='alice', status=TaskStatus.completed) == []

        changed = await store.release('base', owner='alice', status=TaskStatus.completed)
        assert changed == [
            BoardChange(
                item=OwnedPlanItem(id='base', content='Base', status=TaskStatus.completed, owner='alice'),
                previous=OwnedPlanItem(id='base', content='Base', status=TaskStatus.in_progress, owner='alice'),
            ),
            BoardChange(
                item=OwnedPlanItem(id='dep', content='Dependent', depends_on=['base']),
                previous=OwnedPlanItem(id='dep', content='Dependent', status=TaskStatus.blocked, depends_on=['base']),
            ),
        ]
        assert await store.release('base', owner='alice', status=TaskStatus.completed) == []

        dependent = await store.claim('dep', owner='bob')
        assert dependent is not None
        reopened = await store.release('base', owner='alice', status=TaskStatus.pending)
        assert [change.item for change in reopened] == [
            OwnedPlanItem(id='base', content='Base'),
            OwnedPlanItem(id='dep', content='Dependent', status=TaskStatus.blocked, depends_on=['base'], owner='bob'),
        ]
        assert [change.previous.status for change in reopened] == [TaskStatus.completed, TaskStatus.in_progress]
        assert (await store.get_owned_item('other')) == OwnedPlanItem(
            id='other', content='Unrelated', status=TaskStatus.blocked, depends_on=['missing-but-blocked']
        )

    async def test_releasing_to_pending_lets_the_owner_reclaim(self, boards: BoardFactory) -> None:
        store = boards()
        await store.set_items([_item('Base', id='base'), _item('Dependent', id='dep', depends_on=['base'])])
        await store.claim('base', owner='alice')
        await store.release('base', owner='alice', status=TaskStatus.completed)
        await store.claim('dep', owner='bob')
        await store.release('base', owner='alice', status=TaskStatus.cancelled)
        await store.release('base', owner='alice', status=TaskStatus.pending)

        assert await store.claim('dep', owner='carol') is None, 'a blocked step stays assigned to bob'
        await store.claim('base', owner='alice')
        await store.release('base', owner='alice', status=TaskStatus.completed)
        assert await store.claim('dep', owner='carol') is None
        reclaimed = await store.claim(None, owner='bob')
        assert reclaimed is not None and reclaimed.item.id == 'dep' and reclaimed.item.owner == 'bob'


class TestClaimableConcurrency:
    async def test_concurrent_claimers_of_one_step_have_exactly_one_winner(self, boards: BoardFactory) -> None:
        await boards().set_items([_item('Contested', id='contested')])
        results: dict[str, BoardChange | None] = {}

        def claimer(name: str) -> Callable[[], Coroutine[object, object, None]]:
            async def claim() -> None:
                results[name] = await boards().claim('contested', owner=name)

            return claim

        await _race(*(claimer(f'member-{index}') for index in range(CLAIMERS)))

        winners = [name for name, item in results.items() if item is not None]
        assert len(results) == CLAIMERS and len(winners) == 1
        stored = await boards().get_owned_item('contested')
        assert stored is not None and stored.owner == winners[0] and stored.status is TaskStatus.in_progress

    async def test_concurrent_next_claims_take_each_step_once(self, boards: BoardFactory) -> None:
        await boards().set_items([_item(f'Step {index}', id=f'step-{index}') for index in range(CLAIMERS // 2)])
        claimed: list[BoardChange | None] = []

        def claimer(name: str) -> Callable[[], Coroutine[object, object, None]]:
            async def claim() -> None:
                claimed.append(await boards().claim(None, owner=name))

            return claim

        await _race(*(claimer(f'member-{index}') for index in range(CLAIMERS)))

        taken = [change.item for change in claimed if change is not None]
        assert sorted(item.id for item in taken) == sorted(f'step-{index}' for index in range(CLAIMERS // 2))
        assert len({item.owner for item in taken}) == CLAIMERS // 2
        assert {item.owner for item in await boards().get_owned_items()} == {item.owner for item in taken}

    async def test_concurrent_completions_unblock_dependents_exactly_once(self, boards: BoardFactory) -> None:
        prerequisites = [f'pre-{index}' for index in range(CLAIMERS // 2)]
        await boards().set_items(
            [
                *(_item(item_id, id=item_id) for item_id in prerequisites),
                _item('Join', id='join', status=TaskStatus.blocked, depends_on=prerequisites),
                *(
                    _item(f'After {item_id}', id=f'after-{item_id}', status=TaskStatus.blocked, depends_on=[item_id])
                    for item_id in prerequisites
                ),
            ]
        )
        for item_id in prerequisites:
            await boards().claim(item_id, owner=f'owner-{item_id}')
        changes: list[BoardChange] = []

        def completer(item_id: str) -> Callable[[], Coroutine[object, object, None]]:
            async def complete() -> None:
                changes.extend(await boards().release(item_id, owner=f'owner-{item_id}', status=TaskStatus.completed))

            return complete

        # Each prerequisite is completed twice at once: the duplicate must change nothing.
        await _race(*(completer(item_id) for item_id in prerequisites * 2))

        unblocked = sorted(change.item.id for change in changes if change.item.status is TaskStatus.pending)
        assert unblocked == sorted(['join', *(f'after-{item_id}' for item_id in prerequisites)])
        completed = sorted(change.item.id for change in changes if change.item.status is TaskStatus.completed)
        assert all(
            change.previous.status is TaskStatus.blocked for change in changes if change.item.id not in completed
        )
        assert completed == sorted(prerequisites)
        statuses = {item.id: item.status for item in await boards().get_owned_items()}
        assert statuses['join'] is TaskStatus.pending

    async def test_concurrent_set_items_does_not_lose_an_owner(self, boards: BoardFactory) -> None:
        ids = [f'step-{index}' for index in range(CLAIMERS // 2)]
        await boards().set_items([_item(item_id, id=item_id) for item_id in ids])

        def claimer(item_id: str) -> Callable[[], Coroutine[object, object, None]]:
            async def claim() -> None:
                await boards().claim(item_id, owner=f'owner-{item_id}')

            return claim

        def rewriter(round_: int) -> Callable[[], Coroutine[object, object, None]]:
            async def rewrite() -> None:
                await boards().set_items([_item(f'{item_id} v{round_}', id=item_id) for item_id in ids])

            return rewrite

        await _race(*(claimer(item_id) for item_id in ids), *(rewriter(round_) for round_ in range(CLAIMERS // 2)))

        owners = {item.id: item.owner for item in await boards().get_owned_items()}
        assert owners == {item_id: f'owner-{item_id}' for item_id in ids}


class TestBackends:
    def test_sqlite_refuses_an_in_memory_database(self) -> None:
        with pytest.raises(ValueError, match=':memory:'):
            SqliteClaimablePlanStore(':memory:')

    def test_sql_stores_refuse_an_unsafe_table_name(self) -> None:
        with pytest.raises(ValueError, match='invalid table name'):
            SqliteClaimablePlanStore('board.db', table='x; DROP TABLE y')
        pool = SqlitePool()
        with pytest.raises(ValueError, match='invalid table name'):
            PostgresClaimablePlanStore(pool, table='x; DROP TABLE y')
        pool.close()

    async def test_sqlite_sessions_and_tables_are_separate(self, tmp_path: Path) -> None:
        database = tmp_path / 'board.db'
        await SqliteClaimablePlanStore(database, session='one').add_item(_item('A'))

        assert await SqliteClaimablePlanStore(database, session='two').get_items() == []
        assert await SqliteClaimablePlanStore(database, session='one', table='other').get_items() == []

    async def test_postgres_rolls_back_a_refused_write(self) -> None:
        pool = SqlitePool()
        store = PostgresClaimablePlanStore(pool, session='s')
        await store.add_item(_item('A', id='a'))

        with pytest.raises(ValueError, match='already'):
            await store.add_item(_item('A', id='a'))
        assert await pool.count_rows('claimable_plan_items') == 1
        assert await PostgresClaimablePlanStore(pool, session='other').get_items() == []
        pool.close()

    async def test_postgres_creates_its_table_once_for_concurrent_first_calls(self) -> None:
        pool = SqlitePool()
        store = PostgresClaimablePlanStore(pool)

        await _race(store.get_items, store.get_items)

        assert sum(statement.startswith('CREATE TABLE') for statement in pool.statements) == 1
        pool.close()

    def test_fake_redis_satisfies_the_protocol(self) -> None:
        assert isinstance(FakeRedis(decode=True), ClaimableRedisClient)

    async def test_redis_retries_a_conflicting_write(self) -> None:
        client = FakeRedis(decode=True)
        await RedisClaimablePlanStore(client).set_items([_item('Contested', id='contested')])

        await _race(*(lambda: RedisClaimablePlanStore(client).claim('contested', owner='x') for _ in range(3)))

        assert client.conflicts > 0
        assert (await RedisClaimablePlanStore(client).get_owned_item('contested')) is not None

    async def test_redis_key_prefix_session_and_expiry(self) -> None:
        client = FakeRedis(decode=True)
        store = RedisClaimablePlanStore(client, session='s', key_prefix='board', expire_seconds=60)
        await store.add_item(_item('A'))

        assert list(client.data) == ['board:{s}'] and client.expiries == {'board:{s}': '60'}
        await RedisClaimablePlanStore(client, session='t').add_item(_item('B'))
        assert client.expiries['claimable-plan:{t}'] == ''

    async def test_file_store_writes_one_json_document(self, tmp_path: Path) -> None:
        path = tmp_path / 'board.json'
        store = FileClaimablePlanStore(path)
        await store.add_item(_item('A', id='a'))
        await store.claim('a', owner='alice')

        assert store.path == path
        assert json.loads(path.read_text()) == [
            {
                'id': 'a',
                'content': 'A',
                'status': 'in_progress',
                'active_form': '',
                'parent_id': None,
                'depends_on': [],
                'owner': 'alice',
            }
        ]
        assert sorted(entry.name for entry in tmp_path.iterdir()) == ['board.json', 'board.json.lock']

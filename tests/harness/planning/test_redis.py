"""Tests for `RedisPlanStore` against an in-memory fake redis.asyncio-style client."""

from __future__ import annotations

import pytest

from pydantic_ai_harness.planning import (
    PlanEvent,
    PlanEventEmitter,
    PlanEventType,
    PlanItem,
    RedisClient,
    RedisPlanStore,
    TaskStatus,
)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.filterwarnings('ignore::pydantic_ai_harness.HarnessDeprecationWarning'),
]


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


class FakeRedis:
    """A tiny redis.asyncio-compatible client backed by an in-memory dict.

    `decode_responses=False` (the default) returns `bytes` like a real client;
    set `decode=True` to return `str`.
    """

    def __init__(self, *, decode: bool = False) -> None:
        self._data: dict[str, str] = {}
        self._decode = decode
        self.last_ex: int | None = None

    async def get(self, key: str) -> object:
        value = self._data.get(key)
        if value is None:
            return None
        return value if self._decode else value.encode()

    async def set(self, key: str, value: str, *, ex: int | None = None) -> object:
        self._data[key] = value
        self.last_ex = ex
        return True


def _item(content: str, **kwargs: object) -> PlanItem:
    return PlanItem(content=content, **kwargs)  # type: ignore[arg-type]


class TestProtocol:
    def test_fake_satisfies_protocol(self) -> None:
        assert isinstance(FakeRedis(), RedisClient)


class TestRedisStore:
    async def test_crud_round_trip(self) -> None:
        store = RedisPlanStore(FakeRedis())
        assert await store.get_items() == []
        assert await store.get_item('missing') is None
        a = await store.add_item(_item('A', depends_on=['x']))
        await store.add_item(_item('B'))
        items = await store.get_items()
        assert [i.content for i in items] == ['A', 'B']
        fetched = await store.get_item(a.id)
        assert fetched is not None and fetched.depends_on == ['x']

    async def test_a_duplicate_id_is_refused(self) -> None:
        """Every backend answers the same way, so a plan behaves the same wherever it is stored."""
        store = RedisPlanStore(FakeRedis())
        item = await store.add_item(_item('A'))

        with pytest.raises(ValueError, match='already'):
            await store.add_item(_item('A again', id=item.id))

    async def test_set_replaces(self) -> None:
        store = RedisPlanStore(FakeRedis())
        await store.set_items([_item('one'), _item('two')])
        await store.set_items([_item('only')])
        assert [i.content for i in await store.get_items()] == ['only']

    async def test_update_found_and_missing(self) -> None:
        store = RedisPlanStore(FakeRedis())
        item = await store.add_item(_item('x'))
        updated = await store.update_item(item.id, status=TaskStatus.completed, content='y')
        assert updated is not None and updated.status is TaskStatus.completed and updated.content == 'y'
        # persisted
        assert (await store.get_item(item.id)).status is TaskStatus.completed  # type: ignore[union-attr]
        assert await store.update_item('missing', status=TaskStatus.completed) is None

    async def test_remove_existing_and_missing(self) -> None:
        store = RedisPlanStore(FakeRedis())
        first = await store.add_item(_item('first'))
        second = await store.add_item(_item('second'))
        assert await store.remove_item(second.id) is True
        assert [i.content for i in await store.get_items()] == ['first']
        assert await store.remove_item(first.id) is True
        assert await store.remove_item('missing') is False

    async def test_expire_seconds_passed_through(self) -> None:
        client = FakeRedis()
        store = RedisPlanStore(client, expire_seconds=3600)
        await store.add_item(_item('A'))
        assert client.last_ex == 3600

    async def test_no_expire_by_default(self) -> None:
        client = FakeRedis()
        store = RedisPlanStore(client)
        await store.add_item(_item('A'))
        assert client.last_ex is None

    async def test_decode_responses_client(self) -> None:
        # A client configured with decode_responses=True returns str, not bytes.
        store = RedisPlanStore(FakeRedis(decode=True))
        await store.add_item(_item('A'))
        assert [i.content for i in await store.get_items()] == ['A']

    async def test_sessions_and_prefix_isolated(self) -> None:
        client = FakeRedis()
        s1 = RedisPlanStore(client, session='s1', key_prefix='plans')
        s2 = RedisPlanStore(client, session='s2', key_prefix='plans')
        await s1.add_item(_item('only-s1'))
        assert [i.content for i in await s1.get_items()] == ['only-s1']
        assert await s2.get_items() == []

    async def test_events(self) -> None:
        events: list[PlanEvent] = []
        emitter = PlanEventEmitter()
        for kind in PlanEventType:
            emitter.on(kind, events.append)
        store = RedisPlanStore(FakeRedis(), event_emitter=emitter)
        item = await store.add_item(_item('t'))
        await store.update_item(item.id, status=TaskStatus.completed)
        await store.remove_item(item.id)
        kinds = [e.event_type for e in events]
        assert PlanEventType.created in kinds
        assert PlanEventType.completed in kinds
        assert kinds[-1] is PlanEventType.deleted

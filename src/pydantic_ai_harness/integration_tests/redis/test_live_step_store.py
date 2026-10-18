"""Live Redis conformance tests for `RedisStepStore`.

The unit suite in `tests/harness/step_persistence/test_redis.py` drives the store
through a fake that reproduces the result the write script is meant to produce.
It never runs the Lua, so what it cannot answer decides whether the store is
correct:

- that the script is valid Lua the server accepts, for every operation;
- that the server, not the client, suppresses a replayed event or snapshot,
  including one fired concurrently from several tasks, and including a snapshot
  key whose snapshot pruning already removed;
- that the prune inside the script keeps the newest `complete` snapshot below
  the window, and that `ZRANGE` orders the index by score, so `seq` 10 follows
  `seq` 2 rather than preceding it as the member text would sort;
- that each message is stored once on the server;
- that `EXPIRE` reaches every key kind the store writes (string, list, set,
  zset, hash), and that a later write refreshes the whole run;
- that `SET ... NX` refuses a second registration;
- that a client left on redis-py's default `bytes` replies reads back what a
  `decode_responses=True` client wrote.

This file covers exactly those. It is not a second copy of the unit suite --
API-shape coverage belongs there, where it runs on every matrix leg.

Run against a local server with `make integration-redis` after starting one, e.g.
`docker run -d -p 6379:6379 redis:8`. Without a reachable server the tests skip,
unless `REDIS_REQUIRE_LIVE` is set (CI does), where an unreachable server fails
instead -- a service container that never came up must not pass as a silent skip.

External assumptions, verified 2026-10-18 against Redis 8 (`redis:8`):

- `EVAL` runs the script to completion without interleaving another client's
  command, but does not roll back on a mid-script error. Source:
  <https://redis.io/docs/latest/develop/programmability/eval-intro/>.
- A Lua number passed to `HSET` or `ZADD` is converted to its integer string,
  so `seq` 1 is stored as the field `1`. Source:
  <https://redis.io/docs/latest/develop/programmability/lua-api/>.
- `ZRANGE key 0 -1` returns members ordered by score, low to high. Source:
  <https://redis.io/docs/latest/commands/zrange/>.
- `EXPIRE` on a key that does not exist does nothing and replies 0. Source:
  <https://redis.io/docs/latest/commands/expire/>.
- `SET` with `NX` writes only when the key is absent, replying nil otherwise.
  Source: <https://redis.io/docs/latest/commands/set/>.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from typing import NoReturn
from urllib.parse import urlsplit

import anyio
import pytest
from redis.asyncio import Redis
from redis.exceptions import RedisError

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RedisStepStore,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)


@pytest.fixture
def anyio_backend() -> str:
    """Run live server tests once under asyncio."""
    return 'asyncio'


def _requires_live() -> bool:
    return os.environ.get('REDIS_REQUIRE_LIVE', '').lower() in {'1', 'true', 'yes'}


def _redis_url() -> str:
    return os.environ.get('REDIS_TEST_URL', 'redis://127.0.0.1:6379')


def _redis_target() -> str:
    """Host and port, without the credentials a Redis URL may carry.

    `REDIS_TEST_URL` accepts `redis://user:password@host:6379`, and the only place
    this string is used is a skip or failure message -- which CI keeps. Naming the
    variable rather than echoing an unparsable value keeps that true for a URL
    `urlsplit` cannot make sense of.
    """
    host = urlsplit(_redis_url()).hostname
    if host is None:
        return 'the server named by REDIS_TEST_URL'
    port = urlsplit(_redis_url()).port
    return host if port is None else f'{host}:{port}'


def _unavailable(message: str) -> NoReturn:
    if _requires_live():
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture
def prefix() -> str:
    """A key namespace unique to one test, so concurrent runs cannot collide."""
    return f'harness-test:{uuid.uuid4().hex}'


@pytest.fixture
async def redis_client(prefix: str) -> AsyncGenerator[Redis]:
    """A connected client whose `{prefix}:*` keys are removed after the test."""
    client = Redis.from_url(_redis_url(), decode_responses=True)  # pyright: ignore[reportUnknownMemberType]
    try:
        await client.ping()  # pyright: ignore[reportUnknownMemberType]
    except (RedisError, OSError) as error:
        await client.aclose()
        _unavailable(f'no reachable Redis at {_redis_target()}: {error}')

    try:
        yield client
    finally:
        # Closing sits in its own `finally` so a server that went away mid-test takes the
        # key cleanup down without also leaking the connection into the rest of the suite.
        try:
            keys = [key async for key in client.scan_iter(match=f'{prefix}:*')]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            if keys:
                await client.delete(*keys)  # pyright: ignore[reportUnknownArgumentType]
        finally:
            await client.aclose()


def _messages(content: str = 'hello') -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=content)])]


async def _ttls(client: Redis, prefix: str) -> dict[str, int]:
    """Seconds left on every key under `prefix`. -1 means no expiry."""
    keys: list[str] = [key async for key in client.scan_iter(match=f'{prefix}:*')]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    return {key: await client.ttl(key) for key in keys}


class TestLiveRedisStepStore:
    """What the fake reproduces rather than executes."""

    async def test_every_operation_round_trips(self, redis_client: Redis, prefix: str) -> None:
        store = RedisStepStore(redis_client, prefix=prefix)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', parent_run_id='p1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='e0'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages('a'), idempotency_key='0:0:complete')
        )
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages('b'), state='interrupted')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        )

        assert [r.run_id for r in await store.list_runs(parent_run_id='p1', conversation_id='c1')] == ['r1']
        assert [e.kind for e in await store.list_events(run_id='r1')] == ['run_started']
        settled = await store.latest_snapshot(run_id='r1')
        assert settled is not None and settled.step_index == 0
        frontier = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert frontier is not None and frontier.step_index == 1
        assert [s.step_index for s in await store.list_snapshots(run_id='r1', include_interrupted=True)] == [0, 1]
        assert [r.tool_call_id for r in await store.list_unresolved_tool_effects(run_id='r1')] == ['t1']

    async def test_the_server_suppresses_replays(self, redis_client: Redis, prefix: str) -> None:
        store = RedisStepStore(redis_client, prefix=prefix, max_snapshots_per_run=1)
        event = StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='event:0')
        older = ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages(), idempotency_key='0:1:complete')
        newer = ContinuableSnapshot(run_id='r1', step_index=2, messages=_messages(), idempotency_key='1:2:complete')

        await store.append_event(event)
        await store.append_event(event)
        await store.save_snapshot(older)
        await store.save_snapshot(newer)
        # Pruning removed `older`; its key still has to keep the replay out.
        await store.save_snapshot(older)

        assert await store.list_events(run_id='r1') == [event]
        assert await store.list_snapshots(run_id='r1') == [newer]

    async def test_concurrent_replays_store_one_copy(self, redis_client: Redis, prefix: str) -> None:
        """The check and the write are one script, so no two replays both pass the check."""
        store = RedisStepStore(redis_client, prefix=prefix)
        event = StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='event:0')
        snapshot = ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages(), idempotency_key='0:0:complete')

        async with anyio.create_task_group() as group:
            for _ in range(20):
                group.start_soon(store.append_event, event)
                group.start_soon(store.save_snapshot, snapshot)

        assert await store.list_events(run_id='r1') == [event]
        assert await store.list_snapshots(run_id='r1') == [snapshot]

    async def test_prune_orders_by_score_and_keeps_the_newest_complete(self, redis_client: Redis, prefix: str) -> None:
        """`10:interrupted` sorts before `2:...` as text; the score has to decide what is newest."""
        store = RedisStepStore(redis_client, prefix=prefix, max_snapshots_per_run=3)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        for step in range(1, 11):
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages(), state='interrupted')
            )

        retained = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in retained] == [0, 8, 9, 10]
        resumable = await store.latest_snapshot(run_id='r1')
        assert resumable is not None and resumable.step_index == 0
        assert await redis_client.hlen(f'{prefix}:snapshot_payloads:{{r1}}') == 4

    async def test_each_message_is_stored_once(self, redis_client: Redis, prefix: str) -> None:
        store = RedisStepStore(redis_client, prefix=prefix)
        history = _messages()
        for step in range(5):
            history = [*history, ModelResponse(parts=[TextPart(content=f'r{step}')])]
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=history))

        assert await redis_client.hlen(f'{prefix}:messages:{{r1}}') == 6
        assert [len(s.messages) for s in await store.list_snapshots(run_id='r1')] == [2, 3, 4, 5, 6]

    async def test_expire_seconds_reaches_every_key(self, redis_client: Redis, prefix: str) -> None:
        store = RedisStepStore(redis_client, prefix=prefix, expire_seconds=60)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='e'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages(), idempotency_key='s')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        )

        ttls = await _ttls(redis_client, prefix)
        assert len(ttls) == 11
        assert all(0 < ttl <= 60 for ttl in ttls.values()), ttls

    async def test_a_later_write_refreshes_the_whole_run(self, redis_client: Redis, prefix: str) -> None:
        """A second store with a longer TTL stands in for time passing: its write lifts every key of the run."""
        short = RedisStepStore(redis_client, prefix=prefix, expire_seconds=60)
        await short.register_run(RunRecord(run_id='r1'))
        await short.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))

        long = RedisStepStore(redis_client, prefix=prefix, expire_seconds=3600)
        await long.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        )

        ttls = await _ttls(redis_client, prefix)
        assert all(ttl > 60 for ttl in ttls.values()), ttls

    async def test_set_nx_refuses_a_second_registration(self, redis_client: Redis, prefix: str) -> None:
        store = RedisStepStore(redis_client, prefix=prefix)
        await store.register_run(RunRecord(run_id='r1', agent_name='first'))

        with pytest.raises(ValueError, match="run_id 'r1' is already in the store"):
            await store.register_run(RunRecord(run_id='r1', agent_name='second'))

        kept = await store.get_run(run_id='r1')
        assert kept is not None and kept.agent_name == 'first'

    async def test_a_bytes_client_reads_what_a_decoded_client_wrote(self, redis_client: Redis, prefix: str) -> None:
        """redis-py replies with `bytes` unless asked otherwise, and both must work."""
        decoded = RedisStepStore(redis_client, prefix=prefix)
        await decoded.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        await decoded.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await decoded.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=4, messages=_messages('a')))
        await decoded.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        )

        raw_client = Redis.from_url(_redis_url())  # pyright: ignore[reportUnknownMemberType]
        try:
            raw = RedisStepStore(raw_client, prefix=prefix)
            assert [r.run_id for r in await raw.list_runs(conversation_id='c1')] == ['r1']
            assert [e.kind for e in await raw.list_events(run_id='r1')] == ['run_started']
            snapshot = await raw.latest_snapshot(run_id='r1')
            assert snapshot is not None and snapshot.step_index == 4
            assert [r.tool_call_id for r in await raw.list_unresolved_tool_effects(run_id='r1')] == ['t1']
        finally:
            await raw_client.aclose()

"""Tests for `RedisStepStore` against a fake implementing the `RedisClient` protocol.

The store speaks only the commands on that protocol, so a dict-backed fake
reaches every path without a running server. What the fake cannot answer --
that the server runs the write script as written, and that `EXPIRE` really
removes a key -- lives in `integration_tests/redis/test_live_step_store.py`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.media import DiskMediaStore
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RedisClient,
    RedisStepStore,
    RunRecord,
    StepEvent,
    StepPersistence,
    StepStore,
    ToolEffectRecord,
)

_P = 'pydantic-ai-harness:step'


class FakeRedis:
    """The commands `RedisStepStore` uses, over plain dicts.

    `eval` stands in for the server running the write script: the same replay
    check, the same writes, the same pruning and expiry, applied as one step,
    which is what Redis guarantees for a script. It reproduces the result the
    script is meant to produce and does not run the Lua, so what it pins is the
    store's contract rather than the script's correctness -- the live suite is
    where the script itself runs.

    Replies are `bytes` by default, as `redis.asyncio.Redis` returns them unless
    it was built with `decode_responses=True`; `decode=True` covers the other
    configuration. Expiry is recorded, not enforced -- tests that need a key to
    be gone remove it directly, which is what an elapsed TTL looks like from the
    store's side.
    """

    def __init__(self, *, decode: bool = False) -> None:
        self.strings: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, int]] = {}
        self.expiries: dict[str, int] = {}
        self._decode = decode

    def _out(self, value: str) -> str | bytes:
        return value if self._decode else value.encode()

    def _exists(self, name: str) -> bool:
        return any(name in store for store in (self.strings, self.lists, self.sets, self.hashes, self.zsets))

    async def get(self, name: str) -> object:
        value = self.strings.get(name)
        return None if value is None else self._out(value)

    async def set(self, name: str, value: str, *, ex: int | None = None, nx: bool = False) -> object:
        if nx and name in self.strings:
            return None
        self.strings[name] = value
        if ex is not None:
            self.expiries[name] = ex
        return True

    async def expire(self, name: str, time: int) -> object:
        if not self._exists(name):
            return False
        self.expiries[name] = time
        return True

    async def sadd(self, name: str, *values: str) -> object:
        self.sets.setdefault(name, set()).update(values)
        return len(values)

    async def srem(self, name: str, *values: str) -> object:
        self.sets.get(name, set()).difference_update(values)
        return len(values)

    async def smembers(self, name: str) -> list[str | bytes]:
        return [self._out(member) for member in sorted(self.sets.get(name, set()))]

    async def lrange(self, name: str, start: int, end: int) -> list[str | bytes]:
        stop = None if end == -1 else end + 1
        return [self._out(item) for item in self.lists.get(name, [])[start:stop]]

    async def zrange(self, name: str, start: int, end: int) -> list[str | bytes]:
        ordered = sorted(self.zsets.get(name, {}).items(), key=lambda entry: entry[1])
        stop = None if end == -1 else end + 1
        return [self._out(member) for member, _ in ordered[start:stop]]

    async def hget(self, name: str, key: str) -> object:
        value = self.hashes.get(name, {}).get(key)
        return None if value is None else self._out(value)

    async def hmget(self, name: str, keys: Sequence[str]) -> list[object]:
        fields = self.hashes.get(name, {})
        return [None if key not in fields else self._out(fields[key]) for key in keys]

    async def hgetall(self, name: str) -> Mapping[str | bytes, str | bytes]:
        return {self._out(key): self._out(value) for key, value in self.hashes.get(name, {}).items()}

    async def eval(self, script: str, numkeys: int, *keys_and_args: str) -> object:
        keys, args = list(keys_and_args[:numkeys]), list(keys_and_args[numkeys:])
        run, events, event_keys, index, seq_key, payloads, snapshot_keys, messages, tool_effects = keys
        op, ttl = args[0], int(args[1])
        if op == 'event':
            has_key, key, event = args[2:5]
            if has_key == '0' or key not in self.sets.get(event_keys, set()):
                self.lists.setdefault(events, []).append(event)
                if has_key == '1':
                    self.sets.setdefault(event_keys, set()).add(key)
        elif op == 'snapshot':
            has_key, key, state, payload, keep_text = args[2:7]
            if has_key == '0' or key not in self.sets.get(snapshot_keys, set()):
                pairs = args[7:]
                stored = self.hashes.setdefault(messages, {})
                for sha256, message in zip(pairs[::2], pairs[1::2]):
                    stored.setdefault(sha256, message)
                seq = int(self.strings.get(seq_key, '0')) + 1
                self.strings[seq_key] = str(seq)
                self.hashes.setdefault(payloads, {})[str(seq)] = payload
                self.zsets.setdefault(index, {})[f'{seq}:{state}'] = seq
                if has_key == '1':
                    self.sets.setdefault(snapshot_keys, set()).add(key)
                self._prune(index, payloads, int(keep_text))
        else:
            tool_call_id, record = args[2:4]
            self.hashes.setdefault(tool_effects, {})[tool_call_id] = record
        if ttl > 0:
            for name in (run, events, event_keys, index, seq_key, payloads, snapshot_keys, messages, tool_effects):
                await self.expire(name, ttl)
        return 1

    def _prune(self, index: str, payloads: str, keep: int) -> None:
        if keep <= 0:
            return
        members = sorted(self.zsets[index], key=lambda member: self.zsets[index][member])
        complete = [i for i, member in enumerate(members) if member.endswith(':complete')]
        newest_complete = complete[-1] if complete else None
        for i, member in enumerate(members[: len(members) - keep]):
            if i != newest_complete:
                del self.hashes[payloads][member.split(':')[0]]
                del self.zsets[index][member]


class UntypedReplyRedis(FakeRedis):
    """A client whose `GET` hands back something that is neither `str` nor `bytes`."""

    async def get(self, name: str) -> object:
        return 1


class RegisterDuringPruneRedis(FakeRedis):
    """Writes a run key the moment `list_runs` removes the stale member for it.

    That is the interleaving a registration under a recycled `run_id` can produce:
    the store read the run key as missing, and the registration wrote it before
    the store's `SREM` landed.
    """

    def __init__(self, run_key: str, record: str) -> None:
        super().__init__()
        self._run_key = run_key
        self._record = record

    async def srem(self, name: str, *values: str) -> object:
        removed = await super().srem(name, *values)
        self.strings[self._run_key] = self._record
        return removed


def _messages(content: str = 'hello') -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(content=content)])]


def _at(minute: int) -> datetime:
    return datetime(2026, 1, 1, 12, minute, tzinfo=UTC)


class TestRedisStepStoreConstruction:
    def test_fake_satisfies_the_client_protocol(self) -> None:
        assert isinstance(FakeRedis(), RedisClient)

    def test_store_satisfies_the_step_store_protocol(self) -> None:
        assert isinstance(RedisStepStore(FakeRedis()), StepStore)

    async def test_media_is_inline_by_default(self) -> None:
        """With no `media_store`, a part above the threshold stays in the stored message."""
        client = FakeRedis()
        store = RedisStepStore(client)
        big_text = 'Z' * 100_000
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content=big_text)])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        [stored] = client.hashes[f'{_P}:messages:{{r1}}'].values()
        assert big_text in stored

    def test_rejects_invalid_max_snapshots_per_run(self) -> None:
        with pytest.raises(ValueError, match='max_snapshots_per_run must be'):
            RedisStepStore(FakeRedis(), max_snapshots_per_run=0)

    @pytest.mark.parametrize('value', [0, -1, 1.5])
    def test_rejects_invalid_expire_seconds(self, value: object) -> None:
        with pytest.raises(ValueError, match='expire_seconds must be an int >= 1 or None'):
            RedisStepStore(FakeRedis(), expire_seconds=value)  # pyright: ignore[reportArgumentType]

    @pytest.mark.parametrize('prefix', ['{tenant}', 'a{b', 'a}b'])
    def test_rejects_a_prefix_with_braces(self, prefix: str) -> None:
        """A brace in the prefix would decide the hash tag instead of `{run_id}`."""
        with pytest.raises(ValueError, match='prefix must not contain'):
            RedisStepStore(FakeRedis(), prefix=prefix)

    @pytest.mark.parametrize('run_id', ['', '}x'])
    async def test_rejects_a_run_id_that_makes_an_empty_hash_tag(self, run_id: str) -> None:
        """`{}` is not a hash tag, so the run's keys would spread across slots."""
        store = RedisStepStore(FakeRedis())
        with pytest.raises(ValueError, match='run_id must be non-empty'):
            await store.get_run(run_id=run_id)

    async def test_prefix_namespaces_every_key(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, prefix='tenant-a')
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='e'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages(), idempotency_key='s')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )

        touched = {*client.strings, *client.lists, *client.sets, *client.hashes, *client.zsets}
        assert touched == {
            'tenant-a:run:{r1}',
            'tenant-a:runs:all',
            'tenant-a:events:{r1}',
            'tenant-a:event_keys:{r1}',
            'tenant-a:snapshots:{r1}',
            'tenant-a:snapshots:seq:{r1}',
            'tenant-a:snapshot_payloads:{r1}',
            'tenant-a:snapshot_keys:{r1}',
            'tenant-a:messages:{r1}',
            'tenant-a:tool_effects:{r1}',
        }


class TestRedisStepStoreProtocol:
    async def test_register_and_get_run(self) -> None:
        store = RedisStepStore(FakeRedis())
        record = RunRecord(run_id='r1', conversation_id='c1', agent_name='a', metadata={'k': 'v'})
        await store.register_run(record)

        assert await store.get_run(run_id='r1') == record

    async def test_register_duplicate_run_raises_and_leaves_the_indexes_alone(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        with pytest.raises(ValueError, match="run_id 'r1' is already in the store"):
            await store.register_run(RunRecord(run_id='r1', conversation_id='c2'))

        assert f'{_P}:runs:conversation:c2' not in client.sets

    async def test_missing_lookups_return_none_or_empty(self) -> None:
        store = RedisStepStore(FakeRedis())
        assert await store.get_run(run_id='nope') is None
        assert await store.list_events(run_id='nope') == []
        assert await store.latest_snapshot(run_id='nope') is None
        assert await store.list_snapshots(run_id='nope') == []
        assert await store.get_tool_effect(run_id='nope', tool_call_id='t1') is None
        assert await store.list_unresolved_tool_effects(run_id='nope') == []
        assert await store.list_runs() == []

    async def test_list_runs_chronological(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.register_run(RunRecord(run_id='b', started_at=_at(2)))
        await store.register_run(RunRecord(run_id='a', started_at=_at(1)))
        await store.register_run(RunRecord(run_id='c', started_at=_at(3)))

        assert [r.run_id for r in await store.list_runs()] == ['a', 'b', 'c']

    async def test_list_runs_filters(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', parent_run_id='p1', started_at=_at(1)))
        await store.register_run(RunRecord(run_id='r2', conversation_id='c2', parent_run_id='p1', started_at=_at(2)))
        await store.register_run(RunRecord(run_id='r3', conversation_id='c1', started_at=_at(3)))

        assert [r.run_id for r in await store.list_runs(parent_run_id='p1')] == ['r1', 'r2']
        assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['r1', 'r3']
        both = await store.list_runs(parent_run_id='p1', conversation_id='c2')
        assert [r.run_id for r in both] == ['r2']

    async def test_list_runs_checks_the_record_not_only_the_set(self) -> None:
        """A write that carried other ids than the run's record does not list the run under them."""
        store = RedisStepStore(FakeRedis())
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, conversation_id='c2'))

        assert await store.list_runs(conversation_id='c2') == []

    async def test_list_runs_sorts_by_instant_not_iso_string(self) -> None:
        """`13:00+02:00` precedes `12:00Z` as an instant but follows it lexicographically."""
        store = RedisStepStore(FakeRedis())
        earlier = datetime(2026, 1, 1, 13, 0, tzinfo=timezone(timedelta(hours=2)))
        later = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        await store.register_run(RunRecord(run_id='later', started_at=later))
        await store.register_run(RunRecord(run_id='earlier', started_at=earlier))

        assert [r.run_id for r in await store.list_runs()] == ['earlier', 'later']

    async def test_list_runs_drops_a_member_whose_run_expired(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, expire_seconds=60)
        await store.register_run(RunRecord(run_id='alive', conversation_id='c1', started_at=_at(1)))
        await store.register_run(RunRecord(run_id='gone', conversation_id='c1', started_at=_at(2)))
        del client.strings[f'{_P}:run:{{gone}}']

        assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['alive']
        assert client.sets[f'{_P}:runs:conversation:c1'] == {'alive'}

    async def test_list_runs_keeps_a_run_registered_while_it_pruned(self) -> None:
        record = RunRecord(run_id='r1', started_at=_at(1))
        run_key = f'{_P}:run:{{r1}}'
        client = RegisterDuringPruneRedis(run_key, json.dumps({'run_id': 'r1', 'started_at': _at(1).isoformat()}))
        client.sets[f'{_P}:runs:all'] = {'r1'}
        store = RedisStepStore(client)

        assert await store.list_runs() == [record]
        assert client.sets[f'{_P}:runs:all'] == {'r1'}

    async def test_a_later_write_indexes_a_run_whose_registration_stopped_short(self) -> None:
        """`StepPersistence` skips `register_run` on a retry that finds the run key, so the next write indexes it."""
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', parent_run_id='p1'))
        client.sets.clear()

        await store.append_event(
            StepEvent(run_id='r1', kind='run_started', step_index=0, conversation_id='c1', parent_run_id='p1')
        )

        assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['r1']
        assert [r.run_id for r in await store.list_runs(parent_run_id='p1')] == ['r1']

    async def test_append_and_list_events(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(
            StepEvent(run_id='r1', kind='tool_call_started', step_index=1, tool_name='search', tool_call_id='t1')
        )
        await store.append_event(StepEvent(run_id='r2', kind='run_started', step_index=0))

        events = await store.list_events(run_id='r1')
        assert [(e.kind, e.step_index, e.tool_name) for e in events] == [
            ('run_started', 0, None),
            ('tool_call_started', 1, 'search'),
        ]

    async def test_keyed_event_replay_is_suppressed(self) -> None:
        store = RedisStepStore(FakeRedis())
        keyed = StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='event:0')
        await store.append_event(keyed)
        await store.append_event(keyed)
        await store.append_event(StepEvent(run_id='r1', kind='run_completed', step_index=0))
        await store.append_event(StepEvent(run_id='r1', kind='run_completed', step_index=0))

        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_started', 'run_completed', 'run_completed']
        assert events[0] == keyed

    async def test_save_and_load_snapshot(self) -> None:
        store = RedisStepStore(FakeRedis())
        snapshot = ContinuableSnapshot(
            run_id='r1', step_index=3, messages=_messages('a'), conversation_id='c1', idempotency_key='0:3:complete'
        )
        await store.save_snapshot(snapshot)

        assert await store.latest_snapshot(run_id='r1') == snapshot

    async def test_an_empty_history_round_trips(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[]))

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None and snap.messages == []

    async def test_each_message_is_stored_once(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        history = _messages('a')
        for step in range(3):
            history = [*history, ModelResponse(parts=[TextPart(content=f'r{step}')])]
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=history))

        assert len(client.hashes[f'{_P}:messages:{{r1}}']) == 4
        assert [len(s.messages) for s in await store.list_snapshots(run_id='r1')] == [2, 3, 4]

    async def test_keyed_snapshot_replay_is_suppressed_per_state(self) -> None:
        store = RedisStepStore(FakeRedis())
        complete = ContinuableSnapshot(run_id='r1', step_index=3, messages=_messages(), idempotency_key='3:complete')
        interrupted = ContinuableSnapshot(
            run_id='r1', step_index=3, messages=_messages(), state='interrupted', idempotency_key='3:interrupted'
        )
        for snapshot in (complete, interrupted, complete, interrupted):
            await store.save_snapshot(snapshot)

        assert await store.list_snapshots(run_id='r1', include_interrupted=True) == [complete, interrupted]

    async def test_a_pruned_snapshot_key_stays_suppressed(self) -> None:
        store = RedisStepStore(FakeRedis(), max_snapshots_per_run=1)
        older = ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages(), idempotency_key='0:1:complete')
        newer = ContinuableSnapshot(run_id='r1', step_index=2, messages=_messages(), idempotency_key='1:2:complete')
        await store.save_snapshot(older)
        await store.save_snapshot(newer)
        await store.save_snapshot(older)

        assert await store.list_snapshots(run_id='r1') == [newer]

    async def test_latest_snapshot_default_skips_interrupted(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages(), state='interrupted')
        )

        default = await store.latest_snapshot(run_id='r1')
        assert default is not None and default.step_index == 0
        opted = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert opted is not None and (opted.step_index, opted.state) == (1, 'interrupted')

    async def test_only_interrupted_snapshots_read_as_none_by_default(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages(), state='interrupted')
        )

        assert await store.latest_snapshot(run_id='r1') is None

    async def test_latest_snapshot_skips_a_payload_pruned_after_the_index_read(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages()))
        del client.hashes[f'{_P}:snapshot_payloads:{{r1}}']['2']

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None and snap.step_index == 0

    async def test_snapshot_seq_monotonic_across_reset_step_index(self) -> None:
        """A reused `run_id` whose `step_index` restarts must not clobber the earlier snapshot."""
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=7, messages=_messages('first')))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages('second')))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [7, 0]
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None and snap.step_index == 0

    async def test_tool_effect_upsert_and_run_scope(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r1', status='started')
        )
        completed = ToolEffectRecord(
            tool_call_id='t1',
            tool_name='ship',
            run_id='r1',
            status='completed',
            ended_at=_at(5),
            effect_summary='shipped',
        )
        await store.record_tool_effect(completed)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='ship', run_id='r2', status='started')
        )

        assert await store.get_tool_effect(run_id='r1', tool_call_id='t1') == completed
        other = await store.get_tool_effect(run_id='r2', tool_call_id='t1')
        assert other is not None and other.status == 'started'

    async def test_list_unresolved_tool_effects(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='a', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t2', tool_name='b', run_id='r1', status='completed')
        )
        await store.record_tool_effect(ToolEffectRecord(tool_call_id='t3', tool_name='c', run_id='r1', status='failed'))

        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['t1']

    async def test_decode_responses_client(self) -> None:
        """A client built with `decode_responses=True` returns `str`, and reads still work."""
        store = RedisStepStore(FakeRedis(decode=True))
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )

        assert [r.run_id for r in await store.list_runs(conversation_id='c1')] == ['r1']
        assert [e.kind for e in await store.list_events(run_id='r1')] == ['run_started']
        assert await store.latest_snapshot(run_id='r1') is not None
        assert [r.tool_call_id for r in await store.list_unresolved_tool_effects(run_id='r1')] == ['t1']

    async def test_a_reply_that_is_neither_str_nor_bytes_raises(self) -> None:
        store = RedisStepStore(UntypedReplyRedis())
        with pytest.raises(ValueError, match='expected a string reply, got int'):
            await store.get_run(run_id='r1')

    async def test_a_corrupted_snapshot_payload_raises(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        payloads = client.hashes[f'{_P}:snapshot_payloads:{{r1}}']
        payloads['1'] = payloads['1'].replace('"step_index":0', '"step_index":"0"', 1)

        # `ValidationError` subclasses `ValueError`, so the store's rejection contract holds.
        with pytest.raises(ValueError, match='step_index'):
            await store.latest_snapshot(run_id='r1')

    async def test_a_snapshot_missing_a_stored_message_raises(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        client.hashes[f'{_P}:messages:{{r1}}'].clear()

        with pytest.raises(ValueError, match='references 1 stored message'):
            await store.latest_snapshot(run_id='r1')

    async def test_non_ascii_metadata_round_trips(self) -> None:
        store = RedisStepStore(FakeRedis())
        await store.register_run(RunRecord(run_id='r1', metadata={'ключ': 'значение'}))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, metadata={'emoji': '🚀'}))

        run = await store.get_run(run_id='r1')
        assert run is not None and run.metadata == {'ключ': 'значение'}
        assert [e.metadata for e in await store.list_events(run_id='r1')] == [{'emoji': '🚀'}]


class TestRedisStepStoreListSnapshots:
    async def test_write_order_and_interrupted_filter(self) -> None:
        store = RedisStepStore(FakeRedis())
        # Descending then ascending `step_index` so write order is neither a
        # `step_index` sort nor its reverse.
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=_messages()))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages(), state='interrupted')
        )
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=9, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [2, 1]
        opted = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in opted] == [2, 0, 1]
        assert [s.state for s in opted] == ['complete', 'interrupted', 'complete']

    async def test_skips_a_payload_pruned_after_the_index_read(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages()))
        del client.hashes[f'{_P}:snapshot_payloads:{{r1}}']['1']

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [1]

    async def test_skips_an_unparsable_payload(self, caplog: pytest.LogCaptureFixture) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=_messages()))
        client.hashes[f'{_P}:snapshot_payloads:{{r1}}']['1'] = '{"not": "a snapshot"}'

        with caplog.at_level(logging.WARNING):
            snapshots = await store.list_snapshots(run_id='r1')
        assert [s.step_index for s in snapshots] == [1]
        assert 'Skipping unparsable snapshot for run r1' in caplog.text

    async def test_store_is_accepted_as_a_search_substrate(self) -> None:
        """`SnapshotHistorySource` rejects stores lacking `list_snapshots` at construction."""
        store = RedisStepStore(FakeRedis())
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages('remember this')))

        source = SnapshotHistorySource(store)
        assert [type(m).__name__ for m in await source.run_history(run_id='r1')] == ['ModelRequest']


class TestRedisStepStoreRetention:
    async def test_unbounded_keeps_every_snapshot(self) -> None:
        store = RedisStepStore(FakeRedis())
        for step in range(4):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0, 1, 2, 3]

    async def test_prunes_to_the_newest_window(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client, max_snapshots_per_run=2)
        for step in range(5):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [3, 4]
        assert set(client.zsets[f'{_P}:snapshots:{{r1}}']) == {'4:complete', '5:complete'}
        assert set(client.hashes[f'{_P}:snapshot_payloads:{{r1}}']) == {'4', '5'}

    async def test_keeps_newest_complete_below_the_window(self) -> None:
        store = RedisStepStore(FakeRedis(), max_snapshots_per_run=2)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        for step in (1, 2, 3):
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=step, messages=_messages(), state='interrupted')
            )

        retained = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in retained] == [0, 2, 3]
        resumable = await store.latest_snapshot(run_id='r1')
        assert resumable is not None and resumable.step_index == 0

    async def test_pruning_is_scoped_to_the_written_run(self) -> None:
        store = RedisStepStore(FakeRedis(), max_snapshots_per_run=1)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        for step in range(3):
            await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=step, messages=_messages()))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0]
        assert [s.step_index for s in await store.list_snapshots(run_id='r2')] == [2]


class TestRedisStepStoreExpiry:
    async def test_no_expiry_by_default(self) -> None:
        client = FakeRedis()
        store = RedisStepStore(client)
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages()))
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )

        assert client.expiries == {}

    async def test_any_write_refreshes_every_key_of_the_run(self) -> None:
        """The run expires as a unit, so a tool-effect write keeps its snapshots alive too."""
        client = FakeRedis()
        store = RedisStepStore(client, expire_seconds=120)
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='e'))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=0, messages=_messages(), idempotency_key='s')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )
        client.expiries.clear()

        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='completed')
        )

        assert client.expiries == {
            f'{_P}:runs:all': 120,
            f'{_P}:run:{{r1}}': 120,
            f'{_P}:events:{{r1}}': 120,
            f'{_P}:event_keys:{{r1}}': 120,
            f'{_P}:snapshots:{{r1}}': 120,
            f'{_P}:snapshots:seq:{{r1}}': 120,
            f'{_P}:snapshot_payloads:{{r1}}': 120,
            f'{_P}:snapshot_keys:{{r1}}': 120,
            f'{_P}:messages:{{r1}}': 120,
            f'{_P}:tool_effects:{{r1}}': 120,
        }

    async def test_index_sets_take_the_ttl_from_the_writes_that_carry_their_ids(self) -> None:
        """A set expires with the last run it holds, so it cannot grow without bound."""
        client = FakeRedis()
        store = RedisStepStore(client, expire_seconds=120)
        all_runs = f'{_P}:runs:all'
        index_keys = {all_runs, f'{_P}:runs:conversation:c1', f'{_P}:runs:parent:p1'}
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', parent_run_id='p1'))
        assert all(client.expiries[key] == 120 for key in index_keys)

        client.expiries.clear()
        await store.append_event(
            StepEvent(run_id='r1', kind='run_started', step_index=0, conversation_id='c1', parent_run_id='p1')
        )
        assert index_keys <= set(client.expiries)

        client.expiries.clear()
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id='r1', step_index=0, messages=_messages(), conversation_id='c1', parent_run_id='p1'
            )
        )
        assert index_keys <= set(client.expiries)

        # `ToolEffectRecord` carries no lineage ids, so only `runs:all` is refreshed.
        client.expiries.clear()
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='x', run_id='r1', status='started')
        )
        assert set(client.expiries) & index_keys == {all_runs}


class TestRedisStepStoreMedia:
    async def test_large_binary_and_text_externalized_and_restored(self, tmp_path: Path) -> None:
        media = tmp_path / 'media'
        store = RedisStepStore(FakeRedis(), media_store=DiskMediaStore(media), media_threshold_bytes=64 * 1024)
        big_binary = b'\xab' * 100_000
        big_text = 'Z' * 100_000
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(content=[BinaryContent(data=big_binary, media_type='image/png')]),
                    ToolReturnPart(tool_name='scrape', content=big_text, tool_call_id='t1'),
                ]
            ),
            ModelResponse(parts=[TextPart(content='done')]),
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        assert len(list(media.iterdir())) == 2

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = prompt.content[0]
        assert isinstance(binary, BinaryContent)
        assert binary.data == big_binary
        tool_return = request.parts[1]
        assert isinstance(tool_return, ToolReturnPart)
        assert tool_return.content == big_text


class TestRedisStepStoreThroughAgent:
    async def test_agent_run_round_trips_through_step_persistence(self) -> None:
        store = RedisStepStore(FakeRedis(), expire_seconds=3600)
        agent: Agent[None, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='vision')],
        )
        result = await agent.run('classify this')
        assert isinstance(result.output, str)

        runs = await store.list_runs()
        assert len(runs) == 1
        assert runs[0].agent_name == 'vision'
        kinds = [e.kind for e in await store.list_events(run_id=runs[0].run_id)]
        assert kinds[0] == 'run_started' and kinds[-1] == 'run_completed'
        snap = await store.latest_snapshot(run_id=runs[0].run_id)
        assert snap is not None
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert prompt.content == 'classify this'

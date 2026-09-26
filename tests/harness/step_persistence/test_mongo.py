"""Tests for `MongoStepStore` against an in-memory `mongomock-motor` client.

The store uses only the async collection surface `mongomock-motor` provides
(`insert_one`, `find_one`, `update_one` upsert, `find_one_and_update` with
`$inc`, sorted `find`), so it reaches full coverage without a running mongod.
`_mock_client` is the single type-boundary shim (the fake is not a
`pymongo.AsyncMongoClient`).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest
from mongomock_motor import AsyncMongoMockClient
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
from pymongo import AsyncMongoClient

import pydantic_ai_harness.step_persistence as sp
from pydantic_ai_harness.conversation_search import SnapshotHistorySource
from pydantic_ai_harness.media import MongoMediaStore
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    MongoStepStore,
    RunRecord,
    StepEvent,
    StepPersistence,
    ToolEffectRecord,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def _mock_client() -> AsyncMongoClient[dict[str, object]]:
    return AsyncMongoMockClient()  # pyright: ignore[reportUnknownVariableType, reportReturnType]


class TestMongoStepStoreConstruction:
    def test_requires_exactly_one_of_client_or_db_url(self) -> None:
        with pytest.raises(ValueError, match='exactly one'):
            MongoStepStore(database='t')
        with pytest.raises(ValueError, match='exactly one'):
            MongoStepStore(client=_mock_client(), db_url='mongodb://x', database='t')

    def test_requires_database(self) -> None:
        with pytest.raises(ValueError, match='`database=` is required'):
            MongoStepStore(client=_mock_client())

    def test_auto_media_store_shares_client(self) -> None:
        client = _mock_client()
        store = MongoStepStore(client=client, database='t')
        media_store = store._media_store  # pyright: ignore[reportPrivateUsage]
        assert isinstance(media_store, MongoMediaStore)
        assert media_store._client is client  # pyright: ignore[reportPrivateUsage]

    def test_media_store_none_disables_media(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        assert store._media_store is None  # pyright: ignore[reportPrivateUsage]

    def test_rejects_invalid_max_snapshots_per_run(self) -> None:
        with pytest.raises(ValueError, match='must be an int >= 1 or None'):
            MongoStepStore(client=_mock_client(), database='t', max_snapshots_per_run=1.5)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match='must be >= 1 or None'):
            MongoStepStore(client=_mock_client(), database='t', max_snapshots_per_run=0)

    async def test_db_url_owns_client_and_aclose_closes_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        closed: list[object] = []

        async def _record(client: AsyncMongoClient[dict[str, object]]) -> None:
            closed.append(client)

        monkeypatch.setattr(AsyncMongoClient, 'close', _record)
        store = MongoStepStore(db_url='mongodb://localhost:59017', database='t', media_store=None)
        await store.aclose()
        assert len(closed) == 1
        assert closed[0] is store._client  # pyright: ignore[reportPrivateUsage]

        monkeypatch.undo()
        await store.aclose()

    async def test_shared_client_aclose_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _mock_client()
        closed: list[object] = []

        async def _record() -> None:
            closed.append(client)

        monkeypatch.setattr(client, 'close', _record)
        store = MongoStepStore(client=client, database='t', media_store=None)
        await store.aclose()
        assert closed == []

        # Prove the spy would have recorded a close, so the empty list above is
        # an observation rather than a spy that never wired up.
        await client.close()
        assert closed == [client]


class TestMongoStepStoreProtocol:
    async def test_pruned_snapshot_key_remains_suppressed(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None, max_snapshots_per_run=1)
        older = ContinuableSnapshot(run_id='r1', step_index=1, messages=[], idempotency_key='0:1:complete')
        newer = ContinuableSnapshot(run_id='r1', step_index=2, messages=[], idempotency_key='1:2:complete')

        await store.save_snapshot(older)
        await store.save_snapshot(newer)
        await store.save_snapshot(older)

        assert await store.latest_snapshot(run_id='r1') == newer

    async def test_snapshot_document_repairs_missing_suppression_ledger(self) -> None:
        client = _mock_client()
        store = MongoStepStore(client=client, database='t', media_store=None)
        snapshot = ContinuableSnapshot(run_id='r1', step_index=1, messages=[], idempotency_key='0:1:complete')
        await store.save_snapshot(snapshot)
        await client['t']['snapshot_idempotency_keys'].delete_many({})

        await store.save_snapshot(snapshot)

        assert await store.list_snapshots(run_id='r1') == [snapshot]
        assert await client['t']['snapshot_idempotency_keys'].count_documents({}) == 1

    async def test_register_and_get_run(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1', agent_name='agent', metadata={'k': 'v'}))
        fetched = await store.get_run(run_id='r1')
        assert fetched is not None
        assert fetched.conversation_id == 'c1'
        assert fetched.metadata == {'k': 'v'}

    async def test_register_duplicate_run_raises_value_error(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        with pytest.raises(ValueError, match="run_id 'r1' is already in the store"):
            await store.register_run(RunRecord(run_id='r1', agent_name='racing-run'))

    async def test_list_runs_chronological(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        await store.register_run(RunRecord(run_id='r3', started_at=base + timedelta(seconds=3)))
        await store.register_run(RunRecord(run_id='r1', started_at=base + timedelta(seconds=1)))
        await store.register_run(RunRecord(run_id='r2', started_at=base + timedelta(seconds=2)))
        records = await store.list_runs()
        assert [r.run_id for r in records] == ['r1', 'r2', 'r3']

    async def test_list_runs_filters(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1', conversation_id='a', parent_run_id='p'))
        await store.register_run(RunRecord(run_id='r2', conversation_id='a', parent_run_id='q'))
        await store.register_run(RunRecord(run_id='r3', conversation_id='b', parent_run_id='p'))

        assert {r.run_id for r in await store.list_runs(conversation_id='a')} == {'r1', 'r2'}
        assert {r.run_id for r in await store.list_runs(parent_run_id='p')} == {'r1', 'r3'}
        assert [r.run_id for r in await store.list_runs(parent_run_id='p', conversation_id='a')] == ['r1']

    async def test_list_runs_sorts_by_instant_not_iso_string(self) -> None:
        """Mixed-offset timestamps sort by instant, matching the base stores' contract.

        A lexicographic sort of the stored ISO string would put `late` first;
        by instant `early` (an earlier moment behind a +05:00 offset) comes first.
        """
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        early_instant = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=5)))  # 2023-12-31T20:00Z
        late_instant = datetime(2024, 1, 1, 0, 30, 0, tzinfo=timezone.utc)
        await store.register_run(RunRecord(run_id='late', started_at=late_instant))
        await store.register_run(RunRecord(run_id='early', started_at=early_instant))
        assert [r.run_id for r in await store.list_runs()] == ['early', 'late']

    async def test_append_and_list_events(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(
            StepEvent(run_id='r1', kind='tool_call_started', step_index=1, tool_call_id='t1', tool_name='add')
        )
        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_started', 'tool_call_started']
        assert events[1].tool_call_id == 't1'

    async def test_save_and_load_snapshot(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hello')]),
            ModelResponse(parts=[TextPart(content='hi back')]),
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=messages))
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 2
        assert len(snap.messages) == 2

    async def test_latest_snapshot_default_skips_interrupted(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=msgs, state='interrupted'))

        default = await store.latest_snapshot(run_id='r1')
        assert default is not None and default.state == 'complete' and default.step_index == 0
        opted = await store.latest_snapshot(run_id='r1', include_interrupted=True)
        assert opted is not None and opted.state == 'interrupted' and opted.step_index == 1

    async def test_snapshot_seq_monotonic_across_reset_step(self) -> None:
        """A reused run_id whose step_index resets to 0 must not clobber the prior snapshot."""
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=5, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 0  # last write wins, not the highest step_index

    async def test_keyed_snapshot_replays_preserve_complete_and_interrupted_at_same_step(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        complete = ContinuableSnapshot(
            run_id='r1', step_index=2, messages=msgs, state='complete', idempotency_key='2:complete'
        )
        interrupted = ContinuableSnapshot(
            run_id='r1', step_index=2, messages=msgs, state='interrupted', idempotency_key='2:interrupted'
        )

        await store.save_snapshot(complete)
        await store.save_snapshot(interrupted)
        await store.save_snapshot(complete)
        await store.save_snapshot(interrupted)

        snapshots = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [(snapshot.step_index, snapshot.state) for snapshot in snapshots] == [
            (2, 'complete'),
            (2, 'interrupted'),
        ]
        assert await store.latest_snapshot(run_id='r1') == complete

    async def test_keyed_event_replay_is_suppressed_but_unkeyed_events_append(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        keyed = StepEvent(run_id='r1', kind='run_started', step_index=0, idempotency_key='event:0')
        await store.append_event(keyed)
        await store.append_event(keyed)
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        assert len(await store.list_events(run_id='r1')) == 3

    async def test_tool_effect_upsert_and_scope(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='completed', effect_summary='ok')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r2', status='started')
        )
        effect = await store.get_tool_effect(run_id='r1', tool_call_id='t1')
        assert effect is not None and effect.status == 'completed' and effect.effect_summary == 'ok'
        other = await store.get_tool_effect(run_id='r2', tool_call_id='t1')
        assert other is not None and other.status == 'started'

    async def test_list_unresolved_tool_effects(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='t2', tool_name='mul', run_id='r1', status='completed')
        )
        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['t1']

    async def test_missing_lookups_return_none_or_empty(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        assert await store.get_run(run_id='nope') is None
        assert await store.latest_snapshot(run_id='nope') is None
        assert await store.get_tool_effect(run_id='nope', tool_call_id='x') is None
        assert await store.list_events(run_id='nope') == []
        assert await store.list_unresolved_tool_effects(run_id='nope') == []
        assert await store.list_runs() == []

    async def test_non_ascii_metadata_round_trips(self) -> None:
        """Run and event `metadata` survives as a native BSON subdocument.

        Unlike `MongoMediaStore`, which JSON-encodes its metadata, the run and
        event documents nest it, so both keys and values go through BSON's own
        UTF-8 encoding.
        """
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        metadata = {'user': 'Ada Lovelace ✨', 'ключ': 'ジョブ完了', 'emoji': '🙂'}
        await store.register_run(RunRecord(run_id='r1', metadata=metadata))
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0, metadata=metadata))

        run = await store.get_run(run_id='r1')
        assert run is not None
        assert run.metadata == metadata
        assert [e.metadata for e in await store.list_events(run_id='r1')] == [metadata]

    async def test_corrupted_snapshot_document_raises(self) -> None:
        """A poked snapshot with a wrong-typed field surfaces as a ValueError."""
        client = _mock_client()
        store = MongoStepStore(client=client, database='t', media_store=None)
        await store.register_run(RunRecord(run_id='r1'))
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='x')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        await client['t']['snapshots'].update_one({'run_id': 'r1'}, {'$set': {'timestamp': 123}})
        with pytest.raises(ValueError, match='snapshot document has wrong types'):
            await store.latest_snapshot(run_id='r1')


class TestMongoStepStoreListSnapshots:
    async def test_write_order_and_interrupted_filter(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        # Descending then ascending `step_index` so write order is neither a
        # `step_index` sort nor its reverse.
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs, state='interrupted'))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=9, messages=msgs))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [2, 1]
        opted = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in opted] == [2, 0, 1]
        assert [s.state for s in opted] == ['complete', 'interrupted', 'complete']
        assert await store.list_snapshots(run_id='nope') == []

    async def test_restores_externalized_media(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_threshold_bytes=1024)
        big_text = 'Z' * 5_000
        messages: list[ModelMessage] = [
            ModelRequest(parts=[ToolReturnPart(tool_name='scrape', content=big_text, tool_call_id='t1')])
        ]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))

        snapshots = await store.list_snapshots(run_id='r1')
        assert len(snapshots) == 1
        request = snapshots[0].messages[0]
        assert isinstance(request, ModelRequest)
        tool_return = request.parts[0]
        assert isinstance(tool_return, ToolReturnPart)
        assert tool_return.content == big_text

    async def test_store_is_accepted_as_a_search_substrate(self) -> None:
        """`SnapshotHistorySource` rejects stores lacking `list_snapshots` at construction."""
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='remember this')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))

        source = SnapshotHistorySource(store)
        assert [type(m).__name__ for m in await source.run_history(run_id='r1')] == ['ModelRequest']

    async def test_skips_unparsable_document(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _mock_client()
        store = MongoStepStore(client=client, database='t', media_store=None)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=msgs))
        await client['t']['snapshots'].update_one({'step_index': 0}, {'$set': {'timestamp': 123}})

        with caplog.at_level(logging.WARNING):
            snapshots = await store.list_snapshots(run_id='r1')
        assert [s.step_index for s in snapshots] == [1]
        assert 'Skipping unparsable snapshot document for run r1' in caplog.text


class TestMongoStepStoreRetention:
    async def test_unbounded_keeps_every_snapshot(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        for step in range(4):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=msgs))
        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [0, 1, 2, 3]

    async def test_prunes_to_the_newest_window(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None, max_snapshots_per_run=2)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        for step in range(5):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=msgs))
        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [3, 4]

    async def test_keeps_newest_complete_below_the_window(self) -> None:
        """A `complete` snapshot pushed out of the window by `interrupted` writes survives."""
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None, max_snapshots_per_run=2)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        for step in (1, 2, 3):
            await store.save_snapshot(
                ContinuableSnapshot(run_id='r1', step_index=step, messages=msgs, state='interrupted')
            )

        retained = await store.list_snapshots(run_id='r1', include_interrupted=True)
        assert [s.step_index for s in retained] == [0, 2, 3]
        resumable = await store.latest_snapshot(run_id='r1')
        assert resumable is not None
        assert resumable.step_index == 0

    async def test_pruning_is_scoped_to_the_written_run(self) -> None:
        store = MongoStepStore(client=_mock_client(), database='t', media_store=None, max_snapshots_per_run=1)
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='a')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r2', step_index=0, messages=msgs))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=msgs))

        assert [s.step_index for s in await store.list_snapshots(run_id='r1')] == [1]
        assert [s.step_index for s in await store.list_snapshots(run_id='r2')] == [0]


class TestMongoStepStoreMedia:
    async def test_large_binary_and_text_externalized_and_restored(self) -> None:
        """Both a large binary part and a large text tool-return round-trip through media."""
        client = _mock_client()
        store = MongoStepStore(client=client, database='t', media_threshold_bytes=64 * 1024)
        await store.register_run(RunRecord(run_id='r1'))
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

        # Two blobs (binary + text) live in the shared media collections.
        assert await client['t']['media'].count_documents({}) == 2

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

    async def test_below_threshold_stays_inline(self) -> None:
        client = _mock_client()
        store = MongoStepStore(client=client, database='t', media_threshold_bytes=64 * 1024)
        await store.register_run(RunRecord(run_id='r1'))
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='small')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=messages))
        assert await client['t']['media'].count_documents({}) == 0

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        response = snap.messages[0]
        assert isinstance(response, ModelResponse)
        text = response.parts[0]
        assert isinstance(text, TextPart)
        assert text.content == 'small'

    async def test_agent_run_round_trips_through_step_persistence(self) -> None:
        """An Agent run with a large BinaryContent prompt persists and restores via Mongo."""
        store = MongoStepStore(client=_mock_client(), database='t')
        agent: Agent[None, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='vision')],
        )
        big = b'\xab' * 100_000
        result = await agent.run(['classify this image', BinaryContent(data=big, media_type='image/png')])
        assert isinstance(result.output, str)

        runs = await store.list_runs()
        assert len(runs) == 1
        snap = await store.latest_snapshot(run_id=runs[0].run_id)
        assert snap is not None
        request = snap.messages[0]
        assert isinstance(request, ModelRequest)
        prompt = request.parts[0]
        assert isinstance(prompt, UserPromptPart)
        assert isinstance(prompt.content, list)
        binary = next(p for p in prompt.content if isinstance(p, BinaryContent))
        assert binary.data == big


class TestStepPersistenceLazyExport:
    def test_mongo_step_store_lazily_exported(self) -> None:

        assert sp.MongoStepStore is MongoStepStore

    def test_unknown_attribute_raises(self) -> None:

        with pytest.raises(AttributeError, match='has no attribute'):
            _ = sp.NoSuchStore  # type: ignore[attr-defined]

"""Tests for the `StepPersistence` capability.

Exercises the public capability behavior through `Agent(...)`/`TestModel` and
covers the helper / store branches that are awkward to reach through a real
agent run (e.g. the path-traversal guard on `FileStepStore`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai._agent_graph import GraphAgentState  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    InMemoryStepStore,
    RunRecord,
    SqliteStepStore,
    StepEvent,
    StepPersistence,
    StepStore,
    ToolEffectRecord,
    continue_run,
    fork_run,
    is_provider_valid,
)
from pydantic_ai_harness.step_persistence._store import _validate_id  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    """Restrict async tests to asyncio (Agent.run uses `asyncio.create_task`)."""
    return 'asyncio'


def build_run_context(
    deps: object = None,
    *,
    run_id: str | None = None,
    run_step: int = 0,
    conversation_id: str | None = None,
) -> RunContext[Any]:
    """Fabricate a minimal `RunContext` for direct hook invocation."""
    return RunContext[Any](
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
        run_id=run_id,
        conversation_id=conversation_id,
    )


def make_simple_agent(capabilities: list[Any]) -> Agent[object, str]:
    agent: Agent[object, str] = Agent(TestModel(), capabilities=capabilities)

    @agent.tool_plain
    def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
        return a + b

    return agent


async def first_run_id(store: StepStore) -> str:
    runs = await store.list_runs()
    assert len(runs) >= 1
    return runs[0].run_id


# ---------------------------------------------------------------------------
# is_provider_valid
# ---------------------------------------------------------------------------


class TestIsProviderValid:
    def test_empty_history_is_valid(self) -> None:
        assert is_provider_valid([]) is True

    def test_matched_tool_call_is_valid(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='c1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content=3, tool_call_id='c1')]),
        ]
        assert is_provider_valid(messages) is True

    def test_unmatched_tool_call_is_invalid(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='c1')])
        ]
        assert is_provider_valid(messages) is False

    def test_retry_prompt_resolves_a_tool_call(self) -> None:
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='c1')]),
            ModelRequest(parts=[RetryPromptPart(content='try again', tool_name='add', tool_call_id='c1')]),
        ]
        assert is_provider_valid(messages) is True

    def test_request_only_history_is_valid(self) -> None:
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
        assert is_provider_valid(messages) is True

    def test_text_only_response_is_valid(self) -> None:
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content='hi')])]
        assert is_provider_valid(messages) is True

    def test_orphan_tool_return_is_invalid(self) -> None:
        """Return whose `tool_call_id` was never opened by any prior call -> reject."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content=1, tool_call_id='ghost')]),
        ]
        assert is_provider_valid(messages) is False

    def test_duplicate_tool_return_is_invalid(self) -> None:
        """Two returns for the same `tool_call_id` -> the second has no open call."""
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='c1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content=1, tool_call_id='c1')]),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content=2, tool_call_id='c1')]),
        ]
        assert is_provider_valid(messages) is False

    def test_out_of_order_tool_return_is_invalid(self) -> None:
        """Return appearing before its call in a later response -> reject."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content=1, tool_call_id='c1')]),
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='c1')]),
        ]
        assert is_provider_valid(messages) is False

    def test_orphan_retry_prompt_is_invalid(self) -> None:
        """`RetryPromptPart` with no matching open call is also rejected."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[RetryPromptPart(content='retry', tool_name='add', tool_call_id='ghost')]),
        ]
        assert is_provider_valid(messages) is False


# ---------------------------------------------------------------------------
# continue_run / fork_run
# ---------------------------------------------------------------------------


class TestContinueAndForkRun:
    async def test_continue_run_returns_snapshot_messages(self) -> None:
        store = InMemoryStepStore()
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))

        loaded = await continue_run(store, run_id='r1')
        assert len(loaded) == 1
        assert loaded is not msgs  # caller gets an independent list

    async def test_continue_run_raises_when_no_snapshot(self) -> None:
        store = InMemoryStepStore()
        with pytest.raises(LookupError, match="no continuable snapshot for run_id 'missing'"):
            await continue_run(store, run_id='missing')

    async def test_fork_run_delegates_to_continue_run(self) -> None:
        store = InMemoryStepStore()
        msgs: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content='hi')])]
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=msgs))

        forked = await fork_run(store, run_id='r1')
        assert len(forked) == 1


# ---------------------------------------------------------------------------
# _validate_id
# ---------------------------------------------------------------------------


class TestValidateId:
    @pytest.mark.parametrize('bad', ['../evil', 'a/b', '', 'a..b', '..', 'has space', 'x' * 201])
    def test_rejects_bad_ids(self, bad: str) -> None:
        with pytest.raises(ValueError, match='invalid run_id'):
            _validate_id(bad, field='run_id')

    @pytest.mark.parametrize('good', ['good', 'a.b-c_d', 'A1', 'x' * 200])
    def test_accepts_safe_ids(self, good: str) -> None:
        _validate_id(good, field='run_id')


# ---------------------------------------------------------------------------
# InMemoryStepStore
# ---------------------------------------------------------------------------


class TestInMemoryStepStore:
    async def test_register_and_get_run(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1', agent_name='a'))

        record = await store.get_run(run_id='r1')
        assert record is not None
        assert record.agent_name == 'a'
        assert await store.get_run(run_id='missing') is None

    async def test_list_runs_with_and_without_parent_filter(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1', parent_run_id=None))
        await store.register_run(RunRecord(run_id='r2', parent_run_id='r1'))
        await store.register_run(RunRecord(run_id='r3', parent_run_id='r1'))
        await store.register_run(RunRecord(run_id='r4', parent_run_id='other'))

        assert {r.run_id for r in await store.list_runs()} == {'r1', 'r2', 'r3', 'r4'}
        children = await store.list_runs(parent_run_id='r1')
        assert {r.run_id for r in children} == {'r2', 'r3'}

    async def test_list_runs_filters_by_conversation_id(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1', conversation_id='conv-A'))
        await store.register_run(RunRecord(run_id='r2', conversation_id='conv-A'))
        await store.register_run(RunRecord(run_id='r3', conversation_id='conv-B'))

        a_runs = await store.list_runs(conversation_id='conv-A')
        assert {r.run_id for r in a_runs} == {'r1', 'r2'}

    async def test_list_runs_combines_parent_and_conversation_filters(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1', parent_run_id='p', conversation_id='conv-A'))
        await store.register_run(RunRecord(run_id='r2', parent_run_id='p', conversation_id='conv-B'))
        await store.register_run(RunRecord(run_id='r3', parent_run_id='other', conversation_id='conv-A'))

        narrowed = await store.list_runs(parent_run_id='p', conversation_id='conv-A')
        assert [r.run_id for r in narrowed] == ['r1']

    async def test_append_and_list_events(self) -> None:
        store = InMemoryStepStore()
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(StepEvent(run_id='r1', kind='run_completed', step_index=1))

        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_started', 'run_completed']
        assert await store.list_events(run_id='missing') == []

    async def test_latest_snapshot_returns_last_appended(self) -> None:
        store = InMemoryStepStore()
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[]))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=[]))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=[]))

        latest = await store.latest_snapshot(run_id='r1')
        assert latest is not None
        # InMemoryStepStore returns the last *appended* snapshot.
        assert latest.step_index == 1
        assert await store.latest_snapshot(run_id='missing') is None

    async def test_tool_effects_started_then_completed(self) -> None:
        store = InMemoryStepStore()
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='started')
        )
        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['c1']

        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='completed')
        )
        assert await store.list_unresolved_tool_effects(run_id='r1') == []

        # mix completed and another started; only the started one is unresolved.
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c2', tool_name='add', run_id='r1', status='started')
        )
        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['c2']

    async def test_get_tool_effect_returns_latest_or_none(self) -> None:
        store = InMemoryStepStore()
        assert await store.get_tool_effect(run_id='r1', tool_call_id='missing') is None

        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='completed')
        )

        record = await store.get_tool_effect(run_id='r1', tool_call_id='c1')
        assert record is not None
        assert record.status == 'completed'


# ---------------------------------------------------------------------------
# FileStepStore
# ---------------------------------------------------------------------------


class TestFileStepStore:
    async def test_runs_round_trip(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.register_run(RunRecord(run_id='r1', parent_run_id='p1', agent_name='a', metadata={'k': 'v'}))

        record = await store.get_run(run_id='r1')
        assert record is not None
        assert record.parent_run_id == 'p1'
        assert record.agent_name == 'a'
        assert record.metadata == {'k': 'v'}
        assert await store.get_run(run_id='missing') is None

    async def test_list_runs_returns_empty_when_root_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'does-not-exist')
        assert await store.list_runs() == []

    async def test_list_runs_skips_directories_without_run_json(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        (tmp_path / 'orphan').mkdir()
        await store.register_run(RunRecord(run_id='real', agent_name='a'))

        runs = await store.list_runs()
        assert [r.run_id for r in runs] == ['real']

    async def test_list_runs_filters_by_parent(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.register_run(RunRecord(run_id='r1', parent_run_id=None))
        await store.register_run(RunRecord(run_id='r2', parent_run_id='r1'))

        children = await store.list_runs(parent_run_id='r1')
        assert [r.run_id for r in children] == ['r2']

    async def test_list_runs_filters_by_conversation_id(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.register_run(RunRecord(run_id='r1', conversation_id='conv-A'))
        await store.register_run(RunRecord(run_id='r2', conversation_id='conv-B'))

        assert [r.run_id for r in await store.list_runs(conversation_id='conv-A')] == ['r1']

    async def test_events_round_trip_skips_blank_lines(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.append_event(StepEvent(run_id='r1', kind='run_started', step_index=0))
        await store.append_event(
            StepEvent(
                run_id='r1',
                kind='tool_call_started',
                step_index=1,
                tool_call_id='c1',
                tool_name='add',
                metadata={'k': 'v'},
            )
        )

        # Inject a blank line to exercise the strip() branch on read.
        events_file = tmp_path / 'r1' / 'events.jsonl'
        events_file.write_text(events_file.read_text(encoding='utf-8') + '\n', encoding='utf-8')

        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_started', 'tool_call_started']
        assert events[1].metadata == {'k': 'v'}
        assert events[1].tool_call_id == 'c1'

    async def test_list_events_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        assert await store.list_events(run_id='nonexistent') == []

    async def test_snapshot_round_trip(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='ok')]),
        ]
        await store.save_snapshot(
            ContinuableSnapshot(
                run_id='r1',
                step_index=0,
                messages=messages,
                conversation_id='c1',
                parent_run_id='p1',
                agent_name='a',
            )
        )

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 0
        assert snap.conversation_id == 'c1'
        assert snap.parent_run_id == 'p1'
        assert snap.agent_name == 'a'
        assert len(snap.messages) == 2

    async def test_latest_snapshot_picks_most_recent_seq(self, tmp_path: Path) -> None:
        """Latest is by physical write order (filename seq), not by `step_index`."""
        store = FileStepStore(tmp_path)
        for step in (0, 2, 1):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=step, messages=[]))

        # Drop a non-integer filename to exercise the ValueError branch.
        (tmp_path / 'r1' / 'snapshots' / 'not-a-number.json').write_text('{}', encoding='utf-8')

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        # The last save had step_index=1 and was written as the highest seq.
        assert snap.step_index == 1

    async def test_lower_step_index_save_supersedes_earlier(self, tmp_path: Path) -> None:
        """A reused `run_id` whose later save has a LOWER `step_index` is still
        treated as the latest -- the physical seq wins."""
        store = FileStepStore(tmp_path)
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=5, messages=[]))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=2, messages=[]))

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 2

    async def test_snapshot_seq_counter_increments(self, tmp_path: Path) -> None:
        """Three consecutive saves produce files `0.json`, `1.json`, `2.json`."""
        store = FileStepStore(tmp_path)
        for _ in range(3):
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[]))

        snap_dir = tmp_path / 'r1' / 'snapshots'
        names = sorted(p.name for p in snap_dir.glob('*.json'))
        assert names == ['0.json', '1.json', '2.json']

    async def test_snapshot_seq_counter_skips_non_integer_filenames(self, tmp_path: Path) -> None:
        """`_next_snapshot_seq` ignores `*.json` files whose stem is not an int."""
        store = FileStepStore(tmp_path)
        snap_dir = tmp_path / 'r1' / 'snapshots'
        snap_dir.mkdir(parents=True)
        (snap_dir / 'not-a-number.json').write_text('{}', encoding='utf-8')

        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[]))

        # Despite the foreign file, the new snapshot is written as `0.json`.
        assert (snap_dir / '0.json').exists()

    async def test_snapshot_seq_counter_keeps_max_across_unordered_iter(self, tmp_path: Path) -> None:
        """`_next_snapshot_seq` keeps the highest seq even when `glob` yields lower ones later."""
        store = FileStepStore(tmp_path)
        snap_dir = tmp_path / 'r1' / 'snapshots'
        snap_dir.mkdir(parents=True)
        # Pre-populate multiple numeric files so `glob` iteration hits both
        # the `seq > max_seq` true branch and its false branch (a lower seq
        # seen after a higher one already set the max). Insertion order on
        # APFS / ext4 is the directory iteration order: write the high stem
        # first so the lower ones that follow hit the False branch.
        # Lexicographic glob order puts `10.json` before `1.json`, so the
        # iteration encounters a high seq first and then several lower seqs,
        # forcing the `seq > max_seq` False branch.
        for seq in (10, 1, 2, 9):
            (snap_dir / f'{seq}.json').write_text('{}', encoding='utf-8')

        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[]))

        # Next seq must be 11 (one above the highest existing numeric stem).
        assert (snap_dir / '11.json').exists()

    async def test_latest_snapshot_returns_none_when_dir_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        assert await store.latest_snapshot(run_id='nope') is None

    async def test_latest_snapshot_returns_none_when_dir_empty(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.register_run(RunRecord(run_id='r1'))  # creates snapshots/ but no files
        assert await store.latest_snapshot(run_id='r1') is None

    async def test_tool_effects_round_trip(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='r1', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(
                tool_call_id='c2',
                tool_name='mul',
                run_id='r1',
                status='completed',
                idempotency_key='k',
                effect_summary='ok',
            )
        )

        # Blank line to exercise the strip branch on read.
        path = tmp_path / 'r1' / 'tool_effects.jsonl'
        path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')

        unresolved = await store.list_unresolved_tool_effects(run_id='r1')
        assert [r.tool_call_id for r in unresolved] == ['c1']

    async def test_list_unresolved_tool_effects_empty_when_file_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        assert await store.list_unresolved_tool_effects(run_id='nonexistent') == []

    async def test_get_tool_effect_returns_latest_for_run(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='runA', status='started')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c1', tool_name='add', run_id='runA', status='completed')
        )
        await store.record_tool_effect(
            ToolEffectRecord(tool_call_id='c2', tool_name='mul', run_id='runB', status='started')
        )
        # Blank line to exercise the strip branch on read.
        (tmp_path / 'runA' / 'tool_effects.jsonl').write_text(
            (tmp_path / 'runA' / 'tool_effects.jsonl').read_text(encoding='utf-8') + '\n',
            encoding='utf-8',
        )

        record = await store.get_tool_effect(run_id='runA', tool_call_id='c1')
        assert record is not None
        assert record.status == 'completed'
        assert record.run_id == 'runA'

        other = await store.get_tool_effect(run_id='runB', tool_call_id='c2')
        assert other is not None and other.status == 'started'

        assert await store.get_tool_effect(run_id='runA', tool_call_id='missing') is None

    async def test_get_tool_effect_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        assert await store.get_tool_effect(run_id='absent', tool_call_id='anything') is None

    async def test_register_run_rejects_bad_run_id(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        with pytest.raises(ValueError, match='invalid run_id'):
            await store.register_run(RunRecord(run_id='../evil'))

    async def test_event_deserialization_rejects_unknown_kind(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'events.jsonl').write_text(
            json.dumps(
                {
                    'run_id': 'r1',
                    'kind': 'made_up',
                    'step_index': 0,
                    'timestamp': '2024-01-01T00:00:00+00:00',
                }
            )
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='unknown event kind'):
            await store.list_events(run_id='r1')

    async def test_event_deserialization_rejects_wrong_types(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'events.jsonl').write_text(
            json.dumps(
                {
                    'run_id': 1,  # wrong type
                    'kind': 'run_started',
                    'step_index': 0,
                    'timestamp': '2024-01-01T00:00:00+00:00',
                }
            )
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='event payload has wrong types'):
            await store.list_events(run_id='r1')

    async def test_event_deserialization_rejects_non_string_optional(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'events.jsonl').write_text(
            json.dumps(
                {
                    'run_id': 'r1',
                    'kind': 'run_started',
                    'step_index': 0,
                    'timestamp': '2024-01-01T00:00:00+00:00',
                    'agent_name': 5,  # neither None nor str
                }
            )
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='expected str'):
            await store.list_events(run_id='r1')

    async def test_run_deserialization_rejects_wrong_types(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'run.json').write_text(json.dumps({'run_id': 1, 'started_at': 'x'}), encoding='utf-8')
        with pytest.raises(ValueError, match='run record has wrong types'):
            await store.get_run(run_id='r1')

    async def test_tool_effect_deserialization_rejects_wrong_types(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'tool_effects.jsonl').write_text(
            json.dumps({'tool_call_id': 1, 'tool_name': 'add', 'run_id': 'r1', 'status': 'started', 'started_at': 'x'})
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='tool effect record has wrong types'):
            await store.list_unresolved_tool_effects(run_id='r1')

    async def test_tool_effect_deserialization_rejects_unknown_status(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'tool_effects.jsonl').write_text(
            json.dumps(
                {
                    'tool_call_id': 'c1',
                    'tool_name': 'add',
                    'run_id': 'r1',
                    'status': 'pending',
                    'started_at': '2024-01-01T00:00:00+00:00',
                }
            )
            + '\n',
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='unknown tool effect status'):
            await store.list_unresolved_tool_effects(run_id='r1')

    async def test_snapshot_deserialization_rejects_wrong_types(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path)
        snap_dir = tmp_path / 'r1' / 'snapshots'
        snap_dir.mkdir(parents=True)
        (snap_dir / '0.json').write_text(
            json.dumps({'step_index': 'wrong', 'timestamp': 'x', 'messages': []}),
            encoding='utf-8',
        )
        with pytest.raises(ValueError, match='snapshot has wrong types'):
            await store.latest_snapshot(run_id='r1')


# ---------------------------------------------------------------------------
# Capability behavior via Agent + TestModel
# ---------------------------------------------------------------------------


class TestStepPersistenceCapability:
    async def test_basic_run_records_lifecycle_and_snapshot(self) -> None:
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, agent_name='librarian')])

        result = await agent.run('add 1 and 2')

        rid = await first_run_id(store)
        events = await store.list_events(run_id=rid)
        kinds = [e.kind for e in events]
        assert kinds[0] == 'run_started'
        assert kinds[-1] == 'run_completed'
        assert 'tool_call_started' in kinds
        assert 'tool_call_completed' in kinds

        # RunRecord lineage was registered.
        record = await store.get_run(run_id=rid)
        assert record is not None
        assert record.agent_name == 'librarian'
        assert record.parent_run_id is None

        # Snapshot is provider-valid and round-trips through `continue_run`.
        snap = await store.latest_snapshot(run_id=rid)
        assert snap is not None
        assert is_provider_valid(snap.messages) is True
        assert len(snap.messages) == len(result.all_messages())

        # All events tagged with the same run_id and agent_name.
        assert {e.run_id for e in events} == {rid}
        assert {e.agent_name for e in events} == {'librarian'}

    async def test_helper_based_continuation_replays_prior_messages(self) -> None:
        """`continue_run(store, run_id=...) -> Agent.run(message_history=...)`."""
        store = InMemoryStepStore()
        agent1 = make_simple_agent([StepPersistence(store=store, agent_name='a')])
        result1 = await agent1.run('add 1 and 2')
        first_rid = await first_run_id(store)

        history = await continue_run(store, run_id=first_rid)
        assert len(history) == len(result1.all_messages())

        # Second run uses a fresh capability instance + the helper-provided history.
        agent2 = make_simple_agent([StepPersistence(store=store, agent_name='b')])
        result2 = await agent2.run('add 3 and 4', message_history=history)

        msgs = result2.all_messages()
        assert len(msgs) > len(history)
        # The prior messages appear at the head of the second run's history.
        for prior_msg, replayed in zip(history, msgs[: len(history)]):
            assert type(prior_msg) is type(replayed)

    async def test_agent_name_derived_run_id_prefix(self) -> None:
        """No explicit run_id + agent_name -> `{agent_name}-{8-hex}`."""
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, agent_name='librarian')])
        await agent.run('add 1 and 2')

        runs = await store.list_runs()
        assert len(runs) == 1
        assert runs[0].run_id.startswith('librarian-')
        # 'librarian-' + 8 hex chars.
        assert len(runs[0].run_id) == len('librarian-') + 8

    async def test_single_capability_instance_reused_gets_fresh_ids(self) -> None:
        """One `StepPersistence(agent_name=...)` reused for two runs -> two distinct ids."""
        store = InMemoryStepStore()
        cap: StepPersistence[object] = StepPersistence(store=store, agent_name='librarian')

        agent1: Agent[object, str] = Agent(TestModel(), capabilities=[cap])

        @agent1.tool_plain
        def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return a + b

        await agent1.run('add 1 and 2')
        await agent1.run('add 3 and 4')

        runs = await store.list_runs()
        assert len(runs) == 2
        rids = {r.run_id for r in runs}
        assert len(rids) == 2
        for rid in rids:
            assert rid.startswith('librarian-')

    async def test_parent_run_id_inferred_via_contextvar(self) -> None:
        """Orchestrator tool calls a delegate `Agent.run` -> delegate's `parent_run_id`
        is auto-set to the orchestrator's `run_id` without manual threading."""
        store = InMemoryStepStore()

        delegate: Agent[object, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='delegate')],
        )

        @delegate.tool_plain
        def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return a + b

        orchestrator: Agent[object, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='orchestrator')],
        )

        @orchestrator.tool_plain
        async def delegate_work() -> str:  # pyright: ignore[reportUnusedFunction]
            res = await delegate.run('add 1 and 2')
            return res.output

        await orchestrator.run('coordinate')

        runs = await store.list_runs()
        orch = next(r for r in runs if r.agent_name == 'orchestrator')
        dele = next(r for r in runs if r.agent_name == 'delegate')

        assert dele.parent_run_id == orch.run_id
        # And the delegate's events also carry that parent_run_id.
        dele_events = await store.list_events(run_id=dele.run_id)
        assert {e.parent_run_id for e in dele_events} == {orch.run_id}

    async def test_conversation_id_groups_two_runs(self) -> None:
        """Passing the same `conversation_id` to two `Agent.run` calls -> store.list_runs
        finds both."""
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, agent_name='c')])

        await agent.run('add 1 and 2', conversation_id='conv-1')
        await agent.run('add 3 and 4', conversation_id='conv-1')

        runs = await store.list_runs(conversation_id='conv-1')
        assert len(runs) == 2
        assert {r.conversation_id for r in runs} == {'conv-1'}

    async def test_list_runs_parent_and_conversation_filters_combine(self) -> None:
        store = InMemoryStepStore()

        delegate: Agent[object, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='delegate')],
        )

        @delegate.tool_plain
        def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return a + b

        orchestrator: Agent[object, str] = Agent(
            TestModel(),
            capabilities=[StepPersistence(store=store, agent_name='orchestrator')],
        )

        @orchestrator.tool_plain
        async def delegate_work() -> str:  # pyright: ignore[reportUnusedFunction]
            res = await delegate.run('add 1 and 2', conversation_id='target-conv')
            return res.output

        await orchestrator.run('coordinate', conversation_id='target-conv')
        orch_rid = next(r.run_id for r in await store.list_runs() if r.agent_name == 'orchestrator')

        # Sibling unrelated run under the same conversation but no parent.
        unrelated = make_simple_agent([StepPersistence(store=store, agent_name='unrelated')])
        await unrelated.run('add 5 and 6', conversation_id='target-conv')

        narrowed = await store.list_runs(parent_run_id=orch_rid, conversation_id='target-conv')
        assert [r.agent_name for r in narrowed] == ['delegate']

    async def test_step_event_carries_conversation_id(self) -> None:
        """`ctx.conversation_id` propagates onto each emitted `StepEvent`."""
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, agent_name='c')])
        await agent.run('add 1 and 2', conversation_id='conv-evt')

        rid = await first_run_id(store)
        events = await store.list_events(run_id=rid)
        assert events  # sanity
        assert {e.conversation_id for e in events} == {'conv-evt'}

    async def test_tool_effect_is_scoped_per_run_in_memory(self) -> None:
        """Same `tool_call_id` across two runs must NOT share the effect record."""
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, agent_name='a')])

        await agent.run('add 1 and 2')
        await agent.run('add 3 and 4')

        runs = await store.list_runs()
        assert len(runs) == 2
        run1_id, run2_id = runs[0].run_id, runs[1].run_id

        e1 = await store.get_tool_effect(run_id=run1_id, tool_call_id='pyd_ai_tool_call_id__add')
        e2 = await store.get_tool_effect(run_id=run2_id, tool_call_id='pyd_ai_tool_call_id__add')
        assert e1 is not None and e2 is not None
        assert e1.run_id == run1_id
        assert e2.run_id == run2_id
        # The second run's record is independent: its started_at is not inherited from run 1.
        assert e2.started_at >= e1.started_at
        assert e2.started_at != e1.started_at or e2 is not e1

        # Cross-lookups return only the owning run's record.
        cross = await store.get_tool_effect(run_id=run1_id, tool_call_id='pyd_ai_tool_call_id__add')
        assert cross is not None
        assert cross.run_id == run1_id

    async def test_tool_effect_is_scoped_per_run_file_store(self, tmp_path: Path) -> None:
        """Same correctness contract under `FileStepStore`."""
        store = FileStepStore(tmp_path)
        agent = make_simple_agent([StepPersistence(store=store, agent_name='a')])

        await agent.run('add 1 and 2')
        await agent.run('add 3 and 4')

        runs = await store.list_runs()
        assert len(runs) == 2
        run1_id, run2_id = runs[0].run_id, runs[1].run_id

        e1 = await store.get_tool_effect(run_id=run1_id, tool_call_id='pyd_ai_tool_call_id__add')
        e2 = await store.get_tool_effect(run_id=run2_id, tool_call_id='pyd_ai_tool_call_id__add')
        assert e1 is not None and e2 is not None
        assert e1.run_id == run1_id
        assert e2.run_id == run2_id
        # The other run's directory does not leak into this run's lookup.
        assert await store.get_tool_effect(run_id=run1_id, tool_call_id='missing') is None

    async def test_tool_failure_records_failed_status_and_event(self) -> None:
        store = InMemoryStepStore()
        agent: Agent[object, str] = Agent(TestModel(), capabilities=[StepPersistence(store=store)])

        @agent.tool_plain
        def boom() -> int:  # pyright: ignore[reportUnusedFunction]
            raise ValueError('kaboom')

        with pytest.raises(ValueError, match='kaboom'):
            await agent.run('boom please')

        rid = await first_run_id(store)
        events = await store.list_events(run_id=rid)
        kinds = [e.kind for e in events]
        assert 'tool_call_started' in kinds
        assert 'tool_call_failed' in kinds
        assert 'run_failed' in kinds
        # The failure event records the exception repr.
        failed_event = next(e for e in events if e.kind == 'tool_call_failed')
        assert failed_event.error is not None and 'kaboom' in failed_event.error

        effect = await store.get_tool_effect(run_id=rid, tool_call_id='pyd_ai_tool_call_id__boom')
        assert effect is not None
        assert effect.status == 'failed'
        assert effect.effect_summary is not None and 'kaboom' in effect.effect_summary

    async def test_explicit_run_id_wins_over_ctx_run_id(self) -> None:
        store = InMemoryStepStore()
        agent = make_simple_agent(
            [StepPersistence(store=store, run_id='librarian-001', agent_name='librarian')],
        )

        await agent.run('add 1 and 2')

        # Caller-supplied run_id is the persisted identity, not the auto-generated ctx.run_id.
        record = await store.get_run(run_id='librarian-001')
        assert record is not None
        assert record.agent_name == 'librarian'
        events = await store.list_events(run_id='librarian-001')
        assert {e.run_id for e in events} == {'librarian-001'}

    async def test_explicit_parent_run_id_overrides_contextvar(self) -> None:
        """Manual `parent_run_id=` wins over the auto-inferred contextvar value."""
        store = InMemoryStepStore()
        agent = make_simple_agent(
            [StepPersistence(store=store, agent_name='delegate', parent_run_id='manual-parent')],
        )

        await agent.run('add 1 and 2')

        runs = await store.list_runs(parent_run_id='manual-parent')
        assert len(runs) == 1
        assert runs[0].agent_name == 'delegate'

    async def test_from_spec_memory_backend(self) -> None:
        cap = StepPersistence.from_spec()
        assert isinstance(cap.store, InMemoryStepStore)

    async def test_from_spec_explicit_memory_backend_with_kwargs(self) -> None:
        cap = StepPersistence.from_spec(backend='memory', agent_name='a')
        assert isinstance(cap.store, InMemoryStepStore)
        assert cap.agent_name == 'a'

    async def test_from_spec_file_backend(self, tmp_path: Path) -> None:
        cap = StepPersistence.from_spec(backend='file', directory=tmp_path)
        assert isinstance(cap.store, FileStepStore)

    async def test_from_spec_file_backend_default_directory(self) -> None:
        cap = StepPersistence.from_spec(backend='file')
        assert isinstance(cap.store, FileStepStore)


# ---------------------------------------------------------------------------
# Headline acceptance test
# ---------------------------------------------------------------------------


class TestCrashMidToolCallContract:
    """The signature acceptance test from the PR comment.

    A run killed after a tool starts but before its return is persisted must
    leave a visible event trail without exposing the killed point as a
    valid `message_history` continuation. The latest snapshot must be older
    than the in-flight call, and that snapshot must be provider-valid.
    """

    async def test_visible_trail_no_false_continuation_point(self) -> None:
        store = InMemoryStepStore()
        cap: StepPersistence[object] = StepPersistence(store=store, agent_name='delegate')
        agent: Agent[object, str] = Agent(TestModel(), capabilities=[cap])

        @agent.tool_plain
        def add(a: int, b: int) -> int:  # pyright: ignore[reportUnusedFunction]
            return a + b

        # 1) Drive a full successful run so a provider-valid snapshot exists.
        result = await agent.run('add 1 and 2')
        rid = await first_run_id(store)
        snap_before_crash = await store.latest_snapshot(run_id=rid)
        assert snap_before_crash is not None
        assert is_provider_valid(snap_before_crash.messages) is True
        snap_step = snap_before_crash.step_index

        # 2) Simulate a crash mid-tool-call by calling `before_tool_execute`
        # directly with a synthesised ToolCallPart and never firing
        # `after_tool_execute` / `on_tool_execute_error`. Use the resolved
        # `cap.run_id` if set; otherwise rely on the discovered rid via ctx.
        crash_ctx = build_run_context(deps=None, run_id=rid, run_step=snap_step + 1)
        crash_call = ToolCallPart(tool_name='add', args={'a': 9, 'b': 9}, tool_call_id='crash-call-1')
        tool_def = ToolDefinition(name='add', description='Add two numbers.')
        await cap.before_tool_execute(crash_ctx, call=crash_call, tool_def=tool_def, args={'a': 9, 'b': 9})

        # 3) Assert the event log shows the started call with no terminal update.
        events = await store.list_events(run_id=rid)
        started = [e for e in events if e.kind == 'tool_call_started' and e.tool_call_id == 'crash-call-1']
        completed = [e for e in events if e.kind == 'tool_call_completed' and e.tool_call_id == 'crash-call-1']
        failed = [e for e in events if e.kind == 'tool_call_failed' and e.tool_call_id == 'crash-call-1']
        assert len(started) == 1
        assert completed == []
        assert failed == []

        # 4) The unresolved-effect ledger surfaces the in-flight tool call.
        unresolved = await store.list_unresolved_tool_effects(run_id=rid)
        crash_records = [r for r in unresolved if r.tool_call_id == 'crash-call-1']
        assert len(crash_records) == 1
        assert crash_records[0].status == 'started'

        # 5) Resume point is the snapshot from step 1 — older than the crash —
        # and is still provider-valid.
        snap_after_crash = await store.latest_snapshot(run_id=rid)
        assert snap_after_crash is not None
        assert snap_after_crash.step_index == snap_step
        assert is_provider_valid(snap_after_crash.messages) is True
        # And the snapshot is consistent with what the prior successful run produced.
        assert len(snap_after_crash.messages) == len(result.all_messages())


class TestOnRunErrorSnapshot:
    """`on_run_error` rescues a provider-valid resume point no snapshot captured (#253).

    A provider-valid history can exist at a boundary that no snapshot records:
    the model request after a clean tool cycle (the resolved tool return only
    enters the history as that request is built) or the `CallToolsNode` after a
    text response (output validation raising there skips its `after_node_run`).
    `on_run_error` persists the last such history, and skips when the crash left
    a dangling tool call -- so `latest_snapshot` never regresses to an
    unsendable point.
    """

    async def test_rescues_provider_valid_history_when_next_model_request_raises(self) -> None:
        """Issue #253 acceptance test: request 1 tool call, request 2 raises.

        The resolved tool cycle `[prompt, response, tool-return]` is
        provider-valid with no open tool calls, but it is never a completed
        node boundary, so without this behavior the run leaves no snapshot.
        `on_run_error` rescues it from the `before_model_request` boundary.
        """
        store = InMemoryStepStore()

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prior_responses = [m for m in messages if isinstance(m, ModelResponse)]
            if not prior_responses:
                return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='lookup-1')])
            raise RuntimeError('provider down on request 2')

        agent: Agent[object, str] = Agent(
            FunctionModel(model),
            capabilities=[StepPersistence(store=store, agent_name='delegate')],
        )

        @agent.tool_plain
        def lookup() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'ok'

        with pytest.raises(RuntimeError, match='provider down on request 2'):
            await agent.run('go')

        rid = await first_run_id(store)
        assert 'run_failed' in [e.kind for e in await store.list_events(run_id=rid)]

        snap = await store.latest_snapshot(run_id=rid)
        assert snap is not None
        assert is_provider_valid(snap.messages) is True
        # The resolved tool cycle: the tool call has its return, no open calls.
        assert any(
            isinstance(part, ToolReturnPart) and part.tool_call_id == 'lookup-1'
            for msg in snap.messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

    async def test_rescues_text_response_when_output_validation_raises(self) -> None:
        """Text response then a hard output-validation error: the clean tail is rescued.

        The terminal `CallToolsNode` raises inside output validation before its
        `after_node_run` fires; `on_run_error` captures the provider-valid
        `[prompt, text-response]` history from the `after_node_run` boundary.
        """
        store = InMemoryStepStore()
        agent: Agent[object, str] = Agent(
            TestModel(custom_output_text='final answer'),
            capabilities=[StepPersistence(store=store, agent_name='delegate')],
        )

        @agent.output_validator
        def reject(value: str) -> str:  # pyright: ignore[reportUnusedFunction]
            raise RuntimeError('validator down')

        with pytest.raises(RuntimeError, match='validator down'):
            await agent.run('answer please')

        rid = await first_run_id(store)
        assert 'run_failed' in [e.kind for e in await store.list_events(run_id=rid)]

        snap = await store.latest_snapshot(run_id=rid)
        assert snap is not None
        assert is_provider_valid(snap.messages) is True
        assert isinstance(snap.messages[-1], ModelResponse)
        assert any(isinstance(part, TextPart) for part in snap.messages[-1].parts)

    async def test_skips_snapshot_when_crash_leaves_dangling_tool_call(self) -> None:
        """A crash mid-tool-call leaves a dangling call that is never a resume point."""
        store = InMemoryStepStore()

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('boom', {}, tool_call_id='boom-1')])

        agent: Agent[object, str] = Agent(
            FunctionModel(model),
            capabilities=[StepPersistence(store=store, agent_name='delegate')],
        )

        @agent.tool_plain
        def boom() -> str:  # pyright: ignore[reportUnusedFunction]
            raise ValueError('kaboom')

        with pytest.raises(ValueError, match='kaboom'):
            await agent.run('go')

        rid = await first_run_id(store)
        assert 'run_failed' in [e.kind for e in await store.list_events(run_id=rid)]
        # The only history reached is the bare prompt (skipped) and the dangling
        # `boom` call (provider-invalid, skipped): no false continuation point.
        assert await store.latest_snapshot(run_id=rid) is None

    async def test_does_not_regress_to_stale_stash_after_model_request_save(self) -> None:
        """A stale completed-node stash must not supersede a newer resolved-cycle save.

        req1 text -> output-validation `ModelRetry` (run continues; the text
        history is stashed into `latest_node_history`). req2 tool call resolves
        cleanly. req3 raises: `on_model_request_error` saves the newer resolved
        tool cycle, and `on_run_error` must not re-save the older text stash
        over it (both save sites write to the same store, and `latest_snapshot`
        returns the last write).
        """
        store = InMemoryStepStore()

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            n_resp = len([m for m in messages if isinstance(m, ModelResponse)])
            if n_resp == 0:
                return ModelResponse(parts=[TextPart('draft')])
            if n_resp == 1:
                return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='lookup-1')])
            raise RuntimeError('provider down on request 3')

        agent: Agent[object, str] = Agent(
            FunctionModel(model),
            capabilities=[StepPersistence(store=store, agent_name='delegate')],
            output_type=str,
        )

        @agent.output_validator
        def gate(value: str) -> str:  # pyright: ignore[reportUnusedFunction]
            raise ModelRetry('call a tool first')

        @agent.tool_plain
        def lookup() -> str:  # pyright: ignore[reportUnusedFunction]
            return 'ok'

        with pytest.raises(RuntimeError, match='provider down on request 3'):
            await agent.run('go')

        rid = await first_run_id(store)
        snap = await store.latest_snapshot(run_id=rid)
        assert snap is not None
        assert is_provider_valid(snap.messages) is True
        # The newest safe resume point is the resolved tool cycle, not the older text history.
        assert any(
            isinstance(part, ToolReturnPart) and part.tool_call_id == 'lookup-1'
            for msg in snap.messages
            if isinstance(msg, ModelRequest)
            for part in msg.parts
        )

    async def test_rescues_newest_text_history_over_earlier_saved_snapshot(self) -> None:
        """`on_run_error` still saves its stash when it is newer than the store's latest snapshot.

        Two text responses each pass an output-validation `ModelRetry`, then the
        second validator raises hard. The first cycle already saved a snapshot;
        the `on_run_error` stash (the newer text history) supersedes it.
        """
        store = InMemoryStepStore()

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            n_resp = len([m for m in messages if isinstance(m, ModelResponse)])
            return ModelResponse(parts=[TextPart('v1' if n_resp == 0 else 'v2')])

        agent: Agent[object, str] = Agent(
            FunctionModel(model),
            capabilities=[StepPersistence(store=store, agent_name='delegate')],
            output_type=str,
        )
        calls = {'n': 0}

        @agent.output_validator
        def gate(value: str) -> str:  # pyright: ignore[reportUnusedFunction]
            calls['n'] += 1
            if calls['n'] == 1:
                raise ModelRetry('retry once')
            raise RuntimeError('validator down')

        with pytest.raises(RuntimeError, match='validator down'):
            await agent.run('go')

        rid = await first_run_id(store)
        snap = await store.latest_snapshot(run_id=rid)
        assert snap is not None
        assert is_provider_valid(snap.messages) is True
        # The latest resume point is the newer history including `v2`, not the earlier `v1`-only snapshot.
        assert any(
            isinstance(part, TextPart) and part.content == 'v2'
            for msg in snap.messages
            if isinstance(msg, ModelResponse)
            for part in msg.parts
        )

    async def test_on_run_error_without_stashed_history_saves_nothing(self) -> None:
        """Direct-hook: no provider-valid boundary reached -> no snapshot, `run_failed` still emitted.

        Covers the `stashed is None` branch -- `on_run_error` invoked before any
        boundary refreshed `latest_node_history`.
        """
        store = InMemoryStepStore()
        cap: StepPersistence[object] = StepPersistence(store=store)
        ctx = build_run_context(run_id='r1', run_step=1)

        with pytest.raises(RuntimeError, match='boom'):
            await cap.on_run_error(ctx, error=RuntimeError('boom'))

        assert await store.latest_snapshot(run_id='r1') is None
        assert [e.kind for e in await store.list_events(run_id='r1')] == ['run_failed']


# ---------------------------------------------------------------------------
# Hook-level branches awkward to reach through Agent
# ---------------------------------------------------------------------------


class TestCapabilityHookBranches:
    async def test_effective_run_id_falls_back_to_capability_field(self) -> None:
        """When `ctx.run_id` is missing, the capability uses its own `run_id`."""
        store = InMemoryStepStore()
        cap: StepPersistence[object] = StepPersistence(store=store, run_id='configured', agent_name='a')
        ctx_no_run_id = build_run_context(deps=None, run_id=None)
        await cap.before_run(ctx_no_run_id)

        record = await store.get_run(run_id='configured')
        assert record is not None
        events = await store.list_events(run_id='configured')
        assert [e.kind for e in events] == ['run_started']

    async def test_after_run_skips_snapshot_when_history_not_provider_valid(self) -> None:
        """`after_run` only persists a snapshot when the history is provider-valid."""
        store = InMemoryStepStore()
        cap: StepPersistence[object] = StepPersistence(store=store)
        ctx = build_run_context(deps=None, run_id='r1')

        unmatched: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[ToolCallPart(tool_name='add', args={}, tool_call_id='orphan')]),
        ]
        result: AgentRunResult[str] = AgentRunResult(
            output='out',
            _state=GraphAgentState(message_history=unmatched, run_id='r1'),
        )

        await cap.after_run(ctx, result=result)

        assert await store.latest_snapshot(run_id='r1') is None
        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['run_completed']

    async def test_after_run_saves_fallback_snapshot_when_no_node_snapshot(self) -> None:
        """With no `CallToolsNode` snapshot taken, `after_run` saves the final valid history."""
        store = InMemoryStepStore()
        cap: StepPersistence[object] = StepPersistence(store=store)
        ctx = build_run_context(deps=None, run_id='r1', run_step=3)

        valid: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='done')]),
        ]
        result: AgentRunResult[str] = AgentRunResult(
            output='out',
            _state=GraphAgentState(message_history=valid, run_id='r1'),
        )

        await cap.after_run(ctx, result=result)

        snap = await store.latest_snapshot(run_id='r1')
        assert snap is not None
        assert snap.step_index == 3
        assert len(snap.messages) == 2

    async def test_on_model_request_error_records_event_and_reraises(self) -> None:
        store = InMemoryStepStore()
        cap: StepPersistence[object] = StepPersistence(store=store)
        ctx = build_run_context(deps=None, run_id='r1')
        request_context = ModelRequestContext(
            model=ctx.model,
            messages=[],
            model_settings=None,
            model_request_parameters=ModelRequestParameters(),
        )
        boom = RuntimeError('nope')

        with pytest.raises(RuntimeError, match='nope'):
            await cap.on_model_request_error(ctx, request_context=request_context, error=boom)

        events = await store.list_events(run_id='r1')
        assert [e.kind for e in events] == ['model_request_failed']
        assert events[0].error is not None and 'nope' in events[0].error

    async def test_for_run_returns_self_when_resolution_is_no_op(self) -> None:
        """When `run_id` is explicit and no contextvar is set, `for_run` returns `self`."""
        store = InMemoryStepStore()
        cap: StepPersistence[object] = StepPersistence(store=store, run_id='fixed')
        ctx = build_run_context(deps=None, run_id='ignored')

        result = await cap.for_run(ctx)
        assert result is cap

    async def test_run_record_load_with_missing_metadata(self, tmp_path: Path) -> None:
        """`_str_str_dict(None)` returns `{}` when metadata is absent in storage."""
        store = FileStepStore(tmp_path)
        run_dir = tmp_path / 'r1'
        run_dir.mkdir()
        (run_dir / 'run.json').write_text(
            json.dumps({'run_id': 'r1', 'started_at': '2024-01-01T00:00:00+00:00'}),
            encoding='utf-8',
        )

        record = await store.get_run(run_id='r1')
        assert record is not None
        assert record.metadata == {}


# ---------------------------------------------------------------------------
# Round-2 review fixes (RetryPromptPart, list_runs ordering, effect metadata,
# from_spec validation, annotate_tool_effect)
# ---------------------------------------------------------------------------


class TestNonToolRetryPrompt:
    def test_non_tool_retry_prompt_is_valid(self) -> None:
        """`RetryPromptPart(tool_name=None)` is an output-validation retry, not a tool result."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content='hi')]),
            ModelResponse(parts=[TextPart(content='wrong shape')]),
            ModelRequest(parts=[RetryPromptPart(content='try again', tool_name=None)]),
        ]
        assert is_provider_valid(messages) is True

    def test_tool_retry_prompt_still_needs_open_call(self) -> None:
        """`RetryPromptPart(tool_name=...)` still requires a matching open `ToolCallPart`."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[RetryPromptPart(content='retry', tool_name='add', tool_call_id='ghost')]),
        ]
        assert is_provider_valid(messages) is False


class TestListRunsChronologicalOrdering:
    async def test_in_memory_returns_started_at_order(self) -> None:
        """`InMemoryStepStore.list_runs` sorts by `started_at`, not insertion order."""
        from datetime import datetime, timezone

        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='z-newer', started_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc)))
        await store.register_run(RunRecord(run_id='a-older', started_at=datetime(2026, 5, 24, 10, tzinfo=timezone.utc)))
        runs = await store.list_runs()
        assert [r.run_id for r in runs] == ['a-older', 'z-newer']

    async def test_file_store_returns_started_at_order(self, tmp_path: Path) -> None:
        """`FileStepStore.list_runs` sorts by `started_at`, not by directory name."""
        from datetime import datetime, timezone

        store = FileStepStore(tmp_path)
        # `a-new` is lexicographically first but chronologically last.
        await store.register_run(RunRecord(run_id='z-old', started_at=datetime(2026, 5, 24, 10, tzinfo=timezone.utc)))
        await store.register_run(RunRecord(run_id='a-new', started_at=datetime(2026, 5, 24, 12, tzinfo=timezone.utc)))
        runs = await store.list_runs()
        assert [r.run_id for r in runs] == ['z-old', 'a-new']


class TestToolEffectMetadataPreservation:
    async def test_completed_preserves_idempotency_key_and_effect_summary(self) -> None:
        """Metadata written during the tool call survives the terminal `completed` record."""
        from pydantic_ai_harness.step_persistence import annotate_tool_effect

        store = InMemoryStepStore()
        agent: Agent[object, str] = Agent(TestModel(), capabilities=[StepPersistence(store=store, run_id='r1')])

        @agent.tool
        async def write_label(ctx: RunContext[object], label: str) -> str:  # pyright: ignore[reportUnusedFunction]
            await annotate_tool_effect(
                store,
                ctx,
                idempotency_key=f'label::{label}',
                effect_summary=f'set label to {label}',
            )
            return f'wrote {label}'

        await agent.run('apply a label please')

        effect = await store.get_tool_effect(run_id='r1', tool_call_id='pyd_ai_tool_call_id__write_label')
        assert effect is not None
        assert effect.status == 'completed'
        assert effect.idempotency_key is not None and effect.idempotency_key.startswith('label::')
        assert effect.effect_summary is not None and 'set label to' in effect.effect_summary

    async def test_failed_preserves_idempotency_key(self) -> None:
        """Metadata written before a tool raises still appears on the `failed` record."""
        from pydantic_ai_harness.step_persistence import annotate_tool_effect

        store = InMemoryStepStore()
        agent: Agent[object, str] = Agent(TestModel(), capabilities=[StepPersistence(store=store, run_id='r1')])

        @agent.tool
        async def boom(ctx: RunContext[object]) -> int:  # pyright: ignore[reportUnusedFunction]
            await annotate_tool_effect(store, ctx, idempotency_key='boom-key')
            raise ValueError('kaboom')

        with pytest.raises(ValueError, match='kaboom'):
            await agent.run('please boom')

        effect = await store.get_tool_effect(run_id='r1', tool_call_id='pyd_ai_tool_call_id__boom')
        assert effect is not None
        assert effect.status == 'failed'
        assert effect.idempotency_key == 'boom-key'
        # default summary still records the error when no summary was annotated:
        assert effect.effect_summary is not None and 'kaboom' in effect.effect_summary

    async def test_annotate_tool_effect_outside_step_persistence_is_a_noop(self) -> None:
        """No `current_run_id` → `annotate_tool_effect` returns without writing."""
        from pydantic_ai_harness.step_persistence import annotate_tool_effect

        store = InMemoryStepStore()
        ctx = build_run_context(deps=None, run_id='r1')
        # No StepPersistence active and ctx.tool_call_id is None.
        await annotate_tool_effect(store, ctx, idempotency_key='ignored')
        assert await store.get_tool_effect(run_id='r1', tool_call_id='whatever') is None

    async def test_annotate_tool_effect_noop_when_prior_record_missing(self) -> None:
        """`current_run_id` set + ctx tool fields set, but no prior record → no-op."""
        from pydantic_ai_harness.step_persistence import annotate_tool_effect
        from pydantic_ai_harness.step_persistence._context import current_run_id

        store = InMemoryStepStore()
        ctx = RunContext[Any](
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            prompt=None,
            messages=[],
            run_step=0,
            run_id='r1',
            tool_call_id='tc-1',
            tool_name='write_label',
        )
        token = current_run_id.set('r1')
        try:
            await annotate_tool_effect(store, ctx, idempotency_key='label::x')
        finally:
            current_run_id.reset(token)
        # `before_tool_execute` hasn't fired, so the prior record doesn't exist.
        # The helper returns without inventing one.
        assert await store.get_tool_effect(run_id='r1', tool_call_id='tc-1') is None


class TestFromSpecBackendValidation:
    def test_unknown_backend_raises(self) -> None:
        """A typo like `backend='disk'` raises instead of silently using memory."""
        with pytest.raises(ValueError, match='unknown backend'):
            StepPersistence.from_spec(backend='disk')

    def test_memory_backend_still_works(self) -> None:
        cap: StepPersistence[Any] = StepPersistence.from_spec(backend='memory')
        assert isinstance(cap.store, InMemoryStepStore)

    def test_sqlite_backend(self, tmp_path: Path) -> None:
        cap: StepPersistence[Any] = StepPersistence.from_spec(backend='sqlite', database=str(tmp_path / 'runs.db'))
        assert isinstance(cap.store, SqliteStepStore)


# ---------------------------------------------------------------------------
# Identity model contract: run_id is per-call; conversation_id groups turns
# ---------------------------------------------------------------------------


class TestRunIdIsPerCall:
    """`run_id` is one `Agent.run` invocation; `conversation_id` groups turns."""

    async def test_multi_turn_orchestrator_uses_conversation_id(self) -> None:
        """Recommended multi-turn pattern: shared `conversation_id`, distinct `run_id` per call."""
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, agent_name='orchestrator')])

        for prompt in ('first turn', 'second turn', 'third turn'):
            await agent.run(prompt, conversation_id='orch-conv')

        records = await store.list_runs(conversation_id='orch-conv')
        assert len(records) == 3
        # All distinct ids, all carrying the same conversation_id.
        assert len({r.run_id for r in records}) == 3
        assert all(r.conversation_id == 'orch-conv' for r in records)
        assert all(r.run_id.startswith('orchestrator-') for r in records)

    async def test_explicit_run_id_reuse_raises(self) -> None:
        """Reusing an explicit `run_id` across `.run()` calls raises ValueError.

        The tool-effect ledger keys on `(run_id, tool_call_id)`, so a
        silent second run would collide with the first. `before_run`
        rejects the second run explicitly, pointing the caller to
        `conversation_id` for multi-turn grouping.
        """
        store = InMemoryStepStore()
        agent = make_simple_agent([StepPersistence(store=store, run_id='shared')])

        await agent.run('first')

        with pytest.raises(ValueError, match=r"run_id 'shared' is already in the store"):
            await agent.run('second')

        # First run's records remain untouched.
        record = await store.get_run(run_id='shared')
        assert record is not None
        effect = await store.get_tool_effect(run_id='shared', tool_call_id='pyd_ai_tool_call_id__add')
        assert effect is not None
        assert effect.status == 'completed'

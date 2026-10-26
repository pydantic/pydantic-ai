"""Tests for the `ConversationSearch` capability.

End-to-end recall (persist with `StepPersistence`, compact, search) is driven
through `Agent(..., capabilities=[...])`. The search tool's ranking, rendering,
and source edge cases are exercised through the public
`ConversationSearchToolset` and `SnapshotHistorySource` directly, over seeded
`InMemoryStepStore` instances.
"""

from __future__ import annotations

import re
import warnings
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from pydantic_core import ErrorDetails

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import (
    AgentInstructionSource,
    BinaryContent,
    InstructionBaselinePart,
    InstructionDeltaPart,
    InstructionId,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    RetryInfo,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.compaction import SlidingWindowCompaction, SummarizingCompaction
from pydantic_ai_harness.compaction._summarizing_compaction import SUMMARY_PREFIX
from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    ConversationSearchToolset,
    HistorySource,
    SearchScope,
    SnapshotHistorySource,
    SnapshotStore,
)
from pydantic_ai_harness.conversation_search._source import message_hash
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    InMemoryStepStore,
    PostgresConnection,
    PostgresStepStore,
    RunRecord,
    SqliteStepStore,
    StepPersistence,
)
from tests.conftest import legacy_retry_prompt_part


def _run_context(conversation_id: str | None = None) -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        conversation_id=conversation_id,
    )


def _user(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _reply(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


async def _seed_run(
    store: InMemoryStepStore,
    run_id: str,
    *snapshots: list[ModelMessage],
    conversation_id: str | None = None,
) -> None:
    """Register a run and save one snapshot per message list, in order."""
    await store.register_run(RunRecord(run_id=run_id, conversation_id=conversation_id))
    for step_index, messages in enumerate(snapshots):
        await store.save_snapshot(ContinuableSnapshot(run_id=run_id, step_index=step_index, messages=messages))


async def _search(
    source: HistorySource,
    query: str,
    *,
    run_id: str | None = None,
    max_matches: int = 10,
    context_lines: int = 5,
    scope: SearchScope = 'all',
    conversation_id: str | None = None,
) -> str:
    """Invoke the search tool directly against a source."""
    toolset: ConversationSearchToolset[None] = ConversationSearchToolset(
        source,
        tool_id='conversation-search',
        max_matches=max_matches,
        context_lines=context_lines,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope=scope,
    )
    return await toolset.search_conversation_history(_run_context(conversation_id), query, run_id=run_id)


class _StubSource:
    """A `HistorySource` that yields messages verbatim, without artifact filtering."""

    def __init__(self, histories: dict[str, list[ModelMessage]]) -> None:
        self._histories = histories

    async def list_runs(self) -> list[RunRecord]:
        return [RunRecord(run_id=run_id) for run_id in self._histories]

    async def run_history(self, *, run_id: str) -> list[ModelMessage]:
        return self._histories[run_id]


@dataclass
class _Todo:
    status: str = 'pending'


_todos = Capability[_Todo](id='todos')
"""Appends a change to its instruction when `finish_todo` runs, as core's `on_change='append'` docs show."""


@_todos.instructions(name='state', on_change='append')
def _todo_state(ctx: RunContext[_Todo]) -> str:
    return f'Todo status: {ctx.deps.status}.'


@_todos.tool(name='finish_todo')
def _finish_todo(ctx: RunContext[_Todo]) -> str:
    ctx.deps.status = 'done'
    return 'Saved.'


_status_mode = Capability[_Todo](id='status_mode')
"""Rewrites its instruction as the todo status changes, so `finish_todo` opens a new instruction window."""


@_status_mode.instructions(name='mode')
def _status_mode_text(ctx: RunContext[_Todo]) -> str:
    return f'Mode: {ctx.deps.status}.'


class TestSnapshotHistorySource:
    async def test_union_recovers_precompaction_originals(self) -> None:
        store = InMemoryStepStore()
        original = _user('the ZEBRA passphrase is 42')
        reply = _reply('noted')
        summary = ModelRequest(parts=[SystemPromptPart(content='Summary of previous conversation:\n\nolder context')])
        follow_up = _user('what was it again?')
        # Snapshot 1 is pre-compaction; snapshot 2 shows compaction having replaced
        # the original with a summary artifact and the conversation moving on.
        await _seed_run(store, 'r1', [original, reply], [summary, reply, follow_up])

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history == [original, reply, follow_up]

    async def test_repeated_identical_messages_keep_distinct_positions(self) -> None:
        store = InMemoryStepStore()
        repeated = _user('repeat this exactly')
        reply = _reply('done')
        await _seed_run(store, 'r1', [repeated], [repeated, repeated, reply])

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history == [repeated, repeated, reply]

    async def test_summary_skip_stays_in_sync_with_compaction(self) -> None:
        # Drift guard: the source hand-copies the prefix `SummarizingCompaction` emits. If
        # that prefix ever changes, an artifact built with the real (current) prefix would no
        # longer be recognized and would pollute the corpus. Building the artifact from the
        # compaction module's own constant makes this test fail the moment the two drift
        # apart. The cross-capability import is deliberately test-only; the source keeps a
        # local literal to avoid runtime coupling.

        store = InMemoryStepStore()
        artifact = ModelRequest(parts=[SystemPromptPart(content=f'{SUMMARY_PREFIX}older summarized context')])
        original = _user('KEEPME original detail')
        await _seed_run(store, 'r1', [artifact, original])

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history == [original]

    async def test_user_authored_summary_lookalike_stays_in_the_corpus(self) -> None:
        # The artifact marker is compaction's full literal, blank line included. A
        # developer's own system prompt that merely opens with the same sentence is
        # not an artifact and must stay searchable.
        store = InMemoryStepStore()
        lookalike = ModelRequest(
            parts=[SystemPromptPart(content='Summary of previous conversation: the ONYX migration is done.')]
        )
        await _seed_run(store, 'r1', [lookalike, _user('carry on')])

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history[0] == lookalike
        assert 'ONYX' in await _search(SnapshotHistorySource(store), 'ONYX')

    async def test_interrupted_snapshots_stay_out_of_the_corpus(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1'))
        settled = _user('the settled SIGNAL detail')
        frontier = _user('the unsettled FRONTIER detail')
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[settled]))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=1, messages=[settled, frontier], state='interrupted')
        )

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history == [settled]

    async def test_unknown_run_yields_empty(self) -> None:
        source = SnapshotHistorySource(InMemoryStepStore())
        assert await source.run_history(run_id='never-ran') == []

    async def test_recovery_round_trips_through_durable_stores(self, tmp_path: Path) -> None:
        # `message_hash` claims stability across snapshot round-trips; the durable
        # stores actually re-serialize and re-validate messages on read, unlike
        # `InMemoryStepStore`, so dedup must hold on reconstructed objects too.
        stores: list[FileStepStore | SqliteStepStore] = [
            FileStepStore(tmp_path / 'runs'),
            SqliteStepStore(database=tmp_path / 'runs.db'),
        ]
        for store in stores:
            original = _user('the ZEBRA passphrase is 42')
            reply = _reply('noted')
            summary = ModelRequest(parts=[SystemPromptPart(content='Summary of previous conversation:\n\nolder')])
            await store.register_run(RunRecord(run_id='r1'))
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[original, reply]))
            await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=[summary, reply]))

            history = await SnapshotHistorySource(store).run_history(run_id='r1')
            rendered = [str(part) for message in history for part in message.parts]
            assert sum('ZEBRA' in text for text in rendered) == 1
            assert sum('noted' in text for text in rendered) == 1
            assert not any('Summary of previous conversation' in text for text in rendered)

    async def test_message_hash_ignores_framework_stamped_metadata(self) -> None:
        # pydantic-ai-harness#810: pydantic-ai stamps `timestamp`, `run_id`, and
        # `conversation_id` (and fills request `instructions`) after the boundary
        # snapshot that first carries a message is serialized. The overlap key
        # must hash the same logical message equally before and after stamping,
        # so those fields are cleared before hashing, as are the instruction changes
        # core appends to an already-saved request; distinct parts still hash
        # differently.
        request = ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='result', tool_call_id='call-1')])
        stamped_request = replace(request, timestamp=datetime.now(UTC), run_id='r1', conversation_id='c1')
        assert message_hash(request) == message_hash(stamped_request)
        delta = InstructionDeltaPart(
            id=InstructionId(AgentInstructionSource(), name='state'), content='Todo status: done.'
        )
        assert message_hash(request) == message_hash(replace(request, parts=[*request.parts, delta]))
        baseline = InstructionBaselinePart(instructions=[])
        assert message_hash(request) == message_hash(replace(request, parts=[*request.parts, baseline]))
        other_return = ToolReturnPart(tool_name='lookup', content='other', tool_call_id='call-1')
        assert message_hash(request) != message_hash(ModelRequest(parts=[other_return, delta]))

        reply = _reply('done')
        stamped_reply = replace(reply, timestamp=datetime.now(UTC), run_id='r1', conversation_id='c1')
        assert message_hash(reply) == message_hash(stamped_reply)
        assert message_hash(request) != message_hash(_reply('done'))

    async def test_stamped_boundary_request_does_not_duplicate_serialized_history(self, tmp_path: Path) -> None:
        # pydantic-ai-harness#810: `StepPersistence.after_node_run` saves the
        # boundary snapshot before pydantic-ai stamps the tool-return request it
        # carries, so hashing the full serialized message makes the prefix overlap
        # fail and `run_history` returns each tool-return request twice (14
        # instead of 6). Serializing stores (`SqliteStepStore`, `FileStepStore`)
        # reproduce it; `InMemoryStepStore` hides it because stamping mutates the
        # very objects the in-memory store holds, so their bytes stay aligned.

        def model_fn(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
            returns = sum(isinstance(p, ToolReturnPart) for message in messages for p in message.parts)
            if returns < 2:
                return ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={'q': str(returns)})])
            return ModelResponse(parts=[TextPart(content='done')])

        stores: list[SqliteStepStore | FileStepStore] = [
            SqliteStepStore(database=tmp_path / 'runs.db'),
            FileStepStore(tmp_path / 'runs'),
        ]
        for store in stores:
            agent: Agent[object, str] = Agent(
                FunctionModel(model_fn),
                capabilities=[StepPersistence(store=store, agent_name='a')],
            )

            @agent.tool_plain
            def lookup(q: str) -> str:
                return f'result-{q}'

            result = await agent.run('hi', conversation_id='c1')
            (run,) = await store.list_runs(conversation_id='c1')
            history = await SnapshotHistorySource(store).run_history(run_id=run.run_id)
            expected = result.all_messages()
            # Counts match (6, not 14): each message once, in order. Strict
            # dataclass equality cannot be used because the boundary request is
            # persisted before pydantic-ai stamps it, so the recovered copy
            # carries `None` run metadata while the live one carries the stamped
            # values; the parts (what the corpus indexes) are identical. Compare
            # on the dedup key instead, which is exactly the stability `message_hash`
            # promises.
            assert len(history) == len(expected) == 6
            assert [message_hash(message) for message in history] == [message_hash(message) for message in expected]

    async def test_list_runs_delegates_to_store(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1'))
        await store.register_run(RunRecord(run_id='r2'))
        assert [r.run_id for r in await SnapshotHistorySource(store).list_runs()] == ['r1', 'r2']

    def test_shipped_stores_satisfy_snapshot_store_protocol(self, tmp_path: Path) -> None:
        assert isinstance(InMemoryStepStore(), SnapshotStore)
        assert isinstance(FileStepStore(tmp_path / 'runs'), SnapshotStore)
        assert isinstance(SqliteStepStore(database=tmp_path / 'runs.db'), SnapshotStore)

        class _UnusedPool:
            def acquire(self) -> AbstractAsyncContextManager[PostgresConnection]:
                raise AssertionError('the pool is not queried')  # pragma: no cover

        assert isinstance(PostgresStepStore(_UnusedPool()), SnapshotStore)

    def test_rejects_store_without_snapshot_seam(self) -> None:
        # A third-party store that lists runs but has not implemented `list_snapshots`
        # fails at construction with a clear error, not mid-search with an obscure
        # `AttributeError`.
        class _RunsOnlyStore:
            async def list_runs(
                self,
                *,
                parent_run_id: str | None = None,
                conversation_id: str | None = None,
            ) -> list[RunRecord]:
                return []  # pragma: no cover

        with pytest.raises(TypeError, match='not a supported search substrate'):
            SnapshotHistorySource(_RunsOnlyStore())  # pyright: ignore[reportArgumentType]

    async def test_pruned_precompaction_snapshot_degrades_gracefully(self) -> None:
        # Bounded snapshot retention (pydantic-ai-harness#442) can prune the early,
        # pre-compaction snapshot that held an original before a search runs. With
        # only the post-compaction snapshot surviving, recovery degrades to a
        # partial result (the original is unrecoverable) rather than erroring.
        store = InMemoryStepStore()
        reply = _reply('noted')
        summary = ModelRequest(parts=[SystemPromptPart(content='Summary of previous conversation:\n\nolder context')])
        follow_up = _user('what was it again?')
        # The pre-compaction snapshot ([original, reply]) has been pruned; only the
        # later snapshot remains, and it never carried the original.
        await _seed_run(store, 'r1', [summary, reply, follow_up])

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history == [reply, follow_up]
        # Search over the survivors still works; it simply cannot surface the lost original.
        assert 'No matches' in await _search(SnapshotHistorySource(store), 'ZEBRA')

    async def test_bounded_file_store_reconciles_retained_snapshots(self, tmp_path: Path) -> None:
        store = FileStepStore(tmp_path / 'bounded-runs', media_store=None, max_snapshots_per_run=2)
        original = _user('the pruned ZEBRA detail')
        reply = _reply('noted')
        summary = ModelRequest(parts=[SystemPromptPart(content='Summary of previous conversation:\n\nolder context')])
        follow_up = _user('what was it again?')
        latest = _reply('the surviving ONYX detail')
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[original, reply]))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=[summary, reply, follow_up]))
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=2, messages=[summary, reply, follow_up, latest])
        )

        history = await SnapshotHistorySource(store).run_history(run_id='r1')
        assert history == [reply, follow_up, latest]
        assert 'ONYX' in await _search(SnapshotHistorySource(store), 'ONYX')
        assert 'No matches' in await _search(SnapshotHistorySource(store), 'ZEBRA')

    async def test_an_appended_instruction_change_does_not_duplicate_the_run(self) -> None:
        """Core adds an instruction change to a request an earlier snapshot already holds.

        The overlap must still match, and the record must keep the later copy so the change
        itself stays searchable.
        """
        store = InMemoryStepStore()
        agent = Agent(TestModel(), deps_type=_Todo, capabilities=[_todos, StepPersistence(store=store)])
        result = await agent.run('Finish the PELICAN todo.', deps=_Todo(), conversation_id='c1')
        run_id = (await store.list_runs())[0].run_id
        assert len(await store.list_snapshots(run_id=run_id)) == 2

        history = await SnapshotHistorySource(store).run_history(run_id=run_id)

        assert [message_hash(message) for message in history] == [
            message_hash(message) for message in result.all_messages()
        ]
        assert any(isinstance(part, InstructionDeltaPart) for message in history for part in message.parts)
        source = SnapshotHistorySource(store)
        assert (await _search(source, 'PELICAN', scope='conversation', conversation_id='c1')).count('[score:') == 1
        assert 'Todo status: done.' in await _search(source, 'done', scope='conversation', conversation_id='c1')

    async def test_a_new_instruction_window_does_not_duplicate_the_run(self) -> None:
        """Core records a new baseline on a request an earlier snapshot already holds."""
        store = InMemoryStepStore()
        agent = Agent(TestModel(), deps_type=_Todo, capabilities=[_todos, _status_mode, StepPersistence(store=store)])
        result = await agent.run('Finish the IBIS todo.', deps=_Todo(), conversation_id='c1')
        run_id = (await store.list_runs())[0].run_id
        first, second = await store.list_snapshots(run_id=run_id)
        # The same tool-return request, which core gives a new baseline after the first save.
        assert message_hash(first.messages[2]) == message_hash(second.messages[2])
        assert not any(isinstance(part, InstructionBaselinePart) for part in first.messages[2].parts)
        assert any(isinstance(part, InstructionBaselinePart) for part in second.messages[2].parts)

        history = await SnapshotHistorySource(store).run_history(run_id=run_id)

        assert [message_hash(message) for message in history] == [
            message_hash(message) for message in result.all_messages()
        ]
        rendered = await _search(SnapshotHistorySource(store), 'IBIS', scope='conversation', conversation_id='c1')
        assert rendered.count('[score:') == 1


class TestConversationSearch:
    async def test_cross_run_recall_through_shared_store(self) -> None:
        store = InMemoryStepStore()
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[
                StepPersistence(store=store),
                ConversationSearch(SnapshotHistorySource(store), scope='conversation'),
                SlidingWindowCompaction(max_messages=2),
            ],
        )
        result = await agent.run('remember the ZEBRA passphrase')
        result = await agent.run('now talk about apples', message_history=result.all_messages())
        await agent.run('now talk about oranges', message_history=result.all_messages())

        # A `SlidingWindowCompaction` trim narrows what each request sends to the model;
        # the persisted snapshots keep the originals, and search reaches them.
        recall = await _search(SnapshotHistorySource(store), 'ZEBRA passphrase')
        assert 'ZEBRA' in recall
        assert 'persisted messages' in recall

    async def test_recall_survives_persistent_compaction(self) -> None:
        store = InMemoryStepStore()
        agent = Agent(
            TestModel(custom_output_text='reply'),
            capabilities=[
                StepPersistence(store=store),
                ConversationSearch(SnapshotHistorySource(store), scope='conversation'),
                SummarizingCompaction(
                    max_messages=2,
                    keep_messages=1,
                    model=TestModel(custom_output_text='SUMMARY TEXT'),
                ),
            ],
        )
        result = await agent.run('the CRIMSON marker is important')
        result = await agent.run('second turn', message_history=result.all_messages())
        result = await agent.run('third turn', message_history=result.all_messages())

        # `SummarizingCompaction` persists its edits: the live history now carries
        # the summary artifact. The corpus recovers the original and excludes the
        # derived summary.
        live = ' '.join(str(part) for message in result.all_messages() for part in message.parts)
        assert 'SUMMARY TEXT' in live
        recall = await _search(SnapshotHistorySource(store), 'CRIMSON marker')
        assert 'CRIMSON' in recall
        assert 'SUMMARY TEXT' not in recall

    async def test_recall_through_agent_tool(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(
            store, 'past-run', [_user('the ZEBRA passphrase is secret')], conversation_id='test-conversation'
        )
        calls: list[int] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not calls:
                calls.append(1)
                assert any(tool.name == 'search_conversation_history' for tool in info.function_tools)
                return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'ZEBRA'})])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[ConversationSearch(SnapshotHistorySource(store), scope='conversation')],
        )
        result = await agent.run('find it', conversation_id='test-conversation')
        returned = next(
            str(part.content)
            for message in result.all_messages()
            for part in getattr(message, 'parts', [])
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert 'ZEBRA' in returned

    async def test_capability_params_flow_into_the_toolset(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(
            store, 'r1', [_user(f'needle entry {i} filler') for i in range(5)], conversation_id='test-conversation'
        )
        calls: list[int] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not calls:
                calls.append(1)
                return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'needle'})])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[
                ConversationSearch(SnapshotHistorySource(store), scope='conversation', max_matches=1, context_lines=0)
            ],
        )
        result = await agent.run('recall', conversation_id='test-conversation')
        rendered = next(
            str(part.content)
            for message in result.all_messages()
            for part in getattr(message, 'parts', [])
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert rendered.count('[score:') == 1
        *_, excerpt = rendered.split('\n\n')
        assert len(excerpt.splitlines()) == 2  # the score line plus a single-line window

    async def test_model_recovers_compacted_original_through_the_tool(self) -> None:
        # The full user story in one flow: compaction drops the original from the
        # live history, and a model tool call over the shared store gets it back.
        store = InMemoryStepStore()
        writer = Agent(
            TestModel(custom_output_text='reply'),
            capabilities=[
                StepPersistence(store=store),
                SummarizingCompaction(
                    max_messages=2,
                    keep_messages=1,
                    model=TestModel(custom_output_text='SUMMARY TEXT'),
                ),
            ],
        )
        result = await writer.run('the CRIMSON marker is important', conversation_id='test-conversation')
        result = await writer.run(
            'second turn', message_history=result.all_messages(), conversation_id='test-conversation'
        )
        result = await writer.run(
            'third turn', message_history=result.all_messages(), conversation_id='test-conversation'
        )
        live = ' '.join(str(part) for message in result.all_messages() for part in message.parts)
        assert 'SUMMARY TEXT' in live

        calls: list[int] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not calls:
                calls.append(1)
                return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'CRIMSON'})])
            return ModelResponse(parts=[TextPart('done')])

        searcher: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[ConversationSearch(SnapshotHistorySource(store), scope='conversation')],
        )
        recall = await searcher.run('what was the marker?', conversation_id='test-conversation')
        returned = next(
            str(part.content)
            for message in recall.all_messages()
            for part in getattr(message, 'parts', [])
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert 'CRIMSON' in returned
        assert 'SUMMARY TEXT' not in returned

    def test_add_instructions_toggle(self) -> None:
        source = SnapshotHistorySource(InMemoryStepStore())
        assert ConversationSearch(source, scope='conversation').get_instructions() is not None
        assert ConversationSearch(source, scope='conversation', add_instructions=False).get_instructions() is None

    def test_instructions_follow_the_configured_scope(self) -> None:
        """`all` must not inherit conversation-only wording that talks the model out of recall."""
        source = SnapshotHistorySource(InMemoryStepStore())
        scoped = ConversationSearch(source, scope='conversation').get_instructions()
        store_wide = ConversationSearch(source, scope='all').get_instructions()
        assert isinstance(scoped, str) and isinstance(store_wide, str)
        assert 'same conversation' in scoped
        assert 'same store' in store_wide
        assert 'same conversation' not in store_wide


class TestSearchTool:
    async def test_results_carry_run_provenance(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('the needle is here')], conversation_id='conv-A')
        rendered = await _search(SnapshotHistorySource(store), 'needle')
        assert 'run: r1' in rendered
        assert 'conversation: conv-A' in rendered

    async def test_run_id_scopes_the_search(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('needle in the first run')])
        await _seed_run(store, 'r2', [_user('needle in the second run')])
        source = SnapshotHistorySource(store)

        scoped = await _search(source, 'needle', run_id='r2')
        assert 'second run' in scoped
        assert 'first run' not in scoped

        unscoped = await _search(source, 'needle')
        assert 'first run' in unscoped
        assert 'second run' in unscoped

    async def test_unknown_run_id_reports_missing(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('some content')])
        rendered = await _search(SnapshotHistorySource(store), 'anything', run_id='nope')
        assert "No persisted history for run 'nope'" in rendered

    async def test_empty_store_message(self) -> None:
        rendered = await _search(SnapshotHistorySource(InMemoryStepStore()), 'anything')
        assert 'No persisted conversation history to search yet' in rendered

    async def test_run_without_snapshots_is_skipped(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='registered-only'))
        rendered = await _search(SnapshotHistorySource(store), 'anything')
        assert 'No persisted conversation history to search yet' in rendered


class TestSearchScope:
    """A store shared by several conversations is one corpus unless `scope` narrows it."""

    @staticmethod
    async def _two_conversation_store() -> InMemoryStepStore:
        store = InMemoryStepStore()
        await _seed_run(store, 'r-alice', [_user('the passport number is X99881234')], conversation_id='conv-alice')
        await _seed_run(store, 'r-bob', [_user('remind me about the passport thing')], conversation_id='conv-bob')
        return store

    async def test_all_scope_spans_every_conversation(self) -> None:
        source = SnapshotHistorySource(await self._two_conversation_store())
        rendered = await _search(source, 'passport', scope='all', conversation_id='conv-bob')
        assert 'X99881234' in rendered
        assert 'conversation: conv-alice' in rendered

    async def test_conversation_scope_hides_other_conversations(self) -> None:
        source = SnapshotHistorySource(await self._two_conversation_store())
        rendered = await _search(source, 'passport', scope='conversation', conversation_id='conv-bob')
        assert 'X99881234' not in rendered
        assert 'conversation: conv-alice' not in rendered
        assert 'remind me about the passport thing' in rendered

    async def test_conversation_scope_blocks_an_out_of_scope_run_id(self) -> None:
        """An explicit `run_id` cannot reach past the scope, and says nothing about the run."""
        source = SnapshotHistorySource(await self._two_conversation_store())
        rendered = await _search(source, 'passport', run_id='r-alice', scope='conversation', conversation_id='conv-bob')
        assert rendered == "No persisted history for run 'r-alice'."

    async def test_conversation_scope_without_a_conversation_id_searches_nothing(self) -> None:
        """Fail closed: matching `conversation_id is None` would expose every unlabelled run."""
        store = await self._two_conversation_store()
        await _seed_run(store, 'r-anon', [_user('an unlabelled passport mention')])
        rendered = await _search(SnapshotHistorySource(store), 'passport', scope='conversation')
        assert 'Conversation-scoped search needs a conversation id' in rendered
        assert 'passport' not in rendered.replace('Conversation-scoped', '')
        assert 'X99881234' not in rendered

    async def test_empty_conversation_corpus_names_the_scope_without_naming_other_conversations(self) -> None:
        """The default's common miss is a per-run id, not an unpaired `StepPersistence`."""
        source = SnapshotHistorySource(await self._two_conversation_store())
        rendered = await _search(source, 'passport', scope='conversation', conversation_id='conv-unseen')
        assert 'No persisted history in this conversation yet' in rendered
        assert '`conversation_id=`' in rendered
        assert 'conv-alice' not in rendered
        assert 'X99881234' not in rendered

    async def test_empty_store_under_all_scope_points_at_step_persistence(self) -> None:
        rendered = await _search(SnapshotHistorySource(InMemoryStepStore()), 'passport', scope='all')
        assert 'No persisted conversation history to search yet' in rendered
        assert '`StepPersistence`' in rendered

    async def test_conversation_scope_is_announced_in_the_tool_description(self) -> None:
        source = SnapshotHistorySource(InMemoryStepStore())
        scoped: ConversationSearchToolset[None] = ConversationSearchToolset(
            source, tool_id='t', max_matches=10, context_lines=5, bm25_k1=1.5, bm25_b=0.75, scope='conversation'
        )
        unscoped: ConversationSearchToolset[None] = ConversationSearchToolset(
            source, tool_id='t', max_matches=10, context_lines=5, bm25_k1=1.5, bm25_b=0.75, scope='all'
        )
        scoped_tools = await scoped.get_tools(_run_context())
        unscoped_tools = await unscoped.get_tools(_run_context())
        scoped_description = scoped_tools['search_conversation_history'].tool_def.description or ''
        unscoped_description = unscoped_tools['search_conversation_history'].tool_def.description or ''
        assert 'only the current conversation' in scoped_description
        assert 'only the current conversation' not in unscoped_description

    async def test_capability_default_scope_reaches_the_tool(self) -> None:
        """The end-to-end path defaults to keeping another conversation out of the tool result."""
        store = await self._two_conversation_store()
        agent = Agent(
            TestModel(call_tools=['search_conversation_history']),
            capabilities=[
                ConversationSearch(SnapshotHistorySource(store), scope='conversation'),
            ],
        )
        result = await agent.run('find the passport detail', conversation_id='conv-bob')
        returned = '\n'.join(
            str(part.content)
            for message in result.all_messages()
            for part in getattr(message, 'parts', [])
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert 'conv-alice' not in returned
        assert 'X99881234' not in returned

    async def test_conversation_scope_follows_a_message_history_chain(self) -> None:
        """A follow-up run inherits the id from `message_history`, so it reaches the run behind it.

        Pins the resolution order the docs describe: explicit argument, then the most recent
        `conversation_id` on `message_history`, then a fresh id. Without the inherited id the
        follow-up would be its own conversation and recall nothing.
        """
        store = InMemoryStepStore()
        capabilities = [
            StepPersistence(store=store),
            ConversationSearch(SnapshotHistorySource(store), scope='conversation'),
        ]
        writer = Agent(TestModel(), capabilities=capabilities)
        first = await writer.run('the ZEBRA passphrase is secret')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(part.part_kind == 'tool-return' for part in messages[-1].parts):
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'ZEBRA'})])

        reader = Agent(FunctionModel(model_fn), capabilities=capabilities)
        threaded = await reader.run('find it', message_history=first.all_messages())
        returned = [
            str(part.content)
            for message in threaded.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        ]
        assert any('ZEBRA' in rendered for rendered in returned)

        detached = await reader.run('find it')
        detached_returned = next(
            str(part.content)
            for message in detached.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert 'ZEBRA' not in detached_returned

    def test_unset_scope_resolves_to_conversation_and_warns(self) -> None:
        """The default moved from `all` to `conversation`, and nothing else would say so.

        A caller who relied on the old default keeps working with a narrower corpus rather
        than erroring, so the change is invisible without the warning.
        """
        source = SnapshotHistorySource(InMemoryStepStore())
        with pytest.warns(HarnessDeprecationWarning) as record:
            capability = ConversationSearch(source)
        assert capability.scope is None
        assert capability.effective_scope == 'conversation'
        message = str(record[0].message)
        assert "scope='all'" in message
        assert "scope='conversation'" in message
        assert 'warning will be removed in the next breaking release' in message

    def test_unset_scope_warns_once_per_instance(self) -> None:
        """Per instance, not per search: warning on every tool invocation would be noise."""
        source = SnapshotHistorySource(InMemoryStepStore())
        with pytest.warns(HarnessDeprecationWarning) as record:
            capability = ConversationSearch(source)
            capability.get_toolset()
            capability.get_toolset()
            capability.get_instructions()
        assert len(record) == 1

    def test_unset_toolset_scope_warns(self) -> None:
        """The toolset is public and its default moved too, so it announces the same change."""
        source = SnapshotHistorySource(InMemoryStepStore())
        with pytest.warns(HarnessDeprecationWarning):
            ConversationSearchToolset[None](
                source, tool_id='t', max_matches=10, context_lines=5, bm25_k1=1.5, bm25_b=0.75
            )

    def test_explicit_scope_never_warns(self) -> None:
        """Choosing either value opts out of the warning; neither value is deprecated."""
        source = SnapshotHistorySource(InMemoryStepStore())
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any warning at all fails this test
            assert ConversationSearch(source, scope='conversation').effective_scope == 'conversation'
            assert ConversationSearch(source, scope='all').effective_scope == 'all'
            ConversationSearch(source, scope='conversation').get_toolset()
            ConversationSearchToolset[None](
                source, tool_id='t', max_matches=10, context_lines=5, bm25_k1=1.5, bm25_b=0.75, scope='all'
            )

    async def test_explicit_all_scope_still_searches_store_wide(self) -> None:
        """The opt-out has to actually restore the old behavior, not just silence the warning."""
        store = await self._two_conversation_store()

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(part.part_kind == 'tool-return' for part in messages[-1].parts):
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'passport'})])

        agent = Agent(
            FunctionModel(model_fn),
            capabilities=[ConversationSearch(SnapshotHistorySource(store), scope='all')],
        )
        result = await agent.run('find the passport detail', conversation_id='conv-bob')
        returned = next(
            str(part.content)
            for message in result.all_messages()
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert 'X99881234' in returned
        assert 'conv-alice' in returned

    async def test_context_window_stays_within_run(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('unrelated lead-in'), _user('the needle target')])
        await _seed_run(store, 'r2', [_user('OTHERRUN content that must not leak')])
        rendered = await _search(SnapshotHistorySource(store), 'needle', context_lines=5)
        assert 'needle target' in rendered
        assert 'OTHERRUN' not in rendered

    async def test_no_match_returns_message(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('alpha beta gamma'), _reply('delta epsilon')])
        # `zzzmissing` appears nowhere -> df 0 -> idf 0 -> every score 0 -> no matches.
        assert 'No matches' in await _search(SnapshotHistorySource(store), 'zzzmissing')

    async def test_empty_query_returns_no_match(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('alpha beta')])
        # A query with no word tokens ranks nothing.
        assert 'No matches' in await _search(SnapshotHistorySource(store), '!!!')

    async def test_unicode_terms_are_searchable(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('Grüße aus München'), _user('unrelated filler line')])
        rendered = await _search(SnapshotHistorySource(store), 'münchen')
        assert 'München' in rendered

    async def test_rendering_covers_all_part_types(self) -> None:
        # A stub source is used deliberately: it does not filter summary artifacts,
        # exercising the renderer's defensive `[Compaction summary]` collapse for
        # custom sources that pass them through.
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='system guidance ' * 30),
                    UserPromptPart(content='find the ZEBRA passphrase mango'),
                    ToolReturnPart(tool_name='readfile', content='X' * 600, tool_call_id='c1'),
                    ToolReturnPart(tool_name='readfile', content='please retry', tool_call_id='c1', outcome='retried'),
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content='assistant reply about mango'),
                    ToolCallPart(tool_name='search', args={'q': 'Y' * 300}, tool_call_id='c2'),
                    ThinkingPart(content='internal thought'),
                ]
            ),
            ModelRequest(parts=[SystemPromptPart(content='Summary of previous conversation:\n\nolder kiwi context')]),
        ]
        source = _StubSource({'r1': messages})

        # `mango` appears in two adjacent messages, so the second match's window
        # overlaps the first and is skipped (overlapping-window dedup).
        rendered = await _search(source, 'mango', context_lines=5, max_matches=10)
        assert 'Assistant: assistant reply about mango' in rendered
        assert '[Compaction summary]' in rendered
        assert 'Tool Call [search]' in rendered
        assert 'Tool [readfile]' in rendered
        assert 'Retry [readfile]: please retry' in rendered  # a retried tool return is searchable
        assert '...' in rendered  # truncation applied to the long tool return / args

    async def test_retry_feedback_is_indexed(self) -> None:
        """A retry that named no tool is recalled like one that did."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(content='mango question'),
                    RetryFeedbackPart(
                        content=[{'type': 'missing', 'loc': ('mango',), 'msg': 'Field required', 'input': {}}],
                        retry=RetryInfo(cause='validation_error'),
                    ),
                ]
            ),
            ModelResponse(parts=[TextPart(content='mango answer')]),
        ]
        source = _StubSource({'r1': messages})

        rendered = await _search(source, 'mango')

        # Core's compact JSON, not `str(part.content)`, which would index a Python repr of the error list.
        assert 'Retry: [{"type":"missing","loc":["mango"]' in rendered
        assert "{'type': 'missing'" not in rendered

    async def test_retry_feedback_truncates_but_stays_searchable(self) -> None:
        """The display excerpt is capped while the index keeps the full text."""
        errors: list[ErrorDetails] = [
            {'type': 'missing', 'loc': (f'field_{i}',), 'msg': 'Field required', 'input': {}} for i in range(40)
        ]
        errors.append({'type': 'missing', 'loc': ('rambutan',), 'msg': 'Field required', 'input': {}})
        messages: list[ModelMessage] = [
            ModelRequest(parts=[RetryFeedbackPart(content=errors, retry=RetryInfo(cause='validation_error'))]),
        ]
        source = _StubSource({'r1': messages})

        rendered = await _search(source, 'rambutan')

        assert 'Retry: ' in rendered
        assert '...' in rendered

    async def test_a_hand_built_legacy_retry_prompt_part_is_indexed(self) -> None:
        """Code that still builds one and passes it as `message_history` reaches the indexer.

        A stored history loads as one of the two parts above, and the model translates an
        in-memory instance before the request goes out, but the index reads the history first.
        """
        content = 'please retry mango ' + 'padding ' * 80 + 'rambutan'
        part = legacy_retry_prompt_part(content, tool_name='readfile', tool_call_id='c1')
        source = _StubSource({'r1': [ModelRequest(parts=[part])]})

        rendered = await _search(source, 'rambutan')

        # `rambutan` sits past the display cutoff: matching on it proves the index kept the full
        # text, and the excerpt is the clipped window around it.
        assert 'Retry [readfile]: ...' in rendered
        assert rendered.endswith(' rambutan')

    async def test_tool_availability_delta_is_not_indexed(self) -> None:
        """Tool-list bookkeeping is not conversation content, so it contributes no line.

        Built with no arguments deliberately: the indexer rejects the part on its type and
        never reads the names it carries, and the field holding them was renamed
        (`added` -> `tools_added`) in pydantic-ai 2.26. Naming it here would pin the test to
        a release later than the floor the runtime actually needs.
        """
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(content='mango question'),
                    ToolAvailabilityDeltaPart(),
                ]
            ),
            ModelResponse(parts=[TextPart(content='mango answer')]),
        ]
        source = _StubSource({'r1': messages})

        rendered = await _search(source, 'mango')

        assert 'User: mango question' in rendered
        assert 'Assistant: mango answer' in rendered
        # The delta contributes nothing. Before it was recognized it fell through to the retry
        # branch, which read `part.tool_name` and raised `AttributeError`.
        assert 'Retry [' not in rendered

    async def test_long_parts_clip_around_a_term_past_the_cutoff(self) -> None:
        """A term past the 500-char cutoff matches and is shown: the excerpt shows the text around it."""
        long_user = 'pomegranate ' * 60 + 'USERTAIL'
        long_text = 'cranberry ' * 80 + 'TEXTTAIL'
        source = _StubSource(
            {
                'r1': [
                    ModelRequest(parts=[UserPromptPart(content=long_user)]),
                    ModelResponse(parts=[TextPart(content=long_text)]),
                ]
            }
        )
        rendered = await _search(source, 'USERTAIL', context_lines=0)
        excerpt = rendered.split(':\n\n', 1)[1]
        assert 'Found 1 match(es)' in rendered
        assert '[0] User: ...' in excerpt
        assert excerpt.endswith('USERTAIL')
        assert len(excerpt) < len(long_user)
        rendered = await _search(source, 'TEXTTAIL', context_lines=0)
        excerpt = rendered.split(':\n\n', 1)[1]
        assert '[1] Assistant: ...' in excerpt
        assert excerpt.endswith('TEXTTAIL')

    async def test_clip_keeps_the_head_when_the_term_is_inside_it(self) -> None:
        long_user = 'HEADTERM ' + 'filler ' * 100
        rendered = await _search(_StubSource({'r1': [_user(long_user)]}), 'HEADTERM')
        excerpt = rendered.split(':\n\n', 1)[1]
        assert '[0] User: HEADTERM filler' in excerpt
        assert excerpt.endswith('...')

    async def test_clip_centres_a_term_in_the_middle_with_ellipses_on_both_sides(self) -> None:
        long_user = 'before ' * 100 + 'MIDTERM' + ' after' * 100
        rendered = await _search(_StubSource({'r1': [_user(long_user)]}), 'MIDTERM')
        excerpt = rendered.split(':\n\n', 1)[1]
        assert '[0] User: ...' in excerpt
        assert 'MIDTERM' in excerpt
        assert excerpt.endswith('...')

    async def test_context_lines_without_the_term_keep_their_head(self) -> None:
        source = _StubSource({'r1': [_user('lead ' * 200), _user('the TARGET line')]})
        rendered = await _search(source, 'TARGET', context_lines=1)
        assert '[0] User: lead lead' in rendered
        assert '[1] User: the TARGET line' in rendered

    async def test_instruction_updates_stay_searchable_past_display_cutoff(self) -> None:
        history = ModelMessagesTypeAdapter.validate_python(
            [
                {
                    'kind': 'request',
                    'parts': [
                        {'part_kind': 'instruction-delta', 'id': 'agent:state', 'content': 'cedar ' * 50 + 'DELTATAIL'}
                    ],
                }
            ]
        )
        rendered = await _search(_StubSource({'r1': history}), 'DELTATAIL')
        assert rendered == snapshot("""\
Found 1 match(es) for 'DELTATAIL' in 1 persisted messages:

[score: 0.3 | run: r1]
[0] System: ...cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar cedar DELTATAIL
</context>\
""")

    async def test_max_matches_and_context_lines_honored(self) -> None:
        store = InMemoryStepStore()
        corpus: list[ModelMessage] = [_user(f'needle term entry number {i} filler') for i in range(20)]
        await _seed_run(store, 'r1', corpus)
        rendered = await _search(SnapshotHistorySource(store), 'needle', max_matches=1, context_lines=0)
        assert rendered.count('[score:') == 1
        *_, excerpt = rendered.split('\n\n')
        assert len(excerpt.splitlines()) == 2  # the score line plus a single-line window

    async def test_media_content_stays_searchable_around_the_binary(self) -> None:
        store = InMemoryStepStore()
        media_prompt = ModelRequest(
            parts=[
                UserPromptPart(
                    content=['the FIGURE caption text', BinaryContent(data=b'\x89PNG fake', media_type='image/png')]
                )
            ]
        )
        await _seed_run(store, 'r1', [media_prompt, _user('an unrelated entry')])
        rendered = await _search(SnapshotHistorySource(store), 'FIGURE')
        assert '[score:' in rendered
        assert 'FIGURE caption' in rendered

    async def test_externalized_media_binary_never_enters_the_corpus(self, tmp_path: Path) -> None:
        # The durable stores externalize a large `BinaryContent` and restore it on
        # read. The text index must render only the textual parts of the prompt:
        # str()-ing the whole content would fold the image's byte representation
        # into the corpus (a 70 KiB image is ~280k escaped-byte characters),
        # bloating the index and leaking binary into excerpts.
        store = FileStepStore(tmp_path / 'runs')
        blob = b'\x89PNG\r\n' + b'\x00' * (70 * 1024)
        media_prompt = ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        'the FIGURE caption text',
                        TextContent(content='ANNOTATED note'),
                        BinaryContent(data=blob, media_type='image/png'),
                    ]
                )
            ]
        )
        await store.register_run(RunRecord(run_id='r1'))
        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=0, messages=[media_prompt]))

        source = SnapshotHistorySource(store)
        rendered = await _search(source, 'FIGURE')
        assert 'FIGURE caption' in rendered
        # `TextContent` carries searchable text and is indexed alongside plain strings.
        assert 'ANNOTATED' in await _search(source, 'ANNOTATED')
        assert 'BinaryContent' not in rendered
        assert 'image/png' not in rendered
        # The escaped-byte tokens the raw representation would contribute (e.g. `x00`,
        # `x89png`) are absent from the index, so they match nothing.
        assert 'No matches' in await _search(source, 'x00')
        assert 'No matches' in await _search(source, 'x89png')

    async def test_index_prefix_is_not_a_searchable_token(self) -> None:
        # The `[N]` excerpt numbering must stay display-only: indexed, each
        # ordinal would be a rare high-IDF token and numeric queries would
        # match messages by position instead of content.
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('alpha'), _user('beta'), _user('gamma')])
        rendered = await _search(SnapshotHistorySource(store), '1')
        assert 'No matches' in rendered

    async def test_unrenderable_message_gets_placeholder_and_stays_unindexed(self) -> None:
        store = InMemoryStepStore()
        thinking = ModelResponse(parts=[ThinkingPart(content='private reasoning')])
        await _seed_run(store, 'r1', [_user('the needle target'), thinking, _user('after')])

        rendered = await _search(SnapshotHistorySource(store), 'needle', context_lines=5)
        assert '[1] [no text content]' in rendered
        # Neither the thinking text nor the placeholder wording is searchable.
        assert 'No matches' in await _search(SnapshotHistorySource(store), 'reasoning')
        assert 'No matches' in await _search(SnapshotHistorySource(store), 'content')

    async def test_overlap_skip_backfills_lower_ranked_matches(self) -> None:
        # Two top-ranked matches sharing one excerpt window must not exhaust
        # `max_matches`: distinct lower-ranked matches take the freed slots.
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('needle needle needle first'), _user('needle needle needle second')])
        await _seed_run(store, 'r2', [_user('a single needle far away')])
        rendered = await _search(SnapshotHistorySource(store), 'needle', max_matches=2, context_lines=5)
        assert rendered.count('[score:') == 2
        assert 'run: r2' in rendered


class TestConfigValidation:
    """Out-of-range tuning is rejected where the toolset is built, not silently honored."""

    def test_invalid_tuning_is_rejected(self) -> None:
        source = SnapshotHistorySource(InMemoryStepStore())
        with pytest.raises(ValueError, match=re.escape('max_matches must be non-negative, got -1.')):
            ConversationSearch(source, scope='conversation', max_matches=-1).get_toolset()
        with pytest.raises(ValueError, match=re.escape('context_lines must be non-negative, got -1.')):
            ConversationSearch(source, scope='conversation', context_lines=-1).get_toolset()
        with pytest.raises(ValueError, match=re.escape('bm25_k1 must be a non-negative finite number, got -1.0.')):
            ConversationSearch(source, scope='conversation', bm25_k1=-1.0).get_toolset()
        # `nan` and `inf` pass an ordering check and would poison every score instead.
        with pytest.raises(ValueError, match='bm25_k1 must be a non-negative finite number'):
            ConversationSearch(source, scope='conversation', bm25_k1=float('nan')).get_toolset()
        with pytest.raises(ValueError, match='bm25_k1 must be a non-negative finite number'):
            ConversationSearch(source, scope='conversation', bm25_k1=float('inf')).get_toolset()
        with pytest.raises(ValueError, match=re.escape('bm25_b must be between 0.0 and 1.0, got nan.')):
            ConversationSearch(source, scope='conversation', bm25_b=float('nan')).get_toolset()
        with pytest.raises(ValueError, match=re.escape('bm25_b must be between 0.0 and 1.0, got 1.5.')):
            ConversationSearch(source, scope='conversation', bm25_b=1.5).get_toolset()
        with pytest.raises(ValueError, match=re.escape('bm25_b must be between 0.0 and 1.0, got -0.5.')):
            ConversationSearch(source, scope='conversation', bm25_b=-0.5).get_toolset()

    async def test_negative_k1_would_have_divided_by_zero(self) -> None:
        # The guard's reason for existing: k1=-1 with b=0 zeroes the BM25 denominator
        # for any single-occurrence term, so an unvalidated toolset raises mid-search.
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user('the needle target')])
        with pytest.raises(ValueError, match='bm25_k1 must be a non-negative finite number'):
            ConversationSearchToolset[None](
                SnapshotHistorySource(store),
                tool_id='conversation-search',
                max_matches=10,
                context_lines=5,
                bm25_k1=-1.0,
                bm25_b=0.0,
                scope='conversation',
            )

    async def test_negative_max_matches_no_longer_uncaps_output(self) -> None:
        # An unvalidated negative bound left `len(results) == max_matches` unsatisfiable,
        # emitting every ranked excerpt instead of capping them.
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user(f'needle entry {i} filler') for i in range(5)])
        with pytest.raises(ValueError, match='max_matches must be non-negative'):
            await _search(SnapshotHistorySource(store), 'needle', max_matches=-1)


class TestSpeechIsSearchable:
    """A realtime `SpeechPart` carries a transcript; the index renders it so speech is recallable."""

    async def test_an_assistant_speech_turn_is_indexed_and_rendered(self) -> None:
        history: list[ModelMessage] = [
            _user('boot up'),
            ModelResponse(parts=[SpeechPart(speaker='assistant', transcript='the PANGOLIN protocol is ready')]),
        ]
        rendered = await _search(_StubSource({'r1': history}), 'PANGOLIN')
        assert 'PANGOLIN' in rendered
        assert 'Speech [assistant]:' in rendered

    async def test_a_user_speech_turn_is_indexed_and_rendered(self) -> None:
        history: list[ModelMessage] = [
            ModelRequest(parts=[SpeechPart(speaker='user', transcript='remember the MARMOT codeword')]),
        ]
        rendered = await _search(_StubSource({'r1': history}), 'MARMOT')
        assert 'MARMOT' in rendered
        assert 'Speech [user]:' in rendered

    async def test_audio_only_speech_contributes_no_searchable_text(self) -> None:
        """A `SpeechPart` with no transcript renders no line, so it cannot match a query."""
        history: list[ModelMessage] = [ModelRequest(parts=[SpeechPart(speaker='user')])]
        assert 'No matches' in await _search(_StubSource({'r1': history}), 'anything')

    async def test_a_long_assistant_transcript_clips_around_the_term(self) -> None:
        # A spoken turn is clipped in the display excerpt like `TextPart`, while the full transcript
        # stays in the index so a term past the 500-char cutoff still matches and is shown.
        long_transcript = 'nightingale ' * 60 + 'SPEECHTAIL'
        source = _StubSource(
            {'r1': [_user('go'), ModelResponse(parts=[SpeechPart(speaker='assistant', transcript=long_transcript)])]}
        )
        rendered = await _search(source, 'SPEECHTAIL')
        excerpts = rendered.split(':\n\n', 1)[1]
        assert 'Found 1 match(es)' in rendered
        assert 'Speech [assistant]: ...' in excerpts
        assert excerpts.endswith('SPEECHTAIL')

    async def test_a_long_user_transcript_clips_around_the_term(self) -> None:
        long_transcript = 'petrichor ' * 60 + 'USERSPEECHTAIL'
        source = _StubSource({'r1': [ModelRequest(parts=[SpeechPart(speaker='user', transcript=long_transcript)])]})
        rendered = await _search(source, 'USERSPEECHTAIL')
        excerpts = rendered.split(':\n\n', 1)[1]
        assert 'Found 1 match(es)' in rendered
        assert 'Speech [user]: ...' in excerpts
        assert excerpts.endswith('USERSPEECHTAIL')


class TestCorpusAcrossRuns:
    """A run started with `message_history` persists that history again; the corpus holds it once."""

    async def test_threaded_runs_index_inherited_history_once(self) -> None:
        store = InMemoryStepStore()
        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[StepPersistence(store=store)])
        result = await agent.run('the ZEBRA passphrase is 42', conversation_id='c1')
        for turn in range(3):
            result = await agent.run(f'filler turn {turn}', message_history=result.all_messages(), conversation_id='c1')

        rendered = await _search(SnapshotHistorySource(store), 'ZEBRA', scope='conversation', conversation_id='c1')
        assert rendered.count('[score:') == 1
        assert rendered.count('the ZEBRA passphrase is 42') == 1
        assert rendered.startswith(f"Found 1 match(es) for 'ZEBRA' in {len(result.all_messages())} persisted messages:")

        # `run_id` still resolves one run's full record, inherited history included.
        last_run = (await store.list_runs(conversation_id='c1'))[-1]
        scoped = await _search(
            SnapshotHistorySource(store), 'ZEBRA', run_id=last_run.run_id, scope='conversation', conversation_id='c1'
        )
        assert 'the ZEBRA passphrase is 42' in scoped
        assert f'run: {last_run.run_id}' in scoped

    async def test_only_the_inherited_head_is_left_out(self) -> None:
        """A message that repeats after the run's own first message stays in the corpus.

        Numbering follows the run's full record, so `r2`'s hit matches a `run_id` search of `r2`.
        """
        repeated = _user('the ORCHID note')
        reply = _reply('noted')
        source = _StubSource({'r1': [repeated, reply], 'r2': [repeated, reply, _user('new turn'), repeated]})

        rendered = await _search(source, 'ORCHID', context_lines=0)

        assert rendered == snapshot("""\
Found 2 match(es) for 'ORCHID' in 4 persisted messages:

[score: 0.6 | run: r1]
[0] User: the ORCHID note

---

[score: 0.6 | run: r2]
[3] User: the ORCHID note\
""")

    async def test_a_run_holding_only_inherited_history_adds_no_section(self) -> None:
        message = _user('the LOTUS detail')
        source = _StubSource({'r1': [message], 'r2': [message]})
        rendered = await _search(source, 'LOTUS')
        assert 'run: r1' in rendered
        assert 'run: r2' not in rendered

    async def test_fresh_runs_asking_the_same_question_stay_distinct(self) -> None:
        """Without `message_history`, a repeated prompt is new history, not an inherited copy.

        The prompt part's own timestamp keeps the two hashes apart; only the message-level
        timestamp is excluded from `message_hash`.
        """
        store = InMemoryStepStore()
        agent = Agent(TestModel(custom_output_text='ok'), capabilities=[StepPersistence(store=store)])
        await agent.run('what is the QUOKKA status', conversation_id='c1')
        await agent.run('what is the QUOKKA status', conversation_id='c1')

        rendered = await _search(SnapshotHistorySource(store), 'QUOKKA', scope='conversation', conversation_id='c1')

        runs = await store.list_runs(conversation_id='c1')
        assert len(runs) == 2
        assert rendered.count('[score:') == 2
        assert all(f'run: {run.run_id}' in rendered for run in runs)

    async def test_an_instruction_change_on_inherited_history_stays_indexed(self) -> None:
        """A resumed run can append an instruction change to the request it inherited.

        The hash ignores the change, so the request must still count as new in that run, or the
        change would never be indexed.
        """
        user = _user('finish the HERON task')
        call = ModelResponse(parts=[ToolCallPart(tool_name='finish', args={}, tool_call_id='c1')])
        tool_return = ModelRequest(parts=[ToolReturnPart(tool_name='finish', content='ok', tool_call_id='c1')])
        changed = replace(
            tool_return,
            parts=[
                *tool_return.parts,
                InstructionDeltaPart(
                    id=InstructionId(AgentInstructionSource(), name='state'), content='status FINISHED'
                ),
            ],
        )
        reply = _reply('all set')
        # `r3` continues `r2`, so it inherits the changed request and must not index the change again.
        source = _StubSource(
            {
                'r1': [user, call, tool_return],
                'r2': [user, call, changed, reply],
                'r3': [user, call, changed, reply, _user('next step')],
            }
        )

        rendered = await _search(source, 'FINISHED', context_lines=0)

        assert rendered == snapshot("""\
Found 1 match(es) for 'FINISHED' in 6 persisted messages:

[score: 1.0 | run: r2]
[2] Tool [finish]: ok
System: <context id="agent:state">
status FINISHED
</context>\
""")
        assert (await _search(source, 'HERON')).count('[score:') == 1
        # The request that carries the change is indexed whole, so its tool return repeats once.
        assert (await _search(source, 'ok')).count('[score:') == 2


class TestExcerptRendering:
    async def test_tool_return_files_are_named_not_spelled_out(self) -> None:
        file = BinaryContent(data=b'\x89PNG\r\n' + b'\x00' * 4096, media_type='image/png', identifier='chart.png')
        source = _StubSource(
            {
                'r1': [
                    ModelRequest(
                        parts=[ToolReturnPart(tool_name='render', content=['chart READY', file], tool_call_id='c1')]
                    )
                ]
            }
        )
        assert await _search(source, 'READY') == snapshot("""\
Found 1 match(es) for 'READY' in 1 persisted messages:

[score: 0.3 | run: r1]
[0] Tool [render]: chart READY See file chart.png.\
""")
        assert await _search(source, 'x00') == snapshot("No matches for 'x00' in 1 persisted messages.")

    async def test_retry_prompt_excerpt_is_clipped(self) -> None:
        source = _StubSource(
            {'r1': [ModelRequest(parts=[legacy_retry_prompt_part('needle ' + 'A' * 5000, tool_name='t')])]}
        )
        rendered = await _search(source, 'needle')
        excerpt = rendered.split(':\n\n', 1)[1]
        assert excerpt == f'[score: 0.3 | run: r1]\n[0] Retry [t]: needle {"A" * 493}...'

    async def test_retry_prompt_validation_errors_render_as_json_without_core_instructions(self) -> None:
        errors: list[ErrorDetails] = [
            {
                'type': 'missing',
                'loc': ('city',),
                'msg': 'Field required',
                'input': {'town': 'Paris'},
                'ctx': {'min_length': 3},
            }
        ]
        part = legacy_retry_prompt_part(errors, tool_name='weather')
        rendered = await _search(_StubSource({'r1': [ModelRequest(parts=[part])]}), 'required')
        assert rendered == snapshot("""\
Found 1 match(es) for 'required' in 1 persisted messages:

[score: 0.3 | run: r1]
[0] Retry [weather]: [{"type":"missing","loc":["city"],"msg":"Field required","input":{"town":"Paris"}}]\
""")

    async def test_retried_tool_return_validation_errors_render_as_json(self) -> None:
        """The part a run records for a tool-call retry indexes the same JSON as the legacy part."""
        errors: list[ErrorDetails] = [
            {'type': 'missing', 'loc': ('city',), 'msg': 'Field required', 'input': {'town': 'Paris'}}
        ]
        part = ToolReturnPart(tool_name='weather', content=errors, tool_call_id='c1', outcome='retried')
        rendered = await _search(_StubSource({'r1': [ModelRequest(parts=[part])]}), 'required')
        assert rendered == snapshot("""\
Found 1 match(es) for 'required' in 1 persisted messages:

[score: 0.3 | run: r1]
[0] Retry [weather]: [{"type":"missing","loc":["city"],"msg":"Field required","input":{"town":"Paris"}}]\
""")

    async def test_a_long_term_past_the_cutoff_is_shown_whole(self) -> None:
        token = 'k' * 300
        text = 'lead ' * 120 + token + ' trail' * 100
        rendered = await _search(_StubSource({'r1': [_user(text)]}), token)
        excerpt = rendered.split('[0] User: ', 1)[1]
        # The term spans 600-900, so the window ends where it does.
        assert excerpt == '...' + text[400:900] + '...'

    async def test_a_term_near_the_end_keeps_a_full_window(self) -> None:
        text = 'x' * 1000 + ' ENDTERM'
        rendered = await _search(_StubSource({'r1': [_user(text)]}), 'ENDTERM')
        excerpt = rendered.split('[0] User: ', 1)[1]
        assert excerpt == '...' + text[-500:]


def test_unset_toolset_scope_warning_points_at_the_constructing_code() -> None:
    """A subscripted-generic construction adds a `typing` frame the warning must skip."""
    source = SnapshotHistorySource(InMemoryStepStore())
    for build in (ConversationSearchToolset[None], ConversationSearchToolset):
        with pytest.warns(HarnessDeprecationWarning) as record:
            build(source, tool_id='t', max_matches=10, context_lines=5, bm25_k1=1.5, bm25_b=0.75)
        assert record[0].filename == __file__

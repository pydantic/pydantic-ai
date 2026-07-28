"""Tests for the `ConversationSearch` capability.

End-to-end recall (persist with `StepPersistence`, compact, search) is driven
through `Agent(..., capabilities=[...])`. The search tool's ranking, rendering,
and source edge cases are exercised through the public
`ConversationSearchToolset` and `SnapshotHistorySource` directly, over seeded
`InMemoryStepStore` instances.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from pydantic_ai_harness.compaction import SlidingWindowCompaction, SummarizingCompaction
from pydantic_ai_harness.conversation_search import (
    ConversationSearch,
    ConversationSearchToolset,
    HistorySource,
    SearchScope,
    SnapshotHistorySource,
    SnapshotStore,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    InMemoryStepStore,
    RunRecord,
    SqliteStepStore,
    StepPersistence,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


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
        from pydantic_ai_harness.compaction._summarizing_compaction import _SUMMARY_PREFIX

        store = InMemoryStepStore()
        artifact = ModelRequest(parts=[SystemPromptPart(content=f'{_SUMMARY_PREFIX}older summarized context')])
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

    async def test_list_runs_delegates_to_store(self) -> None:
        store = InMemoryStepStore()
        await store.register_run(RunRecord(run_id='r1'))
        await store.register_run(RunRecord(run_id='r2'))
        assert [r.run_id for r in await SnapshotHistorySource(store).list_runs()] == ['r1', 'r2']

    def test_shipped_stores_satisfy_snapshot_store_protocol(self, tmp_path: Path) -> None:
        assert isinstance(InMemoryStepStore(), SnapshotStore)
        assert isinstance(FileStepStore(tmp_path / 'runs'), SnapshotStore)
        assert isinstance(SqliteStepStore(database=tmp_path / 'runs.db'), SnapshotStore)

    def test_rejects_store_without_snapshot_seam(self) -> None:
        # A store that lists runs but has not implemented `list_snapshots` (for
        # example `MongoStepStore`, pydantic-ai-harness#446) fails at construction
        # with a clear error, not mid-search with an obscure `AttributeError`.
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


class TestConversationSearch:
    async def test_cross_run_recall_through_shared_store(self) -> None:
        store = InMemoryStepStore()
        agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[
                StepPersistence(store=store),
                ConversationSearch(SnapshotHistorySource(store)),
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
                ConversationSearch(SnapshotHistorySource(store)),
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
        await _seed_run(store, 'past-run', [_user('the ZEBRA passphrase is secret')])
        calls: list[int] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not calls:
                calls.append(1)
                assert any(tool.name == 'search_conversation_history' for tool in info.function_tools)
                return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'ZEBRA'})])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[ConversationSearch(SnapshotHistorySource(store))],
        )
        result = await agent.run('find it')
        returned = next(
            str(part.content)
            for message in result.all_messages()
            for part in getattr(message, 'parts', [])
            if isinstance(part, ToolReturnPart) and part.tool_name == 'search_conversation_history'
        )
        assert 'ZEBRA' in returned

    async def test_capability_params_flow_into_the_toolset(self) -> None:
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user(f'needle entry {i} filler') for i in range(5)])
        calls: list[int] = []

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not calls:
                calls.append(1)
                return ModelResponse(parts=[ToolCallPart('search_conversation_history', {'query': 'needle'})])
            return ModelResponse(parts=[TextPart('done')])

        agent: Agent[None, str] = Agent(
            FunctionModel(model_fn),
            capabilities=[ConversationSearch(SnapshotHistorySource(store), max_matches=1, context_lines=0)],
        )
        result = await agent.run('recall')
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
        result = await writer.run('the CRIMSON marker is important')
        result = await writer.run('second turn', message_history=result.all_messages())
        result = await writer.run('third turn', message_history=result.all_messages())
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
            capabilities=[ConversationSearch(SnapshotHistorySource(store))],
        )
        recall = await searcher.run('what was the marker?')
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
        assert ConversationSearch(source).get_instructions() is not None
        assert ConversationSearch(source, add_instructions=False).get_instructions() is None


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

    async def test_default_scope_spans_every_conversation(self) -> None:
        source = SnapshotHistorySource(await self._two_conversation_store())
        rendered = await _search(source, 'passport', conversation_id='conv-bob')
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

    async def test_conversation_scope_is_announced_in_the_tool_description(self) -> None:
        source = SnapshotHistorySource(InMemoryStepStore())
        scoped: ConversationSearchToolset[None] = ConversationSearchToolset(
            source, tool_id='t', max_matches=10, context_lines=5, bm25_k1=1.5, bm25_b=0.75, scope='conversation'
        )
        unscoped: ConversationSearchToolset[None] = ConversationSearchToolset(
            source, tool_id='t', max_matches=10, context_lines=5, bm25_k1=1.5, bm25_b=0.75
        )
        scoped_tools = await scoped.get_tools(_run_context())
        unscoped_tools = await unscoped.get_tools(_run_context())
        scoped_description = scoped_tools['search_conversation_history'].tool_def.description or ''
        unscoped_description = unscoped_tools['search_conversation_history'].tool_def.description or ''
        assert 'only the current conversation' in scoped_description
        assert 'only the current conversation' not in unscoped_description

    async def test_capability_scope_reaches_the_tool(self) -> None:
        """The end-to-end path: `ConversationSearch(scope=...)` through `Agent` to the tool result."""
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

    def test_scope_default_is_store_wide(self) -> None:
        assert ConversationSearch(SnapshotHistorySource(InMemoryStepStore())).scope == 'all'

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
                    RetryPromptPart(content='please retry', tool_name='readfile', tool_call_id='c1'),
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
        assert 'Retry [readfile]: please retry' in rendered  # RetryPromptPart is searchable
        assert '...' in rendered  # truncation applied to the long tool return / args

    async def test_user_and_text_parts_truncate_but_stay_searchable(self) -> None:
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
        # Terms past the 500-char display cutoff still match (the index is
        # untruncated), while the displayed excerpt stays truncated.
        rendered = await _search(source, 'USERTAIL')
        excerpts = rendered.split(':\n\n', 1)[1]
        assert 'Found 1 match(es)' in rendered
        assert 'USERTAIL' not in excerpts
        assert '...' in excerpts
        rendered = await _search(source, 'TEXTTAIL')
        excerpts = rendered.split(':\n\n', 1)[1]
        assert 'Found 1 match(es)' in rendered
        assert 'TEXTTAIL' not in excerpts
        assert '...' in excerpts

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
            ConversationSearch(source, max_matches=-1).get_toolset()
        with pytest.raises(ValueError, match=re.escape('context_lines must be non-negative, got -1.')):
            ConversationSearch(source, context_lines=-1).get_toolset()
        with pytest.raises(ValueError, match=re.escape('bm25_k1 must be a non-negative finite number, got -1.0.')):
            ConversationSearch(source, bm25_k1=-1.0).get_toolset()
        # `nan` and `inf` pass an ordering check and would poison every score instead.
        with pytest.raises(ValueError, match='bm25_k1 must be a non-negative finite number'):
            ConversationSearch(source, bm25_k1=float('nan')).get_toolset()
        with pytest.raises(ValueError, match='bm25_k1 must be a non-negative finite number'):
            ConversationSearch(source, bm25_k1=float('inf')).get_toolset()
        with pytest.raises(ValueError, match=re.escape('bm25_b must be between 0.0 and 1.0, got nan.')):
            ConversationSearch(source, bm25_b=float('nan')).get_toolset()
        with pytest.raises(ValueError, match=re.escape('bm25_b must be between 0.0 and 1.0, got 1.5.')):
            ConversationSearch(source, bm25_b=1.5).get_toolset()
        with pytest.raises(ValueError, match=re.escape('bm25_b must be between 0.0 and 1.0, got -0.5.')):
            ConversationSearch(source, bm25_b=-0.5).get_toolset()

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
            )

    async def test_negative_max_matches_no_longer_uncaps_output(self) -> None:
        # An unvalidated negative bound left `len(results) == max_matches` unsatisfiable,
        # emitting every ranked excerpt instead of capping them.
        store = InMemoryStepStore()
        await _seed_run(store, 'r1', [_user(f'needle entry {i} filler') for i in range(5)])
        with pytest.raises(ValueError, match='max_matches must be non-negative'):
            await _search(SnapshotHistorySource(store), 'needle', max_matches=-1)

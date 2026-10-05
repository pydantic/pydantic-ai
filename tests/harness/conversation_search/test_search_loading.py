"""How much persisted history one `search_conversation_history` call loads.

A conversation-scoped search lists only its conversation's runs, and
`SnapshotHistorySource` reuses a run's reconstructed record until the run gains
or loses a snapshot, rather than reloading and rehashing every snapshot per query.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

import pydantic_ai_harness.conversation_search._source as source_module
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness.conversation_search import (
    ConversationSearchToolset,
    HistorySource,
    SearchScope,
    SnapshotHistorySource,
)
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    InMemoryStepStore,
    RunRecord,
    SqliteStepStore,
)


def _user(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _reply(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


_Store = InMemoryStepStore | FileStepStore | SqliteStepStore


class _RecordingStore:
    """A `SnapshotStore` that records which reads a search makes."""

    def __init__(self, inner: _Store) -> None:
        self.inner = inner
        self.listed_conversations: list[str | None] = []
        self.snapshot_loads: list[str] = []
        self.latest_loads: list[str] = []

    async def list_runs(
        self,
        *,
        parent_run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> list[RunRecord]:
        self.listed_conversations.append(conversation_id)
        return await self.inner.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]:
        self.snapshot_loads.append(run_id)
        return await self.inner.list_snapshots(run_id=run_id)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        self.latest_loads.append(run_id)
        return await self.inner.latest_snapshot(run_id=run_id, include_interrupted=include_interrupted)


STORE_FACTORIES: dict[str, Callable[[Path], _Store]] = {
    'memory': lambda _: InMemoryStepStore(),
    'file': lambda tmp_path: FileStepStore(tmp_path / 'runs'),
    'sqlite': lambda tmp_path: SqliteStepStore(database=tmp_path / 'runs.db'),
}


@pytest.fixture(params=list(STORE_FACTORIES))
def store(request: pytest.FixtureRequest, tmp_path: Path) -> _Store:
    return STORE_FACTORIES[request.param](tmp_path)


@pytest.fixture
def hashed(monkeypatch: pytest.MonkeyPatch) -> list[ModelMessage]:
    """Every message `SnapshotHistorySource` hashes, in order."""
    calls: list[ModelMessage] = []
    real_hash = source_module.message_hash

    def counting_hash(message: ModelMessage) -> str:
        calls.append(message)
        return real_hash(message)

    monkeypatch.setattr(source_module, 'message_hash', counting_hash)
    return calls


async def _save(store: _Store, run_id: str, step_index: int, messages: list[ModelMessage]) -> None:
    await store.save_snapshot(ContinuableSnapshot(run_id=run_id, step_index=step_index, messages=messages))


def _texts(history: list[ModelMessage]) -> list[str]:
    texts: list[str] = []
    for message in history:
        for part in message.parts:
            assert isinstance(part, UserPromptPart | TextPart) and isinstance(part.content, str)
            texts.append(part.content)
    return texts


async def _search(source: HistorySource, query: str, *, scope: SearchScope, conversation_id: str | None = None) -> str:
    toolset: ConversationSearchToolset[None] = ConversationSearchToolset(
        source, tool_id='conversation-search', max_matches=10, context_lines=0, bm25_k1=1.5, bm25_b=0.75, scope=scope
    )
    ctx = RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        conversation_id=conversation_id,
    )
    return await toolset.search_conversation_history(ctx, query)


class TestConversationFilter:
    async def test_conversation_scope_lists_only_its_conversation(self) -> None:
        inner = InMemoryStepStore()
        await inner.register_run(RunRecord(run_id='mine', conversation_id='c1'))
        await inner.register_run(RunRecord(run_id='theirs', conversation_id='c2'))
        await _save(inner, 'mine', 0, [_user('ZEBRA mine')])
        await _save(inner, 'theirs', 0, [_user('ZEBRA theirs')])
        store = _RecordingStore(inner)

        rendered = await _search(SnapshotHistorySource(store), 'ZEBRA', scope='conversation', conversation_id='c1')

        assert 'ZEBRA mine' in rendered
        assert 'theirs' not in rendered
        assert store.listed_conversations == ['c1']
        assert store.snapshot_loads == ['mine']

    async def test_all_scope_lists_every_run(self) -> None:
        store = _RecordingStore(InMemoryStepStore())
        await _search(SnapshotHistorySource(store), 'ZEBRA', scope='all')
        assert store.listed_conversations == [None]

    async def test_source_ignoring_the_filter_still_cannot_cross_conversations(self) -> None:
        class _UnfilteredSource:
            async def list_runs(self, **_: object) -> list[RunRecord]:
                return [
                    RunRecord(run_id='mine', conversation_id='c1'),
                    RunRecord(run_id='theirs', conversation_id='c2'),
                ]

            async def run_history(self, *, run_id: str) -> list[ModelMessage]:
                return [_user(f'ZEBRA {run_id}')]

        rendered = await _search(_UnfilteredSource(), 'ZEBRA', scope='conversation', conversation_id='c1')
        assert 'ZEBRA mine' in rendered
        assert 'theirs' not in rendered

    async def test_legacy_source_warns_and_is_filtered_by_the_toolset(self) -> None:
        class _LegacySource:
            async def list_runs(self) -> list[RunRecord]:
                return [
                    RunRecord(run_id='mine', conversation_id='c1'),
                    RunRecord(run_id='theirs', conversation_id='c2'),
                ]

            async def run_history(self, *, run_id: str) -> list[ModelMessage]:
                return [_user(f'ZEBRA {run_id}')]

        with pytest.warns(
            HarnessDeprecationWarning, match=r'`_LegacySource.list_runs\(\)` does not accept `conversation_id=`'
        ):
            rendered = await _search(_LegacySource(), 'ZEBRA', scope='conversation', conversation_id='c1')  # pyright: ignore[reportArgumentType]
        assert 'ZEBRA mine' in rendered
        assert 'theirs' not in rendered

    async def test_positional_only_conversation_id_counts_as_legacy(self) -> None:
        class _PositionalSource:
            async def list_runs(self, conversation_id: str | None = None, /) -> list[RunRecord]:
                return [RunRecord(run_id='mine', conversation_id='c1')]

            async def run_history(self, *, run_id: str) -> list[ModelMessage]:
                return [_user(f'ZEBRA {run_id}')]

        with pytest.warns(HarnessDeprecationWarning, match='does not accept `conversation_id=`'):
            rendered = await _search(_PositionalSource(), 'ZEBRA', scope='conversation', conversation_id='c1')  # pyright: ignore[reportArgumentType]
        assert 'ZEBRA mine' in rendered


class TestReconstructionCache:
    async def test_unchanged_run_is_not_reloaded(self, store: _Store) -> None:
        await store.register_run(RunRecord(run_id='r1', conversation_id='c1'))
        question = _user('ZEBRA question')
        await _save(store, 'r1', 0, [question])
        await _save(store, 'r1', 1, [question, _reply('ZEBRA answer')])
        recording = _RecordingStore(store)
        source = SnapshotHistorySource(recording)

        first = await source.run_history(run_id='r1')
        second = await source.run_history(run_id='r1')

        assert second == first
        assert _texts(second) == ['ZEBRA question', 'ZEBRA answer']
        # The second call confirms the newest snapshot is unchanged instead of reloading all of them.
        assert recording.snapshot_loads == ['r1']
        assert recording.latest_loads == ['r1', 'r1']

    async def test_appended_snapshots_are_folded_in_incrementally(
        self, store: _Store, hashed: list[ModelMessage]
    ) -> None:
        await store.register_run(RunRecord(run_id='r1'))
        turn_one: list[ModelMessage] = [_user('first'), _reply('one')]
        await _save(store, 'r1', 0, turn_one)
        source = SnapshotHistorySource(store)
        await source.run_history(run_id='r1')

        turn_two: list[ModelMessage] = [*turn_one, _user('second'), _reply('two')]
        await _save(store, 'r1', 1, turn_two)

        hashed.clear()
        history = await source.run_history(run_id='r1')

        # Only the appended snapshot is hashed; the first one is not reprocessed.
        assert len(hashed) == len(turn_two)
        assert _texts(history) == ['first', 'one', 'second', 'two']
        assert history == await SnapshotHistorySource(store, max_cached_runs=0).run_history(run_id='r1')

    async def test_pruned_snapshots_rebuild_from_scratch(self) -> None:
        store = InMemoryStepStore(max_snapshots_per_run=1)
        await store.register_run(RunRecord(run_id='r1'))
        await _save(store, 'r1', 0, [_user('ZEBRA original')])
        source = SnapshotHistorySource(store)
        assert len(await source.run_history(run_id='r1')) == 1

        # Retention drops the first snapshot on this save, so the cached record no longer
        # matches the store's snapshots and is rebuilt rather than extended.
        await _save(store, 'r1', 1, [_user('replacement')])

        history = await source.run_history(run_id='r1')
        assert history == await SnapshotHistorySource(store, max_cached_runs=0).run_history(run_id='r1')
        assert _texts(history) == ['replacement']

    async def test_interrupted_save_that_prunes_complete_snapshots_rebuilds(self) -> None:
        store = InMemoryStepStore(max_snapshots_per_run=2)
        await store.register_run(RunRecord(run_id='r1'))
        await _save(store, 'r1', 0, [_user('ZEBRA original')])
        await _save(store, 'r1', 1, [_user('replacement')])
        source = SnapshotHistorySource(store)
        assert _texts(await source.run_history(run_id='r1')) == ['ZEBRA original', 'replacement']

        # The newest `complete` snapshot is unchanged, but this `interrupted` save makes
        # retention prune the first one, so the cached record must not be served.
        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=2, messages=[_user('unsettled')], state='interrupted')
        )

        history = await source.run_history(run_id='r1')
        assert _texts(history) == ['replacement']
        assert history == await SnapshotHistorySource(store, max_cached_runs=0).run_history(run_id='r1')

    async def test_run_without_snapshots_is_not_cached(self) -> None:
        store = _RecordingStore(InMemoryStepStore())
        source = SnapshotHistorySource(store)
        assert await source.run_history(run_id='missing') == []
        assert await source.run_history(run_id='missing') == []
        assert store.snapshot_loads == ['missing', 'missing']

    async def test_vanished_snapshots_drop_the_cached_run(self) -> None:
        inner = InMemoryStepStore()
        await inner.register_run(RunRecord(run_id='r1'))
        await _save(inner, 'r1', 0, [_user('ZEBRA')])
        source = SnapshotHistorySource(inner)
        assert len(await source.run_history(run_id='r1')) == 1

        inner._snapshots.clear()  # pyright: ignore[reportPrivateUsage]

        assert await source.run_history(run_id='r1') == []

    async def test_least_recently_searched_run_is_evicted(self) -> None:
        inner = InMemoryStepStore()
        for run_id in ('r1', 'r2'):
            await inner.register_run(RunRecord(run_id=run_id))
            await _save(inner, run_id, 0, [_user(run_id)])
        store = _RecordingStore(inner)
        source = SnapshotHistorySource(store, max_cached_runs=1)

        await source.run_history(run_id='r1')
        await source.run_history(run_id='r2')
        await source.run_history(run_id='r2')
        await source.run_history(run_id='r1')

        assert store.snapshot_loads == ['r1', 'r2', 'r1']

    async def test_zero_disables_the_cache(self) -> None:
        inner = InMemoryStepStore()
        await inner.register_run(RunRecord(run_id='r1'))
        await _save(inner, 'r1', 0, [_user('ZEBRA')])
        store = _RecordingStore(inner)
        source = SnapshotHistorySource(store, max_cached_runs=0)

        await source.run_history(run_id='r1')
        await source.run_history(run_id='r1')

        assert store.snapshot_loads == ['r1', 'r1']
        assert store.latest_loads == []

    async def test_store_without_latest_snapshot_still_folds_in_only_new_snapshots(
        self, hashed: list[ModelMessage]
    ) -> None:
        class _NoLatestStore:
            """A `SnapshotStore` without `latest_snapshot`, so every call relists snapshots."""

            def __init__(self, inner: InMemoryStepStore) -> None:
                self.inner = inner
                self.snapshot_loads = 0

            async def list_runs(
                self,
                *,
                parent_run_id: str | None = None,
                conversation_id: str | None = None,
            ) -> list[RunRecord]:
                return await self.inner.list_runs(parent_run_id=parent_run_id, conversation_id=conversation_id)

            async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]:
                self.snapshot_loads += 1
                return await self.inner.list_snapshots(run_id=run_id)

        inner = InMemoryStepStore()
        await inner.register_run(RunRecord(run_id='r1'))
        question = _user('ZEBRA')
        await _save(inner, 'r1', 0, [question])
        store = _NoLatestStore(inner)
        source = SnapshotHistorySource(store)
        await source.run_history(run_id='r1')
        assert hashed == [question]

        assert _texts(await source.run_history(run_id='r1')) == ['ZEBRA']
        assert hashed == [question]
        assert store.snapshot_loads == 2
        assert [run.run_id for run in await source.list_runs()] == ['r1']

    def test_rejects_negative_bound(self) -> None:
        with pytest.raises(ValueError, match='max_cached_runs must be non-negative'):
            SnapshotHistorySource(InMemoryStepStore(), max_cached_runs=-1)

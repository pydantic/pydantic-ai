"""Tests for the SQLite FTS5 index `ConversationSearch` keeps beside a `SqliteStepStore`.

The index must give the same answers as the scan it replaces, so most tests compare a
search over a `SqliteStepStore` with the same search over an `InMemoryStepStore` seeded
identically, which always scans.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import anyio
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NoOpTracer, Tracer

from pydantic_ai.messages import (
    AgentInstructionSource,
    InstructionDeltaPart,
    InstructionId,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.conversation_search import ConversationSearchToolset, SearchScope, SnapshotHistorySource
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, InMemoryStepStore, RunRecord, SqliteStepStore

_LOGGER = 'pydantic_ai_harness.conversation_search._toolset'


def _user(content: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(content=content)])


def _reply(content: str) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content=content)])


def _started(minute: int) -> datetime:
    """Distinct start times, so both stores list runs in the same order."""
    return datetime(2026, 1, 1, 0, minute, tzinfo=UTC)


async def _seed_run(
    store: InMemoryStepStore | SqliteStepStore,
    run_id: str,
    *snapshots: list[ModelMessage],
    conversation_id: str | None = None,
    minute: int = 0,
) -> None:
    await store.register_run(RunRecord(run_id=run_id, conversation_id=conversation_id, started_at=_started(minute)))
    for step_index, messages in enumerate(snapshots):
        await store.save_snapshot(ContinuableSnapshot(run_id=run_id, step_index=step_index, messages=messages))


async def _seed(store: InMemoryStepStore | SqliteStepStore) -> None:
    """Two conversations exercising dedup, instruction changes, clipping, and an empty run."""
    user = _user('the ZEBRA passphrase is 42, and snake_case names stay whole')
    call = ModelResponse(parts=[ToolCallPart(tool_name='finish', args={'task': 'zebra'}, tool_call_id='t1')])
    tool_return = ModelRequest(parts=[ToolReturnPart(tool_name='finish', content='ok', tool_call_id='t1')])
    changed = replace(
        tool_return,
        parts=[
            *tool_return.parts,
            InstructionDeltaPart(id=InstructionId(AgentInstructionSource(), name='state'), content='status FINISHED'),
        ],
    )
    long_text = 'filler ' * 120 + 'the café OCELOT sits past the cutoff ' + 'tail ' * 40
    await _seed_run(store, 'r1', [user, call], [user, call, tool_return], conversation_id='c1', minute=1)
    # `r2` continues `r1`: its inherited head stays out, its instruction change stays in.
    await _seed_run(store, 'r2', [user, call, changed, _reply(long_text)], conversation_id='c1', minute=2)
    await _seed_run(
        store, 'r3', [user, call, changed, _reply(long_text), _user('and zebra again')], conversation_id='c1', minute=3
    )
    await _seed_run(store, 'empty', conversation_id='c1', minute=4)
    # Characters inside a word that FTS5's tokenizer keeps but the scan's splits on: a combining
    # accent, as macOS file names carry it, and a private-use character.
    await _seed_run(
        store,
        'decomposed',
        [_user('open nai\u0308ve cafe\u0301.txt and foo\ue000bar, then _ = 1')],
        conversation_id='c1',
        minute=6,
    )
    await _seed_run(
        store,
        'other',
        [_user('a zebra secret from another conversation'), _reply('zebra zebra zebra ok')],
        conversation_id='c2',
        minute=5,
    )


def _toolset(
    store: InMemoryStepStore | SqliteStepStore, *, scope: SearchScope = 'all', context_lines: int = 5
) -> ConversationSearchToolset[None]:
    return ConversationSearchToolset(
        SnapshotHistorySource(store),
        tool_id='conversation-search',
        max_matches=10,
        context_lines=context_lines,
        bm25_k1=1.5,
        bm25_b=0.75,
        scope=scope,
    )


def _ctx(conversation_id: str | None = None, tracer: Tracer | None = None) -> RunContext[None]:
    return RunContext[None](
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
        conversation_id=conversation_id,
        tracer=tracer if tracer is not None else NoOpTracer(),
    )


async def _search(
    toolset: ConversationSearchToolset[None],
    query: str,
    *,
    conversation_id: str | None = None,
    run_id: str | None = None,
    tracer: Tracer | None = None,
) -> str:
    return await toolset.search_conversation_history(_ctx(conversation_id, tracer), query, run_id=run_id)


def _rows(database: Path, sql: str, parameters: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    conn = sqlite3.connect(database)
    try:
        return conn.execute(sql, parameters).fetchall()
    finally:
        conn.close()


def _recording_tracer() -> tuple[Tracer, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer('test'), exporter


_QUERIES = (
    'ZEBRA',
    'zebra passphrase',
    'ok',
    'FINISHED',
    'ocelot',
    'cafe',
    'café',
    'snake_case',
    'snake',
    'nai',
    'foo',
    '_',
    'and',
    'NEAR(zebra ok)',
    'zebra*',
    'text:zebra',
    '^zebra',
    'zebra OR -ok',
    '"',
    '"zebra',
    "zebra's",
    '',
    '   ',
    'nothing-matches',
)


class TestFullTextIndex:
    async def test_index_answers_match_the_scan(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Every query, scope, `run_id`, and context width answers exactly as the scan does.

        The queries include FTS5 query syntax, which must be searched as text, not interpreted.
        """
        memory = InMemoryStepStore()
        sqlite = SqliteStepStore(database=tmp_path / 'sessions.db')
        await _seed(memory)
        await _seed(sqlite)

        cases: list[tuple[SearchScope, str | None, str | None]] = [
            ('all', None, None),
            ('conversation', 'c1', None),
            ('conversation', 'c2', None),
            ('conversation', 'c1', 'r3'),
            ('conversation', 'c1', 'other'),
            ('all', None, 'empty'),
        ]
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            for scope, conversation_id, run_id in cases:
                for context_lines in (0, 2):
                    scanned = _toolset(memory, scope=scope, context_lines=context_lines)
                    indexed = _toolset(sqlite, scope=scope, context_lines=context_lines)
                    for query in _QUERIES:
                        expected = await _search(scanned, query, conversation_id=conversation_id, run_id=run_id)
                        actual = await _search(indexed, query, conversation_id=conversation_id, run_id=run_id)
                        assert actual == expected, (scope, conversation_id, run_id, context_lines, query)

        assert caplog.records == []
        indexed_runs = _rows(tmp_path / 'sessions.db', 'SELECT run_id FROM conversation_search_runs ORDER BY run_id')
        assert indexed_runs == [('decomposed',), ('empty',), ('other',), ('r1',), ('r2',), ('r3',)]

    async def test_other_conversations_neither_appear_nor_move_the_scores(self, tmp_path: Path) -> None:
        """FTS5's own `bm25()` would take statistics from every conversation in the table."""
        alone = SqliteStepStore(database=tmp_path / 'alone.db')
        shared = SqliteStepStore(database=tmp_path / 'shared.db')
        for store in (alone, shared):
            await _seed_run(store, 'mine', [_user('the HERON plan'), _reply('noted')], conversation_id='c1', minute=1)
        await _seed_run(
            shared,
            'theirs',
            [_user('heron heron SECRET'), *[_user(f'heron note {i}') for i in range(20)]],
            conversation_id='c2',
            minute=2,
        )

        expected = await _search(_toolset(alone, scope='conversation'), 'heron secret', conversation_id='c1')
        actual = await _search(_toolset(shared, scope='conversation'), 'heron secret', conversation_id='c1')

        assert actual == expected
        assert 'SECRET' not in actual
        assert 'theirs' not in actual
        out_of_scope = await _search(
            _toolset(shared, scope='conversation'), 'heron', conversation_id='c1', run_id='theirs'
        )
        assert out_of_scope == "No persisted history for run 'theirs'."

    async def test_new_snapshots_and_runs_reach_the_next_search(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'sessions.db')
        toolset = _toolset(store)
        await _seed_run(store, 'r1', [_user('first turn')], minute=1)
        assert 'No matches' in await _search(toolset, 'pelican')

        await store.save_snapshot(
            ContinuableSnapshot(run_id='r1', step_index=1, messages=[_user('first turn'), _reply('a PELICAN')])
        )
        await _seed_run(store, 'r2', [_user('another pelican')], minute=2)

        rendered = await _search(toolset, 'pelican', run_id=None)
        assert rendered.count('[score:') == 2
        assert 'a PELICAN' in rendered
        assert 'another pelican' in rendered

    async def test_pruned_snapshots_leave_the_index(self, tmp_path: Path) -> None:
        store = SqliteStepStore(database=tmp_path / 'sessions.db', max_snapshots_per_run=1)
        toolset = _toolset(store)
        await _seed_run(store, 'r1', [_user('the IBIS original')], minute=1)
        assert 'the IBIS original' in await _search(toolset, 'ibis')

        await store.save_snapshot(ContinuableSnapshot(run_id='r1', step_index=1, messages=[_user('a summary')]))

        assert 'No matches' in await _search(toolset, 'ibis')

    async def test_deleting_a_run_deletes_its_indexed_text(self, tmp_path: Path) -> None:
        database = tmp_path / 'sessions.db'
        store = SqliteStepStore(database=database)
        await _seed_run(store, 'r1', [_user('a private LYNX detail')], conversation_id='c1', minute=1)
        await _seed_run(store, 'r2', [_user('a public lynx detail')], conversation_id='c2', minute=2)
        assert (await _search(_toolset(store), 'lynx')).count('[score:') == 2

        conn = sqlite3.connect(database)
        with conn:
            conn.execute("DELETE FROM snapshots WHERE run_id = 'r1'")
            conn.execute("DELETE FROM runs WHERE run_id = 'r1'")
        conn.close()

        assert _rows(database, "SELECT COUNT(*) FROM conversation_search_messages WHERE run_id = 'r1'") == [(0,)]
        assert _rows(database, "SELECT COUNT(*) FROM conversation_search_runs WHERE run_id = 'r1'") == [(0,)]
        assert _rows(
            database, 'SELECT COUNT(*) FROM conversation_search_fts WHERE conversation_search_fts MATCH ?', ('private',)
        ) == [(0,)]
        assert 'private' not in await _search(_toolset(store), 'lynx')

    async def test_an_existing_database_is_indexed_once(self, tmp_path: Path) -> None:
        """A database written before the index existed is indexed by its first search, and only then."""
        database = tmp_path / 'sessions.db'
        await _seed(SqliteStepStore(database=database))
        memory = InMemoryStepStore()
        await _seed(memory)
        expected = await _search(_toolset(memory), 'zebra')

        # Each toolset over its own store instance stands in for another process on the same file.
        assert await _search(_toolset(SqliteStepStore(database=database)), 'zebra') == expected
        rows = _rows(database, 'SELECT id, run_id, position FROM conversation_search_messages ORDER BY id')
        assert await _search(_toolset(SqliteStepStore(database=database)), 'zebra') == expected
        assert _rows(database, 'SELECT id, run_id, position FROM conversation_search_messages ORDER BY id') == rows

        # Rows written by a build with another rendering are rebuilt.
        conn = sqlite3.connect(database)
        with conn:
            conn.execute('UPDATE conversation_search_runs SET format_version = 0')
        conn.close()
        assert await _search(_toolset(SqliteStepStore(database=database)), 'zebra') == expected
        rebuilt = _rows(database, 'SELECT id, run_id, position FROM conversation_search_messages ORDER BY id')
        assert [row[1:] for row in rebuilt] == [row[1:] for row in rows]
        assert rebuilt != rows

    async def test_concurrent_searches_share_one_database(self, tmp_path: Path) -> None:
        database = tmp_path / 'sessions.db'
        await _seed(SqliteStepStore(database=database))
        memory = InMemoryStepStore()
        await _seed(memory)
        expected = await _search(_toolset(memory), 'zebra ok')
        results: list[str] = []

        async def search() -> None:
            results.append(await _search(_toolset(SqliteStepStore(database=database)), 'zebra ok'))

        async with anyio.create_task_group() as tg:
            for _ in range(4):
                tg.start_soon(search)

        assert results == [expected] * 4

    async def test_a_run_indexed_meanwhile_is_kept(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Another process indexing the same run first leaves its rows in place."""
        database = tmp_path / 'sessions.db'
        store = SqliteStepStore(database=database)
        await _seed_run(store, 'r1', [_user('the MOOSE note')], minute=1)
        original = SnapshotHistorySource.run_history
        raced: list[bool] = []

        async def run_history(source: SnapshotHistorySource, *, run_id: str) -> list[ModelMessage]:
            if not raced:
                raced.append(True)
                # The other process indexes the run between this search's check and its write.
                await _search(_toolset(SqliteStepStore(database=database)), 'moose')
                conn = sqlite3.connect(database)
                with conn:
                    conn.execute("UPDATE conversation_search_messages SET hash = 'written-first'")
                conn.close()
            return await original(source, run_id=run_id)

        monkeypatch.setattr(SnapshotHistorySource, 'run_history', run_history)

        assert 'the MOOSE note' in await _search(_toolset(store), 'moose')
        assert _rows(database, 'SELECT run_id, hash FROM conversation_search_messages') == [('r1', 'written-first')]

    async def test_a_newer_index_is_not_overwritten_by_an_older_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A search that loaded a run before a newer snapshot keeps the newer rows another process wrote."""
        database = tmp_path / 'sessions.db'
        store = SqliteStepStore(database=database)
        await _seed_run(store, 'r1', [_user('the first turn')], minute=1)
        original = SnapshotHistorySource.run_history
        raced: list[bool] = []

        async def run_history(source: SnapshotHistorySource, *, run_id: str) -> list[ModelMessage]:
            history = await original(source, run_id=run_id)
            if not raced:
                raced.append(True)
                await store.save_snapshot(
                    ContinuableSnapshot(run_id='r1', step_index=1, messages=[*history, _reply('the YAK reply')])
                )
                await _search(_toolset(SqliteStepStore(database=database)), 'yak')
            return history

        monkeypatch.setattr(SnapshotHistorySource, 'run_history', run_history)

        assert 'the YAK reply' in await _search(_toolset(store), 'yak')

    async def test_a_run_deleted_while_indexing_stays_out(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        database = tmp_path / 'sessions.db'
        store = SqliteStepStore(database=database)
        await _seed_run(store, 'r1', [_user('the BISON note')], minute=1)
        original = SnapshotHistorySource.run_history

        async def run_history(source: SnapshotHistorySource, *, run_id: str) -> list[ModelMessage]:
            history = await original(source, run_id=run_id)
            conn = sqlite3.connect(database)
            with conn:
                conn.execute('DELETE FROM runs WHERE run_id = ?', (run_id,))
            conn.close()
            return history

        monkeypatch.setattr(SnapshotHistorySource, 'run_history', run_history)

        assert 'No persisted conversation history' in await _search(_toolset(store), 'bison')
        assert _rows(database, 'SELECT COUNT(*) FROM conversation_search_messages') == [(0,)]

    async def test_a_subclassed_source_or_store_scans(self, tmp_path: Path) -> None:
        """A subclass can change what a run's history holds, so it never shares the index."""
        database = tmp_path / 'sessions.db'
        store = SqliteStepStore(database=database)
        await _seed_run(store, 'r1', [_user('the OTTER password is hunter2')], minute=1)
        assert 'hunter2' in await _search(_toolset(store), 'otter')

        class _RedactingSource(SnapshotHistorySource):
            async def run_history(self, *, run_id: str) -> list[ModelMessage]:
                return [_user('the OTTER password is [redacted]') for _ in await super().run_history(run_id=run_id)]

        class _Store(SqliteStepStore):
            pass

        redacting = ConversationSearchToolset[None](
            _RedactingSource(store), tool_id='t', max_matches=10, context_lines=5, bm25_k1=1.5, bm25_b=0.75, scope='all'
        )
        assert '[redacted]' in await _search(redacting, 'otter')
        assert 'hunter2' not in await _search(redacting, 'otter')

        other = _Store(database=tmp_path / 'other.db')
        await _seed_run(other, 'r1', [_user('an OTTER')], minute=1)
        assert 'an OTTER' in await _search(_toolset(other), 'otter')
        assert (
            _rows(tmp_path / 'other.db', "SELECT name FROM sqlite_master WHERE name LIKE 'conversation_search%'") == []
        )


def _option_unused(_name: str) -> int:
    return 0


class TestIndexFallback:
    async def test_sqlite_without_fts5_scans_instead(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        database = tmp_path / 'sessions.db'
        conn = sqlite3.connect(database, check_same_thread=False, isolation_level=None)
        # Report a build without FTS5, as some system Pythons ship.
        conn.create_function('sqlite_compileoption_used', 1, _option_unused)
        store = SqliteStepStore(connection=conn)
        await _seed(store)
        memory = InMemoryStepStore()
        await _seed(memory)
        toolset = _toolset(store)
        tracer, exporter = _recording_tracer()

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert await _search(toolset, 'zebra', tracer=tracer) == await _search(_toolset(memory), 'zebra')
            assert await _search(toolset, 'ocelot', tracer=tracer) == await _search(_toolset(memory), 'ocelot')

        # Detected once per toolset, not retried on every search.
        assert [record.getMessage() for record in caplog.records] == [
            'Conversation search index unavailable, scanning persisted history instead: '
            'the SQLite library was built without FTS5'
        ]
        (span,) = exporter.get_finished_spans()
        assert span.name == 'conversation_search index fallback'
        assert dict(span.attributes or {}) == {'conversation_search.fallback_reason': 'Fts5Unavailable'}
        assert conn.execute("SELECT name FROM sqlite_master WHERE name LIKE 'conversation_search%'").fetchall() == []
        conn.close()

    async def test_a_locked_database_scans_instead(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        database = tmp_path / 'sessions.db'
        conn = sqlite3.connect(database, check_same_thread=False, isolation_level=None, timeout=0)
        conn.execute('PRAGMA journal_mode=WAL')
        store = SqliteStepStore(connection=conn)
        await _seed(store)
        memory = InMemoryStepStore()
        await _seed(memory)
        expected = await _search(_toolset(memory), 'zebra')
        toolset = _toolset(store)

        writer = sqlite3.connect(database, isolation_level=None)
        writer.execute('BEGIN IMMEDIATE')
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert await _search(toolset, 'zebra') == expected
        assert [record.getMessage() for record in caplog.records] == [
            'Conversation search index unavailable, scanning persisted history instead: database is locked'
        ]
        writer.execute('ROLLBACK')
        writer.close()

        # Once the lock is gone the same toolset builds and uses the index.
        assert await _search(toolset, 'zebra') == expected
        assert conn.execute('SELECT COUNT(*) FROM conversation_search_runs').fetchone() == (6,)
        conn.close()

    @pytest.mark.parametrize('open_transaction', [False, True])
    async def test_a_broken_index_scans_instead(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, open_transaction: bool
    ) -> None:
        """A failed index write rolls back, including inside a transaction the caller holds open."""
        database = tmp_path / 'sessions.db'
        conn = sqlite3.connect(database, check_same_thread=False)
        # Occupies the FTS5 table's name, so writing to the index fails.
        conn.execute('CREATE TABLE conversation_search_fts (unrelated TEXT)')
        conn.commit()
        store = SqliteStepStore(connection=conn)
        await _seed(store)
        memory = InMemoryStepStore()
        await _seed(memory)
        if open_transaction:
            conn.execute("INSERT INTO conversation_search_fts VALUES ('caller work')")
        else:
            conn.commit()
        assert conn.in_transaction is open_transaction

        toolset = _toolset(store)
        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            for _ in range(2):
                assert await _search(toolset, 'zebra') == await _search(_toolset(memory), 'zebra')

        # Retried on each search, since the cause may pass; nothing half-written is left.
        assert len(caplog.records) == 2
        assert conn.in_transaction is open_transaction
        assert conn.execute('SELECT COUNT(*) FROM conversation_search_messages').fetchone() == (0,)
        assert conn.execute('SELECT COUNT(*) FROM conversation_search_fts').fetchone() == (int(open_transaction),)
        conn.close()

    async def test_an_index_rolled_back_with_the_caller_is_rebuilt(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        database = tmp_path / 'sessions.db'
        conn = sqlite3.connect(database, check_same_thread=False)
        store = SqliteStepStore(connection=conn)
        await _seed_run(store, 'r1', [_user('the GECKO note')], minute=1)
        conn.commit()
        toolset = _toolset(store)

        conn.execute('BEGIN')
        assert 'the GECKO note' in await _search(toolset, 'gecko')
        conn.rollback()

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            assert 'the GECKO note' in await _search(toolset, 'gecko')
        assert caplog.records == []
        assert _rows(database, 'SELECT run_id FROM conversation_search_runs') == [('r1',)]
        conn.close()

    async def test_an_open_caller_transaction_is_joined_not_committed(self, tmp_path: Path) -> None:
        database = tmp_path / 'sessions.db'
        conn = sqlite3.connect(database, check_same_thread=False)
        store = SqliteStepStore(connection=conn)
        await _seed_run(store, 'r1', [_user('the WALRUS note')], minute=1)
        assert conn.in_transaction

        assert 'the WALRUS note' in await _search(_toolset(store), 'walrus')

        assert conn.in_transaction
        assert _rows(database, 'SELECT COUNT(*) FROM runs') == [(0,)]
        conn.commit()
        assert _rows(database, 'SELECT COUNT(*) FROM conversation_search_messages') == [(1,)]
        conn.close()

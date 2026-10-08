"""Hackathon sessions in Logfire: chunk spans per run, read back with checks, continue or fork on resume.

The query API is replaced by `SpanQuery`, which answers the module's SQL from the spans the test exported, so
writing and reading meet exactly as they would in Logfire, without a live project.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.test import TestModel
from pydantic_clai2.builtin_plugins import logfire_sessions
from pydantic_clai2.builtin_plugins.logfire_sessions import (
    SPAN_NAME,
    ChunkStates,
    LogfireQuery,
    LogfireSessions,
    SessionChunks,
    SessionCorrupt,
    privacy_notice,
    trace_from_link,
    trace_link,
    without_binary,
)

ALICE = 'alice@example.com'
BOB = 'bob@example.com'


class SpanQuery(LogfireQuery):
    """Answers `LogfireSessions`' queries from exported spans, as Logfire's `records` table would."""

    def __init__(self, exporter: InMemorySpanExporter) -> None:
        super().__init__(base_url='https://logfire.example', key='test')
        self.exporter = exporter
        self.edit: dict[tuple[str, int, int], dict[str, Any]] = {}
        """Rows to change, by (session, seq, part), to simulate damage in storage."""
        self.drop: set[tuple[str, int, int]] = set()

    def _records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for span in self.exporter.get_finished_spans():
            if span.name != SPAN_NAME:
                continue
            attributes = dict(span.attributes or {})
            record: dict[str, Any] = {
                name.removeprefix('clai2.session.'): attributes[name]
                for name in attributes
                if name.startswith('clai2.session.')
            }
            record['trace_id'] = f'{span.context.trace_id:032x}' if span.context else ''
            record['repo'] = attributes.get('clai2.repo_slug')
            record['start_timestamp'] = str(span.start_time)
            key = (str(record['id']), int(record['seq']), int(record['part']))
            if key in self.drop:
                continue
            records.append({**record, **self.edit.get(key, {})})
        return records

    async def rows(self, sql: str, *, limit: int = 10_000) -> list[dict[str, Any]]:
        records = self._records()
        if match := re.search(r"trace_id = '([0-9a-f]{32})'", sql):
            return [record for record in records if record['trace_id'] == match.group(1)][:limit]
        if match := re.search(r"'clai2.session.owner' = '([^']+)'", sql):
            return [
                record for record in reversed(records) if record.get('owner') == match.group(1) and record['part'] == 0
            ][:limit]
        session = re.search(r"'clai2.session.id' = '([^']+)'", sql)
        assert session is not None
        upto = re.search(r'<= (\d+)', sql)
        return [
            {**record, 'seq': str(record['seq'])}
            for record in records
            if record['id'] == session.group(1) and (upto is None or record['seq'] <= int(upto.group(1)))
        ][:limit]


class Machine:
    """One clai2 on one machine: its chunk state, its signed-in user, and an agent that writes chunks."""

    def __init__(self, tmp_path: Path, name: str, owner: str, provider: TracerProvider, query: SpanQuery) -> None:
        self.owner = owner
        self.session_id = str(uuid4())
        self.states = ChunkStates(tmp_path / name / 'session_chunks.json')
        self.chunks = SessionChunks(
            tracer_provider=provider,
            states=self.states,
            session_id=lambda: self.session_id,
            owner=lambda: self.owner,
            attributes=lambda: {'clai2.repo_slug': 'acme/widgets'},
        )
        self.agent = Agent(
            TestModel(custom_output_text='ok'),
            capabilities=[
                Instrumentation(settings=InstrumentationSettings(tracer_provider=provider)),
                self.chunks,
            ],
        )
        self.sessions = LogfireSessions(query=query, states=self.states, owner=lambda: self.owner, project='acme/clai2')
        self.messages: list[ModelMessage] = []

    async def prompt(self, text: str) -> None:
        result = await self.agent.run(text, message_history=self.messages)
        self.messages = result.all_messages()


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def query(exporter: InMemorySpanExporter) -> SpanQuery:
    return SpanQuery(exporter)


def chunk_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == SPAN_NAME]


async def test_each_run_writes_its_new_messages_under_the_run_span(
    tmp_path: Path, provider: TracerProvider, exporter: InMemorySpanExporter, query: SpanQuery
) -> None:
    alice = Machine(tmp_path, 'a', ALICE, provider, query)
    await alice.prompt('Fix the flaky test')
    await alice.prompt('Now add a changelog entry')
    spans = chunk_spans(exporter)
    runs = {
        span.context.span_id: span for span in exporter.get_finished_spans() if span.name.startswith('invoke_agent')
    }
    assert all(span.parent is not None and span.parent.span_id in runs for span in spans)
    manifests = [
        {key: value for key, value in (span.attributes or {}).items() if key != 'clai2.session.payload'}
        for span in spans
    ]
    assert [(m['clai2.session.seq'], m['clai2.session.reset'], m['clai2.session.messages']) for m in manifests] == [
        (0, True, 2),
        (1, False, 4),
    ]
    assert manifests[0]['clai2.session.owner'] == ALICE
    assert manifests[0]['clai2.session.first_prompt'] == 'Fix the flaky test'
    assert manifests[1]['clai2.repo_slug'] == 'acme/widgets'

    loaded = await alice.sessions.load(alice.session_id)
    assert loaded is not None and loaded.messages == alice.messages and loaded.last_seq == 1

    # A history that no longer starts with what was stored (after /compact) is written whole.
    alice.messages = alice.messages[2:]
    await alice.prompt('And the docs')
    assert chunk_spans(exporter)[-1].attributes['clai2.session.reset'] is True  # pyright: ignore[reportOptionalSubscript]
    loaded = await alice.sessions.load(alice.session_id)
    assert loaded is not None and loaded.messages == alice.messages

    [listed] = await alice.sessions.listed()
    assert (listed.session_id, listed.runs, listed.repo) == (alice.session_id, 3, 'acme/widgets')
    assert await alice.sessions.session_for('not an id!') is None


async def test_your_session_continues_in_place_and_someone_elses_forks(
    tmp_path: Path, provider: TracerProvider, exporter: InMemorySpanExporter, query: SpanQuery
) -> None:
    alice = Machine(tmp_path, 'a', ALICE, provider, query)
    await alice.prompt('Fix the flaky test')
    trace = f'{chunk_spans(exporter)[0].context.trace_id:032x}'  # pyright: ignore[reportOptionalMemberAccess]
    link = trace_link('https://logfire.example', 'acme/clai2', trace)
    assert trace_from_link(link) == trace

    # The same person on a laptop that never saw it: continue in place, as the next run.
    laptop = Machine(tmp_path, 'laptop', ALICE, provider, query)
    resumed = await laptop.sessions.resume(link, local=False)
    assert resumed is not None and resumed.conversation_id == alice.session_id
    assert resumed.messages == alice.messages
    assert resumed.notice.startswith('Continuing your session from Logfire (1 runs')
    laptop.session_id, laptop.messages = resumed.conversation_id, resumed.messages
    await laptop.prompt('Also the other test')
    assert [span.attributes['clai2.session.seq'] for span in chunk_spans(exporter)] == [0, 1]  # pyright: ignore[reportOptionalSubscript]

    # Back on the first machine, which is now behind: a fork, not a second run 1.
    stale = await alice.sessions.resume(alice.session_id, local=True)
    assert stale is not None and stale.conversation_id != alice.session_id
    assert stale.notice.startswith('Forked your session, which continued elsewhere')

    # Up to date and saved locally: resume the local copy as it is.
    current = await laptop.sessions.resume(alice.session_id, local=True)
    assert current is not None and current.messages is None and current.conversation_id == alice.session_id

    # Someone else: always a fork, which records where it came from and leaves the original alone.
    bob = Machine(tmp_path, 'b', BOB, provider, query)
    fork = await bob.sessions.resume(alice.session_id, local=False)
    assert fork is not None and fork.messages == laptop.messages
    assert fork.notice.startswith(f"Forked {ALICE}'s session {alice.session_id[:8]} at run 1 from Logfire.")
    before = len(chunk_spans(exporter))
    bob.session_id, bob.messages = fork.conversation_id, fork.messages
    await bob.prompt('Bob adds a test')
    [bob_chunk] = chunk_spans(exporter)[before:]
    attributes = bob_chunk.attributes or {}
    assert attributes['clai2.session.id'] == fork.conversation_id
    assert attributes['clai2.session.reset'] is False
    assert (attributes['clai2.session.parent_id'], attributes['clai2.session.parent_seq']) == (alice.session_id, 1)
    assert attributes['clai2.session.parent_owner'] == ALICE
    assert attributes['clai2.session.parent_trace'] == f'{chunk_spans(exporter)[1].context.trace_id:032x}'  # pyright: ignore[reportOptionalMemberAccess]

    # The fork reads back as the original up to the fork point, plus Bob's run; the original is unchanged.
    loaded_fork = await bob.sessions.load(fork.conversation_id)
    assert loaded_fork is not None and loaded_fork.messages == bob.messages
    original = await alice.sessions.load(alice.session_id)
    assert original is not None and original.messages == laptop.messages and original.last_seq == 1

    assert await bob.sessions.resume('0123456789abcdef', local=False) is None
    with pytest.raises(LookupError, match='No clai2 session is stored in trace'):
        await bob.sessions.resume(trace_link('https://logfire.example', 'acme/clai2', '0' * 32), local=False)
    with pytest.raises(LookupError, match='names no trace'):
        await bob.sessions.resume('https://logfire.example/acme/clai2', local=False)


async def test_a_damaged_or_incomplete_session_fails_loudly(
    tmp_path: Path, provider: TracerProvider, query: SpanQuery, monkeypatch: pytest.MonkeyPatch
) -> None:
    alice = Machine(tmp_path, 'a', ALICE, provider, query)
    await alice.prompt('one')
    await alice.prompt('two')
    session = alice.session_id

    query.edit[(session, 1, 0)] = {'payload': 'AAAA'}
    with pytest.raises(SessionCorrupt, match=f'Run 1 of session {session} is truncated or damaged'):
        await alice.sessions.load(session)
    query.edit[(session, 1, 0)] = {'payload': '!!'}
    with pytest.raises(SessionCorrupt, match='not valid base64'):
        await alice.sessions.load(session)
    query.edit.clear()

    query.drop.add((session, 0, 0))
    with pytest.raises(SessionCorrupt, match='missing run 0 of 2'):
        await alice.sessions.resume(session, local=False)
    query.drop.clear()

    query.edit[(session, 1, 0)] = {'parts': 2}
    with pytest.raises(SessionCorrupt, match='missing part 2 of 2'):
        await alice.sessions.load(session)
    query.edit.clear()

    # Two machines both wrote run 2.
    other = Machine(tmp_path, 'other', ALICE, provider, query)
    other.session_id, other.messages = session, alice.messages
    other.states.set(session, alice.states.get(session))  # pyright: ignore[reportArgumentType]
    await other.prompt('three, here')
    await alice.prompt('three, there')
    with pytest.raises(SessionCorrupt, match='continued in two places at run 2'):
        await alice.sessions.load(session)

    # Large runs are split across spans and read back whole.
    monkeypatch.setattr(logfire_sessions, 'PART_CHARS', 64)
    big = Machine(tmp_path, 'big', ALICE, provider, query)
    await big.prompt('x' * 2_000)
    loaded = await big.sessions.load(big.session_id)
    assert loaded is not None and loaded.messages == big.messages


def test_binary_content_is_replaced_and_the_privacy_line_shows_once(tmp_path: Path) -> None:
    image = BinaryContent(data=b'\x89PNG', media_type='image/png')
    [request], dropped = without_binary([ModelRequest(parts=[UserPromptPart(['look', image])])])
    assert dropped == 1
    assert isinstance(request, ModelRequest)
    assert request.parts[0].content == ['look', '[image/png attachment not stored in Logfire]']  # pyright: ignore[reportAttributeAccessIssue]
    marker = tmp_path / 'notice.json'
    assert privacy_notice(marker, 'acme/clai2') == 'Sessions are stored in acme/clai2 and visible to its members.'
    assert privacy_notice(marker, 'acme/clai2') is None


async def test_resume_saves_a_remote_session_and_continues_it_here(tmp_path: Path) -> None:
    from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
    from pydantic_clai2.cli.command_context import CommandContext
    from pydantic_clai2.config import Settings
    from pydantic_clai2.config.settings_store import SettingsStore
    from pydantic_clai2.runtime import remote_sessions
    from pydantic_clai2.runtime._session import Session
    from pydantic_clai2.runtime.remote_sessions import RemoteListing, RemoteResume
    from pydantic_clai2.runtime.sessions import Sessions

    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(
        Agent(TestModel(custom_output_text='continued')), deps=None, conversations=store, workspace=tmp_path
    )
    context = CommandContext(
        settings=Settings(model=None, session_namer=False),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=session.clear,
        apply_setting=lambda key, settings: None,
    )
    service = Sessions(session=session, store=store, context=context)
    history = (await Agent(TestModel(custom_output_text='earlier')).run('Fix the flaky test')).all_messages()
    asked: list[tuple[str, bool]] = []

    class Remote:
        def __init__(self, answer: RemoteResume | None) -> None:
            self.answer = answer

        async def resume(self, reference: str, *, local: bool) -> RemoteResume | None:
            asked.append((reference, local))
            return self.answer

        async def listing(self, *, workspace: str) -> RemoteListing:
            return RemoteListing(entries=[], unavailable='Logfire sessions unavailable: test.')

    fork = RemoteResume(
        conversation_id=str(uuid4()), messages=history, title='Fork: Fix the flaky test', notice='Forked it.'
    )
    remote_sessions.install(Remote(fork))
    try:
        notice = await service.command(['https://logfire.example/acme/clai2?traceId=' + '1' * 32])
        assert notice.startswith('Forked it. Resumed Fork: Fix the flaky test')
        assert service.session.summary.id == fork.conversation_id and service.session.messages == history
        assert (await store.get(conversation_id=fork.conversation_id)).summary.tags == ('logfire',)

        # Saved here already and up to date: the local copy is resumed as it is; a newer remote copy replaces it.
        remote_sessions.install(Remote(RemoteResume(conversation_id=fork.conversation_id, messages=None)))
        assert (await service.command([fork.conversation_id])).startswith('Resumed Fork: Fix the flaky test')
        remote_sessions.install(Remote(fork))
        await service.command([fork.conversation_id])
        assert asked[-2:] == [(fork.conversation_id, True), (fork.conversation_id, True)]

        listing = await service.listing_command([])
        assert f'{fork.conversation_id}' in listing and 'this machine' in listing
        assert 'Logfire sessions unavailable: test.' in listing

        # Unknown to the remote store: the usual local lookup.
        remote_sessions.install(Remote(None))
        with pytest.raises(LookupError):
            await service.command(['missing'])
    finally:
        remote_sessions.install(None)
    assert remote_sessions.current() is None


async def test_logfire_sessions_are_listed_for_sessions_and_resume(
    tmp_path: Path, provider: TracerProvider, query: SpanQuery
) -> None:
    alice = Machine(tmp_path, 'a', ALICE, provider, query)
    await alice.prompt('Fix the flaky test')
    listing = await alice.sessions.listing(workspace=str(tmp_path))
    [entry] = listing.entries
    assert (entry.id, entry.title, entry.tags) == (alice.session_id, 'Fix the flaky test', ('logfire',))
    assert entry.subtitle == 'Logfire · other machine · acme/widgets · 1 runs'
    assert listing.unavailable is None

    class Refused(SpanQuery):
        async def rows(self, sql: str, *, limit: int = 10_000) -> list[dict[str, Any]]:
            raise PermissionError('no query scope')

    refused = LogfireSessions(query=Refused(query.exporter), states=alice.states, owner=lambda: ALICE, project='p')
    assert (await refused.listing(workspace='.')).unavailable == 'Logfire sessions unavailable: no query scope'

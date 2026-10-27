"""Auxiliary naming never changes the conversation head or stalls a foreground turn."""

from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid5

import anyio
import pytest
from pydantic import ValidationError

from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from pydantic_clai2.runtime.session import Session
from pydantic_clai2.runtime.session_naming import NamingClient, NamingResult, SessionName, SessionNamer, generate_name


async def test_structured_name_and_auxiliary_usage() -> None:
    result = await generate_name(model=TestModel(), prompt='Fix renderer')
    assert isinstance(result.name, SessionName)
    assert result.tokens > 0
    with pytest.raises(ValidationError):
        SessionName(title='')
    with pytest.raises(ValidationError, match='printable'):
        SessionName(title='\x1b  ')
    name = SessionName(title='Fix\nrenderer\x1b', tags=['#SQLite', 'sqlite', '  ', 'tools'])
    assert name.title == 'Fix renderer'
    assert name.tags == ['sqlite', 'tools']


async def test_naming_uses_revision_and_bounded_tail(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    summary = await store.save(
        summary=ConversationSummary(workspace='/a', title='Fallback'),
        messages=[ModelRequest(parts=[UserPromptPart('old' * 2000 + ' fix renderer')])],
    )
    prompts: list[str] = []

    async def generate(prompt: str) -> NamingResult:
        prompts.append(prompt)
        return NamingResult(
            name=SessionName(title='Fix renderer', subtitle='Cancellation ordering', tags=['tests']), tokens=5
        )

    namer, client = SessionNamer(store=store), NamingClient(generate=generate)
    assert await namer.name(conversation_id=summary.id, client=client)
    # The tail, plus the start of a conversation still under its fallback title.
    assert len(prompts[0]) < 2900
    assert 'fix renderer' in prompts[0]
    named = (await store.get(conversation_id=summary.id)).summary
    assert named.title == 'Fix renderer'
    assert named.updated_at == summary.updated_at
    assert named.naming_tokens == 5
    assert not await namer.name(conversation_id=summary.id, client=client)
    assert SessionNamer.needed(replace(named, revision=named.revision + 16))
    assert not SessionNamer.needed(replace(named, title_source='user', revision=1000))


async def test_empty_missing_model_and_late_result(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    first = await store.save(summary=ConversationSummary(workspace='/a'), messages=[])

    async def no_model(prompt: str) -> None:
        return None

    namer = SessionNamer(store=store)
    assert not await namer.name(conversation_id=first.id, client=NamingClient(generate=no_model))
    second = await store.save(summary=first, messages=[ModelRequest(parts=[UserPromptPart('hello')])])
    assert not await namer.name(conversation_id=first.id, client=NamingClient(generate=no_model))

    async def delete_during_name(prompt: str) -> NamingResult:
        await store.delete(source=second)
        return NamingResult(name=SessionName(title='Too late'))

    assert not await namer.name(conversation_id=first.id, client=NamingClient(generate=delete_during_name))
    assert await store.listing() == []


async def test_worker_is_bounded_single_flight_and_drained(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    summary = await store.save(
        summary=ConversationSummary(workspace='/a'), messages=[ModelRequest(parts=[UserPromptPart('hello')])]
    )
    entered, stopped = anyio.Event(), anyio.Event()
    active = 0

    async def generate(prompt: str) -> NamingResult:
        nonlocal active
        active += 1
        assert active == 1
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            active -= 1
            stopped.set()
        raise AssertionError('unreachable')  # pragma: no cover

    namer, client = SessionNamer(store=store), NamingClient(generate=generate)
    assert namer.submit(summary.id, client=client)
    assert not namer.submit(summary.id, client=client)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(namer.run)
            await entered.wait()
            for i in range(10):
                assert namer.submit(str(i), client=client)
            assert not namer.submit('over capacity', client=client)
            group.cancel_scope.cancel()
    assert stopped.is_set()
    assert active == 0
    assert (await store.get(conversation_id=summary.id)).summary.title_source == 'fallback'


async def test_backfill_disabled_and_failure_containment(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    summary = await store.save(
        summary=ConversationSummary(workspace='/a'), messages=[ModelRequest(parts=[UserPromptPart('hello')])]
    )
    called = anyio.Event()

    async def fail(prompt: str) -> NamingResult:
        called.set()
        raise ValueError('invalid response')

    enabled = False
    namer, client = SessionNamer(store=store), NamingClient(generate=fail, enabled=lambda: enabled)
    assert not namer.submit(summary.id, client=client)
    enabled = True
    namer.backfill([replace(summary, title_source='user'), summary], client=client)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(namer.run)
            await called.wait()
            group.cancel_scope.cancel()
    assert (await store.get(conversation_id=summary.id)).summary.title_source == 'fallback'


async def test_queued_work_disabled_before_execution(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    checked = anyio.Event()

    async def generate(prompt: str) -> NamingResult:
        pytest.fail('Disabled work must not call the model')  # pragma: no cover

    def disabled() -> bool:
        checked.set()
        return False

    queueing = True
    namer = SessionNamer(store=store)
    assert namer.submit('queued', client=NamingClient(generate=generate, enabled=lambda: queueing or disabled()))
    queueing = False
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(namer.run)
            await checked.wait()
            group.cancel_scope.cancel()


async def test_a_slow_notice_outlives_the_naming_timeout(tmp_path: Path) -> None:
    """The shell may hold the notice until a long turn ends; the saved name still reaches it."""
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    summary = await store.save(
        summary=ConversationSummary(workspace='/a'), messages=[ModelRequest(parts=[UserPromptPart('fix renderer')])]
    )
    entered, release, done = anyio.Event(), anyio.Event(), anyio.Event()
    notified: list[tuple[str, str]] = []

    async def generate(prompt: str) -> NamingResult:
        return NamingResult(name=SessionName(title='Fix renderer'))

    async def on_named(conversation_id: str, title: str) -> None:
        entered.set()
        await release.wait()
        notified.append((conversation_id, title))
        done.set()

    named_by_user = await store.save(
        summary=ConversationSummary(workspace='/a', title_source='user'),
        messages=[ModelRequest(parts=[UserPromptPart('keep my title')])],
    )
    namer, client = SessionNamer(store=store, timeout=0.01), NamingClient(generate=generate)
    namer.listeners.append(on_named)
    # A conversation that needs no name saves nothing, so the shell hears nothing about it.
    assert namer.submit(named_by_user.id, client=client)
    assert namer.submit(summary.id, client=client)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(namer.run)
            # Hold the notice, once it is waiting, past the naming timeout.
            await entered.wait()
            await anyio.sleep(0.05)
            release.set()
            await done.wait()
            group.cancel_scope.cancel()
    assert notified == [(summary.id, 'Fix renderer')]


async def test_one_worker_names_each_conversation_with_the_model_of_the_session_that_queued_it(
    tmp_path: Path,
) -> None:
    """The process's one namer uses the asking session's model, and tells every live session the title."""
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    first, second = [
        await store.save(
            summary=ConversationSummary(workspace='/a'), messages=[ModelRequest(parts=[UserPromptPart(text)])]
        )
        for text in ('fix renderer', 'write docs')
    ]
    heard: list[tuple[str, str, str]] = []
    done = anyio.Event()

    def client(name: str) -> NamingClient:
        async def generate(prompt: str) -> NamingResult:
            return NamingResult(name=SessionName(title=f'Named by {name}'))

        return NamingClient(generate=generate)

    def listener(name: str) -> Callable[[str, str], Awaitable[None]]:
        async def named(conversation_id: str, title: str) -> None:
            heard.append((name, conversation_id, title))
            if len(heard) == 4:
                done.set()

        return named

    namer = SessionNamer(store=store)
    namer.listeners.extend([listener('a'), listener('b')])
    assert namer.submit(first.id, client=client('a'))
    namer.backfill([second], client=client('b'))
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(namer.run)
            await done.wait()
            group.cancel_scope.cancel()
    assert heard == [
        ('a', first.id, 'Named by a'),
        ('b', first.id, 'Named by a'),
        ('a', second.id, 'Named by b'),
        ('b', second.id, 'Named by b'),
    ]


async def test_a_session_waiting_to_hear_a_title_holds_up_no_other_naming(tmp_path: Path) -> None:
    """A listener waits for its terminal; meanwhile the next conversation is named, and a failing one is logged."""
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    first, second = [
        await store.save(
            summary=ConversationSummary(workspace='/a'), messages=[ModelRequest(parts=[UserPromptPart(text)])]
        )
        for text in ('fix renderer', 'write docs')
    ]
    waiting, release, named_second, done = anyio.Event(), anyio.Event(), anyio.Event(), anyio.Event()
    heard: list[str] = []

    async def generate(prompt: str) -> NamingResult:
        return NamingResult(name=SessionName(title='Named'))

    async def busy(conversation_id: str, title: str) -> None:
        if conversation_id == first.id:
            waiting.set()
            await release.wait()
        heard.append(conversation_id)
        (done if len(heard) == 2 else named_second).set()

    async def broken(conversation_id: str, title: str) -> None:
        raise RuntimeError('a listener failed')

    namer, client = SessionNamer(store=store), NamingClient(generate=generate)
    namer.listeners.extend([busy, broken])
    assert namer.submit(first.id, client=client)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(namer.run)
            await waiting.wait()
            assert namer.submit(second.id, client=client)
            await named_second.wait()
            release.set()
            await done.wait()
            group.cancel_scope.cancel()
    assert heard == [second.id, first.id]


async def test_naming_keeps_where_an_imported_session_came_from(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    summary = await store.save(
        summary=ConversationSummary(
            id=str(uuid5(UUID(int=0), 'claude:session')), workspace='/a', title='Fallback', tags=('claude',)
        ),
        messages=[ModelRequest(parts=[UserPromptPart('fix renderer')])],
    )
    # A fork of an import has an ID of its own, so it is CLAI's, whatever tag it copied.
    native = await store.save(
        summary=ConversationSummary(workspace='/a', title='Fallback', tags=('claude',)),
        messages=[ModelRequest(parts=[UserPromptPart('ask claude')])],
    )

    async def generate(prompt: str) -> NamingResult:
        return NamingResult(name=SessionName(title='Fix renderer', tags=['codex', 'claude', 'tests']))

    namer, client = SessionNamer(store=store), NamingClient(generate=generate)
    assert await namer.name(conversation_id=summary.id, client=client)
    assert (await store.get(conversation_id=summary.id)).summary.tags == ('claude', 'tests')
    # A source the model named as a topic would pass for an import, so it is left out.
    assert await namer.name(conversation_id=native.id, client=client)
    assert (await store.get(conversation_id=native.id)).summary.tags == ('tests',)


async def test_templated_sessions_get_their_own_titles_not_the_prompt_prefix(tmp_path: Path) -> None:
    """The prompt-prefix fallback is a placeholder, not a previous title to keep."""
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    # Long enough that the issue number, past the fallback's cut, also leaves the naming tail.
    template = (
        'You are fixing GitHub issue https://github.com/pydantic/pydantic-ai/issues/{} ' + 'Follow the steps. ' * 200
    )
    sessions = [Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path) for _ in range(2)]
    for number, session in zip(('11281', '11282'), sessions):
        await session.prompt(template.format(number))
    fallbacks = {(await store.get(conversation_id=s.conversation_id)).summary.title for s in sessions}
    assert fallbacks == {'You are fixing GitHub issue https://github.com/pydantic/pydantic'}
    prompts: list[str] = []

    async def generate(prompt: str) -> NamingResult:
        """Keeps any previous title it is given, as the namer is instructed to; else names the issue."""
        prompts.append(prompt)
        if prompt.startswith('Previous title: '):
            return NamingResult(name=SessionName(title=prompt.splitlines()[0].removeprefix('Previous title: ')))
        number = prompt.split('/issues/')[1].split()[0]
        return NamingResult(name=SessionName(title=f'Fix issue {number}'))

    namer, client = SessionNamer(store=store), NamingClient(generate=generate)
    for session in sessions:
        assert await namer.name(conversation_id=session.conversation_id, client=client)
    titles = [(await store.get(conversation_id=s.conversation_id)).summary.title for s in sessions]
    assert titles == ['Fix issue 11281', 'Fix issue 11282']
    # Once a title is the model's own, later renames are told it, to keep it stable.
    named = await store.get(conversation_id=sessions[0].conversation_id)
    for _ in range(16):
        named = await store.get(conversation_id=(await store.save(summary=named.summary, messages=named.messages)).id)
    assert await namer.name(conversation_id=named.summary.id, client=client)
    assert prompts[-1].startswith('Previous title: Fix issue 11281\n')

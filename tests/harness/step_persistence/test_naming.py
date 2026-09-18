"""Auxiliary naming never changes the conversation head or stalls a foreground turn."""

from dataclasses import replace
from pathlib import Path

import anyio
import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from pydantic_ai_harness.step_persistence.naming import NamingResult, SessionName, SessionNamer, generate_name


async def test_structured_name_and_auxiliary_usage(anyio_backend: str) -> None:
    if anyio_backend == 'trio':
        pytest.skip('Pydantic AI Agent requires asyncio; worker tests also run on Trio')
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

    namer = SessionNamer(store=store, generate=generate)
    assert await namer.name(conversation_id=summary.id)
    assert len(prompts[0]) < 2600
    assert 'fix renderer' in prompts[0]
    named = (await store.get(conversation_id=summary.id)).summary
    assert named.title == 'Fix renderer'
    assert named.updated_at == summary.updated_at
    assert named.naming_tokens == 5
    assert not await namer.name(conversation_id=summary.id)
    assert SessionNamer.needed(replace(named, revision=named.revision + 16))
    assert not SessionNamer.needed(replace(named, title_source='user', revision=1000))


async def test_empty_missing_model_and_late_result(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    first = await store.save(summary=ConversationSummary(workspace='/a'), messages=[])

    async def no_model(prompt: str) -> None:
        return None

    namer = SessionNamer(store=store, generate=no_model)
    assert not await namer.name(conversation_id=first.id)
    second = await store.save(summary=first, messages=[ModelRequest(parts=[UserPromptPart('hello')])])
    assert not await namer.name(conversation_id=first.id)

    async def delete_during_name(prompt: str) -> NamingResult:
        await store.delete(source=second)
        return NamingResult(name=SessionName(title='Too late'))

    namer = SessionNamer(store=store, generate=delete_during_name)
    assert not await namer.name(conversation_id=first.id)
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

    namer = SessionNamer(store=store, generate=generate)
    assert namer.submit(summary.id)
    assert not namer.submit(summary.id)
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(namer.run)
            await entered.wait()
            for i in range(10):
                assert namer.submit(str(i))
            assert not namer.submit('over capacity')
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

    namer = SessionNamer(store=store, generate=fail, enabled=lambda: False)
    assert not namer.submit(summary.id)
    namer.enabled = lambda: True
    namer.backfill([replace(summary, title_source='user'), summary])
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

    namer = SessionNamer(store=store, generate=generate)
    assert namer.submit('queued')
    namer.enabled = disabled
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(namer.run)
            await checked.wait()
            group.cancel_scope.cancel()

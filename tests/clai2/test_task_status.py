"""A running subagent's own figures on the status row, and the main run's once it settles."""

import asyncio
import io
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
from rich.console import Console

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.subagents import DelegationTaskEvent
from pydantic_clai2._app import create_shell, create_stock_agent
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.tasks import Tasks
from pydantic_clai2.ui.rendering.status import Status
from pydantic_clai2.ui.rendering.theme import INFO, MUTED
from tests.clai2.test_tasks import task


def tasks() -> Tasks:
    return Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)


async def test_observe_records_the_child_model_window_and_context() -> None:
    ui = tasks()
    record = task()
    ui.owner.records[record.id] = record
    await ui.observe(DelegationTaskEvent(task=record))
    progress = ui.progress[record.id]
    assert (progress.agent, progress.model, progress.context_tokens) == ('worker [aaaaaaaa]', '', None)
    record.messages = [
        ModelRequest(parts=[UserPromptPart('inspect')]),
        ModelResponse(parts=[TextPart('first')], usage=RequestUsage(input_tokens=1_000, output_tokens=200)),
        ModelRequest(parts=[UserPromptPart('more')]),
        ModelResponse(parts=[TextPart('unreported')]),
    ]
    await ui.observe(DelegationTaskEvent(task=record, model_name='child-model', context_window=200_000))
    assert (progress.model, progress.context_window, progress.context_tokens) == ('child-model', 200_000, 1_200)
    # Lifecycle updates carry no model and keep what the child's stream reported.
    await ui.observe(DelegationTaskEvent(task=record))
    assert (progress.model, progress.context_window) == ('child-model', 200_000)


async def test_focus_follows_the_newest_child_the_main_run_waits_on() -> None:
    ui = tasks()
    assert ui.focused() is None
    first, second = task(), task(task_id='b' * 32)
    nested = task(task_id='c' * 32, parent_id=second.id)
    first.started_at, second.started_at, nested.started_at = 1.0, 2.0, 3.0
    for record in (first, second, nested):
        ui.owner.records[record.id] = record
        await ui.observe(DelegationTaskEvent(task=record))
    assert ui.focused() is ui.progress[nested.id]
    # A background ancestor leaves its descendants to the panel, not the row.
    second.background = True
    assert ui.focused() is ui.progress[first.id]
    second.background = False
    nested.status = 'finished'
    assert ui.focused() is ui.progress[second.id]
    second.status = first.status = 'finished'
    assert ui.focused() is None
    # Another conversation's children never take the row.
    other = task(task_id='d' * 32, conversation_id='other')
    ui.owner.records[other.id] = other
    assert ui.focused() is None


def test_row_shows_the_subagent_then_restores_the_main_run() -> None:
    child = Status(agent='Explore [abcd1234]', model='', context_tokens=8_000, context_window=200_000)
    child.activity = 'running: grep'
    shown: list[Status] = [child]
    status = Status(
        model='main',
        workspace='/srv/app',
        context_tokens=30_000,
        status_segments=(lambda: 'plugin',),
        subagent=lambda: shown[0] if shown else None,
    )
    assert status.text() == (
        'Explore [abcd1234] | context: 8k/200k tokens | ~0 streamed tokens | running: grep | plugin'
    )
    assert status.toolbar()[:2] == [(INFO, 'Explore [abcd1234]'), (MUTED, ' | context: ')]
    assert ''.join(text for _, text in status.toolbar()) == status.text()
    child.model = 'child-model'
    assert status.text().startswith('Explore [abcd1234]: child-model | context: ')
    assert ''.join(text for _, text in status.toolbar()) == status.text()
    shown.clear()
    assert status.text().startswith('main | /srv/app | context: 30k/? tokens')
    assert status.toolbar()[0] == (MUTED, 'main | /srv/app | context: ')


async def test_foreground_delegation_takes_the_row_until_it_settles(tmp_path: Path) -> None:
    child_started, release = asyncio.Event(), asyncio.Event()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if prompt == 'child':
            yield 'child '
            child_started.set()
            await release.wait()
            yield 'evidence'
        elif len(messages) == 1:
            yield {0: DeltaToolCall(name='delegate_task', json_args='{"agent_name":"self","task":"child"}')}
        else:
            yield 'done'

    ui = tasks()
    status = Status(model='main', subagent=ui.focused)
    session = Session(
        create_stock_agent(FunctionModel(stream_function=stream, profile={'context_window': 64_000})),
        deps=None,
        workspace=tmp_path,
        plugins=[Coder(repo_context=False)],
    )
    session.delegations = ui.owner
    ui.conversation_id = lambda: session.summary.id
    with anyio.fail_after(10):
        async with ui.owner.opened(), anyio.create_task_group() as group:
            group.start_soon(session.prompt, 'parent')
            await child_started.wait()
            assert status.text().startswith('general-purpose [')
            assert ': function::stream | context: ?/64k tokens | ' in status.text()
            release.set()
        assert status.text().startswith('main | context: ')


def test_shell_status_row_follows_its_tasks(tmp_path: Path) -> None:
    shell = create_shell(
        create_stock_agent(TestModel()),
        deps=None,
        plugins=[],
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    assert shell.status.subagent == shell.tasks.focused

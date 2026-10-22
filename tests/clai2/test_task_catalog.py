"""`host.tasks`, the `TaskCatalog` over `/tasks`, and the `on_task_event` moment."""

import asyncio
import io
from collections.abc import AsyncIterator
from pathlib import Path
from typing import ClassVar

import anyio
import pytest
from rich.console import Console

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.subagents import DelegationEndEvent, DelegationStartEvent, DelegationTask, DelegationTaskEvent
from pydantic_clai2._app import create_stock_agent
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart, TaskEvent, TaskInfo
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.runtime.tasks import Tasks, resume_prompt
from pydantic_clai2.ui.rendering.status import Status
from tests.clai2.test_tasks import task


class Watcher(Plugin):
    """Keeps its host, every task event, and what `host.tasks` listed while the child ran."""

    hosts: ClassVar[list[PluginHost[None]]] = []
    events: ClassVar[list[TaskEvent]] = []
    listed: ClassVar[list[list[TaskInfo]]] = []

    async def on_session_start(self, event: SessionStart) -> None:
        self.hosts.append(self.host)

    async def on_task_event(self, event: TaskEvent) -> None:
        self.events.append(event)


@pytest.fixture(autouse=True)
def fresh_watcher() -> None:
    Watcher.hosts.clear()
    Watcher.events.clear()
    Watcher.listed.clear()


def first_prompt(messages: list[ModelMessage]) -> str:
    request = messages[0]
    assert isinstance(request, ModelRequest)
    prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
    assert isinstance(prompt, str)
    return prompt


async def test_tasks_reach_plugins_while_the_shell_runs(tmp_path: Path) -> None:
    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if first_prompt(messages) == 'child' and len(messages) == 1:
            catalog = Watcher.hosts[0].tasks
            assert catalog is not None
            listed = catalog.listing()
            Watcher.listed.append(listed)
            assert catalog.steer(listed[0].id, 'look at the tests too')
            yield 'child done'
        elif first_prompt(messages) == 'child':
            assert 'look at the tests too' in str(messages[-1])
            yield 'child steered'
        elif len(messages) == 1:
            yield {0: DeltaToolCall(name='delegate_task', json_args='{"agent_name":"self","task":"child"}')}
        else:
            yield 'parent done'

    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Coder(repo_context=False)],
        usage_limits=None,
        console=Console(file=io.StringIO(), width=120),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[PluginSettings(id='watcher', factory='tests.clai2.test_task_catalog:Watcher')],
        project=ProjectSettings(),
        headless=True,
    )
    await shell.loader.load_all()
    [host] = Watcher.hosts
    assert host.tasks is None, 'no task owner runs before the prompt opens'
    host.send_input('parent')
    host.send_input('/exit')
    with anyio.fail_after(10):
        assert await shell.run() == 'exit'
    assert host.tasks is None, 'nor after it closes'
    await shell.loader.close('exit')

    [[running]] = Watcher.listed
    assert (running.agent, running.prompt, running.status, running.parent_id) == (
        'general-purpose',
        'child',
        'running',
        None,
    )
    first, last = Watcher.events[0], Watcher.events[-1]
    assert first.event is None and first.task.status == 'running' and first.task.activity == 'starting'
    assert any(isinstance(event.event, DelegationStartEvent) for event in Watcher.events)
    assert any(isinstance(event.event, DelegationEndEvent) for event in Watcher.events)
    assert last.event is None
    assert (last.task.status, last.task.outcome, last.task.usage.requests) == ('finished', 'ok', 2)
    assert last.task.finished_at is not None and last.task.finished_at >= last.task.started_at
    assert last.task.model is not None and last.task.model.startswith('function:')


async def test_catalog_lists_and_controls_the_current_conversations_tasks() -> None:
    entered: list[str] = []
    fired: list[TaskEvent] = []

    async def fire(event: TaskEvent) -> None:
        fired.append(event)

    ui = Tasks(
        console=Console(file=io.StringIO()),
        conversation_id=lambda: 'root',
        directory=None,
        fire=fire,
        send_input=entered.append,
    )
    parent, child, elsewhere = task(), task(task_id='b' * 32, parent_id='a' * 32), task(task_id='c' * 32)
    elsewhere.conversation_id = 'other'
    parent.model = 'openai:gpt-5'
    parent.messages = [
        ModelRequest(parts=[UserPromptPart('inspect')]),
        ModelResponse(
            parts=[TextPart('found'), ToolCallPart('shell', {'command': 'ls'}, 'ls'), ToolCallPart('shell', {}, 'bad')],
            usage=RequestUsage(input_tokens=10, output_tokens=3),
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('shell', 'README.md', 'ls'),
                ToolReturnPart('shell', 'command is required', tool_call_id='bad', outcome='retried'),
            ]
        ),
    ]
    ui.owner.records = {child.id: child, parent.id: parent, elsewhere.id: elsewhere}
    child.started_at = parent.started_at + 1
    ui.progress[child.id] = Status(model='anthropic:claude', activity='running: shell')

    listed = ui.listing()
    assert [info.id for info in listed] == [parent.id, child.id]
    assert (listed[0].model, listed[0].activity, listed[0].usage.output_tokens) == ('openai:gpt-5', 'starting', 3)
    assert (listed[0].usage.requests, listed[0].usage.tool_calls) == (1, 1)
    assert (listed[1].model, listed[1].activity, listed[1].parent_id) == (
        'anthropic:claude',
        'running: shell',
        parent.id,
    )
    assert ui.history(parent.id) == parent.messages
    with pytest.raises(LookupError, match='in this conversation'):
        ui.history(elsewhere.id)

    await ui.observe(DelegationTaskEvent(task=elsewhere))
    assert fired == [], 'only the current conversation is told'
    ui.listening = lambda: False
    await ui.observe(DelegationTaskEvent(task=child))
    assert fired == [], 'no event is built while no plugin handles one'
    ui.listening = lambda: True
    await ui.observe(DelegationTaskEvent(task=child))
    assert [event.task.id for event in fired] == [child.id]

    started = asyncio.Event()

    async def work(record: DelegationTask) -> str:
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    async with ui.owner.opened():
        waiting = asyncio.create_task(
            ui.owner.delegate(
                agent_name='worker',
                prompt='',
                conversation_id='root',
                model=None,
                background=False,
                resume=None,
                run=work,
            )
        )
        await started.wait()
        [worker] = [
            record for record in ui.owner.records.values() if record.id not in (parent.id, child.id, elsewhere.id)
        ]
        ui.background(worker.id)
        assert 'not a result' in await waiting
        assert worker.background
        await ui.stop(worker.id)
        assert worker.user_stopped and worker.status == 'finished'
        assert not ui.steer(worker.id, 'too late')
        await ui.resume(worker.id)
        assert not worker.user_stopped
        assert entered == [resume_prompt(worker)]
        with pytest.raises(ValueError, match='resumable'):
            await ui.resume(parent.id)


async def test_resume_needs_a_prompt() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    record = task()
    record.status = 'finished'
    ui.owner.records[record.id] = record
    with pytest.raises(RuntimeError, match='no prompt'):
        await ui.resume(record.id)


async def test_resume_keeps_the_task_stopped_when_the_prompt_refuses() -> None:
    def refuse(text: str) -> None:
        raise RuntimeError('line by line')

    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None, send_input=refuse)
    record = task()
    record.status, record.user_stopped = 'finished', True
    ui.owner.records[record.id] = record
    with pytest.raises(RuntimeError, match='line by line'):
        await ui.resume(record.id)
    assert record.user_stopped, "the user's stop holds when the request cannot be entered"

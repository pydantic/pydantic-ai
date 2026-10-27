"""Task rendering, live inspection, and stock-agent integration."""

import asyncio
import io
import re
import time
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_ai import Agent, AgentRunResult, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextContent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.subagents import (
    DelegationEndEvent,
    DelegationStartEvent,
    DelegationTask,
    DelegationTaskEvent,
    SubAgent,
    SubAgents,
    TaskControl,
    TaskQuestionEvent,
    is_task_report,
)
from pydantic_clai2._app import create_stock_agent
from pydantic_clai2.builtin_plugins.web.protocol import history_items
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import TurnStart
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.sandbox_calls import DelegationToolCallEvent
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.runtime.tasks import TaskPresentation, Tasks, task_row, task_tree
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.menus.task_menu import TaskDetail, TaskMenu, open_tasks, run_tasks
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.prompt.terminal import Terminal
from pydantic_clai2.ui.rendering.history import turn_starts
from tests.clai2.test_forks import Model as ForkModel, shell_for


def task(*, task_id: str = 'a' * 32, parent_id: str | None = None, conversation_id: str = 'root') -> DelegationTask:
    return DelegationTask(
        id=task_id, parent_id=parent_id, agent_name='worker', prompt='inspect', conversation_id=conversation_id
    )


def test_panel_tree_and_completion_retention() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    parent, child = task(), task(task_id='b' * 32, parent_id='a' * 32)
    ui.owner.records = {parent.id: parent, child.id: child}
    assert [(depth, record.id) for depth, record in task_tree(ui.records())] == [(0, parent.id), (1, child.id)]
    assert '1 descendants' in '\n'.join(ui.rows('*'))
    child.status, child.outcome, child.finished_at = 'finished', 'ok', time.time()
    assert '[bbbbbbbb]' not in '\n'.join(ui.rows('*'))
    child.outcome = 'failed'
    assert '[bbbbbbbb]' in '\n'.join(ui.rows('*'))
    child.finished_at -= 31
    assert '[bbbbbbbb]' not in '\n'.join(ui.rows('*'))


def test_live_menu_keeps_selected_child_after_completion_and_insert() -> None:
    parent, sibling = task(), task(task_id='c' * 32)
    records = [parent, sibling]
    actions: list[tuple[str, str]] = []
    picker = TaskMenu(snapshot=lambda: records, action=lambda name, task_id: actions.append((name, task_id)))
    item = picker.items()[1]
    assert item.value == sibling.id
    assert 'running' in picker.preview(item)
    records.insert(1, task(task_id='b' * 32, parent_id=parent.id))
    sibling.status, sibling.outcome, sibling.output = 'finished', 'ok', 'child result'
    picker.records = records
    assert picker.items()[1].value == sibling.id
    assert 'child result' in picker.preview(item)
    assert picker.selected == sibling.id
    picker.scroll(10)
    assert picker.offset == 10
    picker.scroll(-20)
    assert picker.offset == 0
    assert 'child result' in picker.preview(item)


def test_menu_empty_and_untrusted_transcript() -> None:
    picker = TaskMenu(snapshot=lambda: (), action=lambda name, task_id: None)
    assert picker.items()[0].disabled
    assert picker.preview(MenuItem('none')) == ''
    record = task()
    record.prompt = 'unsafe\x1b]52;c;payload\x07'
    picker.records = [record]
    assert '\x1b' not in picker.preview(picker.items()[0])


@pytest.mark.parametrize('name', ['Explore', 'Plan', 'general-purpose'])
async def test_stock_managed_specialists_and_general_purpose(tmp_path: Path, name: str) -> None:
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        assert isinstance(prompt, str)
        if len(messages) == 1:
            prompts.append(prompt)
            names = {tool.name for tool in info.function_tools}
            if prompt == 'parent':
                assert 'delegate_task' in names
                return ModelResponse(parts=[ToolCallPart('delegate_task', {'agent_name': name, 'task': 'child'})])
            if name in ('Explore', 'Plan'):
                assert names == {'read_file', 'list_directory', 'search_files', 'find_files', 'file_info'}
                assert not {'shell', 'run_code', 'write_file', 'edit_file', 'delegate_task'} & names
            else:
                assert {'shell', 'write_file', 'delegate_task'} <= names
        return ModelResponse(parts=[TextPart('evidence')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for index, part in enumerate(respond(messages, info).parts):
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    session = Session(
        create_stock_agent(FunctionModel(function=respond, stream_function=stream)),
        deps=None,
        workspace=tmp_path,
        plugins=[Coder(repo_context=False)],
    )
    session.delegations = ui.owner
    async with ui.owner.opened():
        result = await session.prompt('parent')
        assert result.output == 'evidence'
        (record,) = ui.owner.records.values()
        assert record.outcome == 'ok', record.output
        assert record.resumable == (name == 'general-purpose')
        assert prompts == ['parent', 'child']


def test_detail_wraps_cell_width_and_refreshes() -> None:
    record = task()
    record.prompt = '界' * 80
    records = [record]
    sizes = [(60, 20)]

    def poll() -> str:
        record.status, record.outcome, record.output = 'finished', 'ok', 'completed result'
        sizes[0] = (25, 20)
        return ''

    picker = TaskMenu(snapshot=lambda: records, action=lambda name, task_id: None, key_source=poll)
    detail = TaskDetail(picker=picker, task_id=record.id, size=lambda: sizes[0])
    assert all(cell_len(item.label) <= 55 for item in detail.items())
    menu = detail.build()
    # Drive the public widget with a controlled input source and stream.
    assert '界' * 80 in ''.join(item.label for item in detail.items())
    poll()
    assert all(cell_len(item.label) <= 20 for item in detail.items())
    assert 'completed result' in ''.join(item.label for item in detail.items())
    assert menu.highlighted is not None


def test_task_menu_flow_and_action_errors() -> None:
    record = task()
    actions: list[tuple[str, str]] = []

    def action(name: str, task_id: str) -> None:
        actions.append((name, task_id))
        if name == 'stop':
            raise ValueError('already stopped')

    picker = TaskMenu(snapshot=lambda: [record], action=action)
    menu = picker.build()
    item = picker.items()[0]
    picker.act(menu, item, 'background')
    assert 'background requested' in picker.notice
    picker.act(menu, item, 'stop')
    assert picker.notice == 'already stopped'
    picker.act(menu, MenuItem('empty', value=42), 'stop')
    assert len(actions) == 2
    calls = 0

    def choose(menu: Menu) -> MenuResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return MenuResult(item=item)
        assert menu.highlighted is not None and menu.highlighted.value == record.id
        return MenuResult(cancelled=True)

    details: list[Menu] = []

    def inspect(menu: Menu) -> MenuResult:
        details.append(menu)
        return MenuResult(cancelled=True)

    run_tasks(picker, runners=Runners(run_list=choose, run_choice=inspect))
    assert len(details) == 1
    run_tasks(picker, runners=Runners(run_list=lambda menu: MenuResult()))
    run_tasks(picker, runners=Runners(run_list=lambda menu: MenuResult(item=MenuItem('invalid', value=42))))


async def test_open_tasks_injectable_runner() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)

    def close(menu: Menu) -> MenuResult:
        assert menu.highlighted is not None
        assert menu.highlighted.disabled
        return MenuResult(cancelled=True)

    assert await open_tasks(ui, runners=Runners(run_list=close)) == ''


@pytest.mark.parametrize('detail', [False, True])
def test_widget_polling_controls_and_resize(monkeypatch: pytest.MonkeyPatch, detail: bool) -> None:
    monkeypatch.setattr('termflow.tui.menu.raw_mode', nullcontext)

    def alternate(output: object) -> nullcontext[None]:
        return nullcontext()

    monkeypatch.setattr('termflow.tui.menu.alt_screen', alternate)
    record = task()
    keys = iter(['', 'down', 'b', 'x', ']', '[', 'end', 'home', 'escape'])
    actions: list[str] = []

    def poll() -> str:
        key = next(keys)
        record.status, record.outcome, record.output = 'finished', 'ok', 'live completed'
        return key

    picker = TaskMenu(snapshot=lambda: [record], action=lambda name, task_id: actions.append(name), key_source=poll)
    widget = TaskDetail(picker=picker, task_id=record.id).build() if detail else picker.build()
    assert widget.run().cancelled
    assert actions == ['background', 'stop']
    assert 'live completed' in picker.preview(picker.items()[0])


async def test_partial_stream_and_lifecycle_routing() -> None:
    output = io.StringIO()
    console = Console(file=output)
    ui = Tasks(console=console, conversation_id=lambda: 'root', directory=None)
    record = task()
    ui.owner.records[record.id] = record
    assert ui.snapshots()[0].messages == []
    start = PartStartEvent(index=0, part=TextPart('first'))
    await ui.observe(DelegationTaskEvent(task=record, event=start))
    await ui.observe(DelegationTaskEvent(task=record, event=PartDeltaEvent(index=0, delta=TextPartDelta(' second'))))
    assert 'first second' in str(ui.snapshots()[0].messages)
    assert record.messages == []
    await ui.observe(DelegationTaskEvent(task=record, event=PartEndEvent(index=0, part=TextPart('first second'))))
    assert ui.snapshots()[0].messages == []
    assert task_row(start) is None
    event = DelegationStartEvent(agent_name='self', task='hello', truncated=False, inherits_tools=True, model=None)
    event.task_id = record.id
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    assert 'general-purpose' not in output.getvalue()
    await ui.observe(DelegationTaskEvent(task=record, event=start, model_name='child-model'))
    assert 'general-purpose [aaaaaaaa] · child-model hello' in output.getvalue()
    assert output.getvalue().count('general-purpose') == 1
    received: list[object] = []

    async def sink(event: object) -> None:
        received.append(event)

    ui.sink = sink
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    assert received == []
    await ui.observe(DelegationTaskEvent(task=record, event=start, model_name='other-model'))
    assert received == [event]
    record.conversation_id = 'other'
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    assert received == [event]
    record.conversation_id = 'root'
    record.status = 'finished'
    await ui.observe(DelegationTaskEvent(task=record))
    assert ui.rows('*') == ()
    assert 'No foreground' in ui.promote()
    assert ui.resolve('aaaa') is record
    with pytest.raises(ValueError, match='ambiguous'):
        ui.resolve('z')
    record.status = 'running'
    record.backgroundable = False
    assert 'workspace' in ui.promote()


@pytest.mark.parametrize('show_model', [True, False])
async def test_task_start_row_waits_for_model_unless_hidden(show_model: bool) -> None:
    output = io.StringIO()
    ui = Tasks(
        console=Console(file=output),
        conversation_id=lambda: 'root',
        directory=None,
        show_task_models=lambda: show_model,
    )
    record = task()
    event = DelegationStartEvent(agent_name='worker', task='inspect', truncated=False, inherits_tools=False, model=None)
    event.task_id = record.id
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    assert ('worker [aaaaaaaa] inspect' in output.getvalue()) is not show_model
    await ui.observe(
        DelegationTaskEvent(task=record, event=PartStartEvent(index=0, part=TextPart('ok')), model_name='actual-model')
    )
    assert output.getvalue().count('worker [aaaaaaaa]') == 1
    assert ('· actual-model' in output.getvalue()) is show_model
    await ui.observe(
        DelegationTaskEvent(task=record, event=PartEndEvent(index=0, part=TextPart('ok')), model_name='actual-model')
    )
    assert output.getvalue().count('worker [aaaaaaaa]') == 1


@pytest.mark.parametrize('first', ['end', 'question'])
async def test_task_ending_or_asking_before_its_model_gets_its_start_row_without_one(
    first: Literal['end', 'question'],
) -> None:
    output = io.StringIO()
    ui = Tasks(console=Console(file=output), conversation_id=lambda: 'root', directory=None)
    record = task()
    event = DelegationStartEvent(agent_name='worker', task='inspect', truncated=False, inherits_tools=False, model=None)
    event.task_id = record.id
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    released: AgentStreamEvent = (
        DelegationEndEvent(
            agent_name='worker',
            outcome='failed',
            output='failed',
            truncated=False,
            usage=None,
            duration_seconds=1,
            task_id=record.id,
        )
        if first == 'end'
        else TaskQuestionEvent(task_id=record.id, agent_name='worker', question_id='q', question='Which?')
    )
    await ui.observe(DelegationTaskEvent(task=record, event=released))
    lines = output.getvalue().splitlines()
    assert lines[0] == '● worker [aaaaaaaa] inspect'
    assert lines[1] == (
        '  └ failed · 1.0s · /tasks to inspect' if first == 'end' else '? worker [aaaaaaaa] asks: Which?'
    )
    await ui.observe(DelegationTaskEvent(task=record, model_name='too-late'))
    assert output.getvalue().splitlines() == lines


async def test_resumed_task_ending_before_its_model_does_not_show_the_earlier_model() -> None:
    output = io.StringIO()
    ui = Tasks(console=Console(file=output), conversation_id=lambda: 'root', directory=None)
    record = task()

    def start() -> DelegationTaskEvent:
        event = DelegationStartEvent(
            agent_name='worker', task='inspect', truncated=False, inherits_tools=False, model=None
        )
        event.task_id = record.id
        return DelegationTaskEvent(task=record, event=event)

    end = DelegationEndEvent(
        agent_name='worker',
        outcome='failed',
        output='failed',
        truncated=False,
        usage=None,
        duration_seconds=1,
        task_id=record.id,
    )
    await ui.observe(start())
    await ui.observe(
        DelegationTaskEvent(task=record, event=PartStartEvent(index=0, part=TextPart('ok')), model_name='old-model')
    )
    await ui.observe(DelegationTaskEvent(task=record, event=end))
    await ui.observe(start())
    await ui.observe(DelegationTaskEvent(task=record, event=end))
    starts = [line for line in output.getvalue().splitlines() if line.startswith('●')]
    assert starts == ['● worker [aaaaaaaa] · old-model inspect', '● worker [aaaaaaaa] inspect']


async def test_presentation_classifies_capability_owned_tool() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('renamed_delegate', {'agent_name': 'worker', 'task': 'go'})])
        if len(messages) == 3:
            return ModelResponse(parts=[ToolCallPart('list_tasks', {})])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        for part in respond(messages, info).parts:
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    events: list[AgentStreamEvent] = []

    async def receive(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    from pydantic_ai.models.test import TestModel

    child = Agent(TestModel(custom_output_text='child'), deps_type=object, name='worker')
    parent = Agent(
        FunctionModel(function=respond, stream_function=stream),
        deps_type=object,
        capabilities=[
            SubAgents(
                agents=[SubAgent(child)], tool_name='renamed_delegate', agent_folders=None, task_control=TaskControl()
            ),
            TaskPresentation(),
        ],
    )
    await parent.run('go', event_stream_handler=receive)
    calls = [event for event in events if isinstance(event, FunctionToolCallEvent)]
    # The delegate tool shows as the task's rows, and a task-control tool of the same capability as a tool.
    assert [(event.part.tool_name, isinstance(event, DelegationToolCallEvent)) for event in calls] == [
        ('renamed_delegate', True),
        ('list_tasks', False),
    ]


async def test_open_tasks_controls_on_application_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('termflow.tui.menu.raw_mode', nullcontext)

    def alternate(output: object) -> nullcontext[None]:
        return nullcontext()

    monkeypatch.setattr('termflow.tui.menu.alt_screen', alternate)
    keys = iter(['b', 'x', 'escape'])
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)

    def run(menu: Menu) -> MenuResult:
        # TaskMenu's default source resolves menu_key on construction; replace the reader's dependency.
        return menu.run()

    def read(*, timeout: float) -> str:
        return next(keys)

    monkeypatch.setattr('pydantic_clai2.ui.menus.menu_worker.read_key', read)

    async def child(record: DelegationTask) -> str:
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    async with ui.owner.opened():
        await ui.owner.delegate(
            agent_name='worker', prompt='', conversation_id='root', model=None, background=True, resume=None, run=child
        )
        assert await open_tasks(ui, runners=Runners(run_list=run)) == ''
        (record,) = ui.owner.records.values()
        assert record.user_stopped


async def test_open_tasks_does_not_background_a_prompted_task(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backgrounded, a prompted task would report to the conversation, which never asked for it."""
    keys = iter(['b', 'escape'])

    def alternate(output: object) -> nullcontext[None]:
        return nullcontext()

    def read(*, timeout: float) -> str:
        return next(keys)

    monkeypatch.setattr('termflow.tui.menu.raw_mode', nullcontext)
    monkeypatch.setattr('termflow.tui.menu.alt_screen', alternate)
    monkeypatch.setattr('pydantic_clai2.ui.menus.menu_worker.read_key', read)
    gate = asyncio.Event()

    async def work(record: DelegationTask) -> str:
        await gate.wait()
        return 'again'

    async def prompt_task(record: DelegationTask, text: str) -> str:
        return await ui.owner.delegate(
            agent_name=record.agent_name,
            prompt=text,
            conversation_id='root',
            model=None,
            background=False,
            resume=record.id,
            run=work,
        )

    ui = Tasks(
        console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None, prompt_task=prompt_task
    )
    record = task()
    record.status, record.outcome = 'finished', 'ok'
    ui.owner.records[record.id] = record
    async with ui.router.opened(view=ui):
        await ui.prompt(record.id, 'and the docs?')
        assert await open_tasks(ui, runners=Runners(run_list=lambda menu: menu.run())) == ''
        assert (record.status, record.background) == ('running', False)
        gate.set()


async def test_promote_all_foreground_siblings() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    started = asyncio.Event()

    async def child(record: DelegationTask) -> str:
        started.set()
        await asyncio.Event().wait()
        return 'unreachable'  # pragma: no cover

    async with ui.owner.opened():
        pending = asyncio.create_task(
            ui.owner.delegate(
                agent_name='worker',
                prompt='',
                conversation_id='root',
                model=None,
                background=False,
                resume=None,
                run=child,
            )
        )
        await started.wait()
        assert '1 task(s) moved' in ui.promote()
        assert 'not a result' in await pending


async def test_a_question_shows_once() -> None:
    """Shown as the asking task's own event, not as the copy in the stream of the run that delegated it."""
    output = io.StringIO()
    ui = Tasks(console=Console(file=output, width=200), conversation_id=lambda: 'root', directory=None)
    record = task()
    ui.owner.records[record.id] = record
    question = TaskQuestionEvent(task_id=record.id, agent_name='self', question_id='q', question='Which?')
    received: list[object] = []

    async def sink(event: object) -> None:
        received.append(event)

    ui.sink = sink
    await ui.observe(DelegationTaskEvent(task=record, event=question))
    assert received == [question]
    child = task(task_id='b' * 32, parent_id=record.id)
    ui.owner.records[child.id] = child
    nested = TaskQuestionEvent(task_id=child.id, agent_name='self', question_id='q', question='Which?')
    await ui.observe(DelegationTaskEvent(task=record, event=nested))
    assert received == [question], 'not the copy in its parent task stream'
    await ui.observe(DelegationTaskEvent(task=child, event=nested))
    assert received == [question, nested]
    ui.sink = None
    await ui.observe(DelegationTaskEvent(task=record, event=question))
    assert 'asks: Which?' in output.getvalue()


def test_question_row() -> None:
    event = TaskQuestionEvent(task_id='a' * 32, agent_name='self', question_id='q', question='Which\nflavour?')
    row = task_row(event)
    assert row is not None
    assert row.plain == '? general-purpose [aaaaaaaa] asks: Which flavour?'


@pytest.mark.parametrize('success', [True, False])
def test_terminal_row_outcomes(success: bool) -> None:
    event = DelegationEndEvent(
        agent_name='worker',
        outcome='ok' if success else 'failed',
        output='untrusted',
        truncated=False,
        usage=None,
        duration_seconds=1.0,
    )
    row = task_row(event)
    if success:
        assert row is None
    else:
        assert row is not None
        assert '/tasks to inspect' in row.plain and 'untrusted' not in row.plain


async def test_presentation_keeps_ordinary_tool_events() -> None:
    from pydantic_ai.models.test import TestModel

    def ordinary() -> str:
        return 'tool result'

    agent = Agent(TestModel(), deps_type=object, tools=[ordinary], capabilities=[TaskPresentation()])
    events: list[AgentStreamEvent] = []

    async def receive(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    await agent.run('go', event_stream_handler=receive)
    assert any(isinstance(event, FunctionToolCallEvent) for event in events)
    assert not any(isinstance(event, DelegationToolCallEvent) for event in events)


async def test_task_commands_and_plugin_lifetime(tmp_path: Path) -> None:

    from pydantic_clai2.ui.prompt.live_prompt import LivePrompt

    shell = shell_for(tmp_path, ForkModel(), io.StringIO())
    assert shell.tasks.owner.directory == tmp_path / 'sessions.db.tasks'
    record = task()
    record.conversation_id = shell.session.summary.id
    shell.tasks.owner.records[record.id] = record
    assert shell.plugins_busy('/plugins list')
    assert not shell.plugins_busy('/tasks')
    with pytest.raises(ValueError, match='Usage'):
        await shell.tasks_command(['bad'])
    async with shell.tasks.owner.opened():
        assert 'Stopping' in await shell.tasks_command(['stop', 'aaaa'])
        assert record.user_stopped
        assert not shell.plugins_busy('/plugins list')
        assert 'may now be resumed' in await shell.tasks_command(['resume', 'aaaa'])
        assert not record.user_stopped
        assert 'moved to background' in await shell.tasks_command(['background', 'aaaa'])
        shell.editor = LivePrompt(
            history=InMemoryHistory(),
            console=shell.console,
            commands=shell.commands,
            images=shell.images,
            interrupts=shell.interrupts,
            toolbar=lambda: [],
        )
        assert 'Resume requested' in await shell.tasks_command(['resume', 'aaaa'])
        assert record.id in shell.editor.queued_messages[0]
    await shell.forks.close()


async def test_tasks_without_arguments_opens_the_live_task_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def escape(timeout: float) -> str:
        return 'escape'

    monkeypatch.setattr('termflow.tui.menu.raw_mode', nullcontext)
    monkeypatch.setattr('pydantic_clai2.ui.menus.menu_worker.read_key', escape)
    shell = shell_for(tmp_path, ForkModel(), io.StringIO())
    record = task(conversation_id=shell.session.summary.id)
    shell.tasks.owner.records[record.id] = record
    assert await shell.commands.execute_async('/tasks') == ''
    assert 'worker [aaaaaaaa] running' in capsys.readouterr().out
    await shell.forks.close()


async def test_live_partial_tail_is_bounded_without_changing_history() -> None:
    ui = Tasks(console=Console(file=io.StringIO()), conversation_id=lambda: 'root', directory=None)
    record = task()
    ui.owner.records[record.id] = record
    text = 'x' * 100000
    await ui.observe(DelegationTaskEvent(task=record, event=PartStartEvent(index=0, part=TextPart(text))))
    assert len(ui.partial[record.id]) == 65536
    assert 'Live preview tail' in str(ui.snapshots()[0].messages)
    assert not record.messages
    record.messages = [ModelResponse(parts=[TextPart(text)])]
    await ui.observe(DelegationTaskEvent(task=record, event=PartEndEvent(index=0, part=TextPart(text))))
    assert ui.snapshots()[0].messages == record.messages
    assert not ui.partial_truncated


async def test_main_user_interrupt_marks_foreground_child_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json
    import signal

    from pydantic_clai2 import chat
    from pydantic_clai2.config.settings_store import SettingsStore
    from tests.clai2.test_app_edges import inputs

    inputs(monkeypatch, ['parent', '/exit'])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if prompt == 'child':
            signal.raise_signal(signal.SIGINT)
            await asyncio.Event().wait()
        elif len(messages) == 1:
            yield {0: DeltaToolCall(name='delegate_task', json_args='{"agent_name":"self","task":"child"}')}
        else:
            yield 'parent'  # pragma: lax no cover

    await chat(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Coder(repo_context=False)],
        builtin_plugins=(),
        store=SettingsStore(tmp_path / 'config.db'),
        console=Console(file=io.StringIO()),
    )
    files = list((tmp_path / 'sessions.db.tasks').glob('*.json'))
    assert len(files) == 1
    record = json.loads(files[0].read_text())
    assert record['user_stopped'] is True
    assert record['outcome'] == 'cancelled'


async def test_managed_code_mode_delegation_has_one_typed_row(tmp_path: Path) -> None:
    pytest.importorskip('pydantic_monty')
    from pydantic_clai2.runtime.sandbox_calls import DelegationCallStartedEvent, SandboxCallOrder
    from pydantic_clai2.runtime.speculation import SpeculationCounters
    from pydantic_clai2.runtime.speculative_mode import speculative_capabilities
    from pydantic_clai2.ui.rendering._rendering import StreamRenderer
    from tests.clai2.test_speculative_mode import streamed

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        prompt = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if prompt == 'parent' and len(messages) == 1:
            return ModelResponse(
                parts=[ToolCallPart('run_code', {'code': 'await delegate_task(agent_name="self", task="child")'})]
            )
        return ModelResponse(parts=[TextPart('evidence')])

    output = io.StringIO()
    console = Console(file=output)
    ui = Tasks(console=console, conversation_id=lambda: 'root', directory=None)
    renderer = StreamRenderer(console, stop_loading=lambda: None, renderers=[ui.row])
    ui.sink = renderer.on_stream_event
    seen: list[AgentStreamEvent] = []

    async def receive(event: AgentStreamEvent) -> None:
        seen.append(event)
        await renderer.on_stream_event(event)

    coder = Coder[None](repo_context=False)
    session = Session(
        create_stock_agent(streamed(respond)),
        deps=None,
        workspace=tmp_path,
        plugins=[coder, *speculative_capabilities(SpeculationCounters(), (coder,))],
        on_stream_event=receive,
    )
    ui.conversation_id = lambda: session.summary.id
    session.delegations = ui.owner
    async with ui.owner.opened():
        assert (await session.prompt('parent')).output == 'evidence'
    await renderer.finish()
    starts = [event for event in seen if isinstance(event, DelegationCallStartedEvent)]
    assert len(starts) == 1
    assert SandboxCallOrder().tool_events(starts[0]) == []
    assert '● delegate_task' not in output.getvalue()
    assert output.getvalue().count('general-purpose [') == 1


@pytest.mark.parametrize('timing', ['idle', 'active', 'restored'])
async def test_background_completion_continues_parent_without_user_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timing: str
) -> None:
    import anyio
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from pydantic_ai_harness.subagents import DelegationTaskEvent
    from pydantic_clai2.config.project_settings import ProjectSettings
    from pydantic_clai2.config.settings_store import SettingsStore
    from pydantic_clai2.runtime.session_runtime import create_shell
    from pydantic_clai2.ui.prompt.live_prompt import LivePrompt

    child_started, release, idle, continued, finished = (asyncio.Event() for _ in range(5))
    captured: list[list[ModelMessage]] = []
    output = io.StringIO()
    reads = 0
    original_read = LivePrompt.read

    async def read(live: LivePrompt) -> str:
        nonlocal reads
        reads += 1
        if reads == 2:
            idle.set()
        return await original_read(live)

    monkeypatch.setattr(LivePrompt, 'read', read)

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        request = messages[0]
        assert isinstance(request, ModelRequest)
        first = next(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if first == 'child':
            child_started.set()
            await release.wait()
            yield 'Admiral Fluff commands the cheese fleet.'
        elif any(
            is_task_report(part) for message in messages if isinstance(message, ModelRequest) for part in message.parts
        ):
            captured.append(messages)
            yield 'Pirate hamster reporting for duty.'
            continued.set()
        elif len(messages) == 1:
            yield {
                0: DeltaToolCall(
                    name='delegate_task', json_args='{"agent_name":"self","task":"child","background":true}'
                )
            }
        else:
            if timing == 'active':
                await release.wait()
            yield 'Background work launched.'

    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Coder(repo_context=False)],
        usage_limits=None,
        console=Console(file=output, force_terminal=True, width=100, height=24),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )

    original_observer = shell.tasks.owner.observer

    async def observe(progress: DelegationTaskEvent) -> None:
        assert original_observer is not None
        await original_observer(progress)
        if progress.task.status == 'finished':
            finished.set()

    shell.tasks.owner.observer = observe
    if timing == 'restored':
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
            async with anyio.create_task_group() as group:
                group.start_soon(shell.run)
                pipe.send_text('launch\r')
                await idle.wait()
                shell.tasks.wake = None
                release.set()
                await finished.wait()
                assert shell.editor is not None
                shell.editor.submit('/exit')
        assert not captured
        shell = create_shell(
            create_stock_agent(FunctionModel(stream_function=stream)),
            deps=None,
            plugins=[Coder(repo_context=False)],
            usage_limits=None,
            console=Console(file=output, force_terminal=True, width=100, height=24),
            settings=None,
            store=SettingsStore(tmp_path / 'config.db'),
            builtin_plugins=(),
            project=ProjectSettings(),
            headless=True,
            summary=shell.session.summary,
            message_history=shell.session.messages,
        )
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(shell.run)
            if timing != 'restored':
                pipe.send_text('launch\r')
                await child_started.wait()
                if timing == 'idle':
                    await idle.wait()
                assert shell.editor is not None
                shell.editor.buffer.replace('keep this unfinished draft')
                release.set()
            await continued.wait()
            assert shell.editor is not None
            assert shell.editor.buffer.text == ('' if timing == 'restored' else 'keep this unfinished draft')
            shell.editor.submit('/exit')
    assert len(captured) == 1
    prompts = [
        part.content
        for message in shell.session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not is_task_report(part)
    ]
    assert prompts == ['launch']
    # The report rides the user channel, but a restored transcript doesn't show it as something the user typed.
    assert len(turn_starts(shell.session.messages)) == 1
    assert [item.text for item in history_items(shell.session.messages) if item.kind == 'user'] == ['launch']
    (record,) = shell.tasks.owner.records.values()
    assert record.delivered
    last = shell.session.messages[-1]
    assert isinstance(last, ModelResponse)
    assert any(
        isinstance(part, TextPart) and part.content == 'Pirate hamster reporting for duty.' for part in last.parts
    )
    assert shell.tasks.wake is None


async def test_consumed_background_report_does_not_start_another_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from pydantic_clai2.ui.prompt.live_prompt import LivePrompt, PromptWakeup

    shell = shell_for(tmp_path, ForkModel(), io.StringIO())
    shell.console = Console(file=io.StringIO(), force_terminal=True, width=100, height=24)
    Terminal[None, str](console=shell.console, transcript=TranscriptBuffer(), history=InMemoryHistory()).attach(shell)
    reads = 0

    async def read(live: LivePrompt) -> str:
        nonlocal reads
        reads += 1
        if reads == 1:
            raise PromptWakeup
        raise EOFError

    monkeypatch.setattr(LivePrompt, 'read', read)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        assert await shell.run() == 'eof'
    assert reads == 2
    assert shell.session.messages == []


def task_texts(request: ModelMessage) -> list[str]:
    """The task reports and questions a request carries."""
    assert isinstance(request, ModelRequest)
    return [
        item.content
        for part in request.parts
        if is_task_report(part)
        for item in part.content
        if isinstance(item, TextContent)
    ]


@pytest.mark.parametrize('timing', ['idle', 'active'])
async def test_a_background_task_question_reaches_the_parent_which_answers_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timing: Literal['idle', 'active']
) -> None:
    """The question shows once, whether it comes during a turn or wakes the idle parent, as a report does."""
    release, idle, continued, asked = (asyncio.Event() for _ in range(4))
    activities: list[str] = []
    output = io.StringIO()
    reads = 0
    original_read = LivePrompt.read

    async def read(live: LivePrompt) -> str:
        nonlocal reads
        reads += 1
        if reads == 2:
            idle.set()
        return await original_read(live)

    monkeypatch.setattr(LivePrompt, 'read', read)
    questions = 0

    def counted(event: AgentStreamEvent, *, model_name: str | None = None, show_model: bool = True) -> Text | None:
        nonlocal questions
        questions += isinstance(event, TaskQuestionEvent)
        return task_row(event, model_name=model_name, show_model=show_model)

    # The task panel's rows and the main run's stream both render through `Tasks.row`, which calls it.
    monkeypatch.setattr('pydantic_clai2.runtime.tasks.task_row', counted)

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        first, last = messages[0], messages[-1]
        assert isinstance(first, ModelRequest) and isinstance(last, ModelRequest)
        prompt = next(part.content for part in first.parts if isinstance(part, UserPromptPart))
        if prompt == 'child':
            if len(messages) == 1:
                await release.wait()
                yield {0: DeltaToolCall(name='ask_parent', json_args='{"question":"Which flavour?"}')}
            else:
                answer = next(part.content for part in last.parts if isinstance(part, ToolReturnPart))
                yield f'Chose {answer}'
        elif any(text.startswith('<task_output ') for text in task_texts(last)):
            yield 'Report read.'
            continued.set()
        elif question := next((text for text in task_texts(last) if text.startswith('<task_question ')), None):
            match = re.search(r'task_id="([0-9a-f]{32})"', question)
            assert match is not None
            arguments = f'{{"task_id":"{match[1]}","answer":"cheddar"}}'
            yield {0: DeltaToolCall(name='answer_task', json_args=arguments)}
        elif len(messages) == 1:
            arguments = '{"agent_name":"self","task":"child","background":true}'
            yield {0: DeltaToolCall(name='delegate_task', json_args=arguments)}
        else:
            if timing == 'active' and len(messages) == 3:
                # The question comes while this turn runs, and is queued on it.
                release.set()
                await asked.wait()
            yield 'Waiting.'

    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Coder(repo_context=False, task_control=TaskControl())],
        usage_limits=None,
        console=Console(file=output, force_terminal=True, width=100, height=24),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    original_observer = shell.tasks.owner.observer

    async def observe(update: DelegationTaskEvent) -> None:
        assert original_observer is not None
        await original_observer(update)
        activities.append(shell.tasks.info(update.task).activity)
        # The owner tells its observer once the question is pending.
        if isinstance(update.event, TaskQuestionEvent):
            asked.set()

    shell.tasks.owner.observer = observe
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(shell.run)
            pipe.send_text('launch\r')
            await idle.wait()
            release.set()
            await continued.wait()
            assert shell.editor is not None
            shell.editor.submit('/exit')
    (record,) = shell.tasks.owner.records.values()
    assert (record.outcome, record.output) == ('ok', 'Chose cheddar')
    assert 'asks: Which flavour?' in output.getvalue()
    assert questions == 1
    # Once the question is pending, the task's row says it waits.
    assert 'waiting for an answer' in activities
    prompts = [
        part.content
        for message in shell.session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not is_task_report(part)
    ]
    assert prompts == ['launch']
    # The question rides the user channel too, but a restored transcript doesn't show it as something the user typed.
    assert len(turn_starts(shell.session.messages)) == 1
    assert [item.text for item in history_items(shell.session.messages) if item.kind == 'user'] == ['launch']


async def test_a_custom_agent_with_task_control_shows_its_tasks_questions(tmp_path: Path) -> None:
    """Without the tasks' feed, the run's own stream shows the question."""

    async def child(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='ask_parent', json_args='{"question":"Which flavour?"}')}
        else:
            yield 'Chose it.'

    async def parent(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        returns = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        ]
        if not returns:
            yield {0: DeltaToolCall(name='delegate_task', json_args='{"agent_name":"helper","task":"Pick."}')}
        elif len(returns) == 1:
            match = re.search(r'([0-9a-f]{32})', str(returns[0]))
            assert match is not None
            yield {0: DeltaToolCall(name='answer_task', json_args=f'{{"task_id":"{match[1]}","answer":"cheddar"}}')}
        else:
            yield 'Done.'

    helper = Agent(FunctionModel(stream_function=child), name='helper')
    agent = Agent(
        FunctionModel(stream_function=parent),
        capabilities=[SubAgents(agents=[SubAgent(helper, description='Helps.')], task_control=TaskControl())],
    )
    output = io.StringIO()
    shell = create_shell(
        agent,
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=output, width=200),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    with anyio.fail_after(10):
        ended = await shell.run_turn(TurnStart(text='Go.'))
    assert ended.outcome == 'completed', ended.error
    assert output.getvalue().count('asks: Which flavour?') == 1


async def test_a_question_asked_after_a_turns_last_read_starts_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A question asked once the turn read its last queued message, as it still receives, waits for the next turn."""
    ending, queued, idle, continued = (asyncio.Event() for _ in range(4))
    reads = 0
    original_read = LivePrompt.read

    async def read(live: LivePrompt) -> str:
        nonlocal reads
        reads += 1
        if reads == 2:
            idle.set()
        return await original_read(live)

    monkeypatch.setattr(LivePrompt, 'read', read)

    @dataclass
    class AskAfterTheLastRead(AbstractCapability[None]):
        """Holds the first turn's end, past its last read of queued messages, until the child asked."""

        held: bool = False

        async def after_run(self, ctx: RunContext[None], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            if not self.held:
                self.held = True
                ending.set()
                await queued.wait()
            return result

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        first, last = messages[0], messages[-1]
        assert isinstance(first, ModelRequest) and isinstance(last, ModelRequest)
        prompt = next(part.content for part in first.parts if isinstance(part, UserPromptPart))
        if prompt == 'child':
            if len(messages) == 1:
                await ending.wait()
                yield {0: DeltaToolCall(name='ask_parent', json_args='{"question":"Which flavour?"}')}
            else:
                yield f'Chose {next(part.content for part in last.parts if isinstance(part, ToolReturnPart))}'
        elif question := next((text for text in task_texts(last) if text.startswith('<task_question ')), None):
            match = re.search(r'task_id="([0-9a-f]{32})"', question)
            assert match is not None
            yield {0: DeltaToolCall(name='answer_task', json_args=f'{{"task_id":"{match[1]}","answer":"cheddar"}}')}
        elif any(text.startswith('<task_output ') for text in task_texts(last)):
            yield 'Report read.'
            continued.set()
        elif len(messages) == 1:
            yield {
                0: DeltaToolCall(
                    name='delegate_task', json_args='{"agent_name":"self","task":"child","background":true}'
                )
            }
        else:
            yield 'Started.'

    shell = create_shell(
        create_stock_agent(FunctionModel(stream_function=stream)),
        deps=None,
        plugins=[Coder(repo_context=False, task_control=TaskControl()), AskAfterTheLastRead()],
        usage_limits=None,
        console=Console(file=io.StringIO(), force_terminal=True, width=100, height=24),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    original_observer = shell.tasks.owner.observer

    async def observe(update: DelegationTaskEvent) -> None:
        assert original_observer is not None
        await original_observer(update)
        if isinstance(update.event, TaskQuestionEvent):
            queued.set()

    shell.tasks.owner.observer = observe
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(shell.run)
            pipe.send_text('launch\r')
            await idle.wait()
            # Nothing was typed: the question woke the prompt for the turn that answered it, and the report the next.
            await continued.wait()
            assert shell.editor is not None
            shell.editor.submit('/exit')
    (record,) = shell.tasks.owner.records.values()
    assert (record.outcome, record.output) == ('ok', 'Chose cheddar')

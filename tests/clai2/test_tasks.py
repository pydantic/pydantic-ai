"""Task rendering, live inspection, and stock-agent integration."""

import io
import time
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import nullcontext
from pathlib import Path

import pytest
from rich.cells import cell_len
from rich.console import Console
from termflow.tui import MenuItem
from termflow.tui.menu import Menu, MenuResult

from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
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
)
from pydantic_clai2._app import create_stock_agent
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.sandbox_calls import DelegationToolCallEvent
from pydantic_clai2.runtime.tasks import TaskPresentation, Tasks, task_row, task_tree
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.menus.task_menu import TaskDetail, TaskMenu, open_tasks, run_tasks
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
    await ui.observe(DelegationTaskEvent(task=record, event=event))
    assert 'general-purpose' in output.getvalue()
    received: list[object] = []

    async def sink(event: object) -> None:
        received.append(event)

    ui.sink = sink
    await ui.observe(DelegationTaskEvent(task=record, event=event))
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


async def test_presentation_classifies_capability_owned_tool() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('renamed_delegate', {'agent_name': 'worker', 'task': 'go'})])
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
            SubAgents(agents=[SubAgent(child)], tool_name='renamed_delegate', agent_folders=None),
            TaskPresentation(),
        ],
    )
    await parent.run('go', event_stream_handler=receive)
    assert any(isinstance(event, DelegationToolCallEvent) for event in events)


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
    import asyncio

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


async def test_promote_all_foreground_siblings() -> None:
    import asyncio

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
    from pydantic_ai.messages import FunctionToolCallEvent
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
    from prompt_toolkit.history import InMemoryHistory

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
    import asyncio
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
    renderer = StreamRenderer(console, stop_loading=lambda: None, renderers=[task_row])
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
    import asyncio

    import anyio
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from pydantic_ai.messages import SystemPromptPart
    from pydantic_ai_harness.subagents import DelegationTaskEvent
    from pydantic_clai2._app import create_shell
    from pydantic_clai2.config.project_settings import ProjectSettings
    from pydantic_clai2.config.settings_store import SettingsStore
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
            isinstance(part, (SystemPromptPart, UserPromptPart))
            and isinstance(part.content, str)
            and 'Automated subagent task report' in part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
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
        if isinstance(part, UserPromptPart)
    ]
    assert prompts == ['launch']
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

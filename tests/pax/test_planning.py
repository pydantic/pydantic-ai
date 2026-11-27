"""The `planning` built-in keeps one plan per saved conversation and draws it after each change."""

import io
import stat
from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path

import pytest
from inline_snapshot import snapshot
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.planning import PlanCreatedEvent, PlanItem, Planning, SqlitePlanStore, TaskStatus
from pydantic_ai_harness.step_persistence.conversations import SqliteSessionStore
from pydantic_ai_pax import DEFAULT_PLUGINS
from pydantic_ai_pax.builtin_plugins.planning import PLAN_DATABASE, PlanningPlugin, render_plan_panel
from pydantic_ai_pax.commands import Commands
from pydantic_ai_pax.config import Settings
from pydantic_ai_pax.config.settings_store import SettingsStore
from pydantic_ai_pax.plugins import LoadedPlugin, PluginHost, SessionStart, load_plugin
from pydantic_ai_pax.plugins.loader import PluginLoader
from pydantic_ai_pax.runtime.session import Session
from pydantic_ai_pax.ui.rendering.rendering import StreamRenderer

PLAN = [
    {'id': 'migrate', 'content': 'Add the migration', 'status': 'completed'},
    {'id': 'endpoint', 'content': 'Add the endpoint', 'status': 'in_progress', 'active_form': 'Adding the endpoint'},
    {'id': 'tests', 'content': 'Write the tests'},
]

CALLS = {
    'plan': ToolCallPart('write_plan', {'items': PLAN}),
    'cancel': ToolCallPart('write_plan', {'items': [*PLAN[:2], {**PLAN[2], 'status': 'cancelled'}]}),
    'drop': ToolCallPart('write_plan', {'items': PLAN[:2]}),
    'add': ToolCallPart('add_task', {'content': 'Ship it'}),
    'read': ToolCallPart('read_plan', {}),
}
"""The plan tool call the model answers each prompt with."""


class Script:
    """Answers each prompt with its plan tool call, then records what the tool returned."""

    def __init__(self) -> None:
        self.results: list[str] = []
        self.tools: list[list[str]] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        part = request.parts[-1]
        if isinstance(part, UserPromptPart):
            self.tools.append([tool.name for tool in info.function_tools])
            assert isinstance(part.content, str)
            call = CALLS.get(part.content)
            return ModelResponse(parts=[call if call is not None else TextPart('hi')])
        assert isinstance(part, ToolReturnPart)
        self.results.append(str(part.content))
        return ModelResponse(parts=[TextPart('done')])

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        [part] = self.respond(messages, info).parts
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    def agent(self) -> Agent[None, str]:
        return Agent(FunctionModel(self.respond, stream_function=self.stream))


class Shell:
    """A saved session with the `planning` plugin loaded, drawing what the plugin renders."""

    def __init__(self, tmp_path: Path, script: Script, *, settings_database: Path | None) -> None:
        self.output = io.StringIO()
        self.console = Console(file=self.output, width=60, color_system=None)
        self.session = Session(
            script.agent(),
            deps=None,
            conversations=SqliteSessionStore(database=tmp_path / 'sessions.db'),
            workspace=tmp_path,
            on_stream_event=self.draw,
        )
        host = PluginHost[None](
            name='planning',
            console=self.console,
            settings={},
            conversation=self.session,
            settings_database=settings_database,
        )
        self.plugin: LoadedPlugin[None] = load_plugin(PlanningPlugin, host)
        self.session.plugins = self.plugin.capabilities

    async def start(self) -> None:
        await self.plugin.dispatch(SessionStart(agent=Agent(TestModel()), settings=Settings()))

    async def draw(self, event: AgentStreamEvent) -> None:
        renderable = self.plugin.plugin.render(event)
        if renderable is not None:
            self.console.print(renderable)

    def drawn(self) -> str:
        return self.output.getvalue()


async def test_the_plan_follows_the_conversation_through_resume_and_new(tmp_path: Path) -> None:
    script = Script()
    # The folder holding the settings database may not exist yet.
    settings_database = tmp_path / 'config' / 'config.db'
    first = Shell(tmp_path, script, settings_database=settings_database)
    await first.start()
    await first.session.prompt('plan')
    conversation_id = first.session.conversation_id

    # Another Pax resuming the conversation gets the plan back from `plan.db`.
    second = Shell(tmp_path, script, settings_database=settings_database)
    await second.start()
    await second.session.resume(conversation_id)
    await second.session.prompt('read')
    assert script.results[-1].endswith('Summary: 1 completed, 1 in progress, 1 pending')

    # `/new` starts without one, and leaves the resumed conversation's plan alone.
    await second.session.clear()
    await second.session.prompt('read')
    assert 'No plan yet.' in script.results[-1]
    database = settings_database.with_name(PLAN_DATABASE)
    stored = await SqlitePlanStore(str(database), session=conversation_id).get_items()
    assert [item.status for item in stored] == [TaskStatus.completed, TaskStatus.in_progress, TaskStatus.pending]
    # Created owner-only, like `sessions.db`.
    assert stat.S_IMODE(database.stat().st_mode) == 0o600


async def test_each_change_draws_the_plan_once(tmp_path: Path) -> None:
    script = Script()
    shell = Shell(tmp_path, script, settings_database=tmp_path / 'config.db')
    for prompt in ('plan', 'read', 'cancel', 'drop', 'drop', 'add'):
        await shell.session.prompt(prompt)
    # The steps keep their ids, so `cancel` only updates one, `drop` only deletes one, and `add` only creates
    # one. `read_plan` and the second `drop` change nothing, so they draw nothing.
    assert shell.drawn() == snapshot("""\
  ✓ Add the migration
  ◼ Adding the endpoint
  ◻ Write the tests
  ✓ Add the migration
  ◼ Adding the endpoint
  ✗ Write the tests
  ✓ Add the migration
  ◼ Adding the endpoint
  ✓ Add the migration
  ◼ Adding the endpoint
  ◻ Ship it
""")


async def test_only_the_plan_tools_draw_the_plan(tmp_path: Path) -> None:
    """Another tool's result arriving between a plan change and its tool's result, as in parallel calls, keeps its own display."""
    shell = Shell(tmp_path, Script(), settings_database=tmp_path / 'config.db')
    await shell.session.prompt('plan')
    plugin = shell.plugin.plugin
    [item, *_] = await SqlitePlanStore(str(tmp_path / PLAN_DATABASE), session=shell.session.conversation_id).get_items()
    assert plugin.render(PlanCreatedEvent(item=item)) is None
    assert plugin.render(FunctionToolResultEvent(part=ToolReturnPart('read_file', 'text'))) is None
    assert plugin.render(FunctionToolResultEvent(part=ToolReturnPart('add_task', 'added'))) is not None


def test_the_panel_marks_every_status() -> None:
    items = [
        PlanItem(content='Plan', status=TaskStatus.blocked),
        PlanItem(content='Run', status=TaskStatus.in_progress),
        PlanItem(content='Done', status=TaskStatus.completed, active_form='Doing'),
        PlanItem(content='Dropped', status=TaskStatus.cancelled),
        PlanItem(content='Line\nbreak'),
    ]
    rows = [row for row in render_plan_panel(items).renderables if isinstance(row, Text)]
    # A step shows its present-tense label only while it runs, and its content without one; a line break
    # cannot forge a row.
    assert [(row.plain, str(row.style), [str(span.style) for span in row.spans]) for row in rows] == snapshot(
        [
            ('  ! Plan', '#D0FF71', ['#D0FF71']),
            ('  ◼ Run', 'bold #E520E9', ['bold #E520E9']),
            ('  ✓ Done', '#8F888E', ['#8F888E strike']),
            ('  ✗ Dropped', '#8F888E', ['#8F888E strike']),
            ('  ◻ Line\\x0abreak', '', []),
        ]
    )


async def test_code_mode_draws_the_plan_under_the_sandbox_call(tmp_path: Path) -> None:
    """Pax folds the plan tools into `run_code` by default; the plan is drawn under the call made from inside it."""
    pytest.importorskip('pydantic_monty')
    from pydantic_ai_pax.runtime.speculation import SpeculationCounters
    from pydantic_ai_pax.runtime.speculative_mode import speculative_capabilities

    code = f'await write_plan(items={PLAN!r})'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code})])
        return ModelResponse(parts=[TextPart('Planned.')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        [part] = respond(messages, info).parts
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    shell = Shell(tmp_path, Script(), settings_database=tmp_path / 'config.db')
    renderer = StreamRenderer(
        shell.console, stop_loading=lambda: None, renderers=(shell.plugin.plugin.render,), smooth=False
    )
    shell.session.agent = Agent(FunctionModel(respond, stream_function=stream))
    shell.session.plugins = [
        *shell.plugin.capabilities,
        *speculative_capabilities(SpeculationCounters(), None, speculate=False),
    ]
    shell.session.on_stream_event = renderer.on_stream_event
    await shell.session.prompt('plan')
    await renderer.finish()
    assert shell.drawn() == snapshot("""\
● run_code code="await write_plan(items=[{'id': 'migrat…

● write_plan items=[{"id":"migrate","content":"Add the mig…

  ✓ Add the migration
  ◼ Adding the endpoint
  ◻ Write the tests

Planned.

""")


async def test_other_conversations_plan_in_memory(tmp_path: Path) -> None:
    """A run under another conversation ID, such as a delegated task, keeps its plan out of `plan.db` and the panel."""
    script = Script()
    shell = Shell(tmp_path, script, settings_database=tmp_path / 'config.db')

    async def draw(_ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            await shell.draw(event)

    await script.agent().run(
        'plan', capabilities=shell.plugin.capabilities, conversation_id='delegated', event_stream_handler=draw
    )
    assert script.results[-1].startswith('Plan updated: 3 step(s).')
    for conversation_id in ('delegated', shell.session.conversation_id):
        assert await SqlitePlanStore(str(tmp_path / PLAN_DATABASE), session=conversation_id).get_items() == []
    assert shell.drawn() == ''


async def test_a_host_without_a_settings_database_plans_in_memory(tmp_path: Path) -> None:
    script = Script()
    shell = Shell(tmp_path, script, settings_database=None)
    await shell.start()
    await shell.session.prompt('plan')
    await shell.session.prompt('read')
    assert 'No plan yet.' in script.results[-1]
    assert list(tmp_path.iterdir()) == [tmp_path / 'sessions.db']
    assert shell.drawn() == ''


async def test_the_disabled_builtin_adds_no_tools(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    [planning] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'planning']
    assert not planning.enabled
    plugins = PluginLoader[None](
        store=store,
        console=Console(file=io.StringIO()),
        commands=Commands(),
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=(planning,),
    )
    script = Script()
    try:
        await plugins.load_all()
        await script.agent().run('hello', capabilities=plugins.capabilities())
        await plugins.enable('planning')
        assert any(isinstance(capability, Planning) for capability in plugins.capabilities())
        await script.agent().run('hello', capabilities=plugins.capabilities())
    finally:
        await plugins.close('exit')
    assert script.tools == [
        [],
        ['write_plan', 'read_plan', 'add_task', 'update_task_status', 'update_task_statuses', 'remove_task'],
    ]
    # Loading it from the shell creates the plans' database beside the settings.
    assert (tmp_path / PLAN_DATABASE).exists()

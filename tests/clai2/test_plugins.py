"""Declarative plugins: settings, host events, capabilities, commands, and renderers."""

import io
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Generic, TypeVar

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, ValidationError
from rich.console import Console, RenderableType

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    CapabilityEvent,
    FunctionToolCallEvent,
    ModelRequest,
    RunContext,
    ToolCallPart,
    ToolDefinition,
)
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, Capability, Hooks, ValidatedToolArgs, on_event
from pydantic_ai.exceptions import PydanticAIDeprecationWarning
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_clai2 import Session, StreamRenderer, plugins as plugins_module
from pydantic_clai2.commands import Command
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import (
    InstallProgress,
    NoSettings,
    Plugin,
    PluginHost,
    SessionEnd,
    SessionStart,
    TaskInfo,
    TaskProgress,
    Transcript,
    TurnEnd,
    TurnStart,
    load_plugin,
)
from pydantic_clai2.ui.rendering.status import StatusSegment


@dataclass(kw_only=True)
class Ping(CapabilityEvent, namespace='clai_test'):
    value: str


class Options(BaseModel):
    greeting: str
    loud: bool = False


def host(**settings: JsonValue) -> PluginHost[None]:
    return PluginHost(name='test', console=Console(file=io.StringIO()), settings=dict(settings))


class Greeter(Plugin[Options]):
    pass


class Bare(Plugin):
    pass


def test_settings_model_comes_from_the_plugin_parameter() -> None:
    assert Greeter.settings_type is Options
    assert Greeter.from_host(host(greeting='hi')).settings == Options(greeting='hi')
    with pytest.raises(ValidationError):
        Greeter.from_host(host())
    assert Bare.settings_type is NoSettings
    with pytest.raises(ValidationError):
        Bare.from_host(host(unexpected=True))


def test_settings_model_is_inherited_by_subclasses() -> None:
    class LoudGreeter(Greeter):
        pass

    assert LoudGreeter.settings_type is Options


class ServiceSettings(BaseModel):
    endpoint: str


class RequestDeps(BaseModel):
    user_id: int


ServiceDepsT = TypeVar('ServiceDepsT')


class ServicePlugin(Plugin[ServiceSettings, ServiceDepsT]):
    pass


class RequestServicePlugin(ServicePlugin[RequestDeps]):
    pass


def test_specializing_only_the_deps_keeps_the_settings_model() -> None:
    assert RequestServicePlugin.settings_type is ServiceSettings
    loaded = load_plugin(
        RequestServicePlugin,
        PluginHost[RequestDeps](
            name='test', console=Console(file=io.StringIO()), settings={'endpoint': 'https://example.com'}
        ),
    )
    assert loaded.plugin.settings == ServiceSettings(endpoint='https://example.com')


ValueT = TypeVar('ValueT')
ChosenT = TypeVar('ChosenT', bound=BaseModel)


class Value(BaseModel, Generic[ValueT]):
    value: ValueT


class ValuePlugin(Plugin[Value[ValueT]], Generic[ValueT]):
    pass


class NumberPlugin(ValuePlugin[int]):
    pass


class ChosenPlugin(Plugin[ChosenT], Generic[ChosenT]):
    pass


class ChosenNumberPlugin(ChosenPlugin[Value[int]]):
    pass


@pytest.mark.parametrize('plugin', [NumberPlugin, ChosenNumberPlugin])
def test_a_generic_settings_model_is_specialized_by_the_plugin(plugin: type[Plugin[Value[int]]]) -> None:
    assert plugin.settings_type is Value[int]
    assert plugin.from_host(host(value='123')).settings == Value[int](value=123)
    with pytest.raises(ValidationError):
        load_plugin(plugin, host(value='not-an-int'))


def test_settings_validate_against_plugin_model() -> None:
    assert host(greeting='hi').settings(Options) == Options(greeting='hi')
    with pytest.raises(ValidationError):
        host().settings(Options)


def test_a_plugin_that_overrides_nothing_contributes_nothing() -> None:
    loaded = load_plugin(Bare, host())
    assert loaded.capabilities == ()
    assert loaded.summary() == 'nothing yet'
    assert not loaded.plugin.has_configure
    assert not loaded.plugin.has_render
    assert loaded.plugin.render(FunctionToolCallEvent(part=ToolCallPart(tool_name='shout', args={}))) is None


async def test_a_plugin_without_a_settings_menu_says_so() -> None:
    with pytest.raises(NotImplementedError, match='Bare has no settings menu'):
        await Bare.from_host(host()).configure()


async def test_host_hooks_dispatch_by_event_type() -> None:
    seen: list[str] = []

    class Moments(Plugin):
        async def on_session_start(self, event: SessionStart) -> None:
            seen.append(f'start:{event.settings.model}')

        async def on_session_end(self, event: SessionEnd) -> None:
            seen.append(f'end:{event.reason}')

        async def on_turn_start(self, event: TurnStart) -> None:
            seen.append(f'turn:{event.text}')

        async def on_turn_end(self, event: TurnEnd) -> None:
            seen.append(f'done:{event.outcome}')

    loaded = load_plugin(Moments, host())
    store = SettingsStore()
    for event in (
        SessionStart(agent=Agent(TestModel()), settings=store.load()),
        TurnStart(text='hi'),
        TurnEnd(text='hi', outcome='completed'),
        SessionEnd(reason='eof'),
    ):
        await loaded.dispatch(event)
    assert seen == ['start:openai-codex:gpt-6-astra', 'turn:hi', 'done:completed', 'end:eof']
    assert loaded.capabilities == ()
    assert loaded.summary() == '4 hooks'


def progress_events() -> tuple[TaskProgress, InstallProgress]:
    now = datetime.now(UTC)
    task = TaskInfo(
        id='t1',
        parent_id=None,
        agent='Explore',
        model=None,
        prompt='look',
        status='running',
        outcome=None,
        started_at=now,
        finished_at=None,
        background=False,
        resumable=False,
        activity='starting',
        usage=RunUsage(),
    )
    return TaskProgress(task=task), InstallProgress(stage='installing', target='1.2.3')


async def test_task_and_install_progress_dispatch_to_their_on_hooks() -> None:
    seen: list[str] = []

    class Progress(Plugin):
        async def on_task_progress(self, event: TaskProgress) -> None:
            seen.append(f'task:{event.task.id}')

        async def on_install_progress(self, event: InstallProgress) -> None:
            seen.append(f'install:{event.stage}:{event.target}')

    loaded = load_plugin(Progress, host())
    assert loaded.handles(TaskProgress)
    assert loaded.handles(InstallProgress)
    for event in progress_events():
        await loaded.dispatch(event)
    assert seen == ['task:t1', 'install:installing:1.2.3']
    assert loaded.summary() == '2 hooks'


async def test_old_progress_hook_names_still_dispatch_with_a_deprecation_warning() -> None:
    seen: list[str] = []

    with pytest.warns(PydanticAIDeprecationWarning) as warned:

        class Legacy(Plugin):
            async def on_task_event(self, event: TaskProgress) -> None:
                with pytest.warns(PydanticAIDeprecationWarning, match='renamed to `on_task_progress`'):
                    await super().on_task_event(event)  # pyright: ignore[reportDeprecated]
                seen.append(f'old task:{event.task.id}')

            async def on_update_progress(self, event: InstallProgress) -> None:
                seen.append(f'old install:{event.stage}')

    assert [str(warning.message) for warning in warned] == [
        '`Legacy.on_task_event` is deprecated; rename it to `on_task_progress`. '
        'The old name will be removed in a future release.',
        '`Legacy.on_update_progress` is deprecated; rename it to `on_install_progress`. '
        'The old name will be removed in a future release.',
    ]
    assert {warning.filename for warning in warned} == {__file__}
    loaded = load_plugin(Legacy, host())
    assert loaded.handles(TaskProgress)
    assert loaded.handles(InstallProgress)
    for event in progress_events():
        await loaded.dispatch(event)
    assert seen == ['old task:t1', 'old install:installing']
    assert loaded.summary() == '2 hooks'

    seen.clear()
    with pytest.warns(PydanticAIDeprecationWarning, match='`Both.on_task_event` is deprecated'):

        class Both(Plugin):
            async def on_task_event(self, event: TaskProgress) -> None:
                seen.append('old')  # pragma: no cover

            async def on_task_progress(self, event: TaskProgress) -> None:
                seen.append('new')

    await load_plugin(Both, host()).dispatch(progress_events()[0])
    assert seen == ['new']


async def test_an_old_progress_hook_from_a_mixin_still_dispatches() -> None:
    seen: list[str] = []

    class LegacyMixin:
        async def on_update_progress(self, event: InstallProgress) -> None:
            seen.append(f'mixin:{event.stage}')

    with pytest.warns(PydanticAIDeprecationWarning, match='`Mixed.on_update_progress` is deprecated'):

        class Mixed(LegacyMixin, Plugin):
            pass

    class Child(Mixed):
        pass

    for plugin in (Mixed, Child):
        loaded = load_plugin(plugin, host())
        assert loaded.handles(InstallProgress)
        await loaded.dispatch(progress_events()[1])
    assert seen == ['mixin:installing', 'mixin:installing']


def test_old_progress_event_names_import_with_a_deprecation_warning() -> None:
    with pytest.warns(
        PydanticAIDeprecationWarning, match='`pydantic_clai2.plugins.TaskEvent` has been renamed to `TaskProgress`'
    ):
        assert plugins_module.TaskEvent is TaskProgress
    with pytest.warns(
        PydanticAIDeprecationWarning,
        match='`pydantic_clai2.plugins.UpdateProgress` has been renamed to `InstallProgress`',
    ):
        assert plugins_module.UpdateProgress is InstallProgress
    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        plugins_module.Missing


@dataclass
class PingListener(AbstractCapability[None]):
    calls: list[str]

    @on_event(Ping)
    async def _ping(self, ctx: RunContext[None], event: Ping) -> None:
        self.calls.append(f'ping:{event.value}')


class Everything(Plugin):
    def __init__(self, host: PluginHost[None], settings: NoSettings) -> None:
        super().__init__(host, settings)
        self.calls: list[str] = []

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        tools = Capability[None](instructions='Be brief.')

        @tools.tool_plain
        def shout(text: str) -> str:
            return text.upper()

        hooks = Hooks[None]()

        @hooks.on.before_tool_execute
        async def guard(
            ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
        ) -> ValidatedToolArgs:
            self.calls.append(f'before:{call.tool_name}')
            await ctx.emit(Ping(value='pong'))
            return args

        return (tools, hooks, PingListener(self.calls))

    def get_commands(self) -> Sequence[Command]:
        return (Command(name='noop', description='Nothing', handler=lambda _: ''),)

    def get_status_segments(self) -> Sequence[StatusSegment]:
        return (lambda: 'here',)

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        if isinstance(event, FunctionToolCallEvent):
            return f'drew {event.part.tool_name}'
        return None


async def test_contributions_are_collected_once_and_reach_the_run() -> None:
    loaded = load_plugin(Everything, host())
    plugin = loaded.plugin
    assert isinstance(plugin, Everything)
    assert loaded.summary() == '1 command, 3 capabilities, 1 renderer, 1 status segment'
    assert loaded.status_segments[0]() == 'here'
    assert [command.name for command in loaded.commands] == ['noop']
    session = Session(Agent(TestModel(call_tools=['shout'])), deps=None, plugins=list(loaded.capabilities))
    await session.prompt('hello')
    assert plugin.calls == ['before:shout', 'ping:pong']
    assert plugin.render(FunctionToolCallEvent(part=ToolCallPart(tool_name='shout', args={}))) == 'drew shout'
    assert plugin.render(Ping(value='x')) is None


def test_duplicate_command_names_fail_the_collection() -> None:
    class Twice(Plugin):
        def get_commands(self) -> Sequence[Command]:
            command = Command(name='same', description='Nothing', handler=lambda _: '')
            return (command, command)

    with pytest.raises(ValueError, match='duplicate command: same'):
        load_plugin(Twice, host())


async def test_plugin_renderers_run_before_the_default_display() -> None:
    class Draws(Plugin):
        def render(self, event: AgentStreamEvent) -> RenderableType | None:
            if isinstance(event, FunctionToolCallEvent) and event.part.tool_name == 'mine':
                return f'custom {event.part.tool_name}'
            return None

    output = io.StringIO()
    plugin = Draws.from_host(host())
    stream = StreamRenderer(Console(file=output), stop_loading=lambda: None, renderers=[plugin.render])
    await stream.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(tool_name='mine', args={})))
    await stream.on_stream_event(FunctionToolCallEvent(part=ToolCallPart(tool_name='theirs', args={})))
    await stream.finish()
    assert 'custom mine' in output.getvalue()
    assert 'custom theirs' not in output.getvalue()
    assert '● theirs' in output.getvalue()


def test_transcript_hands_out_snapshots_like_session() -> None:
    first = ModelRequest.user_text_prompt('one')
    transcript = Transcript(messages=[first], model='test')
    transcript.messages.append(ModelRequest.user_text_prompt('sneaky'))
    assert transcript.messages == [first]
    second = ModelRequest.user_text_prompt('two')
    transcript.replace_messages([second])
    assert transcript.messages == [second]
    assert host().conversation.messages == []


class AliasedSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')
    api_key_name: str = Field(alias='api-key-name')


def test_saved_settings_round_trip_through_aliases() -> None:
    saved: list[dict[str, JsonValue]] = []
    plugin = PluginHost[None](
        name='test', console=Console(file=io.StringIO()), settings={'api-key-name': 'OLD'}, save_settings=saved.append
    )
    plugin.save_settings(AliasedSettings.model_validate({'api-key-name': 'NEW'}))
    assert saved == [{'api-key-name': 'NEW'}]
    assert plugin.settings(AliasedSettings).api_key_name == 'NEW'


def test_settings_that_fail_to_persist_are_not_kept() -> None:
    def refuse(settings: dict[str, JsonValue]) -> None:
        raise OSError('read-only database')

    plugin = PluginHost[None](
        name='test', console=Console(file=io.StringIO()), settings={'api-key-name': 'OLD'}, save_settings=refuse
    )
    with pytest.raises(OSError, match='read-only database'):
        plugin.save_settings(AliasedSettings.model_validate({'api-key-name': 'NEW'}))
    assert plugin.settings(AliasedSettings).api_key_name == 'OLD'


def test_host_outside_the_loader_keeps_saved_settings_for_this_load() -> None:
    plugin = host(**{'api-key-name': 'OLD'})
    plugin.save_settings(AliasedSettings.model_validate({'api-key-name': 'NEW'}))
    assert plugin.settings(AliasedSettings).api_key_name == 'NEW'


class SerializationOnlyAlias(BaseModel):
    api_key_name: str = Field(serialization_alias='api-key-name')


@pytest.mark.parametrize('settings', [RootModel[list[str]](['a']), SerializationOnlyAlias(api_key_name='NEW')])
def test_settings_that_do_not_read_back_are_not_saved(settings: BaseModel) -> None:
    saved: list[dict[str, JsonValue]] = []
    plugin = PluginHost[None](name='test', console=Console(file=io.StringIO()), settings={}, save_settings=saved.append)
    with pytest.raises(ValueError, match='cannot be saved as plugin settings'):
        plugin.save_settings(settings)
    assert saved == []

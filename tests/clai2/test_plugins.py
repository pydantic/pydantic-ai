"""The plugin host: named hooks, typed events, tools, renderers, settings."""

import io
from dataclasses import dataclass
from typing import get_args

import pytest
from pydantic import BaseModel, JsonValue, ValidationError
from pydantic_ai import (
    Agent,
    CapabilityEvent,
    FunctionToolCallEvent,
    ModelRequest,
    RunContext,
    ToolCallPart,
    ToolDefinition,
)
from pydantic_ai.capabilities import Capability, Hooks, ValidatedToolArgs
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import Session, StreamRenderer
from pydantic_clai2.commands import Command
from pydantic_clai2.plugins import CoreHookName, PluginHost, SessionEnd, SessionStart, Transcript, TurnEnd, TurnStart
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@dataclass(kw_only=True)
class Ping(CapabilityEvent, namespace='clai_test'):
    value: str


class Options(BaseModel):
    greeting: str
    loud: bool = False


def host(**settings: JsonValue) -> PluginHost[None]:
    return PluginHost(name='test', console=Console(file=io.StringIO()), settings=dict(settings))


def test_core_hook_names_match_hooks_on() -> None:
    public = {name for name in dir(Hooks[None]().on) if not name.startswith('_')}
    assert set(get_args(CoreHookName)) == public


def test_unknown_hook_name_is_rejected() -> None:
    with pytest.raises(ValueError, match='Unknown hook'):
        host().on('agent_run_start')  # pyright: ignore[reportArgumentType, reportCallIssue]


def test_settings_validate_against_plugin_model() -> None:
    assert host(greeting='hi').settings(Options) == Options(greeting='hi')
    with pytest.raises(ValidationError):
        host().settings(Options)


async def test_host_hooks_dispatch_by_event_type() -> None:
    plugin = host()
    seen: list[str] = []

    @plugin.on('session_start')
    async def started(event: SessionStart) -> None:
        seen.append(f'start:{event.settings.model}')

    @plugin.on('session_end')
    async def ended(event: SessionEnd) -> None:
        seen.append(f'end:{event.reason}')

    @plugin.on('turn_start')
    async def before(event: TurnStart) -> None:
        seen.append(f'turn:{event.text}')

    @plugin.on('turn_end')
    async def after(event: TurnEnd) -> None:
        seen.append(f'done:{event.outcome}')

    store = SettingsStore()
    for event in (
        SessionStart(agent=Agent(TestModel()), settings=store.load()),
        TurnStart(text='hi'),
        TurnEnd(text='hi', outcome='completed'),
        SessionEnd(reason='eof'),
    ):
        for handler in plugin.handlers:
            await handler(event)
    assert seen == ['start:openai-codex:gpt-6-astra', 'turn:hi', 'done:completed', 'end:eof']
    assert plugin.capabilities == []
    assert plugin.summary() == '0 commands, 4 hooks, 0 capabilities, 0 renderers'


async def test_core_hooks_events_tools_and_renderers_reach_the_run() -> None:
    plugin = host()
    calls: list[str] = []
    tools = Capability[None](instructions='Be brief.')

    @tools.tool_plain
    def shout(text: str) -> str:
        return text.upper()

    plugin.add(tools)
    plugin.commands.register(Command(name='noop', description='Nothing', handler=lambda _: ''))

    @plugin.on('before_tool_execute')
    async def guard(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        calls.append(f'before:{call.tool_name}')
        await ctx.emit(Ping(value='pong'))
        return args

    @plugin.on(Ping)
    async def on_ping(ctx: RunContext[None], event: Ping) -> None:
        calls.append(f'ping:{event.value}')

    @plugin.render(FunctionToolCallEvent)
    def draw(event: FunctionToolCallEvent) -> str:
        return f'drew {event.part.tool_name}'

    assert plugin.summary() == '1 commands, 0 hooks, 1 capabilities, 1 renderers'
    assert len(plugin.capabilities) == 2
    session = Session(Agent(TestModel(call_tools=['shout'])), deps=None, plugins=plugin.capabilities)
    await session.prompt('hello')
    assert calls == ['before:shout', 'ping:pong']
    renderer = plugin.renderers[0]
    assert renderer(FunctionToolCallEvent(part=ToolCallPart(tool_name='shout', args={}))) == 'drew shout'
    assert renderer(Ping(value='x')) is None


async def test_plugin_renderers_run_before_the_default_display() -> None:
    plugin = host()

    @plugin.render(FunctionToolCallEvent)
    def draw(event: FunctionToolCallEvent) -> str | None:
        return f'custom {event.part.tool_name}' if event.part.tool_name == 'mine' else None

    output = io.StringIO()
    stream = StreamRenderer(Console(file=output), stop_loading=lambda: None, renderers=plugin.renderers)
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

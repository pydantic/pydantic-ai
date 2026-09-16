"""Public behavior of sessions, settings, plugins, completion, and rendering."""

import io
from dataclasses import dataclass
from pathlib import Path

import anyio
import pytest
from pydantic import ValidationError
from pydantic_ai import (
    Agent,
    CapabilityEvent,
    ModelRequestContext,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    models,
)
from pydantic_ai.capabilities import AbstractCapability, on_event
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models.test import TestModel
from rich.console import Console
from termflow.tui.completion import CompleteEvent, Document  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import Session, StreamRenderer
from pydantic_clai2.commands import Command, Commands, config_command, config_completions, plugins_command
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.splash import Splash


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(autouse=True)
def no_model_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, 'ALLOW_MODEL_REQUESTS', False)


@dataclass(kw_only=True)
class Ready(CapabilityEvent, namespace='clai_test'):
    value: str


class Publisher(AbstractCapability[None]):
    async def before_model_request(
        self, ctx: RunContext[None], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        await ctx.emit(Ready(value='ready'))
        return request_context


class Recorder(AbstractCapability[None]):
    def __init__(self) -> None:
        self.events: list[str] = []

    @on_event(Ready)
    async def record(self, ctx: RunContext[None], event: Ready) -> None:
        self.events.append(event.value)


async def test_history_tools_and_plugins() -> None:
    recorder = Recorder()
    calls: list[str] = []
    agent = Agent(TestModel(custom_output_text='done'), deps_type=type(None), capabilities=[Publisher()])

    @agent.tool_plain
    def inspect_workspace() -> str:
        calls.append('tool')
        return 'files'

    session = Session(agent, deps=None, plugins=[recorder])
    assert (await session.prompt('first')).output == 'done'
    assert (await session.prompt('second')).output == 'done'
    prompts = [
        part.content
        for message in session.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert prompts == ['first', 'second']
    assert calls
    assert recorder.events
    snapshot = session.messages
    snapshot.clear()
    assert session.messages
    session.clear()
    assert not session.messages


async def test_cancel_preserves_history_and_rejects_concurrency() -> None:
    started = anyio.Event()
    release = anyio.Event()
    agent = Agent(TestModel(custom_output_text='done'))

    @agent.tool_plain
    async def wait() -> str:
        started.set()
        await release.wait()
        return 'ok'

    session = Session(agent, deps=None, message_history=[ModelRequest(parts=[UserPromptPart('first')])])
    prior = session.messages
    scope = anyio.CancelScope()

    async def interrupted_turn() -> None:
        with scope:
            await session.prompt('make a personality plugin')

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(interrupted_turn)
        await started.wait()
        with pytest.raises(RuntimeError):
            await session.prompt('overlap')
        with pytest.raises(RuntimeError):
            session.clear()
        scope.cancel()
    assert scope.cancelled_caught
    assert session.messages[: len(prior)] == prior
    assert len(session.messages) > len(prior)
    release.set()
    result = await session.prompt('actually, combine playful and pedantic')
    prompts = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    assert prompts == ['first', 'make a personality plugin', 'actually, combine playful and pedantic']
    session.clear()
    assert session.messages == []


async def test_renderer_flushes_and_hides_thinking() -> None:
    output = io.StringIO()
    renderer = StreamRenderer(Console(file=output), stop_loading=lambda: None)
    await renderer.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='Consider this')))
    await renderer.on_stream_event(PartEndEvent(index=0, part=ThinkingPart(content='Consider this')))
    await renderer.on_stream_event(PartStartEvent(index=0, part=TextPart(content='# Answer\n')))
    await renderer.on_stream_event(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='last line')))
    await renderer.finish()
    assert 'last line' in output.getvalue()
    assert 'Consider this' in output.getvalue()
    hidden = StreamRenderer(Console(file=output), stop_loading=lambda: None, show_thinking=False)
    await hidden.on_stream_event(PartStartEvent(index=0, part=ThinkingPart(content='secret')))
    await hidden.finish()
    assert 'secret' not in output.getvalue()


def test_settings_round_trip_and_validation(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    assert store.load().request_limit == 10000
    store.set('display.thinking', False)
    assert not SettingsStore(store.path).load().thinking
    with pytest.raises(ValidationError):
        store.set('run.request_limit', -1)
    with pytest.raises(ValueError):
        store.set('typo', True)
    store.reset('display.thinking')
    assert store.load().thinking
    assert config_command(store, ['get', 'display.thinking']) == 'true'
    config_command(store, ['set', 'model', 'test'])
    assert store.load().model == 'test'
    plugins_command(store, ['add', 'audit', 'missing.module:Plugin'])
    plugins_command(store, ['disable', 'audit'])
    assert [plugin.enabled for plugin in store.plugins()] == [False]
    plugins_command(store, ['remove', 'audit'])
    assert store.plugins() == []


def test_completion_uses_registry(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    commands = Commands()
    commands.register(
        Command(
            name='config',
            description='Settings',
            handler=lambda args: config_command(store, args),
            complete=config_completions,
        )
    )
    assert [c.text for c in commands.get_completions(Document('/co'), CompleteEvent())] == ['config']
    assert 'display.thinking' in [
        c.text for c in commands.get_completions(Document('/config set display.'), CompleteEvent())
    ]
    result = commands.execute('/config set display.thinking false')
    assert isinstance(result, str) and result.startswith('Saved')
    assert not store.load().thinking
    with pytest.raises(ValueError, match='Unknown command'):
        commands.execute('/oops')
    with pytest.raises(ValueError, match='duplicate'):
        commands.register(Command(name='config', description='Duplicate', handler=lambda _: ''))


def test_splash_noop_and_frames() -> None:
    splash = Splash(enabled=False)
    splash.start()
    assert splash.frame(0) != splash.frame(10)
    splash.stop()
    splash.stop()

"""Terminal and CLI integration without external model requests."""

import io
import os
import subprocess
import sys
import threading
from collections.abc import AsyncIterable
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent, AgentStreamEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from rich.console import Console
from termflow.tui.completion import CompleteEvent, Document  # pyright: ignore[reportMissingTypeStubs]

from pydantic_clai2 import Session, chat
from pydantic_clai2.command_context import CommandContext, CommandProvider
from pydantic_clai2.commands import Command, Commands, set_completions
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.plugins import load_plugins
from pydantic_clai2.settings_store import SettingsStore
from pydantic_clai2.splash import Splash


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_existing_handler_and_structured_output() -> None:
    existing: list[AgentStreamEvent] = []
    observed: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in events:
            existing.append(event)

    async def observe(event: AgentStreamEvent) -> None:
        observed.append(event)

    class ObservedAgent(Agent[None, list[int]]):
        @property
        def event_stream_handler(self):
            return handler

    agent = ObservedAgent(TestModel(), output_type=list[int], deps_type=type(None))
    session = Session(agent, deps=None, on_stream_event=observe)
    result = await session.prompt('numbers')
    assert isinstance(result.output, list)
    assert existing == observed
    assert existing


async def test_set_without_initial_model(tmp_path: Path) -> None:
    output = io.StringIO()
    store = SettingsStore(tmp_path / 'config.db')
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(
            'hello\n/set model test\n/set display.thinking false\n/set run.request_limit 123\nhello\n/exit\n'
        )
        await chat(Agent(), deps=None, console=Console(file=output), store=store)
    assert 'Choose a model first' in output.getvalue()
    assert 'Applied.' in output.getvalue()
    assert store.load().model == 'test'
    assert store.load().request_limit == 123
    assert not store.load().thinking
    assert 'success' in output.getvalue()


async def test_plugin_commands(tmp_path: Path) -> None:
    class GreetingPlugin(AbstractCapability[None], CommandProvider):
        def get_commands(self, context: CommandContext) -> list[Command]:
            return [
                Command(
                    name='greet',
                    description='Plugin greeting',
                    handler=lambda args: f'Hello {args[0]}',
                    complete=lambda _: ('Mike',),
                )
            ]

    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/help\n/greet Mike\n/exit\n')
        await chat(
            Agent(TestModel()),
            deps=None,
            plugins=[GreetingPlugin()],
            console=Console(file=output),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    assert '/greet: Plugin greeting' in output.getvalue()
    assert 'Hello Mike' in output.getvalue()


def test_set_validation_preserves_active_and_saved_settings(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    context = CommandContext(
        settings=store.load(), store=store, clear_history=lambda: None, apply_setting=lambda key, settings: None
    )
    with pytest.raises(ValueError):
        context.set_setting(['run.request_limit', '-1'])
    assert context.settings.request_limit == store.load().request_limit == 10000
    with pytest.raises(ValueError, match='Usage'):
        context.set_setting(['typo', 'false'])
    assert store.overrides() == {}


def test_registry_registration_is_atomic() -> None:
    commands = Commands()
    commands.register(Command(name='help', description='Help', handler=lambda _: ''))
    with pytest.raises(ValueError, match='duplicate'):
        commands.register_many(
            [
                Command(name='greet', description='Greeting', handler=lambda _: ''),
                Command(name='help', description='Collision', handler=lambda _: ''),
            ]
        )
    assert 'greet' not in commands.help([])
    assert 'help' not in Commands().help([])


def test_set_autocomplete() -> None:
    commands = Commands()
    commands.register(Command(name='set', description='Settings', handler=lambda _: '', complete=set_completions))
    assert 'model' in [c.text for c in commands.get_completions(Document('/set mo'), CompleteEvent())]
    assert 'false' in [c.text for c in commands.get_completions(Document('/set display.thinking f'), CompleteEvent())]
    models = list(commands.get_completions(Document('/set model anthropic:'), CompleteEvent()))
    assert models
    codex = list(commands.get_completions(Document('/set model openai-codex'), CompleteEvent()))
    assert [item.text for item in codex] == ['openai-codex:', 'openai-codex:gpt-6-astra']
    assert codex[0].start_position == -len('openai-codex')
    assert all(c.text.startswith('anthropic:') for c in models)


async def test_prompt_loop_commands(tmp_path: Path) -> None:
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/help\nhello\n/new\n/exit\n')
        await chat(
            Agent(TestModel(custom_output_text='hello back')),
            deps=None,
            console=Console(file=output),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    assert '/config' in output.getvalue()
    assert 'hello back' in output.getvalue()
    assert '\n\nConversation cleared.' in output.getvalue()


def test_cli_settings(tmp_path: Path) -> None:
    base = [sys.executable, '-m', 'pydantic_clai2', '--database', str(tmp_path / 'config.db')]
    result = subprocess.run(
        [*base, 'config', 'set', 'display.thinking', 'false'], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    result = subprocess.run([*base, 'config', 'get', 'display.thinking'], check=False, capture_output=True, text=True)
    assert result.stdout.strip() == 'false'
    result = subprocess.run(
        [*base, 'config', 'set', 'run.request_limit', '-1'], check=False, capture_output=True, text=True
    )
    assert result.returncode != 0


def test_import_is_light() -> None:
    result = subprocess.run(
        [sys.executable, '-c', 'import pydantic_clai2, sys; assert "pydantic_ai" not in sys.modules'],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_file_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / 'example.py').touch()
    monkeypatch.chdir(tmp_path)
    completions = list(Commands().get_completions(Document('read @exam'), CompleteEvent()))
    assert any('ple.py' in completion.text for completion in completions)


def test_invalid_plugin() -> None:
    with pytest.raises(TypeError):
        load_plugins([PluginSettings(id='wrong', factory='pathlib:Path')])


def test_splash_restores_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    frame_written = threading.Event()

    class Terminal(io.StringIO):
        def isatty(self) -> bool:
            return True

        def write(self, text: str) -> int:
            result = super().write(text)
            if '\x1b[?2026l' in text:
                frame_written.set()
            return result

    terminal = Terminal()
    monkeypatch.setattr(sys, 'stdout', terminal)
    monkeypatch.setenv('COLUMNS', '100')
    monkeypatch.setenv('LINES', '40')
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm-256color')
    splash = Splash()
    splash.start()
    try:
        assert frame_written.wait(timeout=5)
        print('captured startup output')
    finally:
        splash.stop()
    assert sys.stdout is terminal
    assert 'captured startup output' in terminal.getvalue()
    assert '\x1b[?1049l\x1b[?25h' in terminal.getvalue()


def test_config_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('XDG_CONFIG_HOME', os.fspath(tmp_path))
    assert SettingsStore().path == tmp_path / 'pydantic-clai2/config.db'

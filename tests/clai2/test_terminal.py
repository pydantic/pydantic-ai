"""Terminal and CLI integration without external model requests."""

import io
import os
import subprocess
import sys
import threading
from collections.abc import AsyncIterable
from pathlib import Path
from types import ModuleType

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text
from termflow.tui.completion import CompleteEvent, Document

from pydantic_ai import Agent, AgentStreamEvent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.models.test import TestModel
from pydantic_clai2 import DEFAULT_PLUGINS, Session, chat
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Command, Commands, set_completions
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import Plugin, TurnEnd, TurnStart
from pydantic_clai2.ui.prompt.prompt_surface import LEAVE, PromptSurface
from pydantic_clai2.ui.rendering.splash import Splash


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
            'hello\r/set model test\r/set display.thinking false\r/set run.request_limit 123\rhello\r/exit\r'
        )
        await chat(Agent(), deps=None, console=Console(file=output), store=store)
    assert 'Choose a model first' in output.getvalue()
    assert 'Applied.' in output.getvalue()
    assert store.load().model == 'test'
    assert store.load().request_limit == 123
    assert not store.load().thinking
    assert 'success' in output.getvalue()


async def test_drop_in_plugin_commands_and_hooks(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'greeter.py').write_text(
        'from pydantic_clai2.commands import Command\n'
        'from pydantic_clai2.plugins import Plugin, SessionEnd, SessionStart, TurnEnd, TurnStart\n'
        'class Greeter(Plugin):\n'
        '    def get_commands(self):\n'
        "        return [Command(name='greet', description='Plugin greeting', "
        "handler=lambda args: f'Hello {args[0]}')]\n"
        '    async def on_session_start(self, event: SessionStart) -> None:\n'
        "        self.host.console.print(f'started with model {event.settings.model}')\n"
        '    async def on_turn_start(self, event: TurnStart) -> None:\n'
        '        event.text = event.text.upper()\n'
        '    async def on_turn_end(self, event: TurnEnd) -> None:\n'
        "        self.host.console.print(f'turn {event.outcome}: {event.text}')\n"
        '    async def on_session_end(self, event: SessionEnd) -> None:\n'
        "        self.host.console.print(f'stopped: {event.reason}')\n"
    )
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/help\r/greet Mike\rhello\r/plugins list\r/exit\r')
        await chat(
            Agent(TestModel(custom_output_text='hi')),
            deps=None,
            console=Console(file=output),
            store=store,
        )
    text = output.getvalue()
    assert 'started with model None' in text
    assert '/greet: Plugin greeting' in text
    assert 'Hello Mike' in text
    assert 'turn completed: HELLO' in text
    assert 'greeter:' in text and 'loaded)' in text
    assert 'stopped: exit' in text


async def test_coder_is_a_builtin_plugin(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    offered: list[set[str]] = []
    hooks = Hooks[None]()

    @hooks.on.before_model_request
    async def record(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        offered.append({tool.name for tool in request_context.model_request_parameters.function_tools})
        return request_context

    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/plugins list\rhello\r/plugins disable coder\rhello\r/plugins remove coder\r/exit\r')
        await chat(
            Agent(TestModel(call_tools=[], custom_output_text='hi'), deps_type=type(None)),
            deps=None,
            plugins=[hooks],
            console=Console(file=output, width=200),
            store=store,
            builtin_plugins=DEFAULT_PLUGINS,
        )
    text = output.getvalue()
    assert 'coder: pydantic_clai2.builtin_plugins.coder (built-in) (enabled, loaded)' in text
    assert 'Disabled coder.' in text
    assert 'coder is built in; restored its defaults.' in text
    assert store.plugins() == []
    assert len(offered) == 2
    assert 'shell' in offered[0] and 'shell' not in offered[1]


async def test_plugin_can_cancel_a_turn(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir()
    (store.plugins_dir / 'gate.py').write_text(
        'from pydantic_clai2.plugins import Plugin, TurnStart\n'
        'class Gate(Plugin):\n'
        '    async def on_turn_start(self, event: TurnStart) -> None:\n'
        "        if event.text == 'stop':\n"
        "            event.cancel('not today')\n"
        "        elif event.text == 'boom':\n"
        "            raise RuntimeError('gate exploded')\n"
    )
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('stop\rboom\r/exit\r')
        await chat(Agent(TestModel(custom_output_text='never')), deps=None, console=Console(file=output), store=store)
    text = output.getvalue()
    assert 'Turn cancelled by a plugin: not today' in text
    assert "Plugin 'gate': RuntimeError: gate exploded" in text
    assert 'never' not in text


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
    assert {item.text for item in codex} >= {
        'openai-codex:',
        'openai-codex:gpt-6.1-sol',
        'openai-codex:gpt-6-astra',
        'openai-codex:gpt-6-sol',
        'openai-codex:gpt-6-luna',
    }
    assert len(codex) == len({item.text for item in codex})
    assert codex[0].start_position == -len('openai-codex')
    assert all('anthropic:' in c.text for c in models)


@pytest.mark.parametrize('height', [24, 45])
async def test_prompt_frame_stays_visible_during_tools(
    tmp_path: Path, height: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    painted = anyio.Event()
    completions = anyio.Event()
    searched = anyio.Event()
    pasted = anyio.Event()
    drafted = anyio.Event()
    queued = anyio.Event()
    second = anyio.Event()
    calls = 0
    frame: list[str] = []
    working = anyio.Event()
    finish = anyio.Event()
    done = anyio.Event()

    class Surface(PromptSurface):
        def paint(self, rows: tuple[str, ...]) -> None:
            nonlocal frame
            super().paint(rows)
            frame = [''] * (height - len(rows)) + [Text.from_ansi(row).plain for row in rows]
            for text, event in (
                ('ready', painted),
                ('display.thinking', completions),
                ('reverse-i-search', searched),
                ('second line', pasted),
                ('next message', drafted),
                ('retained draft', queued),
            ):
                if any(text in line for line in frame):
                    event.set()

    monkeypatch.setattr('pydantic_clai2.ui.prompt.live_prompt.PromptSurface', Surface)
    output = io.StringIO()
    store = SettingsStore(tmp_path / 'config.db')
    terminal = DummyOutput()
    hooks = Hooks[None]()

    @hooks.on.before_model_request
    async def observe(ctx: RunContext[None], request: ModelRequestContext) -> ModelRequestContext:
        if ctx.prompt == 'next message':
            assert finish.is_set()
            second.set()
        return request

    agent = Agent(
        TestModel(call_tools=['work'], custom_output_text='Finished work'), deps_type=type(None), capabilities=[hooks]
    )

    @agent.tool_plain
    async def work() -> str:
        nonlocal calls
        calls += 1
        working.set()
        await finish.wait()
        return 'done'

    async def run() -> None:
        await chat(
            agent,
            deps=None,
            console=Console(file=output, force_terminal=True, width=80, height=height),
            store=store,
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=terminal), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            await painted.wait()
            top = next(row for row, line in enumerate(frame) if line and set(line) == {'─'})
            bottom = max(row for row, line in enumerate(frame) if line and set(line) == {'─'})
            assert bottom - top == 2
            assert bottom == len(frame) - 2
            pipe.send_text('\x12')
            await searched.wait()
            top = next(row for row, line in enumerate(frame) if line and set(line) == {'─'})
            bottom = max(row for row, line in enumerate(frame) if line and set(line) == {'─'})
            assert bottom - top == 2
            pipe.send_text('\x07/set ')
            await completions.wait()
            top = next(row for row, line in enumerate(frame) if line and set(line) == {'─'})
            bottom = max(row for row, line in enumerate(frame) if line and set(line) == {'─'})
            assert bottom - top == 8
            assert any('display.thinking' in row for row in frame[top + 1 : bottom])
            assert all('display.thinking' not in row for row in frame[bottom + 1 :])
            assert bottom == len(frame) - 2
            pipe.send_text('\x15\x1b[200~first line\nsecond line\x1b[201~')
            await pasted.wait()
            top = next(row for row, line in enumerate(frame) if line and set(line) == {'─'})
            bottom = max(row for row, line in enumerate(frame) if line and set(line) == {'─'})
            assert bottom - top == 3
            pipe.send_text('\r')
            await working.wait()
            pipe.send_text('next message')
            await drafted.wait()
            assert any(line.startswith('next message') for line in frame)
            pipe.send_text('\r/set display.thinking false\rretained draft')
            await queued.wait()
            follow_up = next(row for row, line in enumerate(frame) if 'Follow-up: next message' in line)
            command = next(row for row, line in enumerate(frame) if 'Command: /set display.thinking false' in line)
            editor_top = next(row for row, line in enumerate(frame) if 'Working ' in line)
            assert follow_up < command < editor_top
            indicator = next(row for row, line in enumerate(frame) if 'Working ' in line)
            draft = next(row for row, line in enumerate(frame) if 'retained draft' in line)
            assert editor_top == indicator
            assert draft == editor_top + 1
            assert calls == 1
            assert store.load().thinking
            finish.set()
            await second.wait()
            assert any('retained draft' in line for line in frame)
            pipe.send_text('\x15/exit\r')
            await done.wait()
    assert 'Goodbye.' in output.getvalue()
    assert 'Turn not saved' not in output.getvalue()
    assert calls == 1
    assert not store.load().thinking
    printed = Text.from_ansi(output.getvalue().rsplit(LEAVE, 1)[1]).plain
    assert printed.index('Finished work') < printed.index('> next message\n')


async def test_prompt_loop_commands(tmp_path: Path) -> None:
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/help\rhello\r/new\r/exit\r')
        await chat(
            Agent(TestModel(custom_output_text='hello back')),
            deps=None,
            console=Console(file=output),
            store=SettingsStore(tmp_path / 'config.db'),
        )
    assert '/config' in output.getvalue()
    assert 'hello back' in output.getvalue()
    # `/new` is `/clear`: the banner again, with nothing after it.
    assert output.getvalue().count('/new starts a session') == 2


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
    assert [(item.text, item.start_position) for item in completions] == [('example.py', -4)]


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


async def test_model_adds_and_selects_any_model(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/model unknown\r/add_model test\r/model test\rhello\r/exit\r')
        await chat(Agent(), deps=None, console=Console(file=output, width=200), store=store)
    assert 'Model not added' not in output.getvalue()
    assert 'success' in output.getvalue()
    assert store.models() == ['test', 'unknown']
    assert store.load().model == 'test'


@pytest.mark.parametrize('phase', ['start', 'end'])
@pytest.mark.parametrize('key', ['\x03', '\x1b'])
async def test_live_editor_interrupts_slow_turn_hooks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, key: str
) -> None:
    started = anyio.Event()
    cleaned = anyio.Event()
    done = anyio.Event()
    module = ModuleType('slow_turn_test')

    async def wait() -> None:
        try:
            started.set()
            await anyio.sleep_forever()
        finally:
            cleaned.set()

    class Slow(Plugin):
        async def on_turn_start(self, event: TurnStart) -> None:
            if phase == 'start':
                await wait()

        async def on_turn_end(self, event: TurnEnd) -> None:
            if phase == 'end':
                await wait()

    Slow.__module__ = module.__name__
    module.__dict__['Slow'] = Slow
    monkeypatch.setitem(sys.modules, module.__name__, module)
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(PluginSettings(id='slow', factory=module.__name__))
    output = io.StringIO()

    async def run() -> None:
        await chat(
            Agent(TestModel(custom_output_text='completed')),
            deps=None,
            console=Console(file=output, force_terminal=True),
            store=store,
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            pipe.send_text('hello\r')
            await started.wait()
            pipe.send_text('draft' + key)
            await cleaned.wait()
            pipe.send_text('\x15/exit\r')
            await done.wait()
    assert 'Goodbye.' in output.getvalue()
    if phase == 'start':
        assert 'Turn cancelled. Use /exit to quit.' in output.getvalue()
        assert 'Press Ctrl-C again' not in output.getvalue()
        assert 'completed' not in output.getvalue()


@pytest.mark.parametrize('terminal', [False, True])
@pytest.mark.parametrize(
    'text',
    [
        '/Users/test/Desktop/Screenshot 2026-09-19.png',
        r'/Users/test/Desktop/Screen\ Shot.png explain this',
        "/tmp/screenshot.png What's wrong here?",
        '/screenshot.PNG',
        r'/Screen\ Shot.png',
        '/help/screenshot.png',
        '"/Users/test/Screen Shot.png"',
        "'/Users/test/Screen Shot.png'",
        '/tmp/shot.png\nDescribe this screenshot.',
    ],
)
async def test_absolute_screenshot_paths_are_prompts(tmp_path: Path, terminal: bool, text: str) -> None:
    prompts: list[str] = []
    hooks = Hooks[None]()

    @hooks.on.before_model_request
    async def record(ctx: RunContext[None], request: ModelRequestContext) -> ModelRequestContext:
        assert isinstance(ctx.prompt, str)
        prompts.append(ctx.prompt)
        return request

    output = io.StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(f'\x1b[200~{text}\x1b[201~\r/missing-command\r/exit\r')
        await chat(
            Agent(TestModel(custom_output_text='received screenshot path'), deps_type=type(None), capabilities=[hooks]),
            deps=None,
            console=Console(file=output, force_terminal=terminal),
            store=SettingsStore(tmp_path / 'settings.db'),
        )
    assert prompts == [text]
    assert 'Unknown command' in output.getvalue()
    assert 'Goodbye.' in output.getvalue()

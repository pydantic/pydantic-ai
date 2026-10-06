"""Menus opened while a turn streams: the run's output waits behind the menu, in order."""

import asyncio
import io
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import anyio
import pytest
from anyio.to_thread import run_sync as in_worker
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from pydantic_ai import Agent, ModelRequestContext, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_clai2 import Session, chat
from pydantic_clai2._app import create_shell
from pydantic_clai2.builtin_plugins.google_workspace import GoogleWorkspacePlugin
from pydantic_clai2.builtin_plugins.grain import GrainPlugin
from pydantic_clai2.builtin_plugins.pylon import PylonPlugin
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, load_plugin
from pydantic_clai2.plugins.loader import TURN_NOTICE
from pydantic_clai2.runtime.session_settings import SessionSettings
from pydantic_clai2.ui.menus.field_menu import FieldMenu
from pydantic_clai2.ui.menus.menu_worker import holding_output, run_worker
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource
from pydantic_clai2.ui.prompt.prompt_surface import LEAVE, PromptSurface
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.prompt.screen import Screen
from tests.clai2.test_tasks import task


def test_only_bare_opted_in_commands_run_during_a_turn() -> None:
    commands = Commands()
    commands.register(Command(name='menu', description='', handler=lambda _: '', during_turn=True))
    commands.register(Command(name='plain', description='', handler=lambda _: ''))
    commands.register(
        Command(name='sub', description='', handler=lambda _: '', during_turn_subcommands=('add',)),
    )
    commands.register(
        Command(name='apply', description='', handler=lambda _: '', during_turn=True, args_during_turn=True)
    )
    commands.register(Command(name='args', description='', handler=lambda _: '', args_during_turn=True))
    assert commands.runs_during_turn('/menu')
    assert not commands.runs_during_turn('/menu value')
    assert not commands.runs_during_turn('/sub')
    assert commands.runs_during_turn('/sub add')
    assert not commands.runs_during_turn('/sub other')
    assert not commands.runs_during_turn('/sub add value')
    assert not commands.runs_during_turn('/unknown add')
    assert commands.runs_during_turn('/apply')
    assert commands.runs_during_turn('/apply some value')
    assert not commands.runs_during_turn('/args')
    assert commands.runs_during_turn('/args value')
    assert not commands.runs_during_turn('/plain')
    assert not commands.runs_during_turn('/unknown')
    assert not commands.runs_during_turn('/tmp/menu')
    assert not commands.runs_during_turn('menu')


async def test_key_login_resume_and_plugin_settings_menus_open_mid_turn(tmp_path: Path) -> None:
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model='test'),
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    assert all(shell.commands.runs_during_turn(text) for text in ('/keys', '/login', '/resume'))
    assert not any(shell.commands.runs_during_turn(text) for text in ('/login openai-codex', '/resume ID'))

    def host(name: str) -> PluginHost[None]:
        return PluginHost(name=name, console=Console(file=io.StringIO()), settings={})

    for name, loaded in (
        ('google_workspace', load_plugin(GoogleWorkspacePlugin, host('google_workspace'))),
        ('grain', load_plugin(GrainPlugin, host('grain'))),
        ('pylon', load_plugin(PylonPlugin, host('pylon'))),
    ):
        assert loaded.commands.runs_during_turn(f'/{name}')
        assert not loaded.commands.runs_during_turn(f'/{name} status')


async def test_run_worker_holds_output_only_while_the_widget_runs() -> None:
    log: list[str] = []

    @contextmanager
    def hold(*, leave_screen: bool) -> Generator[None]:
        log.append(f'hold leave_screen={leave_screen}')
        yield
        log.append('replay')

    with holding_output(hold):
        assert await run_worker(lambda: log.append('menu') or 'done') == 'done'
        await run_worker(lambda: log.append('inline'), inline=True)
    await run_worker(lambda: log.append('unheld'))
    assert log == [
        'hold leave_screen=True',
        'menu',
        'replay',
        'hold leave_screen=False',
        'inline',
        'replay',
        'unheld',
    ]


async def test_overlay_takes_turns_with_widgets_without_pausing_the_stream() -> None:
    log: list[str] = []
    screen = Screen()
    asked, entered = anyio.Event(), anyio.Event()

    @asynccontextmanager
    async def take() -> AsyncGenerator[None]:
        log.append('paused stream')
        yield

    async def question() -> None:
        # The agent's run started before the menu, so its tool calls are not the menu's own code.
        await asked.wait()
        async with screen.full():
            entered.set()

    with screen.bound(take):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(question)
            async with screen.overlay():
                asked.set()
                await anyio.wait_all_tasks_blocked()
                assert not entered.is_set()
                assert log == []
            await entered.wait()
    assert log == ['paused stream']


def test_session_changes_saved_during_a_turn_wait_for_it_to_end() -> None:
    session = Session(Agent(TestModel()), deps=None)
    applied = SessionSettings(session=session, console=Console(file=io.StringIO()), settings=Settings())
    applied('model', Settings(model='test:first'))
    assert session.model == 'test:first'
    with applied.turn():
        applied('model', Settings(model='test:second'))
        applied('run.tool_retries', Settings(tool_retries=7))
        applied('model', Settings(model='test:third'))
        applied('display.theme', Settings(theme='default'))
        assert (session.model, session.tool_retries) == ('test:first', None)
    assert (session.model, session.tool_retries) == ('test:third', 7)
    applied('run.request_limit', Settings(request_limit=12))
    assert session.usage_limits is not None and session.usage_limits.request_limit == 12


async def test_model_settings_saved_mid_turn_reach_the_running_models_next_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An edit saved from a menu worker thread while a tool runs applies to the same turn's next request."""
    working, finish, streamed, done = anyio.Event(), anyio.Event(), anyio.Event(), anyio.Event()

    class Surface(PromptSurface):
        def changed(self) -> None:
            super().changed()
            if 'Finished work' in _plain(self.transcript):
                streamed.set()

    monkeypatch.setattr('pydantic_clai2.ui.prompt.live_prompt.PromptSurface', Surface)
    store = SettingsStore(tmp_path / 'config.db')
    store.save_model_settings('test', {'seed': 1})
    seen: list[ModelSettings | None] = []
    hooks = Hooks[None]()

    @hooks.on.before_model_request
    async def capture(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        seen.append(request_context.model_settings)
        return request_context

    def edit() -> str:
        source = ModelSettingsSource(store, 'test')
        row = FieldMenu(source).row_for('max_tokens')
        assert row is not None
        return source.apply(row, '42')

    async def model_settings(args: list[str]) -> str:
        saved = await run_worker(edit)
        finish.set()
        return saved

    agent = Agent(TestModel(call_tools=['work'], custom_output_text='Finished work'), deps_type=type(None))

    @agent.tool_plain
    async def work() -> str:
        working.set()
        await finish.wait()
        return 'done'

    output = io.StringIO()
    menu = Command(name='edit_settings', description='Edit', handler=model_settings, during_turn=True)

    async def run() -> None:
        await chat(
            agent,
            deps=None,
            plugins=[hooks, _MenuCommand(menu)],
            console=Console(file=output, force_terminal=True, width=80, height=24),
            store=store,
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            pipe.send_text('start\r')
            await working.wait()
            pipe.send_text('/edit_settings\r')
            await streamed.wait()
            pipe.send_text('/exit\r')
            await done.wait()
    assert seen == [{'seed': 1}, {'seed': 1, 'max_tokens': 42}]
    assert 'Saved max_tokens for test.' in output.getvalue()


class _MenuCommand(AbstractCapability[None]):
    def __init__(self, menu: Command) -> None:
        self.menu = menu

    def get_commands(self, context: CommandContext) -> Sequence[Command]:
        return (self.menu,)


async def test_menu_opened_mid_turn_holds_the_run_output_until_it_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _open_menu_mid_turn(tmp_path, monkeypatch)


def test_menu_opens_mid_turn_on_a_plain_asyncio_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI uses `asyncio.run`, so key callbacks run with no task and no anyio backend marker."""
    asyncio.run(_open_menu_mid_turn(tmp_path, monkeypatch))


async def _open_menu_mid_turn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    working, finish, streamed, done = anyio.Event(), anyio.Event(), anyio.Event(), anyio.Event()
    opened, close = threading.Event(), threading.Event()

    class Surface(PromptSurface):
        def changed(self) -> None:
            super().changed()
            if 'Finished work' in _plain(self.transcript):
                streamed.set()

    monkeypatch.setattr('pydantic_clai2.ui.prompt.live_prompt.PromptSurface', Surface)

    def menu() -> str:
        opened.set()
        close.wait(5)
        return 'menu closed'

    async def open_menu(args: list[str]) -> str:
        return await run_worker(menu)

    agent = Agent(TestModel(call_tools=['work'], custom_output_text='Finished work'), deps_type=type(None))

    @agent.tool_plain
    async def work() -> str:
        working.set()
        await finish.wait()
        return 'done'

    output = io.StringIO()

    async def run() -> None:
        await chat(
            agent,
            deps=None,
            plugins=[_MenuCommand(Command(name='menu', description='Menu', handler=open_menu, during_turn=True))],
            console=Console(file=output, force_terminal=True, width=80, height=24),
            store=SettingsStore(tmp_path / 'config.db'),
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            pipe.send_text('start\r')
            await working.wait()
            pipe.send_text('/menu\r')
            assert await in_worker(opened.wait, 5)
            finish.set()
            await streamed.wait()
            assert 'Finished work' not in output.getvalue()
            close.set()
            pipe.send_text('/exit\r')
            await done.wait()
    text = Text.from_ansi(output.getvalue().rsplit(LEAVE, 1)[1]).plain
    assert text.index('> /menu') < text.index('Finished work') < text.index('menu closed') < text.index('Goodbye.')


async def test_overlay_commands_take_the_screen_without_waiting_for_themselves() -> None:
    """Plugin code a mid-turn `/plugins` runs, here or from its menu thread, already owns the screen."""
    screen = Screen()
    taken: list[str] = []
    asked, entered = anyio.Event(), anyio.Event()

    @asynccontextmanager
    async def take() -> AsyncGenerator[None]:
        taken.append('stream paused')
        yield

    async def from_tool() -> None:
        await asked.wait()
        async with screen.full():
            entered.set()

    def from_menu_thread(loop: asyncio.AbstractEventLoop) -> None:
        async def action() -> None:
            async with screen.full():
                taken.append('menu action')

        asyncio.run_coroutine_threadsafe(action(), loop).result()

    with screen.bound(take):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(from_tool)
            async with screen.overlay():
                asked.set()
                async with screen.full():
                    taken.append('configure')
                loop = asyncio.get_running_loop()
                await run_worker(lambda: from_menu_thread(loop))
                await anyio.wait_all_tasks_blocked()
                assert not entered.is_set()
            await entered.wait()
            async with screen.full():
                taken.append('later')
    assert taken == ['configure', 'menu action', 'stream paused', 'stream paused', 'later']


ALPHA = """
from pydantic_ai.capabilities import Hooks
from pydantic_clai2.plugins import Plugin, SessionEnd


class Alpha(Plugin):
    def get_capabilities(self):
        return (Hooks(),)

    async def on_session_end(self, event: SessionEnd) -> None:
        self.host.console.print('alpha ended')
"""


@pytest.mark.parametrize('delegating', [False, True])
async def test_plugins_typed_mid_turn_apply_at_once_and_end_after_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delegating: bool
) -> None:
    """`/plugins` with arguments runs as soon as it is entered; the turn keeps what it bound until it ends."""
    working, finish, applied, done = anyio.Event(), anyio.Event(), anyio.Event(), anyio.Event()
    written: list[str] = []
    reply = 'Plugin changes wait for delegated tasks' if delegating else 'Disabled alpha.'

    class Surface(PromptSurface):
        def write(self, text: str) -> int:
            written.append(text)
            if reply in ''.join(written):
                applied.set()
            return super().write(text)

    monkeypatch.setattr('pydantic_clai2.ui.prompt.live_prompt.PromptSurface', Surface)
    store = SettingsStore(tmp_path / 'config.db')
    store.plugins_dir.mkdir(parents=True)
    (store.plugins_dir / 'alpha.py').write_text(ALPHA)
    agent = Agent(TestModel(call_tools=['work'], custom_output_text='Finished work'), deps_type=type(None))
    output = io.StringIO()
    shell = create_shell(
        agent,
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=output, force_terminal=True, width=120, height=24),
        settings=None,
        store=store,
        builtin_plugins=(),
        project=ProjectSettings(),
    )

    @agent.tool_plain
    async def work() -> str:
        if delegating:
            child = task(conversation_id=shell.session.summary.id)
            shell.tasks.owner.records[child.id] = child
        working.set()
        await finish.wait()
        shell.tasks.owner.records.clear()
        return 'done'

    async def run() -> None:
        await shell.loader.load_all()
        assert await shell.run() == 'exit'
        await shell.loader.close('exit')
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            pipe.send_text('start\r')
            await working.wait()
            pipe.send_text('/plugins disable alpha\r')
            await applied.wait()
            assert 'alpha ended' not in ''.join(written)
            assert [plugin.enabled for plugin in store.plugins()] == ([] if delegating else [False])
            finish.set()
            pipe.send_text('/exit\r')
            await done.wait()
    # The queued `/exit` echoes once the turn is over and its plugin hooks have run.
    text = output.getvalue()
    assert text.index('> /plugins disable alpha') < text.index(reply) < text.index('> /exit')
    if delegating:
        assert TURN_NOTICE not in text
        assert text.index('Goodbye.') < text.index('alpha ended')
    else:
        assert f'Disabled alpha.\n{TURN_NOTICE}' in text
        assert text.index(reply) < text.index('alpha ended') < text.index('> /exit')


def _plain(transcript: TranscriptBuffer) -> str:
    """Streamed Markdown lands in the transcript, not in `write`."""
    return Text.from_ansi('\n'.join(transcript.frame(width=200, height=500).rows)).plain


async def test_speculation_toggled_mid_turn_shows_at_once_and_binds_on_the_next_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Ctrl+X Ctrl+S` saves the switch and repaints at once; the running turn keeps its tools."""
    working, finish, noticed, answered, done = (anyio.Event() for _ in range(5))
    frames: list[str] = []

    class Surface(PromptSurface):
        def changed(self) -> None:
            super().changed()
            if _plain(self.transcript).count('Finished work') >= 2:
                answered.set()

        def paint(self, rows: tuple[str, ...]) -> None:
            frames.append('\n'.join(Text.from_ansi(row).plain for row in rows))
            if 'this turn keeps its tools' in frames[-1]:
                noticed.set()
            super().paint(rows)

    monkeypatch.setattr('pydantic_clai2.ui.prompt.live_prompt.PromptSurface', Surface)
    store = SettingsStore(tmp_path / 'config.db')
    tools: list[list[str]] = []
    hooks = Hooks[None]()

    @hooks.on.before_model_request
    async def capture(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        tools.append(sorted(tool.name for tool in request_context.model_request_parameters.function_tools))
        return request_context

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='work', json_args='{}')}
        else:
            yield 'Finished work'

    agent = Agent(FunctionModel(stream_function=respond), deps_type=type(None))

    @agent.tool_plain
    async def work() -> str:
        working.set()
        await finish.wait()
        return 'done'

    async def run() -> None:
        await chat(
            agent,
            deps=None,
            plugins=[hooks],
            console=Console(file=io.StringIO(), force_terminal=True, width=120, height=24),
            store=store,
        )
        done.set()

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run)
            pipe.send_text('start\r')
            await working.wait()
            pipe.send_text('\x18\x13')
            await noticed.wait()
            assert store.overrides() == {'run.speculative_code_mode': True}
            assert 'Speculative Execution  0 hits' in frames[-1]
            assert 'on from the next prompt' in frames[-1]
            finish.set()
            pipe.send_text('again\r')
            await answered.wait()
            assert 'next prompt' not in frames[-1]
            pipe.send_text('/exit\r')
            await done.wait()
    first, after_tool, next_prompt = tools
    assert first == after_tool == ['work']
    assert 'run_code' in next_prompt and 'work' not in next_prompt

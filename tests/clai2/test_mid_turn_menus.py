"""Menus opened while a turn streams: the run's output waits behind the menu, in order."""

import io
import threading
from collections.abc import AsyncGenerator, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import anyio
import pytest
from anyio.to_thread import run_sync as in_worker
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from rich.console import Console

from pydantic_clai2 import Session, chat
from pydantic_clai2.command_context import CommandContext
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.menu_worker import holding_output, run_worker
from pydantic_clai2.prompt_surface import PromptSurface
from pydantic_clai2.screen import Screen
from pydantic_clai2.session_settings import SessionSettings
from pydantic_clai2.settings_store import SettingsStore


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def test_only_bare_opted_in_commands_run_during_a_turn() -> None:
    commands = Commands()
    commands.register(Command(name='menu', description='', handler=lambda _: '', during_turn=True))
    commands.register(Command(name='plain', description='', handler=lambda _: ''))
    assert commands.runs_during_turn('/menu')
    assert not commands.runs_during_turn('/menu value')
    assert not commands.runs_during_turn('/plain')
    assert not commands.runs_during_turn('/unknown')
    assert not commands.runs_during_turn('/tmp/menu')
    assert not commands.runs_during_turn('menu')


async def test_run_worker_holds_output_only_while_the_widget_runs() -> None:
    log: list[str] = []

    @contextmanager
    def hold() -> Generator[None]:
        log.append('hold')
        yield
        log.append('replay')

    with holding_output(hold):
        assert await run_worker(lambda: log.append('menu') or 'done') == 'done'
    await run_worker(lambda: log.append('unheld'))
    assert log == ['hold', 'menu', 'replay', 'unheld']


async def test_overlay_takes_turns_with_widgets_without_pausing_the_stream() -> None:
    log: list[str] = []
    screen = Screen()
    entered = anyio.Event()

    @asynccontextmanager
    async def take() -> AsyncGenerator[None]:
        log.append('paused stream')
        yield

    async def question() -> None:
        async with screen.full():
            entered.set()

    with screen.bound(take):
        async with anyio.create_task_group() as tasks:
            async with screen.overlay():
                tasks.start_soon(question)
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


class _MenuCommand(AbstractCapability[None]):
    def __init__(self, menu: Command) -> None:
        self.menu = menu

    def get_commands(self, context: CommandContext) -> Sequence[Command]:
        return (self.menu,)


async def test_menu_opened_mid_turn_holds_the_run_output_until_it_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    working, finish, streamed, done = anyio.Event(), anyio.Event(), anyio.Event(), anyio.Event()
    opened, close = threading.Event(), threading.Event()
    written: list[str] = []

    class Surface(PromptSurface):
        def write(self, text: str) -> int:
            written.append(text)
            if 'Finished work' in ''.join(written):
                streamed.set()
            return super().write(text)

    monkeypatch.setattr('pydantic_clai2.live_prompt.PromptSurface', Surface)

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
            pipe.send_text('start\n')
            await working.wait()
            pipe.send_text('/menu\n')
            await in_worker(opened.wait)
            finish.set()
            await streamed.wait()
            assert 'Finished work' not in output.getvalue()
            close.set()
            pipe.send_text('/exit\n')
            await done.wait()
    text = output.getvalue()
    assert text.index('> /menu') < text.index('Finished work') < text.index('menu closed') < text.index('Goodbye.')

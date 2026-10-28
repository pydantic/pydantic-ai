"""The terminal shows one live session at a time: `/resume` of one the browser opened shows it there."""

import io
import signal
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from pathlib import Path
from typing import ClassVar, cast

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.menu import MenuResult

from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent, ModelMessage, UserContent, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2 import chat
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import LiveSessions, Plugin, SessionStart
from pydantic_clai2.runtime.live_runtimes import LiveRuntimes
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.ui.menus import dialogs
from pydantic_clai2.ui.menus.session_browser import SessionBrowser
from pydantic_clai2.ui.menus.update_menu import VERSION
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.prompt.terminal import NOWHERE, Terminal
from tests.clai2.dialog_script import ScriptedDialogs
from tests.clai2.menu_script import Script, pick
from tests.clai2.surface_terminal import SurfaceTerminal
from tests.clai2.test_live_runtimes import Fleet, Recorder, blocked, until


class Desk(Fleet):
    """A terminal CLAI: the session it started with, shown in a terminal, beside sessions the browser opens."""

    def __init__(self, root: Path, *, agent: Agent[None, str] | None = None) -> None:
        super().__init__(root)
        self.screen = SurfaceTerminal(width=100, height=30)
        self.home = create_shell(
            agent or Agent(TestModel()),
            deps=None,
            plugins=(),
            usage_limits=None,
            # A terminal by `isatty`, as a real one: `force_terminal` would hide a file handed elsewhere.
            console=Console(file=self.screen, width=100, height=30),
            settings=None,
            store=None,
            builtin_plugins=(),
            project=ProjectSettings(),
            workspace=self.folder('home'),
            process=self.process,
        )
        assert self.home.terminal is not None
        self.terminal = self.home.terminal

    def shown(self) -> str:
        return '\n'.join([*self.screen.history, *self.screen.lines()])

    async def run(self, body: Callable[[], Awaitable[None]]) -> str:
        """Run the terminal's session while `body` drives the others; the reason the session shown last ended."""
        reasons: list[str] = []

        async def drive() -> None:
            reasons.append(await self.home.run())

        await self.home.loader.load_all()
        # Switching a new database to WAL can fail at once when two first saves race, so make it first.
        await self.process.conversations.listing()
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(20):
            async with self.live.serving(), anyio.create_task_group() as group:
                group.start_soon(drive)
                await body()
        return reasons[0]


def toolbar(runtime: SessionRuntime[None, str]) -> str:
    return ''.join(text for _, text in runtime.status.toolbar())


async def test_resume_of_a_session_live_in_the_browser_shows_it_without_forking(tmp_path: Path) -> None:
    """Focus moves mid-turn: the terminal shows its history and the rest of its turn; Ctrl-C follows."""
    started, release, calls = {'b': anyio.Event()}, {'b': anyio.Event()}, {'b': 0}
    desk = Desk(tmp_path)
    desk.agents['b'] = blocked('b', started, release, calls)
    home, terminal = desk.home, desk.terminal
    original = signal.getsignal(signal.SIGINT)

    async def body() -> None:
        home.send_input('/cost')
        await until(desk.live, lambda: home.idle)
        assert ' live' not in toolbar(home), 'one live session shows nothing new'
        b = await desk.live.new(workspace=desk.folder('b'))
        other = desk.runtimes[b.id]
        b.send_input('go')
        await started['b'].wait()
        assert ' 2 live' in toolbar(home)
        conversation = b.conversation.conversation_id
        # A copy saved to continue it while another process ran it: the live session holding it still wins.
        await desk.process.conversations.save(
            summary=ConversationSummary(workspace=str(desk.folder('b')), title='Copy', forked_from=conversation),
            messages=[],
        )
        home.send_input(f'/resume {conversation}')
        await until(desk.live, lambda: terminal.shown is other and home.idle)
        assert desk.live.focused == b.id and other.terminal is terminal and home.terminal is None
        assert other.editor is terminal.editor and home.editor is None
        assert terminal.editor is not None and terminal.editor.interrupts.active, 'Esc interrupts the turn shown'
        assert other.interrupts.signals and not home.interrupts.signals
        assert signal.getsignal(signal.SIGINT) is not original, 'the turn shown takes Ctrl-C'
        assert other.session.conversation_id == conversation, 'no fork'
        assert [session.id for session in desk.live.listing()] == [home.live_session.id, b.id]
        assert ' 2 live' in toolbar(other)
        release['b'].set()
        await until(desk.live, lambda: other.idle)
        assert signal.getsignal(signal.SIGINT) is original
        # The session no longer shown still takes input, as the browser gives it, and prints nowhere.
        assert home.console.file is NOWHERE
        home.send_input('/cost')
        assert home.queued == 1
        await until(desk.live, lambda: home.idle)
        # The session shown moves the terminal on in turn: back to the first one, and to this one again.
        b.send_input(f'/resume {home.session.conversation_id}')
        await until(desk.live, lambda: terminal.shown is home and other.idle)
        home.send_input(f'/resume {conversation}')
        await until(desk.live, lambda: terminal.shown is other and home.idle)
        b.send_input('/exit')

    assert await desk.run(body) == 'exit'
    (other,) = desk.runtimes.values()
    assert terminal.ended_in is other
    assert terminal.shown is home and home.terminal is terminal
    assert f'Showing session {other.session.conversation_id}' in desk.shown()
    assert 'b answered' in desk.shown()
    assert home.console.file is not NOWHERE
    await home.close('exit')


async def test_restarts_are_refused_while_other_sessions_run_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = {name: anyio.Event() for name in 'bc'}
    release = {name: anyio.Event() for name in 'bc'}
    calls = {'b': 0, 'c': 0}
    desk = Desk(tmp_path)
    desk.agents = {name: blocked(name, started, release, calls) for name in 'bc'}
    home = desk.home

    async def update(args: list[str]) -> str:
        return 'Updated.'

    monkeypatch.setattr(home.updates, 'command', update)
    # Switching a new database to WAL can fail at once when two first saves race, so make it first.
    await desk.process.conversations.listing()
    with anyio.fail_after(20):
        async with desk.live.serving():
            b = await desk.live.new(workspace=desk.folder('b'))
            c = await desk.live.new(workspace=desk.folder('c'))
            b.send_input('go')
            c.send_input('go')
            await started['b'].wait()
            await started['c'].wait()
            for command in ('/reload', '/update now'):
                with pytest.raises(ValueError, match='2 other sessions are running turns; try again when they finish'):
                    await home.commands.execute_async(command)
            # Bare `/update` can still show its read-only status and channel while an install is refused; its
            # Update action says why and stays in the menu.
            shown = Script(lists=[pick(VERSION), MenuResult(cancelled=True)], choices=[], texts=[])
            monkeypatch.setattr(dialogs, 'RUNNERS', shown.runners)
            home.updates.fetch = lambda channel: home.updates.current.version
            home.updates.spawn = lambda work: work()
            assert (
                await home.commands.execute_async('/update')
                == '2 other sessions are running turns; try again when they finish.'
            )
            assert shown.opened == ['list', 'list']
            # The web UI's About menu refuses an install as an error, so the page shows no saved row.
            about = next(command for command in home.commands if command.name == 'update').menu
            assert about is not None
            with pytest.raises(ValueError, match='2 other sessions are running turns; try again when they finish'):
                await about().actions[VERSION](ScriptedDialogs([]))
            release['c'].set()
            await until(desk.live, lambda: not c.running)
            with pytest.raises(ValueError, match='1 other session is running a turn; try again when it finishes'):
                await home.commands.execute_async('/reload')
            release['b'].set()
            await until(desk.live, lambda: not b.running)
            assert await home.commands.execute_async('/update now') == 'Updated.'
            assert await home.commands.execute_async('/reload') == 'Reloading CLAI2...'
    assert home.reload_requested
    await home.close('exit')


async def test_a_session_shown_in_the_terminal_restarts_clai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A hidden session cannot open a terminal menu or restart CLAI; the shown session can restart."""
    desk = Desk(tmp_path)
    home, terminal = desk.home, desk.terminal
    monkeypatch.setattr(dialogs, 'RUNNERS', Script(lists=[MenuResult(cancelled=True)], choices=[], texts=[]).runners)
    home.updates.fetch = lambda channel: home.updates.current.version
    home.updates.spawn = lambda work: work()

    async def body() -> None:
        b = await desk.live.new()
        other = desk.runtimes[b.id]
        with pytest.raises(ValueError, match='Enter them in the session the terminal shows'):
            await other.commands.execute_async('/reload')
        assert await home.show_live(b.conversation.conversation_id) == ''
        # Hidden now, the session CLAI started with cannot borrow the terminal's menu or restart.
        for command in ('/reload', '/update', '/update now'):
            with pytest.raises(ValueError, match='Enter them in the session the terminal shows'):
                await home.commands.execute_async(command)
        assert not home.reload_requested
        assert await home.show_live(home.session.conversation_id) is None, 'not while hidden'
        assert await other.show_live(other.session.conversation_id) is None, 'its own'
        assert await other.show_live('unknown') is None
        b.send_input('/reload')

    assert await desk.run(body) == 'exit'
    assert terminal.ended_in is not None and terminal.ended_in.reload_requested
    assert not terminal.home_closed
    await home.close('exit')


async def test_a_session_shown_from_another_folder_does_not_restart_clai(tmp_path: Path) -> None:
    """The restart resumes the session shown where CLAI started, which works only in the session's own folder."""
    desk = Desk(tmp_path)
    home = desk.home

    async def body() -> None:
        b = await desk.live.new(workspace=desk.folder('b'))
        assert await home.show_live(b.conversation.conversation_id) == ''
        for command in ('/reload', '/update now'):
            with pytest.raises(ValueError, match=r'where it started, and this session works in .*b; restart it'):
                await desk.runtimes[b.id].commands.execute_async(command)
        b.send_input('/exit')

    assert await desk.run(body) == 'exit'
    await home.close('exit')


async def test_the_session_clai_started_with_closes_when_idle_while_another_is_shown(tmp_path: Path) -> None:
    desk = Desk(tmp_path)
    home, terminal = desk.home, desk.terminal
    now = [0.0]
    desk.live.clock = lambda: now[0]

    async def body() -> None:
        now[0] += desk.live.idle_after
        await desk.live.close_idle()
        assert home.live_session in desk.live.listing(), 'never while shown'
        b = await desk.live.new(workspace=desk.folder('b'))
        with pytest.raises(ValueError, match='runs under its own driver'):
            await desk.live.close(home.live_session.id)
        home.send_input(f'/resume {b.conversation.conversation_id}')
        await until(desk.live, lambda: terminal.shown is desk.runtimes[b.id] and home.idle)
        now[0] += desk.live.idle_after
        await desk.live.close_idle()
        assert [session.id for session in desk.live.listing()] == [b.id]
        assert ('home', 'exit') in Recorder.ended
        b.send_input('/exit')

    assert await desk.run(body) == 'exit'
    assert terminal.home_closed


async def test_the_session_clai_started_with_ends_with_its_own_input_while_another_is_shown(tmp_path: Path) -> None:
    desk = Desk(tmp_path)
    home, terminal = desk.home, desk.terminal

    async def body() -> None:
        b = await desk.live.new(workspace=desk.folder('b'))
        await home.show_live(b.conversation.conversation_id)
        home.send_input('/exit')
        await until(desk.live, lambda: home.live_session not in desk.live.listing())
        assert ('home', 'exit') in Recorder.ended
        b.send_input('/exit')

    assert await desk.run(body) == 'exit'
    assert terminal.home_closed


async def test_ctrl_c_follows_the_terminal_between_running_operations() -> None:
    original = signal.getsignal(signal.SIGINT)
    shown, hidden = Interrupts(), Interrupts(signals=False)
    hidden.focus(signals=False)  # Nothing runs yet: only the switch changes.
    started = {name: anyio.Event() for name in ('shown', 'hidden')}
    release, interrupted = anyio.Event(), anyio.Event()
    results: dict[str, bool] = {}

    async def wait(name: str) -> None:
        started[name].set()
        await release.wait()

    async def run(name: str, interrupts: Interrupts) -> None:
        results[name] = await interrupts.run(wait(name))
        if name == 'hidden':
            interrupted.set()

    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(run, 'shown', shown)
            await started['shown'].wait()
            group.start_soon(run, 'hidden', hidden)
            await started['hidden'].wait()
            taken = signal.getsignal(signal.SIGINT)
            assert taken is not original
            shown.focus(signals=False)
            hidden.focus(signals=True)
            hidden.focus(signals=True)
            assert signal.getsignal(signal.SIGINT) not in (original, taken)
            signal.raise_signal(signal.SIGINT)
            await interrupted.wait()
            assert signal.getsignal(signal.SIGINT) is original
            release.set()
    assert results == {'shown': True, 'hidden': False}
    assert signal.getsignal(signal.SIGINT) is original


async def test_line_mode_terminal_follows_the_session_shown(tmp_path: Path) -> None:
    """Without a terminal console, the line prompt completes the commands of the session shown."""
    desk = Desk(tmp_path)
    terminal = Terminal[None, str](
        console=Console(file=io.StringIO()), transcript=TranscriptBuffer(), history=InMemoryHistory()
    )
    async with desk.live.serving():
        b = desk.runtimes[(await desk.live.new(workspace=desk.folder('b'))).id]
        terminal.attach(desk.home)
        prompt = terminal.prompt
        terminal.attach(b)
        assert terminal.prompt is prompt and b.prompt is prompt and desk.home.prompt is None
        assert b.transcript is terminal.transcript and desk.home.transcript is None
        assert terminal._completer.commands is b.commands  # pyright: ignore[reportPrivateUsage]
    await desk.home.close('exit')


async def test_the_terminal_raises_what_its_session_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    desk = Desk(tmp_path)

    async def broken() -> str:
        raise RuntimeError('the loop broke')

    monkeypatch.setattr(desk.home, 'loop', broken)
    # The editor's own task group wraps it once more; `chat` unwraps that.
    with pytest.raises(ExceptionGroup) as caught:
        await desk.home.run()
    assert caught.group_contains(RuntimeError, match='the loop broke', depth=1)
    await desk.home.close('error')


def test_sessions_opened_beside_the_terminal_print_in_its_colours() -> None:
    def colours(console: Console) -> str | None:
        terminal = Terminal[None, str](console=console, transcript=TranscriptBuffer(), history=InMemoryHistory())
        session = terminal.session_console()
        assert session.file is NOWHERE and not session.is_terminal
        return session.color_system

    assert colours(Console(file=io.StringIO(), force_terminal=True, color_system='truecolor')) == 'truecolor'
    assert colours(Console(file=io.StringIO(), color_system=None)) is None


async def test_serving_raises_what_its_block_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='from the block'):
        async with Fleet(tmp_path).live.serving():
            raise RuntimeError('from the block')


async def test_the_resume_browser_shows_a_live_session_and_wakes_it_for_its_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desk = Desk(tmp_path)
    home, terminal = desk.home, desk.terminal
    owner = home.tasks.owner
    reported: list[str] = []

    async def body() -> None:
        b = await desk.live.new(workspace=desk.folder('b'))
        other = desk.runtimes[b.id]

        def reports(*, conversation_id: str) -> bool:
            # One report waits for the session shown next, once.
            if conversation_id != b.conversation.conversation_id:
                return False
            reported.append(conversation_id)
            return len(reported) == 1

        def pick(browser: SessionBrowser) -> str:
            return b.conversation.conversation_id

        monkeypatch.setattr(SessionBrowser, 'run', pick)
        monkeypatch.setattr(owner, 'reports', reports)
        home.send_input('/resume')
        await until(desk.live, lambda: terminal.shown is other and home.idle)
        await until(desk.live, lambda: len(reported) > 1 and other.idle)
        b.send_input('/exit')

    assert await desk.run(body) == 'exit'
    assert 'Showing session' in desk.shown()
    await home.close('exit')


async def test_prompts_queued_for_a_session_stay_its_own_when_the_terminal_shows_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The resume browser opens over a running turn; the prompt queued behind that turn runs in its session."""
    started, release, calls = {'home': anyio.Event()}, {'home': anyio.Event()}, {'home': 0}
    desk = Desk(tmp_path, agent=blocked('home', started, release, calls))
    home, terminal = desk.home, desk.terminal

    async def body() -> None:
        b = await desk.live.new(workspace=desk.folder('b'))
        other = desk.runtimes[b.id]

        def pick(browser: SessionBrowser) -> str:
            return b.conversation.conversation_id

        monkeypatch.setattr(SessionBrowser, 'run', pick)
        home.send_input('first')
        await started['home'].wait()
        home.send_input('second')
        assert home.queued == 1
        home.send_input('/resume')
        await until(desk.live, lambda: terminal.shown is other)
        assert (home.queued, other.queued) == (1, 0)
        release['home'].set()
        await until(desk.live, lambda: calls['home'] == 2 and home.idle)
        b.send_input('/exit')

    assert await desk.run(body) == 'exit'
    (other,) = desk.runtimes.values()
    assert (prompts(home), prompts(other)) == (['first', 'second'], [])
    await home.close('exit')


async def test_a_session_shown_mid_answer_streams_the_rest_in_the_terminal(tmp_path: Path) -> None:
    streaming, release = anyio.Event(), anyio.Event()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'early words\nhalf a '
        streaming.set()
        await release.wait()
        yield 'line\nlate words\n'

    desk = Desk(tmp_path)
    desk.agents['b'] = Agent(FunctionModel(stream_function=stream))
    home, terminal = desk.home, desk.terminal

    async def body() -> None:
        b = await desk.live.new(workspace=desk.folder('b'))
        other = desk.runtimes[b.id]
        b.send_input('go')
        await streaming.wait()
        home.send_input(f'/resume {b.conversation.conversation_id}')
        await until(desk.live, lambda: terminal.shown is other and home.idle)
        release.set()
        await until(desk.live, lambda: other.idle)
        b.send_input('/exit')

    assert await desk.run(body) == 'exit'
    assert 'half a line' in desk.shown() and 'late words' in desk.shown()
    assert 'early words' not in desk.shown(), 'streamed while hidden'
    await home.close('exit')


async def test_queued_images_stay_with_their_prompt_as_the_terminal_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prompts queued with images behind running turns keep them, whether the terminal leaves or shows their session."""
    started = {name: anyio.Event() for name in ('home', 'b')}
    release = {name: anyio.Event() for name in ('home', 'b')}
    calls = {'home': 0, 'b': 0}
    desk = Desk(tmp_path, agent=blocked('home', started, release, calls))
    desk.agents['b'] = blocked('b', started, release, calls)
    home, terminal = desk.home, desk.terminal
    red, blue = BinaryContent(b'red', media_type='image/png'), BinaryContent(b'blue', media_type='image/png')
    green = BinaryContent(b'green', media_type='image/png')

    async def body() -> None:
        b = await desk.live.new(workspace=desk.folder('b'))
        other = desk.runtimes[b.id]

        def pick(browser: SessionBrowser) -> str:
            return b.conversation.conversation_id

        monkeypatch.setattr(SessionBrowser, 'run', pick)
        home.send_input('first')
        b.send_input('first')
        await started['home'].wait()
        await started['b'].wait()
        # Home's in the terminal's editor, B's in its inbox; then the resume browser, open over the turn, shows B.
        home.send_input('home look', images=[red])
        b.send_input('b look', images=[blue])
        # As a prompt rejected for want of a model keeps its images for the retry.
        rejected = f'retry {home.images.attach([green])}'
        home.images.retry_text = rejected
        home.send_input('/resume')
        await until(desk.live, lambda: terminal.shown is other)
        held = {
            name: sorted(image.data for image in images.pending.values())
            for name, images in [('home', home.images), ('terminal', terminal.images)]
        }
        assert held == {'home': [b'green', b'red'], 'terminal': [b'blue']}, 'no image is held, and counted, twice'
        assert home.images.retry_text == rejected and home.images.resolve(rejected)[1] == [green]
        assert terminal.images.retry_text == '', 'it went with its session, not left for the one shown'
        release['home'].set()
        release['b'].set()
        await until(desk.live, lambda: calls == {'home': 2, 'b': 2} and home.idle and other.idle)
        b.send_input('/exit')

    assert await desk.run(body) == 'exit'
    (other,) = desk.runtimes.values()
    assert (user_content(home), user_content(other)) == (
        ['first', ['home look', red]],
        ['first', ['b look', blue]],
    )
    await home.close('exit')


def user_content(runtime: SessionRuntime[None, str]) -> list[str | Sequence[UserContent]]:
    return [
        part.content
        for message in runtime.session.messages
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


def prompts(runtime: SessionRuntime[None, str]) -> list[str]:
    return [
        part.content
        for message in runtime.session.messages
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


async def test_clai2_web_keeps_its_terminal_on_the_session_it_started_with(tmp_path: Path) -> None:
    fleet = Fleet(tmp_path)
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=SurfaceTerminal(width=100, height=30), width=100, height=30),
        settings=None,
        store=None,
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
        workspace=fleet.folder('web'),
        process=fleet.process,
    )
    assert shell.terminal is not None and shell.editor is None
    assert not shell.terminal.shows_others
    assert await shell.show_live(shell.session.conversation_id) is None
    await shell.close('exit')


class Opener(Plugin):
    """A process-wide plugin that keeps `host.live`, as the web UI does, to open sessions beside the terminal's."""

    scope = 'process'
    live: ClassVar[list[LiveSessions]] = []
    started: ClassVar[anyio.Event] = anyio.Event()

    async def on_session_start(self, event: SessionStart) -> None:
        assert self.host.live is not None
        Opener.live.append(self.host.live)
        Opener.started.set()


async def test_chat_opens_sessions_for_the_browser_beside_the_terminal(tmp_path: Path) -> None:
    """Opened beside the terminal's, a session gets the settings of its folder and moves the terminal's focus."""
    Opener.live.clear()
    Opener.started = anyio.Event()
    screen = SurfaceTerminal(width=100, height=30)
    elsewhere = tmp_path / 'elsewhere'
    (elsewhere / '.git').mkdir(parents=True)
    (elsewhere / '.clai').mkdir()
    (elsewhere / '.clai' / 'settings.json').write_text('{"instructions": "Work elsewhere."}')
    seen: list[dict[str, JsonValue]] = []

    def settings_for(project: ProjectSettings) -> Settings:
        seen.append(project.overrides)
        return Settings(model='test')

    async def drive() -> None:
        await Opener.started.wait()
        live = Opener.live[0]
        assert isinstance(live, LiveRuntimes)
        (home,) = live.listing()
        b = await live.new(workspace=elsewhere)
        here = await live.new()
        assert b.workspace == elsewhere.resolve() and here.workspace == home.workspace
        home.send_input(f'/resume {b.conversation.conversation_id}')
        await until(live, lambda: live.focused == b.id)
        # The session CLAI started with ends first, from the browser; the terminal goes on showing the other.
        home.send_input('/exit')
        await until(live, lambda: home not in live.listing())
        b.send_input('/exit')

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(20):
        async with anyio.create_task_group() as group:
            group.start_soon(drive)
            await chat(
                Agent(TestModel()),
                deps=None,
                console=Console(file=screen, force_terminal=True, width=100, height=30),
                store=SettingsStore(tmp_path / 'config.db'),
                builtin_plugins=[PluginSettings(id='opener', factory=f'{__name__}:Opener')],
                settings_for=settings_for,
            )
    assert seen == [{'run.instructions': 'Work elsewhere.'}, {}]
    assert 'Showing session' in '\n'.join([*screen.history, *screen.lines()])


async def test_a_failed_reload_from_a_shown_session_keeps_its_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    Opener.live.clear()
    Opener.started = anyio.Event()
    screen = SurfaceTerminal(width=100, height=30)

    def broken(build: Callable[[], object]) -> object:
        raise RuntimeError('broken source')

    monkeypatch.setattr('pydantic_clai2.app.reload_clai', broken)
    conversations: list[str] = []

    async def drive() -> None:
        await Opener.started.wait()
        live = Opener.live[0]
        assert isinstance(live, LiveRuntimes)
        (home,) = live.listing()
        b = await live.new()
        b.send_input('hello')
        await until(live, lambda: b.session_id is not None and not b.running)
        conversations.append(b.conversation.conversation_id)
        home.send_input(f'/resume {conversations[0]}')
        await until(live, lambda: live.focused == b.id)
        # CLAI starts again in the session it started with, whose plugins load once its conversation is chosen.
        Opener.started = anyio.Event()
        b.send_input('/reload')
        await Opener.started.wait()
        (again,) = live.listing()
        assert again.conversation.conversation_id == conversations[0]
        served = live.runtime(again.conversation.conversation_id)
        assert isinstance(served, SessionRuntime)
        runtime = cast(SessionRuntime[None, str], served)
        # Closed for the reload, it hears titles and task events again, once.
        assert runtime.process.namer.listeners.count(runtime.sessions.named) == 1
        assert runtime.process.tasks.views.count(runtime.tasks) == 1
        again.send_input('/exit')

    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(20):
        async with anyio.create_task_group() as group:
            group.start_soon(drive)
            await chat(
                Agent(TestModel()),
                deps=None,
                console=Console(file=screen, force_terminal=True, width=100, height=30),
                store=SettingsStore(tmp_path / 'config.db'),
                builtin_plugins=[PluginSettings(id='opener', factory=f'{__name__}:Opener')],
            )
    assert 'Reload failed: RuntimeError: broken source' in '\n'.join([*screen.history, *screen.lines()])

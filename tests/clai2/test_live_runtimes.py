"""Several live sessions in one CLAI process: `LiveRuntimes` opens, runs, and closes them, each in its own folder."""

import gc
import io
import os
import re
import weakref
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import ClassVar

import anyio
import anyio.lowlevel
import httpx
import pytest
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.builtin_plugins.web.app import HEADER
from pydantic_clai2.cli import web as web_module
from pydantic_clai2.cli.web import run_web
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import LiveSession, LiveSessions, Plugin, SessionEnd, SessionEndReason
from pydantic_clai2.plugins.process_plugins import JoinedSessions, handling
from pydantic_clai2.runtime.live_runtimes import LiveRuntimes
from pydantic_clai2.runtime.process import ClaiProcess
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell


class Recorder(Plugin):
    """Records how each session's plugins ended, by the session's folder name."""

    ended: ClassVar[list[tuple[str, SessionEndReason]]] = []

    async def on_session_end(self, event: SessionEnd) -> None:
        Recorder.ended.append((self.host.workspace.name, event.reason))


RECORDER = PluginSettings(id='recorder', factory=f'{__name__}:Recorder')


class Gate(Plugin):
    """Holds the end of the session in a folder named `held` until `release` is set."""

    ending: ClassVar[anyio.Event] = anyio.Event()
    release: ClassVar[anyio.Event] = anyio.Event()

    async def on_session_end(self, event: SessionEnd) -> None:
        if self.host.workspace.name == 'held':
            Gate.ending.set()
            await Gate.release.wait()


class _Output(io.StringIO):
    """Console output that tells when `clai2 --web` is ready for its user."""

    def __init__(self) -> None:
        super().__init__()
        self.ready = anyio.Event()

    def write(self, s: str) -> int:
        if 'Press Ctrl-C to quit.' in s:
            self.ready.set()
        return super().write(s)


class Fleet:
    """One process whose `live.build` makes a runtime per folder, with that folder's agent, project, and console."""

    def __init__(self, root: Path, *, builtin: tuple[PluginSettings, ...] = (RECORDER,)) -> None:
        Recorder.ended.clear()
        self.root = root
        self.process = ClaiProcess[None].open(SettingsStore(root / 'config.db'), builtin_plugins=builtin)
        self.agents: dict[str, Agent[None, str]] = {}
        self.projects: dict[str, ProjectSettings] = {}
        self.runtimes: dict[str, SessionRuntime[None, str]] = {}
        """By live session ID."""
        self.outputs: dict[str, io.StringIO] = {}
        """By folder name."""
        self.built: list[str] = []
        self.process.live.build = self.build

    @property
    def live(self) -> LiveRuntimes:
        return self.process.live

    def folder(self, name: str) -> Path:
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        return directory

    def build(self, directory: Path | None) -> SessionRuntime[None, str]:
        directory = directory or self.folder('home')
        output = self.outputs[directory.name] = io.StringIO()
        runtime = create_shell(
            self.agents.get(directory.name) or Agent(TestModel()),
            deps=None,
            plugins=(),
            usage_limits=None,
            console=Console(file=output, force_terminal=False, width=200),
            settings=None,
            store=None,
            builtin_plugins=(),
            project=self.projects.get(directory.name, ProjectSettings()),
            headless=True,
            workspace=directory,
            process=self.process,
            signals=False,
        )
        self.runtimes[runtime.live_session.id] = runtime
        self.built.append(directory.name)
        return runtime


async def until(live: LiveSessions, check: Callable[[], bool]) -> None:
    """Wait, without polling, until `check` holds after a change `live` announces."""
    reached = anyio.Event()

    def listener() -> None:
        if check():
            reached.set()

    unsubscribe = live.subscribe(listener)
    try:
        listener()
        await reached.wait()
    finally:
        unsubscribe()


def blocked(
    name: str, started: dict[str, anyio.Event], release: dict[str, anyio.Event], calls: dict[str, int]
) -> Agent[None, str]:
    """An agent whose turns wait for `release[name]`, counting them in `calls`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        calls[name] += 1
        started[name].set()
        await release[name].wait()
        yield f'{name} answered'

    return Agent(FunctionModel(stream_function=stream))


async def test_sessions_stream_at_once_and_an_interrupt_stops_only_its_own(tmp_path: Path) -> None:
    """Two turns overlap; input sent meanwhile is queued; interrupting A leaves B running to completion."""
    fleet = Fleet(tmp_path)
    started = {name: anyio.Event() for name in 'ab'}
    release = {name: anyio.Event() for name in 'ab'}
    calls = {'a': 0, 'b': 0}
    fleet.agents = {name: blocked(name, started, release, calls) for name in 'ab'}
    launched_in = Path.cwd()
    with anyio.fail_after(20):
        async with fleet.live.serving():
            a = await fleet.live.new(workspace=fleet.folder('a'))
            b = await fleet.live.new(workspace=fleet.folder('b'))
            assert [session.id for session in fleet.live.listing()] == [a.id, b.id]
            assert fleet.live.get(b.id) is b
            assert (a.workspace, b.workspace) == ((tmp_path / 'a').resolve(), (tmp_path / 'b').resolve())
            a.send_input('go')
            b.send_input('go')
            await started['a'].wait()
            await started['b'].wait()
            assert a.running and b.running, 'both turns run at once'
            b.send_input('and then')
            assert (a.queued, b.queued) == (0, 1)
            assert a.interrupt()
            await until(fleet.live, lambda: not a.running)
            assert b.running and b.queued == 1
            release['b'].set()
            second = fleet.runtimes[b.id]
            await until(fleet.live, lambda: calls['b'] == 2 and second.idle)
    assert 'Turn cancelled' in fleet.outputs['a'].getvalue()
    assert 'a answered' not in fleet.outputs['a'].getvalue()
    assert fleet.outputs['b'].getvalue().count('b answered') == 2
    assert not fleet.live.listing()
    # Leaving `serving` closed both, as idle closing does.
    assert sorted(Recorder.ended) == [('a', 'exit'), ('b', 'exit')]
    assert Path.cwd() == launched_in


async def test_a_conversation_is_live_in_one_session(tmp_path: Path) -> None:
    """Opening a conversation twice, at once too, returns one session; another session cannot resume it."""
    fleet = Fleet(tmp_path)
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi')]),
        ModelResponse(parts=[TextPart('hello')]),
    ]
    saved = await fleet.process.conversations.save(
        summary=ConversationSummary(workspace=str(fleet.folder('a').resolve()), title='Saved'), messages=history
    )
    with anyio.fail_after(20):
        async with fleet.live.serving():
            opened: list[LiveSession] = []

            async def open_it(conversation_id: str = saved.id) -> None:
                opened.append(await fleet.live.open(conversation_id))

            async with anyio.create_task_group() as group:
                group.start_soon(open_it)
                group.start_soon(open_it)
            first = opened[0]
            assert opened == [first, first] and fleet.built == ['a']
            assert await fleet.live.open(saved.id) is first
            assert fleet.live.holding(saved.id) is first
            assert first.conversation.conversation_id == saved.id
            assert first.workspace == (tmp_path / 'a').resolve()
            assert fleet.runtimes[first.id].session.messages == history

            other = await fleet.live.new()
            with pytest.raises(ValueError, match='is open in another live session'):
                await fleet.runtimes[other.id].sessions.resume(saved.id)
            # Plugins resume through `host.conversation`, which refuses it too.
            with pytest.raises(ValueError, match='is open in another live session'):
                await fleet.runtimes[other.id].session.resume(saved.id)
            # Nor may another session delete or archive it.
            with pytest.raises(ValueError, match='is open in another live session'):
                await fleet.runtimes[other.id].sessions.delete(saved.id)
            with pytest.raises(ValueError, match='is open in another live session'):
                await fleet.runtimes[other.id].sessions.archive(saved.id)
            # The session holding it may restore it again.
            assert 'Resumed Saved' in await fleet.runtimes[first.id].sessions.resume(saved.id)
            # While one session resumes a conversation, `open` waits for it, and no other session may take it.
            later = await fleet.process.conversations.save(
                summary=ConversationSummary(workspace=str(fleet.folder('home').resolve()), title='Later'),
                messages=history,
            )
            async with anyio.create_task_group() as group:
                with fleet.live.claim(later.id, holder=other):
                    with pytest.raises(ValueError, match='is open in another live session'):
                        await fleet.runtimes[first.id].sessions.resume(later.id)
                    group.start_soon(open_it, later.id)
                    await anyio.lowlevel.checkpoint()
                    assert len(opened) == 2
                    await fleet.runtimes[other.id].session.resume(later.id)
            assert opened[2] is other and fleet.built == ['a', 'home']
            # Commands that restart CLAI belong to the session CLAI started with.
            other.send_input('/reload')
            other.send_input('/update')
            await until(fleet.live, lambda: fleet.outputs['home'].getvalue().count('restart CLAI.') == 2)
            assert other in fleet.live.listing()
            # `/exit` closes only the session it is entered in.
            other.send_input('/exit')
            await until(fleet.live, lambda: other not in fleet.live.listing())
            assert Recorder.ended == [('home', 'exit')]

            # Opening it while it closes waits for the close, then resumes it in a new session.
            async with anyio.create_task_group() as group:
                group.start_soon(fleet.live.close, first.id)
                await anyio.lowlevel.checkpoint()
                # Closing, it still holds the conversation: nothing else may resume it yet.
                assert fleet.live.holding(saved.id) is first
                opened.append(await fleet.live.open(saved.id))
                reopened = opened[-1]
                assert reopened is not first and fleet.built == ['a', 'home', 'a']
                with pytest.raises(LookupError):
                    fleet.live.get(first.id)
                assert fleet.runtimes[reopened.id].session.messages == history

            # While a session switches away from a conversation, `open` waits, then opens it anew.
            reopened = opened[-1]
            third = await fleet.process.conversations.save(
                summary=ConversationSummary(workspace=str(fleet.folder('a').resolve()), title='Third'),
                messages=history,
            )
            async with anyio.create_task_group() as group:
                with fleet.live.claim(third.id, holder=reopened):
                    with pytest.raises(ValueError, match='already switching'):
                        fleet.live.claim('x', holder=reopened).__enter__()
                    group.start_soon(open_it, saved.id)
                    await anyio.wait_all_tasks_blocked()
                    assert len(opened) == 4
                    await fleet.runtimes[reopened.id].session.resume(third.id)
            assert opened[4] is not reopened and opened[4].conversation.conversation_id == saved.id
            assert fleet.live.holding(third.id) is reopened
    assert Recorder.ended.count(('a', 'exit')) == 3


async def test_opening_and_closing_refuse_what_they_cannot_do(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    fleet = Fleet(tmp_path)
    gone = await fleet.process.conversations.save(
        summary=ConversationSummary(workspace=str(tmp_path / 'gone'), title='Gone'), messages=[]
    )
    held = fleet.build(fleet.folder('held'))
    with pytest.raises(RuntimeError, match='opens no other live sessions'):
        await fleet.live.new()
    async with fleet.live.serving():
        with pytest.raises(LookupError):
            await fleet.live.open('unknown')
        with pytest.raises(ValueError, match='does not exist'):
            await fleet.live.open(gone.id)
        with pytest.raises(ValueError, match='does not exist'):
            await fleet.live.new(workspace=tmp_path / 'missing')
        with pytest.raises(LookupError):
            await fleet.live.close('unknown')
        with pytest.raises(ValueError, match='runs under its own driver'):
            await fleet.live.close(held.live_session.id)
    fleet.live.build = None
    async with fleet.live.serving():
        with pytest.raises(RuntimeError, match='opens no other live sessions'):
            await fleet.live.new()
    assert fleet.built == ['held']

    joined = JoinedSessions(lambda: [held.live_session])
    assert joined.listing() == [held.live_session] and joined.current() is None
    for refused in (joined.open('x'), joined.new(), joined.close('x')):
        with pytest.raises(RuntimeError, match='opens no other live sessions'):
            await refused
    heard: list[str] = []

    def failing() -> None:
        raise RuntimeError('listener broke')

    joined.subscribe(failing)
    unsubscribe = joined.subscribe(lambda: heard.append('changed'))
    joined.changed(held.live_session)
    unsubscribe()
    unsubscribe()
    joined.changed(held.live_session)
    # A failing listener is logged; the others still hear the change.
    assert heard == ['changed']
    assert [record.message for record in caplog.records] == ['A live sessions listener failed'] * 2


async def test_a_session_that_fails_leaves_the_others_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure while opening reaches the caller; one after is logged, and the session is gone."""
    fleet = Fleet(tmp_path)
    build = fleet.build

    def failing(directory: Path | None) -> SessionRuntime[None, str]:
        runtime = build(directory)
        assert directory is not None

        async def loading(*args: object, **kwargs: object) -> None:
            # A session still loading is not listed, though plugins handling its moments can reach it.
            assert fleet.live.get(runtime.live_session.id) is runtime.live_session
            assert runtime.live_session not in fleet.live.listing()
            raise RuntimeError('early broke')

        async def running(*args: object, **kwargs: object) -> None:
            raise RuntimeError('late broke')

        if directory.name == 'early':
            monkeypatch.setattr(runtime.loader, 'load_all', loading)
        else:
            monkeypatch.setattr(runtime, 'run', running)
        return runtime

    fleet.live.build = failing
    with anyio.fail_after(20):
        async with fleet.live.serving():
            with pytest.raises(RuntimeError, match='early broke'):
                await fleet.live.new(workspace=fleet.folder('early'))
            await fleet.live.new(workspace=fleet.folder('late'))
            await until(fleet.live, lambda: not fleet.live.listing())
            fleet.live.build = build
            survivor = await fleet.live.new(workspace=fleet.folder('survivor'))
            survivor.send_input('hello')
            await until(fleet.live, lambda: bool(fleet.runtimes[survivor.id].session.messages))
    assert 'late broke' in caplog.text
    assert ('late', 'error') in Recorder.ended and ('survivor', 'exit') in Recorder.ended


async def test_each_session_works_in_its_own_folder(tmp_path: Path) -> None:
    """Each session's plugins read its own folder's `AGENTS.md`; CLAI's working directory never changes."""
    fleet = Fleet(
        tmp_path, builtin=(PluginSettings(id='repo_context', factory='pydantic_clai2.builtin_plugins.repo_context'),)
    )
    seen: dict[str, str] = {}

    def reader(name: str) -> Agent[None, str]:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            seen[name] = info.instructions or ''
            yield 'read'

        return Agent(FunctionModel(stream_function=stream))

    for name in ('alpha', 'beta'):
        (fleet.folder(name) / 'AGENTS.md').write_text(f'Rules of {name}.')
        fleet.agents[name] = reader(name)
    launched_in = Path.cwd()
    with anyio.fail_after(20):
        async with fleet.live.serving():
            sessions = [await fleet.live.new(workspace=fleet.folder(name)) for name in ('alpha', 'beta')]
            for session in sessions:
                session.send_input('read the rules')
            await until(fleet.live, lambda: len(seen) == 2 and all(fleet.runtimes[s.id].idle for s in sessions))
            assert Path.cwd() == launched_in
    assert 'Rules of alpha.' in seen['alpha'] and 'Rules of beta.' not in seen['alpha']
    assert 'Rules of beta.' in seen['beta'] and 'Rules of alpha.' not in seen['beta']
    assert Path.cwd() == launched_in


async def test_idle_sessions_close_after_the_timeout(tmp_path: Path) -> None:
    """Only an idle, unwatched, unfocused session closes, its plugins ending with 'exit'; reopening resumes it."""
    fleet = Fleet(tmp_path)
    now = [0.0]
    fleet.live.clock = lambda: now[0]
    fleet.live.idle_after = 10
    started = {'busy': anyio.Event()}
    release = {'busy': anyio.Event()}
    fleet.agents['busy'] = blocked('busy', started, release, {'busy': 0})
    with anyio.fail_after(20):
        async with fleet.live.serving():
            idle, watched, busy, focused = [
                await fleet.live.new(workspace=fleet.folder(name)) for name in ('idle', 'watched', 'busy', 'focused')
            ]
            idle.send_input('hello')
            await until(
                fleet.live, lambda: fleet.runtimes[idle.id].idle and bool(fleet.runtimes[idle.id].session.messages)
            )
            conversation = idle.conversation.conversation_id
            history = fleet.runtimes[idle.id].session.messages
            unwatch = watched.watch()
            assert watched.watchers == 1
            busy.send_input('work')
            await started['busy'].wait()
            fleet.live.focused = focused.id

            now[0] = 11
            await fleet.live.close_idle()
            assert fleet.live.listing() == [watched, busy, focused]
            assert Recorder.ended == [('idle', 'exit')]

            unwatch()
            unwatch()
            assert watched.watchers == 0
            # Watched until now, so idle from now on.
            await fleet.live.close_idle()
            assert watched in fleet.live.listing()
            now[0] = 22
            release['busy'].set()
            await until(fleet.live, lambda: fleet.runtimes[busy.id].idle)
            await fleet.live.close_idle()
            assert fleet.live.listing() == [busy, focused]
            now[0] = 40
            await fleet.live.close_idle()
            assert fleet.live.listing() == [focused]

            reopened = await fleet.live.open(conversation)
            assert fleet.runtimes[reopened.id].session.messages == history
    assert Recorder.ended[:3] == [('idle', 'exit'), ('watched', 'exit'), ('busy', 'exit')]


async def test_a_session_that_ends_while_another_closes_is_skipped(tmp_path: Path) -> None:
    """Idle closing waits on one session's end; another that `/exit`s meanwhile is not closed twice."""
    Gate.ending, Gate.release = anyio.Event(), anyio.Event()
    fleet = Fleet(tmp_path, builtin=(RECORDER, PluginSettings(id='gate', factory=f'{__name__}:Gate')))
    now = [0.0]
    fleet.live.clock = lambda: now[0]
    fleet.live.idle_after = 10
    with anyio.fail_after(20):
        async with fleet.live.serving():
            await fleet.live.new(workspace=fleet.folder('held'))
            other = await fleet.live.new(workspace=fleet.folder('other'))
            now[0] = 11
            async with anyio.create_task_group() as group:
                group.start_soon(fleet.live.close_idle)
                await Gate.ending.wait()
                other.send_input('/exit')
                await until(fleet.live, lambda: other not in fleet.live.listing())
                Gate.release.set()
            assert not fleet.live.listing()
    assert sorted(Recorder.ended) == [('held', 'exit'), ('other', 'exit')]


async def test_a_conversation_another_process_runs_opens_one_copy(tmp_path: Path) -> None:
    """Opening it twice returns the session continuing its one copy."""
    fleet = Fleet(tmp_path, builtin=())
    busy = await fleet.process.conversations.save(
        summary=ConversationSummary(
            workspace=str(fleet.folder('a').resolve()), title='Busy', outcome='running', owner_pid=os.getpid()
        ),
        messages=[],
    )
    with anyio.fail_after(20):
        async with fleet.live.serving():
            first = await fleet.live.open(busy.id)
            assert first.conversation.conversation_id != busy.id
            assert await fleet.live.open(busy.id) is first
            assert len(await fleet.process.conversations.listing()) == 2
            # Once it moves on to another conversation, opening the busy one continues the same copy afresh.
            copy = first.conversation.conversation_id
            first.send_input('/new')
            await until(fleet.live, lambda: first.conversation.conversation_id != copy and not first.running)
            again = await fleet.live.open(busy.id)
            assert again is not first
            assert again.conversation.conversation_id == copy
            assert len(await fleet.process.conversations.listing()) == 2


async def test_a_conversation_another_process_runs_has_one_copy_across_processes(tmp_path: Path) -> None:
    """A CLAI started later, or another `--web`, continues the copy an earlier one made rather than another."""
    store = Fleet(tmp_path, builtin=()).process.conversations
    busy = await store.save(
        summary=ConversationSummary(
            workspace=str((tmp_path / 'a').resolve()), title='Busy', outcome='running', owner_pid=os.getpid()
        ),
        messages=[],
    )
    copies: list[str] = []
    with anyio.fail_after(20):
        for _ in range(2):
            fleet = Fleet(tmp_path, builtin=())
            fleet.folder('a')
            async with fleet.live.serving():
                copies.append((await fleet.live.open(busy.id)).conversation.conversation_id)
    assert copies[0] == copies[1] != busy.id
    summaries = await store.listing()
    assert sorted((entry.title, entry.forked_from) for entry in summaries) == [
        ('Busy', None),
        ('Busy (fork)', busy.id),
    ]


async def test_a_copy_this_process_left_running_opens_without_another_copy(tmp_path: Path) -> None:
    """A turn in the copy cut off without saving its end leaves this process's mark on it; opening continues it."""
    fleet = Fleet(tmp_path, builtin=())
    store = fleet.process.conversations
    workspace = str(fleet.folder('a').resolve())
    busy = await store.save(
        summary=ConversationSummary(workspace=workspace, title='Busy', outcome='running', owner_pid=os.getpid()),
        messages=[],
    )
    copy = await store.save(
        summary=ConversationSummary(
            workspace=workspace, title='Busy (fork)', outcome='running', owner_pid=os.getpid(), forked_from=busy.id
        ),
        messages=[],
    )
    # A copy of the copy is not opened instead: nothing runs the copy.
    await store.save(summary=ConversationSummary(workspace=workspace, forked_from=copy.id), messages=[])
    with anyio.fail_after(20):
        async with fleet.live.serving():
            opened = await fleet.live.open(busy.id)
            assert opened.conversation.conversation_id == copy.id
    assert len(await store.listing()) == 3


async def test_a_session_runs_as_the_one_being_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One session opening another does not make the new one run as the opener."""
    fleet = Fleet(tmp_path, builtin=())
    build = fleet.build
    current: list[LiveSession | None] = []

    def recording(directory: Path | None) -> SessionRuntime[None, str]:
        runtime = build(directory)
        serve = runtime.serve

        async def serving(*, resume: str | None, ready: Callable[[], None]) -> None:
            current.append(fleet.live.current())
            await serve(resume=resume, ready=ready)

        monkeypatch.setattr(runtime, 'serve', serving)
        return runtime

    fleet.live.build = recording
    with anyio.fail_after(20):
        async with fleet.live.serving():
            opener = await fleet.live.new(workspace=fleet.folder('opener'))
            with handling(fleet.runtimes[opener.id].live_session):
                opened = await fleet.live.new(workspace=fleet.folder('opened'))
    assert current == [opener, opened]


async def test_the_registry_counts_each_session_once(tmp_path: Path) -> None:
    """Adding twice keeps one entry; leaving or removing a session no longer counted does nothing."""
    fleet = Fleet(tmp_path, builtin=())
    runtime = fleet.build(fleet.folder('once'))
    heard: list[str] = []
    fleet.live.subscribe(lambda: heard.append('changed'))
    fleet.live.add(runtime)
    assert fleet.live.listing() == [runtime.live_session] and not heard
    await runtime.close('exit')
    assert not fleet.live.listing() and heard == ['changed', 'changed']
    fleet.live.leaving(runtime)
    fleet.live.remove(runtime)
    assert heard == ['changed', 'changed']


async def test_a_closed_session_is_freed(tmp_path: Path) -> None:
    """Closing a live session stops it hearing the process's titles, so nothing keeps it alive (#11307)."""
    fleet = Fleet(tmp_path, builtin=())
    with anyio.fail_after(20):
        async with fleet.live.serving():
            kept = await fleet.live.new(workspace=fleet.folder('kept'))
            closed = await fleet.live.new(workspace=fleet.folder('closed'))
            runtime = weakref.ref(fleet.runtimes.pop(closed.id))
            assert len(fleet.process.namer.listeners) == 2
            await fleet.live.close(closed.id)
            del closed
            # Let the task that served it finish.
            await anyio.lowlevel.checkpoint()
            gc.collect()
            assert runtime() is None
            assert fleet.process.namer.listeners == [fleet.runtimes[kept.id].sessions.named]
            assert fleet.process.tasks.views == [fleet.runtimes[kept.id].tasks]


async def test_the_idle_check_runs_on_its_own(tmp_path: Path) -> None:
    fleet = Fleet(tmp_path)
    fleet.live.idle_after = fleet.live.check_every = 0
    with anyio.fail_after(20):
        async with fleet.live.serving():
            await fleet.live.new(workspace=fleet.folder('brief'))
            await until(fleet.live, lambda: not fleet.live.listing())
    assert Recorder.ended == [('brief', 'exit')]


async def test_settings_other_than_the_model_and_run_reach_every_live_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """`/set` in one session applies `display.*` and the like everywhere, except where a project file sets it."""
    fleet = Fleet(tmp_path, builtin=())
    fleet.projects['pinned'] = ProjectSettings(overrides={'display.thinking': True})
    origin, other, pinned = (fleet.build(fleet.folder(name)) for name in ('origin', 'other', 'pinned'))

    origin.context.set_setting(['display.thinking', 'false'])
    assert [runtime.context.settings.thinking for runtime in (origin, other, pinned)] == [False, False, True]
    origin.context.set_setting(['run.code_mode', 'true'])
    origin.context.set_setting(['model', 'test'])
    assert other.context.settings.code_mode is not True and other.session.model is None
    assert origin.session.model == 'test'

    origin.context.reset_setting('display.thinking')
    assert 'thinking' not in other.context.settings.model_fields_set
    assert other.context.settings.thinking == Settings().thinking

    # A session that fails to follow is logged; the setting stays saved, and the others still follow.
    def broken(key: str, source: Settings) -> None:
        raise RuntimeError('cannot follow')

    monkeypatch.setattr(other.context, 'adopt', broken)
    assert origin.context.set_setting(['display.grep_lines', '7']).startswith('Saved display.grep_lines.')
    assert pinned.context.settings.grep_lines == 7 and 'did not follow display.grep_lines' in caplog.text
    monkeypatch.undo()

    await origin.close('exit')
    assert origin.live_session not in fleet.live.listing()
    other.context.set_setting(['display.thinking', 'false'])
    assert origin.context.settings.thinking == Settings().thinking


async def test_run_web_opens_more_sessions_in_their_own_folders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`clai2 --web` builds the sessions `host.live` opens: in the launch folder, or another with its project file."""
    opened: list[str] = []
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.web.open_browser', opened.append)
    launch, other = tmp_path / 'launch', tmp_path / 'other'
    launch.mkdir()
    other.mkdir()
    monkeypatch.chdir(launch)
    shells: list[SessionRuntime[None, str]] = []
    serve = web_module._serve_web  # pyright: ignore[reportPrivateUsage]

    async def recorded(shell: SessionRuntime[None, str], **kwargs: object) -> None:
        shells.append(shell)
        await serve(shell, **kwargs)  # pyright: ignore[reportArgumentType]

    monkeypatch.setattr(web_module, '_serve_web', recorded)
    output = _Output()
    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(
                lambda: run_web(
                    settings_for=lambda project: Settings(model='test'),
                    store=SettingsStore(tmp_path / 'config.db'),
                    project=ProjectSettings(),
                    console=Console(file=output, force_terminal=False, width=500),
                    token='secret',
                    port=0,
                )
            )
            await output.ready.wait()
            live = shells[0].process.live
            here = await live.new()
            there = await live.new(workspace=other)
            assert (here.workspace, there.workspace) == (launch.resolve(), other.resolve())
            assert [session.id for session in live.listing()] == [shells[0].live_session.id, here.id, there.id]
            there.send_input('hello')
            await until(live, lambda: not there.running and bool(there.conversation.messages))
            found = re.search(r'Web UI running at (http://[^/\s]+)', output.getvalue())
            assert found is not None
            async with httpx.AsyncClient(base_url=found[1], headers={HEADER: 'secret'}) as client:
                assert (await client.get('/api/state')).status_code == 200
            assert Path.cwd() == launch
            group.cancel_scope.cancel()
    assert Path.cwd() == launch

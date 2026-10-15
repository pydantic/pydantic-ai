"""Plugins read the conversation's ID and title, and hear `ConversationChanged` when either changes."""

from collections.abc import Sequence
from decimal import Decimal
from io import StringIO
from pathlib import Path
from typing import ClassVar

import anyio
import pytest
from inline_snapshot import snapshot
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import chat
from pydantic_clai2._app import create_shell, create_stock_agent
from pydantic_clai2.cli import headless
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.commands import Command
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import ConversationChanged, Plugin, PluginHost, SessionStart, Transcript, load_plugin
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.session_naming import NamingResult, SessionName
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.ui.menus.session_browser import SessionBrowser
from tests.conftest import IsStr


class Recorder(Plugin):
    """Records what it sees at load and every identity change, for the shell-level tests below."""

    seen: ClassVar[list[tuple[str, str, str | None]]] = []

    async def on_session_start(self, event: SessionStart) -> None:
        conversation = self.host.conversation
        self.seen.append(('start', conversation.conversation_id, conversation.title))

    async def on_conversation_changed(self, event: ConversationChanged) -> None:
        self.seen.append(('changed', event.conversation_id, event.title))


RECORDER = PluginSettings(id='recorder', factory='tests.clai2.test_conversation_identity:Recorder')


@pytest.fixture(autouse=True)
def fresh_recorder() -> None:
    Recorder.seen.clear()


def recording(session: Session[None, str]) -> list[ConversationChanged]:
    events: list[ConversationChanged] = []

    async def record(event: ConversationChanged) -> None:
        events.append(event)

    session.on_change = record
    return events


def test_transcript_identity() -> None:
    transcript = Transcript()
    assert transcript.conversation_id != Transcript().conversation_id
    assert transcript.title is None
    named = Transcript(conversation_id='abc', title='Named')
    assert (named.conversation_id, named.title) == ('abc', 'Named')


async def test_session_publishes_each_change_once(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)
    events = recording(session)
    first = session.conversation_id
    assert session.title is None

    await session.prompt('hello there')
    await session.prompt('again')
    assert events == [ConversationChanged(conversation_id=first, title='hello there')]

    await session.renamed(conversation_id='another', title='Ignored')
    await session.renamed(conversation_id=first, title='Renamed')
    assert events[-1] == ConversationChanged(conversation_id=first, title='Renamed')
    assert session.title == 'Renamed'

    await session.clear()
    assert session.conversation_id != first
    assert events[-1] == ConversationChanged(conversation_id=session.conversation_id, title=None)

    await session.resume(first)
    assert events[-1] == ConversationChanged(conversation_id=first, title='hello there')
    await session.resume(first)
    assert len(events) == 4


async def test_naming_and_browser_renames_retitle_the_current_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)
    context = CommandContext(
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        apply_setting=lambda key, settings: None,
    )
    service = Sessions(session=session, store=store, context=context)
    await session.prompt('fix the renderer')
    events = recording(session)

    async def generate(prompt: str) -> NamingResult:
        return NamingResult(name=SessionName(title='Renderer fix'))

    service.namer.generate = generate
    assert await service.namer.name(conversation_id=session.conversation_id)
    assert events == [ConversationChanged(conversation_id=session.conversation_id, title='Renderer fix')]
    await service.named('another conversation', 'Ignored')
    assert len(events) == 1

    def rename(browser: SessionBrowser) -> str:
        browser.reload()
        assert browser.selected is not None
        browser.rename(browser.selected, 'Manual name')
        # The event waits for the browser to close.
        assert len(events) == 1
        return ''

    monkeypatch.setattr(SessionBrowser, 'run', rename)
    assert await service.command([]) == ''
    assert events[-1] == ConversationChanged(conversation_id=session.conversation_id, title='Manual name')


async def test_startup_resume_precedes_plugins_and_commands_notify_them(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'settings.db')
    saved = Session(
        Agent(TestModel()), deps=None, conversations=SqliteConversationStore(database=tmp_path / 'sessions.db')
    )
    await saved.prompt('earlier work')
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text(f'/new\n/resume {saved.conversation_id}\n/exit\n')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=StringIO()),
            store=store,
            builtin_plugins=[RECORDER],
            resume=saved.conversation_id,
        )
    saved_id = saved.conversation_id
    assert Recorder.seen == snapshot(
        [
            ('start', saved_id, 'earlier work'),
            ('changed', IsStr(), None),
            ('changed', saved_id, 'earlier work'),
        ]
    )
    assert Recorder.seen[1][1] != saved_id


async def test_startup_browser_opens_once_plugins_have_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    saved = Session(
        Agent(TestModel()), deps=None, conversations=SqliteConversationStore(database=tmp_path / 'sessions.db')
    )
    await saved.prompt('earlier work')

    def pick(browser: SessionBrowser) -> str:
        # Plugins have started, so models they offer can name the sessions listed here.
        assert [event for event, *_ in Recorder.seen] == ['start']
        return saved.conversation_id

    monkeypatch.setattr(SessionBrowser, 'run', pick)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/exit\n')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=StringIO()),
            store=SettingsStore(tmp_path / 'settings.db'),
            settings=Settings(model=None, session_namer=False),
            builtin_plugins=[RECORDER],
            resume='',
        )
    assert Recorder.seen[1:] == [('changed', saved.conversation_id, 'earlier work')]


async def test_background_names_wait_for_the_terminal(tmp_path: Path) -> None:
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'settings.db'),
        builtin_plugins=[],
        project=ProjectSettings(),
        headless=True,
    )
    session = shell.session
    events = recording(session)
    await session.prompt('fix the renderer')
    events.clear()
    async with anyio.create_task_group() as tasks:
        async with shell.forks.busy():
            tasks.start_soon(shell.sessions.named, session.conversation_id, 'Renderer fix')
            await anyio.wait_all_tasks_blocked()
            assert events == []
        await anyio.wait_all_tasks_blocked()
        assert events == [ConversationChanged(conversation_id=session.conversation_id, title='Renderer fix')]
        # A manual rename made while a generated name waits for the terminal wins.
        async with shell.forks.busy():
            tasks.start_soon(shell.sessions.named, session.conversation_id, 'Generated')
            await anyio.wait_all_tasks_blocked()
            await session.renamed(conversation_id=session.conversation_id, title='Mine')
        await anyio.wait_all_tasks_blocked()
    assert session.title == 'Mine'
    assert events[-1] == ConversationChanged(conversation_id=session.conversation_id, title='Mine')


async def test_a_name_a_save_already_adopted_still_reaches_plugins(tmp_path: Path) -> None:
    """A turn's save copies a name naming stored meanwhile; the waiting notice must still be sent."""
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'settings.db'),
        builtin_plugins=[],
        project=ProjectSettings(),
        headless=True,
    )
    session = shell.session
    store = session.conversations
    assert store is not None
    events = recording(session)
    await session.prompt('fix the renderer')
    events.clear()
    async with anyio.create_task_group() as tasks:
        async with shell.forks.busy():
            saved = (await store.get(conversation_id=session.conversation_id)).summary
            assert await store.name(source=saved, title='Renderer fix')
            tasks.start_soon(shell.sessions.named, session.conversation_id, 'Renderer fix')
            await anyio.wait_all_tasks_blocked()
            # The running turn's save adopts the stored name, and tells no plugin.
            await session.commit_messages(session.messages)
            assert session.title == 'Renderer fix'
            assert events == []
        await anyio.wait_all_tasks_blocked()
    assert events == [ConversationChanged(conversation_id=session.conversation_id, title='Renderer fix')]


async def test_headless_resume_precedes_plugins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    saved = Session(
        Agent(TestModel()), deps=None, conversations=SqliteConversationStore(database=tmp_path / 'sessions.db')
    )
    await saved.prompt('earlier work')
    monkeypatch.setattr(headless, 'create_agent', lambda: create_stock_agent(TestModel(call_tools=[])))
    store = SettingsStore(tmp_path / 'config.db')
    store.save_plugin(RECORDER)
    assert (
        await headless.run_headless(
            text='continue',
            settings=Settings(model=None),
            store=store,
            project=ProjectSettings(),
            resume=saved.conversation_id,
        )
        == 0
    )
    assert Recorder.seen == [('start', saved.conversation_id, 'earlier work')]


class Resumer(Plugin):
    """Restores a known conversation as soon as it loads."""

    target: ClassVar[str] = ''

    async def on_session_start(self, event: SessionStart) -> None:
        Recorder.seen.append(('resumed', self.target, await self.host.conversation.resume(self.target)))


async def test_plugin_resumes_a_saved_conversation(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    saved = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)
    await saved.prompt('earlier work')
    session = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)
    events = recording(session)
    host = PluginHost(name='resumer', console=Console(file=StringIO()), settings={}, conversation=session)
    Resumer.target = saved.conversation_id
    loaded = load_plugin(Resumer, host)
    await loaded.dispatch(SessionStart(agent=session.agent, settings=Settings()))
    assert Recorder.seen == [('resumed', saved.conversation_id, f'Resumed earlier work ({saved.conversation_id}).')]
    assert session.messages == saved.messages
    assert events == [ConversationChanged(conversation_id=saved.conversation_id, title='earlier work')]


async def test_transcript_has_nothing_to_resume() -> None:
    with pytest.raises(LookupError, match='No saved session: missing'):
        await Transcript().resume('missing')


class Switcher(Plugin):
    """Offers `/switch ID`, a plugin command that resumes another conversation."""

    def get_commands(self) -> Sequence[Command]:
        async def switch(args: list[str]) -> str:
            return await self.host.conversation.resume(args[0])

        return (Command(name='switch', description='Resume a conversation', handler=switch),)


async def test_a_plugin_command_switching_conversations_clears_the_footer(tmp_path: Path) -> None:
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=StringIO()),
        settings=None,
        store=SettingsStore(tmp_path / 'settings.db'),
        builtin_plugins=[PluginSettings(id='switcher', factory='tests.clai2.test_conversation_identity:Switcher')],
        project=ProjectSettings(),
        headless=True,
    )
    await shell.loader.load_all()
    try:
        session = shell.session
        await session.prompt('earlier work')
        saved = session.conversation_id
        await shell.commands.execute_async('/new')
        shell.status.context_tokens = 90
        shell.status.cost = Decimal('0.01')
        # Titling the current conversation keeps its figures.
        await session.prompt('current work')
        assert shell.status.context_tokens == 90
        assert 'Resumed earlier work' in await shell.commands.execute_async(f'/switch {saved}')
        assert session.conversation_id == saved
        assert (shell.status.context_tokens, shell.status.cost) == (None, None)
    finally:
        await shell.loader.close('exit')


@pytest.mark.parametrize('failure', ['error', 'cancel'])
async def test_a_failed_title_publication_still_finalizes_the_turn(tmp_path: Path, failure: str) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)

    async def fail(event: ConversationChanged) -> None:
        if failure == 'error':
            raise RuntimeError('observer failed')
        raise anyio.get_cancelled_exc_class()()

    session.on_change = fail
    with pytest.raises((RuntimeError, anyio.get_cancelled_exc_class())):
        await session.prompt('first')
    # Not left `running` under this process, which a later resume would take for a busy session.
    saved = await store.get(conversation_id=session.conversation_id)
    assert (saved.summary.outcome, saved.summary.owner_pid) == ('failed' if failure == 'error' else 'cancelled', None)

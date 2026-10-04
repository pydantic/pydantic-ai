"""Plugins read the conversation's ID and title, and hear `ConversationChanged` when either changes."""

from io import StringIO
from pathlib import Path
from typing import ClassVar

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import chat
from pydantic_clai2._app import create_stock_agent
from pydantic_clai2.cli import headless
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import ConversationChanged, Plugin, SessionStart, Transcript
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.session_naming import NamingResult, SessionName
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.ui.menus.session_browser import SessionBrowser


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
    assert Recorder.seen[0] == ('start', saved.conversation_id, 'earlier work')
    assert Recorder.seen[1][0] == 'changed' and Recorder.seen[1][1] != saved.conversation_id
    assert Recorder.seen[1][2] is None
    assert Recorder.seen[2:] == [('changed', saved.conversation_id, 'earlier work')]


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

"""`--session-id` and `--fork-session` name a new conversation or continue in a copy, as Claude Code's do."""

from io import StringIO
from pathlib import Path
from uuid import UUID

import anyio
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
from pydantic_clai2.cli import _cli, headless
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.ui.menus.session_browser import SessionBrowser
from tests.clai2.test_conversation_identity import RECORDER, Recorder

NEW_ID = '0b4f5d8e-3c1a-4e6b-9f2d-7a8c9b0d1e2f'
OTHER_ID = '6f1c2b3a-4d5e-4f60-8a7b-9c0d1e2f3a4b'


@pytest.fixture(autouse=True)
def fresh_recorder() -> None:
    Recorder.seen.clear()


def session_in(tmp_path: Path) -> Session[None, str]:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    return Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)


async def test_clear_names_the_new_conversation(tmp_path: Path) -> None:
    session = session_in(tmp_path)
    await session.prompt('first')
    await session.clear(NEW_ID)
    assert session.conversation_id == NEW_ID
    await session.prompt('second')
    assert session.conversations is not None
    assert (await session.conversations.get(conversation_id=NEW_ID)).summary.title == 'second'
    with pytest.raises(ValueError, match=f'A saved session already uses the ID {NEW_ID}.'):
        await session.clear(NEW_ID)
    assert session.conversation_id == NEW_ID

    unsaved = Session(Agent(TestModel()), deps=None)
    await unsaved.clear(NEW_ID)
    assert unsaved.conversation_id == NEW_ID


async def test_fork_copies_and_leaves_the_original(tmp_path: Path) -> None:
    session = session_in(tmp_path)
    store = session.conversations
    assert store is not None
    await session.prompt('original work')
    original = session.summary
    notice = await session.fork(NEW_ID)
    assert notice == f'Forked original work ({original.id}) into {NEW_ID}.'
    assert session.conversation_id == NEW_ID
    assert session.title == 'original work'
    await session.prompt('only in the copy')
    copy = await store.get(conversation_id=NEW_ID)
    source = await store.get(conversation_id=original.id)
    assert copy.messages[: len(source.messages)] == source.messages
    assert len(copy.messages) > len(source.messages)
    assert source.summary.revision == original.revision

    assert UUID(await session.fork() and session.conversation_id)
    with pytest.raises(ValueError, match='already uses the ID'):
        await session.fork(original.id)
    with pytest.raises(ValueError, match='not configured'):
        await Session(Agent(TestModel()), deps=None).fork()


async def test_fork_refuses_a_running_conversation(tmp_path: Path) -> None:
    started = anyio.Event()
    release = anyio.Event()
    agent = Agent(TestModel())

    @agent.tool_plain
    async def wait() -> str:
        started.set()
        await release.wait()
        return 'ok'

    session = Session(
        agent,
        deps=None,
        conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
        workspace=tmp_path,
    )
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(session.prompt, 'go')
        await started.wait()
        with pytest.raises(RuntimeError, match='Cannot fork a running conversation'):
            await session.fork()
        release.set()


async def test_launch_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = session_in(tmp_path)
    store = session.conversations
    assert store is not None
    context = CommandContext(
        settings=Settings(model=None, session_namer=False),
        store=SettingsStore(tmp_path / 'config.db'),
        apply_setting=lambda key, settings: None,
    )
    service = Sessions(session=session, store=store, context=context)
    await session.prompt('saved work')
    saved = session.conversation_id
    await session.clear()

    assert (
        await service.start(resume=saved, session_id=NEW_ID, fork=True) == f'Forked saved work ({saved}) into {NEW_ID}.'
    )
    assert session.conversation_id == NEW_ID

    def cancel(browser: SessionBrowser) -> str:
        return ''

    monkeypatch.setattr(SessionBrowser, 'run', cancel)
    assert await service.start(resume='', session_id=OTHER_ID, fork=True) == ''
    assert (session.conversation_id, session.messages) == (OTHER_ID, [])
    assert 'Resumed' in await service.start(resume=saved)
    assert session.conversation_id == saved


async def test_chat_applies_launch_options_before_plugins(tmp_path: Path) -> None:
    saved = Session(
        Agent(TestModel()), deps=None, conversations=SqliteConversationStore(database=tmp_path / 'sessions.db')
    )
    await saved.prompt('earlier work')
    for resume, session_id in ((None, NEW_ID), (saved.conversation_id, None)):
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text('/exit\n')
            await chat(
                Agent(TestModel()),
                deps=None,
                console=Console(file=StringIO()),
                store=SettingsStore(tmp_path / 'settings.db'),
                builtin_plugins=[RECORDER],
                resume=resume,
                session_id=session_id,
                fork_session=resume is not None,
            )
    [(_, new, untitled), (_, fork, title)] = Recorder.seen
    assert (new, untitled) == (NEW_ID, None)
    assert fork not in (NEW_ID, saved.conversation_id) and title == 'earlier work'


async def test_headless_fork(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    saved = Session(
        Agent(TestModel()), deps=None, conversations=SqliteConversationStore(database=tmp_path / 'sessions.db')
    )
    await saved.prompt('earlier work')
    monkeypatch.setattr(headless, 'create_agent', lambda: create_stock_agent(TestModel(call_tools=[])))
    result = await headless.run_headless(
        text='continue',
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        project=ProjectSettings(),
        resume=saved.conversation_id,
        session_id=NEW_ID,
        fork_session=True,
    )
    assert result == 0
    assert saved.conversations is not None
    copy = await saved.conversations.get(conversation_id=NEW_ID)
    original = await saved.conversations.get(conversation_id=saved.conversation_id)
    assert len(copy.messages) > len(original.messages) == len(saved.messages)


@pytest.mark.parametrize(
    'args',
    [
        ['--session-id', 'not-a-uuid'],
        ['--fork-session'],
        ['--resume', 'x', '--session-id', NEW_ID],
        ['--session-id', NEW_ID, 'config'],
        ['--fork-session', 'plugins'],
    ],
)
def test_invalid_flag_combinations(monkeypatch: pytest.MonkeyPatch, args: list[str]) -> None:
    monkeypatch.setattr('sys.argv', ['clai2', *args])
    with pytest.raises(SystemExit) as error:
        _cli.run()
    assert error.value.code == 2


@pytest.mark.parametrize('prompt', [False, True])
def test_flags_reach_the_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prompt: bool) -> None:
    argv = ['--database', str(tmp_path / 'config.db'), '--resume', 'x', '--fork-session']
    monkeypatch.setattr('sys.argv', ['clai2', *argv, '--session-id', NEW_ID.upper(), *(['-p', 'hi'] if prompt else [])])
    seen: list[tuple[str | None, bool]] = []

    async def launch(*, session_id: str | None, fork_session: bool, **_: object) -> int:
        seen.append((session_id, fork_session))
        return 0

    async def chat(*args: object, session_id: str | None, fork_session: bool, **_: object) -> None:
        await launch(session_id=session_id, fork_session=fork_session)

    monkeypatch.setattr(headless, 'run_headless', launch)
    monkeypatch.setattr('pydantic_clai2._app.chat', chat)
    if prompt:
        with pytest.raises(SystemExit):
            _cli.run()
    else:
        _cli.run()
    assert seen == [(NEW_ID, True)]

"""The command, startup flag, naming resolver and terminal worker share one service."""

import os
import subprocess
import sys
from builtins import BaseExceptionGroup
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from io import StringIO
from pathlib import Path

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from termflow.tui.keys import Key

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent, ToolEffectRecord
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2 import DEFAULT_PLUGINS, chat
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.runtime._session import Session
from pydantic_clai2.runtime.session_naming import SessionNamer
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.ui.menus.session_browser import SessionBrowser


async def test_reload_keeps_saved_conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def rebuild(factory: Callable[[], object]) -> object:
        return factory()

    monkeypatch.setattr('pydantic_clai2._app.reload_clai', rebuild)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('first\r/reload\rsecond\r/exit\r')
        await chat(
            Agent(TestModel(call_tools=[], custom_output_text='answer')),
            deps=None,
            console=Console(file=StringIO()),
            store=SettingsStore(tmp_path / 'settings.db'),
        )
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    summaries = await store.listing()
    assert len(summaries) == 1
    saved = await store.get(conversation_id=summaries[0].id)
    assert [
        part.content
        for message in saved.messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ] == ['first', 'second']


def cancel_browser(browser: SessionBrowser) -> str:
    return ''


async def test_resume_command_uses_browser_and_naming_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)
    context = CommandContext(
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=session.clear,
        apply_setting=lambda key, settings: None,
    )
    service = Sessions(session=session, store=store, context=context)
    await session.prompt('name this session')
    saved_id = session.summary.id
    session.clear()

    def select(browser: SessionBrowser) -> str:
        browser.reload()
        assert browser.selected is not None
        assert 'name this session' in browser.preview(browser.selected.id)
        browser.rename(browser.selected, 'Manual name')
        return browser.selected.id

    monkeypatch.setattr(SessionBrowser, 'run', select)
    assert 'Manual name' in await service.command([])
    assert session.summary.id == saved_id
    with pytest.raises(ValueError, match='Usage:'):
        await service.command(['one', 'two'])
    assert 'Resumed' in await service.command([saved_id])
    monkeypatch.setattr(SessionBrowser, 'run', cancel_browser)
    assert await service.command([]) == ''
    assert 'Background naming:' in await service.usage(console=Console(file=StringIO()))
    result = await service.generate('Fix tests')
    assert result is not None
    context.set_setting(['sessions.naming_model', 'test'])
    assert await service.generate('Fix tests') is not None

    async def resolve(name: str) -> TestModel:
        return TestModel()

    session.resolve_model = resolve
    assert await service.generate('Fix tests') is not None
    context.set_setting(['sessions.naming', 'false'])
    assert not service.namer.submit(saved_id)
    context.set_setting(['sessions.naming_model', 'null'])
    assert context.settings.session_namer_model is None


async def test_browser_opened_mid_turn_defers_the_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    prior = Session(Agent(TestModel()), deps=None, conversations=store, workspace=tmp_path)
    await prior.prompt('saved turn')
    started, release = anyio.Event(), anyio.Event()

    async def wait(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        started.set()
        await release.wait()
        yield 'done'

    session = Session(Agent(FunctionModel(stream_function=wait)), deps=None, conversations=store, workspace=tmp_path)
    context = CommandContext(
        settings=Settings(model=None, session_namer=False),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=session.clear,
        apply_setting=lambda key, settings: None,
    )
    service = Sessions(session=session, store=store, context=context)

    def select(browser: SessionBrowser) -> str:
        return prior.summary.id

    monkeypatch.setattr(SessionBrowser, 'run', select)
    async with anyio.create_task_group() as running:
        running.start_soon(session.prompt, 'busy')
        await started.wait()
        assert await service.command([]) == (
            f'A turn is running. Enter /resume {prior.summary.id} to restore that session once it ends.'
        )
        release.set()
    assert session.summary.id != prior.summary.id
    assert 'Resumed' in await service.command([])
    assert session.summary.id == prior.summary.id


async def test_browser_resumes_other_directory_without_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    previous_workspace = tmp_path / 'previous'
    previous_workspace.mkdir()
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    agent = Agent(TestModel(call_tools=[], custom_output_text='saved answer'))
    prior = Session(agent, deps=None, conversations=store, workspace=previous_workspace)
    await prior.prompt('saved turn')
    session = Session(agent, deps=None, conversations=store, workspace=tmp_path)
    context = CommandContext(
        settings=Settings(model=None, session_namer=False),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=session.clear,
        apply_setting=lambda key, settings: None,
    )
    service = Sessions(session=session, store=store, context=context)

    def select(browser: SessionBrowser) -> str:
        keys = iter([Key.ENTER, '', Key.ENTER, '', 'ctrl-c'])
        browser.key_source = lambda: next(keys)
        browser.output = StringIO()
        return browser.loop()

    monkeypatch.setattr(SessionBrowser, 'run', select)
    assert 'Resumed' in await service.command([])
    assert session.summary.id == prior.summary.id
    assert session.messages == prior.messages
    assert session.workspace == str(tmp_path)
    assert Path.cwd() == tmp_path
    assert (await store.get(conversation_id=prior.summary.id)).summary.workspace == str(previous_workspace)


async def test_startup_restore_and_new_session_are_persisted(tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    prior = Session(Agent(TestModel()), deps=None, conversations=store)
    await prior.prompt('earlier turn')
    output = StringIO()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('followup\r/new\rother session\r/exit\r')
        await chat(
            Agent(TestModel(call_tools=[], custom_output_text='answer')),
            deps=None,
            console=Console(file=output),
            store=SettingsStore(tmp_path / 'settings.db'),
            settings=Settings(model=None, session_namer=False),
            builtin_plugins=DEFAULT_PLUGINS,
            resume=prior.summary.id,
        )
    entries = await store.listing()
    assert len(entries) == 2
    assert (await store.get(conversation_id=prior.summary.id)).summary.message_count > prior.summary.message_count
    assert 'Resumed' in output.getvalue()
    # `/new` is `/clear`, so the banner follows the resumed conversation.
    assert '/resume restores one' in output.getvalue().rsplit('Resumed', 1)[1]


async def test_empty_startup_browser_and_invalid_restore(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SessionBrowser, 'run', cancel_browser)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text('/exit\r')
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=StringIO()),
            store=SettingsStore(tmp_path / 'settings.db'),
            resume='',
        )
    with pytest.raises(LookupError, match='No saved session'):
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=StringIO()),
            store=SettingsStore(tmp_path / 'settings.db'),
            resume='missing',
        )


@pytest.mark.parametrize('args', [['--resume', 'missing'], ['--resume=missing', 'config']])
def test_resume_cli_errors_are_normal_parser_errors(tmp_path: Path, args: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, '-m', 'pydantic_clai2', '--database', str(tmp_path / 'settings.db'), *args],
        input='/exit\n',
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
        env=dict(os.environ, CLAI_NO_SPLASH='1'),
    )
    assert result.returncode == 2
    assert 'error:' in result.stderr
    assert 'Traceback' not in result.stderr


async def test_empty_usage_missing_model_and_recovery_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(Agent(), deps=None, conversations=store, workspace=tmp_path)
    context = CommandContext(
        settings=Settings(model=None),
        store=SettingsStore(tmp_path / 'config.db'),
        clear_history=session.clear,
        apply_setting=lambda key, settings: None,
    )
    service = Sessions(session=session, store=store, context=context)
    assert await service.generate('no model') is None
    assert 'Background naming:' not in await service.usage(console=Console(file=StringIO()))
    saved = await store.save(summary=replace(session.summary, outcome='running', run_id='interrupted'), messages=[])
    steps = session.step_store
    assert steps is not None
    await steps.register_run(RunRecord(run_id='interrupted', conversation_id=saved.id))
    await steps.save_snapshot(
        ContinuableSnapshot(
            run_id='interrupted', step_index=0, messages=[ModelRequest(parts=[UserPromptPart('checkpoint text')])]
        )
    )
    await steps.append_event(
        StepEvent(run_id='interrupted', kind='tool_call_completed', step_index=0, tool_name='done')
    )
    await steps.append_event(StepEvent(run_id='interrupted', kind='tool_call_failed', step_index=0, tool_name='failed'))
    await steps.record_tool_effect(
        ToolEffectRecord(run_id='interrupted', tool_call_id='t1', tool_name='unknown', status='started')
    )

    no_snapshot = await store.save(summary=replace(saved, id='no-snapshot', revision=0, run_id='absent'), messages=[])

    def inspect(browser: SessionBrowser) -> str:
        browser.selected_id = saved.id
        assert browser.selected is not None
        text = browser.preview(saved.id)
        assert 'checkpoint text' in text and 'unknown (t1)' in text and 'done' in text and 'failed' in text
        assert 'checkpoint text' not in browser.preview(no_snapshot.id)
        browser.rename(browser.selected, 'first')
        with pytest.raises(ValueError, match='Session changed'):
            browser.rename(browser.selected, 'stale')
        return ''

    monkeypatch.setattr(SessionBrowser, 'run', inspect)
    await service.command([])


async def test_simultaneous_service_failures_remain_grouped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:

    entered = anyio.Event()

    async def broken_worker(self: SessionNamer) -> None:
        entered.set()
        try:
            await anyio.sleep_forever()
        finally:
            raise ValueError('worker cleanup failed')

    async def broken_command(self: Sessions[None, str], args: list[str]) -> str:
        await entered.wait()
        raise ValueError('browser failed')

    monkeypatch.setattr(SessionNamer, 'run', broken_worker)
    monkeypatch.setattr(Sessions, 'command', broken_command)
    with pytest.raises(BaseExceptionGroup) as caught:
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=StringIO()),
            store=SettingsStore(tmp_path / 'settings.db'),
            resume='',
        )
    assert len(caught.value.exceptions) == 2

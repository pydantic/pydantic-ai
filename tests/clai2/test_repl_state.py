"""Code mode keeps each session's `run_code` REPL between prompts, through resume, forks, and deletion."""

import pytest

pytest.importorskip('pydantic_monty')

import io
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

from rich.console import Console

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.step_persistence.conversations import SqliteConversationStore
from pydantic_clai2.cli import headless
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import TurnStart
from pydantic_clai2.runtime import repl_state
from pydantic_clai2.runtime.monty_cells import NOTE_CHARS, MontyCell
from pydantic_clai2.runtime.repl_state import ReplStates
from pydantic_clai2.runtime.session import Session
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.runtime.speculative_mode import guidance
from pydantic_clai2.ui.menus.session_browser import SessionBrowser

SESSION_ID = '0b4f5d8e-3c1a-4e6b-9f2d-7a8c9b0d1e2f'
FORK_ID = '6f1c2b3a-4d5e-4f60-8a7b-9c0d1e2f3a4b'


class Script:
    """Answers each prompt with its `run_code` snippet, then records what the snippet returned."""

    CODE = {'set': 'x = 41', 'get': 'x + 1', 'bump': 'x = x + 100', 'restart': 'y = 1'}

    def __init__(self) -> None:
        self.results: list[str] = []
        self.instructions: list[str] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        part = request.parts[-1]
        if isinstance(part, UserPromptPart):
            assert isinstance(part.content, str)
            self.instructions.append(info.instructions or '')
            args: dict[str, object] = {'code': self.CODE[part.content], 'restart': part.content == 'restart'}
            return ModelResponse(parts=[ToolCallPart('run_code', args)])
        assert isinstance(part, ToolReturnPart)
        self.results.append(f'retry: {part.content}' if part.outcome == 'retried' else str(part.content))
        return ModelResponse(parts=[TextPart('done')])

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        [part] = self.respond(messages, info).parts
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}

    def agent(self) -> Agent[None, str]:
        return Agent(FunctionModel(self.respond, stream_function=self.stream))


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


async def turn(workspace: Path, script: Script, text: str, **launch: str | bool) -> None:
    """One headless prompt in a fresh shell, as each `clai2 -p` would run it."""
    result = await headless.run_headless(
        text=text,
        settings=Settings(model=None, code_mode=True),
        store=SettingsStore(workspace / 'config.db'),
        project=ProjectSettings(),
        agent=script.agent(),
        **launch,  # pyright: ignore[reportArgumentType]
    )
    assert result == 0


def dump(workspace: Path, conversation_id: str) -> Path:
    return workspace / 'sessions.repl' / f'{conversation_id}.bin'


async def test_state_carries_over_to_the_next_prompt_and_resume(workspace: Path) -> None:
    script = Script()
    await turn(workspace, script, 'set', session_id=SESSION_ID)
    await turn(workspace, script, 'get', resume=SESSION_ID)

    assert script.results == ['{}', '42']
    assert dump(workspace, SESSION_ID).stat().st_mode & 0o777 == 0o600
    # The model is told its state outlives the prompt; a run without a saved session is told it does not.
    assert 'State persists between snippets and is saved with the session' in script.instructions[-1]
    assert 'is saved with the session' not in guidance(None, speculate=False)


async def test_restart_replaces_the_saved_state(workspace: Path) -> None:
    script = Script()
    await turn(workspace, script, 'set', session_id=SESSION_ID)
    await turn(workspace, script, 'restart', resume=SESSION_ID)
    await turn(workspace, script, 'get', resume=SESSION_ID)

    assert script.results[-1].startswith('retry: Runtime error') and "name 'x' is not defined" in script.results[-1]


async def test_fork_session_continues_in_its_own_copy(workspace: Path) -> None:
    script = Script()
    await turn(workspace, script, 'set', session_id=SESSION_ID)
    await turn(workspace, script, 'bump', resume=SESSION_ID, session_id=FORK_ID, fork_session=True)
    await turn(workspace, script, 'get', resume=SESSION_ID)
    await turn(workspace, script, 'get', resume=FORK_ID)

    assert script.results[-2:] == ['42', '142']


async def test_a_fork_never_inherits_a_stale_state(workspace: Path) -> None:
    """A REPL left under an ID does not reach a fork given that ID, even when the forked session has none."""
    script = Script()
    await turn(workspace, script, 'set', session_id=FORK_ID)
    store = SqliteConversationStore(database=workspace / 'sessions.db')
    # Deleted outside the `/resume` browser, so its REPL file stays behind.
    await store.delete(source=(await store.get(conversation_id=FORK_ID)).summary)
    saved = Session(Agent(TestModel()), deps=None, conversations=store, workspace=workspace)
    await saved.prompt('earlier work')

    await turn(workspace, script, 'get', resume=saved.conversation_id, session_id=FORK_ID, fork_session=True)

    assert "name 'x' is not defined" in script.results[-1]


async def test_background_fork_starts_from_the_session_state(workspace: Path) -> None:
    script = Script()
    shell = create_shell(
        script.agent(),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model=None, code_mode=True),
        store=SettingsStore(workspace / 'config.db'),
        builtin_plugins=(),
        project=ProjectSettings(),
        headless=True,
    )
    shell.session.plugins = shell.run_plugins()
    await shell.session.prompt('set')
    await shell.commands.execute_async('/fork get')
    (record,) = shell.forks.records
    await record.task

    assert script.results == ['{}', '42']
    assert record.session_id is not None and dump(workspace, record.session_id).exists()


async def test_busy_session_resumes_as_a_fork_with_a_copy(workspace: Path) -> None:
    store = SqliteConversationStore(database=workspace / 'sessions.db')
    session = Session(Agent(TestModel()), deps=None, conversations=store, workspace=workspace)
    await session.prompt('earlier work')
    states = session.repl_states
    assert states is not None
    source = states.path(session.conversation_id)
    assert source is not None
    source.parent.mkdir()
    source.write_bytes(b'state')
    # This process is alive, so a head it is recorded as running is busy.
    await store.save(
        summary=replace(session.summary, outcome='running', owner_pid=os.getpid()), messages=session.messages
    )

    resumed = Session(Agent(TestModel()), deps=None, conversations=store, workspace=workspace)
    await resumed.resume(session.conversation_id)

    assert resumed.conversation_id != session.conversation_id
    assert dump(workspace, resumed.conversation_id).read_bytes() == b'state'


def catalog(workspace: Path) -> 'Sessions[None, str]':
    """The saved sessions as `/resume` and `PluginHost.sessions` manage them, from a shell on another session."""
    store = SqliteConversationStore(database=workspace / 'sessions.db')
    context = CommandContext(
        settings=Settings(model=None),
        store=SettingsStore(workspace / 'config.db'),
        apply_setting=lambda key, settings: None,
    )
    return Sessions(
        session=Session(Agent(TestModel()), deps=None, conversations=store, workspace=workspace),
        store=store,
        context=context,
    )


async def test_deleting_a_session_removes_its_state(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = Script()
    await turn(workspace, script, 'set', session_id=SESSION_ID)
    assert dump(workspace, SESSION_ID).exists()
    service = catalog(workspace)

    def delete(browser: SessionBrowser) -> str:
        browser.reload()
        assert browser.selected is not None and browser.selected.id == SESSION_ID
        browser.delete(browser.selected)
        return ''

    monkeypatch.setattr(SessionBrowser, 'run', delete)
    assert await service.command([]) == ''
    assert not dump(workspace, SESSION_ID).exists()


async def test_the_session_catalog_forks_and_deletes_the_state(workspace: Path) -> None:
    """A fork or delete from `host.sessions`, as the web UI's sidebar makes, treats the Monty state as `/resume` does."""
    script = Script()
    await turn(workspace, script, 'set', session_id=SESSION_ID)
    service = catalog(workspace)
    copy = await service.fork(SESSION_ID)
    await turn(workspace, script, 'get', resume=copy.id)
    assert script.results[-1] == '42', 'the copy continues from the Monty state the original had'

    await service.delete(SESSION_ID)
    assert not dump(workspace, SESSION_ID).exists()
    assert dump(workspace, copy.id).exists(), 'the copy keeps its own'


async def test_unreadable_state_starts_a_fresh_repl(workspace: Path) -> None:
    script = Script()
    await turn(workspace, script, 'set', session_id=SESSION_ID)
    dump(workspace, SESSION_ID).write_bytes(b'not a monty dump')
    await turn(workspace, script, 'get', resume=SESSION_ID)
    await turn(workspace, script, 'set', resume=SESSION_ID)
    await turn(workspace, script, 'get', resume=SESSION_ID)

    assert script.results[1].startswith('retry: Runtime error') and "name 'x' is not defined" in script.results[1]
    assert script.results[-1] == '42'


async def test_oversized_state_is_not_saved(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    script = Script()
    await turn(workspace, script, 'set', session_id=SESSION_ID)
    monkeypatch.setattr(repl_state, 'MAX_STATE_BYTES', 10)
    with caplog.at_level(logging.WARNING):
        await turn(workspace, script, 'bump', resume=SESSION_ID)

    assert not dump(workspace, SESSION_ID).exists()
    assert 'over the 10 byte limit' in caplog.text


def run_context(conversation_id: str) -> RunContext[object]:
    return RunContext[object](deps=None, model=TestModel(), usage=RunUsage(), conversation_id=conversation_id)


class TestReplStates:
    """`ReplStates` on its own, for the paths no session run reaches."""

    async def test_only_the_session_own_run_is_kept(self, tmp_path: Path) -> None:
        """A sub-agent runs under its own conversation ID, so its REPL ends with its run."""
        states = ReplStates(directory=tmp_path, session_id=lambda: SESSION_ID)
        child = run_context(FORK_ID)
        assert not states.keeps(child)
        await states.save(child, b'state')
        assert await states.load(child) is None
        assert list(tmp_path.iterdir()) == []

        # An ID that is not a plain lowercase file name never reaches the file system.
        for unsafe_id in ('../escape', SESSION_ID.upper()):
            unsafe = ReplStates(directory=tmp_path, session_id=lambda unsafe_id=unsafe_id: unsafe_id)
            assert not unsafe.keeps(run_context(unsafe_id))
            unsafe.copy(source=unsafe_id, target=SESSION_ID)
            unsafe.delete(unsafe_id)
        assert list(tmp_path.iterdir()) == []

    async def test_a_new_conversation_never_inherits_a_stale_state(self, tmp_path: Path) -> None:
        """A REPL left under an ID, by a session deleted outside CLAI, does not reach a new session reusing it."""
        session = Session(
            Agent(TestModel()),
            deps=None,
            conversations=SqliteConversationStore(database=tmp_path / 'sessions.db'),
            workspace=tmp_path,
        )
        stale = dump(tmp_path, SESSION_ID)
        stale.parent.mkdir()
        stale.write_bytes(b'state')
        await session.clear(SESSION_ID)
        assert not stale.exists()

    async def test_file_errors_are_logged_not_raised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        own = run_context(SESSION_ID)
        blocked = ReplStates(directory=tmp_path / 'file', session_id=lambda: SESSION_ID)
        (tmp_path / 'file').write_text('not a directory')
        unreadable = ReplStates(directory=tmp_path / 'dir', session_id=lambda: SESSION_ID)
        path = unreadable.path(SESSION_ID)
        assert path is not None
        path.mkdir(parents=True)

        def no_links(source: Path, target: Path) -> None:
            raise OSError('hard links are not supported')

        monkeypatch.setattr(os, 'link', no_links)
        states = ReplStates(directory=tmp_path / 'states', session_id=lambda: SESSION_ID)
        await states.save(own, b'state')
        with caplog.at_level(logging.WARNING):
            await blocked.save(own, b'state')
            assert await unreadable.load(own) is None
            unreadable.delete(SESSION_ID)
            states.copy(source=SESSION_ID, target=FORK_ID)

        assert 'Could not save the REPL state' in caplog.text
        assert 'Could not read the saved REPL state' in caplog.text
        assert 'Could not delete the REPL state' in caplog.text
        assert 'hard links are not supported' in caplog.text
        assert states.path(FORK_ID) is not None and not (tmp_path / 'states' / f'{FORK_ID}.bin').exists()

    async def test_a_failed_write_keeps_the_saved_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        states = ReplStates(directory=tmp_path, session_id=lambda: SESSION_ID)
        own = run_context(SESSION_ID)
        await states.save(own, b'saved')

        def interrupted(source: str, target: Path) -> None:
            raise OSError('interrupted')

        monkeypatch.setattr(os, 'replace', interrupted)
        with caplog.at_level(logging.WARNING):
            await states.save(own, b'newer')

        assert 'Could not save the REPL state' in caplog.text
        assert [path.name for path in tmp_path.iterdir()] == [f'{SESSION_ID}.bin']
        assert await states.load(own) == b'saved'

    async def test_a_write_in_flight_does_not_outlive_a_delete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A save abandoned to its thread may land after the session is deleted; the delete still wins."""
        states = ReplStates(directory=tmp_path, session_id=lambda: SESSION_ID)
        replace_file = os.replace

        def deleted_meanwhile(source: str, target: Path) -> None:
            states.delete(SESSION_ID)
            replace_file(source, target)

        monkeypatch.setattr(os, 'replace', deleted_meanwhile)
        await states.save(run_context(SESSION_ID), b'state')

        assert list(tmp_path.iterdir()) == []


class Asker:
    """Answers each prompt with one `run_code` cell, and records the prompts and what the cells returned."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.prompts: list[str] = []
        self.results: list[str] = []

    def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        part = request.parts[-1]
        if isinstance(part, UserPromptPart):
            content = part.content
            self.prompts.append(content if isinstance(content, str) else '\n'.join(str(item) for item in content))
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': self.code})])
        assert isinstance(part, ToolReturnPart)
        self.results.append(part.model_response_str())
        return ModelResponse(parts=[TextPart('done')])

    async def stream(self, messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        [part] = self.respond(messages, info).parts
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {0: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}


def monty_shell(workspace: Path, asker: Asker, *, code_mode: bool = True) -> 'SessionRuntime[None, str]':
    return create_shell(
        Agent(FunctionModel(asker.respond, stream_function=asker.stream)),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO()),
        settings=Settings(model=None, code_mode=code_mode),
        store=SettingsStore(workspace / 'config.db'),
        project=ProjectSettings(),
        builtin_plugins=[],
        headless=True,
    )


async def test_a_user_cell_runs_in_the_session_monty(workspace: Path) -> None:
    """`/monty` runs the user's own cell in the Monty sandbox code mode keeps, and the next prompt tells the model."""
    asker = Asker('x + 1')
    shell = monty_shell(workspace, asker)
    assert await shell.commands.execute_async('/monty x = 41') == '(no output)'
    assert await shell.commands.execute_async('/monty print("x is", x)\nx') == 'x is 41\n-> 41'
    assert shell.session.messages == [], 'cells stay out of the conversation'

    await shell.run_turn(TurnStart(text='what is x + 1?'), headless=True)
    assert asker.results == ['42']
    [prompt] = asker.prompts
    assert prompt.count('Between turns, the user ran this cell in Monty, the sandbox your `run_code` runs in') == 2
    assert 'x = 41' in prompt and prompt.endswith('what is x + 1?')

    # Told once: the next prompt carries no note.
    await shell.run_turn(TurnStart(text='again'), headless=True)
    assert asker.prompts[-1] == 'again'


async def test_a_failing_cell_is_reported_not_retried(workspace: Path) -> None:
    shell = monty_shell(workspace, Asker('pass'))
    with pytest.raises(ValueError, match='The cell failed') as failed:
        await shell.commands.execute_async('/monty 1 / 0')
    assert 'ZeroDivisionError' in str(failed.value)


async def test_notes_stay_with_their_conversation(workspace: Path) -> None:
    asker = Asker('pass')
    shell = monty_shell(workspace, asker)
    await shell.commands.execute_async('/monty y = 1')
    await shell.commands.execute_async('/new')
    await shell.run_turn(TurnStart(text='fresh'), headless=True)
    assert asker.prompts == ['fresh'], 'a cell run in another conversation is not mentioned'


async def test_monty_refusals(workspace: Path) -> None:
    shell = monty_shell(workspace, Asker('pass'), code_mode=False)
    with pytest.raises(ValueError, match='Usage: /monty CODE'):
        await shell.commands.execute_async('/monty   ')
    with pytest.raises(ValueError, match='Code mode is off'):
        await shell.commands.execute_async('/monty x = 1')
    on = monty_shell(workspace, Asker('pass'))
    on.session.repl_states = None
    with pytest.raises(ValueError, match='saves no sessions'):
        await on.commands.execute_async('/monty x = 1')


async def test_no_cell_runs_during_a_turn(workspace: Path) -> None:
    shell = monty_shell(workspace, Asker('pass'))
    shell.session._running = True  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match='A turn is running'):
        await shell.session.run_in_monty('x = 1')


def test_a_cell_report_reads_its_result() -> None:
    assert MontyCell(code='', returned='not json').report() == 'not json'
    assert MontyCell(code='', returned='[1, 2]').report() == '[1, 2]'
    assert MontyCell(code='', returned='{"result": {"a": 1}}').report() == '-> {"a": 1}'
    assert MontyCell(code='', returned='{"result": "text"}').report() == '-> text'
    assert MontyCell(code='', returned=None).report() == ''
    long = MontyCell(code='x', returned='y' * (NOTE_CHARS + 3)).note()
    assert long.endswith('\n... 3 more characters') and 'It returned:' in long
    assert MontyCell(code='x', returned='', failed=True).note().endswith('It failed:\n(nothing)')

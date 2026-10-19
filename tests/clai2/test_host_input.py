"""`PluginHost.send_input`, `setting`, `interrupt`, `complete`, `sessions`, and the `on_command_end` moment."""

import io
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import ClassVar

import anyio
import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, conversation_text
from pydantic_clai2 import chat
from pydantic_clai2.commands import Command
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import CommandEnd, Plugin, PluginHost, SessionStart, TurnEnd
from pydantic_clai2.runtime.session_runtime import create_shell
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt
from tests.clai2.surface_terminal import SurfaceTerminal
from tests.clai2.test_tasks import task


class Recorder(Plugin):
    """Keeps its host and what it hears, so a test can drive the shell the way a remote UI would."""

    hosts: ClassVar[list[PluginHost[None]]] = []
    commands: ClassVar[list[CommandEnd]] = []
    turns: ClassVar[list[TurnEnd]] = []
    interrupted: ClassVar[list[bool]] = []

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(name='ask', description='Queue a prompt', handler=self.ask, complete=self.many),
            Command(name='broken', description='A completer that fails', handler=self.ask, complete=self.fail),
        )

    async def ask(self, args: list[str]) -> str:
        self.host.submit_prompt('queued by /ask')
        return 'Asked.'

    def many(self, args: list[str]) -> list[str]:
        return [f'option{index}' for index in range(150)]

    def fail(self, args: list[str]) -> list[str]:
        raise OSError('completer broke')

    async def on_session_start(self, event: SessionStart) -> None:
        self.hosts.append(self.host)

    async def on_command_end(self, event: CommandEnd) -> None:
        self.commands.append(event)
        if event.text == '/ask':
            # As Esc pressed while a slow handler runs: the command has finished, so nothing is cancelled.
            self.interrupted.append(self.host.interrupt())

    async def on_turn_end(self, event: TurnEnd) -> None:
        self.turns.append(event)


RECORDER = PluginSettings(id='recorder', factory='tests.clai2.test_host_input:Recorder')


@pytest.fixture(autouse=True)
def fresh_recorder() -> None:
    Recorder.hosts.clear()
    Recorder.commands.clear()
    Recorder.turns.clear()
    Recorder.interrupted.clear()


def remote_shell(tmp_path: Path, agent: Agent[None, str], *, output: io.StringIO, headless: bool = True):
    return create_shell(
        agent,
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=output, width=120),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[RECORDER],
        project=ProjectSettings(),
        headless=headless,
    )


async def test_send_input_drives_a_shell_without_an_editor(tmp_path: Path) -> None:
    """Input entered from a plugin runs as typed: prompts as turns, commands with their output reported."""
    output = io.StringIO()
    shell = remote_shell(tmp_path, Agent(TestModel(custom_output_text='Hello back')), output=output)
    await shell.loader.load_all()
    [host] = Recorder.hosts
    for text in ('/cost', '/missing', '  ', 'hello there', '/exit'):
        host.send_input(text)
    with anyio.fail_after(10):
        assert await shell.run() == 'exit'
    assert [(event.text, event.output, type(event.error)) for event in Recorder.commands] == [
        ('/cost', 'No usage yet.', type(None)),
        ('/missing', '', ValueError),
        ('/exit', 'Goodbye.', type(None)),
    ]
    assert [(turn.text, turn.outcome) for turn in Recorder.turns] == [('hello there', 'completed')]
    text = output.getvalue()
    assert '> /cost\n' in text and '> hello there\n' in text and 'Hello back' in text
    assert not host.interrupt(), 'nothing runs once the shell has exited'
    await shell.loader.close('exit')


async def test_silent_command_reports_no_output(tmp_path: Path) -> None:
    output = io.StringIO()
    shell = remote_shell(tmp_path, Agent(TestModel()), output=output)
    await shell.loader.load_all()
    [host] = Recorder.hosts
    host.send_input('/new')
    host.send_input('/exit')
    await shell.run()
    assert [(event.text, event.output, event.error) for event in Recorder.commands] == [
        ('/new', '', None),
        ('/exit', 'Goodbye.', None),
    ]
    await shell.loader.close('exit')


async def test_interrupt_cancels_the_running_turn(tmp_path: Path) -> None:
    """`interrupt` cancels as Esc does; a `during_turn` command entered meanwhile opens over the turn."""
    started = anyio.Event()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'Working'
        started.set()
        await anyio.sleep_forever()

    output = io.StringIO()
    shell = remote_shell(tmp_path, Agent(FunctionModel(stream_function=stream)), output=output)
    await shell.loader.load_all()
    [host] = Recorder.hosts

    async def stop_then_exit() -> None:
        await started.wait()
        host.send_input('/plugins list')
        while not Recorder.commands:
            await anyio.sleep(0)
        assert host.interrupt()
        host.send_input('/exit')

    host.send_input('wait forever')
    # Bounded, so a regression in mid-turn dispatch fails here instead of hanging the run.
    with anyio.fail_after(10):
        async with anyio.create_task_group() as group:
            group.start_soon(stop_then_exit)
            assert await shell.run() == 'exit'
    assert [(turn.text, turn.outcome) for turn in Recorder.turns] == [('wait forever', 'cancelled')]
    assert [event.text for event in Recorder.commands] == ['/plugins list', '/exit']
    assert 'recorder' in Recorder.commands[0].output, 'ran over the turn, before it was interrupted'
    assert 'Turn cancelled' in output.getvalue()
    await shell.loader.close('exit')


async def test_input_is_normalized_as_enter_does(tmp_path: Path) -> None:
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    await shell.loader.load_all()
    [host] = Recorder.hosts
    for text in ('   ', '  /cost  ', ' clear ', '/exit'):
        host.send_input(text)
    await shell.run()
    assert [event.text for event in Recorder.commands] == ['/cost', '/clear', '/exit']
    assert Recorder.turns == []
    await shell.loader.close('exit')


async def test_a_refused_plugins_command_reports_its_end(tmp_path: Path) -> None:
    """`/plugins` waits for running delegated tasks; a remote UI hears the refusal as the command's end."""
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    await shell.loader.load_all()
    [host] = Recorder.hosts
    shell.tasks.owner.records['child'] = task(conversation_id=shell.session.summary.id)
    host.send_input('/plugins list')
    host.send_input('/exit')
    await shell.run()
    refused = Recorder.commands[0]
    assert refused.text == '/plugins list' and isinstance(refused.error, RuntimeError)
    assert str(refused.error).startswith('Plugin changes wait for delegated tasks.')
    await shell.loader.close('exit')


class Starter(Plugin):
    """Enters input as soon as it loads, before the terminal's editor exists."""

    async def on_session_start(self, event: SessionStart) -> None:
        self.host.send_input('/exit')


async def test_input_entered_while_loading_waits_for_the_editor(tmp_path: Path) -> None:
    terminal = SurfaceTerminal(width=80, height=24)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()), anyio.fail_after(10):
        await chat(
            Agent(TestModel()),
            deps=None,
            console=Console(file=terminal, force_terminal=True, width=80, height=24),
            store=SettingsStore(tmp_path / 'config.db'),
            builtin_plugins=[PluginSettings(id='starter', factory='tests.clai2.test_host_input:Starter')],
        )
    assert 'Goodbye.' in '\n'.join([*terminal.history, *terminal.lines()])


async def test_send_input_queues_in_the_live_editor(tmp_path: Path) -> None:
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    await shell.loader.load_all()
    [host] = Recorder.hosts
    shell.editor = LivePrompt(
        history=InMemoryHistory(),
        console=shell.console,
        commands=shell.commands,
        images=shell.images,
        interrupts=shell.interrupts,
        toolbar=lambda: [],
    )
    host.send_input('from the browser')
    assert shell.editor.queued_messages == ('from the browser',)
    await shell.loader.close('exit')


async def test_line_mode_prompt_refuses_other_input(tmp_path: Path) -> None:
    """Without a terminal the shell reads standard input line by line, which plugin input cannot join."""
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO(), headless=False)
    await shell.loader.load_all()
    [host] = Recorder.hosts
    with pytest.raises(RuntimeError, match='reads its prompts line by line'):
        host.send_input('hello')
    await shell.loader.close('exit')


async def test_complete_offers_what_tab_does(tmp_path: Path) -> None:
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    await shell.loader.load_all()
    [host] = Recorder.hosts
    [cost] = [completion for completion in await host.complete('/cos') if completion.text == 'cost']
    assert (cost.display, cost.display_meta, cost.start_position) == (
        '/cost',
        'Show retained history cost and tokens',
        -3,
    )
    assert await host.complete('plain prompt') == []
    assert [completion.text for completion in await host.complete('/ask ')] == [
        f'option{index}' for index in range(100)
    ]
    assert await host.complete('/broken ') == [], 'a failing completer offers nothing, as in the editor'
    await shell.loader.close('exit')


async def test_a_slow_command_end_handler_cannot_cancel_the_command(tmp_path: Path) -> None:
    """Plugins hear of a command after Esc stops reaching it, so the prompts it queued still run."""
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    await shell.loader.load_all()
    [host] = Recorder.hosts
    host.send_input('/ask')
    host.send_input('/exit')
    await shell.run()
    assert Recorder.interrupted == [False]
    assert [(turn.text, turn.outcome) for turn in Recorder.turns] == [('queued by /ask', 'completed')]
    await shell.loader.close('exit')


async def test_settings_are_readable(tmp_path: Path) -> None:
    """`host.setting` reads the effective value, so a change through `/set` shows at once."""
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    await shell.loader.load_all()
    [host] = Recorder.hosts
    assert host.setting('run.code_mode') is False, 'the tests turn code mode off'
    assert host.setting('display.theme') == 'default'
    host.send_input('/set run.code_mode true')
    host.send_input('/exit')
    await shell.run()
    assert host.setting('run.code_mode') is True
    with pytest.raises(KeyError):
        host.setting('no.such.setting')
    await shell.loader.close('exit')


async def test_a_host_without_a_shell() -> None:
    host = PluginHost[None](name='alone', console=Console(file=io.StringIO()), settings={})
    with pytest.raises(KeyError):
        host.setting('run.code_mode')
    with pytest.raises(KeyError):
        host.setting('display.theme')
    with pytest.raises(RuntimeError, match='has no prompt to enter input at'):
        host.send_input('hello')
    assert not host.interrupt()
    assert await host.complete('/') == []
    assert host.sessions is None
    assert host.tasks is None


def response(text: str) -> list[ModelMessage]:
    return [ModelRequest(parts=[UserPromptPart(text)]), ModelResponse(parts=[TextPart(f'Re: {text}')])]


async def test_session_catalog(tmp_path: Path) -> None:
    """The saved conversations `/resume` lists: read, retitle, copy, and delete them."""
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    await shell.loader.load_all()
    [host] = Recorder.hosts
    catalog = host.sessions
    assert catalog is not None
    store = shell.sessions.store
    other = await store.save(summary=ConversationSummary(workspace='/elsewhere', title='Other'), messages=response('a'))
    await shell.session.commit_messages(response('mine'))
    current = shell.session.conversation_id

    assert [entry.id for entry in await catalog.listing()] == [current, other.id]
    assert [entry.id for entry in await catalog.listing(query='Other')] == [other.id]
    assert conversation_text((await catalog.saved(other.id)).messages) == conversation_text(response('a'))

    await catalog.rename(current, '  Renamed\n session ')
    assert shell.session.title == 'Renamed session'
    with pytest.raises(ValueError, match='cannot be empty'):
        await catalog.rename(current, ' ')

    copy = await catalog.fork(other.id)
    assert (copy.title, copy.workspace) == ('Other (fork)', '/elsewhere')
    assert conversation_text((await catalog.saved(copy.id)).messages) == conversation_text(response('a'))

    with pytest.raises(ValueError, match='current session cannot be deleted'):
        await catalog.delete(current)
    await catalog.delete(other.id)
    with pytest.raises(LookupError):
        await catalog.saved(other.id)
    await shell.loader.close('exit')


async def test_rename_refuses_a_conversation_that_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    saved = await shell.sessions.store.save(
        summary=ConversationSummary(workspace=str(tmp_path), title='Old'), messages=response('a')
    )

    async def stale(**kwargs: object) -> bool:
        return False

    monkeypatch.setattr(shell.sessions.store, 'name', stale)
    with pytest.raises(ValueError, match='Session changed'):
        await shell.sessions.rename(saved.id, 'New')


async def test_fork_closes_open_calls_of_an_interrupted_conversation(tmp_path: Path) -> None:
    shell = remote_shell(tmp_path, Agent(TestModel()), output=io.StringIO())
    call = ModelResponse(parts=[ToolCallPart('shell', {'command': 'ls'}, tool_call_id='call')])
    saved = await shell.sessions.store.save(
        summary=ConversationSummary(workspace=str(tmp_path), title='Broken', outcome='failed'),
        messages=[ModelRequest(parts=[UserPromptPart('go')]), call],
    )
    copy = await shell.sessions.fork(saved.id)
    [_, last] = (await shell.sessions.saved(copy.id)).messages
    assert isinstance(last, ModelResponse) and last.state == 'interrupted'

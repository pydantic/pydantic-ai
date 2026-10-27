"""The `web` plugin's parts on their own: event translation, history, the answerer race, the server, and `--web`."""

import asyncio
import hashlib
import io
import json
import os
import re
import socket
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import anyio
import httpx
import pytest
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import (
    AgentRunResult,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
)
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse, Question, QuestionOption
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent, CommandStartedEvent
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from pydantic_clai2.builtin_plugins.web import WebPlugin, channel as channel_module, server as server_module
from pydantic_clai2.builtin_plugins.web.app import HEADER, Receive, Scope, Send
from pydantic_clai2.builtin_plugins.web.dialogs import BrowserDialogs
from pydantic_clai2.builtin_plugins.web.live import BrowserAnswerer, Hub
from pydantic_clai2.builtin_plugins.web.protocol import (
    RESULT_CHARS,
    ArgsDelta,
    Diff,
    DiffAdded,
    EventTranslator,
    ForkEntry,
    ImageIds,
    Item,
    ItemAdded,
    ItemLog,
    LiveChanged,
    LiveEntry,
    Notice,
    OrchestratorMessage,
    ServerMessage,
    ShellDone,
    StatusChanged,
    SubCall,
    SubCallAdded,
    SubCallDone,
    SubCallOutput,
    TaskItems,
    TextDelta,
    ToolArgs,
    ToolDone,
    ToolOutput,
    TurnEnded,
    TurnStarted,
    dump,
    dump_session,
    history_items,
    legacy_tags,
)
from pydantic_clai2.builtin_plugins.web.server import WebServer
from pydantic_clai2.builtin_plugins.web.service import Observer
from pydantic_clai2.cli import cli
from pydantic_clai2.cli.web import run_web
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import CommandEnd, ForkEnd, ForkInfo, PluginHost, Transcript
from pydantic_clai2.plugins.menus import Option
from pydantic_clai2.runtime.monty_cells import MontyCell
from pydantic_clai2.runtime.sandbox_calls import SandboxCallFinishedEvent, SandboxCallStartedEvent
from pydantic_clai2.runtime.worktrees import Worktree
from pydantic_clai2.ui.rendering.status import Status
from tests.clai2.test_web import WIRE

REQUEST = AskUserRequest(
    questions=(
        Question(
            header='Colour', question='Which?', options=(QuestionOption(label='Red'), QuestionOption(label='Blue'))
        ),
    )
)
RED = AskUserResponse(answers=(AskUserAnswer(header='Colour', selected=('Red',)),))
BLUE = AskUserResponse(answers=(AskUserAnswer(header='Colour', selected=('Blue',)),))


def test_streamed_parts_become_items() -> None:
    translate = EventTranslator().translate
    assert translate(PartStartEvent(index=0, part=TextPart('Hel'))) == [
        ItemAdded(item=Item(id='live1', kind='assistant', text='Hel'))
    ]
    assert translate(PartDeltaEvent(index=0, delta=TextPartDelta('lo'))) == [TextDelta(id='live1', text='lo')]
    assert translate(PartStartEvent(index=1, part=ThinkingPart('Hmm'))) == [
        ItemAdded(item=Item(id='live2', kind='thinking', text='Hmm'))
    ]
    assert translate(PartDeltaEvent(index=1, delta=ThinkingPartDelta(content_delta='.'))) == [
        TextDelta(id='live2', text='.')
    ]
    # The next response restarts its indexes; its part is a new item.
    assert translate(PartStartEvent(index=0, part=TextPart(''))) == [
        ItemAdded(item=Item(id='live3', kind='assistant', text=''))
    ]
    assert translate(PartDeltaEvent(index=0, delta=TextPartDelta('x'))) == [TextDelta(id='live3', text='x')]
    assert translate(PartDeltaEvent(index=0, delta=ThinkingPartDelta(signature_delta='sig'))) == []
    assert translate(PartDeltaEvent(index=5, delta=TextPartDelta('lost'))) == []
    assert translate(PartStartEvent(index=2, part=ToolCallPart('shell', '{}'))) == []
    assert translate(PartStartEvent(index=3, part=NativeToolCallPart('web_search', {}))) == []
    assert translate(PartDeltaEvent(index=2, delta=ToolCallPartDelta(args_delta='{'))) == []


def test_tool_calls_results_shell_output_and_diffs() -> None:
    translate = EventTranslator().translate
    call = ToolCallPart('shell', {'command': 'ls'}, tool_call_id='c1')
    assert translate(FunctionToolCallEvent(part=call)) == [
        ItemAdded(item=Item(id='c1', kind='tool', name='shell', args='{"command":"ls"}', state='running'))
    ]
    assert translate(CommandStartedEvent(tool_call_id='c1', command='ls', pid=1)) == []
    assert translate(CommandOutputEvent(tool_call_id='c1', text='a.py\n')) == [ToolOutput(id='c1', text='a.py\n')]
    assert translate(CommandOutputEvent(tool_call_id=None, text='lost')) == []
    finished = CommandFinishedEvent(
        tool_call_id='c1', pid=1, output_path='o', status_path='s', exit_code=0, truncated=False, total_lines=1
    )
    assert translate(finished) == [ShellDone(id='c1', exit_code=0)]
    assert translate(FunctionToolResultEvent(part=ToolReturnPart('shell', 'a.py', tool_call_id='c1'))) == [
        ToolDone(id='c1', text='a.py', state='done')
    ]
    retry = ToolReturnPart('shell', 'bad arguments', tool_call_id='c2', outcome='retried')
    [failed] = translate(FunctionToolResultEvent(part=retry))
    assert isinstance(failed, ToolDone) and failed.state == 'error' and 'bad arguments' in failed.text

    edited = FileEditedEvent(
        tool_call_id='e1', path='a.py', root_dir='/w', content_hash='h', diff='-a\n+b', truncated=False
    )
    assert translate(edited) == [DiffAdded(id='e1', diff=Diff(path='a.py', diff='-a\n+b'))]
    assert (
        translate(
            FileEditedEvent(tool_call_id=None, path='a.py', root_dir='/w', content_hash='h', diff='', truncated=False)
        )
        == []
    )

    def write(call_id: str, *, cancelled: bool = False) -> FileChangeRequestEvent:
        return FileChangeRequestEvent(
            tool_call_id=call_id,
            path='n.py',
            root_dir='/w',
            operation='write',
            diff='+new',
            truncated=False,
            cancelled=cancelled,
        )

    assert translate(write('w1')) == []
    assert translate(FileWrittenEvent(tool_call_id='w1', path='n.py', root_dir='/w', content_hash='h')) == [
        DiffAdded(id='w1', diff=Diff(path='n.py', diff='+new'))
    ]
    translate(write('w2', cancelled=True))
    assert translate(FileWrittenEvent(tool_call_id='w2', path='n.py', root_dir='/w', content_hash='h')) == []
    assert translate(FileWrittenEvent(tool_call_id='w3', path='n.py', root_dir='/w', content_hash='h')) == []
    assert translate(FileWrittenEvent(tool_call_id=None, path='n.py', root_dir='/w', content_hash='h')) == []
    edit_request = FileChangeRequestEvent(
        tool_call_id='e2', path='a.py', root_dir='/w', operation='edit', diff='', truncated=False
    )
    assert translate(edit_request) == []


def test_a_run_code_cell_streams_with_its_sandbox_calls() -> None:
    """Code mode runs a cell's calls while the model still writes it, so the cell shows from its first token."""
    translate = EventTranslator().translate
    start = PartStartEvent(index=0, part=ToolCallPart('run_code', '', tool_call_id='cell'))
    assert translate(start) == [ItemAdded(item=Item(id='cell', kind='tool', name='run_code', args='', state='running'))]
    assert translate(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"code": "ls'))) == [
        ArgsDelta(id='cell', text='{"code": "ls')
    ]
    assert translate(PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'code': 'x'}))) == []
    nested = ToolCallPart('shell', {'command': 'ls'}, tool_call_id='cell__1')
    assert translate(SandboxCallStartedEvent(call=nested)) == [
        SubCallAdded(id='cell', call=SubCall(id='cell__1', name='shell', args='{"command":"ls"}'))
    ]
    assert translate(CommandOutputEvent(tool_call_id='cell__1', text='a.py\n')) == [
        SubCallOutput(id='cell', call_id='cell__1', text='a.py\n')
    ]
    finished = SandboxCallFinishedEvent(result=ToolReturnPart('shell', 'a.py', tool_call_id='cell__1'))
    assert translate(finished) == [SubCallDone(id='cell', call_id='cell__1', text='a.py', state='done')]
    failed = SandboxCallFinishedEvent(
        result=ToolReturnPart('grep', 'no', tool_call_id='cell__spec_2', outcome='retried')
    )
    [done] = translate(failed)
    assert isinstance(done, SubCallDone) and (done.call_id, done.state) == ('cell__spec_2', 'error')
    call = ToolCallPart('run_code', {'code': 'ls'}, tool_call_id='cell')
    assert translate(FunctionToolCallEvent(part=call)) == [ToolArgs(id='cell', args='{"code":"ls"}')]
    # Events for top-level calls are not sandbox calls.
    assert translate(SandboxCallStartedEvent(call=ToolCallPart('shell', {}, tool_call_id='plain'))) == []
    top = SandboxCallFinishedEvent(result=ToolReturnPart('shell', '', tool_call_id='plain'))
    assert translate(top) == []


@pytest.mark.parametrize(('args', 'shown'), [(None, ''), ({'code': 'x'}, '{"code":"x"}'), ('{"co', '{"co')])
def test_a_streamed_cell_starts_with_its_arguments_so_far(args: str | dict[str, str] | None, shown: str) -> None:
    [added] = EventTranslator().translate(
        PartStartEvent(index=0, part=ToolCallPart('run_code', args, tool_call_id='c'))
    )
    assert isinstance(added, ItemAdded) and added.item.args == shown


def test_the_item_log_follows_a_cell() -> None:
    log = ItemLog()
    log.apply(ItemAdded(item=Item(id='cell', kind='tool', name='run_code', state='running')))
    for message in (
        ArgsDelta(id='cell', text='{"code": '),
        ArgsDelta(id='cell', text='"ls"}'),
        SubCallAdded(id='cell', call=SubCall(id='cell__1', name='shell', args='{}')),
        SubCallOutput(id='cell', call_id='cell__1', text='out'),
        SubCallDone(id='cell', call_id='cell__1', text='done', state='done'),
        SubCallDone(id='cell', call_id='cell__9', text='unknown call', state='error'),
    ):
        log.apply(message)
    [cell] = log.items.values()
    assert cell.args == '{"code": "ls"}'
    assert cell.calls == [SubCall(id='cell__1', name='shell', args='{}', state='done', text='done', output='out')]
    log.apply(ToolArgs(id='cell', args='{"code":"ls"}'))
    assert cell.args == '{"code":"ls"}'


def test_history_items() -> None:
    long = 'x' * (RESULT_CHARS + 5)
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    [
                        'Look',
                        BinaryContent(b'png', media_type='image/png'),
                        BinaryContent(b'pdf', media_type='application/pdf'),
                    ]
                )
            ]
        ),
        ModelResponse(
            parts=[
                ThinkingPart('plan'),
                ThinkingPart(''),
                TextPart(''),
                TextPart('Reading'),
                ToolCallPart('read_file', {'path': 'a'}, tool_call_id='r1'),
                ToolCallPart('shell', {'command': 'x'}, tool_call_id='s1'),
                ToolCallPart('grep', {'pattern': 'p'}, tool_call_id='g1'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('read_file', long, tool_call_id='r1'),
                ToolReturnPart('shell', 'no such command', tool_call_id='s1', outcome='retried'),
                ToolReturnPart('missing', 'orphan', tool_call_id='zz'),
            ]
        ),
    ]
    items = history_items(messages)
    assert [(item.kind, item.name, item.state) for item in items] == [
        ('user', '', 'done'),
        ('thinking', '', 'done'),
        ('assistant', '', 'done'),
        ('tool', 'read_file', 'done'),
        ('tool', 'shell', 'error'),
        ('tool', 'grep', 'interrupted'),
    ]
    assert (items[0].text, items[0].images) == ('Look\n[image]\n[BinaryContent]', []), 'a preview serves no images'
    image = BinaryContent(b'png', media_type='image/png')
    image_only = ModelRequest(parts=[UserPromptPart(['', image])])
    assert history_items([image_only])[0].text == '[image]', 'a prompt of images alone'
    # A live session lists its images by ID, which it serves, rather than naming them.
    ids = ImageIds()
    [listed] = history_items([image_only], image_ids=ids)
    assert (listed.text, listed.images) == ('', [hashlib.sha256(b'png').hexdigest()])
    assert history_items([image_only], image_ids=ids)[0].images == listed.images, 'from the IDs already hashed'
    assert history_items([ModelRequest(parts=[UserPromptPart('plain')])], image_ids=ids)[0].images == []
    assert ids.find(listed.images[0]) is image
    ids.keep([image])
    assert ids.find(listed.images[0]) is image
    ids.keep([])
    assert ids.find(listed.images[0]) is None, 'not once the session no longer holds it'
    note = MontyCell(code='x = 1', returned='{}').note()
    assert history_items([ModelRequest(parts=[UserPromptPart([note, 'what is x?'])])])[0].text == 'what is x?'
    assert items[3].text.endswith('\n… 5 more characters') and len(items[3].text) < len(long) + 40


def test_item_log_follows_what_browsers_were_sent() -> None:
    log = ItemLog()
    log.apply(ItemAdded(item=Item(id='t', kind='tool', name='shell')))
    for message in (
        ToolOutput(id='t', text='out'),
        ShellDone(id='t', exit_code=2),
        DiffAdded(id='t', diff=Diff(path='p', diff='+')),
        ToolDone(id='t', text='done', state='error'),
        TextDelta(id='missing', text='ignored'),
    ):
        log.apply(message)
    log.apply(ItemAdded(item=Item(id='a', kind='assistant', text='Hi')))
    log.apply(TextDelta(id='a', text=' there'))
    assert list(log.items.values()) == [
        Item(
            id='t',
            kind='tool',
            name='shell',
            text='done',
            state='error',
            output='out',
            exit_code=2,
            diffs=[Diff(path='p', diff='+')],
        ),
        Item(id='a', kind='assistant', text='Hi there'),
    ]
    log.clear()
    assert log.items == {}


async def test_hub_fans_out_only_while_subscribed() -> None:
    hub = Hub()
    hub.publish(Notice(text='nobody listens'))
    with hub.subscribe() as first, hub.subscribe() as second:
        assert hub.clients == 2
        hub.publish(Notice(text='hello'))
        assert (
            first.queue.get_nowait()
            == second.queue.get_nowait()
            == '{"type":"notice","text":"hello","level":"info","command":null,"usage":null,"forks":null,"fork":null}'
        )
    assert hub.clients == 0


async def test_the_browser_answers_first() -> None:
    published: list[ServerMessage] = []
    stopped = anyio.Event()

    async def terminal(request: AskUserRequest) -> AskUserResponse:
        try:
            await anyio.sleep_forever()
        finally:
            stopped.set()
        return RED  # pragma: no cover

    answerer = BrowserAnswerer(publish=lambda message, _: published.append(message), terminal=terminal)
    answers: list[AskUserResponse] = []
    async with anyio.create_task_group() as group:

        async def ask() -> None:
            answers.append(await answerer(REQUEST))

        group.start_soon(ask)
        await anyio.wait_all_tasks_blocked()
        assert [questions.id for questions in answerer.questions] == [REQUEST.id]
        assert answerer.request(REQUEST.id) is REQUEST
        answerer.answer(REQUEST.id, BLUE)
        with pytest.raises(LookupError, match='no longer open'):
            answerer.answer(REQUEST.id, RED)
    assert answers == [BLUE] and stopped.is_set()
    assert [type(message).__name__ for message in published] == ['QuestionAsked', 'QuestionClosed']
    with pytest.raises(LookupError):
        answerer.request(REQUEST.id)
    with pytest.raises(LookupError):
        answerer.answer(REQUEST.id, RED)


async def test_the_terminal_answers_first() -> None:
    async def terminal(request: AskUserRequest) -> AskUserResponse:
        return RED

    answerer = BrowserAnswerer(publish=lambda message, _: None, terminal=terminal)
    assert await answerer(REQUEST) == RED
    assert answerer.questions == []


def context(conversation_id: str) -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), conversation_id=conversation_id)


async def test_only_the_held_conversation_reaches_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """A background `/fork` or delegated task runs with the plugin's capability too; the browser shows only the shell's own."""
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript())
    plugin = WebPlugin.from_host(host)
    observer = Observer(plugin.service)
    with plugin.service.hub.subscribe() as client:
        queue = client.queue
        await observer.on_event(context('another'), event=PartStartEvent(index=0, part=TextPart('fork')))
        assert queue.empty()
        monkeypatch.setattr(channel_module, 'STATUS_INTERVAL', 0)
        await observer.on_event(
            context(host.conversation.conversation_id), event=PartStartEvent(index=0, part=TextPart('mine'))
        )
        assert '"text":"mine"' in queue.get_nowait()
        assert '"type":"status"' in queue.get_nowait()
    assert isinstance(StatusChanged(status=plugin.service.default().status()), StatusChanged)


async def test_forks_that_failed_or_were_cancelled() -> None:
    """A failed fork's notice says why, as the terminal's line does; a cancelled one has no text; a lone host has no forks."""
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript())
    plugin = WebPlugin.from_host(host)
    fork = ForkInfo(id=2, model='test', prompt='go', status='failed', elapsed=1.5, session_id=None)
    entry = ForkEntry(id=2, model='test', prompt='go', status='failed', elapsed=1.5, session_id=None)
    with plugin.service.hub.subscribe() as client:
        await plugin.on_fork_end(ForkEnd(fork=fork, error=RuntimeError('provider down\nsecond line')))
        await plugin.on_fork_end(ForkEnd(fork=replace(fork, status='cancelled')))
        await plugin.on_command_end(CommandEnd(text='/forks', output='No forks yet.'))
        messages = [WIRE.validate_json(client.queue.get_nowait()) for _ in range(client.queue.qsize())]
    notices = [message for message in messages if isinstance(message, Notice)]
    assert notices == [
        Notice(text='RuntimeError: provider down', level='error', fork=entry),
        Notice(text='', fork=replace(entry, status='cancelled')),
        Notice(text='No forks yet.', command='/forks'),
    ]


def test_the_browser_status_shows_the_known_window_before_a_request_measures_one() -> None:
    status = Status(context_tokens=0, model_window=200_000)
    host = PluginHost[None](
        name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript(), status=status
    )
    channel = WebPlugin.from_host(host).service.default()
    assert (channel.status().context_tokens, channel.status().context_window) == (0, 200_000)
    status.context_window = 100_000
    assert channel.status().context_window == 100_000


async def test_runs_of_the_held_conversation_are_turns() -> None:
    """A run woken by a finished task has no prompt; a failure without a message is named by its type."""
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript())
    plugin = WebPlugin.from_host(host)
    observer = Observer(plugin.service)

    async def fail() -> AgentRunResult[object]:
        raise RuntimeError()

    with plugin.service.hub.subscribe() as client:
        queue = client.queue
        with pytest.raises(RuntimeError):
            await observer.wrap_run(context('another'), handler=fail)
        assert queue.empty(), 'a background run is not a turn'
        with pytest.raises(RuntimeError):
            await observer.wrap_run(context(host.conversation.conversation_id), handler=fail)
        messages = [WIRE.validate_json(queue.get_nowait()) for _ in range(queue.qsize())]
    assert [message for message in messages if isinstance(message, TurnStarted | TurnEnded)] == [
        TurnStarted(item=None),
        TurnEnded(outcome='failed', error='RuntimeError'),
    ]
    assert not plugin.service.default().running


async def test_a_taken_port_is_reported() -> None:
    taken = socket.socket()
    taken.bind(('127.0.0.1', 0))
    taken.listen()
    try:
        with pytest.raises(OSError):
            WebServer.serve(lambda port: _unused, host='127.0.0.1', port=taken.getsockname()[1])
    finally:
        taken.close()


async def _unused(scope: Scope, receive: Receive, send: Send) -> None:  # pragma: no cover
    pass


async def test_windows_keeps_its_port_exclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows' `SO_REUSEADDR` would let another program bind the same port, so it is left off there."""
    monkeypatch.setattr(server_module.sys, 'platform', 'win32')
    served = WebServer.serve(lambda port: _unused, host='127.0.0.1', port=0)
    assert served.port > 0
    await served.stop()


class _Output(io.StringIO):
    """Console output that tells when `clai2 --web` is ready for its user."""

    def __init__(self) -> None:
        super().__init__()
        self.ready = anyio.Event()

    def write(self, s: str) -> int:
        if 'Press Ctrl-C to quit.' in s:
            self.ready.set()
        return super().write(s)


@pytest.mark.parametrize('resume', [False, True])
async def test_run_web_serves_until_cancelled(resume: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`clai2 --web` loads `web` for this launch unless it is enabled already, starts it, and quits on Ctrl-C."""
    opened: list[str] = []
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.web.open_browser', opened.append)

    store = SettingsStore(tmp_path / 'config.db')
    saved = None
    if resume:
        store.save_plugin(PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', enabled=True))
        sessions = SqliteConversationStore(database=store.path.with_name('sessions.db'))
        saved = await sessions.save(
            summary=ConversationSummary(workspace=str(Path.cwd().resolve()), title='Saved'), messages=[]
        )
    output = _Output()
    console = Console(file=output, force_terminal=False, width=200)
    with anyio.move_on_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(
                lambda: run_web(
                    settings_for=lambda project: Settings(model='test'),
                    store=store,
                    project=ProjectSettings(),
                    console=console,
                    resume=saved.id if saved else None,
                )
            )
            await output.ready.wait()
            group.cancel_scope.cancel()
    text = output.getvalue()
    assert len(opened) == 1
    assert 'Web UI running at http://127.0.0.1:' in text and 'Press Ctrl-C to quit.' in text
    assert ('Resumed Saved' in text) is resume
    assert [plugin.enabled for plugin in store.plugins() if plugin.id == 'web'] == ([True] if resume else []), (
        'nothing saved'
    )


async def test_run_web_works_in_each_sessions_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`clai2 --web` works where a session runs: `--resume` starts in its directory, `/workspace` starts one elsewhere.

    The process's working directory never changes.
    """
    opened: list[str] = []
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.web.open_browser', opened.append)
    launch, other = tmp_path / 'launch', tmp_path / 'other'
    launch.mkdir()
    other.mkdir()
    monkeypatch.chdir(launch)
    store = SettingsStore(tmp_path / 'config.db')
    saved = await SqliteConversationStore(database=store.path.with_name('sessions.db')).save(
        summary=ConversationSummary(workspace=str(other.resolve()), title='Elsewhere'), messages=[]
    )
    output = _Output()
    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(
                lambda: run_web(
                    settings_for=lambda project: Settings(model='test'),
                    store=store,
                    project=ProjectSettings(),
                    console=Console(file=output, force_terminal=False, width=500),
                    resume=saved.id,
                )
            )
            await output.ready.wait()
            assert Path.cwd() == launch
            found = re.search(r'(http://\S+)/\?token=(\S+)', output.getvalue())
            assert found is not None
            async with httpx.AsyncClient(base_url=found[1], headers={HEADER: found[2]}) as client:
                sent = await client.post('/api/input', json={'text': f'/workspace {launch}'})
                assert sent.json() == {'menu': None}
                while 'New session in' not in output.getvalue():
                    await anyio.sleep(0.01)
            group.cancel_scope.cancel()
    assert 'Resumed Elsewhere' in output.getvalue()
    assert f'New session in {launch.resolve()}.' in output.getvalue()
    assert Path.cwd() == launch


async def test_run_web_refuses_a_session_whose_directory_is_gone(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    saved = await SqliteConversationStore(database=store.path.with_name('sessions.db')).save(
        summary=ConversationSummary(workspace=str(tmp_path / 'gone'), title='Gone'), messages=[]
    )
    with pytest.raises(ValueError, match=re.escape(f'does not exist. Session {saved.id} was saved there;')):
        await run_web(
            settings_for=lambda project: Settings(model='test'),
            store=store,
            project=ProjectSettings(),
            console=Console(file=io.StringIO()),
            resume=saved.id,
        )


async def test_run_web_reports_an_unknown_session(tmp_path: Path) -> None:
    with pytest.raises(BaseExceptionGroup) as raised:
        await run_web(
            settings_for=lambda project: Settings(model='test'),
            store=SettingsStore(tmp_path / 'c.db'),
            project=ProjectSettings(),
            console=Console(file=io.StringIO()),
            resume='00000000-0000-0000-0000-000000000000',
        )
    assert raised.group_contains(LookupError, match='No saved session')


async def test_run_web_needs_a_session_id_to_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='explicit --resume'):
        await run_web(
            settings_for=lambda project: Settings(),
            store=SettingsStore(tmp_path / 'c.db'),
            project=ProjectSettings(),
            console=Console(),
            resume='',
        )


@pytest.mark.parametrize(
    ('argv', 'error'),
    [
        (['--web', '-p', 'hi'], '--web cannot be combined'),
        (['--web', '--agent', 'a:b'], '--web cannot be combined'),
        (['--web', '--resume'], '--web requires an explicit --resume SESSION-ID'),
        (['--port', '8000'], '--port and --token-file need --web'),
        (['--token-file', 'token'], '--port and --token-file need --web'),
        (['--web', '--port', '70000'], '--port must be between 0 and 65535'),
    ],
)
def test_web_flag_rejects(
    argv: list[str], error: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, 'argv', ['clai2', *argv])
    with pytest.raises(SystemExit):
        cli.run()
    assert error in capsys.readouterr().err


def test_web_flag_runs_the_web_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_run_web(**kwargs: object) -> None:
        calls.append(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr('pydantic_clai2.cli.web.run_web', fake_run_web)
    monkeypatch.setattr(
        sys, 'argv', ['clai2', '--web', '--database', str(tmp_path / 'c.db'), '-m', 'test', '--request-limit', '7']
    )
    with pytest.raises(SystemExit) as exited:
        cli.run()
    assert exited.value.code == 130
    [call] = calls
    assert (call['resume'], call['session_id'], call['fork_session']) == (None, None, False)
    settings_for = call['settings_for']
    assert callable(settings_for)
    settings = settings_for(ProjectSettings(overrides={'model': 'openai:gpt-5', 'run.instructions': 'Be brief.'}))
    assert isinstance(settings, Settings)
    assert (settings.model, settings.instructions, settings.request_limit) == ('test', 'Be brief.', 7), (
        'launch flags still win over a project file'
    )


def test_web_flag_reads_a_private_token_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--token-file` hands `run_web` the token in a file only you can read, and `--port` the port."""
    calls: list[dict[str, object]] = []

    async def fake_run_web(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr('pydantic_clai2.cli.web.run_web', fake_run_web)
    token = tmp_path / 'token'
    token.write_text('a' * 40 + '\n')
    token.chmod(0o600)
    # Relative to where CLAI was started, though `--worktree` changes directory before the file is read.
    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / 'checkout'
    elsewhere.mkdir()

    def enter_worktree(*, name: str, headless: bool) -> Worktree:
        os.chdir(elsewhere)
        return Worktree(path=elsewhere, branch='clai-x', head='0', created=False)

    monkeypatch.setattr(cli, '_enter_worktree', enter_worktree)
    argv = ['clai2', '--web', '--database', 'c.db', '--port', '4567', '--token-file', 'token', '--worktree', 'x']
    monkeypatch.setattr(sys, 'argv', argv)
    cli.run()
    [call] = calls
    assert (call['token'], call['port']) == ('a' * 40, 4567)
    argv = ['clai2', '--web', '--database', str(tmp_path / 'c.db'), '--token-file', str(token)]
    monkeypatch.setattr(sys, 'argv', argv)
    for mode, text, error in (
        (0o644, 'a' * 40, 'must belong to you and be readable by you alone'),
        (0o600, 'short', 'must hold one URL-safe token of at least 32 characters'),
    ):
        token.write_text(text)
        token.chmod(mode)
        with pytest.raises(SystemExit):
            cli.run()
        assert error in capsys.readouterr().err
    token.unlink()
    with pytest.raises(SystemExit):
        cli.run()
    assert 'No such file' in capsys.readouterr().err


@pytest.mark.parametrize('launcher_token', [False, True])
async def test_run_web_on_a_launchers_port_and_token(
    launcher_token: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A launcher's port replaces the saved one; with its token, CLAI listens on loopback whatever the saved address,
    prints the address without the token, and opens no browser."""
    store = SettingsStore(tmp_path / 'config.db')
    saved: dict[str, JsonValue] = {'host': 'localhost', 'port': 1}
    store.save_plugin(PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', settings=saved))
    opened: list[str] = []
    monkeypatch.setattr('pydantic_clai2.builtin_plugins.web.open_browser', opened.append)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    token = 'b' * 43 if launcher_token else None
    output = io.StringIO()
    with anyio.fail_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(
                lambda: run_web(
                    settings_for=lambda project: Settings(model='test'),
                    store=store,
                    project=ProjectSettings(),
                    console=Console(file=output, force_terminal=False, width=500),
                    port=port,
                    token=token,
                )
            )
            while 'Press Ctrl-C' not in output.getvalue():
                await anyio.sleep(0.01)
            async with httpx.AsyncClient(trust_env=False) as client:
                headers = {HEADER: token or 'wrong'}
                state = await client.get(f'http://127.0.0.1:{port}/api/state', headers=headers)
            assert state.status_code == (200 if token else 403)
            group.cancel_scope.cancel()
    found = re.search(r'Web UI running at (http://\S+)', output.getvalue())
    assert found is not None
    if token is None:
        assert found[1].startswith(f'http://localhost:{port}/?token=') and len(opened) == 1
    else:
        assert found[1] == f'http://127.0.0.1:{port}/' and opened == []
        assert token not in output.getvalue()


def test_web_flag_returns_after_the_web_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_web(**kwargs: object) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr('pydantic_clai2.cli.web.run_web', fake_run_web)
    monkeypatch.setattr(sys, 'argv', ['clai2', '--web', '--database', str(tmp_path / 'c.db')])
    cli.run()


async def test_browser_dialogs_back_out_and_cancel_a_wait() -> None:
    published: list[ServerMessage] = []
    dialogs = BrowserDialogs(publish=published.append)

    async def forever(show: Callable[[str], None]) -> None:
        show('code 1')
        await asyncio.Event().wait()

    async def fails(show: Callable[[str], None]) -> None:
        raise ValueError('denied')

    waiting: list[bool] = []
    async with anyio.create_task_group() as group:

        async def wait() -> None:
            waiting.append(await dialogs.wait(title='Signing in', description='Finish it.', task=forever))

        group.start_soon(wait)
        await anyio.wait_all_tasks_blocked()
        [dialog] = dialogs.dialogs
        assert dialog.shown == 'code 1', 'a browser connecting now sees the latest text'
        # Any answer cancels a wait, and it reports that it did not finish.
        dialogs.answer(dialog.id, None)
    assert waiting == [False]
    with pytest.raises(ValueError, match='denied'):
        await dialogs.wait(title='Signing in', description='Finish it.', task=fails)

    answers: list[bool | str | None] = []
    async with anyio.create_task_group() as group:

        async def confirm() -> None:
            answers.append(await dialogs.confirm(title='Sure?', yes='Yes', no='No'))

        async def choose() -> None:
            answers.append(await dialogs.choose(title='Pick', options=[Option(value='a', label='A')]))

        group.start_soon(confirm)
        group.start_soon(choose)
        await anyio.wait_all_tasks_blocked()
        first, second = dialogs.dialogs
        dialogs.answer(first.id, 'no')
        # One already answered is left as it is; the rest back out.
        dialogs.cancel_all()
        with pytest.raises(LookupError):
            dialogs.answer(second.id, 'a')
    assert sorted(answers, key=str) == [False, None]
    assert [type(message).__name__ for message in published].count('DialogClosed') == 4


def test_wire_tags_follow_their_class_names_in_python_and_typescript() -> None:
    """Each tag is its class in snake case, as `ArgsDelta` is `args_delta`, and `protocol.ts` sends the same."""
    tags = {
        TextDelta: 'text_delta',
        SubCallAdded: 'sub_call',
        SubCallOutput: 'sub_call_output',
        SubCallDone: 'sub_call_done',
        TurnStarted: 'turn_started',
        TurnEnded: 'turn_ended',
        TaskItems: 'task_items',
    }
    assert {message: message.type for message in tags} == tags
    protocol_ts = (Path(__file__).parents[2] / 'src' / 'pydantic_clai2' / 'web-ui' / 'src' / 'protocol.ts').read_text()
    assert [tag for tag in tags.values() if f"{{ type: '{tag}';" not in protocol_ts] == []
    assert 'export type DelegationOutcome =' in protocol_ts
    assert 'TaskOutcome' not in protocol_ts


def test_legacy_tags_give_a_page_from_before_the_renames_the_tags_it_knows() -> None:
    delta = TextDelta(id='i1', text='hi')
    assert json.loads(legacy_tags(dump_session(delta, session='c1'))) == {
        'session': 'c1',
        'type': 'delta',
        'id': 'i1',
        'text': 'hi',
    }
    items = TaskItems(id='t1', messages=[delta, ToolOutput(id='i2', text='out')])
    assert json.loads(legacy_tags(dump(items))) == {
        'type': 'task_event',
        'id': 't1',
        'messages': [{'type': 'delta', 'id': 'i1', 'text': 'hi'}, {'type': 'tool_output', 'id': 'i2', 'text': 'out'}],
    }
    started = OrchestratorMessage(message=TurnStarted())
    assert json.loads(legacy_tags(dump(started))) == {
        'type': 'orchestrator',
        'message': {'type': 'turn_start', 'item': None},
    }
    assert json.loads(legacy_tags('{"type":"task_items","id":"t1","messages":[1]}')) == {
        'type': 'task_event',
        'id': 't1',
        'messages': [1],
    }
    output = ToolOutput(id='i2', text='"type":"text_delta"')
    for unchanged in (dump(output), dump(OrchestratorMessage(message=output))):
        assert legacy_tags(unchanged) == unchanged


def test_dump_names_the_session_of_a_per_session_message() -> None:
    """A per-session message carries its conversation ID first; a global one carries none."""
    assert json.loads(dump_session(TextDelta(id='live1', text='hi'), session='conv "1"')) == {
        'session': 'conv "1"',
        'type': 'text_delta',
        'id': 'live1',
        'text': 'hi',
    }
    live = LiveChanged(sessions=[LiveEntry(id='c1', state='queued', title=None, workspace='/w')])
    assert json.loads(dump(live)) == {
        'type': 'live',
        'sessions': [{'id': 'c1', 'state': 'queued', 'title': None, 'workspace': '/w'}],
    }

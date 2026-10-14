"""The `web` plugin's parts on their own: event translation, history, the answerer race, the server, and `--web`."""

import asyncio
import io
import socket
import sys
from pathlib import Path

import anyio
import pytest
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
    RetryPromptPart,
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
from pydantic_clai2.builtin_plugins.web import WebPlugin, remote as remote_module, server as server_module
from pydantic_clai2.builtin_plugins.web.app import Receive, Scope, Send
from pydantic_clai2.builtin_plugins.web.live import BrowserAnswerer, Hub
from pydantic_clai2.builtin_plugins.web.protocol import (
    RESULT_CHARS,
    Diff,
    DiffAdded,
    EventTranslator,
    Item,
    ItemAdded,
    ItemLog,
    Notice,
    ServerMessage,
    ShellDone,
    StatusChanged,
    TextDelta,
    ToolDone,
    ToolOutput,
    TurnEnded,
    TurnStarted,
    history_items,
)
from pydantic_clai2.builtin_plugins.web.remote import Observer
from pydantic_clai2.builtin_plugins.web.server import WebServer
from pydantic_clai2.cli import _cli
from pydantic_clai2.cli.web import run_web
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, Transcript
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
    retry = RetryPromptPart('bad arguments', tool_name='shell', tool_call_id='c2')
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


def test_history_items() -> None:
    long = 'x' * (RESULT_CHARS + 5)
    messages = [
        ModelRequest(parts=[UserPromptPart(['Look', BinaryContent(b'png', media_type='image/png')])]),
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
                RetryPromptPart('no such command', tool_name='shell', tool_call_id='s1'),
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
    assert items[0].text == 'Look\n[BinaryContent]'
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
            first.get_nowait()
            == second.get_nowait()
            == '{"type":"notice","text":"hello","level":"info","command":null}'
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

    answerer = BrowserAnswerer(publish=published.append, terminal=terminal)
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

    answerer = BrowserAnswerer(publish=lambda message: None, terminal=terminal)
    assert await answerer(REQUEST) == RED
    assert answerer.questions == []


def context(conversation_id: str) -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage(), conversation_id=conversation_id)


async def test_only_the_held_conversation_reaches_the_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """A background `/fork` or delegated task runs with the plugin's capability too; the browser shows only the shell's own."""
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript())
    plugin = WebPlugin.from_host(host)
    observer = Observer(plugin.remote)
    with plugin.hub.subscribe() as queue:
        await observer.on_event(context('another'), event=PartStartEvent(index=0, part=TextPart('fork')))
        assert queue.empty()
        monkeypatch.setattr(remote_module, 'STATUS_INTERVAL', 0)
        await observer.on_event(
            context(host.conversation.conversation_id), event=PartStartEvent(index=0, part=TextPart('mine'))
        )
        assert '"text":"mine"' in queue.get_nowait()
        assert '"type":"status"' in queue.get_nowait()
    assert isinstance(StatusChanged(status=plugin.remote.status()), StatusChanged)


async def test_runs_of_the_held_conversation_are_turns() -> None:
    """A run woken by a finished task has no prompt; a failure without a message is named by its type."""
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript())
    plugin = WebPlugin.from_host(host)
    observer = Observer(plugin.remote)

    async def fail() -> AgentRunResult[object]:
        raise RuntimeError()

    with plugin.hub.subscribe() as queue:
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
    assert not plugin.remote.running


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


@pytest.mark.parametrize('resume', [False, True])
async def test_run_web_serves_until_cancelled(resume: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`clai2 --web` loads `web` for this launch unless it is enabled already, starts it, and quits on Ctrl-C."""
    opened = anyio.Event()

    def open_browser(url: str) -> None:
        opened.set()

    monkeypatch.setattr('pydantic_clai2.builtin_plugins.web.open_browser', open_browser)
    store = SettingsStore(tmp_path / 'config.db')
    saved = None
    if resume:
        store.save_plugin(PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', enabled=True))
        sessions = SqliteConversationStore(database=store.path.with_name('sessions.db'))
        saved = await sessions.save(
            summary=ConversationSummary(workspace=str(Path.cwd().resolve()), title='Saved'), messages=[]
        )
    output = io.StringIO()
    console = Console(file=output, force_terminal=False, width=200)
    with anyio.move_on_after(30):
        async with anyio.create_task_group() as group:
            group.start_soon(
                lambda: run_web(
                    settings=Settings(model='test'),
                    store=store,
                    project=ProjectSettings(),
                    console=console,
                    resume=saved.id if saved else None,
                )
            )
            await opened.wait()
            group.cancel_scope.cancel()
    text = output.getvalue()
    assert 'Web UI running at http://127.0.0.1:' in text and 'Press Ctrl-C to quit.' in text
    assert ('Resumed Saved' in text) is resume
    assert [plugin.enabled for plugin in store.plugins() if plugin.id == 'web'] == ([True] if resume else []), (
        'nothing saved'
    )


async def test_run_web_needs_a_session_id_to_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='explicit --resume'):
        await run_web(
            settings=Settings(),
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
    ],
)
def test_web_flag_rejects(
    argv: list[str], error: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, 'argv', ['clai2', *argv])
    with pytest.raises(SystemExit):
        _cli.run()
    assert error in capsys.readouterr().err


def test_web_flag_runs_the_web_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    async def fake_run_web(**kwargs: object) -> None:
        calls.append(kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr('pydantic_clai2.cli.web.run_web', fake_run_web)
    monkeypatch.setattr(sys, 'argv', ['clai2', '--web', '--database', str(tmp_path / 'c.db'), '-m', 'test'])
    with pytest.raises(SystemExit) as exited:
        _cli.run()
    assert exited.value.code == 130
    [call] = calls
    assert (call['resume'], call['session_id'], call['fork_session']) == (None, None, False)


def test_web_flag_returns_after_the_web_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_web(**kwargs: object) -> None:
        await asyncio.sleep(0)

    monkeypatch.setattr('pydantic_clai2.cli.web.run_web', fake_run_web)
    monkeypatch.setattr(sys, 'argv', ['clai2', '--web', '--database', str(tmp_path / 'c.db')])
    _cli.run()

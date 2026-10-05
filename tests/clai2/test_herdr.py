"""Herdr reports follow real agent runs and use its newline-delimited socket protocol."""

import asyncio
import io
import json
import socket
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory

import anyio
import pytest
from pydantic import JsonValue, TypeAdapter
from rich.console import Console

from pydantic_ai import Agent, ToolDefinition
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.ask_user import AskUser, AskUserRequest, AskUserResponse
from pydantic_ai_harness.compaction import ReportContextUsage
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from pydantic_clai2._app import DEFAULT_PLUGINS, create_shell
from pydantic_clai2.builtin_plugins import _herdr_client, herdr
from pydantic_clai2.builtin_plugins._herdr_client import HerdrClient
from pydantic_clai2.config import Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionEnd, SessionStart, TurnEnd, TurnStart, load_plugin
from pydantic_clai2.runtime._session import Session


class RecordingClient:
    def __init__(self, **kwargs: object) -> None:
        self.reports: list[tuple[str, dict[str, JsonValue]]] = []
        self.closed = False
        self.titles: asyncio.Queue[JsonValue] = asyncio.Queue()

    def submit(self, lane: str, method: str, params: dict[str, JsonValue]) -> None:
        self.reports.append((method, params))
        if method == 'tab.rename':
            self.titles.put_nowait(params['label'])

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> RecordingClient:
    client = RecordingClient()

    def create(**kwargs: object) -> RecordingClient:
        return client

    monkeypatch.setattr(herdr, 'HerdrClient', create)
    monkeypatch.setenv('HERDR_ENV', '1')
    monkeypatch.setenv('HERDR_SOCKET_PATH', '/unused')
    monkeypatch.setenv('HERDR_PANE_ID', 'w1:p1')
    return client


def make_host() -> PluginHost[None]:
    return PluginHost(name='herdr', console=Console(file=io.StringIO()), settings={})


@pytest.mark.parametrize('missing', ['HERDR_ENV', 'HERDR_SOCKET_PATH', 'HERDR_PANE_ID', 'windows'])
async def test_inactive(missing: str, recorded: RecordingClient, monkeypatch: pytest.MonkeyPatch) -> None:
    if missing == 'windows':
        monkeypatch.setattr(herdr.sys, 'platform', 'win32')
    else:
        monkeypatch.delenv(missing)
    host = make_host()
    loaded = load_plugin(herdr.HerdrPlugin, host)
    assert isinstance(loaded.plugin, herdr.HerdrPlugin) and loaded.plugin.reporter is None
    assert loaded.capabilities == ()
    for event in (
        SessionStart(agent=Agent(TestModel()), settings=Settings()),
        TurnStart(text='hello'),
        TurnEnd(text='hello', outcome='completed'),
        SessionEnd(reason='exit'),
    ):
        await loaded.dispatch(event)
    assert recorded.reports == []


class QuestionModel(TestModel):
    def gen_tool_args(self, tool_def: ToolDefinition) -> JsonValue:
        return {
            'questions': [{'header': 'Choice', 'question': 'Choose?', 'options': [{'label': 'Yes'}, {'label': 'No'}]}]
        }


@pytest.mark.parametrize('answer_error', [False, True])
async def test_question_state_and_cleanup(recorded: RecordingClient, answer_error: bool) -> None:
    host = make_host()
    loaded = load_plugin(herdr.HerdrPlugin, host)

    async def answer(request: AskUserRequest) -> AskUserResponse:
        assert recorded.reports[-1] == ('pane.report_agent', {'state': 'blocked', 'message': 'awaiting input'})
        if answer_error:
            raise ValueError('answerer failed')
        return AskUserResponse(cancelled=True)

    agent = Agent(
        QuestionModel(call_tools=['ask_user_question']),
        deps_type=type(None),
        capabilities=[AskUser(answerer=answer), *loaded.capabilities],
    )
    before = asyncio.all_tasks()
    await loaded.dispatch(SessionStart(agent=agent, settings=Settings()))
    try:
        if answer_error:
            with pytest.raises(ValueError, match='answerer failed'):
                await agent.run('hello')
        else:
            await agent.run('hello')
        states = [params['state'] for method, params in recorded.reports if method == 'pane.report_agent']
        assert states[0] == 'idle'
        assert 'working' in states
        assert 'blocked' in states
        assert states[-1] == 'idle'
    finally:
        await loaded.dispatch(SessionEnd(reason='exit'))
    assert recorded.closed
    assert asyncio.all_tasks() == before


async def test_session_titles_metadata(recorded: RecordingClient, tmp_path: Path) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(Agent(TestModel()), deps=None, conversations=store)
    host = PluginHost(name='herdr', console=Console(file=io.StringIO()), settings={}, conversation=session)
    loaded = load_plugin(herdr.HerdrPlugin, host)
    session.plugins = loaded.capabilities
    await session.prompt('hello')
    await loaded.dispatch(TurnEnd(text='hello', outcome='completed'))
    references = [params for method, params in recorded.reports if method == 'pane.report_agent_session']
    assert references == [{'agent_session_id': session.summary.id, 'agent_session_path': str(store.database)}]
    assert await store.name(source=session.summary, title='My conversation', subtitle='', tags=(), manual=True)
    await loaded.dispatch(TurnEnd(text='hello', outcome='completed'))
    metadata = [params for method, params in recorded.reports if method == 'pane.report_metadata'][-1]
    assert metadata['title'] == 'My conversation'
    assert metadata['ttl_ms'] == 86_400_000
    assert metadata['tokens'] == {'model': 'agent default', 'tokens': f'{session.summary.total_tokens:,}'}
    assert ('tab.rename', {'label': 'My conversation'}) in recorded.reports
    session.clear()
    await loaded.dispatch(TurnEnd(text='', outcome='cancelled'))
    assert recorded.reports[-1] == ('tab.rename', {'label': None})
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert recorded.closed


@pytest.mark.parametrize('failure', [False, True])
async def test_title_read_failure_or_session_switch(
    recorded: RecordingClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    session = Session(Agent(TestModel()), deps=None, conversations=store)
    await session.prompt('hello')
    host = PluginHost(name='herdr', console=Console(file=io.StringIO()), settings={}, conversation=session)
    loaded = load_plugin(herdr.HerdrPlugin, host)
    original_listing = store.listing

    async def listing(*, query: str = '', limit: int = 200, offset: int = 0) -> list[ConversationSummary]:
        if failure:
            raise OSError('unavailable')
        summaries = await original_listing(query=query, limit=limit, offset=offset)
        session.clear()
        return summaries

    monkeypatch.setattr(store, 'listing', listing)
    await loaded.dispatch(TurnEnd(text='', outcome='completed'))
    metadata = [params for method, params in recorded.reports if method == 'pane.report_metadata'][-1]
    assert metadata['clear_title'] is True
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert recorded.closed


async def test_nested_run_does_not_report_idle(recorded: RecordingClient) -> None:
    host = make_host()
    loaded = load_plugin(herdr.HerdrPlugin, host)
    child = Agent(TestModel(), deps_type=type(None), capabilities=loaded.capabilities)
    parent = Agent(TestModel(call_tools=['nested']), deps_type=type(None), capabilities=loaded.capabilities)

    @parent.tool_plain
    async def nested() -> str:
        await child.run('child')
        assert recorded.reports[-1][1]['state'] == 'working'
        return 'done'

    await parent.run('parent')
    states = [params['state'] for method, params in recorded.reports if method == 'pane.report_agent']
    assert states.count('idle') == 1
    assert states[-1] == 'idle'
    await loaded.dispatch(SessionEnd(reason='exit'))


async def test_context_percentage(recorded: RecordingClient) -> None:
    host = make_host()
    loaded = load_plugin(herdr.HerdrPlugin, host)
    agent = Agent(
        TestModel(),
        deps_type=type(None),
        capabilities=[ReportContextUsage(context_window=10_000), *loaded.capabilities],
    )
    await agent.run('hello')
    await loaded.dispatch(TurnEnd(text='hello', outcome='completed'))
    metadata = [params for method, params in recorded.reports if method == 'pane.report_metadata'][-1]
    tokens = metadata['tokens']
    assert isinstance(tokens, dict)
    context = tokens['context']
    assert isinstance(context, str) and context.endswith('%')
    await loaded.dispatch(TurnStart(text='new prompt'))
    metadata = [params for method, params in recorded.reports if method == 'pane.report_metadata'][-1]
    tokens = metadata['tokens']
    assert isinstance(tokens, dict) and 'context' not in tokens
    await loaded.dispatch(SessionEnd(reason='exit'))


async def test_loader_enable_disable(tmp_path: Path, recorded: RecordingClient) -> None:
    shell = create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        settings=None,
        project=ProjectSettings(),
        console=Console(file=io.StringIO()),
        store=SettingsStore(tmp_path / 'settings.db'),
        builtin_plugins=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'herdr'],
    )
    await shell.loader.load_all()
    assert recorded.reports == []
    try:
        await shell.loader.command(['enable', 'herdr'])
        await shell.loader.fire(await shell.run_turn(TurnStart(text='hello'), headless=True))
        assert ('pane.report_agent', {'state': 'working', 'message': 'thinking'}) in recorded.reports
        await shell.loader.command(['disable', 'herdr'])
        assert recorded.closed
        count = len(recorded.reports)
        await shell.loader.fire(await shell.run_turn(TurnStart(text='hello again'), headless=True))
        assert len(recorded.reports) == count
    finally:
        await shell.loader.close('exit')


def test_default_is_opt_in() -> None:
    entry = next(plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'herdr')
    assert not entry.enabled


async def test_background_title_watcher(
    recorded: RecordingClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SqliteConversationStore(database=tmp_path / 'sessions.db')
    agent = Agent(TestModel())
    session = Session(agent, deps=None, conversations=store)
    await session.prompt('hello')
    host = PluginHost(name='herdr', console=Console(file=io.StringIO()), settings={}, conversation=session)
    loaded = load_plugin(herdr.HerdrPlugin, host)

    def no_transcript(*args: object, **kwargs: object) -> None:
        raise AssertionError('Title polling must not load transcripts or media')  # pragma: no cover

    monkeypatch.setattr(store, 'get', no_transcript)
    monkeypatch.setattr(store.media, 'get', no_transcript)
    await loaded.dispatch(SessionStart(agent=agent, settings=Settings()))
    try:
        assert await recorded.titles.get() == session.summary.title
        assert await store.name(source=session.summary, title='Background title', subtitle='', tags=(), manual=True)
        assert await asyncio.wait_for(recorded.titles.get(), timeout=10) == 'Background title'
    finally:
        await loaded.dispatch(SessionEnd(reason='exit'))
    assert recorded.closed


@dataclass(kw_only=True)
class Server:
    path: str
    requests: list[dict[str, JsonValue]] = field(default_factory=list[dict[str, JsonValue]])
    condition: threading.Condition = field(default_factory=threading.Condition)
    label: str = 'original'
    panes: int = 1
    error: bool = False
    empty_reply: bool = False
    missing_label: bool = False

    def wait(self, method: str, count: int = 1) -> None:
        with self.condition:
            assert self.condition.wait_for(
                lambda: sum(request['method'] == method for request in self.requests) >= count, timeout=10
            )


@pytest.fixture
def server() -> Iterator[Server]:
    directory = TemporaryDirectory(dir='/tmp')
    path = str(Path(directory.name) / 'herdr.sock')
    state = Server(path=path)
    ready = threading.Event()
    stop = threading.Event()
    adapter = TypeAdapter(dict[str, JsonValue])

    def serve() -> None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            listener.listen()
            listener.settimeout(0.1)
            ready.set()
            while not stop.is_set():
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                with connection:
                    with connection.makefile('rb') as stream:
                        request = adapter.validate_json(stream.readline())
                    params = request['params']
                    assert isinstance(params, dict)
                    if request['method'] == 'tab.get':
                        response: dict[str, JsonValue] = {
                            'result': {
                                'tab': {
                                    'pane_count': state.panes,
                                    'label': None if state.missing_label else state.label,
                                }
                            }
                        }
                    else:
                        if request['method'] == 'tab.rename' and not state.error:
                            label = params['label']
                            assert isinstance(label, str)
                            state.label = label
                        response = {'error': {'message': 'rejected'}} if state.error else {'result': {'ok': True}}
                    # Record before acknowledging: the client may finish and join
                    # immediately after receiving the reply.
                    with state.condition:
                        state.requests.append(request)
                        state.condition.notify_all()
                    if not state.empty_reply:
                        connection.sendall((json.dumps(response) + '\n').encode())

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    assert ready.wait(10)
    try:
        yield state
    finally:
        stop.set()
        thread.join(10)
        assert not thread.is_alive()
        directory.cleanup()


def test_socket_delivery_and_release(server: Server) -> None:
    requests = server.requests
    client = HerdrClient(socket_path=server.path, pane_id='w1:p1')
    client.close()
    client.close()
    client.submit('state', 'pane.report_agent', {'state': 'working'})
    assert len(requests) == 1
    assert requests[0]['method'] == 'pane.release_agent'
    params = requests[0]['params']
    assert isinstance(params, dict)
    assert params['pane_id'] == 'w1:p1'
    assert params['source'] == 'herdr:clai2'
    assert params['agent'] == 'clai2'


def test_missing_socket_is_nonfatal(tmp_path: Path) -> None:
    client = HerdrClient(socket_path=str(tmp_path / 'missing.sock'), pane_id='w1:p1')
    client.close()


def test_incremental_reply_has_total_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    elapsed = 0.0
    chunks = 0

    def clock() -> float:
        nonlocal elapsed
        elapsed += 0.1
        return elapsed

    class DripSocket:
        def __enter__(self) -> 'DripSocket':
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def settimeout(self, timeout: float) -> None:
            assert timeout > 0

        def connect(self, path: str) -> None:
            pass

        def sendall(self, payload: bytes) -> None:
            pass

        def recv(self, size: int) -> bytes:
            nonlocal chunks
            chunks += 1
            return b' '

    def connect(*args: object) -> DripSocket:
        return DripSocket()

    monkeypatch.setattr(_herdr_client, 'monotonic', clock)
    monkeypatch.setattr(_herdr_client.socket, 'socket', connect)
    client = HerdrClient(socket_path='/unused', pane_id='w1:p1', tab_id='w1:t1')
    # An unbounded read loop would hang here, even though each recv succeeds.
    client.close()
    assert 0 < chunks < 10


@pytest.mark.parametrize('manual', [False, True])
def test_tab_ownership_and_restore(server: Server, manual: bool) -> None:
    client = HerdrClient(socket_path=server.path, pane_id='w1:p1', tab_id='w1:t1')
    try:
        client.submit('title', 'tab.rename', {'label': 'Conversation'})
        server.wait('tab.rename')
        assert server.label == 'Conversation'
        if manual:
            server.label = 'User title'
        client.submit('title', 'tab.rename', {'label': 'Next conversation'})
        server.wait('tab.get', 2)
    finally:
        client.close()
    assert server.label == ('User title' if manual else 'original')
    assert server.requests[-1]['method'] == 'pane.release_agent'


@pytest.mark.parametrize('shared', [False, True])
def test_shared_or_rejected_tab_untouched(server: Server, shared: bool) -> None:
    server.panes = 2 if shared else 1
    server.error = not shared
    client = HerdrClient(socket_path=server.path, pane_id='w1:p1', tab_id='w1:t1')
    try:
        client.submit('title', 'tab.rename', {'label': 'Conversation'})
        server.wait('tab.get')
        # A state request is a barrier behind any in-flight tab rename.
        client.submit('state', 'pane.report_agent', {'state': 'idle'})
        server.wait('pane.report_agent')
    finally:
        client.close()
    assert server.label == 'original'


@pytest.mark.parametrize('missing_label', [False, True])
def test_empty_reply_and_missing_label(server: Server, missing_label: bool) -> None:
    server.empty_reply = not missing_label
    server.missing_label = missing_label
    client = HerdrClient(socket_path=server.path, pane_id='w1:p1', tab_id='w1:t1')
    client.close()
    assert server.label == 'original'
    assert server.requests[-1]['method'] == 'pane.release_agent'


def test_protocol_reports_and_sequences(server: Server) -> None:
    client = HerdrClient(socket_path=server.path, pane_id='w1:p1')
    try:
        client.submit('state', 'pane.report_agent', {'state': 'working'})
        server.wait('pane.report_agent')
        client.submit('activity', 'pane.report_agent', {'state': 'working', 'message': 'running tool'})
        server.wait('pane.report_agent', 2)
        client.submit('session', 'pane.report_agent_session', {'agent_session_id': 'stable'})
        server.wait('pane.report_agent_session')
        client.submit('metadata', 'pane.report_metadata', {'tokens': {'model': 'test'}})
        server.wait('pane.report_metadata')
    finally:
        client.close()
    params = [request['params'] for request in server.requests]
    assert all(isinstance(param, dict) for param in params)
    sequences = [param['seq'] for param in params if isinstance(param, dict) and isinstance(param['seq'], int)]
    assert sequences == sorted(sequences)
    assert len(set(sequences)) == 5
    assert server.requests[-1]['method'] == 'pane.release_agent'


async def test_cancelled_question_returns_idle(recorded: RecordingClient) -> None:
    host = make_host()
    loaded = load_plugin(herdr.HerdrPlugin, host)
    started = anyio.Event()

    async def answer(request: AskUserRequest) -> AskUserResponse:
        started.set()
        await anyio.sleep_forever()
        return AskUserResponse(cancelled=True)  # pragma: no cover

    agent = Agent(
        QuestionModel(call_tools=['ask_user_question']),
        deps_type=type(None),
        capabilities=[AskUser(answerer=answer), *loaded.capabilities],
    )
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(agent.run, 'hello')
        await started.wait()
        assert recorded.reports[-1][1]['state'] == 'blocked'
        tasks.cancel_scope.cancel()
    assert recorded.reports[-1] == ('pane.report_agent', {'state': 'idle', 'message': 'ready'})
    await loaded.dispatch(SessionEnd(reason='exit'))
    assert recorded.closed

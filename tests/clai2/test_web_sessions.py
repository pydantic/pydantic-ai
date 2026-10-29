"""The web UI over several live sessions in one CLAI: each browser watches some, by conversation ID."""

import io
import json
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import httpx2
import pytest
from anyio.from_thread import BlockingPortal, start_blocking_portal
from pydantic import JsonValue
from rich.console import Console
from starlette.testclient import TestClient, WebSocketTestSession

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.builtin_plugins.web import WebPlugin
from pydantic_clai2.builtin_plugins.web.app import HEADER, create_app
from pydantic_clai2.builtin_plugins.web.host_session import HostSessions
from pydantic_clai2.builtin_plugins.web.protocol import PROTOCOL
from pydantic_clai2.builtin_plugins.web.service import Observer, WebService
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, Transcript
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell

from .test_web import HOSTS, TOKEN, QuestionModel, static_dir
from .test_web_parts import REQUEST

GLOBAL = {'snapshot', 'orchestrator', 'live', 'sessions', 'plugins', 'settings', 'theme', 'update'}
"""Message types every browser gets, without a `session`."""
GPT = 'openai:gpt-5.6'
"""A saved model with settings of its own, as `/effort` shows them."""
WEB = PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', settings={'open_browser': False})


class Gate:
    """A model whose turns stream once released: `started` is set as a turn asks it, `release` lets it answer."""

    def __init__(self, portal: BlockingPortal, name: str) -> None:
        self.name = name
        self.started: anyio.Event = portal.call(anyio.Event)
        self.release: anyio.Event = portal.call(anyio.Event)
        self.calls = 0

    def agent(self) -> Agent[None, str]:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            self.calls += 1
            self.started.set()
            await self.release.wait()
            yield f'{self.name} answered'

        return Agent(FunctionModel(stream_function=stream), deps_type=type(None))


@dataclass
class Fleet:
    """`clai2 --web` in-process: a launch session serving the web UI, which opens more in their own folders."""

    root: Path
    portal: BlockingPortal
    agents: dict[str, Agent[None, str]]
    launch: SessionRuntime[None, str] = field(init=False)
    client: TestClient = field(init=False)
    runtimes: dict[str, SessionRuntime[None, str]] = field(default_factory=dict[str, SessionRuntime[None, str]])
    """By folder name."""
    running: Future[None] | None = None

    def __post_init__(self) -> None:
        self.launch = self._build(self.folder('home'))
        self.launch.process.live.build = lambda directory: self._build(directory or self.folder('home'))
        self.portal.call(self.launch.loader.load_all)
        [plugin] = [
            entry.loaded.plugin for entry in self.launch.loader.entries() if entry.name == 'web' and entry.loaded
        ]
        assert isinstance(plugin, WebPlugin)
        self.plugin = plugin
        app = create_app(service=plugin.service, token=TOKEN, hosts=HOSTS, static=static_dir(self.root))
        self.client = TestClient(app)
        self.client.portal = self.portal

        serving = self.portal.call(anyio.Event)

        async def serve() -> None:
            # As `clai2 --web` does: the others close as serving ends, then the launch session's plugins.
            async with self.launch.process.live.serving():
                serving.set()
                reason = await self.launch.run()
            await self.launch.close(reason)

        self.running = self.portal.start_task_soon(serve)
        self.portal.call(serving.wait)

    def folder(self, name: str) -> Path:
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        return directory.resolve()

    def _build(self, directory: Path) -> SessionRuntime[None, str]:
        joined = getattr(self, 'launch', None)
        runtime = create_shell(
            self.agents.get(directory.name) or Agent(TestModel()),
            deps=None,
            plugins=(),
            usage_limits=None,
            console=Console(file=io.StringIO(), width=100),
            settings=None,
            store=SettingsStore(self.root / 'config.db') if joined is None else None,
            builtin_plugins=[WEB] if joined is None else (),
            project=ProjectSettings(),
            headless=True,
            workspace=directory,
            process=None if joined is None else joined.process,
            signals=joined is None,
        )
        self.runtimes[directory.name] = runtime
        return runtime

    @property
    def service(self) -> WebService:
        return self.plugin.service

    def post(self, path: str, body: object = None) -> httpx2.Response:
        return self.client.post(path, json=body, headers={HEADER: TOKEN})

    def get(self, path: str) -> httpx2.Response:
        return self.client.get(path, headers={HEADER: TOKEN})

    def socket(self) -> WebSocketTestSession:
        return self.client.websocket_connect(
            f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
        )

    def exit(self) -> None:
        """End the launch session, and with it the process's others and the web UI."""
        assert self.running is not None
        if not self.running.done():
            self.portal.call(lambda: self.launch.send_input('/exit'))
            self.running.result(timeout=10)

    def new(self, name: str) -> str:
        """Start a live session in folder `name`; its conversation ID."""
        response = self.post('/api/sessions/new', {'workspace': str(self.folder(name))})
        assert response.status_code == 200, response.text
        return response.json()['id']


MakeFleet = Callable[..., Fleet]


@pytest.fixture
def make_fleet(tmp_path: Path) -> Iterator[MakeFleet]:
    with start_blocking_portal() as portal:
        made: list[Fleet] = []

        def make(agents: Callable[[BlockingPortal], dict[str, Agent[None, str]]] = lambda _: {}) -> Fleet:
            made.append(Fleet(root=tmp_path, portal=portal, agents=agents(portal)))
            return made[-1]

        yield make
        for fleet in made:
            fleet.exit()


def messages(
    socket: WebSocketTestSession, *, until: Callable[[dict[str, JsonValue]], bool]
) -> list[dict[str, JsonValue]]:
    """Raw messages from the WebSocket up to and including the first `until` accepts."""
    received: list[dict[str, JsonValue]] = []
    while True:
        message = json.loads(socket.receive_text())
        received.append(message)
        if until(message):
            return received


def kind(name: str, **fields: JsonValue) -> Callable[[dict[str, JsonValue]], bool]:
    return lambda message: message['type'] == name and all(message.get(key) == value for key, value in fields.items())


def snapshot_of(conversation_id: str) -> Callable[[dict[str, JsonValue]], bool]:
    def check(message: dict[str, JsonValue]) -> bool:
        session = message.get('session')
        return message['type'] == 'snapshot' and isinstance(session, dict) and session['id'] == conversation_id

    return check


def connected(socket: WebSocketTestSession) -> dict[str, JsonValue]:
    """The default session's snapshot, after reading `live` and the orchestrator's state, as the page gets them."""
    snapshot = json.loads(socket.receive_text())
    assert [json.loads(socket.receive_text())['type'] for _ in range(2)] == ['live', 'orchestrator']
    return snapshot


def at(value: JsonValue, *path: str | int) -> JsonValue:
    """The value at `path` in a decoded message, by key or list index."""
    for step in path:
        if isinstance(step, int):
            assert isinstance(value, list)
            value = value[step]
        else:
            assert isinstance(value, dict)
            value = value[step]
    return value


def watch(socket: WebSocketTestSession, *sessions: str) -> None:
    socket.send_text(json.dumps({'type': 'watch', 'sessions': list(sessions)}))


def live_states(message: dict[str, JsonValue]) -> dict[str, JsonValue]:
    sessions = message['sessions']
    assert isinstance(sessions, list)
    return {
        entry['id']: entry['state'] for entry in sessions if isinstance(entry, dict) and isinstance(entry['id'], str)
    }


def test_each_browser_gets_only_the_sessions_it_watches(make_fleet: MakeFleet) -> None:
    """Two turns stream at once in two folders; each browser sees only its session, every message naming it."""
    gates: dict[str, Gate] = {}

    def agents(portal: BlockingPortal) -> dict[str, Agent[None, str]]:
        gates.update({name: Gate(portal, name) for name in ('home', 'b')})
        return {name: gate.agent() for name, gate in gates.items()}

    fleet = make_fleet(agents)
    a = fleet.launch.session.conversation_id
    launched_in = Path.cwd()
    with fleet.socket() as one, fleet.socket() as two:
        connected(one)
        connected(two)
        b = fleet.new('b')
        assert b != a and fleet.new('c') not in (a, b)
        watch(one, a)
        watch(two, b)
        assert messages(one, until=snapshot_of(a))[-1]['session'] == {
            'id': a,
            'title': None,
            'workspace': str(fleet.folder('home')),
        }
        snapshot = messages(two, until=snapshot_of(b))[-1]
        assert snapshot['session'] == {'id': b, 'title': None, 'workspace': str(fleet.folder('b'))}
        assert fleet.post(f'/api/sessions/{a}/input', {'text': 'go'}).json() == {'menu': None}
        assert fleet.post(f'/api/sessions/{b}/input', {'text': 'go'}).json() == {'menu': None}
        fleet.portal.call(gates['home'].started.wait)
        fleet.portal.call(gates['b'].started.wait)
        # Both run at once, and input sent to one meanwhile waits behind its turn.
        assert fleet.post(f'/api/sessions/{b}/input', {'text': 'and then'}).status_code == 200
        states = live_states(
            messages(one, until=lambda m: m['type'] == 'live' and live_states(m).get(b) == 'queued')[-1]
        )
        assert (states[a], states[b]) == ('running', 'queued')
        sidebar = {
            entry['id']: entry['state']
            for project in fleet.get('/api/sessions').json()
            for worktree in project['worktrees']
            for entry in worktree['sessions']
        }
        assert (sidebar[a], sidebar[b]) == ('running', 'running'), 'live turns are running, never busy elsewhere'
        fleet.portal.call(gates['home'].release.set)
        first = messages(one, until=kind('turn_ended', session=a))
        fleet.portal.call(gates['b'].release.set)
        # The queued input runs as a second turn once the first ends.
        second = messages(two, until=kind('turn_ended', session=b))
        second += messages(two, until=kind('turn_ended', session=b))
        assert gates['b'].calls == 2
    for received, mine, answer in ((first, a, 'home answered'), (second, b, 'b answered')):
        scoped = [message for message in received if message['type'] not in GLOBAL]
        assert scoped and {str(at(message, 'session')) for message in scoped} == {mine}
        assert not any(
            message['type'] in GLOBAL and 'session' in message for message in received if message['type'] != 'snapshot'
        )
        texts = json.dumps(scoped)
        assert answer in texts and ('b answered' if mine == a else 'home answered') not in texts
    assert Path.cwd() == launched_in


def test_watching_switches_sessions_and_keeps_them_open(make_fleet: MakeFleet) -> None:
    """`watch` sends a snapshot of each newly watched live session, counts the browser as a watcher, and ignores IDs not live."""
    fleet = make_fleet()
    a = fleet.launch.session.conversation_id
    b = fleet.new('b')
    session_b = fleet.service.session(b).session
    with fleet.socket() as socket:
        assert at(connected(socket), 'session', 'id') == b, 'a browser that never watched follows the newest session'
        watch(socket, a, 'not-live')
        assert at(messages(socket, until=kind('snapshot'))[-1], 'session', 'id') == a
        assert session_b.watchers == 0
        watch(socket, a, b)
        assert at(messages(socket, until=kind('snapshot'))[-1], 'session', 'id') == b, 'only b is new'
        assert session_b.watchers == 1
        watch(socket, a)
        fleet.post(f'/api/sessions/{a}/input', {'text': '/cost'})
        notice = messages(socket, until=kind('notice'))[-1]
        assert notice['session'] == a and session_b.watchers == 0
        watch(socket, b)
        messages(socket, until=kind('snapshot'))
        assert session_b.watchers == 1
        socket.send_text('not a watch')  # Ignored, as anything else an older or newer page sends.
        socket.send_text(json.dumps({'type': 'watch'}))
    fleet.portal.call(anyio.wait_all_tasks_blocked)
    assert session_b.watchers == 0, 'a browser that leaves stops watching'


def test_watching_the_session_a_browser_connected_to_sends_no_second_snapshot(make_fleet: MakeFleet) -> None:
    """A reloaded page watches the default session it got on connect: that snapshot is not sent again (#11304)."""
    fleet = make_fleet()
    a = fleet.launch.session.conversation_id
    session_a = fleet.service.session(a).session
    with fleet.socket() as socket:
        assert at(connected(socket), 'session', 'id') == a
        watch(socket, a)
        fleet.portal.call(anyio.wait_all_tasks_blocked)  # The server has handled `watch`.
        fleet.post(f'/api/sessions/{a}/input', {'text': '/cost'})
        received = messages(socket, until=kind('notice'))
        assert [message['type'] for message in received if message['type'] == 'snapshot'] == []
        assert received[-1]['session'] == a and session_a.watchers == 1


def test_a_session_that_switches_conversation_takes_its_watchers(make_fleet: MakeFleet) -> None:
    """`/new` typed in a live session sends its watchers the new conversation's snapshot, and `live` names it."""
    fleet = make_fleet()
    b = fleet.new('b')
    with fleet.socket() as socket:
        connected(socket)
        watch(socket, b)
        fleet.post(f'/api/sessions/{b}/input', {'text': '/new'})
        snapshot = messages(socket, until=lambda m: m['type'] == 'snapshot' and at(m, 'session', 'id') != b)[-1]
        renamed = at(snapshot, 'session', 'id')
        assert isinstance(renamed, str)
        assert renamed != b and snapshot['items'] == []
        assert snapshot['replaces'] == b, 'it names the conversation it replaces, for a browser keeping several'
        assert renamed in live_states(
            messages(socket, until=lambda m: m['type'] == 'live' and renamed in live_states(m))[-1]
        )
        assert fleet.get(f'/api/sessions/{b}/state').status_code == 404
        assert fleet.get(f'/api/sessions/{renamed}/state').json()['session']['workspace'] == str(fleet.folder('b'))
        fleet.post(f'/api/sessions/{renamed}/input', {'text': '/cost'})
        assert messages(socket, until=kind('notice'))[-1]['session'] == renamed


def test_session_routes(make_fleet: MakeFleet) -> None:
    """Each route acts on the session it names, 404 when none is live; the legacy ones on the default session."""
    gate: list[Gate] = []

    def agents(portal: BlockingPortal) -> dict[str, Agent[None, str]]:
        gate.append(Gate(portal, 'b'))
        return {'b': gate[0].agent(), 'q': Agent(QuestionModel(), deps_type=type(None))}

    fleet = make_fleet(agents)
    a = fleet.launch.session.conversation_id
    b = fleet.new('b')
    for method, path in (
        ('GET', 'state'),
        ('POST', 'input'),
        ('POST', 'interrupt'),
        ('GET', 'complete?text=/he'),
        ('POST', 'model'),
        ('POST', 'switch'),
        ('POST', 'questions/x'),
        ('POST', 'dialogs/x'),
    ):
        body = {'text': 'hi', 'name': 'test', 'key': 'run.code_mode', 'on': True, 'answers': None, 'value': None}
        response = fleet.client.request(method, f'/api/sessions/nope/{path}', json=body, headers={HEADER: TOKEN})
        assert response.status_code == 404, (path, response.text)
    assert fleet.get('/api/state').json()['session']['id'] == b, 'the legacy routes drive the default session'
    assert 'help' in [completion['text'] for completion in fleet.get(f'/api/sessions/{a}/complete?text=/he').json()]
    assert fleet.post(f'/api/sessions/{a}/dialogs/none', {'value': None}).status_code == 404
    with fleet.socket() as socket:
        connected(socket)
        watch(socket, a, b)
        assert fleet.post(f'/api/sessions/{a}/switch', {'key': 'run.code_mode', 'on': True}).status_code == 204
        assert messages(socket, until=kind('notice', session=a))[-1]['command'] == '/set run.code_mode true'
        assert fleet.post(f'/api/sessions/{a}/model', {'name': 'test'}).status_code == 204
        assert messages(socket, until=kind('notice', session=a))[-1]['command'] == '/model test'
        fleet.post(f'/api/sessions/{b}/input', {'text': 'go'})
        fleet.portal.call(gate[0].started.wait)
        assert fleet.post(f'/api/sessions/{a}/interrupt').json() == {'interrupted': False}
        assert fleet.post(f'/api/sessions/{b}/interrupt').json() == {'interrupted': True}
        ended = messages(socket, until=kind('turn_ended', session=b))[-1]
        assert ended['outcome'] == 'cancelled'

        q = fleet.new('q')
        watch(socket, a, b, q)
        messages(socket, until=snapshot_of(q))
        fleet.post(f'/api/sessions/{q}/input', {'text': 'ask me'})
        asked = messages(socket, until=kind('question', session=q))[-1]
        request = at(asked, 'questions', 'id')
        assert (
            live_states(
                messages(socket, until=lambda m: m['type'] == 'live' and live_states(m).get(q) == 'asking')[-1]
            )[q]
            == 'asking'
        )
        assert fleet.get(f'/api/sessions/{q}/state').json()['questions'][0]['id'] == request
        assert fleet.get(f'/api/sessions/{a}/state').json()['questions'] == []
        answers = {'answers': [{'header': 'Colour', 'selected': ['Red']}]}
        assert fleet.post(f'/api/sessions/{a}/questions/{request}', answers).status_code == 404, "another session's"
        assert fleet.post(f'/api/sessions/{q}/questions/{request}', answers).status_code == 204
        assert messages(socket, until=kind('question_closed', session=q))[-1]['id'] == request
        messages(socket, until=kind('turn_ended', session=q))


def test_open_makes_a_saved_session_live_once(make_fleet: MakeFleet) -> None:
    """Opening a saved session resumes it in its own folder, once; the legacy routes then drive it."""
    fleet = make_fleet()
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('hi')]), ModelResponse(parts=[TextPart('hello')])]
    summary = ConversationSummary(workspace=str(fleet.folder('saved')), title='Saved')
    saved = fleet.portal.call(lambda: fleet.launch.process.conversations.save(summary=summary, messages=history))
    with fleet.socket() as socket, fleet.socket() as legacy:
        connected(socket)
        connected(legacy)
        watch(socket, saved.id)  # Not live yet: followed once it opens.
        assert fleet.post(f'/api/sessions/{saved.id}/open').json() == {'id': saved.id}
        snapshot = messages(socket, until=snapshot_of(saved.id))[-1]
        messages(legacy, until=snapshot_of(saved.id))  # It follows the default, now the opened one.
        assert fleet.service.session(saved.id).session.watchers == 2
        assert [at(snapshot, 'items', index, 'text') for index in range(2)] == ['hi', 'hello'], (
            'the page follows what it opened'
        )
        assert fleet.post(f'/api/sessions/{saved.id}/open').json() == {'id': saved.id}
    assert [runtime for runtime in fleet.runtimes] == ['home', 'saved']
    assert fleet.get('/api/state').json()['session']['id'] == saved.id
    assert fleet.runtimes['saved'].session.workspace == str(fleet.folder('saved'))
    assert fleet.post('/api/sessions/unknown/open').status_code == 404


def test_a_conversation_watched_before_a_session_resumes_it_gets_one_snapshot(make_fleet: MakeFleet) -> None:
    """`/resume` in a live session hands its new conversation to a browser that watched that ID already."""
    fleet = make_fleet()
    a = fleet.launch.session.conversation_id
    summary = ConversationSummary(workspace=str(fleet.folder('home')), title='Saved')
    saved = fleet.portal.call(lambda: fleet.launch.process.conversations.save(summary=summary, messages=[]))
    with fleet.socket() as socket, fleet.socket() as other:
        connected(socket)
        connected(other)
        watch(socket, saved.id)
        watch(other, a)  # Followed since it connected, so no second snapshot comes.
        fleet.portal.call(anyio.wait_all_tasks_blocked)
        fleet.post(f'/api/sessions/{a}/input', {'text': f'/resume {saved.id}'})
        messages(other, until=snapshot_of(saved.id))
        fleet.post(f'/api/sessions/{saved.id}/input', {'text': '/cost'})
        received = messages(socket, until=kind('notice', session=saved.id))
    assert len([message for message in received if snapshot_of(saved.id)(message)]) == 1
    assert fleet.launch.loader.live_session.watchers == 0


def menu_rows(fleet: Fleet, path: str) -> dict[str, dict[str, JsonValue]]:
    """A command menu's rows by key, as the Settings window reads them."""
    response = fleet.get(path)
    assert response.status_code == 200, response.text
    fields = at(response.json(), 'fields')
    assert isinstance(fields, list)
    return {str(at(field, 'key')): field for field in fields if isinstance(field, dict)}


def test_command_menus_act_on_the_session_they_name(make_fleet: MakeFleet) -> None:
    """`/model` chosen in session B's menu switches B only, and B's `/effort` reads B's model."""
    fleet = make_fleet()
    a = fleet.launch.session.conversation_id
    b = fleet.new('b')
    session_a, session_b = fleet.launch.session, fleet.runtimes['b'].session
    fleet.launch.context.store.add_model(name=GPT)
    assert 'openai_reasoning_effort' not in menu_rows(fleet, f'/api/sessions/{b}/commands/effort/fields')
    chosen = fleet.post(f'/api/sessions/{b}/commands/model/fields/model', {'value': GPT})
    assert chosen.json()['messages'] == ['Saved model. Applied.']
    assert (session_a.model, session_b.model) == (None, GPT)
    assert 'openai_reasoning_effort' in menu_rows(fleet, f'/api/sessions/{b}/commands/effort/fields')
    assert 'openai_reasoning_effort' not in menu_rows(fleet, f'/api/sessions/{a}/commands/effort/fields')
    # A model's own settings, one menu down, and its action, both through B's commands.
    nested = f'/api/sessions/{b}/commands/model/fields/openai_reasoning_effort?menu={GPT}'
    assert fleet.post(nested, {'value': 'low'}).status_code == 200
    assert fleet.post(f'/api/sessions/{b}/commands/model/fields/add/action?menu=nope').status_code == 404
    for path in ('commands/model/fields', 'commands/model/fields/model', 'commands/model/fields/add/action'):
        method = 'GET' if path.endswith('fields') else 'POST'
        response = fleet.client.request(
            method, f'/api/sessions/nope/{path}', json={'value': GPT}, headers={HEADER: TOKEN}
        )
        assert response.status_code == 404, path
    # `/set`'s `run.*` rows are per session too.
    assert fleet.post(f'/api/sessions/{b}/commands/set/fields/run.code_mode', {'value': 'true'}).status_code == 200
    live_a, live_b = fleet.launch.loader.live_session, fleet.runtimes['b'].loader.live_session
    assert (live_a.setting('run.code_mode'), live_b.setting('run.code_mode')) == (False, True)
    # The unprefixed routes drive the default session, now B, the newest; A is untouched.
    assert fleet.post('/api/commands/model/fields/model', {'value': 'test'}).status_code == 200
    assert (session_a.model, session_b.model) == (None, 'test')


def test_a_session_the_browser_opened_keeps_update_menu_and_refuses_its_install(make_fleet: MakeFleet) -> None:
    """Settings > About in a session the browser opened: `/update`'s menu, its own description, and its refusal."""
    fleet = make_fleet()
    b = fleet.new('b')
    assert set(menu_rows(fleet, f'/api/sessions/{b}/commands/update/fields')) == {'version', 'updates.channel'}
    descriptions = {command.name: command.description for command in fleet.runtimes['b'].commands}
    assert (descriptions['update'], descriptions['reload']) == (
        'Show whether a newer CLAI is out on updates.channel (stable or main); /update now installs it',
        'Reload CLAI2 code without restarting',
    )
    # No terminal shows it, so Update says why, as `/update now` typed there would, instead of asking first.
    response = fleet.post(f'/api/sessions/{b}/commands/update/fields/version/action')
    assert response.status_code == 400, response.text
    assert response.json()['detail'] == (
        '/reload and /update restart CLAI. Enter them in the session the terminal shows, '
        'or with no terminal, in the session CLAI started with.'
    )


def test_the_orchestrator_runs_on_the_saved_default_model(make_fleet: MakeFleet) -> None:
    """Not one session's choice: the saved `model` setting, else the agent's own."""
    fleet = make_fleet()
    host = fleet.plugin.host
    assert fleet.portal.call(host.default_model) is fleet.launch.agent.model
    assert isinstance(model := fleet.launch.agent.model, Model)
    assert fleet.portal.call(host.default_model_name) == model.model_name
    fleet.launch.context.store.set('model', 'test')
    assert fleet.portal.call(host.default_model) == 'test'
    assert fleet.portal.call(host.default_model_name) == 'test'
    bare = PluginHost[None](name='bare', console=Console(), settings={})
    assert (fleet.portal.call(bare.default_model), fleet.portal.call(bare.default_model_name)) == (None, None)


async def test_a_host_outside_the_loader_is_its_one_session() -> None:
    """A `web` plugin built on a bare host, as in a test, shows that host as its only live session."""
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript())
    live = HostSessions(host)
    [session] = live.listing()
    assert live.get(session.id) is session and live.current() is None
    with pytest.raises(LookupError):
        live.get('other')
    for opening in (live.open('x'), live.new(), live.close(session.id)):
        with pytest.raises(RuntimeError, match='opens no other'):
            await opening
    live.subscribe(lambda: None)()
    unwatch = session.watch()
    unwatch()
    assert (session.session_id, session.workspace, session.conversation) == (None, Path.cwd(), host.conversation)
    assert (session.status, session.sessions, session.tasks) == (host.status, None, None)
    assert (session.running, session.queued, session.watchers, session.interrupt()) == (False, 0, 0, False)
    assert (session.queue, session.steer('hi')) == ([], False), 'a bare host queues nothing and steers nothing'
    for change in (lambda: session.edit_queued('q1', 'x'), lambda: session.remove_queued('q1')):
        with pytest.raises(LookupError, match='no longer queued'):
            change()
    assert await session.complete('/') == []
    assert session.command_menu('set') is None, 'a bare host has no commands'
    with pytest.raises(KeyError):
        session.setting('run.code_mode')
    service = WebService(host=host)
    assert service.live_message().sessions[0].state == 'idle'


def test_closing_a_watched_session_or_the_web_ui_stops_counting_its_browsers(make_fleet: MakeFleet) -> None:
    """A session that closes leaves `live`, and a browser that never sent `watch` follows the next default.

    Ending the launch session stops the web UI counting any browser as a watcher.
    """
    fleet = make_fleet()
    b = fleet.new('b')
    session_b = fleet.service.session(b).session
    session_a = fleet.launch.loader.live_session
    a = session_a.conversation.conversation_id
    with fleet.socket() as socket, fleet.socket() as legacy:
        assert at(connected(legacy), 'session', 'id') == b, 'the newest session is the default'
        connected(socket)
        watch(socket, a, b)
        snapshot = messages(socket, until=kind('snapshot'))[-1]
        assert at(snapshot, 'session', 'id') == a, 'b, the default, has shown since it connected'
        assert (session_a.watchers, session_b.watchers) == (1, 2)
        fleet.portal.call(fleet.launch.process.live.close, session_b.id)
        assert b not in live_states(
            messages(socket, until=lambda m: m['type'] == 'live' and b not in live_states(m))[-1]
        )
        assert at(messages(legacy, until=snapshot_of(a))[-1], 'session', 'id') == a
        assert (session_a.watchers, session_b.watchers) == (2, 0)
        fleet.exit()
        assert session_a.watchers == 0


async def test_a_bare_host_asks_as_its_session_and_has_no_default_once_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Its one session asks in its own name; with no session live, nothing is followed and no run is shown."""
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript())
    service = WebService(host=host)
    with service.hub.subscribe() as client:
        async with anyio.create_task_group() as group:
            group.start_soon(service.answerer, REQUEST)
            await anyio.wait_all_tasks_blocked()
            asked = json.loads(client.queue.get_nowait())
            assert (asked['type'], asked['session']) == ('question', host.conversation.conversation_id)
            service.answerer.reply(REQUEST.id, None)
    monkeypatch.setattr(service.live, 'listing', list)
    with pytest.raises(LookupError, match='No session is open'):
        service.default()
    with pytest.raises(LookupError, match='No session is open'):
        await service.new()
    assert service.hub.default() is None
    result = await Agent(TestModel(), deps_type=type(None), capabilities=[Observer(service=service)]).run('hi')
    assert result.output == 'success (no tool calls)', 'a run no live session holds is not shown'


def test_a_browser_connecting_while_no_session_is_live(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """As while CLAI exits: no snapshot, only an empty `live` and the orchestrator."""
    host = PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={}, conversation=Transcript())
    service = WebService(host=host)
    monkeypatch.setattr(service.live, 'listing', list)
    client = TestClient(create_app(service=service, token=TOKEN, hosts=HOSTS, static=static_dir(tmp_path)))
    with client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        assert json.loads(socket.receive_text()) == {'type': 'live', 'sessions': []}
        assert json.loads(socket.receive_text())['type'] == 'orchestrator'

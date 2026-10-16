"""The `web` plugin end to end: a shell driven over HTTP, its WebSocket stream, and the access checks."""

import asyncio
import html
import io
import re
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar
from urllib.parse import urlparse

import anyio
import httpx
import httpx2
import pytest
from anyio.from_thread import BlockingPortal, start_blocking_portal
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from rich.console import Console
from starlette.testclient import TestClient, WebSocketTestSession
from starlette.websockets import WebSocketDisconnect
from termflow.themes import PALETTES
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus
from websockets.typing import Origin

from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2._app import _Shell, create_shell  # pyright: ignore[reportPrivateUsage]
from pydantic_clai2.builtin_plugins import web
from pydantic_clai2.builtin_plugins.web import WebPlugin, WebSettings, live, remote as remote_module
from pydantic_clai2.builtin_plugins.web.app import (
    HEADER,
    UNBUILT,
    ASGIApp,
    Guard,
    Message,
    Receive,
    Scope,
    Send,
    create_app,
)
from pydantic_clai2.builtin_plugins.web.protocol import (
    ItemAdded,
    Notice,
    PluginsChanged,
    QuestionAsked,
    QuestionClosed,
    ServerMessage,
    SessionsChanged,
    Snapshot,
    StatusChanged,
    TextDelta,
    ThemeChanged,
    TitleChanged,
    ToolDone,
    TurnEnded,
    TurnStarted,
)
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionEndReason
from pydantic_clai2.ui.rendering import theme as terminal_theme

TOKEN = 'secret-token'
HOSTS = frozenset({'testserver'})


class QuestionModel(TestModel):
    """Asks one question with `ask_user_question`, then answers with text."""

    def gen_tool_args(self, tool_def: ToolDefinition) -> JsonValue:
        return {
            'questions': [
                {'question': 'Which colour?', 'header': 'Colour', 'options': [{'label': 'Red'}, {'label': 'Blue'}]}
            ]
        }


@dataclass
class Web:
    shell: _Shell[None, str]
    plugin: WebPlugin
    client: TestClient
    portal: BlockingPortal
    """The event loop the shell and every request share, as uvicorn and CLAI do."""
    runs: list[Future[SessionEndReason]] = field(default_factory=list[Future[SessionEndReason]])

    def run(self) -> Future[SessionEndReason]:
        """Run the shell's prompt loop on the client's event loop until it reads `/exit`."""
        self.runs.append(self.portal.start_task_soon(self.shell.run))
        return self.runs[-1]

    def post(self, path: str, body: object = None) -> httpx2.Response:
        return self.client.post(path, json=body, headers={HEADER: TOKEN})

    def get(self, path: str) -> httpx2.Response:
        return self.client.get(path, headers={HEADER: TOKEN})


MakeWeb = Callable[..., Web]
"""Builds a headless shell with the `web` plugin loaded, and a test client for its app."""


def static_dir(tmp_path: Path) -> Path:
    static = tmp_path / 'static'
    (static / 'assets').mkdir(parents=True)
    (static / 'index.html').write_text(
        '<html><head><script src="/assets/app.js"></script><link href="/assets/index.css"></head></html>'
    )
    (static / 'assets' / 'app.js').write_text('console.log("hi")')
    return static


@pytest.fixture
def make_web(tmp_path: Path) -> Iterator['MakeWeb']:
    made_webs: list[Web] = []
    portals: list[AbstractContextManager[BlockingPortal]] = []

    def make(
        agent: Agent[None, str], settings: dict[str, JsonValue] | None = None, extra: Sequence[PluginSettings] = ()
    ) -> Web:
        shell = create_shell(
            agent,
            deps=None,
            plugins=(),
            usage_limits=None,
            console=Console(file=io.StringIO(), width=100),
            settings=None,
            store=SettingsStore(tmp_path / 'config.db'),
            builtin_plugins=[
                PluginSettings(
                    id='web',
                    factory='pydantic_clai2.builtin_plugins.web',
                    settings={'open_browser': False, **(settings or {})},
                ),
                *extra,
            ],
            project=ProjectSettings(),
            headless=True,
        )
        portals.append(start_blocking_portal())
        portal = portals[-1].__enter__()
        portal.call(shell.loader.load_all)
        [entry] = [entry for entry in shell.loader.entries() if entry.name == 'web']
        plugin = entry.loaded.plugin if entry.loaded else None
        assert isinstance(plugin, WebPlugin)
        client = TestClient(create_app(remote=plugin.remote, token=TOKEN, hosts=HOSTS, static=static_dir(tmp_path)))
        client.portal = portal
        made = Web(shell=shell, plugin=plugin, client=client, portal=portal)
        made_webs.append(made)
        return made

    yield make
    for made in made_webs:
        for run in made.runs:
            run.cancel()
        made.portal.call(made.shell.loader.close, 'exit')
    for opened in portals:
        opened.__exit__(None, None, None)


WIRE = TypeAdapter[ServerMessage](ServerMessage)
MessageT = TypeVar('MessageT')


def receive_until(socket: WebSocketTestSession, kind: type[MessageT]) -> tuple[list[ServerMessage], MessageT]:
    """Messages from the WebSocket up to and including the first `kind`, which is also returned on its own."""
    received: list[ServerMessage] = []
    while True:
        message = WIRE.validate_json(socket.receive_text())
        received.append(message)
        if isinstance(message, kind):
            return received, message


def test_a_turn_streams_to_the_browser(make_web: 'MakeWeb', monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt entered over HTTP runs in the shell; the browser sees the turn, its text, its tools, and the end."""
    monkeypatch.setattr(remote_module, 'STATUS_INTERVAL', 0)
    tools = Capability[None]()

    @tools.tool_plain
    def lookup(*, query: str) -> str:
        """Look something up."""
        return f'found {query}'

    agent = Agent(
        TestModel(call_tools=['lookup'], custom_output_text='All **done**.'), deps_type=type(None), capabilities=[tools]
    )
    web_ = make_web(agent)
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        snapshot = WIRE.validate_json(socket.receive_text())
        assert isinstance(snapshot, Snapshot) and snapshot.items == [] and not snapshot.running
        running = web_.run()
        assert web_.post('/api/input', {'text': 'look it up'}).status_code == 204
        messages, end = receive_until(socket, TurnEnded)
        [started] = [message.item for message in messages if isinstance(message, TurnStarted)]
        assert started is not None and started.text == 'look it up'
        items = [message.item for message in messages if isinstance(message, ItemAdded)]
        [tool] = [item for item in items if item.kind == 'tool']
        assert (tool.name, tool.state) == ('lookup', 'running')
        [done] = [message for message in messages if isinstance(message, ToolDone)]
        assert (done.id, done.text, done.state) == (tool.id, 'found a', 'done')
        [answer] = [item for item in items if item.kind == 'assistant']
        deltas = [message.text for message in messages if isinstance(message, TextDelta) and message.id == answer.id]
        assert answer.text + ''.join(deltas) == 'All **done**.'
        assert end.outcome == 'completed' and end.usage is not None and end.usage.requests == 2
        # Token counts tick while the turn streams: the estimate grows, and so does each request's usage.
        statuses = [message.status for message in messages if isinstance(message, StatusChanged)]
        streamed = [status.streamed_tokens for status in statuses]
        assert streamed == sorted(streamed) and streamed[-2] > 0
        live = [status.run for status in statuses if status.run is not None]
        assert [usage.requests for usage in live][-1] == 1, 'the second request reports only once the run ends'
        assert live[-1].input_tokens > 0
        settled, _ = receive_until(socket, SessionsChanged)
        final = [message.status for message in settled if isinstance(message, StatusChanged)][-1]
        assert final.run is None and final.output_tokens == end.usage.output_tokens
        # A browser connecting now sees the finished turn as history.
        state = web_.get('/api/state').json()
        assert [item['kind'] for item in state['items']] == ['user', 'tool', 'assistant']
        assert state['items'][-1]['text'] == 'All **done**.'
        assert web_.post('/api/input', {'text': '/cost'}).status_code == 204
        _, notice = receive_until(socket, Notice)
        assert (notice.command, notice.level) == ('/cost', 'info') and notice.text.startswith('Retained history')
        assert web_.post('/api/input', {'text': '/missing'}).status_code == 204
        _, notice = receive_until(socket, Notice)
        assert (notice.level, notice.text) == ('error', 'Unknown command /missing. Use /help.')
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


def test_a_background_fork_is_not_a_turn(make_web: 'MakeWeb') -> None:
    """`/fork` fires the turn moments for its background run; the browser shows only the held conversation's turns."""
    web_ = make_web(Agent(TestModel(call_tools=[], custom_output_text='ok'), deps_type=type(None)))
    shell = web_.shell
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': '/fork look elsewhere'})
        forked, _ = receive_until(socket, Notice)

        async def forks_finished() -> None:
            await asyncio.gather(*(record.task for record in shell.forks.records))

        web_.portal.call(forks_finished)
        assert not web_.plugin.remote.running
        web_.post('/api/input', {'text': 'here'})
        messages, end = receive_until(socket, TurnEnded)
        started = [message for message in [*forked, *messages] if isinstance(message, TurnStarted)]
        assert [message.item.text for message in started if message.item] == ['here']
        assert end.outcome == 'completed'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_switches(make_web: 'MakeWeb') -> None:
    """The browser's toggles flip code mode and speculation through `/set`, and its status follows the settings."""
    web_ = make_web(Agent(TestModel()))
    status = web_.get('/api/state').json()['status']
    assert (status['code_mode'], status['speculation']) == (False, False), 'the tests turn code mode off'
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        assert web_.post('/api/switch', {'key': 'run.speculative_code_mode', 'on': True}).status_code == 204
        assert receive_until(socket, Notice)[1].command == '/set run.speculative_code_mode true'
        assert receive_until(socket, StatusChanged)[1].status.speculation is True
        assert web_.post('/api/switch', {'key': 'run.code_mode', 'on': True}).status_code == 204
        assert receive_until(socket, StatusChanged)[1].status.code_mode is True
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    assert web_.post('/api/switch', {'key': 'display.theme', 'on': True}).status_code == 422, 'only the run switches'


def test_theme(make_web: 'MakeWeb') -> None:
    """The browser paints the `/theme` palette, and hears when a command changes it."""
    web_ = make_web(Agent(TestModel()))
    theme = web_.get('/api/state').json()['theme']
    assert (theme['name'], theme['palette']) == ('default', None)
    assert {'default', 'tokyo_night', 'github_light'} <= set(theme['names'])
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': '/theme tokyo_night'})
        changed = receive_until(socket, ThemeChanged)[1].theme
        assert changed.name == 'tokyo_night'
        assert changed.palette is not None
        assert (changed.palette.bg, len(changed.palette.ansi)) == (PALETTES['tokyo_night'].bg, 16)
        web_.post('/api/input', {'text': '/cost'})
        messages, _ = receive_until(socket, Notice)
        assert not any(isinstance(message, ThemeChanged) for message in messages), 'sent only when it changes'
        web_.post('/api/input', {'text': '/theme default'})
        changed = receive_until(socket, ThemeChanged)[1].theme
        assert (changed.name, changed.palette) == ('default', None), "back to Pydantic's colours"
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_a_repl_cell_from_the_browser(make_web: 'MakeWeb') -> None:
    """`/repl` from the browser runs the user's cell in the session's REPL: shown as theirs, and not a turn."""
    pytest.importorskip('pydantic_monty')
    web_ = make_web(Agent(TestModel()))
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': '/set run.code_mode true'})
        receive_until(socket, Notice)
        web_.post('/api/input', {'text': '/repl x = 6 * 7\nprint(x)'})
        messages, notice = receive_until(socket, Notice)
        assert (notice.command, notice.text) == ('/repl x = 6 * 7\nprint(x)', '42')
        [cell] = [message.item for message in messages if isinstance(message, ItemAdded)]
        assert (cell.name, cell.by_user) == ('run_code', True)
        assert not any(isinstance(message, TurnStarted | TurnEnded) for message in messages)
        assert all(message.status.run is None for message in messages if isinstance(message, StatusChanged))
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_theme_slots_match_the_terminal() -> None:
    """`web-ui/src/palette.ts` paints each role from the same ANSI slot as `ui/rendering/theme.py` does."""
    source = (Path(__file__).parents[2] / 'src/pydantic_clai2/web-ui/src/palette.ts').read_text()
    constants = {name: float(value) for name, value in re.findall(r'^const (\w+) = ([\d.]+)$', source, re.MULTILINE)}
    roles = ('ACCENT', 'ERROR', 'THINKING', 'SUCCESS', 'WARNING', 'INFO')
    terminal = {role: terminal_theme._SLOTS[getattr(terminal_theme, role).removeprefix('bold ')] for role in roles}  # pyright: ignore[reportPrivateUsage]
    assert {role: constants[role] for role in roles} == terminal
    assert constants['DIFF_TINT'] == terminal_theme._DIFF_TINT  # pyright: ignore[reportPrivateUsage]


def test_a_failed_turn(make_web: 'MakeWeb') -> None:
    async def fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError('model unavailable')
        yield  # pragma: no cover

    web_ = make_web(Agent(FunctionModel(stream_function=fail), deps_type=type(None)))
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'try'})
        end = receive_until(socket, TurnEnded)[1]
        assert (end.outcome, end.error, end.usage) == ('failed', 'model unavailable', None)
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_input_the_browser_cannot_send(make_web: 'MakeWeb') -> None:
    web_ = make_web(Agent(TestModel()))
    assert web_.post('/api/input', {'text': '   '}).json() == {'detail': 'Type a prompt or a /command.'}
    response = web_.post('/api/input', {'text': '!ls'})
    assert (response.status_code, response.json()) == (400, {'detail': 'Shell lines (!) run only in the terminal.'})


def test_completion_and_models(make_web: 'MakeWeb') -> None:
    web_ = make_web(Agent(TestModel()))
    [web_command] = [entry for entry in web_.get('/api/complete?text=/we').json() if entry['text'] == 'web']
    assert web_command == {
        'text': 'web',
        'start': 2,
        'display': '/web',
        'meta': 'Open this session in your browser; /web stop ends the web UI',
    }
    assert web_.get('/api/models').json() == []
    assert web_.post('/api/model', {'name': 'test:model'}).status_code == 204


def test_questions_are_answered_from_the_browser(make_web: 'MakeWeb') -> None:
    agent = Agent(QuestionModel(call_tools=['ask_user_question']), deps_type=type(None))
    web_ = make_web(agent)
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'ask me'})
        asked = receive_until(socket, QuestionAsked)[1].questions
        assert [question['id'] for question in web_.get('/api/state').json()['questions']] == [asked.id]
        assert [option.label for option in asked.questions[0].options] == ['Red', 'Blue']
        wrong = web_.post(f'/api/questions/{asked.id}', {'answers': [{'header': 'Colour', 'selected': ['Green']}]})
        assert wrong.status_code == 400
        assert web_.post('/api/questions/unknown', {'answers': None}).status_code == 404
        answered = web_.post(f'/api/questions/{asked.id}', {'answers': [{'header': 'Colour', 'selected': ['Blue']}]})
        assert answered.status_code == 204
        assert receive_until(socket, QuestionClosed)[1].id == asked.id
        assert receive_until(socket, TurnEnded)[1].outcome == 'completed'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    tool = next(item for item in web_.get('/api/state').json()['items'] if item['kind'] == 'tool')
    assert 'Blue' in tool['text']


@pytest.mark.parametrize(
    ('answers', 'expected'), [([{'header': 'Colour', 'custom': 'Teal'}], 'Teal'), (None, 'declined')]
)
def test_a_custom_answer_or_declining(make_web: 'MakeWeb', answers: JsonValue, expected: str) -> None:
    agent = Agent(QuestionModel(call_tools=['ask_user_question']), deps_type=type(None))
    web_ = make_web(agent)
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'ask me'})
        asked = receive_until(socket, QuestionAsked)[1].questions
        assert web_.post(f'/api/questions/{asked.id}', {'answers': answers}).status_code == 204
        receive_until(socket, TurnEnded)
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    [result] = [item['text'] for item in web_.get('/api/state').json()['items'] if item['kind'] == 'tool']
    assert expected in result


def test_interrupt_from_the_browser(make_web: 'MakeWeb') -> None:
    started = anyio.Event()

    tools = Capability[None]()

    @tools.tool_plain
    async def wait() -> str:
        """Wait forever."""
        started.set()
        await anyio.sleep_forever()
        return 'never'  # pragma: no cover

    agent = Agent(TestModel(call_tools=['wait']), deps_type=type(None), capabilities=[tools])
    web_ = make_web(agent)
    assert web_.post('/api/interrupt').json() == {'interrupted': False}
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'wait'})
        web_.portal.call(started.wait)
        # A browser connecting mid-turn sees the prompt and the running tool.
        state = web_.get('/api/state').json()
        assert state['running'] is True
        assert [(item['kind'], item['state']) for item in state['items']] == [('user', 'done'), ('tool', 'running')]
        assert web_.post('/api/interrupt').json() == {'interrupted': True}
        assert receive_until(socket, TurnEnded)[1].outcome == 'cancelled'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_session_actions(make_web: 'MakeWeb', tmp_path: Path) -> None:
    """The sidebar lists, reads, renames, forks, deletes, opens, and starts sessions in the store `/resume` uses."""
    web_ = make_web(Agent(TestModel(call_tools=[], custom_output_text='ok'), deps_type=type(None)))
    shell = web_.shell
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'first'})
        receive_until(socket, TurnEnded)
        # The turn's outcome is saved after its run ends; the sidebar is told once it is.
        receive_until(socket, SessionsChanged)
        first = shell.session.conversation_id
        elsewhere = web_.portal.call(
            lambda: shell.sessions.store.save(
                summary=ConversationSummary(workspace='/somewhere/else', title='Elsewhere'), messages=[]
            )
        )

        projects = web_.get('/api/sessions').json()
        assert projects[0]['current'] is True
        assert [(entry['title'], entry['state'], entry['current']) for entry in projects[0]['sessions']] == [
            ('first', 'completed', True)
        ]
        assert [project['name'] for project in projects[1:]] == ['else']
        assert web_.get('/api/sessions?q=Elsewhere').json()[0]['sessions'][0]['id'] == elsewhere.id

        transcript = web_.get(f'/api/sessions/{elsewhere.id}').json()
        assert (transcript['title'], transcript['openable'], transcript['items']) == ('Elsewhere', False, [])
        assert web_.get('/api/sessions/missing').status_code == 404

        assert web_.post(f'/api/sessions/{first}/rename', {'title': 'Renamed'}).status_code == 204
        assert receive_until(socket, TitleChanged)[1].title == 'Renamed'
        receive_until(socket, SessionsChanged)
        assert shell.session.title == 'Renamed'
        assert web_.post(f'/api/sessions/{first}/rename', {'title': ' '}).status_code == 400

        assert web_.client.delete(f'/api/sessions/{first}', headers={HEADER: TOKEN}).status_code == 400
        assert web_.client.delete(f'/api/sessions/{elsewhere.id}', headers={HEADER: TOKEN}).status_code == 204

        assert web_.post('/api/sessions/new').status_code == 204
        snapshot = receive_until(socket, Snapshot)[1]
        assert snapshot.session.id != first and snapshot.items == []

        assert web_.post(f'/api/sessions/{first}/open').status_code == 204
        snapshot = receive_until(socket, Snapshot)[1]
        assert (snapshot.session.id, snapshot.session.title) == (first, 'Renamed')

        copy = web_.post(f'/api/sessions/{first}/fork').json()['id']
        snapshot = receive_until(socket, Snapshot)[1]
        assert (snapshot.session.id, snapshot.session.title) == (copy, 'Renamed (fork)')
        assert [item.text for item in snapshot.items] == ['first', 'ok']

        other = web_.portal.call(
            lambda: shell.sessions.store.save(
                summary=ConversationSummary(workspace='/somewhere/else', title='Other'), messages=[]
            )
        )
        forked = web_.post(f'/api/sessions/{other.id}/fork').json()['id']
        assert shell.session.conversation_id == copy, 'a copy saved in another directory is not opened'
        assert forked != other.id
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_running_sessions_in_the_sidebar(make_web: 'MakeWeb') -> None:
    """A turn this CLAI runs shows as running; one another live process runs, as busy; a dead one, as cancelled."""
    web_ = make_web(Agent(TestModel()))
    store = web_.shell.sessions.store
    import os

    for title, pid in (('Live', os.getpid()), ('Dead', 2**22 + 12345)):
        web_.portal.call(
            lambda title=title, pid=pid: store.save(
                summary=ConversationSummary(workspace='/x', title=title, outcome='running', owner_pid=pid),
                messages=[],
            )
        )
    sessions = {entry['title']: entry['state'] for entry in web_.get('/api/sessions').json()[-1]['sessions']}
    assert sessions == {'Live': 'busy', 'Dead': 'cancelled'}
    web_.plugin.remote.running = True
    web_.portal.call(web_.shell.session.commit_messages, list[ModelMessage]())
    current = web_.get('/api/sessions').json()[0]['sessions'][0]
    assert (current['current'], current['state']) == (True, 'running')


def test_a_shell_without_saved_sessions(make_web: 'MakeWeb') -> None:
    web_ = make_web(Agent(TestModel()))
    web_.plugin.host.sessions = None
    assert web_.get('/api/sessions').status_code == 404


PANEL_PLUGINS = """
\"\"\"Plugins for the plugins panel.\"\"\"

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pydantic_clai2.plugins import Plugin


class SampleSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    mode: Literal['fast', 'slow'] = 'fast'
    limit: int = Field(default=3, ge=1)
    label: str = 'hi'


class Sample(Plugin[SampleSettings]):
    \"\"\"A plugin with settings and a menu of its own.\"\"\"

    def __init__(self, host, settings):
        if settings.label == 'boom':
            raise RuntimeError('boom refused')
        super().__init__(host, settings)

    async def configure(self) -> str:
        return 'Sample menu closed.'


class Plain(Plugin):
    \"\"\"A plugin without settings.\"\"\"


class OpaqueSettings(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    hook: Callable[[], None] | None = None


class Opaque(Plugin[OpaqueSettings]):
    \"\"\"Settings JSON Schema cannot describe.\"\"\"
"""


def panel_plugins(tmp_path: Path) -> list[PluginSettings]:
    """`sample`, `plain`, and `opaque` on; `broken`, which fails to import, off; `ask_user`, which `web` includes."""
    path = tmp_path / 'panel.py'
    path.write_text(PANEL_PLUGINS)
    broken = tmp_path / 'broken.py'
    broken.write_text("raise RuntimeError('cannot import')")
    return [
        *(
            PluginSettings(id=name.lower(), factory=f'panel:{name}', path=str(path))
            for name in ('Sample', 'Plain', 'Opaque')
        ),
        PluginSettings(id='broken', factory='broken', path=str(broken), enabled=False),
        PluginSettings(id='ask_user', factory='pydantic_clai2.builtin_plugins.ask_user_menu'),
    ]


def plugin_rows(web_: Web) -> dict[str, dict[str, JsonValue]]:
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(web_.get('/api/plugins').json())
    return {str(row['name']): row for row in rows}


def loaded_plugin(web_: Web, name: str) -> object:
    [entry] = [entry for entry in web_.shell.loader.entries() if entry.name == name]
    return entry.loaded.plugin if entry.loaded is not None else None


def test_the_plugins_panel_lists_every_plugin_with_a_form_for_its_settings(make_web: 'MakeWeb', tmp_path: Path) -> None:
    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    rows = plugin_rows(web_)
    assert sorted(rows) == ['ask_user', 'broken', 'opaque', 'plain', 'sample', 'web']
    sample = rows['sample']
    assert (sample['description'], sample['status'], sample['origin'], sample['source']) == (
        'A plugin with settings and a menu of its own.',
        'on',
        'built-in',
        str(tmp_path / 'panel.py'),
    )
    # A headless CLAI has no terminal for the plugin's own menu to open in.
    assert (sample['has_menu'], sample['terminal'], sample['serving'], sample['provides']) == (
        True,
        False,
        False,
        'nothing yet',
    )
    assert sample['form'] == {
        'json_schema': SampleSettings_schema(),
        'values': {'mode': 'fast', 'limit': 3, 'label': 'hi'},
    }
    # No settings, settings JSON Schema cannot describe, and a plugin that is off: no form.
    assert (rows['plain']['form'], rows['plain']['has_menu']) == (None, False)
    assert (rows['opaque']['status'], rows['opaque']['form']) == ('on', None)
    assert (rows['broken']['status'], rows['broken']['form']) == ('off', None)
    assert (rows['ask_user']['status'], rows['ask_user']['included_in']) == ('included', 'web')
    web_row = rows['web']
    assert web_row['serving'] is True
    assert isinstance(web_row['form'], dict) and web_row['form']['values'] == {
        'host': '127.0.0.1',
        'port': 0,
        'open_browser': False,
        'ask_user': True,
    }


def SampleSettings_schema() -> JsonValue:
    return {
        'additionalProperties': False,
        'properties': {
            'mode': {'default': 'fast', 'enum': ['fast', 'slow'], 'title': 'Mode', 'type': 'string'},
            'limit': {'default': 3, 'minimum': 1, 'title': 'Limit', 'type': 'integer'},
            'label': {'default': 'hi', 'title': 'Label', 'type': 'string'},
        },
        'title': 'SampleSettings',
        'type': 'object',
    }


def test_settings_from_the_browser_are_validated_saved_and_applied(make_web: 'MakeWeb', tmp_path: Path) -> None:
    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    first = loaded_plugin(web_, 'sample')
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        assert web_.post('/api/plugins/sample/settings', {'changes': {'limit': 5, 'mode': 'slow'}}).status_code == 204
        receive_until(socket, PluginsChanged)
    # Saved, then loaded again with them, as the plugin's own menu does.
    form = plugin_rows(web_)['sample']['form']
    assert isinstance(form, dict) and form['values'] == {'mode': 'slow', 'limit': 5, 'label': 'hi'}
    second = loaded_plugin(web_, 'sample')
    assert second is not first
    assert [saved.settings for saved in web_.shell.loader._store.plugins() if saved.id == 'sample'] == [  # pyright: ignore[reportPrivateUsage]
        {'mode': 'slow', 'limit': 5, 'label': 'hi'}
    ]
    # Nothing changed, nothing reloaded.
    assert web_.post('/api/plugins/sample/settings', {'changes': {'limit': 5}}).status_code == 204
    assert loaded_plugin(web_, 'sample') is second

    rejected = web_.post('/api/plugins/sample/settings', {'changes': {'limit': 0, 'colour': 'red'}})
    assert rejected.status_code == 422
    assert rejected.json()['detail'] == {
        'message': 'Some settings are not valid.',
        'errors': {'limit': 'Input should be greater than or equal to 1', 'colour': 'Extra inputs are not permitted'},
    }
    assert web_.post('/api/plugins/sample/settings', {'reset': ['limit', 'unknown']}).status_code == 204
    form = plugin_rows(web_)['sample']['form']
    assert isinstance(form, dict) and form['values'] == {'mode': 'slow', 'limit': 3, 'label': 'hi'}

    # Settings the plugin refuses when it loads leave it failed, as its own menu would.
    refused = web_.post('/api/plugins/sample/settings', {'changes': {'label': 'boom'}})
    assert refused.status_code == 400 and 'boom refused' in refused.json()['detail']
    assert plugin_rows(web_)['sample']['status'] == 'failed'
    off = web_.post('/api/plugins/sample/settings', {'changes': {'label': 'fine'}})
    assert (off.status_code, off.json()['detail']) == (400, 'Turn sample on to change its settings.')

    none = web_.post('/api/plugins/plain/settings', {'changes': {'x': 1}})
    assert (none.status_code, none.json()['detail']) == (
        400,
        'plain has no settings form; replace its declaration with /plugins add.',
    )
    assert web_.post('/api/plugins/missing/settings', {'changes': {}}).status_code == 404


def test_settings_built_from_a_stale_listing_are_refused(
    make_web: 'MakeWeb', tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload re-imports the plugin's module between the listing and the save, so its model is another class."""

    class Stale(BaseModel):
        pass

    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    loader = web_.shell.loader
    listing = loader.listing
    monkeypatch.setattr(
        loader,
        'listing',
        lambda: [replace(info, settings_model=Stale) if info.settings else info for info in listing()],
    )
    stale = web_.post('/api/plugins/sample/settings', {'changes': {'limit': 5}})
    assert (stale.status_code, stale.json()['detail']) == (400, 'sample changed; reload the panel and try again.')


def test_plugins_turn_on_off_and_reload_from_the_browser(make_web: 'MakeWeb', tmp_path: Path) -> None:
    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        assert web_.post('/api/plugins/sample/disable').status_code == 204
        receive_until(socket, PluginsChanged)
    assert (plugin_rows(web_)['sample']['status'], loaded_plugin(web_, 'sample')) == ('off', None)
    assert web_.post('/api/plugins/sample/enable').status_code == 204
    enabled = loaded_plugin(web_, 'sample')
    assert enabled is not None
    assert web_.post('/api/plugins/sample/reload').status_code == 204
    assert loaded_plugin(web_, 'sample') not in (None, enabled)

    # The page's own plugin stays on: turning it off or reloading it would end this page.
    for action in ('disable', 'reload'):
        refused = web_.post(f'/api/plugins/web/{action}')
        assert (refused.status_code, refused.json()['detail']) == (400, remote_module.SERVING)
    refused = web_.post('/api/plugins/web/settings', {'changes': {'open_browser': True}})
    assert (refused.status_code, refused.json()['detail']) == (400, remote_module.SERVING)
    assert plugin_rows(web_)['web']['status'] == 'on'

    failed = web_.post('/api/plugins/broken/enable')
    assert failed.status_code == 400 and 'cannot import' in failed.json()['detail']
    broken = plugin_rows(web_)['broken']
    assert broken['status'] == 'failed' and 'cannot import' in str(broken['error'])
    assert web_.post('/api/plugins/sample/explode').status_code == 422


def test_a_plugins_own_settings_menu_opens_in_the_terminal(make_web: 'MakeWeb', tmp_path: Path) -> None:
    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        web_.run()
        assert web_.post('/api/plugins/sample/menu').status_code == 204
        notice = receive_until(socket, Notice)[1]
        assert (notice.text, notice.command) == ('Sample menu closed.', '/plugins configure sample')
        receive_until(socket, PluginsChanged)
    assert web_.post('/api/plugins/missing/menu').status_code == 404


def test_a_host_without_plugins(make_web: 'MakeWeb') -> None:
    web_ = make_web(Agent(TestModel()))
    web_.plugin.host.plugins = None
    assert web_.get('/api/plugins').status_code == 404


def test_access_checks(make_web: 'MakeWeb') -> None:
    """Every request needs this launch's token, for the address the server listens on."""
    web_ = make_web(Agent(TestModel()))
    client = web_.client
    assert client.get('/api/state').status_code == 403
    assert client.get('/api/state', headers={HEADER: 'wrong'}).status_code == 403
    assert client.get(f'/api/state?token={TOKEN}').status_code == 403, 'the API takes only the header'
    assert client.get('/api/state', headers={HEADER: TOKEN, 'host': 'evil.example'}).status_code == 403
    page = client.get('/')
    assert (page.status_code, page.text) == (
        403,
        "Open the address CLAI printed: it carries this launch's access token.",
    )
    assert client.get('/assets/app.js').status_code == 403
    assert client.get(f'/?token={TOKEN}', headers={'host': 'evil.example'}).status_code == 403

    page = client.get(f'/?token={TOKEN}')
    assert f'<meta name="clai-token" content="{TOKEN}"></head>' in page.text
    assert f'src="/assets/app.js?token={TOKEN}"' in page.text and f'href="/assets/index.css?token={TOKEN}"' in page.text
    assert "frame-ancestors 'none'" in page.headers['content-security-policy']
    assert 'set-cookie' not in page.headers, 'cookies reach every port of the host'
    assert client.get(f'/assets/app.js?token={TOKEN}').text == 'console.log("hi")'

    for path, headers in (
        ('/ws', {'origin': 'http://testserver'}),
        (f'/ws?token={TOKEN}', {}),
        (f'/ws?token={TOKEN}', {'origin': 'http://evil.example'}),
    ):
        with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(path, headers=headers):
            pass  # pragma: no cover
        assert refused.value.code == 1008


def test_a_checkout_without_a_built_ui(tmp_path: Path) -> None:
    remote = WebPlugin.from_host(PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={})).remote
    with TestClient(create_app(remote=remote, token=TOKEN, hosts=HOSTS, static=tmp_path / 'missing')) as client:
        page = client.get(f'/?token={TOKEN}')
        assert (page.status_code, page.text) == (503, UNBUILT)


def test_a_build_without_assets(tmp_path: Path) -> None:
    static = tmp_path / 'bare'
    static.mkdir()
    (static / 'index.html').write_text('<html><head></head></html>')
    remote = WebPlugin.from_host(PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={})).remote
    with TestClient(create_app(remote=remote, token=TOKEN, hosts=HOSTS, static=static)) as client:
        assert client.get(f'/assets/app.js?token={TOKEN}').status_code == 404
        assert client.post('/api/input', json={'text': 'hi'}, headers={HEADER: TOKEN}).json() == {
            'detail': "Plugin 'web' has no prompt to enter input at."
        }
        state = client.get('/api/state', headers={HEADER: TOKEN}).json()
        assert (state['status']['code_mode'], state['status']['speculation']) == (None, None), 'no settings to read'
        assert state['theme']['name'] == 'default'


async def test_lifespan_passes_through() -> None:
    seen: list[str] = []

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(str(scope['type']))

    guard = Guard(inner, token=TOKEN, hosts=HOSTS)

    async def receive() -> Message:  # pragma: no cover
        return {}

    async def send(message: Message) -> None:  # pragma: no cover
        pass

    await guard({'type': 'lifespan'}, receive, send)
    assert seen == ['lifespan']


def web_shell(
    tmp_path: Path, settings: dict[str, JsonValue] | None = None, *, terminal: bool = True
) -> '_Shell[None, str]':
    return create_shell(
        Agent(TestModel()),
        deps=None,
        plugins=(),
        usage_limits=None,
        console=Console(file=io.StringIO(), force_terminal=terminal),
        settings=None,
        store=SettingsStore(tmp_path / 'config.db'),
        builtin_plugins=[
            PluginSettings(id='web', factory='pydantic_clai2.builtin_plugins.web', settings=settings or {})
        ],
        project=ProjectSettings(),
        headless=True,
    )


async def test_web_command_serves_on_a_free_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`/web` serves on the shell's loop and opens it through a private page; `stop` ends it, and so does unloading."""
    opened: list[str] = []
    loop_thread = threading.get_ident()

    def open_browser(url: str) -> bool:
        assert threading.get_ident() != loop_thread, 'a foreground `$BROWSER` command would block the loop'
        opened.append(url)
        return True

    monkeypatch.setattr(web, 'open_browser', open_browser)
    shell = web_shell(tmp_path)
    await shell.loader.load_all()
    assert await shell.commands.execute_async('/web status') == 'The web UI is not running. Start it with /web.'
    assert await shell.commands.execute_async('/web stop') == 'The web UI is not running.'
    started = await shell.commands.execute_async('/web')
    url = started.removeprefix('Web UI running at ')
    assert url.startswith('http://127.0.0.1:')
    [launcher] = [Path(urlparse(page).path) for page in opened]
    assert launcher.stat().st_mode & 0o077 == 0, 'only you can read the token'
    assert html.escape(url) in launcher.read_text()
    assert await shell.commands.execute_async('/web start') == f'Web UI already running at {url}'
    assert (await shell.commands.execute_async('/web status')).endswith('with 0 browser(s) connected.')
    base, token = url.split('/?token=')
    port = urlparse(base).port
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(), trust_env=False) as client:
        assert (await client.get(url)).status_code != 403
        assert (await client.get(url, headers={'host': f'attacker.example:{port}'})).status_code == 403
    async with connect(f'ws://127.0.0.1:{port}/ws?token={token}', origin=Origin(base)) as socket:
        snapshot = WIRE.validate_json(await socket.recv())
        assert isinstance(snapshot, Snapshot)
    with pytest.raises(InvalidStatus):
        async with connect(f'ws://127.0.0.1:{port}/ws?token={token}', origin=Origin('http://attacker.example')):
            pass  # pragma: no cover
    with pytest.raises(ValueError, match='Usage: /web'):
        await shell.commands.execute_async('/web open now')
    assert await shell.commands.execute_async('/web stop') == 'Web UI stopped.'
    assert not launcher.exists()
    await shell.commands.execute_async('/web')
    await shell.loader.close('exit')


async def test_stopping_the_only_interface_is_refused(tmp_path: Path) -> None:
    """With no terminal prompt, as under `clai2 --web`, stopping the web UI would leave nothing to type at."""
    shell = web_shell(tmp_path, {'open_browser': False}, terminal=False)
    await shell.loader.load_all()
    await shell.commands.execute_async('/web')
    with pytest.raises(ValueError, match='only way to drive this CLAI'):
        await shell.commands.execute_async('/web stop')
    await shell.loader.close('exit')


async def test_the_default_port_admits_bare_host_names(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On port 80, browsers send `Host: 127.0.0.1` without the port, so that form is admitted too."""
    served: list[frozenset[str]] = []

    class Served:
        port = 80

        @classmethod
        def serve(cls, app: Callable[[int], ASGIApp], *, host: str, port: int) -> 'Served':
            guard = app(cls.port)
            assert isinstance(guard, Guard)
            served.append(guard.hosts)
            return cls()

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(web, 'WebServer', Served)
    shell = web_shell(tmp_path, {'port': 80, 'open_browser': False})
    await shell.loader.load_all()
    assert (await shell.commands.execute_async('/web')).startswith('Web UI running at http://127.0.0.1:80/?token=')
    assert {'127.0.0.1', 'localhost', '[::1]', '127.0.0.1:80'} <= served[0]
    await shell.loader.close('exit')


async def test_a_server_beyond_loopback_warns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    served: list[tuple[str, frozenset[str]]] = []

    class Served:
        port = 4321

        @classmethod
        def serve(cls, app: Callable[[int], ASGIApp], *, host: str, port: int) -> 'Served':
            guard = app(cls.port)
            assert isinstance(guard, Guard)
            served.append((host, guard.hosts))
            return cls()

        async def stop(self) -> None:
            pass

    monkeypatch.setattr(web, 'WebServer', Served)
    shell = web_shell(tmp_path, {'host': '[fd00::5]', 'open_browser': False, 'ask_user': False})
    await shell.loader.load_all()
    started = await shell.commands.execute_async('/web')
    assert started.startswith('Web UI running at http://[fd00::5]:4321/?token=')
    assert started.endswith('Other machines can reach it: keep the link private.')
    assert served == [('fd00::5', frozenset({'127.0.0.1:4321', 'localhost:4321', '[::1]:4321', '[fd00::5]:4321'}))]
    await shell.loader.close('exit')


@pytest.mark.parametrize('host', ['0.0.0.0', '::', '[::]', '::0', '0:0:0:0:0:0:0:0', ' '])
def test_every_interface_is_refused(host: str) -> None:
    with pytest.raises(ValidationError, match='listen on one address'):
        WebSettings(host=host)


def test_a_host_name_is_kept() -> None:
    assert WebSettings(host=' localhost ').host == 'localhost'


def test_a_falling_behind_browser_reconnects(make_web: 'MakeWeb', monkeypatch: pytest.MonkeyPatch) -> None:
    """A browser that stops reading is closed with "try again" rather than buffering the whole session."""
    monkeypatch.setattr(live, 'QUEUE_LIMIT', 2)
    web_ = make_web(Agent(TestModel()))
    hub = web_.plugin.hub
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()

        async def flood() -> None:
            # All at once, on the loop, so the WebSocket cannot forward any in between.
            for text in ('one', 'two', 'three'):
                hub.publish(Notice(text=text))

        web_.portal.call(flood)
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_text()
        assert closed.value.code == 1013
    assert hub.clients == 0

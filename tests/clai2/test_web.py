"""The `web` plugin end to end: a shell driven over HTTP, its WebSocket stream, and the access checks."""

import asyncio
import base64
import gzip
import hashlib
import html
import io
import json
import os
import re
import shutil
import sys
import threading
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager, asynccontextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar
from urllib.parse import quote, urlparse

import anyio
import httpx
import httpx2
import pytest
from anyio.from_thread import BlockingPortal, start_blocking_portal
from PIL import Image
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
from pydantic_ai.messages import ModelMessage, ModelRequest, UserContent, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.builtin_plugins import web
from pydantic_clai2.builtin_plugins.web import (
    WebPlugin,
    WebSettings,
    app as app_module,
    channel as channel_module,
    live,
    service as service_module,
)
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
    PROTOCOL,
    DialogClosed,
    DialogOpened,
    DialogShown,
    ForkEntry,
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
    image_id,
)
from pydantic_clai2.commands import Command
from pydantic_clai2.config import PluginSettings, Settings, resolve_settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.plugins import PluginHost, SessionEndReason
from pydantic_clai2.runtime import image_codec
from pydantic_clai2.runtime.image_codec import decode_images
from pydantic_clai2.runtime.project_identity import git_checkout, project_identity, worktree_roots
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.runtime.session_workspace import SessionWorkspace
from pydantic_clai2.ui.menus.menu_worker import require_terminal
from pydantic_clai2.ui.rendering import theme as terminal_theme
from tests.clai2.test_session_browser import git, repository
from tests.clai2.test_tasks import task

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
    shell: SessionRuntime[None, str]
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
        client = TestClient(create_app(service=plugin.service, token=TOKEN, hosts=HOSTS, static=static_dir(tmp_path)))
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
    monkeypatch.setattr(channel_module, 'STATUS_INTERVAL', 0)
    tools = Capability[None]()

    @tools.tool_plain
    def lookup(*, query: str) -> str:
        """Look something up."""
        return f'found {query}'

    agent = Agent(
        TestModel(call_tools=['lookup'], custom_output_text='All **done**.'), deps_type=type(None), capabilities=[tools]
    )
    web_ = make_web(agent)
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        snapshot = WIRE.validate_json(socket.receive_text())
        assert isinstance(snapshot, Snapshot) and snapshot.items == [] and not snapshot.running
        running = web_.run()
        assert web_.post('/api/input', {'text': 'look it up'}).json() == {'menu': None}
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
        assert web_.post('/api/input', {'text': '/cost'}).status_code == 200
        _, notice = receive_until(socket, Notice)
        assert (notice.command, notice.level) == ('/cost', 'info') and notice.text.startswith('Retained history')
        assert web_.post('/api/input', {'text': '/missing'}).status_code == 200
        _, notice = receive_until(socket, Notice)
        assert (notice.level, notice.text) == ('error', 'Unknown command /missing. Use /help.')
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


def test_a_background_fork_is_not_a_turn(make_web: 'MakeWeb') -> None:
    """`/fork` fires the turn moments for its background run; the browser shows only the held conversation's turns."""
    web_ = make_web(Agent(TestModel(call_tools=[], custom_output_text='ok'), deps_type=type(None)))
    shell = web_.shell
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': '/fork look elsewhere'})
        forked, _ = receive_until(socket, Notice)

        async def forks_finished() -> None:
            await asyncio.gather(*(record.task for record in shell.forks.records))

        web_.portal.call(forks_finished)
        assert not web_.plugin.service.default().running
        web_.post('/api/input', {'text': 'here'})
        messages, end = receive_until(socket, TurnEnded)
        started = [message for message in [*forked, *messages] if isinstance(message, TurnStarted)]
        assert [message.item.text for message in started if message.item] == ['here']
        assert end.outcome == 'completed'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_fork_results_and_the_forks_table_reach_the_browser(make_web: 'MakeWeb') -> None:
    """What the terminal prints for `/fork` and `/forks`, the answer and the table, reaches the page as notices."""
    web_ = make_web(Agent(TestModel(call_tools=[], custom_output_text='the fork answer'), deps_type=type(None)))
    shell = web_.shell
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': '/forks'})
        _, empty = receive_until(socket, Notice)
        assert (empty.text, empty.forks) == ('No forks yet. Start one with /fork [@model] PROMPT.', None)
        web_.post('/api/input', {'text': '/fork look elsewhere'})
        # The fork may finish before its command's notice is sent.
        notices = [receive_until(socket, Notice)[1], receive_until(socket, Notice)[1]]
        [started] = [notice for notice in notices if notice.fork is None]
        [ended] = [notice for notice in notices if notice.fork is not None]
        assert started.text.startswith('fork #1 (agent default) started in the background.')
        [record] = shell.forks.records
        assert record.session_id is not None
        assert (ended.text, ended.level, ended.command) == ('the fork answer', 'info', None)
        fork = ended.fork
        assert fork == ForkEntry(
            id=1,
            model='agent default',
            prompt='look elsewhere',
            status='done',
            elapsed=record.elapsed or 0,
            session_id=record.session_id,
        )
        web_.post('/api/input', {'text': '/forks'})
        _, table = receive_until(socket, Notice)
        assert (table.command, table.text, table.forks) == ('/forks', '1 done', [fork])
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_switches(make_web: 'MakeWeb') -> None:
    """The browser's toggles flip code mode and speculation through `/set`, and its status follows the settings."""
    web_ = make_web(Agent(TestModel()))
    status = web_.get('/api/state').json()['status']
    assert (status['code_mode'], status['speculation']) == (False, False), 'the tests turn code mode off'
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
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
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
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


def test_theme_changes_while_a_turn_runs(make_web: 'MakeWeb') -> None:
    """`/theme NAME` from the browser's picker applies at once, not after the running turn."""
    started, release = anyio.Event(), anyio.Event()
    tools = Capability[None]()

    @tools.tool_plain
    async def wait() -> str:
        """Wait to be released."""
        started.set()
        await release.wait()
        return 'released'

    web_ = make_web(Agent(TestModel(call_tools=['wait']), deps_type=type(None), capabilities=[tools]))
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'wait'})
        web_.portal.call(started.wait)
        assert web_.post('/api/input', {'text': '/theme tokyo_night'}).json() == {'menu': None}
        web_.portal.call(anyio.wait_all_tasks_blocked)
        state = web_.get('/api/state').json()
        assert (state['running'], state['theme']['name']) == (True, 'tokyo_night')
        messages, changed = receive_until(socket, ThemeChanged)
        assert changed.theme.name == 'tokyo_night'
        assert not any(isinstance(message, TurnEnded) for message in messages)
        web_.portal.call(release.set)
        assert receive_until(socket, TurnEnded)[1].outcome == 'completed'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_a_monty_cell_from_the_browser(make_web: 'MakeWeb') -> None:
    """`/monty` from the browser runs the user's cell in the session's Monty sandbox: shown as theirs, and not a turn."""
    pytest.importorskip('pydantic_monty')
    web_ = make_web(Agent(TestModel()))
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': '/set run.code_mode true'})
        receive_until(socket, Notice)
        web_.post('/api/input', {'text': '/monty x = 6 * 7\nprint(x)'})
        messages, notice = receive_until(socket, Notice)
        assert (notice.command, notice.text) == ('/monty x = 6 * 7\nprint(x)', '42')
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
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'try'})
        end = receive_until(socket, TurnEnded)[1]
        assert (end.outcome, end.error, end.usage) == ('failed', 'model unavailable', None)
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_input_the_browser_cannot_send(make_web: 'MakeWeb', monkeypatch: pytest.MonkeyPatch) -> None:
    web_ = make_web(Agent(TestModel()))
    assert web_.post('/api/input', {'text': '   '}).json() == {'detail': 'Type a prompt or a /command.'}
    response = web_.post('/api/input', {'text': '!ls'})
    assert (response.status_code, response.json()) == (400, {'detail': 'Shell lines (!) run only in the terminal.'})
    response = web_.post('/api/input', {'text': '/model', 'images': [encoded(png('red'))]})
    assert (response.status_code, response.json()) == (
        400,
        {'detail': 'Images go with a prompt, not with a /command or ! line.'},
    ), 'not even a command whose menu the page would show'
    response = web_.post('/api/input', {'text': 'look', 'images': [encoded(b'not an image')]})
    assert response.status_code == 400 and response.json()['detail'].startswith('Not an image CLAI can read')
    assert web_.shell.images.pending == {}
    response = web_.post('/api/input', {'text': 'look', 'images': ['not base64!']})
    assert (response.status_code, response.json()) == (400, {'detail': 'Images must be base64-encoded.'})
    # Images waiting in the terminal's draft, which the page does not show, leave no room.
    waiting = web_.shell.images.attach(decode_images([encoded(png('red'))])).strip()
    [blue] = decode_images([encoded(png('blue'))])
    monkeypatch.setattr(
        image_codec, 'MAX_PENDING_BYTES', len(web_.shell.images.pending[waiting].data) + len(blue.data) - 1
    )
    response = web_.post('/api/input', {'text': 'look', 'images': [encoded(png('blue'))]})
    assert (response.status_code, response.json()) == (
        400,
        {'detail': "Images waiting in this session's prompt leave no room for these (32 MiB limit)."},
    )
    assert list(web_.shell.images.pending) == [waiting]
    with pytest.raises(ValueError, match='send fewer at a time'):
        web_.shell.send_input('look', images=[blue, blue, blue])


def test_a_terminal_menu_or_prompt_fails_without_a_terminal(
    make_web: 'MakeWeb', monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no terminal anywhere, a command that would read the terminal's keys fails, naming an alternative.

    A menu would read EOF from stdin and close as cancelled, saying nothing useful, as under `clai2 --web` from a
    script; that is so whether the command waits for its turn or opens over a running one.
    """
    monkeypatch.setattr(sys, 'stdin', None)  # As Python leaves it when CLAI starts with stdin closed.
    started, release = anyio.Event(), anyio.Event()
    tools = Capability[None]()

    @tools.tool_plain
    async def wait() -> str:
        """Wait to be released."""
        started.set()
        await release.wait()
        return 'released'

    web_ = make_web(Agent(TestModel(call_tools=['wait']), deps_type=type(None), capabilities=[tools]))
    assert web_.shell.terminal is None
    needs = 'opens a terminal menu or prompt, and this CLAI has no terminal: enter it in CLAI in a terminal'

    def notices(texts: Sequence[str]) -> list[tuple[str, str, str | None]]:
        received: list[tuple[str, str, str | None]] = []
        for text in texts:
            assert web_.post('/api/input', {'text': text}).json() == {'menu': None}
            notice = receive_until(socket, Notice)[1]
            received.append((notice.level, notice.text, notice.command))
        return received

    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        assert notices(('/model add', '/model settings', '/chain', '/set api_key', '/resume claude')) == [
            ('error', f'/model add {needs}, or use /model in this page.', '/model add'),
            ('error', f'/model settings {needs}, or use /model in this page.', '/model settings'),
            ('error', f'/chain {needs}.', '/chain'),
            ('error', f'/set api_key {needs}, or use /keys in this page.', '/set api_key'),
            ('error', f'/resume claude {needs}.', '/resume claude'),
        ]
        web_.post('/api/input', {'text': 'wait'})
        web_.portal.call(started.wait)
        # `during_turn` commands open over the running turn instead of queueing.
        assert notices(('/model add', '/chain')) == [
            ('error', f'/model add {needs}, or use /model in this page.', '/model add'),
            ('error', f'/chain {needs}.', '/chain'),
        ]
        web_.portal.call(release.set)
        assert receive_until(socket, TurnEnded)[1].outcome == 'completed'
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


class _Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_a_terminal_menu_opens_in_the_terminal_that_runs_clai(
    make_web: 'MakeWeb', monkeypatch: pytest.MonkeyPatch
) -> None:
    """`clai2 --web` started in a terminal shows no session there, but its menus still open in it, as before."""
    monkeypatch.setattr(sys, 'stdin', _Terminal())
    web_ = make_web(Agent(TestModel()))
    assert web_.shell.terminal is None

    def probe(args: list[str]) -> str:
        require_terminal()
        return 'A menu would open here.'

    web_.shell.commands.register(Command(name='probe', description='Open a terminal menu', handler=probe))
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': '/probe'})
        notice = receive_until(socket, Notice)[1]
        assert (notice.level, notice.text) == ('info', 'A menu would open here.')
        web_.post('/api/input', {'text': '/exit'})
        assert running.result(timeout=10) == 'exit'


def png(color: str) -> bytes:
    output = io.BytesIO()
    Image.new('RGB', (3, 2), color=color).save(output, format='PNG')
    return output.getvalue()


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode()


def test_images_pasted_in_the_browser_go_with_the_prompt(make_web: 'MakeWeb') -> None:
    """Images sent with a prompt reach the model after its text, also queued behind a running turn, also alone."""
    started, release = anyio.Event(), anyio.Event()
    prompts: list[str | Sequence[UserContent]] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        prompts.extend(part.content for part in request.parts if isinstance(part, UserPromptPart))
        if len(prompts) == 1:
            started.set()
            await release.wait()
        yield 'seen'

    web_ = make_web(Agent(FunctionModel(stream_function=stream)))
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'first'})
        web_.portal.call(started.wait)
        # Queued in the inbox of a session no editor reads, until the running turn ends.
        queued = web_.post('/api/input', {'text': 'what is this?', 'images': [encoded(png('red'))]})
        assert queued.json() == {'menu': None}
        assert web_.post('/api/input', {'text': ' ', 'images': [encoded(png('blue'))]}).status_code == 200
        web_.portal.call(release.set)
        messages: list[ServerMessage] = []
        for _ in range(3):
            messages += receive_until(socket, TurnEnded)[0]
        started_items = [message.item for message in messages if isinstance(message, TurnStarted)]
        red, blue = decode_images([encoded(png('red')), encoded(png('blue'))])
        shown: list[tuple[str, list[str]]] = [('first', []), ('what is this?', [image_id(red)]), ('', [image_id(blue)])]
        assert [(item.text, item.images) for item in started_items if item] == shown
        history = [item for item in web_.get('/api/state').json()['items'] if item['kind'] == 'user']
        assert [(item['text'], item['images']) for item in history] == shown, 'the same IDs after a reload'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    assert prompts == ['first', ['what is this?', red], [blue]], 'no empty text block beside images alone'
    assert web_.shell.images.pending == {}, 'sent images are let go'


def test_the_transcript_fetches_the_images_of_a_session(make_web: 'MakeWeb') -> None:
    """A prompt's images are served by ID, from a running turn's prompt and from the saved history."""
    started, release = anyio.Event(), anyio.Event()

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        started.set()
        await release.wait()
        yield 'seen'

    web_ = make_web(Agent(FunctionModel(stream_function=stream)))
    [red] = decode_images([encoded(png('red'))])
    route = f'/api/sessions/{web_.shell.session.conversation_id}/images/{image_id(red)}'
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'look', 'images': [encoded(png('red'))]})
        web_.portal.call(started.wait)
        for path in (route, f'/api/images/{image_id(red)}'):
            response = web_.get(path)
            assert (response.status_code, response.content) == (200, red.data), 'while its turn runs'
            assert response.headers['content-type'] == 'image/png'
            assert response.headers['cache-control'] == 'private, max-age=31536000, immutable'
        web_.portal.call(release.set)
        receive_until(socket, TurnEnded)
        # A later prompt without images leaves only the saved history to find it in.
        web_.post('/api/input', {'text': 'again'})
        receive_until(socket, TurnEnded)
        assert web_.get(route).content == red.data, 'from the saved history'
        unknown = web_.get(f'/api/images/{"0" * 64}')
        assert (unknown.status_code, unknown.json()) == (404, {'detail': 'No such image in this session.'})
        assert web_.client.get(route).status_code == 403, 'not without the token'
        # A new conversation lets go of the old one's images.
        web_.post('/api/input', {'text': '/new'})
        assert receive_until(socket, Snapshot)[1].items == []
        assert web_.get(f'/api/images/{image_id(red)}').status_code == 404
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_a_prompt_keeps_its_images_while_it_waits_for_the_terminal(
    make_web: 'MakeWeb', monkeypatch: pytest.MonkeyPatch
) -> None:
    """Images are read as the prompt leaves the queue, so the terminal moving here meanwhile cannot let them go."""
    prompts: list[str | Sequence[UserContent]] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        prompts.extend(part.content for part in request.parts if isinstance(part, UserPromptPart))
        yield 'seen'

    web_ = make_web(Agent(FunctionModel(stream_function=stream)))
    waiting, announced = anyio.Event(), anyio.Event()
    busy = web_.shell.forks.busy

    @asynccontextmanager
    async def after_an_announcement() -> AsyncGenerator[None]:
        # As when a fork's announcement holds the terminal.
        if not announced.is_set():
            waiting.set()
            await announced.wait()
        async with busy():
            yield

    monkeypatch.setattr(web_.shell.forks, 'busy', after_an_announcement)
    running = web_.run()
    web_.post('/api/input', {'text': 'what is this?', 'images': [encoded(png('red'))]})
    web_.portal.call(waiting.wait)
    # As the terminal showing this session now does, with nothing left in its queue.
    web_.portal.call(web_.shell.images.retain, list[str]())
    web_.post('/api/input', {'text': '/exit'})
    web_.portal.call(announced.set)
    running.result(timeout=10)
    assert prompts == [['what is this?', *decode_images([encoded(png('red'))])]]


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
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
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
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
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
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
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
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
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
        [here] = projects[0]['worktrees']
        assert [
            (entry['title'], entry['state'], entry['current'], entry['openable']) for entry in here['sessions']
        ] == [('first', 'completed', True, True)]
        assert [(project['name'], [w['label'] for w in project['worktrees']]) for project in projects[1:]] == [
            ('Removed folders', ['else'])
        ]
        [listed] = web_.get('/api/sessions?q=Elsewhere').json()[0]['worktrees'][0]['sessions']
        assert (listed['id'], listed['openable']) == (elsewhere.id, False), 'its row says it only shows'

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

        assert web_.post('/api/sessions/new').status_code == 200
        snapshot = receive_until(socket, Snapshot)[1]
        assert snapshot.session.id != first and snapshot.items == []

        assert web_.post(f'/api/sessions/{first}/open').status_code == 200
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
    """A live session's turn shows as running; one another live process runs, as busy; a dead one, as cancelled."""
    web_ = make_web(Agent(TestModel()))
    store = web_.shell.sessions.store
    for title, pid in (('Live', os.getppid()), ('Mine', os.getpid()), ('Dead', 2**22 + 12345)):
        web_.portal.call(
            lambda title=title, pid=pid: store.save(
                summary=ConversationSummary(workspace='/x', title=title, outcome='running', owner_pid=pid),
                messages=[],
            )
        )
    [removed] = web_.get('/api/sessions').json()[-1]['worktrees']
    assert {entry['title']: entry['state'] for entry in removed['sessions']} == {
        'Live': 'busy',
        'Mine': 'cancelled',
        'Dead': 'cancelled',
    }, "this process's own turn is never busy elsewhere: no live session here holds it, so it ended unsaved"
    web_.shell.turning = True
    web_.portal.call(web_.shell.session.commit_messages, list[ModelMessage]())
    current = web_.get('/api/sessions').json()[0]['worktrees'][0]['sessions'][0]
    assert (current['current'], current['state']) == (True, 'running')


def test_a_session_another_process_runs_shows_read_only_until_forked(make_web: 'MakeWeb') -> None:
    """Viewing it saves nothing; opening it, the banner's explicit fork, continues one copy, reused thereafter."""
    web_ = make_web(Agent(TestModel()))
    store = web_.shell.sessions.store
    busy = web_.portal.call(
        lambda: store.save(
            summary=ConversationSummary(
                workspace=web_.shell.session.workspace, title='Busy', outcome='running', owner_pid=os.getppid()
            ),
            messages=[],
        )
    )
    transcript = web_.get(f'/api/sessions/{busy.id}').json()
    assert (transcript['openable'], transcript['missing'], transcript['owner_pid']) == (False, False, os.getppid())
    assert [entry.id for entry in web_.portal.call(store.listing)] == [busy.id]
    [row] = web_.get('/api/sessions').json()[0]['worktrees'][0]['sessions']
    assert (row['id'], row['state'], row['openable']) == (busy.id, 'busy', False), 'a click fetches the transcript'
    # A fork the user made from the row menu is theirs; continuing the busy session does not take it over.
    mine = web_.portal.call(web_.shell.sessions.fork, busy.id).id

    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        assert web_.post(f'/api/sessions/{busy.id}/open').status_code == 200
        copy = receive_until(socket, Snapshot)[1].session
        assert copy.id not in (busy.id, mine)
        assert copy.title == 'Busy (fork)'
        assert web_.get(f'/api/sessions/{copy.id}').json()['owner_pid'] is None
        web_.post('/api/input', {'text': '/new'})
        receive_until(socket, Snapshot)
        assert web_.post(f'/api/sessions/{busy.id}/open').status_code == 200
        assert receive_until(socket, Snapshot)[1].session.id == copy.id, 'the fork is continued, not forked again'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)
    assert sorted(entry.id for entry in web_.portal.call(store.listing)) == sorted([busy.id, mine, copy.id])


def test_sidebar_groups_sessions_by_project_then_worktree(make_web: 'MakeWeb', tmp_path: Path) -> None:
    """Each repository is one project, its checkouts nested in it, removed worktrees included.

    A removed worktree keeps the checkout saved with its sessions; an older one without it is placed by
    where it was. Folders outside Git are their own project, and removed ones no project claims are last.
    """
    web_ = make_web(Agent(TestModel()))
    worktrees = tmp_path / 'worktrees' / 'alpha'
    repo, feature, gone = tmp_path / 'code' / 'alpha', worktrees / 'feature', worktrees / 'gone'
    repository(repo, linked=feature, branch='feature')
    git('-C', str(repo), 'worktree', 'add', '-b', 'fix/gone', str(gone))
    saved = git_checkout(project_identity(str(gone.resolve())))
    git('-C', str(repo), 'worktree', 'remove', str(gone))
    plain = tmp_path / 'plain'
    plain.mkdir()
    workspaces = {
        'Removed here': (tmp_path / 'one' / 'x', None),
        'Removed elsewhere': (tmp_path / 'two' / 'x', None),
        'Plain': (plain, None),
        'Before recording': (worktrees / 'old' / 'src', None),
        'Recorded': (gone / 'src', saved),
        'Deleted subfolder': (repo / 'deleted', None),
        'Main': (repo, None),
        'Feature': (feature, None),
    }
    for title, (workspace, checkout) in workspaces.items():
        summary = ConversationSummary(workspace=str(workspace.resolve()), title=title, checkout=checkout)
        web_.portal.call(lambda summary=summary: web_.shell.sessions.store.save(summary=summary, messages=[]))
    web_.shell.session.workspace = str(feature.resolve())

    projects = web_.get('/api/sessions').json()
    shown = [
        (
            project['name'],
            project['current'],
            [
                (w['label'], w['current'], w['missing'], [entry['title'] for entry in w['sessions']])
                for w in project['worktrees']
            ],
        )
        for project in projects
    ]
    one, two = (str((tmp_path / name / 'x').resolve()) for name in ('one', 'two'))
    assert shown == [
        (
            'alpha',
            True,
            [
                ('feature', True, False, ['Feature']),
                ('main', False, False, ['Main', 'Deleted subfolder']),
                ('fix/gone', False, True, ['Recorded']),
                ('old', False, True, ['Before recording']),
            ],
        ),
        ('plain', False, [('plain', False, False, ['Plain'])]),
        (
            'Removed folders',
            False,
            [(two, False, True, ['Removed elsewhere']), (one, False, True, ['Removed here'])],
        ),
    ]
    # A search lists only matching sessions, in the same places.
    [found] = web_.get('/api/sessions?q=Before').json()
    assert [(w['label'], w['missing']) for w in found['worktrees']] == [('old', True)]


def test_sidebar_lists_removed_worktrees_after_live_ones(make_web: 'MakeWeb', tmp_path: Path) -> None:
    """A removed worktree comes after the project's live checkouts, even when its session is the newest."""
    web_ = make_web(Agent(TestModel()))
    repo, feature, gone = tmp_path / 'repo', tmp_path / 'wt' / 'feature', tmp_path / 'wt' / 'gone'
    repository(repo, linked=feature, branch='feature')
    git('-C', str(repo), 'worktree', 'add', '-b', 'gone', str(gone))
    for title, workspace in [('Feature', feature), ('Main', repo), ('Gone', gone)]:
        folder = str(workspace.resolve())
        summary = ConversationSummary(workspace=folder, title=title, checkout=git_checkout(project_identity(folder)))
        web_.portal.call(lambda summary=summary: web_.shell.sessions.store.save(summary=summary, messages=[]))
    git('-C', str(repo), 'worktree', 'remove', str(gone))
    plain = tmp_path / 'plain'
    plain.mkdir()
    web_.shell.session.workspace = str(plain.resolve())

    [project] = [p for p in web_.get('/api/sessions').json() if p['name'] == 'repo']
    assert [(w['label'], w['missing']) for w in project['worktrees']] == [
        ('main', False),
        ('feature', False),
        ('gone', True),
    ]


def test_archiving_a_worktree_archives_every_session_in_it(make_web: 'MakeWeb') -> None:
    """One request archives a group's sessions and lists them no more; ones it refuses stay, with the reasons."""
    web_ = make_web(Agent(TestModel()))
    old = [
        web_.portal.call(
            lambda title=title: web_.shell.sessions.store.save(
                summary=ConversationSummary(workspace='/gone', title=title), messages=[]
            )
        )
        for title in ('One', 'Two')
    ]
    web_.portal.call(web_.shell.session.commit_messages, list[ModelMessage]())
    current = web_.shell.session.conversation_id

    def listed() -> list[str]:
        return [
            entry['title']
            for project in web_.get('/api/sessions').json()
            for worktree in project['worktrees']
            for entry in worktree['sessions']
        ]

    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        ids = [summary.id for summary in old]
        assert web_.client.post('/api/sessions/archive', json={'ids': ids}).status_code == 403, 'the token is checked'
        assert web_.post('/api/sessions/archive', {'ids': ids}).status_code == 204
        receive_until(socket, SessionsChanged)
        assert listed() == ['New session']
        assert web_.post(f'/api/sessions/{old[0].id}/unarchive').status_code == 204
        receive_until(socket, SessionsChanged)
        refused = web_.post('/api/sessions/archive', {'ids': [current, old[0].id, 'missing']})
        assert (refused.status_code, refused.json()['detail']) == (
            400,
            f"Archived 1 of 3 sessions. '{current}': The current session cannot be archived. Switch to another one "
            "first. 'missing': No saved session: missing",
        )
        receive_until(socket, SessionsChanged)
        assert web_.post('/api/sessions/archive', {'ids': ['missing']}).json()['detail'] == (
            "Archived 0 of 1 sessions. 'missing': No saved session: missing"
        )
    assert listed() == ['New session']


def test_new_session_offers_every_worktree_of_a_known_repository(
    make_web: 'MakeWeb', tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New session lists a known repository's other worktrees before any session runs there; the sidebar does not.

    One whose folder was deleted behind Git's back is not offered, nor is any during a search, or when Git fails.
    """
    web_ = make_web(Agent(TestModel()))
    repo, feature, deleted = tmp_path / 'repo', tmp_path / 'wt' / 'feature', tmp_path / 'wt' / 'deleted'
    repository(repo, linked=feature, branch='feature')
    git('-C', str(repo), 'worktree', 'add', '--detach', str(deleted))
    shutil.rmtree(deleted)
    summary = ConversationSummary(workspace=str(repo.resolve()), title='Main')
    web_.portal.call(lambda: web_.shell.sessions.store.save(summary=summary, messages=[]))
    plain = tmp_path / 'plain'
    plain.mkdir()
    web_.shell.session.workspace = str(plain.resolve())

    def shown(query: str) -> list[tuple[str, str, list[str]]]:
        [project] = [p for p in web_.get(f'/api/sessions?{query}').json() if p['name'] == 'repo']
        return [(w['label'], w['root'], [entry['title'] for entry in w['sessions']]) for w in project['worktrees']]

    main = ('main', str(repo.resolve()), ['Main'])
    assert shown('include_worktrees=true') == [main, ('feature', str(feature.resolve()), [])]
    assert shown('q=') == [main], 'the sidebar lists checkouts with sessions only'
    assert shown('q=Main&include_worktrees=true') == [main]
    monkeypatch.setenv('PATH', '')
    assert shown('include_worktrees=true') == [main]


def test_new_session_completes_a_typed_path_with_its_repository_worktrees(make_web: 'MakeWeb', tmp_path: Path) -> None:
    """New session completes a typed path to directories, listing each repository's worktrees before any session.

    A subfolder of a checkout is offered as itself, hidden folders only when typed, and a relative,
    unreadable, or empty path offers nothing.
    """
    web_ = make_web(Agent(TestModel()))
    fresh, linked = tmp_path / 'fresh', tmp_path / 'fresh-wt'
    repository(fresh, linked=linked, branch='wt')
    # Git still lists a locked worktree whose folder is gone; the dialog leaves out a missing one.
    git('-C', str(fresh), 'worktree', 'add', '--lock', '-b', 'gone', str(tmp_path / 'gone'))
    shutil.rmtree(tmp_path / 'gone')
    (fresh / 'src').mkdir()
    (tmp_path / 'frozen').mkdir()
    (tmp_path / '.free').mkdir()
    (tmp_path / 'french.txt').touch()
    assert not any(p['name'] == 'fresh' for p in web_.get('/api/sessions?include_worktrees=true').json())

    def completed(path: str) -> list[tuple[str, list[tuple[str, str]]]]:
        response = web_.get(f'/api/directories?path={quote(path)}')
        assert response.status_code == 200, response.text
        return [
            (p['name'], [(w['label'], w['root']) for w in p['worktrees'] if not w['missing']]) for p in response.json()
        ]

    def at(*parts: str) -> str:
        return str(tmp_path.resolve().joinpath(*parts))

    repo = ('fresh', [('main', at('fresh')), ('wt', at('fresh-wt'))])
    assert completed(at('fr')) == [repo, ('frozen', [('frozen', at('frozen'))])]
    assert completed(at('fresh')) == [repo]
    [found] = web_.get(f'/api/directories?path={quote(at("fresh-"))}').json()
    assert (at('gone'), True) in [(w['root'], w['missing']) for w in found['worktrees']]
    assert completed(at('.f')) == [('.free', [('.free', at('.free'))])]
    assert completed(at('fresh') + os.sep) == [('src', [('src', at('fresh', 'src'))])]
    assert completed('fresh') == completed(at('nowhere') + os.sep) == completed(at('a\0', 'b')) == completed('') == []
    # At most `COMPLETIONS` directories, in name order.
    for index in range(13):
        (tmp_path / f'z{index:02}').mkdir()
    assert [name for name, _ in completed(at('z'))] == [f'z{index:02}' for index in range(12)]


def test_a_bare_repository_is_not_a_worktree(tmp_path: Path) -> None:
    """A bare clone's own folder holds no checkout, so only its linked worktrees are offered."""
    source, bare, feature = tmp_path / 'source', tmp_path / 'bare.git', tmp_path / 'feature'
    repository(source, linked=tmp_path / 'unused', branch='unused')
    git('clone', '--bare', str(source), str(bare))
    git('-C', str(bare), 'worktree', 'add', '-b', 'feature', str(feature))
    assert worktree_roots(project_identity(str(feature.resolve()))) == [str(feature.resolve())]


def test_sidebar_notices_a_folder_removed_or_recreated_while_it_runs(make_web: 'MakeWeb', tmp_path: Path) -> None:
    """Each listing resolves again a folder removed or recreated since the last one, as `/resume` does."""
    web_ = make_web(Agent(TestModel()))
    folder = (tmp_path / 'folder').resolve()
    folder.mkdir()
    summary = ConversationSummary(workspace=str(folder), title='Folder')
    web_.portal.call(lambda: web_.shell.sessions.store.save(summary=summary, messages=[]))

    def shown() -> list[tuple[str, bool]]:
        return [
            (worktree['label'], worktree['missing'])
            for project in web_.get('/api/sessions').json()
            for worktree in project['worktrees']
            if [entry['title'] for entry in worktree['sessions']] == ['Folder']
        ]

    assert shown() == [('folder', False)]
    folder.rmdir()
    assert shown() == [('folder', True)]
    folder.mkdir()
    assert shown() == [('folder', False)]


def test_archived_sessions_leave_the_sidebar_until_it_lists_them(make_web: 'MakeWeb') -> None:
    """Archiving hides a session, listed again muted with `include_archived=true`; the open one is refused."""
    web_ = make_web(Agent(TestModel()))
    old = web_.portal.call(
        lambda: web_.shell.sessions.store.save(summary=ConversationSummary(workspace='/x', title='Stale'), messages=[])
    )
    web_.portal.call(web_.shell.session.commit_messages, list[ModelMessage]())
    current = web_.shell.session.conversation_id

    def listed(path: str) -> dict[str, bool]:
        return {
            entry['title']: entry['archived']
            for project in web_.get(path).json()
            for worktree in project['worktrees']
            for entry in worktree['sessions']
        }

    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        assert web_.post(f'/api/sessions/{old.id}/archive').status_code == 204
        receive_until(socket, SessionsChanged)
        assert listed('/api/sessions') == {'New session': False}
        assert listed('/api/sessions?include_archived=true') == {'New session': False, 'Stale': True}
        assert listed('/api/sessions?q=Stale&include_archived=true') == {'Stale': True}
        refused = web_.post(f'/api/sessions/{current}/archive')
        assert (refused.status_code, refused.json()['detail']) == (
            400,
            'The current session cannot be archived. Switch to another one first.',
        )
        assert web_.post('/api/sessions/missing/archive').status_code == 404
        assert web_.client.post(f'/api/sessions/{old.id}/unarchive').status_code == 403, 'the token is checked'
        assert web_.post(f'/api/sessions/{old.id}/unarchive').status_code == 204
        receive_until(socket, SessionsChanged)
    assert listed('/api/sessions') == {'New session': False, 'Stale': False}


def test_a_shell_without_saved_sessions(make_web: 'MakeWeb') -> None:
    web_ = make_web(Agent(TestModel()))
    web_.plugin.host.sessions = None
    assert web_.get('/api/sessions').status_code == 404


PANEL_PLUGINS = """
\"\"\"Plugins for the plugins panel.\"\"\"

from collections.abc import Callable
from typing import Literal

import asyncio

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pydantic_clai2.plugins import Plugin
from pydantic_clai2.plugins.menus import FieldRow, Option, SettingsMenu


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


class DeclaredSettings(BaseModel):
    model_config = ConfigDict(extra='forbid')

    colour: Literal['red', 'blue'] = 'red'
    size: int = Field(default=2, ge=1)
    token: str | None = None


class DeclaredSource:
    title = 'Declared settings'

    def __init__(self, host):
        self.host = host

    def rows(self):
        # Like a source that reads the keyring or `/keys`, it must never be read on the event loop.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError('rows() read on the event loop')
        return (
            FieldRow(
                key='colour',
                label='Colour',
                description='Which colour.',
                default='red',
                choices=('red', 'blue'),
                choice_labels={'red': 'Red', 'blue': 'Blue'},
                allow_custom=False,
            ),
            FieldRow(key='size', label='Size', description='How big.', default='2', note='a note'),
            FieldRow(key='token', label='Token', description='Set by an action.', default='(not set)', secret=True),
        )

    def current(self, row):
        settings = self.host.settings(DeclaredSettings)
        values = {'colour': settings.colour, 'size': str(settings.size), 'token': '(set)' if settings.token else '(not set)'}
        return values[row.key]

    def _updated(self, row, text):
        return DeclaredSettings.model_validate({**self.host.settings(DeclaredSettings).model_dump(), row.key: text})

    def problem(self, row, text):
        try:
            self._updated(row, text)
        except ValueError as error:
            return error.errors()[0]['msg'] if isinstance(error, ValidationError) else str(error)
        return None

    def apply(self, row, raw):
        self.host.save_settings(self._updated(row, raw))
        return f'Saved {row.label}.'

    def reset(self, row):
        data = self.host.settings(DeclaredSettings).model_dump()
        data.pop(row.key)
        self.host.save_settings(DeclaredSettings.model_validate(data))
        return f'Reset {row.label}.'


class Declared(Plugin[DeclaredSettings]):
    \"\"\"A plugin that declares its settings menu.\"\"\"

    def __init__(self, host, settings):
        if settings.size == 13:
            raise RuntimeError('unlucky size')
        super().__init__(host, settings)

    def settings_menu(self):
        return SettingsMenu(source=DeclaredSource(self.host), actions={'token': self._token})

    async def _token(self, dialogs):
        how = await dialogs.choose(
            title='Token',
            options=[Option(value='type', label='Type one', description='Paste it.'), Option(value='sign_in', label='Sign in')],
            current='type',
            description='How to get one.',
        )
        if how is None:
            return []
        if how == 'sign_in':

            async def sign_in(show):
                show('Open https://example.com/device and enter ABCD.')
                await asyncio.sleep(0.05)

            return ['Signed in.'] if await dialogs.wait(title='Signing in', description='Finish there.', task=sign_in) else []
        typed = await dialogs.text(
            title='Token',
            label='Token',
            initial='ignored',
            secret=True,
            check=lambda text: None if text.startswith('tok') else 'Tokens start with tok.',
            description='Find it at https://example.com/tokens.',
        )
        if not typed:
            return []
        if not await dialogs.confirm(title='Replace the token?', yes='Replace', no='Keep', description='Others use it.'):
            return ['Kept the token.']
        self.host.save_settings(self.host.settings(DeclaredSettings).model_copy(update={'token': typed}))
        return ['Saved the token.']
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
            for name in ('Sample', 'Plain', 'Opaque', 'Declared')
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
    assert sorted(rows) == ['ask_user', 'broken', 'declared', 'opaque', 'plain', 'sample', 'web']
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
    # A declared menu replaces the schema form; an undeclared one does not count as declared.
    declared = rows['declared']
    assert (declared['has_menu'], declared['settings_menu'], declared['form']) == (True, True, None)
    assert (sample['settings_menu'], rows['plain']['settings_menu']) == (False, False)
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
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        assert web_.post('/api/plugins/sample/settings', {'changes': {'limit': 5, 'mode': 'slow'}}).status_code == 204
        receive_until(socket, PluginsChanged)
    # Saved, then loaded again with them, as the plugin's own menu does.
    form = plugin_rows(web_)['sample']['form']
    assert isinstance(form, dict) and form['values'] == {'mode': 'slow', 'limit': 5, 'label': 'hi'}
    second = loaded_plugin(web_, 'sample')
    assert second is not first
    assert [saved.settings for saved in web_.shell.process.store.plugins() if saved.id == 'sample'] == [
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
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
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
        assert (refused.status_code, refused.json()['detail']) == (400, service_module.SERVING)
    refused = web_.post('/api/plugins/web/settings', {'changes': {'open_browser': True}})
    assert (refused.status_code, refused.json()['detail']) == (400, service_module.SERVING)
    assert plugin_rows(web_)['web']['status'] == 'on'

    failed = web_.post('/api/plugins/broken/enable')
    assert failed.status_code == 400 and 'cannot import' in failed.json()['detail']
    broken = plugin_rows(web_)['broken']
    assert broken['status'] == 'failed' and 'cannot import' in str(broken['error'])
    assert web_.post('/api/plugins/sample/explode').status_code == 422


def test_a_plugins_own_settings_menu_opens_in_the_terminal(make_web: 'MakeWeb', tmp_path: Path) -> None:
    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
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
    assert client.get('/assets/app.js', headers={'host': 'evil.example'}).status_code == 403
    assert client.get(f'/?token={TOKEN}', headers={'host': 'evil.example'}).status_code == 403

    page = client.get(f'/?token={TOKEN}')
    assert f'<meta name="clai-token" content="{TOKEN}"></head>' in page.text
    assert "frame-ancestors 'none'" in page.headers['content-security-policy']
    assert 'set-cookie' not in page.headers, 'cookies reach every port of the host'

    for path, headers in (
        ('/ws', {'origin': 'http://testserver'}),
        (f'/ws?token={TOKEN}&protocol={PROTOCOL}', {}),
        (f'/ws?token={TOKEN}&protocol={PROTOCOL}', {'origin': 'http://evil.example'}),
    ):
        with pytest.raises(WebSocketDisconnect) as refused, client.websocket_connect(path, headers=headers):
            pass  # pragma: no cover
        assert refused.value.code == 1008


def test_assets_are_versioned_cached_and_compressed(make_web: 'MakeWeb', tmp_path: Path) -> None:
    """The page links each asset by its content's digest, not the launch's token, so it stays cached across launches."""
    web_ = make_web(Agent(TestModel()))
    client = web_.client
    script = tmp_path / 'static' / 'assets' / 'app.js'
    version = hashlib.sha256(script.read_bytes()).hexdigest()[:16]
    page = client.get(f'/?token={TOKEN}').text
    assert f'src="/assets/app.js?v={version}"' in page
    assert 'href="/assets/index.css"' in page, 'a link to a missing file is left alone'
    assert TOKEN not in page.replace(f'<meta name="clai-token" content="{TOKEN}">', '')
    other = TestClient(
        create_app(service=web_.plugin.service, token='next-launch', hosts=HOSTS, static=tmp_path / 'static')
    )
    assert f'src="/assets/app.js?v={version}"' in other.get('/?token=next-launch').text, 'the same link every launch'

    served = client.get(f'/assets/app.js?v={version}', headers={'accept-encoding': 'gzip, br'})
    assert served.status_code == 200 and served.text == 'console.log("hi")', 'public build output needs no token'
    assert served.headers['cache-control'] == 'public, max-age=31536000, immutable'
    assert served.headers['content-encoding'] == 'gzip'
    assert served.headers['vary'] == 'Accept-Encoding'
    assert 'javascript' in served.headers['content-type']
    plain = client.get('/assets/app.js', headers={'accept-encoding': 'identity'})
    assert (plain.headers['cache-control'], plain.text) == ('no-cache', 'console.log("hi")')
    assert 'content-encoding' not in plain.headers
    revalidated = client.get('/assets/app.js', headers={'if-none-match': served.headers['etag']})
    assert (revalidated.status_code, revalidated.content) == (304, b'')

    assert client.get('/assets/missing.js').status_code == 404


async def test_assets_are_built_once_per_version(tmp_path: Path) -> None:
    """Each file is read and compressed once, until a rebuild changes it, and only files of the directory are served."""
    assets = app_module.Assets(static_dir(tmp_path) / 'assets')
    built = await assets.get('app.js')
    assert built is not None and built is await assets.get('app.js')
    (tmp_path / 'static' / 'assets' / 'app.js').write_text('console.log("rebuilt")')
    rebuilt = await assets.get('app.js')
    assert rebuilt is not None and (rebuilt.body, rebuilt.version != built.version) == (b'console.log("rebuilt")', True)
    assert gzip.decompress(rebuilt.gzipped) == rebuilt.body
    assert await assets.get('../index.html') is None


def test_a_checkout_without_a_built_ui(tmp_path: Path) -> None:
    service = WebPlugin.from_host(
        PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={})
    ).service
    with TestClient(create_app(service=service, token=TOKEN, hosts=HOSTS, static=tmp_path / 'missing')) as client:
        page = client.get(f'/?token={TOKEN}')
        assert (page.status_code, page.text) == (503, UNBUILT)


def test_a_build_without_assets(tmp_path: Path) -> None:
    static = tmp_path / 'bare'
    static.mkdir()
    (static / 'index.html').write_text('<html><head></head></html>')
    service = WebPlugin.from_host(
        PluginHost[None](name='web', console=Console(file=io.StringIO()), settings={})
    ).service
    with TestClient(create_app(service=service, token=TOKEN, hosts=HOSTS, static=static)) as client:
        assert client.get('/assets/app.js').status_code == 404
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
) -> 'SessionRuntime[None, str]':
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
    hub = web_.plugin.service.hub
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        socket.receive_text()  # The orchestrator chat's state.
        socket.receive_text()  # The live sessions.

        async def flood() -> None:
            # All at once, on the loop, so the WebSocket cannot forward any in between.
            for text in ('one', 'two', 'three'):
                hub.publish(Notice(text=text))

        web_.portal.call(flood)
        with pytest.raises(WebSocketDisconnect) as closed:
            socket.receive_text()
        assert closed.value.code == 1013
    assert hub.clients == 0


def menu_fields(web_: Web, name: str) -> dict[str, dict[str, JsonValue]]:
    response = web_.get(f'/api/plugins/{name}/fields')
    assert response.status_code == 200, response.text
    menu = TypeAdapter(dict[str, JsonValue]).validate_python(response.json())
    fields = TypeAdapter(list[dict[str, JsonValue]]).validate_python(menu['fields'])
    return {str(field['key']): field for field in fields}


def test_a_declared_settings_menu_edits_from_the_browser(make_web: 'MakeWeb', tmp_path: Path) -> None:
    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    first = loaded_plugin(web_, 'declared')
    fields = menu_fields(web_, 'declared')
    assert fields['colour'] == {
        'key': 'colour',
        'label': 'Colour',
        'description': 'Which colour.',
        'value': 'red',
        'default': 'red',
        'choices': [
            {'value': 'red', 'label': 'Red', 'description': ''},
            {'value': 'blue', 'label': 'Blue', 'description': ''},
        ],
        'allow_custom': False,
        'secret': False,
        'note': '',
        'action': False,
        'menu': False,
        'button': '',
    }
    assert (fields['size']['note'], fields['token']['secret'], fields['token']['action']) == ('a note', True, True)
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        # A value the menu rejects is reported with its own message, and nothing is saved or reloaded.
        rejected = web_.post('/api/plugins/declared/fields/size', {'value': '0'})
        assert (rejected.status_code, rejected.json()['detail']) == (400, 'Input should be greater than or equal to 1')
        assert loaded_plugin(web_, 'declared') is first
        edited = web_.post('/api/plugins/declared/fields/size', {'value': '5'})
        assert edited.status_code == 200
        body = edited.json()
        assert body['messages'] == ['Saved Size.']
        assert [field['value'] for field in body['menu']['fields']] == ['red', '5', '(not set)']
        receive_until(socket, PluginsChanged)
        # The plugin loaded again with what it saved.
        again = loaded_plugin(web_, 'declared')
        assert again is not first
        # Empty text resets, as an empty edit in the terminal menu does.
        assert web_.post('/api/plugins/declared/fields/size', {'value': ' '}).json()['messages'] == ['Reset Size.']
        assert web_.post('/api/plugins/declared/fields/size', {}).json()['menu']['fields'][1]['value'] == '2'
    # The page's own plugin is never edited from it.
    assert web_.post('/api/plugins/web/fields/host', {'value': 'x'}).json()['detail'] == service_module.SERVING
    assert web_.post('/api/plugins/declared/fields/nope', {'value': '1'}).status_code == 404
    assert web_.post('/api/plugins/declared/fields/size/action').status_code == 404
    # A plugin without a declared menu, or one that is off, has no fields here.
    assert web_.get('/api/plugins/sample/fields').status_code == 404
    assert web_.get('/api/plugins/broken/fields').status_code == 404
    assert web_.get('/api/plugins/unknown/fields').status_code == 404
    assert web_.post('/api/plugins/sample/fields/mode', {'value': 'slow'}).status_code == 400
    # Saved, but the plugin refuses to load with it: the error is the answer, and it shows as failed.
    failed = web_.post('/api/plugins/declared/fields/size', {'value': '13'})
    assert failed.status_code == 400 and 'unlucky size' in failed.json()['detail']
    assert plugin_rows(web_)['declared']['status'] == 'failed'


def test_a_settings_menu_action_asks_through_browser_dialogs(make_web: 'MakeWeb', tmp_path: Path) -> None:
    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    with (
        ThreadPoolExecutor(1) as pool,
        web_.client.websocket_connect(
            f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
        ) as socket,
    ):
        socket.receive_text()
        action = pool.submit(web_.post, '/api/plugins/declared/fields/token/action')
        choose = receive_until(socket, DialogOpened)[1].dialog
        assert (choose.kind, choose.title, choose.current, choose.description) == (
            'choose',
            'Token',
            'type',
            'How to get one.',
        )
        assert [(option.value, option.label, option.description) for option in choose.options] == [
            ('type', 'Type one', 'Paste it.'),
            ('sign_in', 'Sign in', ''),
        ]
        # A browser that connects now sees the open dialog in its snapshot.
        assert [dialog['id'] for dialog in web_.get('/api/state').json()['dialogs']] == [choose.id]
        assert web_.post(f'/api/dialogs/{choose.id}', {'value': 'other'}).status_code == 400
        assert web_.post(f'/api/dialogs/{choose.id}', {'value': 'type'}).status_code == 204
        assert receive_until(socket, DialogClosed)[1].id == choose.id
        text = receive_until(socket, DialogOpened)[1].dialog
        # A secret never starts with text in it.
        assert (text.kind, text.secret, text.initial, text.label, text.description) == (
            'text',
            True,
            '',
            'Token',
            'Find it at https://example.com/tokens.',
        )
        rejected = web_.post(f'/api/dialogs/{text.id}', {'value': 'nope'})
        assert (rejected.status_code, rejected.json()['detail']) == (400, 'Tokens start with tok.')
        assert web_.post(f'/api/dialogs/{text.id}', {'value': ' tok-1 '}).status_code == 204
        confirm = receive_until(socket, DialogOpened)[1].dialog
        assert (confirm.kind, confirm.yes, confirm.no, confirm.description) == (
            'confirm',
            'Replace',
            'Keep',
            'Others use it.',
        )
        # While the action runs, the menu is open: another edit of the same plugin waits its turn.
        busy = web_.post('/api/plugins/declared/fields/size', {'value': '3'})
        assert (busy.status_code, busy.json()['detail']) == (
            400,
            'The settings menu of declared is already open; finish there first.',
        )
        assert web_.post(f'/api/dialogs/{confirm.id}', {'value': 'yes'}).status_code == 204
        done = action.result(timeout=10)
        assert done.status_code == 200
        assert done.json()['messages'] == ['Saved the token.']
        assert done.json()['menu']['fields'][2]['value'] == '(set)'
        # Answering what is no longer open is a 404.
        assert web_.post(f'/api/dialogs/{confirm.id}', {'value': 'yes'}).status_code == 404

        # Backing out of the first dialog saves nothing.
        action = pool.submit(web_.post, '/api/plugins/declared/fields/token/action')
        choose = receive_until(socket, DialogOpened)[1].dialog
        assert web_.post(f'/api/dialogs/{choose.id}', {'value': None}).status_code == 204
        assert action.result(timeout=10).json()['messages'] == []

        # A wait shows what its task shows, and closes when the task ends.
        action = pool.submit(web_.post, '/api/plugins/declared/fields/token/action')
        choose = receive_until(socket, DialogOpened)[1].dialog
        assert web_.post(f'/api/dialogs/{choose.id}', {'value': 'sign_in'}).status_code == 204
        waiting = receive_until(socket, DialogOpened)[1].dialog
        assert (waiting.kind, waiting.description) == ('wait', 'Finish there.')
        shown = receive_until(socket, DialogShown)[1]
        assert (shown.id, shown.text) == (waiting.id, 'Open https://example.com/device and enter ABCD.')
        assert action.result(timeout=10).json()['messages'] == ['Signed in.']


def test_stopping_the_web_ui_backs_out_of_open_dialogs(make_web: 'MakeWeb', tmp_path: Path) -> None:
    web_ = make_web(Agent(TestModel()), extra=panel_plugins(tmp_path))
    with (
        ThreadPoolExecutor(1) as pool,
        web_.client.websocket_connect(
            f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
        ) as socket,
    ):
        socket.receive_text()
        action = pool.submit(web_.post, '/api/plugins/declared/fields/token/action')
        receive_until(socket, DialogOpened)
        web_.portal.call(web_.plugin.stop)
        assert action.result(timeout=10).json()['messages'] == []
    assert web_.plugin.service.dialogs.dialogs == []


def follow_workspaces(
    web_: Web, settings: Callable[[ProjectSettings], Settings] = lambda project: resolve_settings(project.overrides)
) -> SessionWorkspace:
    """Wire the shell as `clai2 --web` does, so each session runs in its own directory."""
    shell = web_.shell
    workspace = SessionWorkspace(
        session=shell.session,
        context=shell.context,
        loader=shell.loader,
        status=shell.status,
        settings=settings,
    )
    workspace.attach(sessions=shell.sessions, commands=shell.commands)
    return workspace


def saved_in(web_: Web, workspace: Path, title: str) -> str:
    summary = ConversationSummary(workspace=str(workspace), title=title)
    return web_.portal.call(lambda: web_.shell.sessions.store.save(summary=summary, messages=[])).id


REPO_CONTEXT = PluginSettings(id='repo_context', factory='pydantic_clai2.builtin_plugins.repo_context')


def test_sessions_continue_in_their_own_directories(
    make_web: 'MakeWeb', tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under `clai2 --web`, opening a session moves it to its directory, and a new one asks where it runs.

    The process's working directory stays where CLAI started; plugins read the session's from `host.workspace`.
    """
    launch, other, gone = tmp_path / 'launch', tmp_path / 'other', tmp_path / 'gone'
    launch.mkdir()
    (other / '.clai').mkdir(parents=True)
    (other / '.clai' / 'settings.json').write_text('{"instructions": "Work in other.", "model": "test"}')
    monkeypatch.chdir(launch)
    web_ = make_web(
        Agent(TestModel(call_tools=[], custom_output_text='ok'), deps_type=type(None)), extra=[REPO_CONTEXT]
    )
    follow_workspaces(web_)
    shell = web_.shell
    here = saved_in(web_, launch.resolve(), 'Here')
    there = saved_in(web_, other.resolve(), 'There')
    lost = saved_in(web_, gone, 'Lost')
    plugin = web_.plugin
    repo_context = shell.loader._loaded['repo_context']  # pyright: ignore[reportPrivateUsage]
    assert web_.get('/api/state').json()['follows_workspace'] is True
    assert [web_.get(f'/api/sessions/{id}').json()['openable'] for id in (here, there, lost)] == [True, True, False]
    rows = {
        entry['id']: entry['openable']
        for project in web_.get('/api/sessions').json()
        for worktree in project['worktrees']
        for entry in worktree['sessions']
    }
    assert [rows[id] for id in (here, there, lost)] == [True, True, False], 'each row says what its transcript says'
    assert web_.get(f'/api/sessions/{lost}').json()['missing'] is True
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()

        assert web_.post(f'/api/sessions/{there}/open').status_code == 200
        snapshot = receive_until(socket, Snapshot)[1]
        assert (snapshot.session.id, snapshot.session.workspace) == (there, str(other.resolve()))
        assert (Path.cwd(), shell.session.workspace) == (launch, str(other.resolve()))
        assert (shell.context.settings.instructions, shell.session.instructions) == ('Work in other.',) * 2
        assert (shell.session.model, 'test' in shell.context.store.models()) == ('test', True)
        loaded = shell.loader._loaded  # pyright: ignore[reportPrivateUsage]
        assert list(loaded) == ['repo_context', 'web'], 'the page keeps its plugin, still after the others'
        assert (loaded['web'].plugin, loaded['repo_context'] is repo_context) == (plugin, False)
        assert loaded['repo_context'].plugin.host.workspace == other.resolve()

        assert web_.post('/api/sessions/new', {'workspace': str(launch)}).status_code == 200
        snapshot = receive_until(socket, Snapshot)[1]
        assert (snapshot.items, snapshot.session.workspace) == ([], str(launch.resolve()))
        assert snapshot.session.id not in (here, there) and Path.cwd() == launch.resolve()
        assert (shell.context.settings.instructions, shell.session.model) == ('', resolve_settings({}).model), (
            "the other project's settings are gone"
        )
        assert receive_until(socket, Notice)[1].text == f'New session in {launch.resolve()}.'

        (tmp_path / 'file').touch()
        for workspace, problem in (
            ('relative', 'absolute path'),
            (str(gone), 'does not exist'),
            (str(tmp_path / 'file'), 'is not a directory'),
        ):
            refused = web_.post('/api/sessions/new', {'workspace': workspace})
            assert (refused.status_code, problem in refused.json()['detail']) == (400, True)
        refused = web_.post(f'/api/sessions/{lost}/open')
        assert (refused.status_code, refused.json()['detail']) == (400, f'{gone} does not exist.')

        assert web_.post(f'/api/sessions/{here}/open').status_code == 200
        assert receive_until(socket, Snapshot)[1].session.id == here
        copy = web_.post(f'/api/sessions/{there}/fork').json()['id']
        snapshot = receive_until(socket, Snapshot)[1]
        assert (snapshot.session.id, snapshot.session.workspace) == (copy, str(other.resolve()))

        assert receive_until(socket, Notice)[1].text.startswith('Resumed There (fork)')
        web_.post('/api/input', {'text': '/workspace'})
        assert receive_until(socket, Notice)[1].text == f'This session works in {other.resolve()}.'
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_a_session_moves_only_between_turns(
    make_web: 'MakeWeb', tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A move waits for whatever runs; a project file that does not load, or the current directory, changes nothing."""
    broken = tmp_path / 'broken'
    (broken / '.clai').mkdir(parents=True)
    (broken / '.clai' / 'settings.json').write_text('{"model": 1}')
    monkeypatch.chdir(tmp_path)
    web_ = make_web(Agent(TestModel()), extra=[REPO_CONTEXT])
    workspace = follow_workspaces(web_, lambda project: Settings(model=None))
    shell = web_.shell
    shell.session._running = True  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(RuntimeError, match='Switch directories once it ends'):
        web_.portal.call(workspace.move, str(broken))
    shell.session._running = False  # pyright: ignore[reportPrivateUsage]
    workspace.busy = shell.background_running
    record = task(conversation_id=shell.session.conversation_id)
    shell.tasks.owner.records[record.id] = record
    with pytest.raises(RuntimeError, match='Switch directories once it ends'):
        web_.portal.call(workspace.move, str(broken))
    del shell.tasks.owner.records[record.id]
    with pytest.raises(ValueError, match=r'settings\.json'):
        web_.portal.call(workspace.move, str(broken))
    plugin = web_.plugin
    web_.portal.call(workspace.move, str(tmp_path))
    assert (Path.cwd(), shell.session.workspace) == (tmp_path, str(tmp_path.resolve()))
    (tmp_path / 'next').mkdir()
    web_.portal.call(workspace.move, str(tmp_path / 'next'))
    assert (Path.cwd(), shell.session.workspace, shell.session.model) == (
        tmp_path,
        str((tmp_path / 'next').resolve()),
        None,
    )
    loaded = shell.loader._loaded  # pyright: ignore[reportPrivateUsage]
    assert loaded['web'].plugin is plugin
    repo_context = loaded['repo_context'].plugin
    web_.portal.call(lambda: shell.loader.restart(project=()))
    assert loaded['repo_context'].plugin is not repo_context, 'a session-scoped plugin loads again'
    assert loaded['web'].plugin is plugin, 'a process-scoped plugin follows no session directory'
    assert list(loaded)[-1] == 'web'


def test_a_terminal_session_stays_where_it_started(make_web: 'MakeWeb', tmp_path: Path) -> None:
    """Without `clai2 --web`, CLAI works where it started: no `/workspace`, and a session saved elsewhere is read-only."""
    web_ = make_web(Agent(TestModel()))
    shell = web_.shell
    elsewhere = saved_in(web_, tmp_path, 'Elsewhere')
    assert not shell.sessions.follows_workspace and 'workspace' not in shell.commands
    assert web_.get('/api/state').json()['follows_workspace'] is False
    assert web_.get(f'/api/sessions/{elsewhere}').json() | {'items': []} == {
        'id': elsewhere,
        'title': 'Elsewhere',
        'workspace': str(tmp_path),
        'items': [],
        'openable': False,
        'missing': False,
        'owner_pid': None,
    }
    with pytest.raises(ValueError, match='Select it in /resume to confirm'):
        web_.portal.call(shell.sessions.resume, elsewhere)
    with web_.client.websocket_connect(
        f'/ws?token={TOKEN}&protocol={PROTOCOL}', headers={'origin': 'http://testserver'}
    ) as socket:
        socket.receive_text()
        running = web_.run()
        first = shell.session.conversation_id
        web_.post('/api/input', {'text': '/workspace'})
        assert receive_until(socket, Notice)[1].text == 'Unknown command /workspace. Use /help.'
        refused = web_.post('/api/sessions/new', {'workspace': str(tmp_path)})
        assert (refused.status_code, refused.json()['detail']) == (
            400,
            f'This CLAI works in {shell.session.workspace}. Start CLAI in {tmp_path.resolve()}.',
        )
        assert web_.post('/api/sessions/new', {'workspace': shell.session.workspace}).status_code == 200
        snapshot = receive_until(socket, Snapshot)[1]
        assert (snapshot.session.id != first, snapshot.session.workspace) == (True, shell.session.workspace)
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)


def test_a_page_from_before_the_tag_renames_gets_the_old_tags(make_web: MakeWeb) -> None:
    """A tab open across the `/update` that renamed the tags sends no `protocol`, so it gets the tags it knows."""
    web_ = make_web(Agent(TestModel(call_tools=[], custom_output_text='answer'), deps_type=type(None)))
    with web_.client.websocket_connect(f'/ws?token={TOKEN}', headers={'origin': 'http://testserver'}) as socket:
        socket.receive_text()
        running = web_.run()
        web_.post('/api/input', {'text': 'hello'})
        tags: list[JsonValue] = []
        while 'turn_end' not in tags:
            tags.append(json.loads(socket.receive_text())['type'])
        assert 'turn_start' in tags
        assert not {'turn_started', 'turn_ended', 'text_delta'} & set(map(str, tags))
        web_.post('/api/input', {'text': '/exit'})
        running.result(timeout=10)

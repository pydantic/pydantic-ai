"""The FastAPI app: the built UI, a JSON API for everything the user does, and one WebSocket for what happens.

Anyone who can reach the server can make the agent run commands, so every request is checked:
the `Host` must be the address the server listens on (no DNS rebinding), and the per-launch token
must come with it. API calls carry it in a header, which another site's page cannot set; the page
and its assets carry it in the query, as the printed link does; the WebSocket takes it as a query
parameter and must also come from the server's own origin. No cookie holds it: browsers share
cookies between every port of a host, so one would reach any other local server.
"""

import re
import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from pathlib import Path
from typing import Literal, TypeVar
from urllib.parse import parse_qs

import anyio
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, JsonValue, ValidationError
from starlette.datastructures import Headers

from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins.web.chat_commands import orchestrator_completions
from pydantic_clai2.builtin_plugins.web.framing import FRAME_SOURCES
from pydantic_clai2.builtin_plugins.web.live import RESYNC
from pydantic_clai2.builtin_plugins.web.orchestrator import Orchestrator
from pydantic_clai2.builtin_plugins.web.protocol import (
    CompletionInfo,
    MenuEdited,
    MenuInfo,
    OrchestratorMessage,
    OrchestratorState,
    PanelInfo,
    PluginRow,
    Project,
    TaskTranscript,
    Transcript,
    WorkspaceOpened,
    WorkspacesInfo,
    dump,
)
from pydantic_clai2.builtin_plugins.web.remote import PluginAction, Remote, Switch, TaskAction
from pydantic_clai2.builtin_plugins.web.workspaces import listing, open_workspace

STATIC = Path(__file__).with_name('static')
HEADER = 'x-clai-token'
UNBUILT = 'The web UI is not built in this checkout: run `npm ci && npm run build` in src/pydantic_clai2/web-ui.'

Scope = MutableMapping[str, object]
Message = MutableMapping[str, object]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]
ResultT = TypeVar('ResultT')


class Guard:
    """Refuse any request without the token, for another host name, or, for the WebSocket, from another origin."""

    def __init__(self, app: ASGIApp, *, token: str, hosts: frozenset[str]) -> None:
        self.app = app
        self.token = token
        self.hosts = hosts
        self.origins = frozenset(f'http://{host}' for host in hosts)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        kind = scope['type']
        if kind not in ('http', 'websocket'):
            await self.app(scope, receive, send)
            return
        if self._allowed(kind, str(scope['path']), Headers(scope=scope), _query(scope)):
            await self.app(scope, receive, send)
        elif kind == 'websocket':
            await send({'type': 'websocket.close', 'code': 1008})
        else:
            refused = Response(
                "Open the address CLAI printed: it carries this launch's access token.",
                status_code=403,
                media_type='text/plain',
            )
            await refused(scope, receive, send)

    def _allowed(self, kind: object, path: str, headers: Headers, query: dict[str, list[str]]) -> bool:
        if headers.get('host') not in self.hosts:
            return False
        if kind == 'websocket':
            return headers.get('origin') in self.origins and self._valid(query.get('token', [''])[0])
        if path.startswith('/api/'):
            return self._valid(headers.get(HEADER, ''))
        return self._valid(query.get('token', [''])[0])

    def _valid(self, token: str) -> bool:
        return secrets.compare_digest(token.encode(), self.token.encode())


def _query(scope: Scope) -> dict[str, list[str]]:
    raw = scope.get('query_string', b'')
    return parse_qs(raw.decode() if isinstance(raw, bytes) else '')


class Input(BaseModel):
    """A prompt or `/command` to enter."""

    text: str = Field(max_length=200_000)


class Answer(BaseModel):
    """The answer to one question: picked labels, or a typed answer instead."""

    header: str
    selected: list[str] = []
    custom: str | None = None


class Answers(BaseModel):
    """Answers to every question of a request, or `None` to decline them all."""

    answers: list[Answer] | None = None


class RequestAnswers(Answers):
    """Answers to the orchestrator's question request `id`."""

    id: str


class Rename(BaseModel):
    """A session's new title."""

    title: str = Field(max_length=200)


class ModelChoice(BaseModel):
    """A saved model or chain to select, as `/model NAME` does."""

    name: str = Field(min_length=1, max_length=200)


class OrchestratorModelChoice(BaseModel):
    """The orchestrator's model, as its `/model NAME` picks it; `None` follows the session's model again."""

    name: str | None = Field(default=None, min_length=1, max_length=200)


class SwitchChoice(BaseModel):
    """Turn code mode or speculative execution on or off."""

    key: Switch
    on: bool


class Interrupted(BaseModel):
    """Whether something was running to interrupt."""

    interrupted: bool


class PluginChanges(BaseModel):
    """Settings to change, by their saved names, and settings to reset to their defaults; the rest are kept."""

    changes: dict[str, JsonValue] = {}
    reset: list[str] = []


class FieldValue(BaseModel):
    """A settings menu row's new value, as typed in its menu; empty or `None` resets it."""

    value: str | None = Field(default=None, max_length=100_000)


class DialogAnswer(BaseModel):
    """A dialog's answer: the chosen value, the typed text, or `yes`; `None` backs out."""

    value: str | None = Field(default=None, max_length=100_000)


class NewSession(BaseModel):
    """Where a new session runs: an absolute directory, or `None` for the current session's."""

    workspace: str | None = Field(default=None, max_length=4096)


class Forked(BaseModel):
    """The copy a fork saved."""

    id: str


class Steered(BaseModel):
    """Whether the task was running to take the message; a finished one can be resumed instead."""

    steered: bool


def create_app(*, remote: Remote, token: str, hosts: frozenset[str], static: Path = STATIC) -> ASGIApp:
    """The web UI for `remote`, accepting requests only for `hosts` (`HOST:PORT`) with `token`.

    Every handler is `async`, so it runs on the shell's event loop, never in a worker thread.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    _page(app, token=token, static=static)
    _prompt(app, remote)
    _sessions(app, remote)
    _plugins(app, remote)
    _tasks(app, remote)
    _panels(app, remote)
    _workspaces(app, remote)
    _orchestrator(app, remote.orchestrator)
    _events(app, remote)
    if (static / 'assets').is_dir():
        app.mount('/assets', StaticFiles(directory=static / 'assets'), name='assets')
    return Guard(app, token=token, hosts=hosts)


_ASSET = re.compile(r'((?:src|href)="/assets/[^"?]+)"')


def _page(app: FastAPI, *, token: str, static: Path) -> None:
    """The single page, its asset links and a meta tag carrying the token, which is URL-safe."""

    @app.get('/', include_in_schema=False)
    async def index() -> Response:
        index = static / 'index.html'
        if not index.is_file():
            return Response(UNBUILT, status_code=503, media_type='text/plain')
        page = index.read_text(encoding='utf-8')
        page = _ASSET.sub(rf'\1?token={token}"', page)
        page = page.replace('</head>', f'<meta name="clai-token" content="{token}"></head>', 1)
        return HTMLResponse(
            page,
            headers={
                'Cache-Control': 'no-store',
                'Content-Security-Policy': (
                    "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                    # Side panels frame only pages whose own headers allow it, which the server checks first.
                    f"connect-src 'self'; frame-src {FRAME_SOURCES}; "
                    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
                ),
                'Referrer-Policy': 'no-referrer',
                'X-Content-Type-Options': 'nosniff',
            },
        )


def _prompt(app: FastAPI, remote: Remote) -> None:
    """What the composer does: enter input, interrupt, complete, pick a model, answer questions."""

    @app.get('/api/state')
    async def state() -> Response:
        return Response(dump(remote.snapshot()), media_type='application/json')

    @app.post('/api/input', status_code=204)
    async def send(body: Input) -> None:
        _run(lambda: remote.send(body.text))

    @app.post('/api/interrupt')
    async def interrupt() -> Interrupted:
        return Interrupted(interrupted=remote.host.interrupt())

    @app.get('/api/complete')
    async def complete(text: str) -> list[CompletionInfo]:
        return await remote.complete(text)

    @app.get('/api/models')
    async def models() -> list[str]:
        return await remote.models()

    @app.post('/api/model', status_code=204)
    async def model(body: ModelChoice) -> None:
        _run(lambda: remote.send(f'/model {body.name}'))

    @app.post('/api/switch', status_code=204)
    async def switch(body: SwitchChoice) -> None:
        _run(lambda: remote.switch(body.key, body.on))

    @app.post('/api/questions/{request_id}', status_code=204)
    async def answer(request_id: str, body: Answers) -> None:
        answers = None if body.answers is None else [_answer(answer) for answer in body.answers]
        _run(lambda: remote.answer(request_id, answers))

    @app.post('/api/dialogs/{dialog_id}', status_code=204)
    async def answer_dialog(dialog_id: str, body: DialogAnswer) -> None:
        _run(lambda: remote.answer_dialog(dialog_id, body.value))


def _answer(answer: Answer) -> AskUserAnswer:
    if answer.custom is not None:
        return AskUserAnswer(header=answer.header, custom_answer=answer.custom)
    return AskUserAnswer(header=answer.header, selected=tuple(answer.selected))


def _sessions(app: FastAPI, remote: Remote) -> None:
    """The sidebar: list, read, open, start, rename, fork, and delete saved sessions."""

    @app.get('/api/sessions')
    async def sessions(q: str = '') -> list[Project]:
        return await _wait(remote.projects(q))

    @app.post('/api/sessions/new', status_code=204)
    async def new(body: NewSession | None = None) -> None:
        _run(lambda: remote.new(None if body is None else body.workspace))

    @app.get('/api/sessions/{conversation_id}')
    async def transcript(conversation_id: str) -> Transcript:
        return await _wait(remote.transcript(conversation_id))

    @app.post('/api/sessions/{conversation_id}/open', status_code=204)
    async def open_session(conversation_id: str) -> None:
        _run(lambda: remote.open(conversation_id))

    @app.post('/api/sessions/{conversation_id}/rename', status_code=204)
    async def rename(conversation_id: str, body: Rename) -> None:
        await _wait(remote.rename(conversation_id, body.title))

    @app.post('/api/sessions/{conversation_id}/fork')
    async def fork(conversation_id: str) -> Forked:
        return Forked(id=await _wait(remote.fork(conversation_id)))

    @app.delete('/api/sessions/{conversation_id}', status_code=204)
    async def delete(conversation_id: str) -> None:
        await _wait(remote.delete(conversation_id))


def _plugins(app: FastAPI, remote: Remote) -> None:
    """The plugins panel: list plugins, turn them on and off, reload them, and edit their settings."""

    @app.get('/api/plugins')
    async def plugins() -> list[PluginRow]:
        return _run(remote.plugins)

    # Before the action route, which would otherwise take `settings` and `menu` for actions.
    @app.post('/api/plugins/{name}/settings', status_code=204)
    async def settings(name: str, body: PluginChanges) -> None:
        await _wait(remote.configure_plugin(name, body.changes, body.reset))

    @app.post('/api/plugins/{name}/menu', status_code=204)
    async def menu(name: str) -> None:
        _run(lambda: remote.open_plugin_menu(name))

    @app.get('/api/plugins/{name}/fields')
    async def fields(name: str) -> MenuInfo:
        return await _wait(remote.plugin_menu(name))

    @app.post('/api/plugins/{name}/fields/{key}')
    async def set_field(name: str, key: str, body: FieldValue) -> MenuEdited:
        return await _wait(remote.edit_menu(name, key, body.value))

    @app.post('/api/plugins/{name}/fields/{key}/action')
    async def run_action(name: str, key: str) -> MenuEdited:
        return await _wait(remote.run_menu_action(name, key))

    @app.post('/api/plugins/{name}/{action}', status_code=204)
    async def act(name: str, action: PluginAction) -> None:
        await _wait(remote.plugin_action(name, action))


def _tasks(app: FastAPI, remote: Remote) -> None:
    """The Workers tree: read a delegated task's history, steer it, and stop, background, or resume it."""

    @app.get('/api/tasks/{task_id}')
    async def task(task_id: str) -> TaskTranscript:
        return _run(lambda: remote.task_transcript(task_id))

    # Before the action route, which would otherwise take `steer` for an action.
    @app.post('/api/tasks/{task_id}/steer')
    async def steer(task_id: str, body: Input) -> Steered:
        return Steered(steered=_run(lambda: remote.steer(task_id, body.text)))

    @app.post('/api/tasks/{task_id}/{action}', status_code=204)
    async def act(task_id: str, action: TaskAction) -> None:
        await _wait(remote.task_action(task_id, action))


def _panels(app: FastAPI, remote: Remote) -> None:
    """Plugins' side panels, as they are now; the page asks again when a turn starts or ends."""

    @app.get('/api/panels')
    async def panels(request: Request) -> list[PanelInfo]:
        # `Guard` admitted only the server's own addresses, so `Host` names the origin this page came from.
        return await remote.panels(origin=f'http://{request.headers["host"]}')


def _workspaces(app: FastAPI, remote: Remote) -> None:
    """The sidebar's Workspaces section: list them, and open one's own page."""

    @app.get('/api/workspaces')
    async def workspaces() -> WorkspacesInfo:
        return await listing(remote.host.plugins)

    @app.post('/api/workspaces/{workspace_id}/open')
    async def open_one(workspace_id: str) -> WorkspaceOpened:
        return await _wait(open_workspace(remote.host.plugins, workspace_id))


def _orchestrator(app: FastAPI, orchestrator: Orchestrator) -> None:
    """The orchestrator chat: its own conversation, agent, model, and commands, streamed on the same WebSocket."""

    @app.get('/api/orchestrator/state')
    async def state() -> OrchestratorState:
        return await orchestrator.state()

    @app.post('/api/orchestrator/input', status_code=204)
    async def send(body: Input) -> None:
        await _wait(orchestrator.send(body.text))

    @app.post('/api/orchestrator/interrupt')
    async def interrupt() -> Interrupted:
        return Interrupted(interrupted=orchestrator.interrupt())

    @app.get('/api/orchestrator/complete')
    async def complete(text: str) -> list[CompletionInfo]:
        return await orchestrator_completions(orchestrator.host, text)

    @app.post('/api/orchestrator/model', status_code=204)
    async def model(body: OrchestratorModelChoice) -> None:
        await _wait(orchestrator.select_model(body.name))

    @app.post('/api/orchestrator/answer', status_code=204)
    async def answer(body: RequestAnswers) -> None:
        answers = None if body.answers is None else [_answer(answer) for answer in body.answers]
        _run(lambda: orchestrator.answer(body.id, answers))

    @app.post('/api/orchestrator/clear', status_code=204)
    async def clear() -> None:
        await _wait(orchestrator.clear())


def _events(app: FastAPI, remote: Remote) -> None:
    """The WebSocket: a snapshot on connect, then every message the session publishes."""

    @app.websocket('/ws')
    async def events(socket: WebSocket) -> None:
        await socket.accept()
        # Subscribe first, so nothing published between the snapshot and the subscription is lost.
        with remote.hub.subscribe() as queue:
            await socket.send_text(dump(remote.snapshot()))
            await socket.send_text(dump(OrchestratorMessage(message=await remote.orchestrator.state())))
            async with anyio.create_task_group() as connection:

                async def forward() -> None:
                    try:
                        while (text := await queue.get()) != RESYNC:
                            await socket.send_text(text)
                        # The browser fell too far behind; it reconnects and starts from a fresh snapshot.
                        await socket.close(code=1013)
                    except (WebSocketDisconnect, RuntimeError):  # pragma: no cover -- races the receive loop below.
                        # The browser left mid-send.
                        pass
                    connection.cancel_scope.cancel()

                connection.start_soon(forward)
                try:
                    # The browser sends nothing; this only notices it leave.
                    while True:
                        await socket.receive_text()
                except WebSocketDisconnect:
                    connection.cancel_scope.cancel()


Failure = Literal[400, 404, 409]


def _status(error: Exception) -> Failure:
    if isinstance(error, LookupError):
        return 404
    if isinstance(error, RuntimeError):
        return 409
    return 400


def _run(action: Callable[[], ResultT]) -> ResultT:
    """Run an action, turning the errors it raises for the user into HTTP errors carrying their message."""
    try:
        return action()
    except (ValueError, LookupError, RuntimeError) as error:
        raise _http(error) from error


async def _wait(action: Awaitable[ResultT]) -> ResultT:
    try:
        return await action
    except (ValueError, LookupError, RuntimeError) as error:
        raise _http(error) from error


def _http(error: ValueError | LookupError | RuntimeError) -> HTTPException:
    """The error's message; settings a model rejected answer 422 with what is wrong with each, by its path."""
    if isinstance(error, ValidationError):
        problems = {'.'.join(str(part) for part in problem['loc']): problem['msg'] for problem in error.errors()}
        return HTTPException(422, {'message': 'Some settings are not valid.', 'errors': problems})
    return HTTPException(_status(error), str(error))

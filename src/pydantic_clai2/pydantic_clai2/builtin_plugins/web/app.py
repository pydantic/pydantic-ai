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
from typing import Annotated, Literal, TypeVar
from urllib.parse import parse_qs

import anyio
from anyio import to_thread
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError
from starlette.datastructures import Headers

from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins.web.channel import SessionChannel, Switch
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
    Watch,
    WorkspaceOpened,
    WorkspacesInfo,
    dump,
)
from pydantic_clai2.builtin_plugins.web.service import PluginAction, TaskAction, WebService
from pydantic_clai2.builtin_plugins.web.workspaces import listing, open_workspace
from pydantic_clai2.runtime.image_codec import decode_images

STATIC = Path(__file__).with_name('static')
HEADER = 'x-clai-token'
UNBUILT = 'The web UI is not built in this checkout: run `npm ci && npm run build` in src/pydantic_clai2/web-ui.'

Scope = MutableMapping[str, object]
Message = MutableMapping[str, object]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]
ResultT = TypeVar('ResultT')
MenuPath = Annotated[tuple[str, ...], Query(alias='menu')]
"""`?menu=KEY`, once per row chosen from a declared settings menu to reach the menu that row opens."""


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


class PromptInput(Input):
    """Input for a session's prompt, with the images pasted for it, base64-encoded."""

    # Strings, decoded off the loop once their length is checked; see `send`.
    images: list[str] = []


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


class SwitchChoice(BaseModel):
    """Turn code mode or speculative execution on or off."""

    key: Switch
    on: bool


class Entered(BaseModel):
    """What entering input did: `menu` names the bare command whose menu the page shows instead of running it."""

    menu: str | None = None


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
    """Where a new session runs: an absolute directory, or `None` for the default session's."""

    workspace: str | None = Field(default=None, max_length=4096)


class Opened(BaseModel):
    """The conversation of the live session that opening or starting a session gave."""

    id: str


class Forked(BaseModel):
    """The copy a fork saved."""

    id: str


class Steered(BaseModel):
    """Whether a running turn or task took the message; a finished task can be resumed instead."""

    steered: bool


class Steering(Entered):
    """What Alt+Enter did: `steered` when the running turn took it, else it was entered as Enter enters it."""

    steered: bool


def create_app(*, service: WebService, token: str, hosts: frozenset[str], static: Path = STATIC) -> ASGIApp:
    """The web UI for `service`, accepting requests only for `hosts` (`HOST:PORT`) with `token`.

    Every handler is `async`, so it runs on the shell's event loop, never in a worker thread.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    _page(app, token=token, static=static)
    _prompt(app, service)
    _queue(app, service)
    _images(app, service)
    _sessions(app, service)
    _plugins(app, service)
    _commands(app, service)
    _tasks(app, service)
    _panels(app, service)
    _workspaces(app, service)
    _orchestrator(app, service.orchestrator)
    _events(app, service)
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


def _prompt(app: FastAPI, service: WebService) -> None:
    """What the composer does: enter input, interrupt, complete, pick a model, answer questions.

    Each route is there twice: under `/api/sessions/ID/` for the live session holding conversation ID,
    and unprefixed for the default session, as before CLAI ran several.
    """

    def session(conversation_id: str | None) -> SessionChannel:
        return _run(service.default if conversation_id is None else lambda: service.session(conversation_id))

    async def state(conversation_id: str | None = None) -> Response:
        return Response(dump(session(conversation_id).snapshot()), media_type='application/json')

    async def send(body: PromptInput, conversation_id: str | None = None) -> Entered:
        channel = session(conversation_id)
        # Off the loop: decoding large images would stall every stream meanwhile.
        images = await _wait(to_thread.run_sync(decode_images, body.images)) if body.images else []
        return Entered(menu=_run(lambda: channel.send(body.text, images=images)))

    async def interrupt(conversation_id: str | None = None) -> Interrupted:
        return Interrupted(interrupted=session(conversation_id).session.interrupt())

    async def complete(text: str, conversation_id: str | None = None) -> list[CompletionInfo]:
        return await session(conversation_id).complete(text)

    async def model(body: ModelChoice, conversation_id: str | None = None) -> None:
        channel = session(conversation_id)
        _run(lambda: channel.send(f'/model {body.name}'))

    async def switch(body: SwitchChoice, conversation_id: str | None = None) -> None:
        channel = session(conversation_id)
        _run(lambda: channel.switch(body.key, body.on))

    async def answer(request_id: str, body: Answers, conversation_id: str | None = None) -> None:
        answers = None if body.answers is None else [_answer(answer) for answer in body.answers]
        if conversation_id is None:
            _run(lambda: service.answerer.reply(request_id, answers))
        else:
            channel = session(conversation_id)
            _run(lambda: channel.answer(request_id, answers))

    async def answer_dialog(dialog_id: str, body: DialogAnswer, conversation_id: str | None = None) -> None:
        if conversation_id is not None:
            session(conversation_id)
        _run(lambda: service.answer_dialog(dialog_id, body.value))

    for prefix in ('/api', '/api/sessions/{conversation_id}'):
        app.get(f'{prefix}/state')(state)
        app.post(f'{prefix}/input')(send)
        app.post(f'{prefix}/interrupt')(interrupt)
        app.get(f'{prefix}/complete')(complete)
        app.post(f'{prefix}/model', status_code=204)(model)
        app.post(f'{prefix}/switch', status_code=204)(switch)
        app.post(f'{prefix}/questions/{{request_id}}', status_code=204)(answer)
        app.post(f'{prefix}/dialogs/{{dialog_id}}', status_code=204)(answer_dialog)

    @app.get('/api/models')
    async def models() -> list[str]:
        return await service.models()


def _queue(app: FastAPI, service: WebService) -> None:
    """Steer the running turn, and edit, steer, or remove the input queued behind it, as the prompt does.

    Each route is there twice, as the composer's are: under `/api/sessions/ID/` for the live session
    holding conversation ID, and unprefixed for the default session.
    """

    def session(conversation_id: str | None) -> SessionChannel:
        return _run(service.default if conversation_id is None else lambda: service.session(conversation_id))

    async def steer(body: Input, conversation_id: str | None = None) -> Steering:
        channel = session(conversation_id)
        if _run(lambda: channel.steer(body.text)):
            return Steering(steered=True)
        return Steering(steered=False, menu=_run(lambda: channel.send(body.text)))

    async def edit_queued(input_id: str, body: Input, conversation_id: str | None = None) -> None:
        channel = session(conversation_id)
        _run(lambda: channel.edit_queued(input_id, body.text))

    async def remove_queued(input_id: str, conversation_id: str | None = None) -> None:
        channel = session(conversation_id)
        _run(lambda: channel.remove_queued(input_id))

    async def steer_queued(input_id: str, conversation_id: str | None = None) -> Steered:
        channel = session(conversation_id)
        return Steered(steered=_run(lambda: channel.steer_queued(input_id)))

    for prefix in ('/api', '/api/sessions/{conversation_id}'):
        app.post(f'{prefix}/steer')(steer)
        app.put(f'{prefix}/queue/{{input_id}}', status_code=204)(edit_queued)
        app.delete(f'{prefix}/queue/{{input_id}}', status_code=204)(remove_queued)
        app.post(f'{prefix}/queue/{{input_id}}/steer')(steer_queued)


def _images(app: FastAPI, service: WebService) -> None:
    """The images of a session's prompts, by the IDs its transcript items list, for the page to show.

    Each route is there twice, as the composer's are. The page fetches them with the token header and
    shows them as `data:` URLs, which its content policy allows.
    """

    async def image(image_id: str, conversation_id: str | None = None) -> Response:
        channel = _run(service.default if conversation_id is None else lambda: service.session(conversation_id))
        # On the loop: the session hashed each image once already, when its transcript listed it.
        found = channel.image(image_id)
        if found is None:
            raise HTTPException(status_code=404, detail='No such image in this session.')
        # An ID is the digest of the bytes, so they never change.
        return Response(
            found.data, media_type=found.media_type, headers={'Cache-Control': 'private, max-age=31536000, immutable'}
        )

    for prefix in ('/api', '/api/sessions/{conversation_id}'):
        app.get(f'{prefix}/images/{{image_id}}')(image)


def _answer(answer: Answer) -> AskUserAnswer:
    if answer.custom is not None:
        return AskUserAnswer(header=answer.header, custom_answer=answer.custom)
    return AskUserAnswer(header=answer.header, selected=tuple(answer.selected))


def _sessions(app: FastAPI, service: WebService) -> None:
    """The sidebar: list, read, open, start, rename, fork, archive, and delete saved sessions."""

    @app.get('/api/sessions')
    async def sessions(q: str = '', include_archived: bool = False) -> list[Project]:
        return await _wait(service.projects(q, include_archived=include_archived))

    @app.post('/api/sessions/new')
    async def new(body: NewSession | None = None) -> Opened:
        return Opened(id=await _wait(service.new(None if body is None else body.workspace)))

    @app.get('/api/sessions/{conversation_id}')
    async def transcript(conversation_id: str) -> Transcript:
        return await _wait(service.transcript(conversation_id))

    @app.post('/api/sessions/{conversation_id}/open')
    async def open_session(conversation_id: str) -> Opened:
        return Opened(id=await _wait(service.open(conversation_id)))

    @app.post('/api/sessions/{conversation_id}/rename', status_code=204)
    async def rename(conversation_id: str, body: Rename) -> None:
        await _wait(service.rename(conversation_id, body.title))

    @app.post('/api/sessions/{conversation_id}/fork')
    async def fork(conversation_id: str) -> Forked:
        return Forked(id=await _wait(service.fork(conversation_id)))

    @app.post('/api/sessions/{conversation_id}/archive', status_code=204)
    async def archive(conversation_id: str) -> None:
        await _wait(service.archive(conversation_id, archived=True))

    @app.post('/api/sessions/{conversation_id}/unarchive', status_code=204)
    async def unarchive(conversation_id: str) -> None:
        await _wait(service.archive(conversation_id, archived=False))

    @app.delete('/api/sessions/{conversation_id}', status_code=204)
    async def delete(conversation_id: str) -> None:
        await _wait(service.delete(conversation_id))


def _plugins(app: FastAPI, service: WebService) -> None:
    """The plugins panel: list plugins, turn them on and off, reload them, and edit their settings."""

    @app.get('/api/plugins')
    async def plugins() -> list[PluginRow]:
        return _run(service.plugins)

    # Before the action route, which would otherwise take `settings` and `menu` for actions.
    @app.post('/api/plugins/{name}/settings', status_code=204)
    async def settings(name: str, body: PluginChanges) -> None:
        await _wait(service.configure_plugin(name, body.changes, body.reset))

    @app.post('/api/plugins/{name}/menu', status_code=204)
    async def menu(name: str) -> None:
        _run(lambda: service.open_plugin_menu(name))

    @app.get('/api/plugins/{name}/fields')
    async def fields(name: str, path: MenuPath = ()) -> MenuInfo:
        return await _wait(service.plugin_menu(name, path=path))

    @app.post('/api/plugins/{name}/fields/{key}')
    async def set_field(name: str, key: str, body: FieldValue, path: MenuPath = ()) -> MenuEdited:
        return await _wait(service.edit_menu(name, key, body.value, path=path))

    @app.post('/api/plugins/{name}/fields/{key}/action')
    async def run_action(name: str, key: str, path: MenuPath = ()) -> MenuEdited:
        return await _wait(service.run_menu_action(name, key, path=path))

    @app.post('/api/plugins/{name}/{action}', status_code=204)
    async def act(name: str, action: PluginAction) -> None:
        await _wait(service.plugin_action(name, action))


def _commands(app: FastAPI, service: WebService) -> None:
    """Commands' own menus, such as `/set`'s, shown in the page instead of the terminal.

    Each route is there twice, as the composer's are: under `/api/sessions/ID/` for the live session
    holding conversation ID, whose commands build the menu, and unprefixed for the default session.
    """

    def session(conversation_id: str | None) -> SessionChannel:
        return _run(service.default if conversation_id is None else lambda: service.session(conversation_id))

    async def fields(name: str, path: MenuPath = (), conversation_id: str | None = None) -> MenuInfo:
        return await _wait(service.command_menu(session(conversation_id), name, path=path))

    async def set_field(
        name: str, key: str, body: FieldValue, path: MenuPath = (), conversation_id: str | None = None
    ) -> MenuEdited:
        return await _wait(service.edit_command_menu(session(conversation_id), name, key, body.value, path=path))

    async def run_action(name: str, key: str, path: MenuPath = (), conversation_id: str | None = None) -> MenuEdited:
        return await _wait(service.run_command_menu_action(session(conversation_id), name, key, path=path))

    for prefix in ('/api', '/api/sessions/{conversation_id}'):
        app.get(f'{prefix}/commands/{{name}}/fields')(fields)
        app.post(f'{prefix}/commands/{{name}}/fields/{{key}}')(set_field)
        app.post(f'{prefix}/commands/{{name}}/fields/{{key}}/action')(run_action)


def _tasks(app: FastAPI, service: WebService) -> None:
    """The Workers tree: read a delegated task's history, steer it, and stop, background, or resume it."""

    @app.get('/api/tasks/{task_id}')
    async def task(task_id: str) -> TaskTranscript:
        return _run(lambda: service.task_transcript(task_id))

    # Before the action route, which would otherwise take `steer` for an action.
    @app.post('/api/tasks/{task_id}/steer')
    async def steer(task_id: str, body: Input) -> Steered:
        return Steered(steered=_run(lambda: service.steer(task_id, body.text)))

    @app.post('/api/tasks/{task_id}/{action}', status_code=204)
    async def act(task_id: str, action: TaskAction) -> None:
        await _wait(service.task_action(task_id, action))


def _panels(app: FastAPI, service: WebService) -> None:
    """Plugins' side panels, as they are now; the page asks again when a turn starts or ends."""

    @app.get('/api/panels')
    async def panels(request: Request) -> list[PanelInfo]:
        # `Guard` admitted only the server's own addresses, so `Host` names the origin this page came from.
        return await service.panels(origin=f'http://{request.headers["host"]}')


def _workspaces(app: FastAPI, service: WebService) -> None:
    """The sidebar's Workspaces section: list them, and open one's own page."""

    @app.get('/api/workspaces')
    async def workspaces() -> WorkspacesInfo:
        return await listing(service.host.plugins)

    @app.post('/api/workspaces/{workspace_id}/open')
    async def open_one(workspace_id: str) -> WorkspaceOpened:
        return await _wait(open_workspace(service.host.plugins, workspace_id))


def _orchestrator(app: FastAPI, orchestrator: Orchestrator) -> None:
    """The orchestrator chat: its own conversation with its own agent, streamed on the same WebSocket."""

    @app.get('/api/orchestrator/state')
    async def state() -> OrchestratorState:
        return await orchestrator.state()

    @app.post('/api/orchestrator/input', status_code=204)
    async def send(body: Input) -> None:
        await _wait(orchestrator.send(body.text))

    @app.post('/api/orchestrator/interrupt')
    async def interrupt() -> Interrupted:
        return Interrupted(interrupted=orchestrator.interrupt())

    @app.post('/api/orchestrator/answer', status_code=204)
    async def answer(body: RequestAnswers) -> None:
        answers = None if body.answers is None else [_answer(answer) for answer in body.answers]
        _run(lambda: orchestrator.answer(body.id, answers))

    @app.post('/api/orchestrator/clear', status_code=204)
    async def clear() -> None:
        await _wait(orchestrator.clear())


def _events(app: FastAPI, service: WebService) -> None:
    """The WebSocket: the default session's snapshot, the orchestrator, and `live` on connect, then what is published.

    The browser may send `watch` to choose the live sessions it gets the messages of; until then it gets
    the default session's.
    """

    @app.websocket('/ws')
    async def events(socket: WebSocket) -> None:
        await socket.accept()
        # Subscribe first, so nothing published between the snapshot and the subscription is lost.
        with service.hub.subscribe() as client:
            try:
                try:
                    snapshot = dump(service.default().snapshot())
                except LookupError:
                    snapshot = None  # No session is live, as while CLAI exits; `live` says so.
                # Before the first `await`, so the default it follows is the one in the snapshot, and every
                # `live` already queued, or queued from here on, is no older than this one.
                service.follow(client, snapshots=False)
                live = dump(service.live_message())
                if snapshot is not None:
                    await socket.send_text(snapshot)
                await socket.send_text(dump(OrchestratorMessage(message=await service.orchestrator.state())))
                await socket.send_text(live)
                async with anyio.create_task_group() as connection:

                    async def forward() -> None:
                        try:
                            while (text := await client.queue.get()) != RESYNC:
                                await socket.send_text(text)
                            # The browser fell too far behind; it reconnects and starts from a fresh snapshot.
                            await socket.close(code=1013)
                        except (WebSocketDisconnect, RuntimeError):  # pragma: no cover -- races the receive loop.
                            # The browser left mid-send.
                            pass
                        connection.cancel_scope.cancel()

                    connection.start_soon(forward)
                    try:
                        while True:
                            text = await socket.receive_text()
                            try:
                                watch = _WATCH.validate_json(text)
                            except ValidationError:
                                continue  # Nothing else is sent; an older or newer page's message is ignored.
                            service.watch(client, watch.sessions)
                    except WebSocketDisconnect:
                        connection.cancel_scope.cancel()
            finally:
                service.unwatch(client)


_WATCH = TypeAdapter(Watch)


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

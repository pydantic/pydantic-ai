"""The FastAPI app: the built UI, a JSON API for everything the user does, and one WebSocket for what happens.

Anyone who can reach the server can make the agent run commands, so every request is checked:
the `Host` must be the address the server listens on (no DNS rebinding), and the per-launch token
must come with it. API calls carry it in a header, which another site's page cannot set; the page
carries it in the query, as the printed link does; the WebSocket takes it as a query parameter and
must also come from the server's own origin. No cookie holds it: browsers share cookies between
every port of a host, so one would reach any other local server. The page's assets are the public
build the package ships, so they need no token: their links carry their content's digest instead,
which lets the browser keep them across launches.
"""

import gzip
import hashlib
import mimetypes
import re
import secrets
from collections.abc import Awaitable, Callable, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeVar
from urllib.parse import parse_qs

import anyio
from anyio import to_thread
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field, JsonValue, TypeAdapter, ValidationError
from starlette.datastructures import Headers

from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins.web.changes import Changes, ChangeScope, session_changes
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
    dump,
    legacy_tags,
)
from pydantic_clai2.builtin_plugins.web.service import PluginAction, TaskAction, WebService
from pydantic_clai2.builtin_plugins.web.sidebar import completions
from pydantic_clai2.builtin_plugins.web.workspaces import open_workspace
from pydantic_clai2.runtime import git_actions
from pydantic_clai2.runtime.git_actions import GitStatus, PullRequestDraft
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
        if path.startswith('/assets/'):
            return True
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


class Archive(BaseModel):
    """The sessions to archive at once, such as every one of a worktree."""

    ids: list[str] = Field(max_length=1000)


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


class CommitFiles(BaseModel):
    """The changed files to commit, by path; `None` for every one."""

    files: list[str] | None = None


class CommitRequest(CommitFiles):
    """A commit to make: its message, and the files it takes."""

    message: str = Field(max_length=100_000)


class PushRequest(BaseModel):
    """`confirmed` allows pushing the default branch."""

    confirmed: bool = False


class PullRequestRequest(BaseModel):
    """The pull request to open."""

    title: str = Field(max_length=200)
    body: str = Field(default='', max_length=100_000)


class GitDone(BaseModel):
    """What a git action did, in a line, and the repository after it."""

    message: str
    status: GitStatus | None


class CommitDraft(BaseModel):
    """The commit message the session's model drafted."""

    message: str


class PullRequestOpened(BaseModel):
    """The pull request `gh` opened."""

    url: str


class Opened(BaseModel):
    """The conversation of the live session that opening or starting a session gave."""

    id: str


class Forked(BaseModel):
    """The copy a fork saved."""

    id: str


class Steered(BaseModel):
    """Whether a running turn or task took the message; a finished task can be prompted instead."""

    steered: bool


class Steering(Entered):
    """What Alt+Enter did: `steered` when the running turn took it, else it was entered as Enter enters it."""

    steered: bool


def create_app(*, service: WebService, token: str, hosts: frozenset[str], static: Path = STATIC) -> ASGIApp:
    """The web UI for `service`, accepting requests only for `hosts` (`HOST:PORT`) with `token`.

    Every handler is `async`, so it runs on the shell's event loop, never in a worker thread.
    """
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    assets = Assets(static / 'assets')
    _page(app, token=token, static=static, assets=assets)
    _assets(app, assets)
    _prompt(app, service)
    _queue(app, service)
    _images(app, service)
    _changes(app, service)
    _sessions(app, service)
    _plugins(app, service)
    _commands(app, service)
    _tasks(app, service)
    _panels(app, service)
    _workspaces(app, service)
    _git(app, service)
    _orchestrator(app, service.orchestrator)
    _events(app, service)
    return Guard(app, token=token, hosts=hosts)


@dataclass(frozen=True)
class Asset:
    """A built file, gzipped once, and the digest its links carry, so a link changes only with the file."""

    body: bytes
    gzipped: bytes
    version: str


class Assets:
    """The built UI's files, each read and gzipped off the loop once, until a rebuild changes it."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._built: dict[str, tuple[tuple[int, int], Asset]] = {}

    async def get(self, name: str) -> Asset | None:
        path = self.directory / name
        if path.parent != self.directory or not path.is_file():
            return None
        stat = path.stat()
        key = (stat.st_mtime_ns, stat.st_size)
        built = self._built.get(name)
        if built is None or built[0] != key:
            built = key, await to_thread.run_sync(_build_asset, path)
            self._built[name] = built
        return built[1]


def _build_asset(path: Path) -> Asset:
    """Read the built file at `path`, gzip it, and digest it."""
    body = path.read_bytes()
    return Asset(body=body, gzipped=gzip.compress(body, mtime=0), version=hashlib.sha256(body).hexdigest()[:16])


def _assets(app: FastAPI, assets: Assets) -> None:
    """The built UI's files, gzipped if the browser takes that, and cached for good under the page's versioned links."""

    @app.api_route('/assets/{name}', methods=['GET', 'HEAD'], include_in_schema=False)
    async def asset(name: str, request: Request, v: str = '') -> Response:
        found = await assets.get(name)
        if found is None:
            raise HTTPException(status_code=404)
        etag = f'"{found.version}"'
        headers = {
            'Cache-Control': 'public, max-age=31536000, immutable' if v == found.version else 'no-cache',
            'ETag': etag,
            'Vary': 'Accept-Encoding',
        }
        if request.headers.get('if-none-match') == etag:
            return Response(status_code=304, headers=headers)
        body = found.body
        if 'gzip' in request.headers.get('accept-encoding', ''):
            body = found.gzipped
            headers['Content-Encoding'] = 'gzip'
        return Response(body, media_type=mimetypes.guess_type(name)[0], headers=headers)


_ASSET = re.compile(r'((?:src|href)="/assets/([^"?/]+))"')


def _page(app: FastAPI, *, token: str, static: Path, assets: Assets) -> None:
    """The single page, its versioned asset links and a meta tag carrying the token, which is URL-safe."""

    @app.get('/', include_in_schema=False)
    async def index() -> Response:
        index = static / 'index.html'
        if not index.is_file():
            return Response(UNBUILT, status_code=503, media_type='text/plain')
        page = index.read_text(encoding='utf-8')
        versions = {name: await assets.get(name) for _, name in _ASSET.findall(page)}

        def link(match: re.Match[str]) -> str:
            found = versions[match[2]]
            return match[0] if found is None else f'{match[1]}?v={found.version}"'

        page = _ASSET.sub(link, page)
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
    """What the composer does: enter input, interrupt, complete, find `@` files, pick a model, answer questions.

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

    async def files(q: str = '', conversation_id: str | None = None) -> list[CompletionInfo]:
        return await session(conversation_id).files(q)

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
        app.get(f'{prefix}/files')(files)
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


def _changes(app: FastAPI, service: WebService) -> None:
    """The Changes panel: the files a session changed in a scope, each with its diff, or without for `diffs=false`.

    Each route is there twice, as the composer's are.
    """

    async def changes(
        scope: ChangeScope = 'session', diffs: bool = True, conversation_id: str | None = None
    ) -> Changes:
        session = _run(service.default if conversation_id is None else lambda: service.session(conversation_id)).session
        return await session_changes(session.conversation.messages, session.workspace, scope=scope, diffs=diffs)

    for prefix in ('/api', '/api/sessions/{conversation_id}'):
        app.get(f'{prefix}/changes')(changes)


def _answer(answer: Answer) -> AskUserAnswer:
    if answer.custom is not None:
        return AskUserAnswer(header=answer.header, custom_answer=answer.custom)
    return AskUserAnswer(header=answer.header, selected=tuple(answer.selected))


def _sessions(app: FastAPI, service: WebService) -> None:
    """The sidebar: list, read, open, start, rename, fork, archive, and delete saved sessions."""

    @app.get('/api/sessions')
    async def sessions(q: str = '', include_archived: bool = False, include_worktrees: bool = False) -> list[Project]:
        return await _wait(service.projects(q, include_archived=include_archived, include_worktrees=include_worktrees))

    @app.get('/api/directories')
    async def directories(path: str = '') -> list[Project]:
        """New session's completions of a typed path, so it reaches repositories and worktrees without sessions."""
        return await to_thread.run_sync(completions, path)

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

    @app.post('/api/sessions/archive', status_code=204)
    async def archive_all(body: Archive) -> None:
        await _wait(service.archive_all(body.ids))

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
    """The Workers tree: read a delegated task's history, steer or prompt it, and stop, background, or resume it."""

    @app.get('/api/tasks/{task_id}')
    async def task(task_id: str) -> TaskTranscript:
        return _run(lambda: service.task_transcript(task_id))

    # Before the action route, which would otherwise take `steer` and `prompt` for actions.
    @app.post('/api/tasks/{task_id}/steer')
    async def steer(task_id: str, body: Input) -> Steered:
        return Steered(steered=_run(lambda: service.steer(task_id, body.text)))

    @app.post('/api/tasks/{task_id}/prompt', status_code=204)
    async def prompt(task_id: str, body: Input) -> None:
        await _wait(service.prompt_task(task_id, body.text))

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
    """A workspace's session, which an orchestrator tool card opens in this page."""

    @app.post('/api/workspaces/{workspace_id}/open')
    async def open_one(workspace_id: str) -> Opened:
        return Opened(id=await _wait(open_workspace(service, workspace_id)))


def _git(app: FastAPI, service: WebService) -> None:
    """The git chip and its dialog: a live session's repository, and committing, pushing, or opening a pull request.

    Each acts in the session's own directory, and refuses one that no longer exists.
    """

    def workspace(conversation_id: str) -> Path:
        return _run(lambda: service.session(conversation_id)).session.workspace

    def model(conversation_id: str) -> git_actions.ModelSource:
        return _run(lambda: service.session(conversation_id)).session.conversation.resolved_model

    @app.get('/api/sessions/{conversation_id}/git')
    async def status(conversation_id: str) -> GitStatus | None:
        return await _wait(git_actions.git_status(workspace(conversation_id)))

    @app.post('/api/sessions/{conversation_id}/git/commit/draft')
    async def draft_commit(conversation_id: str, body: CommitFiles) -> CommitDraft:
        drafted = git_actions.draft_commit_message(
            workspace(conversation_id), model=model(conversation_id), files=body.files
        )
        return CommitDraft(message=await _wait(drafted))

    @app.post('/api/sessions/{conversation_id}/git/commit')
    async def commit(conversation_id: str, body: CommitRequest) -> GitDone:
        directory = workspace(conversation_id)
        done = await _wait(git_actions.commit(directory, message=body.message, files=body.files))
        return GitDone(message=done, status=await _wait(git_actions.git_status(directory)))

    @app.post('/api/sessions/{conversation_id}/git/push')
    async def push(conversation_id: str, body: PushRequest) -> GitDone:
        directory = workspace(conversation_id)
        done = await _wait(git_actions.push(directory, confirmed=body.confirmed))
        return GitDone(message=done, status=await _wait(git_actions.git_status(directory)))

    @app.post('/api/sessions/{conversation_id}/git/pr/draft')
    async def draft_pull_request(conversation_id: str) -> PullRequestDraft:
        return await _wait(git_actions.draft_pull_request(workspace(conversation_id), model=model(conversation_id)))

    @app.post('/api/sessions/{conversation_id}/git/pr')
    async def pull_request(conversation_id: str, body: PullRequestRequest) -> PullRequestOpened:
        opened = git_actions.open_pull_request(workspace(conversation_id), title=body.title, body=body.body)
        return PullRequestOpened(url=await _wait(opened))


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
    """The WebSocket: the default session's snapshot, `live`, and the orchestrator on connect, then what is published.

    The browser may send `watch` to choose the live sessions it gets the messages of; until then it gets
    the default session's.
    """

    @app.websocket('/ws')
    async def events(socket: WebSocket) -> None:
        await socket.accept()
        # A page open across the `/update` that renamed the item and turn tags sends no `protocol`.
        legacy = 'protocol' not in socket.query_params
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
                await socket.send_text(live)
                # Last: the session and the sidebar do not wait on the orchestrator's state.
                await socket.send_text(dump(OrchestratorMessage(message=await service.orchestrator.state())))
                async with anyio.create_task_group() as connection:

                    async def forward() -> None:
                        try:
                            while (text := await client.queue.get()) != RESYNC:
                                await socket.send_text(legacy_tags(text) if legacy else text)
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
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _http(exc) from exc


async def _wait(action: Awaitable[ResultT]) -> ResultT:
    try:
        return await action
    except (ValueError, LookupError, RuntimeError) as exc:
        raise _http(exc) from exc


def _http(error: ValueError | LookupError | RuntimeError) -> HTTPException:
    """The error's message; settings a model rejected answer 422 with what is wrong with each, by its path."""
    if isinstance(error, ValidationError):
        problems = {'.'.join(str(part) for part in problem['loc']): problem['msg'] for problem in error.errors()}
        return HTTPException(422, {'message': 'Some settings are not valid.', 'errors': problems})
    return HTTPException(_status(error), str(error))

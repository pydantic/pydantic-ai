"""A typed client for another CLAI's web UI: what its page does, from Python, for a CLAI that drives another one.

`WebClient` makes the page's calls: the state, a prompt or `/command`, an interrupt, a saved session's
transcript, and answers to questions and dialogs, plus the WebSocket stream of `ServerMessage`s.
`SessionView` follows that stream as the page does, keeping the transcript's items, whether a turn runs,
and what is asked, so a caller can wait for a turn to end. It follows any `SessionClient`: `WebClient`, or
`local_client.LocalClient` for a session of this CLAI.

The token goes in a header for API calls and in the WebSocket's query, with the server's own address as
the `Host` and `Origin`, as `app.Guard` requires. Nothing here prints or logs it.
"""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, Self
from urllib.parse import urlsplit

import anyio
import httpx
from pydantic import BaseModel, JsonValue, TypeAdapter, ValidationError
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed
from websockets.typing import Origin

from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins.web.app import HEADER, Answer, Answers, DialogAnswer, Entered, Input, Interrupted
from pydantic_clai2.builtin_plugins.web.protocol import (
    Dialog,
    DialogClosed,
    DialogOpened,
    Item,
    ItemLog,
    ItemMessage,
    Notice,
    PluginsChanged,
    QuestionAsked,
    QuestionClosed,
    Questions,
    ServerMessage,
    SessionInfo,
    Snapshot,
    TitleChanged,
    Transcript,
    TurnEnded,
    TurnStarted,
)

_MESSAGE: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)
_SNAPSHOT = TypeAdapter(Snapshot)
_TRANSCRIPT = TypeAdapter(Transcript)


class _Failure(BaseModel):
    """FastAPI's error body."""

    detail: JsonValue


def _error(response: httpx.Response) -> Exception:
    """The error the server's handler raised, as the app maps them to statuses: unknown, refused now, or invalid."""
    try:
        detail = str(_Failure.model_validate_json(response.content).detail)
    except ValidationError:
        detail = response.text
    message = f'CLAI answered {response.status_code}: {detail}'
    if response.status_code == 404:
        return LookupError(message)
    if response.status_code == 409:
        return RuntimeError(message)
    if response.status_code == 403:
        return PermissionError(message)
    return ValueError(message)


class WebClient:
    """Calls to one CLAI's web UI at `url` (`http://HOST:PORT/`, without the token), with `token`.

    Use it as an async context manager, which owns its HTTP connections. Failed calls raise what the
    server's handler raised: `LookupError` (404), `RuntimeError` (409), `ValueError` (400, 422), and
    `PermissionError` for a wrong token or address (403).
    """

    def __init__(self, *, url: str, token: str) -> None:
        parts = urlsplit(url)
        if parts.scheme != 'http' or not parts.netloc:
            raise ValueError(f'Give the web UI address as http://HOST:PORT/, not {url!r}.')
        self.url = f'http://{parts.netloc}/'
        self._netloc = parts.netloc
        self._token = token
        self._http = httpx.AsyncClient(base_url=f'http://{parts.netloc}', headers={HEADER: token}, trust_env=False)

    async def __aenter__(self) -> Self:
        await self._http.__aenter__()
        return self

    async def __aexit__(
        self, kind: type[BaseException] | None, error: BaseException | None, traceback: TracebackType | None
    ) -> None:
        await self._http.__aexit__(kind, error, traceback)

    async def _call(self, method: Literal['GET', 'POST'], path: str, body: BaseModel | None = None) -> bytes:
        content = None if body is None else body.model_dump_json()
        headers = {} if body is None else {'content-type': 'application/json'}
        response = await self._http.request(method, path, content=content, headers=headers, timeout=30)
        if response.is_error:
            raise _error(response)
        return response.content

    async def state(self) -> Snapshot:
        """What a newly connected page gets: the session, its items, whether a turn runs, and what is asked."""
        return _SNAPSHOT.validate_json(await self._call('GET', '/api/state'))

    async def send(self, text: str) -> None:
        """Enter a prompt or `/command`, as typed in the page; it queues behind a running turn.

        A bare command whose menu the page would show, such as `/set`, raises `ValueError`: it runs nothing.
        """
        entered = Entered.model_validate_json(await self._call('POST', '/api/input', Input(text=text)))
        if entered.menu is not None:
            raise ValueError(f'/{entered.menu} opens a menu in the page; send it with arguments instead.')

    async def interrupt(self) -> bool:
        """Cancel the running turn, as Esc does; whether something was running."""
        return Interrupted.model_validate_json(await self._call('POST', '/api/interrupt')).interrupted

    async def transcript(self, session_id: str) -> Transcript:
        """A saved session's items, without switching to it."""
        return _TRANSCRIPT.validate_json(await self._call('GET', f'/api/sessions/{session_id}'))

    async def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer an open question request: an answer for each question, or `None` to decline them all."""
        body = None if answers is None else [_answer(answer) for answer in answers]
        await self._call('POST', f'/api/questions/{request_id}', Answers(answers=body))

    async def answer_dialog(self, dialog_id: str, value: str | None) -> None:
        """Answer an open dialog: the chosen value, the typed text, or `yes`; `None` backs out."""
        await self._call('POST', f'/api/dialogs/{dialog_id}', DialogAnswer(value=value))

    @asynccontextmanager
    async def messages(self) -> AsyncGenerator[AsyncIterator[ServerMessage]]:
        """The WebSocket's messages, a `Snapshot` first; iteration ends when the connection closes."""
        address = f'ws://{self._netloc}/ws?token={self._token}'
        async with connect(address, origin=Origin(f'http://{self._netloc}'), max_size=None, proxy=None) as socket:

            async def received() -> AsyncIterator[ServerMessage]:
                while True:
                    try:
                        text = await socket.recv()
                    except ConnectionClosed:  # Closed cleanly or not, the stream ends.
                        return
                    yield _MESSAGE.validate_json(text)

            yield received()


def _answer(answer: AskUserAnswer) -> Answer:
    if answer.custom_answer is not None:
        return Answer(header=answer.header, custom=answer.custom_answer)
    return Answer(header=answer.header, selected=list(answer.selected))


class SessionClient(Protocol):
    """What `SessionView` drives a session through: the calls a page makes, and the messages it receives."""

    async def send(self, text: str) -> None:
        """Enter a prompt or `/command`; it queues behind a running turn."""
        ...

    async def interrupt(self) -> bool:
        """Cancel the running turn; whether something was running."""
        ...

    async def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer an open question request, or decline it with `None`."""
        ...

    async def answer_dialog(self, dialog_id: str, value: str | None) -> None:
        """Answer an open dialog; `None` backs out."""
        ...

    def messages(self) -> AbstractAsyncContextManager[AsyncIterator[ServerMessage]]:
        """The session's messages, a `Snapshot` first; iteration ends when the connection closes."""
        ...


WaitOutcome = Literal['idle', 'asking', 'timeout', 'closed']
"""What `SessionView.wait` stopped for: no turn runs or is queued, a question or dialog waits for an answer,
the time ran out, or the connection ended."""


@dataclass(eq=False)
class _Sent:
    """One input sent through a view and not started yet; each is its own, compared by identity."""

    command: bool
    """A `/command`, which ends without a turn."""


class SessionView:
    """One CLAI's session as its page follows it, kept current by `follow` from the client's messages.

    Prompts and commands sent through `send` count as queued until their turn starts or the command ends,
    so `wait` right after `send` waits for that turn, not for a session that has not started it yet. The
    server says a command ended only with `PluginsChanged`, which a plugin panel edit also sends, so a
    command sent while another page edits plugins may count as ended early.
    """

    def __init__(self, client: SessionClient) -> None:
        self.client = client
        self.session: SessionInfo | None = None
        self.running = False
        self.questions: dict[str, Questions] = {}
        self.dialogs: dict[str, Dialog] = {}
        self.last_turn: TurnEnded | None = None
        """How the last turn ended, and what it used."""
        self.notices: list[Notice] = []
        """`/command` results and other lines, oldest first."""
        self.closed = False
        self._log = ItemLog()
        self._queued: list[_Sent] = []
        """Sent and not yet started, oldest first."""
        self._changed = anyio.Event()
        self._ready = anyio.Event()

    @property
    def items(self) -> list[Item]:
        """The session's transcript, its running turn included."""
        return list(self._log.items.values())

    async def follow(self) -> None:
        """Apply every message until the connection ends; run it in a task while using the view."""
        try:
            async with self.client.messages() as messages:
                async for message in messages:
                    self.apply(message)
        finally:
            self.closed = True
            self._ready.set()
            self._changed.set()

    async def ready(self) -> None:
        """Wait until the first snapshot arrived, or the connection ended."""
        await self._ready.wait()

    def apply(self, message: ServerMessage) -> None:
        """Change the view as the page changes on `message`."""
        if isinstance(message, Snapshot):
            self.session, self.running = message.session, message.running
            self.questions = {questions.id: questions for questions in message.questions}
            self.dialogs = {dialog.id: dialog for dialog in message.dialogs}
            self._log.clear()
            self._log.items.update((item.id, item) for item in message.items)
            self._ready.set()
        elif isinstance(message, TurnStarted):
            self.running = True
            self._dequeue(command=False)
            if message.item is not None:
                self._log.items[message.item.id] = message.item
        elif isinstance(message, TurnEnded):
            self.running, self.last_turn = False, message
        elif isinstance(message, QuestionAsked):
            self.questions[message.questions.id] = message.questions
        elif isinstance(message, QuestionClosed):
            self.questions.pop(message.id, None)
        elif isinstance(message, DialogOpened):
            self.dialogs[message.dialog.id] = message.dialog
        elif isinstance(message, DialogClosed):
            self.dialogs.pop(message.id, None)
        elif isinstance(message, Notice):
            self.notices.append(message)
        elif isinstance(message, PluginsChanged):
            self._dequeue(command=True)  # The page hears this once every command ends.
        elif isinstance(message, TitleChanged):
            if self.session is not None:
                self.session.title = message.title
        elif isinstance(message, ItemMessage):
            self._log.apply(message)
        self._changed.set()
        self._changed = anyio.Event()

    def _dequeue(self, *, command: bool) -> None:
        if self._queued and self._queued[0].command is command:
            self._queued.pop(0)

    async def send(self, text: str) -> None:
        """Enter a prompt or `/command`, counted as queued until it starts."""
        sent = _Sent(command=text.strip().startswith('/'))
        self._queued.append(sent)
        try:
            await self.client.send(text)
        except BaseException:
            # Only this input's own marker: concurrent sends each queued theirs, and one may have started.
            if sent in self._queued:
                self._queued.remove(sent)
            raise

    async def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer an open question request, which then counts as closed: see `WebClient.answer`."""
        await self.client.answer(request_id, answers)
        self.questions.pop(request_id, None)

    async def answer_dialog(self, dialog_id: str, value: str | None) -> None:
        """Answer an open dialog, which then counts as closed: see `WebClient.answer_dialog`."""
        await self.client.answer_dialog(dialog_id, value)
        self.dialogs.pop(dialog_id, None)

    async def wait(self, *, timeout: float) -> WaitOutcome:
        """Wait up to `timeout` seconds until no turn runs or is queued, or something is asked."""
        with anyio.move_on_after(timeout):
            while True:
                if self.closed:
                    return 'closed'
                if self.questions or self.dialogs:
                    return 'asking'
                if not self.running and not self._queued:
                    return 'idle'
                await self._changed.wait()
        return 'timeout'

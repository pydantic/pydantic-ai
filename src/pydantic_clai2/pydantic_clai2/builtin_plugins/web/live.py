"""Fan messages out to the connected browsers, and answer `ask_user` questions from whichever side answers first.

Each browser watches some live sessions, by conversation ID: it gets their messages and the global
ones. One that never says which watches the default session, as every browser did before CLAI ran several.
"""

import asyncio
from collections.abc import Callable, Generator, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import anyio

from pydantic_ai_harness.ask_user import Answerer, AskUserAnswer, AskUserRequest, AskUserResponse, check_response
from pydantic_ai_harness.subagents import DelegationTasks
from pydantic_clai2.builtin_plugins.web.protocol import (
    QuestionAsked,
    QuestionClosed,
    Questions,
    ServerMessage,
    SessionMessage,
    Snapshot,
    dump,
    dump_session,
    questions_info,
)

QUEUE_LIMIT = 10_000
"""Messages a browser may fall behind by before it is told to reconnect, so a stalled tab cannot hold a whole session."""
RESYNC = ''
"""Queued in place of what a browser missed: it reconnects and starts from a fresh snapshot."""


@dataclass(eq=False, kw_only=True)
class Client:
    """One connected browser: its queue of JSON messages and the sessions it watches."""

    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue[str])
    watching: set[str] | None = None
    """Conversation IDs; `None` until it sends `watch`, meaning the default session."""
    unwatch: dict[str, Callable[[], None]] = field(default_factory=dict[str, Callable[[], None]])
    """What stops counting it as a watcher of each live session it watches, by `LiveSession.id`."""

    def watches(self, session: str, *, default: str | None) -> bool:
        return session == default if self.watching is None else session in self.watching


class Hub:
    """The connected browsers, all on the shell's event loop; publishing never waits for one."""

    def __init__(self, *, default: Callable[[], str | None] = lambda: None) -> None:
        """`default` names the conversation of the session a browser that sent no `watch` follows."""
        self._clients: set[Client] = set()
        self.default = default

    @property
    def clients(self) -> int:
        """How many browsers are connected."""
        return len(self._clients)

    def connected(self) -> list[Client]:
        return list(self._clients)

    def watchers(self, session: str) -> list[Client]:
        """The browsers watching the session holding conversation `session`."""
        default = self.default()
        return [client for client in self._clients if client.watches(session, default=default)]

    def publish(self, message: ServerMessage) -> None:
        """Queue a global message for every browser."""
        if self._clients:
            self.send(self.connected(), dump(message))

    def publish_session(self, message: SessionMessage | Snapshot, *, session: str) -> None:
        """Queue a message about the session holding conversation `session` for the browsers watching it."""
        if clients := self.watchers(session):
            self.send(
                clients, dump(message) if isinstance(message, Snapshot) else dump_session(message, session=session)
            )

    def rename(self, old: str, new: str) -> None:
        """A session switched from conversation `old` to `new`: the browsers watching it watch `new` now."""
        for client in self._clients:
            if client.watching is not None and old in client.watching:
                client.watching.discard(old)
                client.watching.add(new)

    def send(self, clients: Iterable[Client], text: str) -> None:
        """Queue `text` for `clients`; one too far behind is dropped and told to reconnect."""
        for client in clients:
            queue = client.queue
            if queue.qsize() < QUEUE_LIMIT - 1:
                queue.put_nowait(text)
                continue
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(RESYNC)
            self._clients.discard(client)

    @contextmanager
    def subscribe(self) -> Generator[Client]:
        """Receive what is published while the block runs."""
        client = Client()
        self._clients.add(client)
        try:
            yield client
        finally:
            self._clients.discard(client)


@dataclass(kw_only=True)
class _Open:
    questions: Questions
    request: AskUserRequest
    answered: asyncio.Future[AskUserResponse]
    session: str | None


class BrowserAnswerer:
    """An `Answerer` that shows each question in the browser and races the terminal picker, when there is one.

    Whichever side answers first wins and the other closes, so a question asked while the user sits at
    either the terminal or the browser can be answered there. Each question belongs to the session
    `session` names as it is asked, so only that session's watchers see it.
    """

    def __init__(
        self,
        *,
        publish: Callable[[QuestionAsked | QuestionClosed, str | None], None],
        terminal: Answerer | None,
        session: Callable[[], str | None] = lambda: None,
    ) -> None:
        """`publish` takes each message with the conversation ID of the session asking, from `session`."""
        self._publish = publish
        self._terminal = terminal
        self._session = session
        self._open: dict[str, _Open] = {}

    @property
    def questions(self) -> list[Questions]:
        """The questions waiting for an answer, oldest first."""
        return [entry.questions for entry in self._open.values()]

    def asked(self, session: str) -> list[Questions]:
        """The questions the session holding conversation `session` waits on, oldest first."""
        return [entry.questions for entry in self._open.values() if entry.session == session]

    def answer(self, request_id: str, response: AskUserResponse) -> None:
        """Answer an open question; `LookupError` when it was answered or withdrawn already."""
        entry = self._open.get(request_id)
        if entry is None or entry.answered.done():
            raise LookupError('That question is no longer open.')
        entry.answered.set_result(response)

    def reply(self, request_id: str, answers: Sequence[AskUserAnswer] | None, *, session: str | None = None) -> None:
        """Answer an open question, or decline it with `None`; `ValueError` when the answers do not fit it.

        With `session`, only a question of that session's: another is `LookupError`, as if it were closed.
        """
        request = self.request(request_id, session=session)
        response = AskUserResponse(cancelled=True) if answers is None else AskUserResponse(answers=tuple(answers))
        check_response(request, response)
        self.answer(request_id, response)

    def request(self, request_id: str, *, session: str | None = None) -> AskUserRequest:
        """The open request with this ID, of `session` when given; `LookupError` when it is not open."""
        entry = self._open.get(request_id)
        if entry is None or (session is not None and entry.session != session):
            raise LookupError('That question is no longer open.')
        return entry.request

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        child = DelegationTasks.child_id()
        questions = questions_info(request, asker=None if child is None else f'Task [{child[:8]}]')
        session = self._session()
        entry = _Open(
            questions=questions, request=request, answered=asyncio.get_running_loop().create_future(), session=session
        )
        self._open[request.id] = entry
        self._publish(QuestionAsked(questions=questions), session)
        try:
            if self._terminal is None:
                return await entry.answered
            return await self._race(request, entry.answered, self._terminal)
        finally:
            del self._open[request.id]
            self._publish(QuestionClosed(id=request.id), session)

    async def _race(
        self, request: AskUserRequest, browser: asyncio.Future[AskUserResponse], terminal: Answerer
    ) -> AskUserResponse:
        answers: list[AskUserResponse] = []
        async with anyio.create_task_group() as race:

            async def first(answer: asyncio.Future[AskUserResponse] | Answerer) -> None:
                answers.append(await answer if isinstance(answer, asyncio.Future) else await answer(request))
                race.cancel_scope.cancel()

            race.start_soon(first, browser)
            race.start_soon(first, terminal)
        return answers[0]

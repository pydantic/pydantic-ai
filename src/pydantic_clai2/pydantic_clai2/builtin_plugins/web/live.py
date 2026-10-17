"""Fan messages out to every connected browser, and answer `ask_user` questions from whichever side answers first."""

import asyncio
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import anyio

from pydantic_ai_harness.ask_user import Answerer, AskUserAnswer, AskUserRequest, AskUserResponse, check_response
from pydantic_ai_harness.subagents import DelegationTasks
from pydantic_clai2.builtin_plugins.web.protocol import (
    QuestionAsked,
    QuestionClosed,
    Questions,
    ServerMessage,
    dump,
    questions_info,
)

QUEUE_LIMIT = 10_000
"""Messages a browser may fall behind by before it is told to reconnect, so a stalled tab cannot hold a whole session."""
RESYNC = ''
"""Queued in place of what a browser missed: it reconnects and starts from a fresh snapshot."""


class Hub:
    """The connected browsers, each with its own queue of JSON messages, all on the shell's event loop."""

    def __init__(self) -> None:
        self._clients: set[asyncio.Queue[str]] = set()

    @property
    def clients(self) -> int:
        """How many browsers are connected."""
        return len(self._clients)

    def publish(self, message: ServerMessage) -> None:
        """Queue `message` for every browser; it never waits, so an agent run never blocks on a browser."""
        if not self._clients:
            return
        text = dump(message)
        for queue in list(self._clients):
            if queue.qsize() < QUEUE_LIMIT - 1:
                queue.put_nowait(text)
                continue
            while not queue.empty():
                queue.get_nowait()
            queue.put_nowait(RESYNC)
            self._clients.discard(queue)

    @contextmanager
    def subscribe(self) -> Generator[asyncio.Queue[str]]:
        """Receive every message published while the block runs."""
        queue = asyncio.Queue[str]()
        self._clients.add(queue)
        try:
            yield queue
        finally:
            self._clients.discard(queue)


@dataclass(kw_only=True)
class _Open:
    questions: Questions
    request: AskUserRequest
    answered: asyncio.Future[AskUserResponse]


class BrowserAnswerer:
    """An `Answerer` that shows each question in the browser and races the terminal picker, when there is one.

    Whichever side answers first wins and the other closes, so a question asked while the user sits at
    either the terminal or the browser can be answered there.
    """

    def __init__(self, *, publish: Callable[[QuestionAsked | QuestionClosed], None], terminal: Answerer | None) -> None:
        self._publish = publish
        self._terminal = terminal
        self._open: dict[str, _Open] = {}

    @property
    def questions(self) -> list[Questions]:
        """The questions waiting for an answer, oldest first."""
        return [entry.questions for entry in self._open.values()]

    def answer(self, request_id: str, response: AskUserResponse) -> None:
        """Answer an open question; `LookupError` when it was answered or withdrawn already."""
        entry = self._open.get(request_id)
        if entry is None or entry.answered.done():
            raise LookupError('That question is no longer open.')
        entry.answered.set_result(response)

    def reply(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        """Answer an open question, or decline it with `None`; `ValueError` when the answers do not fit it."""
        request = self.request(request_id)
        response = AskUserResponse(cancelled=True) if answers is None else AskUserResponse(answers=tuple(answers))
        check_response(request, response)
        self.answer(request_id, response)

    def request(self, request_id: str) -> AskUserRequest:
        """The open request with this ID, to check an answer against; `LookupError` when it is not open."""
        entry = self._open.get(request_id)
        if entry is None:
            raise LookupError('That question is no longer open.')
        return entry.request

    async def __call__(self, request: AskUserRequest, /) -> AskUserResponse:
        child = DelegationTasks.child_id()
        questions = questions_info(request, asker=None if child is None else f'Task [{child[:8]}]')
        entry = _Open(questions=questions, request=request, answered=asyncio.get_running_loop().create_future())
        self._open[request.id] = entry
        self._publish(QuestionAsked(questions=questions))
        try:
            if self._terminal is None:
                return await entry.answered
            return await self._race(request, entry.answered, self._terminal)
        finally:
            del self._open[request.id]
            self._publish(QuestionClosed(id=request.id))

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

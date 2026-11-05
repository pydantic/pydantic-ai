"""A `SessionClient` for a live session of this CLAI: what a page watching it does, without leaving the process.

The orchestrator follows the sessions it runs in workspaces with it, through `WebService` as a browser would,
so it counts as one of the session's watchers and the session does not close for being idle meanwhile.
Settings dialogs belong to the page and its user, never to one session, so this client neither shows nor
answers them.
"""

from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING

from pydantic import TypeAdapter

from pydantic_ai_harness.ask_user import AskUserAnswer
from pydantic_clai2.builtin_plugins.web.channel import SessionChannel
from pydantic_clai2.builtin_plugins.web.live import RESYNC, Client
from pydantic_clai2.builtin_plugins.web.protocol import DialogClosed, DialogOpened, LiveChanged, ServerMessage, Snapshot
from pydantic_clai2.plugins import LiveSession

if TYPE_CHECKING:
    from pydantic_clai2.builtin_plugins.web.service import WebService

_MESSAGE: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def live_session(service: 'WebService', live_id: str) -> LiveSession | None:
    """The live session `live_id` names in `service`'s CLAI; `None` once it closed."""
    try:
        return service.live.get(live_id)
    except LookupError:
        return None


class LocalClient:
    """The live session `live_id` of `service`'s CLAI, driven as its page drives it.

    It is named by its live ID, which stays the same when the session switches conversation with `/new`
    or `/resume`; `LookupError` once it closed.
    """

    def __init__(self, service: 'WebService', live_id: str) -> None:
        self.service = service
        self.live_id = live_id

    def _channel(self) -> SessionChannel:
        return self.service.channel(self.service.live.get(self.live_id))

    def _closed(self) -> bool:
        return live_session(self.service, self.live_id) is None

    async def send(self, text: str) -> None:
        """Enter a prompt or `/command`; a bare command that opens a menu in the page raises `ValueError`."""
        if (menu := self._channel().send(text)) is not None:
            raise ValueError(f'/{menu} opens a menu in the page; send it with arguments instead.')

    async def interrupt(self) -> bool:
        return self._channel().session.interrupt()

    async def answer(self, request_id: str, answers: Sequence[AskUserAnswer] | None) -> None:
        self._channel().answer(request_id, answers)

    async def answer_dialog(self, dialog_id: str, value: str | None) -> None:
        raise LookupError(f'No dialog {dialog_id!r}: dialogs belong to the page, not to a session.')

    @asynccontextmanager
    async def messages(self) -> AsyncGenerator[AsyncIterator[ServerMessage]]:
        """The session's messages, its `Snapshot` first; iteration ends once it closes or this falls too far behind."""
        service = self.service
        with service.hub.subscribe() as client:
            service.watch(client, [self._channel().id])
            try:
                yield self._received(client)
            finally:
                service.unwatch(client)

    async def _received(self, client: Client) -> AsyncIterator[ServerMessage]:
        while True:
            text = await client.queue.get()
            if text == RESYNC:
                return
            message = _MESSAGE.validate_json(text)
            if isinstance(message, DialogOpened | DialogClosed):
                continue
            # Asked by live ID, which `/new` and `/resume` keep: a `LiveChanged` queued before a switch names the
            # old conversation, which the client no longer watches.
            if isinstance(message, LiveChanged) and self._closed():
                return
            yield replace(message, dialogs=[]) if isinstance(message, Snapshot) else message

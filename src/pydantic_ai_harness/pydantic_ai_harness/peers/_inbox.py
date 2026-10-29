"""Per-peer inbox: persist before accepting, enqueue during a run, remove only after delivery acknowledgement."""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable
from pathlib import Path

from anyio import Lock
from anyio.to_thread import run_sync
from pydantic import TypeAdapter

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import UserPromptPart
from pydantic_ai_harness.peers._message import make_peer_message
from pydantic_ai_harness.peers._types import MAX_MESSAGE_BYTES, MAX_PENDING, PeerDelivery, PeerMessage

_MESSAGES = TypeAdapter(list[PeerMessage])


class PeerInbox:
    """Incoming peer messages, in memory or in an atomically replaced JSON file.

    Bind one inbox to one recipient. The application supplies a path scoped to that recipient;
    its directory is private to the user. An idle inbox never starts an agent run.
    """

    accepting: bool
    """Whether new messages are accepted. Changing this flag does not discard pending messages."""

    def __init__(
        self,
        *,
        path: Path | None = None,
        max_pending: int = MAX_PENDING,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
        accepting: bool = True,
    ) -> None:
        if max_pending < 1 or max_message_bytes < 1:
            raise UserError('Inbox limits must be positive.')
        self._path = path
        self._max_pending = max_pending
        self._max_message_bytes = max_message_bytes
        self.accepting = accepting
        self._messages: dict[str, PeerMessage] = {}
        self._loaded = False
        self._lock = Lock()
        self._enqueue: Callable[[UserPromptPart], str] | None = None
        self._inflight: dict[str, str] = {}

    @property
    def running(self) -> bool:
        """Whether this inbox is attached to a run that can receive messages now."""
        return self._enqueue is not None

    async def _load(self) -> None:
        if self._loaded:
            return
        if self._path is not None:
            try:
                saved = await run_sync(self._path.read_bytes)
            except FileNotFoundError:
                saved = b'[]'
            self._messages = {message.id: message for message in _MESSAGES.validate_json(saved)}
        self._loaded = True

    async def _save(self, messages: dict[str, PeerMessage]) -> None:
        if self._path is None:
            return
        data = _MESSAGES.dump_json(list(messages.values()))
        await run_sync(self._write, self._path, data)

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = path.with_name(f'.{path.name}.{secrets.token_hex(8)}.tmp')
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _deliver(self, message: PeerMessage) -> None:
        if self._enqueue is not None and message.id not in self._inflight:
            self._inflight[message.id] = self._enqueue(make_peer_message(message))

    async def receive(self, message: PeerMessage) -> PeerDelivery:
        """Accept once by message id; refuse oversized messages or a full or disabled inbox."""
        async with self._lock:
            await self._load()
            if message.id in self._messages:
                return PeerDelivery(
                    message_id=message.id,
                    status='delivered' if message.id in self._inflight else 'queued',
                    pending=len(self._messages),
                )
            reason = (
                'does not accept messages'
                if not self.accepting
                else 'too long'
                if len(message.text.encode('utf-8')) > self._max_message_bytes
                else 'inbox full'
                if len(self._messages) >= self._max_pending
                else ''
            )
            if reason:
                return PeerDelivery(message_id=message.id, status='refused', reason=reason, pending=len(self._messages))
            updated = {**self._messages, message.id: message}
            await self._save(updated)
            self._messages = updated
            self._deliver(message)
            return PeerDelivery(
                message_id=message.id,
                status='delivered' if message.id in self._inflight else 'queued',
                pending=len(self._messages),
            )

    async def pending(self) -> list[PeerMessage]:
        """Undelivered messages, oldest first."""
        async with self._lock:
            await self._load()
            return list(self._messages.values())

    async def _attach(self, enqueue: Callable[[UserPromptPart], str]) -> bool:
        """Attach one run, enqueue pending messages in order. A nested run leaves the outer run attached."""
        async with self._lock:
            await self._load()
            if self._enqueue is not None:
                return False
            self._enqueue = enqueue
            for message in self._messages.values():
                self._deliver(message)
            return True

    def _detach(self) -> None:
        """Forget an ended run's enqueue IDs; its unacknowledged messages stay pending."""
        self._enqueue = None
        self._inflight.clear()

    async def _acknowledge(self, enqueue_id: str) -> None:
        """Remove a message only after core confirms it entered the conversation history."""
        async with self._lock:
            message_id = next((key for key, value in self._inflight.items() if value == enqueue_id), None)
            if message_id is None:
                return
            updated = self._messages.copy()
            del updated[message_id]
            await self._save(updated)
            self._messages = updated
            del self._inflight[message_id]

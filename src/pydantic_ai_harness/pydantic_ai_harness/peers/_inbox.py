"""Per-peer inbox: persist before accepting, hand messages to the reading run, remove only after acknowledgement."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from anyio import Lock
from anyio.to_thread import run_sync
from pydantic import TypeAdapter

from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.peers._types import (
    MAX_MESSAGE_BYTES,
    MAX_PENDING,
    DeliveryStatus,
    PeerDelivery,
    PeerInbox,
    PeerMessage,
)

_MESSAGES = TypeAdapter(list[PeerMessage])


class LocalPeerInbox(PeerInbox):
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
        self._reader: str | None = None
        self._taken: set[str] = set()

    @property
    def running(self) -> bool:
        """Whether a run is reading this inbox, so it reads new messages at its next model request."""
        return self._reader is not None

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

    async def receive(self, message: PeerMessage) -> PeerDelivery:
        """Accept once by message id; refuse oversized messages or a full or disabled inbox."""
        async with self._lock:
            await self._load()
            if message.id in self._messages:
                return PeerDelivery(message_id=message.id, status=self._status, pending=len(self._messages))
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
            return PeerDelivery(message_id=message.id, status=self._status, pending=len(self._messages))

    @property
    def _status(self) -> DeliveryStatus:
        return 'delivered' if self.running else 'queued'

    async def pending(self) -> list[PeerMessage]:
        """Undelivered messages, oldest first."""
        async with self._lock:
            await self._load()
            return list(self._messages.values())

    async def take(self, run_id: str) -> list[PeerMessage]:
        """Claim the inbox for `run_id` if no run holds it; return its messages not yet taken, oldest first."""
        async with self._lock:
            await self._load()
            if self._reader is None:
                self._reader = run_id
            if self._reader != run_id:
                return []
            taken = [message for message in self._messages.values() if message.id not in self._taken]
            self._taken.update(message.id for message in taken)
            return taken

    async def acknowledge(self, message_id: str) -> None:
        """Remove a message once it entered the conversation history."""
        async with self._lock:
            if message_id not in self._messages:
                return
            updated = self._messages.copy()
            del updated[message_id]
            await self._save(updated)
            self._messages = updated
            self._taken.discard(message_id)

    async def release(self, run_id: str) -> None:
        """End `run_id`'s claim; unacknowledged messages stay pending for the next run."""
        async with self._lock:
            if self._reader == run_id:
                self._reader = None
                self._taken.clear()

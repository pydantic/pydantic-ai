"""In-process implementation of the peer transport for tests and agents in one application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import uuid4

from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.peers._inbox import PeerInbox
from pydantic_ai_harness.peers._types import PeerDelivery, PeerInfo, PeerMessage, PeerNetwork, PeerSender


@dataclass(kw_only=True)
class _Peer(PeerNetwork):
    members: dict[str, _Peer]
    info: PeerInfo
    inbox: PeerInbox

    async def peers(self) -> Sequence[PeerInfo]:
        return [
            replace(
                peer.info,
                status='running' if peer.inbox.running else 'idle',
                pending=len(await peer.inbox.pending()),
                accepts_messages=peer.inbox.accepting,
            )
            for peer in tuple(self.members.values())
            if peer is not self
        ]

    async def send(self, recipient: str, text: str, *, reply_to: str | None = None) -> PeerDelivery:
        peer = self.members.get(recipient)
        if peer is None:
            raise LookupError(f'No peer {recipient!r}.')
        message = PeerMessage(
            id=uuid4().hex,
            sender=PeerSender(id=self.info.id, handle=self.info.handle, workspace=self.info.workspace),
            recipient=recipient,
            text=text,
            sent_at=datetime.now(UTC),
            reply_to=reply_to,
        )
        return await peer.inbox.receive(message)


class InMemoryPeerNetwork:
    """Join several agents in one process. The network does not run turns or own their lifetime."""

    def __init__(self) -> None:
        self._members: dict[str, _Peer] = {}

    def join(self, *, handle: str, inbox: PeerInbox | None = None, workspace: str = '', title: str = '') -> PeerNetwork:
        """Join as `handle`; return this peer's network view.

        Pass the same `inbox` to `Peers(inbox=...)` when constructing the receiving agent.
        """
        if any(peer.info.handle == handle for peer in self._members.values()):
            raise UserError(f'Peer handle {handle!r} is already in use.')
        info = PeerInfo(id=uuid4().hex, handle=handle, workspace=workspace, title=title)
        member = _Peer(members=self._members, info=info, inbox=inbox if inbox is not None else PeerInbox())
        self._members[info.id] = member
        return member

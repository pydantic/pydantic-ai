"""In-process implementation of the peer transport for tests and agents in one application."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.peers._inbox import LocalPeerInbox
from pydantic_ai_harness.peers._types import PeerDelivery, PeerInfo, PeerMessage, PeerNetwork, PeerSender


@dataclass(kw_only=True)
class _Peer(PeerNetwork):
    members: dict[str, _Peer]
    info: PeerInfo
    inbox: LocalPeerInbox

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

    async def send(self, recipient: str, text: str, *, key: str, reply_to: str | None = None) -> PeerDelivery:
        peer = self.members.get(recipient)
        if peer is None:
            raise LookupError(f'No peer {recipient!r}.')
        message = PeerMessage(
            # The sender's id scopes the key, so two senders' keys never name the same message.
            id=uuid5(NAMESPACE_URL, f'{self.info.id}/{key}').hex,
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

    def join(
        self, *, handle: str, inbox: LocalPeerInbox | None = None, workspace: str = '', title: str = ''
    ) -> PeerNetwork:
        """Join as `handle`; return this peer's network view.

        Pass the same `inbox` to `Peers(inbox=...)` when constructing the receiving agent.
        With no inbox, this peer can send but refuses incoming messages.
        """
        if any(peer.info.handle == handle for peer in self._members.values()):
            raise UserError(f'Peer handle {handle!r} is already in use.')
        info = PeerInfo(id=uuid4().hex, handle=handle, workspace=workspace, title=title)
        if inbox is None:
            inbox = LocalPeerInbox(accepting=False)
        member = _Peer(members=self._members, info=info, inbox=inbox)
        self._members[info.id] = member
        return member

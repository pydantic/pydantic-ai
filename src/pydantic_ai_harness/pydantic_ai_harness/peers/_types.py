"""The data peers exchange and the network protocol a host implements to connect them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

PeerStatus = Literal['idle', 'running', 'waiting']
DeliveryStatus = Literal['delivered', 'queued', 'refused']
MAX_MESSAGE_BYTES = 16_384
MAX_PENDING = 50


@dataclass(frozen=True, kw_only=True)
class PeerInfo:
    """A peer visible to the agent. `id` addresses it; `handle` is its short display name."""

    id: str
    handle: str
    title: str = ''
    workspace: str = ''
    branch: str = ''
    status: PeerStatus = 'idle'
    waiting_for: str = ''
    status_since: datetime | None = None
    pending: int = 0
    accepts_messages: bool = True


@dataclass(frozen=True, kw_only=True)
class PeerSender:
    """Identity supplied by the network, not the agent's tool arguments."""

    id: str
    handle: str
    workspace: str = ''


@dataclass(frozen=True, kw_only=True)
class PeerMessage:
    """One message; `TypeAdapter(PeerMessage)` is its JSON wire format."""

    id: str
    sender: PeerSender
    recipient: str
    text: str
    sent_at: datetime
    reply_to: str | None = None


@dataclass(frozen=True, kw_only=True)
class PeerDelivery:
    """Result reported by the receiving inbox."""

    message_id: str
    status: DeliveryStatus
    reason: str = ''
    pending: int = 0


class PeerNetwork(Protocol):
    """One peer's view of its neighbours. The host implements transport and authenticates senders."""

    async def peers(self) -> Sequence[PeerInfo]:
        """Other reachable peers, never this one, most relevant first."""
        ...

    async def send(self, recipient: str, text: str, *, reply_to: str | None = None) -> PeerDelivery:
        """Build a message from this peer to `PeerInfo.id` and deliver it.

        Return the recipient's result. Raise `LookupError` if the recipient has disappeared.
        """
        ...

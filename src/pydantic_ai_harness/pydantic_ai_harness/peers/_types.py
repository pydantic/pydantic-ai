"""The data peers exchange and the network protocol a host implements to connect them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

PeerStatus = Literal['idle', 'running', 'waiting']
"""Host-reported activity: no run, an active run, or a run waiting for user input or approval."""

DeliveryStatus = Literal['delivered', 'queued', 'refused']
"""Accepted by an active run, queued for the next run, or rejected without delivery."""

MAX_MESSAGE_BYTES = 16_384
"""Default UTF-8 text limit, shared with transports that reject oversized messages before inbox delivery."""

MAX_PENDING = 50
"""Default inbox capacity, shared with transports that bound their incoming queues."""


@dataclass(frozen=True, kw_only=True)
class PeerInfo:
    """A peer visible to the agent. `id` addresses it; `handle` is its short display name."""

    id: str
    """Address accepted by `PeerNetwork.send`, unique among reachable peers."""
    handle: str
    """Short host-assigned name, also accepted by the agent's `send_message` tool."""
    title: str = ''
    """A human-readable description of the peer's current work."""
    workspace: str = ''
    """The directory the peer works in, if known."""
    branch: str = ''
    """Its Git branch, if known."""
    status: PeerStatus = 'idle'
    """The peer's current activity, independent of whether it accepts messages."""
    waiting_for: str = ''
    """With `status='waiting'`, what the peer is waiting for."""
    status_since: datetime | None = None
    """When the current status began, or `None` if unknown."""
    pending: int = 0
    """Messages awaiting delivery acknowledgement in this peer's inbox."""
    accepts_messages: bool = True
    """Whether the peer accepts messages, independently of its activity status."""


@dataclass(frozen=True, kw_only=True)
class PeerSender:
    """Identity supplied by the network, not the agent's tool arguments."""

    id: str
    """The sender's `PeerInfo.id`, assigned by the host."""
    handle: str
    """The sender's short display name at send time."""
    workspace: str = ''
    """The sender's working directory, if known."""


@dataclass(frozen=True, kw_only=True)
class PeerMessage:
    """One message; `TypeAdapter(PeerMessage)` is its JSON wire format."""

    id: str
    """Host-generated message identifier, used to deduplicate messages still held by an inbox."""
    sender: PeerSender
    recipient: str
    """Recipient's `PeerInfo.id`."""
    text: str
    """Untrusted peer-authored content, bounded by its UTF-8 byte length."""
    sent_at: datetime
    """The sending host's timestamp."""
    reply_to: str | None = None
    """Identifier of an earlier message this one answers, if any."""


@dataclass(frozen=True, kw_only=True)
class PeerDelivery:
    """Result reported by the receiving inbox."""

    message_id: str
    """The handled `PeerMessage.id`."""
    status: DeliveryStatus
    """Enqueued to an active run, held for the next run, or refused."""
    reason: str = ''
    """Explanation of a refusal or a note about queued delivery."""
    pending: int = 0
    """Messages awaiting acknowledgement in the recipient's inbox after this operation."""


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

"""The data peers exchange, and the network and inbox protocols a host implements to connect them."""

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
    """Host-generated identifier, derived from the send's key so a retried send keeps it."""
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

    async def send(self, recipient: str, text: str, *, key: str, reply_to: str | None = None) -> PeerDelivery:
        """Build a message from this peer to `PeerInfo.id` and deliver it.

        `key` identifies this send: a retried send with the same key must not deliver a second message.
        `Peers` passes `run_id:run_step:tool_call_id` of the tool call that sends it.
        Return the recipient's result. Raise `LookupError` if the recipient has disappeared.
        """
        ...


class PeerInbox(Protocol):
    """A recipient's held messages, as `Peers` reads them during a run. The host stores and admits them.

    One run reads an inbox at a time. `Peers` takes new messages before and after each model request,
    and acknowledges each one once it has entered the conversation history.
    """

    async def take(self, run_id: str) -> Sequence[PeerMessage]:
        """Messages held for the reading run that it has not taken yet, oldest first.

        The first call claims the inbox for `run_id`. While another run holds it, return nothing.
        """
        ...

    async def acknowledge(self, message_id: str) -> None:
        """Stop holding a message: it entered the reading run's conversation history."""
        ...

    async def release(self, run_id: str) -> None:
        """End `run_id`'s claim. Messages it took without acknowledging are held for the next run."""
        ...

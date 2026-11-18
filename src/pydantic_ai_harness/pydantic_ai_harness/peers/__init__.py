"""Discover and message peer agents without giving them control of one another's runs."""

from pydantic_ai_harness._warn import warn_class_renamed
from pydantic_ai_harness.peers._capability import Peers
from pydantic_ai_harness.peers._inbox import LocalPeerInbox
from pydantic_ai_harness.peers._message import PeerMessagePart, read_peer_message
from pydantic_ai_harness.peers._network import InMemoryPeerNetwork
from pydantic_ai_harness.peers._types import (
    MAX_MESSAGE_BYTES,
    MAX_PENDING,
    DeliveryStatus,
    PeerConnection,
    PeerDelivery,
    PeerInbox,
    PeerInfo,
    PeerMessage,
    PeerSender,
    PeerStatus,
)

__all__ = [
    'MAX_MESSAGE_BYTES',
    'MAX_PENDING',
    'DeliveryStatus',
    'InMemoryPeerNetwork',
    'LocalPeerInbox',
    'PeerConnection',
    'PeerDelivery',
    'PeerInbox',
    'PeerInfo',
    'PeerMessage',
    'PeerMessagePart',
    'PeerSender',
    'PeerStatus',
    'Peers',
    'read_peer_message',
]


def __getattr__(name: str) -> object:
    if name == 'PeerNetwork':
        warn_class_renamed(name, 'PeerConnection', __name__)
        return PeerConnection
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

"""A peer's words travel as labelled data on the user channel, never as the receiving user's prompt."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter, ValidationError

from pydantic_ai.framing import register_tag
from pydantic_ai.messages import DataCapabilityPart, ModelRequestPart, ModelSource, UserPromptPart
from pydantic_ai_harness._peer_message import PEER_MESSAGE_ORIGIN
from pydantic_ai_harness._provenance import tagged_text
from pydantic_ai_harness.peers._types import PeerMessage

_MESSAGE = TypeAdapter(PeerMessage)
# Registered so a frame that still escapes its body, such as `wrap`'s, neutralizes a copy of the peer message tag.
register_tag('peer_message')


@dataclass(kw_only=True)
class PeerMessagePart(
    DataCapabilityPart[str],
    namespace='peers',
    name='message',
    tag='peer_message',
    attributes=('sender', 'id'),
):
    """A message from a peer agent, as `Peers` delivers it.

    Data on the user channel in a `<peer_message>` tag naming the sender, with a note that the peer wrote it, not the
    user or the application. Find peer messages in history by this type, or with
    [`read_peer_message`][pydantic_ai_harness.peers.read_peer_message].
    """

    sender: str
    """The sender's handle, which the host supplied."""

    id: str
    """The message's id."""

    message: PeerMessage
    """The message as the host delivered it, with its sender's identity and timestamp. Never sent to the model."""


def make_peer_message(message: PeerMessage) -> PeerMessagePart:
    """A peer's message as data its sender wrote, with no authority."""
    return PeerMessagePart(
        content=message.text,
        source=ModelSource(f'peer `{message.sender.handle}`'),
        sender=message.sender.handle,
        id=message.id,
        message=message,
    )


def read_peer_message(part: ModelRequestPart) -> PeerMessage | None:
    """Read a message from its part, without relying on its model-facing text.

    A message an earlier version delivered as a `UserPromptPart` is read from its provenance metadata.
    """
    if isinstance(part, PeerMessagePart):
        return part.message
    if not isinstance(part, UserPromptPart) or (content := tagged_text(part.content, PEER_MESSAGE_ORIGIN)) is None:
        return None
    metadata = content.metadata
    assert isinstance(metadata, dict)
    try:
        return _MESSAGE.validate_python(metadata['message'])
    except (KeyError, ValidationError):
        return None

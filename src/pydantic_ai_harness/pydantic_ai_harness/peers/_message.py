"""A peer's words travel as labelled data on the user channel, never as the receiving user's prompt."""

from __future__ import annotations

import html

from pydantic import TypeAdapter, ValidationError

from pydantic_ai.messages import ModelRequestPart, TextContent, UserPromptPart
from pydantic_ai_harness._peer_message import PEER_MESSAGE_ORIGIN
from pydantic_ai_harness._provenance import neutralize_harness_tags, tagged_text
from pydantic_ai_harness.peers._types import PeerMessage

_MESSAGE = TypeAdapter(PeerMessage)


def make_peer_message(message: PeerMessage) -> UserPromptPart:
    """Frame untrusted peer text using core's tag escaper and retain its typed provenance in metadata."""
    # TODO: register this data frame with the public framing API once #10957 lands.
    sender = html.escape(message.sender.handle)
    message_id = html.escape(message.id)
    body = neutralize_harness_tags(message.text)
    text = f'<peer_message sender="{sender}" id="{message_id}">\n{body}\n</peer_message>'
    return UserPromptPart(
        [
            TextContent(
                text, metadata={'origin': PEER_MESSAGE_ORIGIN, 'message': _MESSAGE.dump_python(message, mode='json')}
            )
        ]
    )


def read_peer_message(part: ModelRequestPart) -> PeerMessage | None:
    """Read a message from its tagged part, without relying on its model-facing text."""
    if not isinstance(part, UserPromptPart) or (content := tagged_text(part.content, PEER_MESSAGE_ORIGIN)) is None:
        return None
    metadata = content.metadata
    assert isinstance(metadata, dict)
    try:
        return _MESSAGE.validate_python(metadata['message'])
    except (KeyError, ValidationError):
        return None

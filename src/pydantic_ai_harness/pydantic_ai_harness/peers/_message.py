"""A peer's words travel as labelled data on the user channel, never as the receiving user's prompt."""

from __future__ import annotations

from pydantic import TypeAdapter, ValidationError

from pydantic_ai._framing import wrap
from pydantic_ai.messages import ModelRequestPart, TextContent, UserPromptPart
from pydantic_ai_harness._provenance import tagged_text
from pydantic_ai_harness.peers._types import PeerMessage

_ORIGIN = 'pydantic-ai-harness.peers.message.v1'
_MESSAGE = TypeAdapter(PeerMessage)


def make_peer_message(message: PeerMessage) -> UserPromptPart:
    """Frame untrusted peer text using core's tag escaper and retain its typed provenance in metadata."""
    text = wrap('tool_result', message.text, tool_name='peer_message', sender=message.sender.handle, id=message.id)
    return UserPromptPart(
        [TextContent(text, metadata={'origin': _ORIGIN, 'message': _MESSAGE.dump_python(message, mode='json')})]
    )


def read_peer_message(part: ModelRequestPart) -> PeerMessage | None:
    """Read a message from its tagged part, without relying on its model-facing text."""
    if not isinstance(part, UserPromptPart) or (content := tagged_text(part.content, _ORIGIN)) is None:
        return None
    metadata = content.metadata
    if not isinstance(metadata, dict):
        return None
    try:
        return _MESSAGE.validate_python(metadata['message'])
    except (KeyError, ValidationError):
        return None

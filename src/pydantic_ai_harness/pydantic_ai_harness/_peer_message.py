"""Peer message provenance shared with guardrails without importing the peers capability."""

from __future__ import annotations

from typing import TypeGuard

from pydantic_ai.messages import ModelRequestPart, UserPromptPart
from pydantic_ai_harness._provenance import tagged_text

PEER_MESSAGE_ORIGIN = 'pydantic-ai-harness.peers.message.v1'


def is_peer_message(part: ModelRequestPart) -> TypeGuard[UserPromptPart]:
    """Whether a part is a peer message an earlier version delivered as a `UserPromptPart`, not the user's own words.

    `Peers` now delivers a `PeerMessagePart`, which is never taken for a user prompt.
    """
    return isinstance(part, UserPromptPart) and tagged_text(part.content, PEER_MESSAGE_ORIGIN) is not None

"""Shared helper for capabilities that append an ephemeral tail part behind a cache breakpoint."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import (
    CachePoint,
    ModelRequestPart,
    RetryFeedbackPart,
    TextContent,
    ToolReturnPart,
    UserPromptPart,
)


def has_cache_point_target(parts: Sequence[ModelRequestPart]) -> bool:
    """Whether these request parts already carry content a leading `CachePoint` can attach to.

    Providers attach a `CachePoint` that opens a user prompt to the content right before it, which is
    the end of the durable history when the prompt is an ephemeral tail part, whether that history ends
    in a user prompt, a tool return, or retry feedback. With nothing before it to attach to, providers
    that honor `CachePoint` reject it, so a tail part only leads with one when the request already
    contributes such content. `SystemPromptPart` maps to the system field, not to message content, so
    it does not count.

    `RetryFeedbackPart` counts: it is translated at request time into a `<validation_errors>`-fenced
    user prompt when it quotes the model's own output, and otherwise into a mid-conversation system
    prompt, which providers without an inline system channel receive as `<system>`-tagged user text.
    """
    for part in parts:
        if isinstance(part, ToolReturnPart | RetryFeedbackPart):
            return True
        if isinstance(part, UserPromptPart):
            content = part.content
            if isinstance(content, str):
                if content:
                    return True
            elif any(_is_user_content_item(item) for item in content):
                return True
    return False


def _is_user_content_item(item: object) -> bool:
    if isinstance(item, CachePoint):
        return False
    if isinstance(item, str):
        return bool(item)
    if isinstance(item, TextContent):
        return bool(item.content)
    return True

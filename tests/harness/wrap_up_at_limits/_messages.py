"""Helpers shared by the `WrapUpAtLimits` tests."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelRequest
from pydantic_ai_harness.wrap_up_at_limits import LimitName, WrapUpPart


def wrap_ups(messages: list[ModelMessage]) -> list[tuple[str, LimitName]]:
    """The content and limit of each wrap-up prompt in `messages`, checking each is request-scoped."""
    prompts = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, WrapUpPart)
    ]
    assert all(part.scope == 'request' for part in prompts)
    return [(str(part.content), part.limit) for part in prompts]

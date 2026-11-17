"""Helpers shared by the `WrapUpAtLimits` tests."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart

ORIGIN = 'pydantic-ai-harness.wrap_up_at_limits.v1'


def wrap_ups(messages: list[ModelMessage]) -> list[tuple[str, object]]:
    """The content and metadata of each system prompt in `messages`, checking each is request-scoped."""
    prompts = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart)
    ]
    assert all(part.scope == 'request' for part in prompts)
    return [(part.content, part.metadata) for part in prompts]

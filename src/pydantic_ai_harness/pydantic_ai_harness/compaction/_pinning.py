"""Pin contract: mark message content that every shipped compaction strategy must preserve.

A *pin* is a lightweight, harness-side convention: content marked with model-invisible
`TextContent.metadata` that compaction treats as load-bearing -- it is never summarized away or
dropped, and if a strategy would have discarded it, the strategy re-injects it verbatim.
Planning's plan does not need pinning because it is re-injected ephemerally every request in
`wrap_model_request`; the motivating consumers here are durable task state and a scratchpad the
model wants guaranteed to ride along.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    SystemPromptPart,
    TextContent,
    UserPromptPart,
)

_PIN_METADATA = 'pydantic-ai-harness.compaction.pin.v1'
"""Model-invisible metadata identifying a pinned user-content item."""


def pin(content: str) -> UserPromptPart:
    """Mark *content* so every shipped compaction strategy preserves it.

    The returned `UserPromptPart` can be placed in a `ModelRequest` in the run's message
    history (e.g. by a capability or by the user); compaction keeps it verbatim.
    """
    return UserPromptPart(content=[TextContent(content, metadata=_PIN_METADATA)])


def is_pinned(part: object) -> bool:
    """Return True if *part* carries the pin metadata marker."""
    return (
        isinstance(part, UserPromptPart)
        and not isinstance(part.content, str)
        and any(isinstance(item, TextContent) and item.metadata == _PIN_METADATA for item in part.content)
    )


def collect_pinned(messages: Sequence[ModelMessage]) -> list[ModelRequestPart]:
    """Return every pinned part found across *messages*, in order."""
    out: list[ModelRequestPart] = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            out.extend(part for part in msg.parts if is_pinned(part))
    return out


def _leading_context_len(messages: Sequence[ModelMessage]) -> int:
    """Count the leading run of system-only requests (system prompts + a summary message)."""
    count = 0
    for msg in messages:
        if isinstance(msg, ModelRequest) and msg.parts and all(isinstance(p, SystemPromptPart) for p in msg.parts):
            count += 1
        else:
            break
    return count


def reinject_pinned(original: Sequence[ModelMessage], compacted: list[ModelMessage]) -> list[ModelMessage]:
    """Re-inject any pinned parts from *original* that *compacted* dropped.

    Pinned parts already present in *compacted* are left where they are; missing ones are
    gathered into a single `ModelRequest` placed right after any leading system/summary
    messages, so they sit near the top of the surviving history. A no-op when *original* has
    no pins or all pins survived, so it is always safe to call.
    """
    pinned = collect_pinned(original)
    if not pinned:
        return compacted
    surviving = collect_pinned(compacted)
    missing: list[ModelRequestPart] = []
    for part in pinned:
        for index, survivor in enumerate(surviving):
            if part == survivor:
                del surviving[index]
                break
        else:
            missing.append(part)
    if not missing:
        return compacted
    index = _leading_context_len(compacted)
    pin_message = ModelRequest(parts=list(missing))
    return [*compacted[:index], pin_message, *compacted[index:]]

"""Built-in history processor functions for common message history repair tasks.

These functions can be passed directly to `Agent(history_processors=[...])` or
used with `capabilities.HistoryProcessor(processor=...)`.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai import messages as _messages

__all__ = ('repair_orphaned_tool_parts',)

logger = logging.getLogger(__name__)


def repair_orphaned_tool_parts(
    messages: list[_messages.ModelMessage],
) -> list[_messages.ModelMessage]:
    """Remove orphaned tool call/return parts from message history.

    Multi-turn agent conversations can accumulate structurally invalid history
    when tool calls and their corresponding results become mismatched. Common
    causes include streaming timeouts, deferred tool result drops, and history
    trimming by other processors.

    Providers like Anthropic strictly enforce that every `ToolCallPart` has a
    matching `ToolReturnPart` (or `RetryPromptPart`) and vice versa; orphaned
    entries cause 400 errors.

    This processor performs a two-pass repair:

    1. **Orphaned returns/retries**: `ToolReturnPart` or `RetryPromptPart` whose
       `tool_call_id` does not match any preceding `ToolCallPart` are removed.
    2. **Orphaned calls**: `ToolCallPart` whose `tool_call_id` does not match
       any following `ToolReturnPart` or `RetryPromptPart` are removed.

    Empty messages (all parts removed) are dropped entirely.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.history_processors import repair_orphaned_tool_parts

        agent = Agent('openai:gpt-5.2', history_processors=[repair_orphaned_tool_parts])
        ```
    """
    call_ids = _collect_tool_call_ids(messages)
    return_ids = _collect_tool_return_ids(messages)

    repaired: list[_messages.ModelMessage] = []
    for message in messages:
        if isinstance(message, _messages.ModelRequest):
            repaired_msg = _repair_request(message, call_ids)
        elif isinstance(message, _messages.ModelResponse):  # pragma: no branch
            repaired_msg = _repair_response(message, return_ids)
        if repaired_msg is not None:
            repaired.append(repaired_msg)

    return repaired


def _collect_tool_call_ids(messages: list[_messages.ModelMessage]) -> set[str]:
    """Collect all tool_call_ids from ToolCallPart in ModelResponse messages."""
    ids: set[str] = set()
    for message in messages:
        if isinstance(message, _messages.ModelResponse):
            for part in message.parts:
                if isinstance(part, _messages.ToolCallPart) and part.tool_call_id:
                    ids.add(part.tool_call_id)
    return ids


def _collect_tool_return_ids(messages: list[_messages.ModelMessage]) -> set[str]:
    """Collect all tool_call_ids from ToolReturnPart/RetryPromptPart in ModelRequest messages."""
    ids: set[str] = set()
    for message in messages:
        if isinstance(message, _messages.ModelRequest):
            for part in message.parts:
                if isinstance(part, (_messages.ToolReturnPart, _messages.RetryPromptPart)) and part.tool_call_id:
                    ids.add(part.tool_call_id)
    return ids


def _is_orphaned_request_part(part: _messages.ModelRequestPart, call_ids: set[str]) -> bool:
    """Check if a request part is orphaned (no matching tool call)."""
    if isinstance(part, _messages.ToolReturnPart):
        return part.tool_call_id not in call_ids
    if isinstance(part, _messages.RetryPromptPart):
        return part.tool_name is not None and part.tool_call_id not in call_ids
    return False


def _repair_request(message: _messages.ModelRequest, call_ids: set[str]) -> _messages.ModelMessage | None:
    """Remove orphaned ToolReturnPart/RetryPromptPart from a ModelRequest."""
    kept: list[_messages.ModelRequestPart] = []
    for part in message.parts:
        if _is_orphaned_request_part(part, call_ids):
            logger.debug(
                'Removing orphaned %s with tool_call_id=%r (no matching ToolCallPart)',
                type(part).__name__,
                getattr(part, 'tool_call_id', None),
            )
            continue
        kept.append(part)
    return _rebuild_or_drop(message, message.parts, kept)


def _repair_response(message: _messages.ModelResponse, return_ids: set[str]) -> _messages.ModelMessage | None:
    """Remove orphaned ToolCallPart from a ModelResponse."""
    kept: list[_messages.ModelResponsePart] = []
    for part in message.parts:
        if isinstance(part, _messages.ToolCallPart) and part.tool_call_id not in return_ids:
            logger.debug(
                'Removing orphaned ToolCallPart with tool_call_id=%r (no matching return)',
                part.tool_call_id,
            )
            continue
        kept.append(part)
    return _rebuild_or_drop(message, message.parts, kept)


def _rebuild_or_drop(
    message: _messages.ModelMessage,
    original_parts: Sequence[object],
    kept_parts: list[object],
) -> _messages.ModelMessage | None:
    """Return the message with filtered parts, or None if all parts were removed."""
    if not kept_parts:
        return None
    if len(kept_parts) != len(original_parts):
        return replace(message, parts=kept_parts)  # type: ignore[arg-type]
    return message

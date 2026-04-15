"""Built-in history processor functions for common message history repair tasks.

These functions can be passed directly to `Agent(history_processors=[...])` or
used with `capabilities.HistoryProcessor(processor=...)`.
"""

from __future__ import annotations

import logging
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

        agent = Agent('openai:gpt-4o-mini', history_processors=[repair_orphaned_tool_parts])
        ```
    """
    call_ids: set[str] = set()
    for message in messages:
        if isinstance(message, _messages.ModelResponse):
            for part in message.parts:
                if isinstance(part, _messages.ToolCallPart) and part.tool_call_id:
                    call_ids.add(part.tool_call_id)

    return_ids: set[str] = set()
    for message in messages:
        if isinstance(message, _messages.ModelRequest):
            for part in message.parts:
                if isinstance(part, (_messages.ToolReturnPart, _messages.RetryPromptPart)) and part.tool_call_id:
                    return_ids.add(part.tool_call_id)

    repaired: list[_messages.ModelMessage] = []
    for message in messages:
        if isinstance(message, _messages.ModelRequest):
            kept_parts: list[_messages.ModelRequestPart] = []
            for part in message.parts:
                if isinstance(part, _messages.ToolReturnPart):
                    if part.tool_call_id not in call_ids:
                        logger.debug(
                            'Removing orphaned ToolReturnPart with tool_call_id=%r (no matching ToolCallPart)',
                            part.tool_call_id,
                        )
                        continue
                elif isinstance(part, _messages.RetryPromptPart):
                    if part.tool_name is not None and part.tool_call_id not in call_ids:
                        logger.debug(
                            'Removing orphaned RetryPromptPart with tool_call_id=%r (no matching ToolCallPart)',
                            part.tool_call_id,
                        )
                        continue
                kept_parts.append(part)

            if kept_parts:
                if len(kept_parts) != len(message.parts):
                    repaired.append(replace(message, parts=kept_parts))
                else:
                    repaired.append(message)

        elif isinstance(message, _messages.ModelResponse):  # pragma: no branch
            kept_response_parts: list[_messages.ModelResponsePart] = []
            for part in message.parts:
                if isinstance(part, _messages.ToolCallPart):
                    if part.tool_call_id not in return_ids:
                        logger.debug(
                            'Removing orphaned ToolCallPart with tool_call_id=%r (no matching return)',
                            part.tool_call_id,
                        )
                        continue
                kept_response_parts.append(part)

            if kept_response_parts:
                if len(kept_response_parts) != len(message.parts):
                    repaired.append(replace(message, parts=kept_response_parts))
                else:
                    repaired.append(message)

    return repaired

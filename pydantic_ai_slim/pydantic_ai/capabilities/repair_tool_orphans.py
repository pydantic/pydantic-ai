"""Capability that repairs broken tool-call / tool-return pairing in message history.

Multi-turn conversations with tools can accumulate structural problems:
dangling tool calls without matching returns, or orphaned returns referencing
calls that don't exist. Providers like Anthropic strictly enforce pairing and
will reject malformed history with a 400 error.

This capability hooks into ``before_model_request`` to transparently repair
these issues before each model request.

Usage::

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import RepairToolOrphans

    agent = Agent('anthropic:claude-sonnet', capabilities=[RepairToolOrphans()])

See https://github.com/pydantic/pydantic-ai/issues/4728 for background.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_ai import messages as _messages
from pydantic_ai.tools import AgentDepsT

from .abstract import AbstractCapability

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.tools import RunContext

__all__ = ['RepairToolOrphans']

_DANGLING_TOOL_CALL_MESSAGE = 'Tool call was not completed.'


@dataclass
class RepairToolOrphans(AbstractCapability[AgentDepsT]):
    """Repair broken tool-call / tool-return pairing in message history.

    Fixes two classes of structural problems:

    1. **Dangling tool calls** -- a ``ToolCallPart`` or ``BuiltinToolCallPart``
       in a ``ModelResponse`` that has no matching return anywhere later in the
       history.

       - For regular ``ToolCallPart``: a synthetic ``ToolReturnPart`` is injected
         into the next ``ModelRequest``.
       - For ``BuiltinToolCallPart``: a synthetic ``BuiltinToolReturnPart`` is
         injected into the *same* ``ModelResponse`` (required by Anthropic).

    2. **Orphaned tool returns** -- a ``ToolReturnPart`` or ``RetryPromptPart``
       whose ``tool_call_id`` does not match any tool call in the preceding
       ``ModelResponse``.  These parts are removed.  If that empties the
       ``ModelRequest``, a placeholder ``UserPromptPart`` is inserted.

    Args:
        warn: If ``True`` (the default), emit a warning when repairs are made
              so the underlying issue can be investigated.
    """

    warn: bool = True

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        request_context.messages = _repair_messages(request_context.messages, warn=self.warn)
        return request_context


def _repair_messages(
    messages: list[_messages.ModelMessage],
    *,
    warn: bool,
) -> list[_messages.ModelMessage]:
    """Core repair logic operating on a message list."""
    if not messages:
        return messages

    repaired = False
    result: list[_messages.ModelMessage] = []

    for i, message in enumerate(messages):
        if isinstance(message, _messages.ModelResponse):
            fixed = _repair_response(message)
            if fixed is not message:
                repaired = True
            result.append(fixed)
        elif isinstance(message, _messages.ModelRequest):
            fixed = _repair_request(message, result)
            if fixed is not message:
                repaired = True
            result.append(fixed)
        else:
            result.append(message)

    # Handle trailing ModelResponse with dangling regular tool calls
    if result and isinstance(result[-1], _messages.ModelResponse):
        trailing = result[-1]
        dangling_calls = [p for p in trailing.parts if isinstance(p, _messages.ToolCallPart)]
        if dangling_calls:
            repaired = True
            result.append(
                _messages.ModelRequest(
                    parts=[
                        _messages.ToolReturnPart(
                            tool_name=part.tool_name,
                            content=_DANGLING_TOOL_CALL_MESSAGE,
                            tool_call_id=part.tool_call_id,
                        )
                        for part in dangling_calls
                    ]
                )
            )

    if repaired and warn:
        warnings.warn(
            'RepairToolOrphans: repaired broken tool-call/return pairing in message history. '
            'This usually means a previous agent run was interrupted before tool results '
            'were recorded. Consider investigating the root cause.',
            UserWarning,
            stacklevel=4,
        )

    return result


def _repair_response(response: _messages.ModelResponse) -> _messages.ModelResponse:
    """Inject synthetic BuiltinToolReturnPart for unanswered BuiltinToolCallParts.

    Builtin tool returns must live in the same ModelResponse as the call
    (required by Anthropic).
    """
    builtin_call_ids: set[str] = set()
    existing_return_ids: set[str] = set()

    for part in response.parts:
        if isinstance(part, _messages.BuiltinToolCallPart):
            builtin_call_ids.add(part.tool_call_id)
        elif isinstance(part, _messages.BuiltinToolReturnPart):
            existing_return_ids.add(part.tool_call_id)

    unanswered = builtin_call_ids - existing_return_ids
    if not unanswered:
        return response

    new_parts: list[_messages.ModelResponsePart] = list(response.parts)
    for part in response.parts:
        if isinstance(part, _messages.BuiltinToolCallPart) and part.tool_call_id in unanswered:
            new_parts.append(
                _messages.BuiltinToolReturnPart(
                    tool_name=part.tool_name,
                    content=_DANGLING_TOOL_CALL_MESSAGE,
                    tool_call_id=part.tool_call_id,
                )
            )

    return _messages.ModelResponse(
        parts=new_parts,
        usage=response.usage,
        model_name=response.model_name,
        timestamp=response.timestamp,
        provider_name=response.provider_name,
        provider_url=response.provider_url,
        provider_details=response.provider_details,
        provider_response_id=response.provider_response_id,
        finish_reason=response.finish_reason,
        run_id=response.run_id,
    )


def _repair_request(
    request: _messages.ModelRequest,
    preceding: list[_messages.ModelMessage],
) -> _messages.ModelRequest:
    """Fix orphaned tool returns and inject synthetic returns for dangling calls.

    1. Find the preceding ModelResponse's regular tool call IDs.
    2. Remove ToolReturnPart / RetryPromptPart whose tool_call_id is not in
       the preceding response.
    3. Inject synthetic ToolReturnPart for any ToolCallPart in the preceding
       response that has no matching return in this request.
    4. If the request ends up empty, insert a placeholder UserPromptPart.
    """
    # Find preceding ModelResponse's tool call IDs
    prev_regular_calls: dict[str, _messages.ToolCallPart] = {}
    for prev in reversed(preceding):
        if isinstance(prev, _messages.ModelResponse):
            for part in prev.parts:
                if isinstance(part, _messages.ToolCallPart):
                    prev_regular_calls[part.tool_call_id] = part
            break

    prev_call_ids = set(prev_regular_calls.keys())

    # Step 1: Filter out orphaned tool returns / retries
    filtered_parts: list[_messages.ModelRequestPart] = []
    answered_ids: set[str] = set()
    changed = False
    for part in request.parts:
        if isinstance(part, (_messages.ToolReturnPart, _messages.RetryPromptPart)):
            if part.tool_call_id in prev_call_ids:
                filtered_parts.append(part)
                answered_ids.add(part.tool_call_id)
            else:
                changed = True  # Dropping orphaned part
        else:
            filtered_parts.append(part)

    # Step 2: Inject synthetic returns for unanswered regular tool calls
    unanswered = prev_call_ids - answered_ids
    synthetic_parts: list[_messages.ModelRequestPart] = []
    if unanswered:
        changed = True
        # Preserve the order of tool calls from the response
        for prev in reversed(preceding):
            if isinstance(prev, _messages.ModelResponse):
                for part in prev.parts:
                    if isinstance(part, _messages.ToolCallPart) and part.tool_call_id in unanswered:
                        synthetic_parts.append(
                            _messages.ToolReturnPart(
                                tool_name=part.tool_name,
                                content=_DANGLING_TOOL_CALL_MESSAGE,
                                tool_call_id=part.tool_call_id,
                            )
                        )
                break

    if not changed:
        return request

    all_parts = synthetic_parts + filtered_parts

    # Step 3: If empty, add placeholder
    if not all_parts:
        all_parts = [_messages.UserPromptPart(content='...')]

    return _messages.ModelRequest(
        parts=all_parts,
        timestamp=request.timestamp,
        instructions=request.instructions,
        run_id=request.run_id,
        metadata=request.metadata,
    )

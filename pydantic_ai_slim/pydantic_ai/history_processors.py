"""Built-in history processors for sanitizing and repairing message history.

These are ready-to-use functions that can be passed directly to
`Agent(history_processors=[...])`.
"""

from __future__ import annotations

from pydantic_ai.messages import (
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

__all__ = ['repair_tool_orphans']

_DANGLING_TOOL_CALL_MESSAGE = 'Tool call was not completed.'


def repair_tool_orphans(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Repair broken tool-call / tool-return pairing in message history.

    This history processor fixes two classes of structural problems that can
    accumulate in multi-turn conversations with tool use:

    1. **Dangling tool calls** -- a `ToolCallPart` or `BuiltinToolCallPart` in a
       `ModelResponse` that has no matching `ToolReturnPart`, `BuiltinToolReturnPart`,
       or `RetryPromptPart` anywhere later in the history.

       - For regular `ToolCallPart`: a synthetic `ToolReturnPart` with an error
         message is inserted into the next `ModelRequest`.
       - For `BuiltinToolCallPart`: a synthetic `BuiltinToolReturnPart` is inserted
         into the *same* `ModelResponse` (required by some providers like Anthropic).

    2. **Orphaned tool returns** -- a `ToolReturnPart` or `RetryPromptPart` in a
       `ModelRequest` whose `tool_call_id` does not match any tool call in the
       preceding `ModelResponse`.  These parts are removed.  If removing them
       empties the `ModelRequest`, a placeholder `UserPromptPart` is inserted to
       maintain valid message alternation.

    Usage::

        from pydantic_ai import Agent
        from pydantic_ai.history_processors import repair_tool_orphans

        agent = Agent('anthropic:claude-sonnet', history_processors=[repair_tool_orphans])
    """
    if not messages:
        return messages

    result: list[ModelMessage] = []

    for i, message in enumerate(messages):
        if isinstance(message, ModelResponse):
            result.append(_repair_model_response(message, messages, i))
        elif isinstance(message, ModelRequest):
            result.append(_repair_model_request(message, result))
        else:
            result.append(message)

    # Handle trailing ModelResponse with dangling regular tool calls
    if result and isinstance(result[-1], ModelResponse):
        trailing = result[-1]
        dangling_calls = [p for p in trailing.parts if isinstance(p, ToolCallPart)]
        if dangling_calls:
            result.append(
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name=part.tool_name,
                            content=_DANGLING_TOOL_CALL_MESSAGE,
                            tool_call_id=part.tool_call_id,
                        )
                        for part in dangling_calls
                    ]
                )
            )

    return result


def _repair_model_response(
    response: ModelResponse,
    messages: list[ModelMessage],
    index: int,
) -> ModelResponse:
    """Fix dangling tool calls in a ModelResponse.

    - For BuiltinToolCallPart without a matching BuiltinToolReturnPart in the same
      response, inject a synthetic BuiltinToolReturnPart into this response.
    - For regular ToolCallPart without a matching return in the next ModelRequest,
      we don't modify the response itself -- those are handled in _repair_model_request.
    """
    # Check for unanswered builtin tool calls
    builtin_call_ids: set[str] = set()
    existing_builtin_return_ids: set[str] = set()

    for part in response.parts:
        if isinstance(part, BuiltinToolCallPart):
            builtin_call_ids.add(part.tool_call_id)
        elif isinstance(part, BuiltinToolReturnPart):
            existing_builtin_return_ids.add(part.tool_call_id)

    unanswered_builtin = builtin_call_ids - existing_builtin_return_ids
    if not unanswered_builtin:
        return response

    # Build new response with synthetic builtin returns appended
    new_parts = list(response.parts)
    for part in response.parts:
        if isinstance(part, BuiltinToolCallPart) and part.tool_call_id in unanswered_builtin:
            new_parts.append(
                BuiltinToolReturnPart(
                    tool_name=part.tool_name,
                    content=_DANGLING_TOOL_CALL_MESSAGE,
                    tool_call_id=part.tool_call_id,
                )
            )

    return ModelResponse(
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


def _repair_model_request(
    request: ModelRequest,
    preceding: list[ModelMessage],
) -> ModelRequest:
    """Fix orphaned tool returns and inject synthetic returns for dangling calls.

    1. Find the preceding ModelResponse in the already-processed result list.
    2. Remove any ToolReturnPart / RetryPromptPart whose tool_call_id is not in
       the preceding response's tool calls.
    3. Inject synthetic ToolReturnPart for any regular ToolCallPart in the
       preceding response that has no matching return in this request.
    4. If the request ends up empty, insert a placeholder UserPromptPart.
    """
    # Find preceding ModelResponse's tool call IDs
    prev_regular_calls: dict[str, ToolCallPart] = {}
    for prev in reversed(preceding):
        if isinstance(prev, ModelResponse):
            for part in prev.parts:
                if isinstance(part, ToolCallPart):
                    prev_regular_calls[part.tool_call_id] = part
            break

    prev_call_ids = set(prev_regular_calls.keys())

    # Step 1: Filter out orphaned tool returns / retries
    filtered_parts = []
    answered_ids: set[str] = set()
    for part in request.parts:
        if isinstance(part, (ToolReturnPart, RetryPromptPart)):
            if part.tool_call_id in prev_call_ids:
                filtered_parts.append(part)
                answered_ids.add(part.tool_call_id)
            # else: orphaned -- drop it
        else:
            filtered_parts.append(part)

    # Step 2: Inject synthetic returns for unanswered regular tool calls
    unanswered = prev_call_ids - answered_ids
    synthetic_parts = []
    if unanswered:
        # Preserve the order of tool calls from the response
        for prev in reversed(preceding):
            if isinstance(prev, ModelResponse):
                for part in prev.parts:
                    if isinstance(part, ToolCallPart) and part.tool_call_id in unanswered:
                        synthetic_parts.append(
                            ToolReturnPart(
                                tool_name=part.tool_name,
                                content=_DANGLING_TOOL_CALL_MESSAGE,
                                tool_call_id=part.tool_call_id,
                            )
                        )
                break

    all_parts = synthetic_parts + filtered_parts

    # Step 3: If empty, add placeholder
    if not all_parts:
        all_parts = [UserPromptPart(content='...')]

    return ModelRequest(
        parts=all_parts,
        timestamp=request.timestamp,
        instructions=request.instructions,
        run_id=request.run_id,
        metadata=request.metadata,
    )

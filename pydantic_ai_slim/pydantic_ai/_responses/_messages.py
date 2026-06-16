"""Convert OpenAI Responses API `input` items into Pydantic AI message history."""

from __future__ import annotations

from collections.abc import Sequence

from ..messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from .request_types import FunctionCall, InputMessage, ResponsesInputItem, ResponsesRequest


def _input_items(request: ResponsesRequest) -> Sequence[ResponsesInputItem]:
    input = request.input
    return [InputMessage(role='user', content=input)] if isinstance(input, str) else input


def load_messages(request: ResponsesRequest) -> list[ModelMessage]:
    """Build Pydantic AI message history from a Responses request's `input`.

    A bare string becomes a single user message. `system`/`developer` role messages are kept as
    [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s (the Responses API expects client
    instructions to be honored), and `function_call`/`function_call_output` items are paired back
    into tool call and return parts.
    """
    messages: list[ModelMessage] = []
    tool_names: dict[str, str] = {}

    def add_request_part(part: ModelRequestPart) -> None:
        last = messages[-1] if messages else None
        if isinstance(last, ModelRequest):
            last.parts = [*last.parts, part]
        else:
            messages.append(ModelRequest(parts=[part]))

    def add_response_part(part: ModelResponsePart) -> None:
        last = messages[-1] if messages else None
        if isinstance(last, ModelResponse):
            last.parts = [*last.parts, part]
        else:
            messages.append(ModelResponse(parts=[part]))

    for item in _input_items(request):
        if isinstance(item, InputMessage):
            content = item.content if isinstance(item.content, str) else '\n'.join(c.text for c in item.content)
            if item.role == 'assistant':
                add_response_part(TextPart(content=content))
            elif item.role == 'user':
                add_request_part(UserPromptPart(content=content))
            else:
                add_request_part(SystemPromptPart(content=content))
        elif isinstance(item, FunctionCall):
            tool_names[item.call_id] = item.name
            add_response_part(ToolCallPart(tool_name=item.name, args=item.arguments, tool_call_id=item.call_id))
        else:
            add_request_part(
                ToolReturnPart(
                    tool_name=tool_names.get(item.call_id, ''), content=item.output, tool_call_id=item.call_id
                )
            )

    return messages

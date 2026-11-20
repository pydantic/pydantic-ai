"""Message histories and model functions shared by the step store tests."""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo


def lookup_twice(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """A `FunctionModel` function: call `lookup` as `call-1`, again as `call-2` once `call-1` is answered, then finish."""
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('lookup', {'q': 'x'}, tool_call_id='call-1')])
    if len(messages) == 3:
        return ModelResponse(parts=[ToolCallPart('lookup', {'q': 'y'}, tool_call_id='call-2')])
    return ModelResponse(parts=[TextPart('done')])


def growing_histories(turns: int) -> list[list[ModelMessage]]:
    """The history after each turn; each is a prefix of the next, as across a conversation's runs."""
    histories: list[list[ModelMessage]] = []
    history: list[ModelMessage] = []
    for turn in range(turns):
        history = [
            *history,
            ModelRequest(parts=[UserPromptPart(content=f'question {turn}')]),
            ModelResponse(parts=[TextPart(content=f'answer {turn}')]),
        ]
        histories.append(history)
    return histories


def history_with_unordered_keys() -> list[ModelMessage]:
    """A tool call, its return and provider details whose dict keys aren't in sorted order."""
    return [
        ModelRequest(parts=[UserPromptPart(content='find shoes')]),
        ModelResponse(
            parts=[ToolCallPart('search', {'query': 'shoes', 'filters': {'zip': '10115', 'brand': 'x'}, 'limit': 5})],
            provider_details={'service_tier': 'default', 'finish_reason': 'tool_calls'},
        ),
        ModelRequest(parts=[ToolReturnPart('search', {'total': 1, 'items': [{'title': 'shoe', 'id': 7}]})]),
    ]

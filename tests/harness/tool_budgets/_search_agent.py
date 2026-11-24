"""A model that keeps searching past its `ToolBudgets` budget, shared by the durable-engine tests."""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo

searches: list[str] = []
"""One entry per execution of the `search` tool body, so a test can tell a replayed result from a new call."""


def searching_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    done = sum(isinstance(message, ModelResponse) for message in messages)
    if done == 3:
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('search', {'query': f'q{done}'}, tool_call_id=f'call-{done}')])


def search(query: str) -> str:
    searches.append(query)
    return f'results for {query}'


def refusal(reason: str) -> dict[str, str]:
    """The `metadata` of the tool result `ToolBudgets` answers a refused call with."""
    return {'origin': 'pydantic-ai-harness.tool_budgets.refusal.v1', 'reason': reason}


def refusals(messages: list[ModelMessage]) -> list[object]:
    return [
        part.metadata
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.metadata is not None
    ]

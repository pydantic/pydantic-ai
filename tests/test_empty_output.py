from __future__ import annotations as _annotations

import pytest

from pydantic_ai import Agent, exceptions
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


@pytest.mark.anyio
async def test_agent_allows_none_output_empty_response():
    """Test that Agent(output_type=str | None) succeeds on empty response."""

    async def empty_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    model = FunctionModel(function=empty_model)
    agent = Agent(model, output_type=str | None)

    result = await agent.run('hello')
    assert result.output is None


@pytest.mark.anyio
async def test_agent_allows_none_output_after_tool():
    """Test that Agent(output_type=str | None) succeeds after tool call with no final text."""
    call_count = 0

    async def tool_then_empty_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='noop', args={}, tool_call_id='123')])
        return ModelResponse(parts=[])

    model = FunctionModel(function=tool_then_empty_model)
    agent = Agent(model, output_type=str | None)

    @agent.tool_plain
    def noop() -> str:
        return 'done'

    result = await agent.run('hello')
    assert result.output is None
    assert call_count == 2


@pytest.mark.anyio
async def test_agent_still_fails_if_none_not_allowed():
    """Test that Agent(output_type=str) still fails on empty response."""

    async def empty_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[])

    model = FunctionModel(function=empty_model)
    agent = Agent(model, output_type=str)

    # It should raise an error after retries.
    # Current codebase behavior is to raise UnexpectedModelBehavior or IncompleteToolCall
    # after max retries are exceeded.
    with pytest.raises(exceptions.UnexpectedModelBehavior, match='Exceeded maximum retries'):
        await agent.run('hello')

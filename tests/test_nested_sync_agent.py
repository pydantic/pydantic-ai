from __future__ import annotations

import pytest

from pydantic_ai import Agent, UserError, _utils
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


def test_run_sync_from_sync_tool_is_rejected() -> None:
    inner_agent = Agent('test')

    def call_tool(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('delegate', '{}')])

    outer_agent = Agent(FunctionModel(call_tool))

    @outer_agent.tool_plain
    def delegate() -> str:
        return inner_agent.run_sync('hello').output

    with pytest.raises(
        UserError,
        match=r'`Agent\.run_sync\(\)` and `Agent\.run_stream_sync\(\)` cannot be used inside a synchronous tool',
    ):
        outer_agent.run_sync('delegate')

    # The guard is scoped to the callback: sync runs from regular application code still work.
    assert inner_agent.run_sync('hello').output == 'success (no tool calls)'


async def test_run_stream_sync_from_sync_output_function_is_rejected() -> None:
    inner_agent = Agent('test')

    def call_output(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"prompt": "hello"}')])

    def delegate(prompt: str) -> str:
        return inner_agent.run_stream_sync(prompt).get_output()

    outer_agent = Agent(FunctionModel(call_output), output_type=delegate)

    with pytest.raises(UserError, match=r'cannot be used inside a synchronous tool'):
        await outer_agent.run('delegate')


async def test_run_sync_from_sync_tool_is_rejected_when_threads_disabled() -> None:
    """Under `disable_threads()` (emscripten, Temporal) sync callbacks run inline; the same rule applies."""
    inner_agent = Agent('test')

    def call_tool(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('delegate', '{}')])

    outer_agent = Agent(FunctionModel(call_tool))

    @outer_agent.tool_plain
    def delegate() -> str:
        return inner_agent.run_sync('hello').output

    with _utils.disable_threads():
        with pytest.raises(UserError, match=r'cannot be used inside a synchronous tool'):
            await outer_agent.run('delegate')


async def test_async_tool_can_delegate_with_await() -> None:
    """The documented delegation pattern: an `async def` tool awaiting the inner run is allowed."""
    inner_agent = Agent('test')

    def call_tool(messages: list[ModelMessage], _: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('delegate', '{}')])
        return ModelResponse(parts=[TextPart('done')])

    outer_agent = Agent(FunctionModel(call_tool))

    @outer_agent.tool_plain
    async def delegate() -> str:
        result = await inner_agent.run('hello')
        return result.output

    result = await outer_agent.run('delegate')
    assert result.output == 'done'

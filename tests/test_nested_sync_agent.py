from __future__ import annotations

import pytest

from pydantic_ai import Agent, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


async def test_run_sync_from_sync_output_function_is_rejected() -> None:
    inner_agent = Agent('test')

    def call_output(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"prompt": "hello"}')])

    def delegate(prompt: str) -> str:
        return inner_agent.run_sync(prompt).output

    outer_agent = Agent(FunctionModel(call_output), output_type=delegate)

    with pytest.raises(
        UserError,
        match=(
            r'`Agent\.run_sync\(\)` cannot be called from a synchronous callback run by Pydantic AI\. '
            r'Make the callback `async def` and use `await agent\.run\(\.\.\.\)` instead\.'
        ),
    ):
        await outer_agent.run('delegate')


async def test_run_stream_sync_from_sync_tool_is_rejected() -> None:
    inner_agent = Agent('test')

    def call_tool(_: list[ModelMessage], __: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('delegate', '{}')])

    outer_agent = Agent(FunctionModel(call_tool))

    @outer_agent.tool_plain
    def delegate() -> str:
        return inner_agent.run_stream_sync('hello').get_output()

    with pytest.raises(UserError, match=r'`Agent\.run_stream_sync\(\)` cannot be called'):
        await outer_agent.run('delegate')

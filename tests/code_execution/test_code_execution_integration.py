"""End-to-end integration tests: Agent + CodeExecutionToolset + FunctionModel.

Validates the full pipeline including how the agent loop interacts with
code execution tool routing, tool execution, and result handling.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

try:
    from pydantic_ai.environments import monty as _monty  # pyright: ignore[reportUnusedImport] # noqa: F401
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic-monty is not installed', allow_module_level=True)
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset
from pydantic_ai.toolsets.function import FunctionToolset

from ..conftest import IsDatetime, IsStr

pytestmark = pytest.mark.anyio


def _make_toolset() -> FunctionToolset[None]:
    """Build a simple FunctionToolset with weather + math tools."""

    def get_weather(city: str) -> dict[str, Any]:
        """Get weather for a city."""
        return {'city': city, 'temp': 20, 'conditions': 'sunny'}

    def add(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(get_weather, takes_ctx=False)
    toolset.add_function(add, takes_ctx=False)
    return toolset


async def test_agent_single_tool_call():
    """Agent calls run_code with a single tool call and returns the result as text."""

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            # First turn: call run_code
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='run_code',
                        args={'code': 'await get_weather(city="Paris")'},
                    )
                ]
            )
        # After tool result: return text
        return ModelResponse(parts=[TextPart('The weather in Paris is sunny at 20 degrees.')])

    agent = Agent(
        FunctionModel(model_function),
        toolsets=[CodeExecutionToolset(toolset=_make_toolset())],
    )
    result = await agent.run('What is the weather in Paris?')
    assert result.output == 'The weather in Paris is sunny at 20 degrees.'


async def test_agent_multiple_tool_calls():
    """Agent writes code that calls multiple tools and processes results."""

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            code = """\
w = await get_weather(city="London")
total = await add(x=w["temp"], y=5)
{"adjusted_temp": total, "city": w["city"]}"""
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code})])
        return ModelResponse(parts=[TextPart('London is 25 degrees adjusted.')])

    agent = Agent(
        FunctionModel(model_function),
        toolsets=[CodeExecutionToolset(toolset=_make_toolset())],
    )
    result = await agent.run('Adjusted temp for London?')
    assert result.output == 'London is 25 degrees adjusted.'


async def test_agent_parallel_fire_then_await():
    """Agent writes code using fire-then-await for parallel execution."""

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            code = """\
f1 = get_weather(city="Paris")
f2 = get_weather(city="Tokyo")
r1 = await f1
r2 = await f2
[r1["city"], r2["city"]]"""
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': code})])
        return ModelResponse(parts=[TextPart('Got weather for both cities.')])

    agent = Agent(
        FunctionModel(model_function),
        toolsets=[CodeExecutionToolset(toolset=_make_toolset())],
    )
    result = await agent.run('Weather in Paris and Tokyo?')
    assert result.output == 'Got weather for both cities.'


async def test_agent_code_error_triggers_retry():
    """Syntax/runtime errors in code trigger ModelRetry, and the agent can recover."""
    call_count = 0

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt: bad code
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': '1 / 0'})])
        if call_count == 2:
            # Second attempt after retry: good code
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'await add(x=1, y=2)'})])
        # Final: return text
        return ModelResponse(parts=[TextPart('The answer is 3.')])

    agent = Agent(
        FunctionModel(model_function),
        toolsets=[CodeExecutionToolset(toolset=_make_toolset())],
    )
    result = await agent.run('Add 1 and 2')
    assert result.output == 'The answer is 3.'
    assert call_count == 3


async def test_concurrent_agent_runs_on_shared_toolset():
    """Two concurrent agent.run() calls sharing a CodeExecutionToolset produce correct independent results."""

    def add(x: int, y: int) -> int:
        return x + y

    def mul(x: int, y: int) -> int:
        return x * y

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)
    toolset.add_function(mul, takes_ctx=False)

    shared_toolset = CodeExecutionToolset(toolset=toolset)

    def add_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'await add(x=1, y=2)'})])
        return ModelResponse(parts=[TextPart('3')])

    def mul_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': 'await mul(x=3, y=4)'})])
        return ModelResponse(parts=[TextPart('12')])

    agent_add = Agent(FunctionModel(add_model), toolsets=[shared_toolset])
    agent_mul = Agent(FunctionModel(mul_model), toolsets=[shared_toolset])

    r1, r2 = await asyncio.gather(
        agent_add.run('add 1 and 2'),
        agent_mul.run('multiply 3 and 4'),
    )
    assert r1.output == '3'
    assert r2.output == '12'
    assert r1.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='add 1 and 2', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='run_code',
                        args={'code': 'await add(x=1, y=2)'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=54, output_tokens=7),
                model_name='function:add_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_code',
                        content=3,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='3')],
                usage=RequestUsage(input_tokens=55, output_tokens=8),
                model_name='function:add_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert r2.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='multiply 3 and 4', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='run_code',
                        args={'code': 'await mul(x=3, y=4)'},
                        tool_call_id=IsStr(),
                    )
                ],
                usage=RequestUsage(input_tokens=54, output_tokens=7),
                model_name='function:mul_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_code',
                        content=12,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='12')],
                usage=RequestUsage(input_tokens=55, output_tokens=8),
                model_name='function:mul_model:',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_agent_no_toolset_pure_code_execution():
    """Agent with CodeExecutionToolset() and no wrapped toolset executes pure Python."""

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if not any(isinstance(m, ModelResponse) for m in messages):
            return ModelResponse(parts=[ToolCallPart(tool_name='run_code', args={'code': '2 ** 10'})])
        return ModelResponse(parts=[TextPart('1024')])

    agent = Agent(
        FunctionModel(model_function),
        toolsets=[CodeExecutionToolset()],
    )
    result = await agent.run('What is 2^10?')
    assert result.output == '1024'

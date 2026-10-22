"""Known-name sandbox calls use execution candidates, not the displayed catalog."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.code_mode import CodeMode

READINESS_WAIT_TIMEOUT = 10


@pytest.mark.parametrize('dynamic_catalog', [False, True])
@pytest.mark.parametrize('eager,speculate', [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize('name', ['lookup', 'api.lookup'])
async def test_hidden_tool_runs_without_appearing_in_catalog(
    dynamic_catalog: bool, eager: bool, speculate: bool, name: str
):
    calls: list[int] = []
    started = asyncio.Event()
    tasks_before = asyncio.all_tasks()

    async def lookup(ctx: RunContext[None], key: int) -> int:
        assert ctx.discovered_tool_names == set()
        calls.append(key)
        started.set()
        return key * 2

    safe_name = name.replace('.', '_')

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tools = {tool.name: tool for tool in info.function_tools}
        assert name not in tools
        assert safe_name not in (tools['run_code'].description or '')
        assert safe_name not in (info.instructions or '')
        for msg in messages:
            for part in msg.parts:
                if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                    assert part.outcome == 'success', part.content
                    return ModelResponse(parts=[TextPart(str(part.content))])
        # A following complete statement closes the assignment for eager execution.
        return ModelResponse(
            parts=[ToolCallPart('run_code', {'code': f'result = await {safe_name}(key=3)\npass\nresult'})]
        )

    async def stream_model(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        response = model(messages, info)
        for part in response.parts:
            if isinstance(part, TextPart):
                yield part.content
            else:
                assert isinstance(part, ToolCallPart)
                args = json.dumps(part.args)
                yield {0: DeltaToolCall(name=part.tool_name, tool_call_id=part.tool_call_id)}
                chunks = [args[i : i + 4] for i in range(0, len(args), 4)]
                for chunk in chunks[:-1]:
                    yield {0: DeltaToolCall(json_args=chunk)}
                # The run owns early execution; prove it starts before the model finishes.
                await asyncio.wait_for(started.wait(), timeout=READINESS_WAIT_TIMEOUT)
                yield {0: DeltaToolCall(json_args=chunks[-1])}

    agent = Agent(
        FunctionModel(model, stream_function=stream_model),
        deps_type=type(None),
        tools=[Tool(lookup, name=name, defer_loading=True)],
        capabilities=[
            ToolSearch(),
            CodeMode(
                tools='all',
                dynamic_catalog=dynamic_catalog,
                eager=eager,
                speculate=[name] if speculate else None,
            ),
        ],
    )
    result = await agent.run('Run saved code')
    assert '6' in result.output
    assert calls == [3]
    assert asyncio.all_tasks() - tasks_before == set()
    assert result.usage.tool_calls == 2
    assert not any(isinstance(part, ToolAvailabilityDeltaPart) for msg in result.all_messages() for part in msg.parts)


@pytest.mark.parametrize('dynamic_catalog', [False, True])
@pytest.mark.parametrize('hidden_first', [False, True])
async def test_disclosed_tool_wins_sandbox_name_collision(dynamic_catalog: bool, hidden_first: bool):
    calls: list[str] = []
    catalogs: list[str] = []

    async def hidden() -> str:
        assert False, 'The hidden colliding tool must not execute'

    async def visible() -> str:
        calls.append('visible')
        return 'visible result'

    def reveal() -> ToolReturn:
        return ToolReturn('Ready', tools=['get_weather'])

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        returns = [part for msg in messages for part in msg.parts if isinstance(part, ToolReturnPart)]
        if not returns:
            return ModelResponse(parts=[ToolCallPart('reveal', {})])
        run_code = next(tool for tool in info.function_tools if tool.name == 'run_code')
        catalogs.append((info.instructions or '') if dynamic_catalog else (run_code.description or ''))
        if returns[-1].tool_name == 'reveal':
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': 'await get_weather()'})])
        assert returns[-1].outcome == 'success', returns[-1].content
        return ModelResponse(parts=[TextPart(str(returns[-1].content))])

    tools = [
        Tool(hidden, name='get-weather', description='Hidden weather implementation.', defer_loading=True),
        Tool(visible, name='get_weather', description='Visible weather implementation.', defer_loading=True),
    ]
    if not hidden_first:
        tools.reverse()
    agent = Agent(
        FunctionModel(model),
        tools=[*tools, Tool(reveal)],
        capabilities=[ToolSearch(), CodeMode(tools=['get-weather', 'get_weather'], dynamic_catalog=dynamic_catalog)],
    )
    with pytest.warns(UserWarning, match='collides'):
        result = await agent.run('Reveal weather and run saved code')
    assert calls == ['visible']
    assert 'visible result' in result.output
    assert all('Visible weather implementation.' in catalog for catalog in catalogs)
    assert all('Hidden weather implementation.' not in catalog for catalog in catalogs)

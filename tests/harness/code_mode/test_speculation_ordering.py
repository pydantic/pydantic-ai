"""Streamed speculation must not overtake earlier model tool calls."""

import json
from collections.abc import AsyncIterator

import pytest

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai_harness.code_mode import CodeMode, SpeculationStats


@pytest.mark.parametrize('native_write', [False, True])
async def test_read_waits_for_preceding_sequential_write(native_write: bool) -> None:
    value = 'before'
    reads: list[str] = []

    def write() -> str:
        nonlocal value
        value = 'after'
        return value

    def read() -> str:
        reads.append(value)
        return value

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) > 1:
            yield 'done'
            return
        code = 'await read()'
        if native_write:
            yield {0: DeltaToolCall(name='write', json_args='{}')}
        else:
            code = 'write()\nawait read()'
        yield {1: DeltaToolCall(name='run_code', json_args=json.dumps({'code': code}))}

    agent = Agent(
        FunctionModel(stream_function=stream),
        tools=[Tool(write, sequential=True), Tool(read)],
        capabilities=[CodeMode(tools=['read'] if native_write else 'all', speculate=['read'])],
    )
    result = await agent.run('go')
    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    assert returns == ['after']
    assert reads == ['after']


_PUBLISH = 'async def publish(x=0):\n    return await write()\n0'


@pytest.mark.parametrize(
    'snippets',
    [
        pytest.param(['await (write)()\nawait read()'], id='parenthesized'),
        pytest.param([_PUBLISH, 'await publish()\nawait read()'], id='earlier-helper'),
        pytest.param(
            [_PUBLISH, 'import asyncio\nawait asyncio.gather(*map(publish, [1]))\nawait read()'], id='mapped-helper'
        ),
    ],
)
async def test_read_launched_before_an_unspelled_write_is_not_adopted(snippets: list[str]) -> None:
    """A launch that predates a call to a non-eligible tool is never claimed after it.

    The text-level barrier cannot see these writes: the parenthesized call does not spell
    `write(`, and the helpers come from an earlier snippet, so `read` launches before the write
    runs. The snippet must call it again rather than adopt the stale value.
    """
    value = 'before'

    def write() -> str:
        nonlocal value
        value = 'after'
        return value

    def read() -> str:
        return value

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        step = sum(isinstance(message, ModelRequest) for message in messages) - 1
        if step == len(snippets):
            yield 'done'
            return
        yield {0: DeltaToolCall(name='run_code', json_args=json.dumps({'code': snippets[step]}))}

    capability = CodeMode(speculate=['read'])
    agent = Agent(FunctionModel(stream_function=stream), tools=[write, read], capabilities=[capability])
    result = await agent.run('go')

    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    assert returns[-1] == 'after'
    assert capability.speculation_stats.launched == 1
    assert capability.speculation_stats.adopted == 0


async def test_retry_does_not_adopt_a_launch_that_predates_a_native_write() -> None:
    """A failed attempt's launch survives for the retry, but not past a native write in between."""
    value = 'before'

    def write() -> str:
        nonlocal value
        value = 'after'
        return value

    def read() -> str:
        return value

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        step = sum(isinstance(message, ModelRequest) for message in messages) - 1
        if step == 0:
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': 'await read()\n1 +'})])
        if step == 1:
            # The retry follows a sequential native write in the same response, so no step boundary
            # evicts the failed attempt's launch before the retry dispatches.
            return ModelResponse(parts=[ToolCallPart('write', {}), ToolCallPart('run_code', {'code': 'await read()'})])
        return ModelResponse(parts=[TextPart('done')])

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        response = model(messages, info)
        for index, part in enumerate(response.parts):
            if isinstance(part, ToolCallPart):
                yield {index: DeltaToolCall(name=part.tool_name, json_args=part.args_as_json_str())}
            else:
                yield 'done'

    capability = CodeMode(tools=['read'], speculate=['read'])
    agent = Agent(
        FunctionModel(model, stream_function=stream),
        tools=[Tool(write, sequential=True), Tool(read)],
        capabilities=[capability],
    )
    result = await agent.run('go')

    returns = [
        part.content
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code' and part.outcome == 'success'
    ]
    assert returns == ['after']
    # The retry launched `read` afresh after the write and adopted that; the stale launch was evicted.
    assert capability.speculation_stats == SpeculationStats(launched=2, adopted=1, evicted=1)

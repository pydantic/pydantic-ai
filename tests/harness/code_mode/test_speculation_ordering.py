"""Streamed speculation must not overtake earlier model tool calls."""

import json
from collections.abc import AsyncIterator

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_harness.code_mode import CodeMode


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.anyio
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

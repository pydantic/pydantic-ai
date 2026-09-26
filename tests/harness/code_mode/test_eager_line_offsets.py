"""Regression tests for multiple statements closing in one streamed update."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from pydantic_ai_harness.code_mode import CodeMode

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('warmup', [False, True])
@pytest.mark.parametrize(
    'prefix, tail, expected',
    [
        ('a = 1\nb = 2\nc = 3\nd = 4\n', 'a + b + c + d', 10),
        ('a = 1\nb = 2\nfor i in range(2):\n    a += i\ndone = True\n', 'a + b', 4),
        ('a = 1\nb = 2\ntext = """first\nsecond"""\ndone = True\n', 'text', 'first\nsecond'),
    ],
    ids=['assignments', 'compound-statement', 'multiline-string'],
)
async def test_multiple_closed_statements_preserve_line_offsets(
    prefix: str, tail: str, expected: int | str, *, warmup: bool
) -> None:
    chunks = [prefix, tail]
    if warmup:
        chunks.insert(0, 'saved = 7\nready = True\n')
    code = ''.join(chunks)
    requests = 0

    async def stream_code(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        nonlocal requests
        requests += 1
        if requests > 1:
            yield 'done'
            return
        yield {0: DeltaToolCall(name='run_code', json_args='{"code":"')}
        for chunk in chunks:
            yield {0: DeltaToolCall(json_args=json.dumps(chunk)[1:-1])}
        yield {0: DeltaToolCall(json_args='"}')}

    agent = Agent(FunctionModel(stream_function=stream_code), capabilities=[CodeMode(eager=True)])
    result = await agent.run('Run the code.')

    returns = [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code'
    ]
    assert returns == [expected], code
    assert requests == 2

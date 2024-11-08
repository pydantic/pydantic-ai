from __future__ import annotations as _annotations

import json
from collections.abc import Iterable

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, AgentError
from pydantic_ai.messages import Message, ToolReturn, UserPrompt
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


async def test_streamed_text_response():
    m = TestModel()

    agent = Agent(m, deps=None)

    @agent.retriever_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result = await agent.run_stream('Hello')

    assert not result.is_structured()
    assert not result.is_complete
    response = await result.get_response()
    assert response == snapshot('{"ret_a":"a-apple"}')
    assert result.is_complete


async def test_streamed_structured_response():
    m = TestModel()

    agent = Agent(m, deps=None, result_type=tuple[str, str])

    result = await agent.run_stream('')

    assert result.is_structured()
    assert not result.is_complete
    response = await result.get_response()
    assert response == snapshot(('a', 'a'))
    assert result.is_complete


async def test_streamed_text_stream():
    m = TestModel(custom_result_text='The cat sat on the mat.')

    agent = Agent(m, deps=None)

    result = await agent.run_stream('Hello')
    assert not result.is_structured()
    # typehint to test (via static typing) that the stream type is correctly inferred
    chunks: list[str] = [c async for c in result.stream()]
    # one chunk due to group_by_temporal
    assert chunks == snapshot(['The cat sat on the mat.'])
    assert result.is_complete

    result = await agent.run_stream('Hello')
    assert [c async for c in result.stream(debounce_by=None)] == snapshot(
        [
            'The ',
            'The cat ',
            'The cat sat ',
            'The cat sat on ',
            'The cat sat on the ',
            'The cat sat on the mat.',
        ]
    )

    result = await agent.run_stream('Hello')
    assert [c async for c in result.stream(text_delta=True, debounce_by=None)] == snapshot(
        ['The ', 'cat ', 'sat ', 'on ', 'the ', 'mat.']
    )


async def test_plain_response():
    call_index = 0

    def text_stream(_messages: list[Message], _: AgentInfo) -> list[str]:
        nonlocal call_index

        call_index += 1
        return ['hello ', 'world']

    agent = Agent(FunctionModel(stream_function=text_stream), deps=None, result_type=tuple[str, str])

    with pytest.raises(AgentError) as exc_info:
        await agent.run_stream('')

    assert str(exc_info.value) == snapshot(
        'Error while running model function:stream-text_stream after 2 messages\n'
        '  caused by unexpected model behavior: Exceeded maximum retries (1) for result validation'
    )


async def test_call_retriever():
    def stream_structured_function(
        messages: list[Message], agent_info: AgentInfo
    ) -> Iterable[DeltaToolCalls] | Iterable[str]:
        if len(messages) == 1:
            assert agent_info.retrievers is not None
            assert len(agent_info.retrievers) == 1
            name = next(iter(agent_info.retrievers))
            first = messages[0]
            assert isinstance(first, UserPrompt)
            json_string = json.dumps({'x': first.content})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(args=json_string[:3])}
            yield {0: DeltaToolCall(args=json_string[3:])}
        else:
            last = messages[-1]
            assert isinstance(last, ToolReturn)
            assert agent_info.result_tools is not None
            assert len(agent_info.result_tools) == 1
            name = agent_info.result_tools[0].name
            json_data = json.dumps({'response': [last.content, 2]})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(args=json_data[:5])}
            yield {0: DeltaToolCall(args=json_data[5:])}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), deps=None, result_type=tuple[str, int])

    @agent.retriever_plain
    async def ret_a(x: str) -> str:
        assert x == 'hello'
        return f'{x} world'

    result = await agent.run_stream('hello')
    assert await result.get_response() == snapshot(('hello world', 2))


async def test_call_retriever_empty():
    def stream_structured_function(_messages: list[Message], _: AgentInfo) -> Iterable[DeltaToolCalls] | Iterable[str]:
        yield {}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), deps=None, result_type=tuple[str, int])

    with pytest.raises(AgentError, match='caused by unexpected model behavior: Received empty tool call message'):
        await agent.run_stream('hello')


async def test_call_retriever_wrong_name():
    def stream_structured_function(_messages: list[Message], _: AgentInfo) -> Iterable[DeltaToolCalls] | Iterable[str]:
        yield {0: DeltaToolCall(name='foobar', args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), deps=None, result_type=tuple[str, int])

    with pytest.raises(AgentError, match="caused by unexpected model behavior: Unknown function name: 'foobar'"):
        await agent.run_stream('hello')

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    ArgsJson,
    ArgsObject,
    Message,
    ModelStructuredResponse,
    ModelTextResponse,
    RetryPrompt,
    ToolCall,
    ToolReturn,
    UserPrompt,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import Cost
from tests.conftest import IsNow

pytestmark = pytest.mark.anyio


async def test_streamed_text_response():
    m = TestModel()

    agent = Agent(m)

    @agent.retriever_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    async with agent.run_stream('Hello') as result:
        assert not result.is_structured
        assert not result.is_complete
        assert result.all_messages() == snapshot(
            [
                UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[ToolCall(tool_name='ret_a', args=ArgsObject(args_object={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturn(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc)),
            ]
        )
        response = await result.get_data()
        assert response == snapshot('{"ret_a":"a-apple"}')
        assert result.is_complete
        assert result.cost() == snapshot(Cost())
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.all_messages() == snapshot(
            [
                UserPrompt(content='Hello', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[ToolCall(tool_name='ret_a', args=ArgsObject(args_object={'x': 'a'}))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturn(tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc)),
                ModelTextResponse(content='{"ret_a":"a-apple"}', timestamp=IsNow(tz=timezone.utc)),
            ]
        )


async def test_streamed_structured_response():
    m = TestModel()

    agent = Agent(m, result_type=tuple[str, str])

    async with agent.run_stream('') as result:
        assert result.is_structured
        assert not result.is_complete
        response = await result.get_data()
        assert response == snapshot(('a', 'a'))
        assert result.is_complete


async def test_structured_response_iter():
    async def text_stream(_messages: list[Message], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert agent_info.result_tools is not None
        assert len(agent_info.result_tools) == 1
        name = agent_info.result_tools[0].name
        json_data = json.dumps({'response': [1, 2, 3, 4]})
        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args=json_data[:15])}
        yield {0: DeltaToolCall(json_args=json_data[15:])}

    agent = Agent(FunctionModel(stream_function=text_stream), result_type=list[int])

    chunks: list[list[int]] = []
    async with agent.run_stream('') as result:
        async for structured_response, last in result.stream_structured(debounce_by=None):
            response_data = await result.validate_structured_result(structured_response, allow_partial=not last)
            chunks.append(response_data)

    assert chunks == snapshot([[1], [1, 2, 3, 4], [1, 2, 3, 4]])

    async with agent.run_stream('Hello') as result:
        with pytest.raises(UserError, match=r'stream_text\(\) can only be used with text responses'):
            async for _ in result.stream_text():
                pass


async def test_streamed_text_stream():
    m = TestModel(custom_result_text='The cat sat on the mat.')

    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        assert not result.is_structured
        # typehint to test (via static typing) that the stream type is correctly inferred
        chunks: list[str] = [c async for c in result.stream()]
        # one chunk due to group_by_temporal
        assert chunks == snapshot(['The cat sat on the mat.'])
        assert result.is_complete

    async with agent.run_stream('Hello') as result:
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

    async with agent.run_stream('Hello') as result:
        assert [c async for c in result.stream_text(delta=True, debounce_by=None)] == snapshot(
            ['The ', 'cat ', 'sat ', 'on ', 'the ', 'mat.']
        )

    async with agent.run_stream('Hello') as result:
        with pytest.raises(UserError, match=r'stream_structured\(\) can only be used with structured responses'):
            async for _ in result.stream_structured():
                pass


async def test_plain_response():
    call_index = 0

    async def text_stream(_messages: list[Message], _: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_index

        call_index += 1
        yield 'hello '
        yield 'world'

    agent = Agent(FunctionModel(stream_function=text_stream), result_type=tuple[str, str])

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for result validation'):
        async with agent.run_stream(''):
            pass

    assert call_index == 2


async def test_call_retriever():
    async def stream_structured_function(
        messages: list[Message], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            assert agent_info.retrievers is not None
            assert len(agent_info.retrievers) == 1
            name = next(iter(agent_info.retrievers))
            first = messages[0]
            assert isinstance(first, UserPrompt)
            json_string = json.dumps({'x': first.content})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(json_args=json_string[:3])}
            yield {0: DeltaToolCall(json_args=json_string[3:])}
        else:
            last = messages[-1]
            assert isinstance(last, ToolReturn)
            assert agent_info.result_tools is not None
            assert len(agent_info.result_tools) == 1
            name = agent_info.result_tools[0].name
            json_data = json.dumps({'response': [last.content, 2]})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(json_args=json_data[:5])}
            yield {0: DeltaToolCall(json_args=json_data[5:])}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), result_type=tuple[str, int])

    @agent.retriever_plain
    async def ret_a(x: str) -> str:
        assert x == 'hello'
        return f'{x} world'

    async with agent.run_stream('hello') as result:
        assert result.all_messages() == snapshot(
            [
                UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[ToolCall(tool_name='ret_a', args=ArgsJson(args_json='{"x": "hello"}'))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturn(tool_name='ret_a', content='hello world', timestamp=IsNow(tz=timezone.utc)),
            ]
        )
        assert await result.get_data() == snapshot(('hello world', 2))
        assert result.all_messages() == snapshot(
            [
                UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[ToolCall(tool_name='ret_a', args=ArgsJson(args_json='{"x": "hello"}'))],
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ToolReturn(tool_name='ret_a', content='hello world', timestamp=IsNow(tz=timezone.utc)),
                ModelStructuredResponse(
                    calls=[
                        ToolCall(
                            tool_name='final_result',
                            args=ArgsJson(args_json='{"response": ["hello world", 2]}'),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )


async def test_call_retriever_empty():
    async def stream_structured_function(_messages: list[Message], _: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), result_type=tuple[str, int])

    with pytest.raises(UnexpectedModelBehavior, match='Received empty tool call message'):
        async with agent.run_stream('hello'):
            pass


async def test_call_retriever_wrong_name():
    async def stream_structured_function(_messages: list[Message], _: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {0: DeltaToolCall(name='foobar', json_args='{}')}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), result_type=tuple[str, int])

    @agent.retriever_plain
    async def ret_a(x: str) -> str:
        return x

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for result validation'):
        async with agent.run_stream('hello'):
            pass
    assert agent.last_run_messages == snapshot(
        [
            UserPrompt(content='hello', timestamp=IsNow(tz=timezone.utc)),
            ModelStructuredResponse(
                calls=[ToolCall(tool_name='foobar', args=ArgsJson(args_json='{}'))], timestamp=IsNow(tz=timezone.utc)
            ),
            RetryPrompt(
                content="Unknown tool name: 'foobar'. Available tools: ret_a, final_result",
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )

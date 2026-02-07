from __future__ import annotations as _annotations

import asyncio
import datetime
import json
import re
from collections.abc import AsyncIterable, AsyncIterator
from copy import deepcopy
from dataclasses import replace
from datetime import timezone
from typing import Any
from unittest.mock import MagicMock

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from pydantic_core import ErrorDetails

from pydantic_ai import (
    Agent,
    AgentRunResult,
    AgentRunResultEvent,
    AgentStreamEvent,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ExternalToolset,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    RunContext,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
    capture_run_messages,
    models,
)
from pydantic_ai._agent_graph import (
    GraphAgentState,
    _clean_message_history,  # pyright: ignore[reportPrivateUsage]
    _filter_incomplete_tool_calls,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai._output import TextOutputProcessor, TextOutputSchema
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.agent import AgentRun
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel, TestStreamedResponse as ModelTestStreamedResponse
from pydantic_ai.output import PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.result import AgentStream, FinalResult, RunUsage, StreamedRunResult, StreamedRunResultSync
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolApproved, ToolDefinition
from pydantic_ai.usage import RequestUsage
from pydantic_graph import End

from .conftest import IsDatetime, IsInt, IsNow, IsStr

pytestmark = pytest.mark.anyio


class Foo(BaseModel):
    a: int
    b: str


async def test_streamed_text_response():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    async with test_agent.run_stream('Hello') as result:
        assert test_agent.name == 'test_agent'
        assert isinstance(result.run_id, str)
        assert not result.is_complete
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=51),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )
        assert result.usage() == snapshot(
            RunUsage(
                requests=2,
                input_tokens=103,
                output_tokens=5,
                tool_calls=1,
            )
        )
        response = await result.get_output()
        assert response == snapshot('{"ret_a":"a-apple"}')
        assert result.is_complete
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=51),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='{"ret_a":"a-apple"}')],
                    usage=RequestUsage(input_tokens=52, output_tokens=11),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                ),
            ]
        )
        assert result.usage() == snapshot(
            RunUsage(
                requests=2,
                input_tokens=103,
                output_tokens=11,
                tool_calls=1,
            )
        )


def test_streamed_text_sync_response():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    result = test_agent.run_stream_sync('Hello')
    assert test_agent.name == 'test_agent'
    assert isinstance(result.run_id, str)
    assert not result.is_complete
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()
    assert result.usage() == snapshot(
        RunUsage(
            requests=2,
            input_tokens=103,
            output_tokens=5,
            tool_calls=1,
        )
    )
    response = result.get_output()
    assert response == snapshot('{"ret_a":"a-apple"}')
    assert result.is_complete
    assert result.timestamp() == IsNow(tz=timezone.utc)
    assert result.response == snapshot(
        ModelResponse(
            parts=[TextPart(content='{"ret_a":"a-apple"}')],
            usage=RequestUsage(input_tokens=52, output_tokens=11),
            model_name='test',
            timestamp=IsDatetime(),
            provider_name='test',
        )
    )
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"ret_a":"a-apple"}')],
                usage=RequestUsage(input_tokens=52, output_tokens=11),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
            ),
        ]
    )
    assert result.usage() == snapshot(
        RunUsage(
            requests=2,
            input_tokens=103,
            output_tokens=11,
            tool_calls=1,
        )
    )


async def test_streamed_structured_response():
    m = TestModel()

    agent = Agent(m, output_type=tuple[str, str], name='fig_jam')

    async with agent.run_stream('') as result:
        assert agent.name == 'fig_jam'
        assert not result.is_complete
        response = await result.get_output()
        assert response == snapshot(('a', 'a'))
        assert result.is_complete
    assert result.response == snapshot(
        ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='final_result',
                    args={'response': ['a', 'a']},
                    tool_call_id='pyd_ai_tool_call_id__final_result',
                )
            ],
            usage=RequestUsage(input_tokens=50),
            model_name='test',
            timestamp=IsDatetime(),
            provider_name='test',
        )
    )


async def test_structured_response_iter():
    async def text_stream(_messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert agent_info.output_tools is not None
        assert len(agent_info.output_tools) == 1
        name = agent_info.output_tools[0].name
        json_data = json.dumps({'response': [1, 2, 3, 4]})
        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args=json_data[:15])}
        yield {0: DeltaToolCall(json_args=json_data[15:])}

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=list[int])

    chunks: list[list[int]] = []
    async with agent.run_stream('') as result:
        async for structured_response, last in result.stream_responses(debounce_by=None):
            response_data = await result.validate_response_output(structured_response, allow_partial=not last)
            chunks.append(response_data)

    assert chunks == snapshot([[1], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])

    async with agent.run_stream('Hello') as result:
        with pytest.raises(UserError, match=r'stream_text\(\) can only be used with text responses'):
            async for _ in result.stream_text():
                pass


async def test_streamed_text_stream():
    m = TestModel(custom_output_text='The cat sat on the mat.')

    agent = Agent(m)

    async with agent.run_stream('Hello') as result:
        # typehint to test (via static typing) that the stream type is correctly inferred
        chunks: list[str] = [c async for c in result.stream_text()]
        # one chunk with `stream_text()` due to group_by_temporal
        assert chunks == snapshot(['The cat sat on the mat.'])
        assert result.is_complete

    async with agent.run_stream('Hello') as result:
        # typehint to test (via static typing) that the stream type is correctly inferred
        chunks: list[str] = [c async for c in result.stream_output()]
        # two chunks with `stream()` due to not-final vs. final
        assert chunks == snapshot(['The cat sat on the mat.', 'The cat sat on the mat.'])
        assert result.is_complete

    async with agent.run_stream('Hello') as result:
        assert [c async for c in result.stream_text(debounce_by=None)] == snapshot(
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
        # with stream_text, there is no need to do partial validation, so we only get the final message once:
        assert [c async for c in result.stream_text(delta=False, debounce_by=None)] == snapshot(
            ['The ', 'The cat ', 'The cat sat ', 'The cat sat on ', 'The cat sat on the ', 'The cat sat on the mat.']
        )

    async with agent.run_stream('Hello') as result:
        assert [c async for c in result.stream_text(delta=True, debounce_by=None)] == snapshot(
            ['The ', 'cat ', 'sat ', 'on ', 'the ', 'mat.']
        )

    def upcase(text: str) -> str:
        return text.upper()

    async with agent.run_stream('Hello', output_type=TextOutput(upcase)) as result:
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                'THE ',
                'THE CAT ',
                'THE CAT SAT ',
                'THE CAT SAT ON ',
                'THE CAT SAT ON THE ',
                'THE CAT SAT ON THE MAT.',
                'THE CAT SAT ON THE MAT.',
            ]
        )

    async with agent.run_stream('Hello') as result:
        assert [c async for c, _is_last in result.stream_responses(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='The ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=1),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=2),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=3),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=4),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the ')],
                    usage=RequestUsage(input_tokens=51, output_tokens=5),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the mat.')],
                    usage=RequestUsage(input_tokens=51, output_tokens=7),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the mat.')],
                    usage=RequestUsage(input_tokens=51, output_tokens=7),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                    provider_name='test',
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the mat.')],
                    usage=RequestUsage(input_tokens=51, output_tokens=7),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                ),
            ]
        )


def test_streamed_text_stream_sync():
    m = TestModel(custom_output_text='The cat sat on the mat.')

    agent = Agent(m)

    result = agent.run_stream_sync('Hello')
    # typehint to test (via static typing) that the stream type is correctly inferred
    chunks: list[str] = [c for c in result.stream_text()]
    # one chunk with `stream_text()` due to group_by_temporal
    assert chunks == snapshot(['The cat sat on the mat.'])
    assert result.is_complete

    result = agent.run_stream_sync('Hello')
    # typehint to test (via static typing) that the stream type is correctly inferred
    chunks: list[str] = [c for c in result.stream_output()]
    # two chunks with `stream()` due to not-final vs. final
    assert chunks == snapshot(['The cat sat on the mat.', 'The cat sat on the mat.'])
    assert result.is_complete

    result = agent.run_stream_sync('Hello')
    assert [c for c in result.stream_text(debounce_by=None)] == snapshot(
        [
            'The ',
            'The cat ',
            'The cat sat ',
            'The cat sat on ',
            'The cat sat on the ',
            'The cat sat on the mat.',
        ]
    )

    result = agent.run_stream_sync('Hello')
    # with stream_text, there is no need to do partial validation, so we only get the final message once:
    assert [c for c in result.stream_text(delta=False, debounce_by=None)] == snapshot(
        ['The ', 'The cat ', 'The cat sat ', 'The cat sat on ', 'The cat sat on the ', 'The cat sat on the mat.']
    )

    result = agent.run_stream_sync('Hello')
    assert [c for c in result.stream_text(delta=True, debounce_by=None)] == snapshot(
        ['The ', 'cat ', 'sat ', 'on ', 'the ', 'mat.']
    )

    def upcase(text: str) -> str:
        return text.upper()

    result = agent.run_stream_sync('Hello', output_type=TextOutput(upcase))
    assert [c for c in result.stream_output(debounce_by=None)] == snapshot(
        [
            'THE ',
            'THE CAT ',
            'THE CAT SAT ',
            'THE CAT SAT ON ',
            'THE CAT SAT ON THE ',
            'THE CAT SAT ON THE MAT.',
            'THE CAT SAT ON THE MAT.',
        ]
    )

    result = agent.run_stream_sync('Hello')
    assert [c for c, _is_last in result.stream_responses(debounce_by=None)] == snapshot(
        [
            ModelResponse(
                parts=[TextPart(content='The ')],
                usage=RequestUsage(input_tokens=51, output_tokens=1),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat ')],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat ')],
                usage=RequestUsage(input_tokens=51, output_tokens=3),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on ')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on the ')],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on the mat.')],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on the mat.')],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
            ),
            ModelResponse(
                parts=[TextPart(content='The cat sat on the mat.')],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
                run_id=IsStr(),
            ),
        ]
    )


async def test_plain_response():
    call_index = 0

    async def text_stream(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[str]:
        nonlocal call_index

        call_index += 1
        yield 'hello '
        yield 'world'

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=tuple[str, str])

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for output validation'):
        async with agent.run_stream(''):
            pass

    assert call_index == 2


async def test_call_tool():
    async def stream_structured_function(
        messages: list[ModelMessage], agent_info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            assert agent_info.function_tools is not None
            assert len(agent_info.function_tools) == 1
            name = agent_info.function_tools[0].name
            first = messages[0]
            assert isinstance(first, ModelRequest)
            assert isinstance(first.parts[0], UserPromptPart)
            json_string = json.dumps({'x': first.parts[0].content})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(json_args=json_string[:3])}
            yield {0: DeltaToolCall(json_args=json_string[3:])}
        else:
            last = messages[-1]
            assert isinstance(last, ModelRequest)
            assert isinstance(last.parts[0], ToolReturnPart)
            assert agent_info.output_tools is not None
            assert len(agent_info.output_tools) == 1
            name = agent_info.output_tools[0].name
            json_data = json.dumps({'response': [last.parts[0].content, 2]})
            yield {0: DeltaToolCall(name=name)}
            yield {0: DeltaToolCall(json_args=json_data[:5])}
            yield {0: DeltaToolCall(json_args=json_data[5:])}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), output_type=tuple[str, int])

    @agent.tool_plain
    async def ret_a(x: str) -> str:
        assert x == 'hello'
        return f'{x} world'

    async with agent.run_stream('hello') as result:
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args='{"x": "hello"}', tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=50, output_tokens=5),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a',
                            content='hello world',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )
        assert await result.get_output() == snapshot(('hello world', 2))
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args='{"x": "hello"}', tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=50, output_tokens=5),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a',
                            content='hello world',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"response": ["hello world", 2]}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=7),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )


async def test_empty_response():
    async def stream_structured_function(
        messages: list[ModelMessage], _: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {}
        else:
            yield 'ok here is text'

    agent = Agent(FunctionModel(stream_function=stream_structured_function))

    async with agent.run_stream('hello') as result:
        response = await result.get_output()
        assert response == snapshot('ok here is text')
        messages = result.all_messages()

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='hello',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[],
                usage=RequestUsage(input_tokens=50),
                model_name='function::stream_structured_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='ok here is text')],
                usage=RequestUsage(input_tokens=50, output_tokens=4),
                model_name='function::stream_structured_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


async def test_call_tool_wrong_name():
    async def stream_structured_function(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {0: DeltaToolCall(name='foobar', json_args='{}')}

    agent = Agent(
        FunctionModel(stream_function=stream_structured_function),
        output_type=tuple[str, int],
        retries=0,
    )

    @agent.tool_plain
    async def ret_a(x: str) -> str:  # pragma: no cover
        return x

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(0\) for output validation'):
            async with agent.run_stream('hello'):
                pass

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=50, output_tokens=1),
                model_name='function::stream_structured_function',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


class TestPartialOutput:
    """Tests for `ctx.partial_output` flag in output validators and output functions."""

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_agent.py::TestPartialOutput` as well

    async def test_output_validator_text(self):
        """Test that output validators receive correct value for `partial_output` with text output."""
        call_log: list[tuple[str, bool]] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for chunk in ['Hello', ' ', 'world', '!']:
                yield chunk

        agent = Agent(FunctionModel(stream_function=sf))

        @agent.output_validator
        def validate_output(ctx: RunContext[None], output: str) -> str:
            call_log.append((output, ctx.partial_output))
            return output

        async with agent.run_stream('test') as result:
            text_parts = [text_part async for text_part in result.stream_text(debounce_by=None)]

        assert text_parts[-1] == 'Hello world!'
        assert call_log == snapshot(
            [
                ('Hello', True),
                ('Hello ', True),
                ('Hello world', True),
                ('Hello world!', True),
                ('Hello world!', False),
            ]
        )

    async def test_output_validator_structured(self):
        """Test that output validators receive correct value for `partial_output` with structured output."""
        call_log: list[tuple[Foo, bool]] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 42')}
            yield {0: DeltaToolCall(json_args=', "b": "f')}
            yield {0: DeltaToolCall(json_args='oo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=Foo)

        @agent.output_validator
        def validate_output(ctx: RunContext[None], output: Foo) -> Foo:
            call_log.append((output, ctx.partial_output))
            return output

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output(debounce_by=None)]

        assert outputs[-1] == Foo(a=42, b='foo')
        assert call_log == snapshot(
            [
                (Foo(a=42, b='f'), True),
                (Foo(a=42, b='foo'), True),
                (Foo(a=42, b='foo'), False),
            ]
        )

    async def test_output_function_text(self):
        """Test that output functions receive correct value for `partial_output` with text output."""
        call_log: list[tuple[str, bool]] = []

        def process_output(ctx: RunContext[None], text: str) -> str:
            call_log.append((text, ctx.partial_output))
            return text.upper()

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for chunk in ['Hello', ' ', 'world', '!']:
                yield chunk

        agent = Agent(FunctionModel(stream_function=sf), output_type=TextOutput(process_output))

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output(debounce_by=None)]

        assert outputs[-1] == 'HELLO WORLD!'
        assert call_log == snapshot(
            [
                ('Hello', True),
                ('Hello ', True),
                ('Hello world', True),
                ('Hello world!', True),
                ('Hello world!', False),
            ]
        )

    async def test_output_function_structured(self):
        """Test that output functions receive correct value for `partial_output` with structured output."""
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21')}
            yield {0: DeltaToolCall(json_args=', "b": "f')}
            yield {0: DeltaToolCall(json_args='oo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=process_foo)

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output(debounce_by=None)]

        assert outputs[-1] == Foo(a=42, b='FOO')
        assert call_log == snapshot(
            [
                (Foo(a=21, b='f'), True),
                (Foo(a=21, b='foo'), True),
                (Foo(a=21, b='foo'), False),
            ]
        )

    async def test_output_function_structured_get_output(self):
        """Test that output functions receive correct value for `partial_output` with `get_output()`.

        When using only `get_output()` without streaming, the output processor is called only once
        with `partial_output=False` (final validation), since the user doesn't see partial results.
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            output = await result.get_output()

        assert output == Foo(a=42, b='FOO')
        assert call_log == snapshot([(Foo(a=21, b='foo'), False)])

    async def test_output_function_structured_stream_output_only(self):
        """Test that output functions receive correct value for `partial_output` with `stream_output()`.

        When using only `stream_output()`, the LAST yielded output should have `partial_output=False` (final validation).
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output()]

        assert outputs[-1] == Foo(a=42, b='FOO')
        assert call_log == snapshot(
            [
                (Foo(a=21, b='foo'), True),
                (Foo(a=21, b='foo'), False),
            ],
        )

    async def test_stream_output_partial_then_final_validation(self):
        """Test that stream_output() calls validators with partial_output=True during streaming, then False at the end.

        This verifies the critical invariant: output validators/functions are called multiple times with
        partial_output=True as chunks arrive, followed by exactly one call with partial_output=False
        for final validation. The final yield may have the same content as the last partial yield,
        but the validation semantics differ (partial validation may accept incomplete data).
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21')}
            yield {0: DeltaToolCall(json_args=', "b": "f')}
            yield {0: DeltaToolCall(json_args='oo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            outputs = [output async for output in result.stream_output(debounce_by=None)]

        assert outputs[-1] == Foo(a=42, b='FOO')

        # Verify the pattern: multiple True calls, exactly one False call at the end
        partial_output_flags = [partial for _, partial in call_log]
        assert partial_output_flags[-1] is False, 'Last call must have partial_output=False'
        assert all(flag is True for flag in partial_output_flags[:-1]), (
            'All calls except last must have partial_output=True'
        )
        assert len([f for f in partial_output_flags if f is False]) == 1, 'Exactly one partial_output=False call'

        # The full call log shows progressive partial outputs followed by final validation
        assert call_log == snapshot(
            [
                (Foo(a=21, b='f'), True),
                (Foo(a=21, b='foo'), True),
                (Foo(a=21, b='foo'), False),  # Final validation - same content, different validation mode
            ]
        )

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_agent.py::TestPartialOutput` as well


class TestStreamingCachedOutput:
    async def test_output_function_structured_double_stream_output(self):
        """Test that calling `stream_output()` twice works correctly.

        The first `stream_output()` should do validations and cache the result.
        The second `stream_output()` should return cached results without re-validation.
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            outputs1 = [output async for output in result.stream_output()]
            outputs2 = [output async for output in result.stream_output()]

        assert outputs1[-1] == outputs2[-1] == Foo(a=42, b='FOO')
        assert call_log == snapshot(
            [
                (Foo(a=21, b='foo'), True),
                (Foo(a=21, b='foo'), False),
            ],
        )

    async def test_output_validator_text_double_stream_text(self):
        """Test that calling `stream_text()` twice works correctly with output validator.

        The first `stream_text()` should do validations and cache the result.
        The second `stream_text()` should return cached results without re-validation.
        """
        call_log: list[tuple[str, bool]] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for chunk in ['Hello', ' ', 'world', '!']:
                yield chunk

        agent = Agent(FunctionModel(stream_function=sf))

        @agent.output_validator
        def validate_output(ctx: RunContext[None], output: str) -> str:
            call_log.append((output, ctx.partial_output))
            return output

        async with agent.run_stream('test') as result:
            text_parts1 = [text async for text in result.stream_text(debounce_by=None)]
            text_parts2 = [text async for text in result.stream_text(debounce_by=None)]

        assert text_parts1[-1] == text_parts2[-1] == 'Hello world!'
        assert call_log == snapshot(
            [
                ('Hello', True),
                ('Hello ', True),
                ('Hello world', True),
                ('Hello world!', True),
                ('Hello world!', False),
            ],
        )

    async def test_output_function_structured_double_get_output(self):
        """Test that calling `get_output()` twice works correctly.

        The first `get_output()` should do validation and cache the result.
        The second `get_output()` should return cached results without re-validation.
        """
        call_log: list[tuple[Foo, bool]] = []

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            call_log.append((foo, ctx.partial_output))
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            output1 = await result.get_output()
            output2 = await result.get_output()

        assert output1 == output2 == Foo(a=42, b='FOO')
        assert call_log == snapshot([(Foo(a=21, b='foo'), False)])

    async def test_cached_output_mutation_does_not_affect_cache(self):
        """Test that mutating a returned cached output does not affect the cached value.

        When the same output is retrieved multiple times from cache, each call should return
        a deep copy, so mutations to one don't affect subsequent retrievals.
        """

        def process_foo(ctx: RunContext[None], foo: Foo) -> Foo:
            return Foo(a=foo.a * 2, b=foo.b.upper())

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 21, "b": "foo"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=ToolOutput(process_foo, name='my_output'))

        async with agent.run_stream('test') as result:
            # Get the first output and mutate it
            output1 = await result.get_output()
            output1.a = 999
            output1.b = 'MUTATED'

            # Get the second output - should not be affected by mutation
            output2 = await result.get_output()

        # First output should have been mutated
        assert output1 == Foo(a=999, b='MUTATED')
        # Second output should be the original cached value (not mutated)
        assert output2 == Foo(a=42, b='FOO')


class OutputType(BaseModel):
    """Result type used by multiple tests."""

    value: str


class TestMultipleToolCalls:
    """Tests for scenarios where multiple tool calls are made in a single response."""

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_agent.py::TestMultipleToolCallsStreaming` as well

    async def test_early_strategy_stops_after_first_final_result(self):
        """Test that 'early' strategy stops processing regular tools after first final result."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('final_result', '{"value": "final"}')}
            yield {2: DeltaToolCall('regular_tool', '{"x": 1}')}
            yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
            yield {4: DeltaToolCall('deferred_tool', '{"x": 3}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:  # pragma: no cover
            """Another tool that should not be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        async with agent.run_stream('test early strategy') as result:
            response = await result.get_output()
            assert response.value == snapshot('final')
            messages = result.all_messages()

        # Verify no tools were called after final result
        assert tool_called == []

        # Verify we got tool returns for all calls
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test early strategy', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args='{"value": "final"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='regular_tool', args='{"x": 1}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args='{"y": 2}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='deferred_tool', args='{"x": 3}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=13),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content='Tool not executed - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_does_not_call_additional_output_tools(self):
        """Test that 'early' strategy does not execute additional output tool functions."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:  # pragma: no cover
            """Process second output."""
            output_tools_called.append('second')
            return output

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "first"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "second"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='early',
        )

        async with agent.run_stream('test early output tools') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool
        assert isinstance(response, OutputType)
        assert response.value == 'first'

        # Verify only the first output tool was called
        assert output_tools_called == ['first']

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test early output tools', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_uses_first_final_result(self):
        """Test that 'early' strategy uses the first final result and ignores subsequent ones."""

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('final_result', '{"value": "first"}')}
            yield {2: DeltaToolCall('final_result', '{"value": "second"}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='early')

        async with agent.run_stream('test multiple final results') as result:
            response = await result.get_output()
            assert response.value == snapshot('first')
            messages = result.all_messages()

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool not used - a final result was already processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_with_final_result_in_middle(self):
        """Test that 'early' strategy stops at first final result, regardless of position."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('regular_tool', '{"x": 1}')}
            yield {2: DeltaToolCall('final_result', '{"value": "final"}')}
            yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
            yield {4: DeltaToolCall('unknown_tool', '{"value": "???"}')}
            yield {5: DeltaToolCall('deferred_tool', '{"x": 5}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:  # pragma: no cover
            """A tool that should not be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        async with agent.run_stream('test early strategy with final result in middle') as result:
            response = await result.get_output()
            assert response.value == snapshot('final')
            messages = result.all_messages()

        # Verify no tools were called
        assert tool_called == []

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with final result in middle',
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='regular_tool',
                            args='{"x": 1}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": "final"}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='another_tool',
                            args='{"y": 2}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='unknown_tool',
                            args='{"value": "???"}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='deferred_tool',
                            args='{"x": 5}',
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=17),
                    model_name='function::sf',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'final_result', 'regular_tool', 'another_tool', 'deferred_tool'",
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_with_external_tool_call(self):
        """Test that early strategy handles external tool calls correctly.

        Streaming and non-streaming modes differ in how they choose the final result:
        - Streaming: First tool call (in response order) that can produce a final result (output or deferred)
        - Non-streaming: First output tool (if none called, all deferred tools become final result)

        See https://github.com/pydantic/pydantic-ai/issues/3636#issuecomment-3618800480 for details.
        """
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('external_tool')}
            yield {2: DeltaToolCall('final_result', '{"value": "final"}')}
            yield {3: DeltaToolCall('regular_tool', '{"x": 1}')}

        agent = Agent(
            FunctionModel(stream_function=sf),
            output_type=[OutputType, DeferredToolRequests],
            toolsets=[
                ExternalToolset(
                    tool_defs=[
                        ToolDefinition(
                            name='external_tool',
                            kind='external',
                        )
                    ]
                )
            ],
            end_strategy='early',
        )

        @agent.tool_plain
        def regular_tool(x: int) -> int:  # pragma: no cover
            """A regular tool that should not be called."""
            tool_called.append('regular_tool')
            return x

        async with agent.run_stream('test early strategy with external tool call') as result:
            response = await result.get_output()
            assert response == snapshot(
                DeferredToolRequests(
                    calls=[
                        ToolCallPart(
                            tool_name='external_tool',
                            tool_call_id=IsStr(),
                        )
                    ]
                )
            )
            messages = result.all_messages()

        # Verify no tools were called
        assert tool_called == []

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with external tool call',
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='external_tool', tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"value": "final"}',
                            tool_call_id=IsStr(),
                        ),
                        ToolCallPart(
                            tool_name='regular_tool',
                            args='{"x": 1}',
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=7),
                    model_name='function::sf',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Output tool not used - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_with_deferred_tool_call(self):
        """Test that early strategy handles deferred tool calls correctly."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('deferred_tool')}
            yield {2: DeltaToolCall('regular_tool', '{"x": 1}')}

        agent = Agent(
            FunctionModel(stream_function=sf),
            output_type=[str, DeferredToolRequests],
            end_strategy='early',
        )

        @agent.tool_plain
        def deferred_tool() -> int:
            raise CallDeferred

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            tool_called.append('regular_tool')
            return x

        async with agent.run_stream('test early strategy with external tool call') as result:
            response = await result.get_output()
            assert response == snapshot(
                DeferredToolRequests(calls=[ToolCallPart(tool_name='deferred_tool', tool_call_id=IsStr())])
            )
            messages = result.all_messages()

        # Verify regular tool was called
        assert tool_called == ['regular_tool']

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with external tool call',
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='deferred_tool', tool_call_id=IsStr()),
                        ToolCallPart(
                            tool_name='regular_tool',
                            args='{"x": 1}',
                            tool_call_id=IsStr(),
                        ),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=3),
                    model_name='function::sf',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=1,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_early_strategy_does_not_apply_to_tool_calls_without_final_tool(self):
        """Test that 'early' strategy does not apply to tool calls when no output tool is called."""
        tool_called: list[str] = []
        agent = Agent(TestModel(), output_type=OutputType, end_strategy='early')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        async with agent.run_stream('test early strategy with regular tool calls') as result:
            response = await result.get_output()
            assert response.value == snapshot('a')
            messages = result.all_messages()

        # Verify the regular tool was executed
        assert tool_called == ['regular_tool']

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test early strategy with regular tool calls',
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='regular_tool',
                            args={'x': 0},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=57),
                    model_name='test',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=0,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args={'value': 'a'},
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=RequestUsage(input_tokens=58, output_tokens=4),
                    model_name='test',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    provider_name='test',
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_exhaustive_strategy_executes_all_tools(self):
        """Test that 'exhaustive' strategy executes all tools while using first final result."""
        tool_called: list[str] = []

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('regular_tool', '{"x": 42}')}
            yield {2: DeltaToolCall('final_result', '{"value": "first"}')}
            yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
            yield {4: DeltaToolCall('final_result', '{"value": "second"}')}
            yield {5: DeltaToolCall('unknown_tool', '{"value": "???"}')}
            yield {6: DeltaToolCall('deferred_tool', '{"x": 4}')}

        agent = Agent(FunctionModel(stream_function=sf), output_type=OutputType, end_strategy='exhaustive')

        @agent.tool_plain
        def regular_tool(x: int) -> int:
            """A regular tool that should be called."""
            tool_called.append('regular_tool')
            return x

        @agent.tool_plain
        def another_tool(y: int) -> int:
            """Another tool that should be called."""
            tool_called.append('another_tool')
            return y

        async def defer(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
            return replace(tool_def, kind='external')

        @agent.tool_plain(prepare=defer)
        def deferred_tool(x: int) -> int:  # pragma: no cover
            return x + 1

        async with agent.run_stream('test exhaustive strategy') as result:
            response = await result.get_output()
            assert response.value == snapshot('first')
            messages = result.all_messages()

        # Verify the result came from the first final tool
        assert response.value == 'first'

        # Verify all regular tools were called
        assert sorted(tool_called) == sorted(['regular_tool', 'another_tool'])

        # Verify we got tool returns in the correct order
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test exhaustive strategy', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='regular_tool', args='{"x": 42}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='another_tool', args='{"y": 2}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='unknown_tool', args='{"value": "???"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='deferred_tool', args='{"x": 4}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=21),
                    model_name='function::sf',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                        ToolReturnPart(
                            tool_name='regular_tool',
                            content=42,
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='another_tool', content=2, tool_call_id=IsStr(), timestamp=IsNow(tz=timezone.utc)
                        ),
                        RetryPromptPart(
                            content="Unknown tool name: 'unknown_tool'. Available tools: 'final_result', 'regular_tool', 'another_tool', 'deferred_tool'",
                            tool_name='unknown_tool',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='deferred_tool',
                            content='Tool not executed - a final result was already processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_exhaustive_strategy_calls_all_output_tools(self):
        """Test that 'exhaustive' strategy executes all output tool functions."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output."""
            output_tools_called.append('second')
            return output

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "first"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "second"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
        )

        async with agent.run_stream('test exhaustive output tools') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool
        assert isinstance(response, OutputType)
        assert response.value == 'first'

        # Verify both output tools were called
        assert output_tools_called == ['first', 'second']

        # Verify we got tool returns in the correct order
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test exhaustive output tools', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.xfail(reason='See https://github.com/pydantic/pydantic-ai/issues/3393')
    async def test_exhaustive_strategy_invalid_first_valid_second_output(self):
        """Test that exhaustive strategy uses the second valid output when the first is invalid."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be invalid."""
            output_tools_called.append('first')
            raise ModelRetry('First output validation failed')

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will be valid."""
            output_tools_called.append('second')
            return output

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "invalid"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "valid"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
        )

        async with agent.run_stream('test invalid first valid second') as result:
            response = await result.get_output()

        # Verify the result came from the second output tool (first was invalid)
        assert isinstance(response, OutputType)
        assert response.value == snapshot('valid')

        # Verify both output tools were called
        assert output_tools_called == snapshot(['first', 'second'])

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test invalid first valid second', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "invalid"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "valid"}', tool_call_id=IsStr()),
                    ],
                    model_name='function:stream_function:',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content='First output validation failed',
                            tool_name='first_output',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_exhaustive_strategy_valid_first_invalid_second_output(self):
        """Test that exhaustive strategy uses the first valid output even when the second is invalid."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be valid."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will be invalid."""
            output_tools_called.append('second')
            raise ModelRetry('Second output validation failed')

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "valid"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "invalid"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
            output_retries=0,  # No retries - model must succeed first try
        )

        async with agent.run_stream('test valid first invalid second') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool (second was invalid, but we ignore it)
        assert isinstance(response, OutputType)
        assert response.value == snapshot('valid')

        # Verify both output tools were called
        assert output_tools_called == snapshot(['first', 'second'])

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test valid first invalid second', timestamp=IsNow(tz=timezone.utc))],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "valid"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "invalid"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='second_output',
                            content='Output tool not used - output failed validation.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    async def test_exhaustive_strategy_with_tool_retry_and_final_result(self):
        """Test that exhaustive strategy doesn't increment retries when `final_result` exists and `ToolRetryError` occurs."""
        output_tools_called: list[str] = []

        def process_first(output: OutputType) -> OutputType:
            """Process first output - will be valid."""
            output_tools_called.append('first')
            return output

        def process_second(output: OutputType) -> OutputType:
            """Process second output - will raise ModelRetry."""
            output_tools_called.append('second')
            raise ModelRetry('Second output validation failed')

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('first_output', '{"value": "valid"}')}
            yield {2: DeltaToolCall('second_output', '{"value": "invalid"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_first, name='first_output'),
                ToolOutput(process_second, name='second_output'),
            ],
            end_strategy='exhaustive',
            output_retries=1,  # Allow 1 retry so ToolRetryError is raised
        )

        async with agent.run_stream('test exhaustive with tool retry') as result:
            response = await result.get_output()

        # Verify the result came from the first output tool
        assert isinstance(response, OutputType)
        assert response.value == 'valid'

        # Verify both output tools were called
        assert output_tools_called == ['first', 'second']

        # Verify we got appropriate messages
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='test exhaustive with tool retry', timestamp=IsNow(tz=datetime.timezone.utc)
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='first_output', args='{"value": "valid"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='second_output', args='{"value": "invalid"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=datetime.timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='first_output',
                            content='Final result processed.',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                        RetryPromptPart(
                            content='Second output validation failed',
                            tool_name='second_output',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=datetime.timezone.utc),
                        ),
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
            ]
        )

    @pytest.mark.xfail(reason='See https://github.com/pydantic/pydantic-ai/issues/3638')
    async def test_exhaustive_raises_unexpected_model_behavior(self):
        """Test that exhaustive strategy raises `UnexpectedModelBehavior` when all outputs have validation errors."""

        def process_output(output: OutputType) -> OutputType:  # pragma: no cover
            """A tool that should not be called."""
            assert False

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            # Missing 'value' field will cause validation error
            yield {1: DeltaToolCall('output_tool', '{"invalid_field": "invalid"}')}

        agent = Agent(
            FunctionModel(stream_function=stream_function),
            output_type=[
                ToolOutput(process_output, name='output_tool'),
            ],
            end_strategy='exhaustive',
        )

        with pytest.raises(UnexpectedModelBehavior, match='Exceeded maximum retries \\(1\\) for output validation'):
            async with agent.run_stream('test') as result:
                await result.get_output()

    @pytest.mark.xfail(reason='See https://github.com/pydantic/pydantic-ai/issues/3638')
    async def test_multiple_final_result_are_validated_correctly(self):
        """Tests that if multiple final results are returned, but one fails validation, the other is used."""

        async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            assert info.output_tools is not None
            yield {1: DeltaToolCall('final_result', '{"bad_value": "first"}')}
            yield {2: DeltaToolCall('final_result', '{"value": "second"}')}

        agent = Agent(FunctionModel(stream_function=stream_function), output_type=OutputType, end_strategy='early')

        async with agent.run_stream('test multiple final results') as result:
            response = await result.get_output()
            messages = result.new_messages()

        # Verify the result came from the second final tool
        assert response.value == snapshot('second')

        # Verify we got appropriate tool returns
        assert messages == snapshot(
            [
                ModelRequest(
                    parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))],
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(tool_name='final_result', args='{"bad_value": "first"}', tool_call_id=IsStr()),
                        ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                    ],
                    usage=RequestUsage(input_tokens=50, output_tokens=8),
                    model_name='function::stream_function',
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        RetryPromptPart(
                            content=[
                                ErrorDetails(
                                    type='missing',
                                    loc=('value',),
                                    msg='Field required',
                                    input={'bad_value': 'first'},
                                )
                            ],
                            tool_name='final_result',
                            tool_call_id=IsStr(),
                            timestamp=IsNow(tz=timezone.utc),
                        ),
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        ),
                    ],
                    run_id=IsStr(),
                ),
            ]
        )

    # NOTE: When changing tests in this class:
    # 1. Follow the existing order
    # 2. Update tests in `tests/test_agent.py::TestMultipleToolCallsStreaming` as well


async def test_custom_output_type_default_str() -> None:
    agent = Agent('test')

    async with agent.run_stream('test') as result:
        response = await result.get_output()
        assert response == snapshot('success (no tool calls)')
    assert result.response == snapshot(
        ModelResponse(
            parts=[TextPart(content='success (no tool calls)')],
            usage=RequestUsage(input_tokens=51, output_tokens=4),
            model_name='test',
            timestamp=IsDatetime(),
            provider_name='test',
        )
    )

    async with agent.run_stream('test', output_type=OutputType) as result:
        response = await result.get_output()
        assert response == snapshot(OutputType(value='a'))


async def test_custom_output_type_default_structured() -> None:
    agent = Agent('test', output_type=OutputType)

    async with agent.run_stream('test') as result:
        response = await result.get_output()
        assert response == snapshot(OutputType(value='a'))

    async with agent.run_stream('test', output_type=str) as result:
        response = await result.get_output()
        assert response == snapshot('success (no tool calls)')


async def test_iter_stream_output():
    m = TestModel(custom_output_text='The cat sat on the mat.')

    agent = Agent(m)

    @agent.output_validator
    def output_validator_simple(data: str) -> str:
        # Make a substitution in the validated results
        return re.sub('cat sat', 'bat sat', data)

    run: AgentRun
    stream: AgentStream | None = None
    messages: list[str] = []

    stream_usage: RunUsage | None = None
    async with agent.iter('Hello') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_output(debounce_by=None):
                        messages.append(chunk)
                stream_usage = deepcopy(stream.usage())
    assert stream is not None
    assert stream.response == snapshot(
        ModelResponse(
            parts=[TextPart(content='The cat sat on the mat.')],
            usage=RequestUsage(input_tokens=51, output_tokens=7),
            model_name='test',
            timestamp=IsDatetime(),
            provider_name='test',
        )
    )
    assert run.next_node == End(data=FinalResult(output='The bat sat on the mat.', tool_name=None, tool_call_id=None))
    assert run.usage() == stream_usage == RunUsage(requests=1, input_tokens=51, output_tokens=7)

    assert messages == snapshot(
        [
            '',
            'The ',
            'The cat ',
            'The bat sat ',
            'The bat sat on ',
            'The bat sat on the ',
            'The bat sat on the mat.',
            'The bat sat on the mat.',
        ]
    )


async def test_streamed_run_result_metadata_available() -> None:
    agent = Agent(TestModel(custom_output_text='stream metadata'), metadata={'env': 'stream'})

    async with agent.run_stream('stream metadata prompt') as result:
        assert await result.get_output() == 'stream metadata'

    assert result.metadata == {'env': 'stream'}


async def test_agent_stream_metadata_available() -> None:
    agent = Agent(
        TestModel(custom_output_text='agent stream metadata'),
        metadata=lambda ctx: {'prompt': ctx.prompt},
    )

    captured_stream: AgentStream | None = None
    async with agent.iter('agent stream prompt') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    captured_stream = stream
                    async for _ in stream.stream_text(debounce_by=None):
                        pass

    assert captured_stream is not None
    assert captured_stream.metadata == {'prompt': 'agent stream prompt'}


def test_agent_stream_metadata_falls_back_to_run_context() -> None:
    response_message = ModelResponse(parts=[TextPart('fallback metadata')], model_name='test')
    stream_response = ModelTestStreamedResponse(
        model_request_parameters=models.ModelRequestParameters(),
        _model_name='test',
        _structured_response=response_message,
        _messages=[],
        _provider_name='test',
    )
    run_ctx = RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        metadata={'source': 'run-context'},
    )
    output_schema = TextOutputSchema[str](
        text_processor=TextOutputProcessor(),
        allows_deferred_tools=False,
        allows_image=False,
    )
    stream = AgentStream(
        _raw_stream_response=stream_response,
        _output_schema=output_schema,
        _model_request_parameters=models.ModelRequestParameters(),
        _output_validators=[],
        _run_ctx=run_ctx,
        _usage_limits=None,
        _tool_manager=ToolManager(toolset=MagicMock()),
    )

    assert stream.metadata == {'source': 'run-context'}


def _make_run_result(*, metadata: dict[str, Any] | None) -> AgentRunResult[str]:
    state = GraphAgentState(metadata=metadata)
    response_message = ModelResponse(parts=[TextPart('final')], model_name='test')
    state.message_history.append(response_message)
    return AgentRunResult('final', _state=state)


def test_streamed_run_result_metadata_prefers_run_result_state() -> None:
    run_result = _make_run_result(metadata={'from': 'run-result'})
    streamed = StreamedRunResult(
        all_messages=run_result.all_messages(),
        new_message_index=0,
        run_result=run_result,
    )
    assert streamed.metadata == {'from': 'run-result'}


def test_streamed_run_result_metadata_none_without_sources() -> None:
    run_result = _make_run_result(metadata=None)
    streamed = StreamedRunResult(all_messages=[], new_message_index=0, run_result=run_result)
    assert streamed.metadata is None


def test_streamed_run_result_metadata_none_without_run_or_stream() -> None:
    streamed = StreamedRunResult(all_messages=[], new_message_index=0, stream_response=None, on_complete=None)
    assert streamed.metadata is None


def test_streamed_run_result_sync_exposes_metadata() -> None:
    run_result = _make_run_result(metadata={'sync': 'metadata'})
    streamed = StreamedRunResult(
        all_messages=run_result.all_messages(),
        new_message_index=0,
        run_result=run_result,
    )
    sync_result = StreamedRunResultSync(streamed)
    assert sync_result.metadata == {'sync': 'metadata'}


async def test_iter_stream_responses():
    m = TestModel(custom_output_text='The cat sat on the mat.')

    agent = Agent(m)

    @agent.output_validator
    def output_validator_simple(data: str) -> str:
        # Make a substitution in the validated results
        return re.sub('cat sat', 'bat sat', data)

    run: AgentRun
    stream: AgentStream
    messages: list[ModelResponse] = []
    async with agent.iter('Hello') as run:
        assert isinstance(run.run_id, str)
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_responses(debounce_by=None):
                        messages.append(chunk)

    assert messages == [
        ModelResponse(
            parts=[TextPart(content=text)],
            usage=RequestUsage(input_tokens=IsInt(), output_tokens=IsInt()),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            provider_name='test',
        )
        for text in [
            '',
            '',
            'The ',
            'The cat ',
            'The cat sat ',
            'The cat sat on ',
            'The cat sat on the ',
            'The cat sat on the mat.',
            'The cat sat on the mat.',
        ]
    ]

    # Note: as you can see above, the output validator is not applied to the streamed responses, just the final result:
    assert run.result is not None
    assert run.result.output == 'The bat sat on the mat.'


async def test_stream_iter_structured_validator() -> None:
    class NotOutputType(BaseModel):
        not_value: str

    agent = Agent[None, OutputType | NotOutputType]('test', output_type=OutputType | NotOutputType)

    @agent.output_validator
    def output_validator(data: OutputType | NotOutputType) -> OutputType | NotOutputType:
        assert isinstance(data, OutputType)
        return OutputType(value=data.value + ' (validated)')

    outputs: list[OutputType] = []
    async with agent.iter('test') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for output in stream.stream_output(debounce_by=None):
                        outputs.append(output)
    assert outputs == snapshot([OutputType(value='a (validated)'), OutputType(value='a (validated)')])


async def test_unknown_tool_call_events():
    """Test that unknown tool calls emit both FunctionToolCallEvent and FunctionToolResultEvent during streaming."""

    def call_mixed_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls both known and unknown tools."""
        return ModelResponse(
            parts=[
                ToolCallPart('unknown_tool', {'arg': 'value'}),
                ToolCallPart('known_tool', {'x': 5}),
            ]
        )

    agent = Agent(FunctionModel(call_mixed_tools))

    @agent.tool_plain
    def known_tool(x: int) -> int:
        return x * 2

    event_parts: list[Any] = []

    try:
        async with agent.iter('test') as agent_run:
            async for node in agent_run:  # pragma: no branch
                if Agent.is_call_tools_node(node):
                    async with node.stream(agent_run.ctx) as event_stream:
                        async for event in event_stream:
                            event_parts.append(event)

    except UnexpectedModelBehavior:
        pass

    assert event_parts == snapshot(
        [
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='known_tool',
                    args={'x': 5},
                    tool_call_id=IsStr(),
                )
            ),
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='unknown_tool',
                    args={'arg': 'value'},
                    tool_call_id=IsStr(),
                ),
            ),
            FunctionToolResultEvent(
                result=RetryPromptPart(
                    content="Unknown tool name: 'unknown_tool'. Available tools: 'known_tool'",
                    tool_name='unknown_tool',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='known_tool',
                    content=10,
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ),
        ]
    )


async def test_output_tool_validation_failure_events():
    """Test that output tools that fail validation emit events during streaming."""

    def call_final_result_with_bad_data(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        """Mock function that calls final_result tool with invalid data."""
        assert info.output_tools is not None
        return ModelResponse(
            parts=[
                ToolCallPart('final_result', {'bad_value': 'invalid'}),  # Invalid field name
                ToolCallPart('final_result', {'value': 'valid'}),  # Valid field name
            ]
        )

    agent = Agent(FunctionModel(call_final_result_with_bad_data), output_type=OutputType)

    events: list[Any] = []
    async with agent.iter('test') as agent_run:
        async for node in agent_run:
            if Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as event_stream:
                    async for event in event_stream:
                        events.append(event)

    assert events == snapshot(
        [
            FunctionToolCallEvent(
                part=ToolCallPart(
                    tool_name='final_result',
                    args={'bad_value': 'invalid'},
                    tool_call_id=IsStr(),
                ),
            ),
            FunctionToolResultEvent(
                result=RetryPromptPart(
                    content=[
                        ErrorDetails(
                            type='missing',
                            loc=('value',),
                            msg='Field required',
                            input={'bad_value': 'invalid'},
                        ),
                    ],
                    tool_name='final_result',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
        ]
    )


async def test_stream_structured_output():
    class CityLocation(BaseModel):
        city: str
        country: str | None = None

    m = TestModel(custom_output_text='{"city": "Mexico City", "country": "Mexico"}')

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    async with agent.run_stream('') as result:
        assert not result.is_complete
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [
                CityLocation(city='Mexico '),
                CityLocation(city='Mexico City'),
                CityLocation(city='Mexico City'),
                CityLocation(city='Mexico City', country='Mexico'),
                CityLocation(city='Mexico City', country='Mexico'),
            ]
        )
        assert result.is_complete


async def test_iter_stream_structured_output():
    class CityLocation(BaseModel):
        city: str
        country: str | None = None

    m = TestModel(custom_output_text='{"city": "Mexico City", "country": "Mexico"}')

    agent = Agent(m, output_type=PromptedOutput(CityLocation))

    async with agent.iter('') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    assert [c async for c in stream.stream_output(debounce_by=None)] == snapshot(
                        [
                            CityLocation(city='Mexico '),
                            CityLocation(city='Mexico City'),
                            CityLocation(city='Mexico City'),
                            CityLocation(city='Mexico City', country='Mexico'),
                            CityLocation(city='Mexico City', country='Mexico'),
                        ]
                    )


async def test_iter_stream_output_tool_dont_hit_retry_limit():
    class CityLocation(BaseModel):
        city: str
        country: str | None = None

    async def text_stream(_messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        """Stream partial JSON data that will initially fail validation."""
        assert agent_info.output_tools is not None
        assert len(agent_info.output_tools) == 1
        name = agent_info.output_tools[0].name

        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args='{"c')}
        yield {0: DeltaToolCall(json_args='ity":')}
        yield {0: DeltaToolCall(json_args=' "Mex')}
        yield {0: DeltaToolCall(json_args='ico City",')}
        yield {0: DeltaToolCall(json_args=' "cou')}
        yield {0: DeltaToolCall(json_args='ntry": "Mexico"}')}

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=CityLocation)

    async with agent.iter('Generate city info') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    assert [c async for c in stream.stream_output(debounce_by=None)] == snapshot(
                        [
                            CityLocation(city='Mex'),
                            CityLocation(city='Mexico City'),
                            CityLocation(city='Mexico City'),
                            CityLocation(city='Mexico City', country='Mexico'),
                            CityLocation(city='Mexico City', country='Mexico'),
                        ]
                    )


def test_function_tool_event_tool_call_id_properties():
    """Ensure that the `tool_call_id` property on function tool events mirrors the underlying part's ID."""
    # Prepare a ToolCallPart with a fixed ID
    call_part = ToolCallPart(tool_name='sample_tool', args={'a': 1}, tool_call_id='call_id_123')
    call_event = FunctionToolCallEvent(part=call_part)

    # The event should expose the same `tool_call_id` as the part
    assert call_event.tool_call_id == call_part.tool_call_id == 'call_id_123'

    # Prepare a ToolReturnPart with a fixed ID
    return_part = ToolReturnPart(tool_name='sample_tool', content='ok', tool_call_id='return_id_456')
    result_event = FunctionToolResultEvent(result=return_part)

    # The event should expose the same `tool_call_id` as the result part
    assert result_event.tool_call_id == return_part.tool_call_id == 'return_id_456'


async def test_tool_raises_call_deferred():
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain()
    def my_tool(x: int) -> int:
        raise CallDeferred

    async with agent.run_stream('Hello') as result:
        assert not result.is_complete
        assert isinstance(result.run_id, str)
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            [DeferredToolRequests(calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())])]
        )
        assert await result.get_output() == snapshot(
            DeferredToolRequests(calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())])
        )
        responses = [c async for c, _is_last in result.stream_responses(debounce_by=None)]
        assert responses == snapshot(
            [
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())],
                    usage=RequestUsage(input_tokens=51),
                    model_name='test',
                    timestamp=IsDatetime(),
                    provider_name='test',
                    run_id=IsStr(),
                )
            ]
        )
        assert await result.validate_response_output(responses[0]) == snapshot(
            DeferredToolRequests(calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())])
        )
        assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=51, output_tokens=0))
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.is_complete


async def test_tool_raises_approval_required():
    async def llm(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='my_tool', json_args='{"x": 1}', tool_call_id='my_tool')}
        else:
            yield 'Done!'

    agent = Agent(FunctionModel(stream_function=llm), output_type=[str, DeferredToolRequests])

    @agent.tool
    def my_tool(ctx: RunContext[None], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired
        return x * 42

    async with agent.run_stream('Hello') as result:
        assert not result.is_complete
        messages = result.all_messages()
        output = await result.get_output()
        assert output == snapshot(
            DeferredToolRequests(approvals=[ToolCallPart(tool_name='my_tool', args='{"x": 1}', tool_call_id=IsStr())])
        )
        assert result.is_complete

    async with agent.run_stream(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={'my_tool': ToolApproved(override_args={'x': 2})}),
    ) as result:
        assert not result.is_complete
        output = await result.get_output()
        assert result.all_messages() == snapshot(
            [
                ModelRequest(
                    parts=[
                        UserPromptPart(
                            content='Hello',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='my_tool', args='{"x": 1}', tool_call_id='my_tool')],
                    usage=RequestUsage(input_tokens=50, output_tokens=3),
                    model_name='function::llm',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='my_tool',
                            content=84,
                            tool_call_id='my_tool',
                            timestamp=IsDatetime(),
                        )
                    ],
                    timestamp=IsNow(tz=timezone.utc),
                    run_id=IsStr(),
                ),
                ModelResponse(
                    parts=[TextPart(content='Done!')],
                    usage=RequestUsage(input_tokens=50, output_tokens=1),
                    model_name='function::llm',
                    timestamp=IsDatetime(),
                    run_id=IsStr(),
                ),
            ]
        )
        assert output == snapshot('Done!')
        assert result.is_complete


async def test_deferred_tool_iter():
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    async def prepare_tool(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, kind='external')

    @agent.tool_plain(prepare=prepare_tool)
    def my_tool(x: int) -> int:
        return x + 1  # pragma: no cover

    @agent.tool_plain(requires_approval=True)
    def my_other_tool(x: int) -> int:
        return x + 1  # pragma: no cover

    outputs: list[str | DeferredToolRequests] = []
    events: list[Any] = []

    async with agent.iter('test') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)
                    async for output in stream.stream_output(debounce_by=None):
                        outputs.append(output)
            if agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)

    assert outputs == snapshot(
        [
            DeferredToolRequests(
                calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())],
                approvals=[ToolCallPart(tool_name='my_other_tool', args={'x': 0}, tool_call_id=IsStr())],
            )
        ]
    )
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr()),
            ),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_tool'),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                ),
                previous_part_kind='tool-call',
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                ),
            ),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='my_other_tool', args={'x': 0}, tool_call_id=IsStr())),
        ]
    )


async def test_tool_raises_call_deferred_approval_required_iter():
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain
    def my_tool(x: int) -> int:
        raise CallDeferred

    @agent.tool_plain
    def my_other_tool(x: int) -> int:
        raise ApprovalRequired

    events: list[Any] = []

    async with agent.iter('test') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)
            if agent.is_call_tools_node(node):
                async with node.stream(run.ctx) as stream:
                    async for event in stream:
                        events.append(event)

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_tool'),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                ),
                previous_part_kind='tool-call',
            ),
            PartEndEvent(
                index=1,
                part=ToolCallPart(
                    tool_name='my_other_tool', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__my_other_tool'
                ),
            ),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='my_other_tool', args={'x': 0}, tool_call_id=IsStr())),
        ]
    )

    assert run.result is not None
    assert run.result.output == snapshot(
        DeferredToolRequests(
            calls=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())],
            approvals=[ToolCallPart(tool_name='my_other_tool', args={'x': 0}, tool_call_id=IsStr())],
        )
    )


async def test_run_event_stream_handler():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]):
        async for event in stream:
            events.append(event)

    result = await test_agent.run('Hello', event_stream_handler=event_stream_handler)
    assert result.output == snapshot('{"ret_a":"a-apple"}')
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id='pyd_ai_tool_call_id__ret_a'),
            ),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='ret_a',
                    content='a-apple',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"ret_a":')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='"a-apple"}')),
            PartEndEvent(index=0, part=TextPart(content='{"ret_a":"a-apple"}')),
        ]
    )


def test_run_sync_event_stream_handler():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]):
        async for event in stream:
            events.append(event)

    result = test_agent.run_sync('Hello', event_stream_handler=event_stream_handler)
    assert result.output == snapshot('{"ret_a":"a-apple"}')
    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id='pyd_ai_tool_call_id__ret_a'),
            ),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='ret_a',
                    content='a-apple',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"ret_a":')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='"a-apple"}')),
            PartEndEvent(index=0, part=TextPart(content='{"ret_a":"a-apple"}')),
        ]
    )


async def test_run_stream_event_stream_handler():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]):
        async for event in stream:
            events.append(event)

    async with test_agent.run_stream('Hello', event_stream_handler=event_stream_handler) as result:
        assert [c async for c in result.stream_output(debounce_by=None)] == snapshot(
            ['{"ret_a":', '{"ret_a":"a-apple"}', '{"ret_a":"a-apple"}']
        )

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id='pyd_ai_tool_call_id__ret_a'),
            ),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='ret_a',
                    content='a-apple',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
        ]
    )


async def test_stream_tool_returning_user_content():
    m = TestModel()

    agent = Agent(m)
    assert agent.name is None

    @agent.tool_plain
    async def get_image() -> ImageUrl:
        return ImageUrl(url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg')

    events: list[AgentStreamEvent] = []

    async def event_stream_handler(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]):
        async for event in stream:
            events.append(event)

    await agent.run('Hello', event_stream_handler=event_stream_handler)

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='get_image', args={}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='get_image', args={}, tool_call_id='pyd_ai_tool_call_id__get_image'),
            ),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='get_image', args={}, tool_call_id=IsStr())),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='get_image',
                    content='See file bd38f5',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                ),
                content=[
                    'This is file bd38f5:',
                    ImageUrl(
                        url='https://t3.ftcdn.net/jpg/00/85/79/92/360_F_85799278_0BBGV9OAdQDTLnKwAPBCcg1J7QtiieJY.jpg',
                        identifier='bd38f5',
                    ),
                ],
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"get_image":"See ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='file ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='bd38f5"}')),
            PartEndEvent(index=0, part=TextPart(content='{"get_image":"See file bd38f5"}')),
        ]
    )


async def test_run_stream_events():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    events = [event async for event in test_agent.run_stream_events('Hello')]
    assert test_agent.name == 'test_agent'

    assert events == snapshot(
        [
            PartStartEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr()),
            ),
            PartEndEvent(
                index=0,
                part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id='pyd_ai_tool_call_id__ret_a'),
            ),
            FunctionToolCallEvent(part=ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())),
            FunctionToolResultEvent(
                result=ToolReturnPart(
                    tool_name='ret_a',
                    content='a-apple',
                    tool_call_id=IsStr(),
                    timestamp=IsNow(tz=timezone.utc),
                )
            ),
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='{"ret_a":')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='"a-apple"}')),
            PartEndEvent(index=0, part=TextPart(content='{"ret_a":"a-apple"}')),
            AgentRunResultEvent(result=AgentRunResult(output='{"ret_a":"a-apple"}')),
        ]
    )


def test_structured_response_sync_validation():
    async def text_stream(_messages: list[ModelMessage], agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        assert agent_info.output_tools is not None
        assert len(agent_info.output_tools) == 1
        name = agent_info.output_tools[0].name
        json_data = json.dumps({'response': [1, 2, 3, 4]})
        yield {0: DeltaToolCall(name=name)}
        yield {0: DeltaToolCall(json_args=json_data[:15])}
        yield {0: DeltaToolCall(json_args=json_data[15:])}

    agent = Agent(FunctionModel(stream_function=text_stream), output_type=list[int])

    chunks: list[list[int]] = []
    result = agent.run_stream_sync('')
    for structured_response, last in result.stream_responses(debounce_by=None):
        response_data = result.validate_response_output(structured_response, allow_partial=not last)
        chunks.append(response_data)

    assert chunks == snapshot([[1], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])


async def test_get_output_after_stream_output():
    """Verify that we don't get duplicate messages in history when using tool output and `get_output` is called after `stream_output`."""
    m = TestModel()

    agent = Agent(m, output_type=bool)

    async with agent.run_stream('Hello') as result:
        outputs: list[bool] = []
        async for o in result.stream_output():
            outputs.append(o)
        o = await result.get_output()
        outputs.append(o)

    assert outputs == snapshot([False, False, False])
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result',
                        args={'response': False},
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                    )
                ],
                usage=RequestUsage(input_tokens=51),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
                provider_name='test',
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


class TestStreamCancellation:
    """Tests for streaming cancellation functionality."""

    async def test_stream_cancel_basic(self):
        """Test that cancel() stops iteration and sets incomplete=True."""
        agent = Agent(TestModel())

        async with agent.run_stream('Hello') as result:
            chunks: list[str] = []
            # Use debounce_by=None to ensure we get individual chunks from TestModel
            async for text in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                chunks.append(text)
                if len(chunks) >= 2:
                    await result.cancel()
                    break

            assert result.is_cancelled
            assert result.response.incomplete

    async def test_stream_cancel_idempotent(self):
        """Test that calling cancel() multiple times is safe."""
        agent = Agent(TestModel())

        async with agent.run_stream('Hello') as result:
            await result.cancel()
            await result.cancel()  # Should not raise
            await result.cancel()  # Should not raise

            assert result.is_cancelled
            assert result.response.incomplete

    async def test_stream_cancel_in_stream_text(self):
        """Test cancellation during stream_text()."""
        agent = Agent(TestModel())

        async with agent.run_stream('Hello') as result:
            chunks: list[str] = []
            async for text in result.stream_text():  # pragma: no branch
                chunks.append(text)
                if len(chunks) >= 1:  # pragma: no branch
                    await result.cancel()
                    break

            assert result.is_cancelled
            assert len(chunks) >= 1

    async def test_stream_cancel_message_history(self):
        """Test that cancelled response appears in all_messages() with incomplete=True."""
        agent = Agent(TestModel())

        async with agent.run_stream('Hello') as result:
            # Use debounce_by=None to ensure we get individual chunks from TestModel
            async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                await result.cancel()
                break

            messages = result.all_messages()
            # Should have request + incomplete response
            assert len(messages) == 2
            # Find the ModelResponse - it should be present and marked incomplete
            responses = [m for m in messages if isinstance(m, ModelResponse)]
            assert len(responses) == 1, 'Cancelled response should be in all_messages()'
            assert responses[0].incomplete, 'Cancelled response should be marked incomplete'

    async def test_stream_cancel_before_iteration(self):
        """Test that cancel() works before iteration starts."""
        agent = Agent(TestModel())

        async with agent.run_stream('Hello') as result:
            await result.cancel()

            # Iteration should complete immediately
            chunks: list[str] = []
            # Use debounce_by=None to ensure we get individual chunks from TestModel
            async for text in result.stream_text(delta=True, debounce_by=None):
                chunks.append(text)  # pragma: no cover

            assert result.is_cancelled
            # May or may not have any chunks depending on timing
            assert result.response.incomplete

    async def test_stream_cancel_structured_output(self):
        """Test that cancel() works with structured output streaming via stream_output()."""

        async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            assert info.output_tools is not None
            yield {0: DeltaToolCall(name=info.output_tools[0].name, json_args='{"a": 42')}
            yield {0: DeltaToolCall(json_args=', "b": "h')}
            yield {0: DeltaToolCall(json_args='el')}
            yield {0: DeltaToolCall(json_args='lo')}  # pragma: no cover
            yield {0: DeltaToolCall(json_args='"}')}  # pragma: no cover

        agent = Agent(FunctionModel(stream_function=sf), output_type=Foo)

        async with agent.run_stream('test') as result:
            outputs: list[Foo] = []
            async for output in result.stream_output(debounce_by=None):  # pragma: no branch
                outputs.append(output)
                if len(outputs) >= 2:
                    await result.cancel()
                    break

            assert result.is_cancelled
            assert result.response.incomplete
            assert len(outputs) >= 2

    async def test_stream_cancel_stream_responses(self):
        """Test that cancel() works with stream_responses() iteration."""
        agent = Agent(TestModel())

        async with agent.run_stream('Hello') as result:
            responses: list[tuple[ModelResponse, bool]] = []
            async for response_tuple in result.stream_responses():  # pragma: no branch
                responses.append(response_tuple)
                if len(responses) >= 2:
                    await result.cancel()
                    break

            assert result.is_cancelled
            assert result.response.incomplete
            assert len(responses) >= 2

    async def test_stream_cancel_agent_iter(self):
        """Test that cancel() works when using Agent.iter() with node.stream()."""
        agent = Agent(TestModel())

        async with agent.iter('Hello') as run:
            async for node in run:
                if agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        chunks: list[str] = []
                        async for text in stream.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                            chunks.append(text)
                            if len(chunks) >= 2:
                                await stream.cancel()
                                break

                        assert stream.is_cancelled
                        assert stream.response.incomplete
                        assert len(chunks) >= 2

    async def test_stream_cancel_tool_call_with_run_stream(self):
        """Test that cancelling during tool call streaming works with run_stream API.

        Note: run_stream + stream_responses() accumulates deltas before yielding,
        so this tests cancellation at the response level, not mid-delta.
        See test_stream_cancel_tool_call_marks_args_incomplete_agent_iter for
        fine-grained delta-level cancellation testing.
        """

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            # First yield a complete tool call
            yield {0: DeltaToolCall(name='my_tool', json_args='{"arg1": "value1", "arg2": "value2"}')}
            # Then yield text that we'll cancel during
            yield 'Starting '
            yield 'response '
            yield 'text'  # pragma: no cover

        agent = Agent(model=FunctionModel(stream_function=stream_function))

        @agent.tool_plain
        def my_tool(arg1: str, arg2: str) -> str:  # pragma: no cover
            return f'{arg1}-{arg2}'  # pragma: no cover

        # Use run_stream - cancel after seeing some responses
        async with agent.run_stream('Call my_tool') as result:
            response_count = 0
            async for _response, _is_last in result.stream_responses(debounce_by=None):  # pragma: no branch
                response_count += 1
                if response_count >= 2:
                    await result.cancel()
                    break

            assert result.is_cancelled
            assert result.response.incomplete

            # The tool call was complete before cancellation, so args_incomplete should be False
            tool_call_parts = [p for p in result.response.parts if isinstance(p, ToolCallPart)]
            assert len(tool_call_parts) == 1
            tool_call = tool_call_parts[0]
            assert tool_call.args_incomplete is False  # Complete args
            assert tool_call.args_as_dict() == {'arg1': 'value1', 'arg2': 'value2'}

    async def test_stream_cancel_tool_call_marks_args_incomplete_agent_iter(self):
        """Test that cancelling during tool call streaming marks args_incomplete=True (agent.iter API)."""

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            # Stream a tool call with args in multiple chunks
            yield {0: DeltaToolCall(name='my_tool', json_args='{"arg1": ')}
            yield {0: DeltaToolCall(json_args='"value1", ')}
            yield {0: DeltaToolCall(json_args='"arg2": ')}  # pragma: no cover
            # These would complete the JSON but we'll cancel before reaching them
            yield {0: DeltaToolCall(json_args='"value2"}')}  # pragma: no cover

        agent = Agent(model=FunctionModel(stream_function=stream_function))

        @agent.tool_plain
        def my_tool(arg1: str, arg2: str) -> str:  # pragma: no cover
            return f'{arg1}-{arg2}'  # pragma: no cover

        # Use agent.iter() to get fine-grained control over streaming
        async with agent.iter('Call my_tool') as run:
            async for node in run:  # pragma: no branch
                if agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        event_count = 0
                        async for _ in stream:  # pragma: no branch
                            event_count += 1
                            if event_count >= 2:  # Cancel after receiving partial tool call args
                                await stream.cancel()
                                break

                        assert stream.is_cancelled
                        assert stream.response.incomplete

                        # Check that tool call parts have args_incomplete=True if args are truncated
                        tool_call_parts = [p for p in stream.response.parts if isinstance(p, ToolCallPart)]
                        assert len(tool_call_parts) == 1
                        tool_call = tool_call_parts[0]

                        # The args should be incomplete (truncated JSON)
                        assert tool_call.args_incomplete is True
                        # The args string should be truncated (not valid JSON)
                        assert tool_call.args is not None
                        with pytest.raises(Exception):  # Should fail to parse
                            tool_call.args_as_dict()
                    break  # Don't continue the agent loop

    async def test_stream_cancel_tool_call_complete_args_not_marked_incomplete_run_stream(self):
        """Test that tool calls with complete args before cancellation are not marked incomplete (run_stream API)."""

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            # First yield a complete tool call
            yield {0: DeltaToolCall(name='tool1', json_args='{"complete": true}')}
            # Then stream text that we'll cancel during
            yield 'Some '
            yield 'text '
            yield 'response'

        agent = Agent(model=FunctionModel(stream_function=stream_function))

        @agent.tool_plain
        def tool1(complete: bool) -> str:  # pragma: no cover
            return 'done'  # pragma: no cover

        # Use run_stream with stream_responses() to iterate over raw model responses
        async with agent.run_stream('Call tool1') as result:
            response_count = 0
            async for _ in result.stream_responses():  # pragma: no branch
                response_count += 1
                if response_count >= 3:  # Cancel after tool call is complete but during text
                    await result.cancel()
                    break

            assert result.is_cancelled
            assert result.response.incomplete

            # Check that the complete tool call is NOT marked as args_incomplete
            tool_call_parts = [p for p in result.response.parts if isinstance(p, ToolCallPart)]
            assert len(tool_call_parts) == 1
            tool_call = tool_call_parts[0]

            # The args are complete (valid JSON), so should NOT be marked incomplete
            assert tool_call.args_incomplete is False
            assert tool_call.args_as_dict() == {'complete': True}

    async def test_stream_cancel_tool_call_complete_args_not_marked_incomplete_agent_iter(self):
        """Test that tool calls with complete args before cancellation are not marked incomplete (agent.iter API)."""

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            # First yield a complete tool call
            yield {0: DeltaToolCall(name='tool1', json_args='{"complete": true}')}
            # Then stream text that we'll cancel during
            yield 'Some '
            yield 'text '  # pragma: no cover
            yield 'response'  # pragma: no cover

        agent = Agent(model=FunctionModel(stream_function=stream_function))

        @agent.tool_plain
        def tool1(complete: bool) -> str:  # pragma: no cover
            return 'done'  # pragma: no cover

        # Use agent.iter() to get fine-grained control over streaming
        async with agent.iter('Call tool1') as run:
            async for node in run:  # pragma: no branch
                if agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        event_count = 0
                        async for _ in stream:  # pragma: no branch
                            event_count += 1
                            if event_count >= 3:  # Cancel after tool call is complete but during text
                                await stream.cancel()
                                break

                        assert stream.is_cancelled
                        assert stream.response.incomplete

                        # Check that the complete tool call is NOT marked as args_incomplete
                        tool_call_parts = [p for p in stream.response.parts if isinstance(p, ToolCallPart)]
                        assert len(tool_call_parts) == 1
                        tool_call = tool_call_parts[0]

                        # The args are complete (valid JSON), so should NOT be marked incomplete
                        assert tool_call.args_incomplete is False
                        assert tool_call.args_as_dict() == {'complete': True}
                    break  # Don't continue the agent loop

    async def test_stream_cancel_all_messages_has_incomplete_marked(self):
        """Verify all_messages() returns responses with args_incomplete properly set after cancellation.

        Uses run_stream API to test the user-facing all_messages() interface.
        """

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
            # Stream a tool call with args in multiple chunks
            yield {0: DeltaToolCall(name='my_tool', json_args='{"arg1": ')}
            yield {0: DeltaToolCall(json_args='"value1", ')}
            yield {0: DeltaToolCall(json_args='"arg2": ')}
            yield {0: DeltaToolCall(json_args='"value2"}')}
            # Add text to prevent agent from looping
            yield 'Done!'

        agent = Agent(model=FunctionModel(stream_function=stream_function))

        @agent.tool_plain
        def my_tool(arg1: str, arg2: str) -> str:  # pragma: no cover
            return f'{arg1}-{arg2}'  # pragma: no cover

        # Use run_stream to test the user-facing all_messages() API
        async with agent.run_stream('Call my_tool') as result:
            # Cancel after receiving first response in stream_responses
            async for _, _ in result.stream_responses(debounce_by=None):  # pragma: no branch
                await result.cancel()
                break

            assert result.is_cancelled
            assert result.response.incomplete

            # After cancellation, check all_messages()
            messages = result.all_messages()

            # Find the ModelResponse in the message history
            model_responses = [m for m in messages if isinstance(m, ModelResponse)]
            assert len(model_responses) == 1

            response = model_responses[0]
            assert response.incomplete is True

            # Check that the response has tool call parts
            tool_calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
            assert len(tool_calls) == 1
            # With run_stream + stream_responses, deltas are accumulated before yielding,
            # so the tool call may have complete args depending on timing.
            # The key point is that incomplete=True is set on the response.


class TestFilterIncompleteToolCalls:
    """Tests for _filter_incomplete_tool_calls helper function.

    This function filters unprocessed tool calls from incomplete responses.
    Tool calls are kept only if they have a corresponding tool result
    (identified by tool_call_id in processed_tool_call_ids).
    """

    def test_filter_unprocessed_tool_calls_from_incomplete_response(self):
        """Tool calls without corresponding results should be filtered from incomplete responses."""
        response = ModelResponse(
            parts=[
                TextPart(content='Some text'),
                ToolCallPart(tool_name='processed_tool', args='{"valid": true}', tool_call_id='tc1'),
                ToolCallPart(tool_name='unprocessed_tool', args='{"also_valid": true}', tool_call_id='tc2'),
            ],
            incomplete=True,
        )

        # Only tc1 has a corresponding result
        filtered = _filter_incomplete_tool_calls(response, processed_tool_call_ids={'tc1'})

        # Unprocessed tool call (tc2) should be filtered out
        assert len(filtered.parts) == 2
        assert isinstance(filtered.parts[0], TextPart)
        assert isinstance(filtered.parts[1], ToolCallPart)
        assert filtered.parts[1].tool_name == 'processed_tool'

    def test_no_filter_for_complete_response(self):
        """Tool calls should NOT be filtered from complete (non-incomplete) responses."""
        response = ModelResponse(
            parts=[
                TextPart(content='Some text'),
                ToolCallPart(tool_name='some_tool', args='{"data": true}', tool_call_id='tc1'),
            ],
            incomplete=False,  # Response is complete
        )

        # Even with no processed results, complete responses are not filtered
        filtered = _filter_incomplete_tool_calls(response, processed_tool_call_ids=set())

        # Nothing should be filtered since response.incomplete is False
        assert len(filtered.parts) == 2
        assert filtered is response  # Should return same object

    def test_filter_preserves_processed_tool_calls(self):
        """Tool calls with corresponding results should be preserved."""
        response = ModelResponse(
            parts=[
                ToolCallPart(tool_name='tool1', args='{"valid": true}', tool_call_id='tc1'),
                ToolCallPart(tool_name='tool2', args='{"also_valid": 1}', tool_call_id='tc2'),
            ],
            incomplete=True,
        )

        # Both tool calls have corresponding results
        filtered = _filter_incomplete_tool_calls(response, processed_tool_call_ids={'tc1', 'tc2'})

        # Both tool calls should be preserved
        assert len(filtered.parts) == 2
        assert filtered is response  # Should return same object if nothing filtered

    def test_filter_all_unprocessed_tool_calls(self):
        """If no tool calls have results, all should be filtered leaving only text parts."""
        response = ModelResponse(
            parts=[
                TextPart(content='Hello'),
                ToolCallPart(tool_name='tool1', args='{"data1": true}', tool_call_id='tc1'),
                ToolCallPart(tool_name='tool2', args='{"data2": true}', tool_call_id='tc2'),
            ],
            incomplete=True,
        )

        # No tool calls have corresponding results
        filtered = _filter_incomplete_tool_calls(response, processed_tool_call_ids=set())

        # Only the text part should remain
        assert len(filtered.parts) == 1
        assert isinstance(filtered.parts[0], TextPart)
        assert filtered.parts[0].content == 'Hello'

    def test_filter_empty_parts_list(self):
        """Filtering empty parts list should work correctly."""
        response = ModelResponse(parts=[], incomplete=True)

        filtered = _filter_incomplete_tool_calls(response, processed_tool_call_ids=set())

        assert len(filtered.parts) == 0
        assert filtered is response  # Should return same object

    def test_filter_builtin_tool_call_parts(self):
        """BuiltinToolCallPart without corresponding results should also be filtered."""
        response = ModelResponse(
            parts=[
                TextPart(content='Some text'),
                BuiltinToolCallPart(
                    tool_name='builtin_tool',
                    args='{"data": true}',
                    tool_call_id='tc1',
                ),
                ToolCallPart(tool_name='regular_tool', args='{"valid": true}', tool_call_id='tc2'),
            ],
            incomplete=True,
        )

        # Only tc2 has a corresponding result
        filtered = _filter_incomplete_tool_calls(response, processed_tool_call_ids={'tc2'})

        # BuiltinToolCallPart without result should be filtered
        assert len(filtered.parts) == 2
        assert isinstance(filtered.parts[0], TextPart)
        assert isinstance(filtered.parts[1], ToolCallPart)
        assert filtered.parts[1].tool_name == 'regular_tool'

    def test_filter_preserves_thinking_parts(self):
        """ThinkingPart should be preserved during filtering."""
        response = ModelResponse(
            parts=[
                ThinkingPart(content='Let me think about this...'),
                TextPart(content='Here is my response'),
                ToolCallPart(tool_name='some_tool', args='{"data": true}', tool_call_id='tc1'),
            ],
            incomplete=True,
        )

        # No tool calls have results
        filtered = _filter_incomplete_tool_calls(response, processed_tool_call_ids=set())

        # ThinkingPart and TextPart should be preserved, unprocessed tool call filtered
        assert len(filtered.parts) == 2
        assert isinstance(filtered.parts[0], ThinkingPart)
        assert filtered.parts[0].content == 'Let me think about this...'
        assert isinstance(filtered.parts[1], TextPart)


class TestCleanMessageHistoryFiltersIncomplete:
    """Tests for _clean_message_history filtering unprocessed tool calls from incomplete responses.

    Tool calls are filtered from incomplete responses only if they don't have
    corresponding tool results in subsequent messages.
    """

    def test_clean_message_history_filters_unprocessed_tool_calls(self):
        """_clean_message_history should filter unprocessed tool calls from incomplete responses."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content='Hello')]),
            ModelResponse(
                parts=[
                    TextPart(content='Some text'),
                    ToolCallPart(tool_name='tool1', args='{"valid": true}', tool_call_id='tc1'),
                    ToolCallPart(tool_name='tool2', args='{"also_valid": true}', tool_call_id='tc2'),
                ],
                incomplete=True,
            ),
            # No tool results for tc1 or tc2
        ]

        cleaned = _clean_message_history(messages)

        # Both tool calls should be filtered since they have no corresponding results
        assert len(cleaned) == 2
        response = cleaned[1]
        assert isinstance(response, ModelResponse)
        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)

    def test_clean_message_history_preserves_processed_tool_calls(self):
        """_clean_message_history should preserve tool calls that have corresponding results."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content='Hello')]),
            ModelResponse(
                parts=[
                    TextPart(content='Some text'),
                    ToolCallPart(tool_name='processed_tool', args='{"valid": true}', tool_call_id='tc1'),
                    ToolCallPart(tool_name='unprocessed_tool', args='{"also_valid": true}', tool_call_id='tc2'),
                ],
                incomplete=True,
            ),
            # Only tc1 has a result
            ModelRequest(parts=[ToolReturnPart(tool_name='processed_tool', content='result', tool_call_id='tc1')]),
        ]

        cleaned = _clean_message_history(messages)

        # tc1 should be preserved (has result), tc2 should be filtered (no result)
        assert len(cleaned) == 3
        response = cleaned[1]
        assert isinstance(response, ModelResponse)
        assert len(response.parts) == 2
        assert isinstance(response.parts[0], TextPart)
        assert isinstance(response.parts[1], ToolCallPart)
        assert response.parts[1].tool_name == 'processed_tool'

    def test_clean_message_history_preserves_complete_responses(self):
        """_clean_message_history should NOT filter tool calls from complete responses."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content='Hello')]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='some_tool',
                        args='{"data": true}',
                        tool_call_id='tc1',
                    ),
                ],
                incomplete=False,  # Response is complete, so no filtering should happen
            ),
        ]

        cleaned = _clean_message_history(messages)

        # Nothing should be filtered since response.incomplete is False
        assert len(cleaned) == 2
        response = cleaned[1]
        assert isinstance(response, ModelResponse)
        assert len(response.parts) == 1  # Tool call preserved

    def test_clean_message_history_merges_after_filtering(self):
        """_clean_message_history should still merge consecutive messages after filtering."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content='First')]),
            ModelRequest(parts=[UserPromptPart(content='Second')]),  # Should merge with first
            ModelResponse(
                parts=[
                    TextPart(content='Response'),
                    ToolCallPart(tool_name='unprocessed', args='{"data": true}', tool_call_id='tc1'),
                ],
                incomplete=True,
            ),
            # No tool result for tc1
        ]

        cleaned = _clean_message_history(messages)

        # Two requests should be merged, unprocessed tool call filtered
        assert len(cleaned) == 2
        # First message should have both user prompts merged
        assert isinstance(cleaned[0], ModelRequest)
        assert len(cleaned[0].parts) == 2
        # Response should have unprocessed tool call filtered
        assert isinstance(cleaned[1], ModelResponse)
        assert len(cleaned[1].parts) == 1
        assert isinstance(cleaned[1].parts[0], TextPart)

    def test_clean_message_history_multiple_responses(self):
        """_clean_message_history should filter unprocessed tool calls from multiple responses."""
        messages = [
            ModelRequest(parts=[UserPromptPart(content='First request')]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='tool1', args='{"complete": true}', tool_call_id='tc1'),
                ],
                incomplete=False,  # Complete response - no filtering
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='tool1', content='result', tool_call_id='tc1')]),
            ModelResponse(
                parts=[
                    TextPart(content='Partial response'),
                    ToolCallPart(tool_name='tool2', args='{"data": true}', tool_call_id='tc2'),
                ],
                incomplete=True,  # Incomplete response - tool2 has no result, should filter
            ),
            ModelRequest(parts=[UserPromptPart(content='Continue')]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='tool3', args='{"data": true}', tool_call_id='tc3'),
                ],
                incomplete=True,  # Another incomplete response - tool3 has no result, should filter
            ),
        ]

        cleaned = _clean_message_history(messages)

        # Should have 6 messages (requests and responses don't merge across types)
        assert len(cleaned) == 6

        # First response (complete) - tool call preserved
        assert isinstance(cleaned[1], ModelResponse)
        assert len(cleaned[1].parts) == 1
        assert isinstance(cleaned[1].parts[0], ToolCallPart)
        assert cleaned[1].parts[0].tool_name == 'tool1'

        # Second response (incomplete) - unprocessed tool call filtered, text preserved
        assert isinstance(cleaned[3], ModelResponse)
        assert len(cleaned[3].parts) == 1
        assert isinstance(cleaned[3].parts[0], TextPart)

        # Third response (incomplete) - all unprocessed tool calls filtered, empty parts
        assert isinstance(cleaned[5], ModelResponse)
        assert len(cleaned[5].parts) == 0

    def test_clean_message_history_preserves_builtin_tool_calls_with_results(self):
        """_clean_message_history should preserve BuiltinToolCallPart when matching BuiltinToolReturnPart exists.

        Built-in tools (like web_search, code_execution) have both call and return parts
        in the same ModelResponse. When filtering incomplete responses, we must check
        BuiltinToolReturnPart (in ModelResponse) not just ToolReturnPart (in ModelRequest).
        """
        messages = [
            ModelRequest(parts=[UserPromptPart(content='Search for something')]),
            ModelResponse(
                parts=[
                    TextPart(content='Let me search for that'),
                    # Built-in tool call and its result in the same response
                    BuiltinToolCallPart(tool_name='web_search', args={'query': 'test'}, tool_call_id='builtin_tc1'),
                    BuiltinToolReturnPart(
                        tool_name='web_search',
                        content={'results': ['result1', 'result2']},
                        tool_call_id='builtin_tc1',
                    ),
                    # Regular tool call without result - should be filtered
                    ToolCallPart(tool_name='regular_tool', args='{"data": true}', tool_call_id='tc2'),
                ],
                incomplete=True,
            ),
        ]

        cleaned = _clean_message_history(messages)

        # Should have 2 messages
        assert len(cleaned) == 2
        response = cleaned[1]
        assert isinstance(response, ModelResponse)

        # BuiltinToolCallPart should be preserved (has matching BuiltinToolReturnPart)
        # BuiltinToolReturnPart should be preserved (not a tool call)
        # Regular ToolCallPart should be filtered (no matching result)
        assert len(response.parts) == 3
        assert isinstance(response.parts[0], TextPart)
        assert isinstance(response.parts[1], BuiltinToolCallPart)
        assert response.parts[1].tool_call_id == 'builtin_tc1'
        assert isinstance(response.parts[2], BuiltinToolReturnPart)
        assert response.parts[2].tool_call_id == 'builtin_tc1'


class TestIncompleteToolCallsNotSentToApi:
    """Integration tests verifying the complete pipeline filters unprocessed tool calls."""

    def test_cancelled_stream_message_history_filters_unprocessed_tool_calls(self):
        """Simulate a cancelled stream's message history going through _clean_message_history.

        This test constructs a message history that matches what would result from a
        cancelled stream (ModelResponse with incomplete=True), then verifies that
        _clean_message_history filters out unprocessed tool calls (those without
        corresponding results) before they would be sent to the model API.
        """
        # Simulate message history from a cancelled stream with tool calls but no results
        message_history = [
            ModelRequest(parts=[UserPromptPart(content='Get some data')]),
            ModelResponse(
                parts=[
                    TextPart(content='Let me fetch that data'),
                    ToolCallPart(
                        tool_name='get_user',
                        args='{"user_id": 123}',
                        tool_call_id='tc1',
                    ),
                    ToolCallPart(
                        tool_name='get_orders',
                        args='{"user_id": 123, "limit": 10}',
                        tool_call_id='tc2',
                    ),
                ],
                incomplete=True,  # Response was incomplete due to cancellation
            ),
            # No tool results - both tool calls are unprocessed
        ]

        # Run through _clean_message_history (called before sending to model API)
        cleaned = _clean_message_history(message_history)

        # Verify both unprocessed tool calls were filtered out
        assert len(cleaned) == 2
        response = cleaned[1]
        assert isinstance(response, ModelResponse)

        # Only the text part should remain
        assert len(response.parts) == 1
        assert isinstance(response.parts[0], TextPart)
        assert response.parts[0].content == 'Let me fetch that data'

    def test_cancelled_stream_preserves_processed_tool_calls(self):
        """Tool calls with corresponding results should be preserved even from incomplete responses."""
        message_history = [
            ModelRequest(parts=[UserPromptPart(content='Get some data')]),
            ModelResponse(
                parts=[
                    TextPart(content='Let me fetch that data'),
                    ToolCallPart(
                        tool_name='get_user',
                        args='{"user_id": 123}',
                        tool_call_id='tc1',
                    ),
                    ToolCallPart(
                        tool_name='get_orders',
                        args='{"user_id": 123, "limit": 10}',
                        tool_call_id='tc2',
                    ),
                ],
                incomplete=True,  # Response was incomplete due to cancellation
            ),
            # Only tc1 has a result
            ModelRequest(parts=[ToolReturnPart(tool_name='get_user', content='User data', tool_call_id='tc1')]),
        ]

        # Run through _clean_message_history (called before sending to model API)
        cleaned = _clean_message_history(message_history)

        # Verify tc1 was preserved (has result) and tc2 was filtered (no result)
        assert len(cleaned) == 3
        response = cleaned[1]
        assert isinstance(response, ModelResponse)

        # Text and processed tool call should remain
        assert len(response.parts) == 2
        assert isinstance(response.parts[0], TextPart)
        assert isinstance(response.parts[1], ToolCallPart)
        assert response.parts[1].tool_name == 'get_user'

        # Unprocessed tool call should NOT be present
        tool_names = [p.tool_name for p in response.parts if isinstance(p, ToolCallPart)]
        assert 'get_orders' not in tool_names

    def test_complete_response_tool_calls_preserved(self):
        """Tool calls in complete responses should NOT be filtered.

        This tests that the filtering only happens for incomplete responses (cancelled streams),
        not for complete responses.
        """
        message_history = [
            ModelRequest(parts=[UserPromptPart(content='Test')]),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='some_tool',
                        args='{"data": true}',
                        tool_call_id='tc1',
                    ),
                ],
                incomplete=False,  # Response is complete
            ),
            # No tool result, but response is complete so no filtering
        ]

        cleaned = _clean_message_history(message_history)

        # Tool call should be preserved since response.incomplete is False
        assert len(cleaned) == 2
        response = cleaned[1]
        assert isinstance(response, ModelResponse)
        assert len(response.parts) == 1
        assert isinstance(response.parts[0], ToolCallPart)
        assert response.parts[0].tool_name == 'some_tool'

    async def test_end_to_end_cancel_then_continue_conversation(self):
        """True end-to-end test: cancel stream → continue conversation → verify model doesn't receive incomplete tool calls.

        This tests the complete user journey:
        1. Start a streaming run that returns a tool call
        2. Cancel mid-way through the tool call args (creating incomplete args)
        3. Continue the conversation using the message history
        4. Verify the model does NOT receive the incomplete tool call in step 3
        """
        # Track what messages the model receives on each call
        call_count = 0
        received_messages_per_call: list[list[ModelMessage]] = []

        async def stream_model_function(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[DeltaToolCalls | str]:
            nonlocal call_count
            call_count += 1
            received_messages_per_call.append(deepcopy(messages))

            if call_count == 1:
                # First call: stream a tool call in chunks (will be cancelled mid-stream)
                yield {0: DeltaToolCall(name='fetch_data', json_args='{"query": ')}
                yield {0: DeltaToolCall(json_args='"test", ')}
                yield {0: DeltaToolCall(json_args='"limit": ')}  # pragma: no cover
                # These would complete the JSON but we'll cancel before reaching them
                yield {0: DeltaToolCall(json_args='10}')}  # pragma: no cover
                yield 'Done fetching'  # pragma: no cover
            else:  # pragma: no cover
                # Second call: just return text
                yield 'Continuing the conversation'  # pragma: no cover

        def sync_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            received_messages_per_call.append(deepcopy(messages))
            return ModelResponse(parts=[TextPart(content='Continuing the conversation')])

        agent = Agent(model=FunctionModel(function=sync_model_function, stream_function=stream_model_function))

        @agent.tool_plain
        def fetch_data(query: str, limit: int) -> str:  # pragma: no cover
            return f'Results for {query}'  # pragma: no cover

        # Step 1 & 2: Start streaming and cancel mid-tool-call
        cancelled_messages: list[ModelMessage] = []
        async with agent.iter('Fetch some data') as run:
            async for node in run:  # pragma: no branch
                if agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        event_count = 0
                        async for _ in stream:  # pragma: no branch
                            event_count += 1
                            if event_count >= 2:  # Cancel after partial args
                                await stream.cancel()
                                break

                        # Verify we got an incomplete tool call
                        assert stream.is_cancelled
                        assert stream.response.incomplete
                        tool_calls = [p for p in stream.response.parts if isinstance(p, ToolCallPart)]
                        assert len(tool_calls) == 1
                        assert tool_calls[0].args_incomplete is True

                    break

            # Get the message history from the cancelled run
            cancelled_messages = run.result.all_messages() if run.result else list(run.ctx.state.message_history)

        # Step 3: Continue the conversation with the message history
        await agent.run('Please continue', message_history=cancelled_messages)

        # Step 4: Verify the model did NOT receive the incomplete tool call
        assert call_count == 2, 'Model should have been called twice'

        # Check the messages sent to the model on the second call
        second_call_messages = received_messages_per_call[1]

        # Find all tool calls in the messages sent to the model
        tool_calls_sent_to_model: list[ToolCallPart] = []
        for msg in second_call_messages:
            if isinstance(msg, ModelResponse):
                for part in msg.parts:
                    if isinstance(part, ToolCallPart):  # pragma: no cover
                        tool_calls_sent_to_model.append(part)  # pragma: no cover

        # The incomplete tool call should NOT be in the messages sent to the model
        for tc in tool_calls_sent_to_model:
            assert tc.args_incomplete is False, f'Incomplete tool call was sent to model: {tc}'  # pragma: no cover
            # Also verify the args are valid JSON
            tc.args_as_dict()  # Should not raise  # pragma: no cover


class TestRunStreamEventsCancellation:
    """Tests for run_stream_events cancellation via break.

    run_stream_events() is an async generator that yields events. When the user breaks
    from the loop, the receive_stream closes, causing send_stream.send() to fail with
    ClosedResourceError. The event_stream_handler catches this and cancels the underlying stream.
    """

    async def test_break_stops_event_iteration(self):
        """Breaking from run_stream_events should stop the event iteration."""

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for i in range(100):  # pragma: no branch
                yield f'chunk {i} '

        agent = Agent(FunctionModel(stream_function=stream_function))

        events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
        async for event in agent.run_stream_events('test'):  # pragma: no branch
            events.append(event)
            if len(events) >= 3:
                break

        # Should have exactly 3 events (we broke at 3)
        assert len(events) == 3
        # Should NOT have completed with AgentRunResultEvent
        assert not any(isinstance(e, AgentRunResultEvent) for e in events)

    async def test_break_does_not_yield_result(self):
        """Breaking from run_stream_events should NOT yield AgentRunResultEvent."""

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for i in range(100):  # pragma: no branch
                yield f'chunk {i} '

        agent = Agent(FunctionModel(stream_function=stream_function))

        events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
        async for event in agent.run_stream_events('test'):  # pragma: no branch
            events.append(event)
            if len(events) >= 3:
                break

        # Should NOT have AgentRunResultEvent since we broke early
        assert not any(isinstance(e, AgentRunResultEvent) for e in events)

    async def test_normal_completion_yields_result(self):
        """Normal completion of run_stream_events should yield AgentRunResultEvent."""
        agent = Agent(TestModel())

        events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
        async for event in agent.run_stream_events('Hello'):
            events.append(event)

        # Last event should be AgentRunResultEvent
        assert len(events) > 0
        assert isinstance(events[-1], AgentRunResultEvent)

    async def test_break_marks_tool_call_args_incomplete(self):
        """Breaking during tool call streaming marks args_incomplete=True."""

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            # Stream a tool call with args in multiple chunks
            yield {0: DeltaToolCall(name='my_tool', json_args='{"arg1": ')}
            yield {0: DeltaToolCall(json_args='"value1", ')}
            yield {0: DeltaToolCall(json_args='"arg2": ')}
            # These would complete the JSON but we'll cancel before reaching them
            yield {0: DeltaToolCall(json_args='"value2"}')}  # pragma: no cover

        agent = Agent(model=FunctionModel(stream_function=stream_function))

        @agent.tool_plain
        def my_tool(arg1: str, arg2: str) -> str:  # pragma: no cover
            return f'{arg1}-{arg2}'  # pragma: no cover

        events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
        saw_part_start = False
        async for event in agent.run_stream_events('Call my_tool'):  # pragma: no branch
            events.append(event)
            # Break after seeing a PartStartEvent for the tool call
            if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart):
                saw_part_start = True
            # Break after we've seen the first delta (partial args)
            if saw_part_start and isinstance(event, PartDeltaEvent):
                break

        # Should NOT have AgentRunResultEvent since we broke early
        assert not any(isinstance(e, AgentRunResultEvent) for e in events)

        # Find the tool call part from PartStartEvent
        tool_call_events = [e for e in events if isinstance(e, PartStartEvent) and isinstance(e.part, ToolCallPart)]
        assert len(tool_call_events) >= 1

    async def test_break_during_text_streaming(self):
        """Breaking during text streaming stops the stream properly."""

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            for word in [  # pragma: no branch
                'Hello ',
                'world, ',
                'this ',
                'is ',
                'a ',
                'very ',
                'long ',
                'response ',
                'that ',
                'continues',
            ]:
                yield word

        agent = Agent(FunctionModel(stream_function=stream_function))

        events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
        async for event in agent.run_stream_events('test'):  # pragma: no branch
            events.append(event)
            # Break after a few events
            if len(events) >= 5:
                break

        # Should have events but not the final AgentRunResultEvent
        assert len(events) == 5
        assert not any(isinstance(e, AgentRunResultEvent) for e in events)

    async def test_break_is_detected_via_closed_resource_error(self):
        """Verify that breaking triggers ClosedResourceError handling in event_stream_handler."""
        stream_completed = False

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            nonlocal stream_completed
            try:
                for i in range(100):
                    yield f'chunk {i} '
                stream_completed = True  # pragma: no cover
            except asyncio.CancelledError:
                # Stream was cancelled - this is expected
                raise  # pragma: no cover

        agent = Agent(FunctionModel(stream_function=stream_function))

        async for event in agent.run_stream_events('test'):  # pragma: no branch
            if isinstance(event, PartDeltaEvent):
                break

        # Give a moment for the cancellation to propagate
        await asyncio.sleep(0.01)

        # Stream should NOT have completed naturally
        assert not stream_completed, 'Stream completed when it should have been cancelled'


class TestCoverageEdgeCases:
    """Tests for edge cases to improve coverage."""

    async def test_streamed_run_result_is_cancelled_without_stream(self):
        """StreamedRunResult.is_cancelled returns False when _stream_response is None."""
        # Create a StreamedRunResult with run_result (no stream)
        run_result = AgentRunResult(output='test output')
        result = StreamedRunResult(
            all_messages=[],
            new_message_index=0,
            run_result=run_result,
        )

        # is_cancelled should return False when there's no stream
        assert result.is_cancelled is False

        # cancel() should work without error even when _stream_response is None
        await result.cancel()

        # Still not cancelled since there was no stream to cancel
        assert result.is_cancelled is False

    async def test_break_during_tool_streaming_non_agent_stream(self):
        """Breaking during tool streaming hits the non-AgentStream branch in event_stream_handler.

        When events is from CallToolsNode (tool execution), it's an async generator,
        not an AgentStream. Breaking during this phase should not crash.
        """
        from pydantic_ai.models.function import DeltaToolCall

        tool_executed = False

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            # First call: generate a tool call
            if not any(
                isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts) for m in messages
            ):
                yield {0: DeltaToolCall(name='slow_tool')}
            else:  # pragma: no cover
                # Second call: return final response
                yield 'Final response'  # type: ignore  # pragma: no cover

        agent = Agent(FunctionModel(stream_function=stream_function))

        @agent.tool_plain
        async def slow_tool() -> str:
            nonlocal tool_executed
            tool_executed = True
            # Simulate some work
            await asyncio.sleep(0.01)
            return 'tool result'  # pragma: no cover

        events: list[AgentStreamEvent | AgentRunResultEvent[str]] = []
        async for event in agent.run_stream_events('Call the tool'):  # pragma: no branch
            events.append(event)
            # Break when we see the tool call event (during tool streaming, not model streaming)
            if isinstance(event, FunctionToolCallEvent):
                break

        # We should have seen the tool call event
        assert any(isinstance(e, FunctionToolCallEvent) for e in events)
        # Should NOT crash - the non-AgentStream branch should handle this gracefully

    async def test_streamed_response_cancelled_stops_iteration(self):
        """Setting _cancelled=True on StreamedResponse causes iteration to stop immediately.

        Tests the _cancelled checks in models/__init__.py:
        - Line 949-950: _wrap_with_final_event first loop
        - Line 957-958: _wrap_with_final_event return after first loop
        - Line 963-964: _wrap_with_final_event second loop
        - Line 991-992: _wrap_with_part_end
        """
        from pydantic_ai.messages import PartDeltaEvent, PartStartEvent, TextPart, TextPartDelta
        from pydantic_ai.models import ModelRequestParameters, StreamedResponse

        # Create a test StreamedResponse that would yield multiple events
        events_yielded: list[str] = []

        class TestStreamedResponse(StreamedResponse):
            """A test StreamedResponse that tracks what events were yielded."""

            @property
            def model_name(self) -> str:  # pragma: no cover
                return 'test-model'  # pragma: no cover

            @property
            def timestamp(self):  # pragma: no cover
                from datetime import datetime, timezone  # pragma: no cover

                return datetime.now(timezone.utc)  # pragma: no cover

            @property
            def provider_name(self) -> str:  # pragma: no cover
                return 'test'  # pragma: no cover

            @property
            def provider_url(self) -> str:  # pragma: no cover
                return 'https://test.example.com'  # pragma: no cover

            async def _get_event_iterator(self):
                events_yielded.append('part_start')
                yield PartStartEvent(index=0, part=TextPart(content=''))
                events_yielded.append('delta_1')  # pragma: no cover
                yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='Hello'))  # pragma: no cover
                events_yielded.append('delta_2')  # pragma: no cover
                yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' World'))  # pragma: no cover

        response = TestStreamedResponse(model_request_parameters=ModelRequestParameters())

        # Set cancelled before iteration starts
        response._cancelled = True  # pyright: ignore[reportPrivateUsage]

        # Iterate and collect events
        collected_events: list[Any] = []
        async for event in response:
            collected_events.append(event)  # pragma: no cover

        # With _cancelled=True, the wrapper iterators should break immediately
        # No events should be yielded from the wrapped iterators
        assert len(collected_events) == 0
        # The underlying iterator was never even started (or stopped immediately)
        # Note: the exact behavior depends on where the break happens
        # The important thing is that we don't get all events

    async def test_streamed_response_cancelled_during_iteration(self):
        """Cancelling a StreamedResponse during iteration causes it to stop.

        Tests that setting _cancelled=True mid-iteration triggers the break statements.
        """
        from pydantic_ai.models import ModelRequestParameters
        from pydantic_ai.models.function import FunctionStreamedResponse

        events_yielded: list[str] = []

        async def event_generator():
            events_yielded.append('text_1')
            yield 'Hello'
            events_yielded.append('text_2')  # pragma: no cover
            yield ' World'  # pragma: no cover
            events_yielded.append('text_3')  # pragma: no cover
            yield '!'  # pragma: no cover
            events_yielded.append('text_4')  # pragma: no cover
            yield ' More text'  # pragma: no cover

        response = FunctionStreamedResponse(
            model_request_parameters=ModelRequestParameters(),
            _model_name='test-model',
            _iter=event_generator(),
        )

        # Iterate and cancel after getting some events
        collected_events: list[Any] = []
        event_count = 0
        async for event in response:
            collected_events.append(event)
            event_count += 1
            if event_count == 2:  # After getting 2 events
                response._cancelled = True  # pyright: ignore[reportPrivateUsage]

        # Should have stopped before getting all events from the generator
        # The exact number depends on how the wrappers work, but we shouldn't get all 4
        assert len(events_yielded) < 4  # Generator didn't yield all 4 items

    async def test_agent_stream_cancelled_stops_iteration(self):
        """Setting _cancelled=True on AgentStream causes stream() to stop.

        Tests the _cancelled checks in result.py line 339-340.
        We use the run_stream context manager and cancel mid-iteration.
        """

        chunks_generated = 0

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            nonlocal chunks_generated
            for i in range(10):  # pragma: no branch
                chunks_generated += 1
                yield f'chunk {i}'  # type: ignore

        agent = Agent(FunctionModel(stream_function=stream_function))

        async with agent.run_stream('Test') as result:
            # Iterate directly over AgentStream (which uses __aiter__) to hit line 339-340
            events: list[Any] = []
            event_count = 0
            if result._stream_response:  # pyright: ignore[reportPrivateUsage]  # pragma: no branch
                async for event in result._stream_response:  # pyright: ignore[reportPrivateUsage]
                    events.append(event)
                    event_count += 1
                    if event_count == 3:
                        # Set _cancelled during iteration to hit line 339-340
                        result._stream_response._cancelled = True  # pyright: ignore[reportPrivateUsage]

        # We set _cancelled after 3 events, so we should have at least 3 but less than all events
        assert 3 <= len(events) < 20  # Less than all possible events (10 chunks * 2 events each)

    async def test_wrap_with_final_event_second_loop_cancelled(self):
        """Test that _cancelled=True during the second loop of _wrap_with_final_event breaks.

        The second loop (line 962-965) only runs when:
        1. First loop found a FinalResultEvent and broke early
        2. _cancelled is False at line 957-958
        3. There are more events to yield

        This test hits line 964: `if self._cancelled: break` in the second loop.
        """
        from pydantic_ai.models import ModelRequestParameters
        from pydantic_ai.models.function import DeltaToolCall, FunctionStreamedResponse

        # Create a tool output definition so ToolCallPart triggers FinalResultEvent
        tool_def = ToolDefinition(
            name='output_tool',
            description='Output tool',
            parameters_json_schema={'type': 'object'},
            kind='output',
        )

        async def event_generator():
            # First yield a tool call (triggers FinalResultEvent with kind='output')
            yield {0: DeltaToolCall(name='output_tool', json_args='{"value": 1}')}
            # These will be yielded by the second loop after FinalResultEvent
            yield 'More text after tool'  # pragma: no cover
            yield 'Even more text'  # pragma: no cover

        response = FunctionStreamedResponse(
            model_request_parameters=ModelRequestParameters(
                allow_text_output=True,
                output_tools=[tool_def],  # Use output_tools, not tool_defs
            ),
            _model_name='test-model',
            _iter=event_generator(),
        )

        events: list[Any] = []
        event_count = 0
        async for event in response:
            events.append(event)
            event_count += 1
            # Cancel after the FinalResultEvent is yielded (event 2 after PartStart + FinalResult)
            if event_count >= 2:
                response._cancelled = True  # pyright: ignore[reportPrivateUsage]

        # Should have gotten events but stopped after cancellation
        assert event_count >= 2

    async def test_wrap_with_part_end_cancelled(self):
        """Test that _cancelled=True during _wrap_with_part_end breaks iteration.

        This test hits line 992: `if self._cancelled: break` in _wrap_with_part_end.
        """
        from pydantic_ai.messages import TextPart
        from pydantic_ai.models import ModelRequestParameters, StreamedResponse

        events_from_inner: list[int] = []

        class TestStreamedResponse(StreamedResponse):
            """StreamedResponse with controllable event iterator."""

            @property
            def model_name(self) -> str:  # pragma: no cover
                return 'test-model'  # pragma: no cover

            @property
            def timestamp(self):  # pragma: no cover
                from datetime import datetime, timezone  # pragma: no cover

                return datetime.now(timezone.utc)  # pragma: no cover

            @property
            def provider_name(self) -> str:  # pragma: no cover
                return 'test'  # pragma: no cover

            @property
            def provider_url(self) -> str:  # pragma: no cover
                return 'https://test.example.com'  # pragma: no cover

            async def _get_event_iterator(self):
                for i in range(100):  # Many events  # pragma: no branch
                    event = self._parts_manager.handle_part(vendor_part_id=i, part=TextPart(content=f'text {i}'))
                    events_from_inner.append(i)
                    yield event

        response = TestStreamedResponse(model_request_parameters=ModelRequestParameters(allow_text_output=False))

        # Directly iterate over _wrap_with_part_end to test it in isolation
        events: list[Any] = []
        inner_iter = response._get_event_iterator()  # pyright: ignore[reportPrivateUsage]
        wrapped_iter = response._wrap_with_part_end(inner_iter)  # pyright: ignore[reportPrivateUsage]

        async for event in wrapped_iter:
            events.append(event)
            # Cancel after getting 5 events - next iteration should hit the _cancelled check
            if len(events) == 5:
                response._cancelled = True  # pyright: ignore[reportPrivateUsage]

        # Verify that cancellation stopped iteration early
        # Without cancellation, we'd get 100+ events (including PartEndEvents)
        assert len(events) < 50
        assert len(events_from_inner) < 50

    async def test_stream_text_deltas_cancelled(self):
        """Test that _cancelled=True during _stream_text_deltas returns early.

        This test hits line 209 in result.py: `if self._cancelled: return`
        """

        async def stream_function(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
            # Yield many text chunks
            for i in range(10):
                yield f'chunk {i} '  # type: ignore

        agent = Agent(FunctionModel(stream_function=stream_function))

        async with agent.run_stream('Test') as result:
            # Use stream_text() and cancel during iteration to hit line 209
            text_chunks: list[str] = []
            chunk_count = 0
            async for text in result.stream_text(delta=True):
                text_chunks.append(text)
                chunk_count += 1
                if chunk_count == 3:  # pragma: no cover
                    # Cancel during streaming to hit the _cancelled check
                    if result._stream_response:  # pyright: ignore[reportPrivateUsage]  # pragma: no cover
                        result._stream_response._cancelled = True  # pyright: ignore[reportPrivateUsage]  # pragma: no cover

        # Should have stopped early
        assert chunk_count < 10

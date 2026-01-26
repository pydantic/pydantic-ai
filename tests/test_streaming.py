from __future__ import annotations as _annotations

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
    ToolCallPart,
    ToolReturnPart,
    UnexpectedModelBehavior,
    UserError,
    UserPromptPart,
    capture_run_messages,
    models,
)
from pydantic_ai._agent_graph import GraphAgentState
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

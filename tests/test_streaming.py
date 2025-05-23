from __future__ import annotations as _annotations

import datetime
import json
import re
from collections.abc import AsyncIterator
from copy import deepcopy
from datetime import timezone
from typing import Union

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, UnexpectedModelBehavior, UserError, capture_run_messages
from pydantic_ai.agent import AgentRun
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.result import AgentStream, FinalResult, Usage
from pydantic_graph import End

from .conftest import IsInt, IsNow, IsStr

pytestmark = pytest.mark.anyio


async def test_streamed_text_response():
    m = TestModel()

    test_agent = Agent(m)
    assert test_agent.name is None

    @test_agent.tool_plain
    async def ret_a(x: str) -> str:
        return f'{x}-apple'

    async with test_agent.run_stream('Hello') as result:
        assert test_agent.name == 'test_agent'
        assert not result.is_complete
        assert result.all_messages() == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                    usage=Usage(request_tokens=51, response_tokens=0, total_tokens=51),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                        )
                    ]
                ),
            ]
        )
        assert result.usage() == snapshot(
            Usage(
                requests=2,
                request_tokens=103,
                response_tokens=5,
                total_tokens=108,
            )
        )
        response = await result.get_output()
        assert response == snapshot('{"ret_a":"a-apple"}')
        assert result.is_complete
        assert result.timestamp() == IsNow(tz=timezone.utc)
        assert result.all_messages() == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args={'x': 'a'}, tool_call_id=IsStr())],
                    usage=Usage(request_tokens=51, response_tokens=0, total_tokens=51),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a', content='a-apple', timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                        )
                    ]
                ),
                ModelResponse(
                    parts=[TextPart(content='{"ret_a":"a-apple"}')],
                    usage=Usage(request_tokens=52, response_tokens=11, total_tokens=63),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
            ]
        )
        assert result.usage() == snapshot(
            Usage(
                requests=2,
                request_tokens=103,
                response_tokens=11,
                total_tokens=114,
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
        async for structured_response, last in result.stream_structured(debounce_by=None):
            response_data = await result.validate_structured_output(structured_response, allow_partial=not last)
            chunks.append(response_data)

    assert chunks == snapshot([[1], [1, 2, 3, 4], [1, 2, 3, 4]])

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
        chunks: list[str] = [c async for c in result.stream()]
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

    async with agent.run_stream('Hello') as result:
        assert [c async for c, _is_last in result.stream_structured(debounce_by=None)] == snapshot(
            [
                ModelResponse(
                    parts=[TextPart(content='The ')],
                    usage=Usage(request_tokens=51, response_tokens=1, total_tokens=52),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat ')],
                    usage=Usage(request_tokens=51, response_tokens=2, total_tokens=53),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat ')],
                    usage=Usage(request_tokens=51, response_tokens=3, total_tokens=54),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on ')],
                    usage=Usage(request_tokens=51, response_tokens=4, total_tokens=55),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the ')],
                    usage=Usage(request_tokens=51, response_tokens=5, total_tokens=56),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the mat.')],
                    usage=Usage(request_tokens=51, response_tokens=7, total_tokens=58),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelResponse(
                    parts=[TextPart(content='The cat sat on the mat.')],
                    usage=Usage(request_tokens=51, response_tokens=7, total_tokens=58),
                    model_name='test',
                    timestamp=IsNow(tz=timezone.utc),
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

    with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(1\) for result validation'):
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
                ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args='{"x": "hello"}', tool_call_id=IsStr())],
                    usage=Usage(request_tokens=50, response_tokens=5, total_tokens=55),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a',
                            content='hello world',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ]
                ),
            ]
        )
        assert await result.get_output() == snapshot(('hello world', 2))
        assert result.all_messages() == snapshot(
            [
                ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
                ModelResponse(
                    parts=[ToolCallPart(tool_name='ret_a', args='{"x": "hello"}', tool_call_id=IsStr())],
                    usage=Usage(request_tokens=50, response_tokens=5, total_tokens=55),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='ret_a',
                            content='hello world',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ]
                ),
                ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name='final_result',
                            args='{"response": ["hello world", 2]}',
                            tool_call_id=IsStr(),
                        )
                    ],
                    usage=Usage(request_tokens=50, response_tokens=7, total_tokens=57),
                    model_name='function::stream_structured_function',
                    timestamp=IsNow(tz=timezone.utc),
                ),
                ModelRequest(
                    parts=[
                        ToolReturnPart(
                            tool_name='final_result',
                            content='Final result processed.',
                            timestamp=IsNow(tz=timezone.utc),
                            tool_call_id=IsStr(),
                        )
                    ]
                ),
            ]
        )


async def test_call_tool_empty():
    async def stream_structured_function(_messages: list[ModelMessage], _: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        yield {}

    agent = Agent(FunctionModel(stream_function=stream_structured_function), output_type=tuple[str, int])

    with pytest.raises(UnexpectedModelBehavior, match='Received empty model response'):
        async with agent.run_stream('hello'):
            pass


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
        with pytest.raises(UnexpectedModelBehavior, match=r'Exceeded maximum retries \(0\) for result validation'):
            async with agent.run_stream('hello'):
                pass

    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hello', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='foobar', args='{}', tool_call_id=IsStr())],
                usage=Usage(request_tokens=50, response_tokens=1, total_tokens=51),
                model_name='function::stream_structured_function',
                timestamp=IsNow(tz=timezone.utc),
            ),
        ]
    )


class OutputType(BaseModel):
    """Result type used by all tests."""

    value: str


async def test_early_strategy_stops_after_first_final_result():
    """Test that 'early' strategy stops processing regular tools after first final result."""
    tool_called: list[str] = []

    async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert info.output_tools is not None
        yield {1: DeltaToolCall('final_result', '{"value": "final"}')}
        yield {2: DeltaToolCall('regular_tool', '{"x": 1}')}
        yield {3: DeltaToolCall('another_tool', '{"y": 2}')}

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

    async with agent.run_stream('test early strategy') as result:
        response = await result.get_output()
        assert response.value == snapshot('final')
        messages = result.all_messages()

    # Verify no tools were called after final result
    assert tool_called == []

    # Verify we got tool returns for all calls
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='test early strategy', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args='{"value": "final"}', tool_call_id=IsStr()),
                    ToolCallPart(tool_name='regular_tool', args='{"x": 1}', tool_call_id=IsStr()),
                    ToolCallPart(tool_name='another_tool', args='{"y": 2}', tool_call_id=IsStr()),
                ],
                usage=Usage(request_tokens=50, response_tokens=10, total_tokens=60),
                model_name='function::sf',
                timestamp=IsNow(tz=timezone.utc),
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
                ]
            ),
        ]
    )


async def test_early_strategy_uses_first_final_result():
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
                parts=[UserPromptPart(content='test multiple final results', timestamp=IsNow(tz=timezone.utc))]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args='{"value": "first"}', tool_call_id=IsStr()),
                    ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                ],
                usage=Usage(request_tokens=50, response_tokens=8, total_tokens=58),
                model_name='function::sf',
                timestamp=IsNow(tz=timezone.utc),
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
                ]
            ),
        ]
    )


async def test_exhaustive_strategy_executes_all_tools():
    """Test that 'exhaustive' strategy executes all tools while using first final result."""
    tool_called: list[str] = []

    async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert info.output_tools is not None
        yield {1: DeltaToolCall('final_result', '{"value": "first"}')}
        yield {2: DeltaToolCall('regular_tool', '{"x": 42}')}
        yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
        yield {4: DeltaToolCall('final_result', '{"value": "second"}')}
        yield {5: DeltaToolCall('unknown_tool', '{"value": "???"}')}

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

    async with agent.run_stream('test exhaustive strategy') as result:
        response = await result.get_output()
        assert response.value == snapshot('first')
        messages = result.all_messages()

    # Verify we got tool returns in the correct order
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='test exhaustive strategy', timestamp=IsNow(tz=timezone.utc))]),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='final_result', args='{"value": "first"}', tool_call_id=IsStr()),
                    ToolCallPart(tool_name='regular_tool', args='{"x": 42}', tool_call_id=IsStr()),
                    ToolCallPart(tool_name='another_tool', args='{"y": 2}', tool_call_id=IsStr()),
                    ToolCallPart(tool_name='final_result', args='{"value": "second"}', tool_call_id=IsStr()),
                    ToolCallPart(tool_name='unknown_tool', args='{"value": "???"}', tool_call_id=IsStr()),
                ],
                usage=Usage(request_tokens=50, response_tokens=18, total_tokens=68),
                model_name='function::sf',
                timestamp=IsNow(tz=timezone.utc),
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
                    RetryPromptPart(
                        tool_name='unknown_tool',
                        content="Unknown tool name: 'unknown_tool'. Available tools: regular_tool, another_tool, final_result",
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    ),
                    ToolReturnPart(
                        tool_name='regular_tool', content=42, timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                    ToolReturnPart(
                        tool_name='another_tool', content=2, timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    ),
                ]
            ),
        ]
    )


async def test_early_strategy_with_final_result_in_middle():
    """Test that 'early' strategy stops at first final result, regardless of position."""
    tool_called: list[str] = []

    async def sf(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        assert info.output_tools is not None
        yield {1: DeltaToolCall('regular_tool', '{"x": 1}')}
        yield {2: DeltaToolCall('final_result', '{"value": "final"}')}
        yield {3: DeltaToolCall('another_tool', '{"y": 2}')}
        yield {4: DeltaToolCall('unknown_tool', '{"value": "???"}')}

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
                        part_kind='user-prompt',
                    )
                ],
                kind='request',
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='regular_tool',
                        args='{"x": 1}',
                        tool_call_id=IsStr(),
                        part_kind='tool-call',
                    ),
                    ToolCallPart(
                        tool_name='final_result',
                        args='{"value": "final"}',
                        tool_call_id=IsStr(),
                        part_kind='tool-call',
                    ),
                    ToolCallPart(
                        tool_name='another_tool',
                        args='{"y": 2}',
                        tool_call_id=IsStr(),
                        part_kind='tool-call',
                    ),
                    ToolCallPart(
                        tool_name='unknown_tool',
                        args='{"value": "???"}',
                        tool_call_id=IsStr(),
                        part_kind='tool-call',
                    ),
                ],
                usage=Usage(request_tokens=50, response_tokens=14, total_tokens=64),
                model_name='function::sf',
                timestamp=IsNow(tz=datetime.timezone.utc),
                kind='response',
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='regular_tool',
                        content='Tool not executed - a final result was already processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=datetime.timezone.utc),
                        part_kind='tool-return',
                    ),
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=datetime.timezone.utc),
                        part_kind='tool-return',
                    ),
                    ToolReturnPart(
                        tool_name='another_tool',
                        content='Tool not executed - a final result was already processed.',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=datetime.timezone.utc),
                        part_kind='tool-return',
                    ),
                    RetryPromptPart(
                        content='Unknown tool name: '
                        "'unknown_tool'. Available tools: "
                        'regular_tool, another_tool, '
                        'final_result',
                        tool_name='unknown_tool',
                        tool_call_id=IsStr(),
                        timestamp=IsNow(tz=datetime.timezone.utc),
                        part_kind='retry-prompt',
                    ),
                ],
                kind='request',
            ),
        ]
    )


async def test_early_strategy_does_not_apply_to_tool_calls_without_final_tool():
    """Test that 'early' strategy does not apply to tool calls without final tool."""
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

    assert tool_called == ['regular_tool']

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='test early strategy with regular tool calls', timestamp=IsNow(tz=timezone.utc)
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='regular_tool', args={'x': 0}, tool_call_id=IsStr())],
                usage=Usage(request_tokens=57, response_tokens=0, total_tokens=57),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='regular_tool', content=0, timestamp=IsNow(tz=timezone.utc), tool_call_id=IsStr()
                    )
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='final_result', args={'value': 'a'}, tool_call_id=IsStr())],
                usage=Usage(request_tokens=58, response_tokens=4, total_tokens=62),
                model_name='test',
                timestamp=IsNow(tz=timezone.utc),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        timestamp=IsNow(tz=timezone.utc),
                        tool_call_id=IsStr(),
                    )
                ]
            ),
        ]
    )


async def test_custom_output_type_default_str() -> None:
    agent = Agent('test')

    async with agent.run_stream('test') as result:
        response = await result.get_output()
        assert response == snapshot('success (no tool calls)')

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
    stream: AgentStream
    messages: list[str] = []

    stream_usage: Usage | None = None
    async with agent.iter('Hello') as run:
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_output(debounce_by=None):
                        messages.append(chunk)
                stream_usage = deepcopy(stream.usage())
    assert run.next_node == End(data=FinalResult(output='The bat sat on the mat.', tool_name=None, tool_call_id=None))
    assert (
        run.usage()
        == stream_usage
        == Usage(requests=1, request_tokens=51, response_tokens=7, total_tokens=58, details=None)
    )

    assert messages == [
        '',
        'The ',
        'The cat ',
        'The bat sat ',
        'The bat sat on ',
        'The bat sat on the ',
        'The bat sat on the mat.',
        'The bat sat on the mat.',
    ]


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
        async for node in run:
            if agent.is_model_request_node(node):
                async with node.stream(run.ctx) as stream:
                    async for chunk in stream.stream_responses(debounce_by=None):
                        messages.append(chunk)

    assert messages == [
        ModelResponse(
            parts=[TextPart(content=text, part_kind='text')],
            usage=Usage(
                requests=0, request_tokens=IsInt(), response_tokens=IsInt(), total_tokens=IsInt(), details=None
            ),
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            kind='response',
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
        ]
    ]

    # Note: as you can see above, the output validator is not applied to the streamed responses, just the final result:
    assert run.result is not None
    assert run.result.output == 'The bat sat on the mat.'


async def test_stream_iter_structured_validator() -> None:
    class NotOutputType(BaseModel):
        not_value: str

    agent = Agent[None, Union[OutputType, NotOutputType]]('test', output_type=Union[OutputType, NotOutputType])  # pyright: ignore[reportArgumentType]

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
    assert outputs == [OutputType(value='a (validated)'), OutputType(value='a (validated)')]

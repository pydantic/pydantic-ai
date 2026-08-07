"""Tests for the Claude Code `stream-json` UI event stream.

The format is not a versioned public spec, so the golden fixtures in
`tests/assets/claude_code_stream_json/` (captured from Claude Code CLI 2.1.222) are the spec:
`test_emitted_shapes_were_observed_in_real_cli_output` asserts every field name we emit was seen in
a real CLI stream. The other acceptance bar is gh-aw's own Claude log parser, vendored in
`tests/assets/ghaw_log_parser/` and run over our output by `test_ghaw_parser_reads_our_stream`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import count
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import (
    BinaryImage,
    CompactionPart,
    ModelMessage,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartEndEvent,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.ui import NativeEvent
from pydantic_ai.ui.claude_code import NDJSON_CONTENT_TYPE, ClaudeCodeEventStream
from pydantic_ai.usage import RunUsage, UsageLimits

pytestmark = pytest.mark.anyio

StreamedChunks = AsyncIterator[DeltaThinkingCalls | DeltaToolCalls | str]


def _event_stream(*, include_partial_messages: bool = False) -> ClaudeCodeEventStream[Any, Any]:
    """An event stream with every nondeterministic input pinned, so its lines can be snapshotted."""
    ids = count(1)
    ticks = count(0)
    return ClaudeCodeEventStream(
        session_id='session-1',
        model='function:stream',
        cwd='/workspace',
        include_partial_messages=include_partial_messages,
        id_factory=lambda: f'id-{next(ids)}',
        now=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(milliseconds=100 * next(ticks)),
    )


async def _lines(stream: ClaudeCodeEventStream[Any, Any], events: AsyncIterator[NativeEvent]) -> list[Any]:
    """Run a native event stream through `stream` and decode the JSONL it encodes."""
    return [json.loads(line) async for line in stream.encode_stream(stream.transform_stream(events))]


async def _native_events(
    agent: Agent[Any, Any], prompt: str, usage_limits: UsageLimits | None = None, usage: RunUsage | None = None
) -> AsyncIterator[NativeEvent]:
    async with agent.run_stream_events(prompt, usage_limits=usage_limits, usage=usage) as events:
        async for event in events:
            yield event


async def _run_lines(agent: Agent[Any, Any], prompt: str, *, include_partial_messages: bool = False) -> list[Any]:
    stream = _event_stream(include_partial_messages=include_partial_messages)
    return await _lines(stream, _native_events(agent, prompt))


async def test_plain_text_run():
    """Text deltas are buffered into one whole-block `assistant` line, then the terminal `result`."""

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield 'hello '
        yield 'world'

    lines = await _run_lines(Agent(FunctionModel(stream_function=stream_function)), 'say hello')

    assert lines == snapshot(
        [
            {
                'type': 'system',
                'subtype': 'init',
                'cwd': '/workspace',
                'session_id': 'session-1',
                'tools': [],
                'mcp_servers': [],
                'permissionMode': 'default',
                'slash_commands': [],
                'output_style': 'default',
                'uuid': 'id-1',
                'model': 'function:stream',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'hello world'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-3',
                'timestamp': '2026-01-01T00:00:00.100Z',
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 1,
                'duration_ms': 200,
                'result': 'hello world',
                'session_id': 'session-1',
                'uuid': 'id-4',
                'usage': {
                    'input_tokens': 50,
                    'output_tokens': 2,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            },
        ]
    )


def test_content_type_is_ndjson():
    """`stream-json` is newline-delimited JSON, not the base class's SSE default."""
    assert _event_stream().content_type == NDJSON_CONTENT_TYPE == 'application/x-ndjson'


async def test_default_ids_and_clock():
    """Left to its defaults, the stream generates its own session id, uuids and UTC timestamps."""

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield 'hello'

    stream = ClaudeCodeEventStream(cwd='/workspace')
    lines = await _lines(stream, _native_events(Agent(FunctionModel(stream_function=stream_function)), 'hi'))

    init, assistant, result = lines
    assert init['session_id'] == stream.session_id == result['session_id']
    assert len({init['uuid'], assistant['uuid'], result['uuid']}) == 3
    assert assistant['timestamp'].endswith('Z')
    assert result['duration_ms'] >= 0
    # No model was configured, so none is claimed on the init record or on the message.
    assert 'model' not in init
    assert 'model' not in assistant['message']

    # Nor on the message the partial-messages mode announces up front.
    partial = ClaudeCodeEventStream(cwd='/workspace', include_partial_messages=True)
    partial_lines = await _lines(partial, _native_events(Agent(FunctionModel(stream_function=stream_function)), 'hi'))
    message_start = next(line for line in partial_lines if line['type'] == 'stream_event')
    assert 'model' not in message_start['event']['message']


async def test_tool_call_and_result():
    """A tool call pairs a `tool_use` block with a `tool_result` block carrying the same id.

    Unpaired ids are what make gh-aw render a call as unresolved, so the pairing is load-bearing.
    The args arrive fragmented, as a provider streams them, and are emitted only once complete.
    """

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='get_weather', json_args='{"city":', tool_call_id='call-1')}
            yield {0: DeltaToolCall(json_args=' "Utrecht"}', tool_call_id='call-1')}
        else:
            yield 'It is sunny in Utrecht.'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    lines = await _run_lines(agent, 'weather?')

    assert lines == snapshot(
        [
            {
                'type': 'system',
                'subtype': 'init',
                'cwd': '/workspace',
                'session_id': 'session-1',
                'tools': [],
                'mcp_servers': [],
                'permissionMode': 'default',
                'slash_commands': [],
                'output_style': 'default',
                'uuid': 'id-1',
                'model': 'function:stream',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [
                        {'type': 'tool_use', 'id': 'call-1', 'name': 'get_weather', 'input': {'city': 'Utrecht'}}
                    ],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-3',
                'timestamp': '2026-01-01T00:00:00.100Z',
            },
            {
                'type': 'user',
                'message': {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'sunny in Utrecht'}],
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-4',
                'timestamp': '2026-01-01T00:00:00.200Z',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-5',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'It is sunny in Utrecht.'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-6',
                'timestamp': '2026-01-01T00:00:00.300Z',
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 2,
                'duration_ms': 400,
                'result': 'It is sunny in Utrecht.',
                'session_id': 'session-1',
                'uuid': 'id-7',
                'usage': {
                    'input_tokens': 100,
                    'output_tokens': 12,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            },
        ]
    )


async def test_failed_tool_is_marked_as_an_error():
    """A retry prompt becomes a `tool_result` with `is_error`, which is what renders it as a failure."""

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='flaky', json_args='{}', tool_call_id='call-1')}
        else:
            yield 'Gave up.'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    def flaky() -> str:
        raise ModelRetry('try something else')

    lines = await _run_lines(agent, 'go')

    assert lines == snapshot(
        [
            {
                'type': 'system',
                'subtype': 'init',
                'cwd': '/workspace',
                'session_id': 'session-1',
                'tools': [],
                'mcp_servers': [],
                'permissionMode': 'default',
                'slash_commands': [],
                'output_style': 'default',
                'uuid': 'id-1',
                'model': 'function:stream',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'tool_use', 'id': 'call-1', 'name': 'flaky', 'input': {}}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-3',
                'timestamp': '2026-01-01T00:00:00.100Z',
            },
            {
                'type': 'user',
                'message': {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'tool_result',
                            'tool_use_id': 'call-1',
                            'content': """\
try something else

Fix the errors and try again.\
""",
                            'is_error': True,
                        }
                    ],
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-4',
                'timestamp': '2026-01-01T00:00:00.200Z',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-5',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'Gave up.'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-6',
                'timestamp': '2026-01-01T00:00:00.300Z',
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 2,
                'duration_ms': 400,
                'result': 'Gave up.',
                'session_id': 'session-1',
                'uuid': 'id-7',
                'usage': {
                    'input_tokens': 100,
                    'output_tokens': 4,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            },
        ]
    )


async def test_thinking_run():
    """Thinking parts become `thinking` blocks, carrying their signature when the model signs them."""

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield {0: DeltaThinkingPart(content='Signed thought')}
        yield {0: DeltaThinkingPart(signature='sig-1')}
        yield {1: DeltaThinkingPart(content='Unsigned thought')}
        yield 'Done thinking.'

    lines = await _run_lines(Agent(FunctionModel(stream_function=stream_function)), 'think')

    assert lines == snapshot(
        [
            {
                'type': 'system',
                'subtype': 'init',
                'cwd': '/workspace',
                'session_id': 'session-1',
                'tools': [],
                'mcp_servers': [],
                'permissionMode': 'default',
                'slash_commands': [],
                'output_style': 'default',
                'uuid': 'id-1',
                'model': 'function:stream',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'thinking', 'thinking': 'Signed thought', 'signature': 'sig-1'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-3',
                'timestamp': '2026-01-01T00:00:00.100Z',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'thinking', 'thinking': 'Unsigned thought'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-4',
                'timestamp': '2026-01-01T00:00:00.200Z',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'Done thinking.'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-5',
                'timestamp': '2026-01-01T00:00:00.300Z',
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 1,
                'duration_ms': 400,
                'result': 'Done thinking.',
                'session_id': 'session-1',
                'uuid': 'id-6',
                'usage': {
                    'input_tokens': 50,
                    'output_tokens': 7,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            },
        ]
    )


async def test_multimodal_tool_result_uses_the_content_array_form():
    """A tool return carrying files becomes the array content form, referencing them by identifier."""

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='screenshot', json_args='{}', tool_call_id='call-1')}
        else:
            yield 'Got it.'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    def screenshot() -> list[str | BinaryImage]:
        return ['here is the screen', BinaryImage(data=b'fake-png', media_type='image/png', identifier='shot')]

    lines = await _run_lines(agent, 'capture')

    assert [line['message']['content'] for line in lines if line['type'] == 'user'] == snapshot(
        [
            [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'call-1',
                    'content': [{'type': 'text', 'text': '["here is the screen","See file shot."]'}],
                }
            ]
        ]
    )


async def test_output_tool_result():
    """A structured output's tool call and its result are recorded like any other tool exchange."""

    class Weather(BaseModel):
        city: str
        summary: str

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield {
            0: DeltaToolCall(
                name='final_result', json_args='{"city": "Utrecht", "summary": "sunny"}', tool_call_id='call-1'
            )
        }

    agent = Agent(FunctionModel(stream_function=stream_function), output_type=Weather)
    lines = await _run_lines(agent, 'weather?')

    assert [(line['type'], line.get('message', {}).get('content')) for line in lines] == snapshot(
        [
            ('system', None),
            (
                'assistant',
                [
                    {
                        'type': 'tool_use',
                        'id': 'call-1',
                        'name': 'final_result',
                        'input': {'city': 'Utrecht', 'summary': 'sunny'},
                    }
                ],
            ),
            ('user', [{'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'Final result processed.'}]),
            ('result', None),
        ]
    )


async def test_result_record_reports_stop_reason_and_cost():
    """A finish reason maps onto Anthropic's `stop_reason`, and a priced run reports its cost.

    Driven from a synthetic result because the test models report neither. `total_cost_usd` is only
    ever emitted for a real cost: a `0` is falsy to the JavaScript consumers, so it reads as absent.
    """
    response = ModelResponse(parts=[TextPart(content='Done.')], finish_reason='stop')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=response.parts[0])
        yield PartEndEvent(index=0, part=response.parts[0])
        result = AgentRunResult(output='Done.')
        result._state.message_history = [response]  # pyright: ignore[reportPrivateUsage]
        result._state.usage = RunUsage(input_tokens=10, output_tokens=3, cost=Decimal('0.0042'))  # pyright: ignore[reportPrivateUsage]
        yield AgentRunResultEvent(result=result)

    lines = await _lines(_event_stream(), events())

    assert lines[-1] == snapshot(
        {
            'type': 'result',
            'subtype': 'success',
            'is_error': False,
            'terminal_reason': 'completed',
            'num_turns': 1,
            'duration_ms': 200,
            'result': 'Done.',
            'session_id': 'session-1',
            'uuid': 'id-4',
            'stop_reason': 'end_turn',
            'usage': {
                'input_tokens': 10,
                'output_tokens': 3,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
            },
            'total_cost_usd': 0.0042,
        }
    )


async def test_partial_messages_mode():
    """`include_partial_messages` interleaves `stream_event` records without dropping the whole blocks.

    The `result` record still comes last: gh-aw's parser reads the last raw JSON line for its entire
    Information section, so a trailing `message_stop` would blank out turns and token usage.
    """

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        if len(messages) == 1:
            yield {0: DeltaThinkingPart(content='I should ')}
            yield {0: DeltaThinkingPart(content='look it up')}
            yield {0: DeltaThinkingPart(signature='sig-1')}
            yield {1: DeltaToolCall(name='get_weather', json_args='{"city":', tool_call_id='call-1')}
            yield {1: DeltaToolCall(json_args=' "Utrecht"}', tool_call_id='call-1')}
        else:
            yield 'It is '
            yield 'sunny.'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    lines = await _run_lines(agent, 'weather?', include_partial_messages=True)

    assert lines[-1]['type'] == 'result'
    assert lines == snapshot(
        [
            {
                'type': 'system',
                'subtype': 'init',
                'cwd': '/workspace',
                'session_id': 'session-1',
                'tools': [],
                'mcp_servers': [],
                'permissionMode': 'default',
                'slash_commands': [],
                'output_style': 'default',
                'uuid': 'id-1',
                'model': 'function:stream',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_id-2',
                        'type': 'message',
                        'role': 'assistant',
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'model': 'function:stream',
                    },
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-3',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_start',
                    'index': 0,
                    'content_block': {'type': 'thinking', 'thinking': '', 'signature': ''},
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-4',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_delta',
                    'index': 0,
                    'delta': {'type': 'thinking_delta', 'thinking': 'I should '},
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-5',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_delta',
                    'index': 0,
                    'delta': {'type': 'thinking_delta', 'thinking': 'look it up'},
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-6',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_delta',
                    'index': 0,
                    'delta': {'type': 'signature_delta', 'signature': 'sig-1'},
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-7',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'thinking', 'thinking': 'I should look it up', 'signature': 'sig-1'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-8',
                'timestamp': '2026-01-01T00:00:00.100Z',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_stop', 'index': 0},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-9',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_start',
                    'index': 1,
                    'content_block': {'type': 'tool_use', 'id': 'call-1', 'name': 'get_weather', 'input': {}},
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-10',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_delta',
                    'index': 1,
                    'delta': {'type': 'input_json_delta', 'partial_json': '{"city":'},
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-11',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'content_block_delta',
                    'index': 1,
                    'delta': {'type': 'input_json_delta', 'partial_json': ' "Utrecht"}'},
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-12',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [
                        {'type': 'tool_use', 'id': 'call-1', 'name': 'get_weather', 'input': {'city': 'Utrecht'}}
                    ],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-13',
                'timestamp': '2026-01-01T00:00:00.200Z',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_stop', 'index': 1},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-14',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'message_delta', 'delta': {'stop_reason': None, 'stop_sequence': None}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-15',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'message_stop'},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-16',
            },
            {
                'type': 'user',
                'message': {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'sunny in Utrecht'}],
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-17',
                'timestamp': '2026-01-01T00:00:00.300Z',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'message_start',
                    'message': {
                        'id': 'msg_id-18',
                        'type': 'message',
                        'role': 'assistant',
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'model': 'function:stream',
                    },
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-19',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-20',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'It is '}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-21',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'sunny.'}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-22',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-18',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'It is sunny.'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-23',
                'timestamp': '2026-01-01T00:00:00.400Z',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_stop', 'index': 0},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-24',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'message_delta', 'delta': {'stop_reason': None, 'stop_sequence': None}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-25',
            },
            {
                'type': 'stream_event',
                'event': {'type': 'message_stop'},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-26',
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 2,
                'duration_ms': 500,
                'result': 'It is sunny.',
                'session_id': 'session-1',
                'uuid': 'id-27',
                'usage': {
                    'input_tokens': 100,
                    'output_tokens': 15,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            },
        ]
    )


async def test_error_run_still_closes_with_a_result():
    """A run that raises still terminates the stream with a `result` record reporting the failure."""

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        raise RuntimeError('the model exploded')
        yield  # Make this an async generator

    lines = await _run_lines(Agent(FunctionModel(stream_function=stream_function)), 'go')

    assert lines == snapshot(
        [
            {
                'type': 'system',
                'subtype': 'init',
                'cwd': '/workspace',
                'session_id': 'session-1',
                'tools': [],
                'mcp_servers': [],
                'permissionMode': 'default',
                'slash_commands': [],
                'output_style': 'default',
                'uuid': 'id-1',
                'model': 'function:stream',
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': True,
                'terminal_reason': 'error',
                'num_turns': 0,
                'duration_ms': 100,
                'result': 'the model exploded',
                'session_id': 'session-1',
                'uuid': 'id-2',
                'errors': ['the model exploded'],
            },
        ]
    )


async def test_usage_limit_exceeded_changes_the_result_subtype():
    """Turn exhaustion is the one failure the CLI reports through `subtype`, not just `is_error`."""

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield {0: DeltaToolCall(name='noop', json_args='{}', tool_call_id=f'call-{len(messages)}')}

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    lines = await _lines(_event_stream(), _native_events(agent, 'go', UsageLimits(request_limit=1)))

    result = lines[-1]
    assert result['type'] == 'result'
    assert (result['subtype'], result['is_error'], result['terminal_reason']) == snapshot(
        ('error_max_turns', True, 'max_turns')
    )


async def _synthetic_events() -> AsyncIterator[NativeEvent]:
    """Parts no test model emits, in a stream that also ends without an `AgentRunResultEvent`."""
    native_call = NativeToolCallPart('web_search', {'query': 'pydantic'}, tool_call_id='ws-1')
    yield PartStartEvent(index=0, part=CompactionPart(content='summary of earlier turns'))
    yield PartStartEvent(index=1, part=native_call)
    yield PartEndEvent(index=1, part=native_call)
    yield PartStartEvent(index=2, part=NativeToolReturnPart('web_search', content='one result', tool_call_id='ws-1'))


async def test_synthetic_parts_without_a_run_result():
    """Compaction and native tool calls/returns all have `stream-json` counterparts.

    Ending without an `AgentRunResultEvent` also pins what the terminal `result` reports when no
    usage was ever gathered: the `usage` block is omitted rather than fabricated as zeros.
    """
    lines = await _lines(_event_stream(), _synthetic_events())

    assert lines == snapshot(
        [
            {
                'type': 'system',
                'subtype': 'init',
                'cwd': '/workspace',
                'session_id': 'session-1',
                'tools': [],
                'mcp_servers': [],
                'permissionMode': 'default',
                'slash_commands': [],
                'output_style': 'default',
                'uuid': 'id-1',
                'model': 'function:stream',
            },
            {
                'type': 'system',
                'subtype': 'compact_boundary',
                'compact_metadata': {'trigger': 'auto'},
                'session_id': 'session-1',
                'uuid': 'id-3',
                'timestamp': '2026-01-01T00:00:00.100Z',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': 'msg_id-2',
                    'type': 'message',
                    'role': 'assistant',
                    'content': [
                        {'type': 'tool_use', 'id': 'ws-1', 'name': 'web_search', 'input': {'query': 'pydantic'}}
                    ],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-4',
                'timestamp': '2026-01-01T00:00:00.200Z',
            },
            {
                'type': 'user',
                'message': {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 'ws-1', 'content': 'one result'}],
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': 'id-5',
                'timestamp': '2026-01-01T00:00:00.300Z',
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 1,
                'duration_ms': 400,
                'result': '',
                'session_id': 'session-1',
                'uuid': 'id-6',
            },
        ]
    )


def _shape(value: Any, label: str, shapes: dict[str, set[str]]) -> None:
    """Record the key set of every record, message, block, event and delta object in `value`."""
    if not is_str_dict(value):
        return
    shapes.setdefault(label, set()).update(value)
    for key, child in value.items():
        if key in ('message', 'event', 'delta', 'content_block'):
            child_type = child.get('type') if is_str_dict(child) else None
            _shape(child, f'{key}:{child_type}' if child_type else key, shapes)
        elif key == 'content' and not isinstance(child, str):
            for block in child:
                block_type = block.get('type') if is_str_dict(block) else None
                _shape(block, f'block:{block_type}', shapes)


def _stream_shapes(lines: list[Any]) -> dict[str, set[str]]:
    shapes: dict[str, set[str]] = {}
    for line in lines:
        label = f'record:{line["type"]}'
        if subtype := line.get('subtype'):
            label = f'{label}:{subtype}'
        _shape(line, label, shapes)
    return shapes


# The captured fixtures contain no failing tool call and no compaction, so these two have no CLI
# precedent to check against. `tool_result.is_error` is read by gh-aw's parser, which renders a
# failure as a success without it; `compact_boundary` is documented by the Claude Agent SDK's
# message types. Everything else we emit has to have been seen in a real stream.
UNOBSERVED_SHAPES = {'record:system:compact_boundary'}
UNOBSERVED_KEYS = {('block:tool_result', 'is_error')}


async def test_emitted_shapes_were_observed_in_real_cli_output(assets_path: Path):
    """Every field name we emit appears in a stream captured from the real Claude Code CLI."""

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        if len(messages) == 1:
            yield {0: DeltaThinkingPart(content='thinking', signature='sig-1')}
            yield {1: DeltaToolCall(name='flaky', json_args='{}', tool_call_id='call-1')}
        else:
            yield 'done'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    def flaky() -> str:
        raise ModelRetry('nope')

    fixture_shapes: dict[str, set[str]] = {}
    fixtures = sorted((assets_path / 'claude_code_stream_json').glob('*.jsonl'))
    assert len(fixtures) == 8
    for fixture in fixtures:
        lines = [json.loads(line) for line in fixture.read_text(encoding='utf-8').splitlines() if line]
        for label, keys in _stream_shapes(lines).items():
            fixture_shapes.setdefault(label, set()).update(keys)

    emitted = await _run_lines(agent, 'go')
    emitted += await _run_lines(agent, 'go', include_partial_messages=True)
    emitted += await _lines(_event_stream(), _synthetic_events())
    emitted_shapes = _stream_shapes(emitted)

    unexpected = {
        (label, key)
        for label, keys in emitted_shapes.items()
        if label not in UNOBSERVED_SHAPES
        for key in keys - fixture_shapes.get(label, set())
    }
    assert unexpected - UNOBSERVED_KEYS == set()
    # And the objects themselves: every kind of record, block and event we emit was seen in the CLI's
    # own output, so no consumer meets a shape the format doesn't have.
    assert set(emitted_shapes) - set(fixture_shapes) == UNOBSERVED_SHAPES


@pytest.mark.skipif(shutil.which('node') is None, reason='node is required to run the vendored gh-aw parser')
async def test_ghaw_parser_reads_our_stream(assets_path: Path):
    """gh-aw's own Claude log parser extracts turns, tokens and tool calls from our stream.

    This is the acceptance bar for "drop-in compatible": if their parser reads our output, an agent
    emitting it can run as gh-aw's `engine: claude` and keep the step-summary rendering and token
    metrics that third-party engines don't get.
    """

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='get_weather', json_args='{"city": "Utrecht"}', tool_call_id='call-1')}
        else:
            yield 'It is sunny in Utrecht.'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    stream = _event_stream()
    # Seeded cache counters so all four token names the parser reads carry a value; `FunctionModel`
    # reports none of its own.
    events = _native_events(agent, 'weather?', usage=RunUsage(cache_write_tokens=11, cache_read_tokens=7))
    jsonl = ''.join([line async for line in stream.encode_stream(stream.transform_stream(events))])

    driver = (
        'const {parseClaudeLog} = require(process.argv[1]);'
        "let log = '';"
        "process.stdin.on('data', chunk => (log += chunk));"
        "process.stdin.on('end', () => process.stdout.write(parseClaudeLog(log).markdown));"
    )
    parser = assets_path / 'ghaw_log_parser' / 'parse_claude_log.cjs'
    markdown = subprocess.run(
        ['node', '-e', driver, str(parser)],
        input=jsonl,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    information = markdown.split('<summary>Information</summary>')[1]
    assert '**Turns:** 2' in information
    assert [line for line in information.splitlines() if line.startswith('- ')] == snapshot(
        ['- Total: 128', '- Input: 100', '- Cache Creation: 11', '- Cache Read: 7', '- Output: 10']
    )

    tools = markdown.split('<summary>Commands and Tools</summary>')[1]
    assert '✅ get_weather' in tools
    # The final answer has to reach the reader as an `assistant` text block: `result.result` is
    # never read by the parser.
    assert 'It is sunny in Utrecht.' in markdown

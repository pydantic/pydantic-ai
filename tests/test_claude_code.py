"""Tests for the Claude Code `stream-json` UI event stream.

The format is not a versioned public spec, so the golden fixtures in
`tests/assets/claude_code_stream_json/` (captured from Claude Code CLI 2.1.222) are the spec:
`test_emitted_shapes_were_observed_in_real_cli_output` asserts every field name we emit was seen in
a real CLI stream. The other acceptance bar is gh-aw's own Claude log parser, vendored in
`tests/assets/ghaw_log_parser/` and run over our output by `test_ghaw_parser_reads_our_stream`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import ModelRetry, RunCancelled
from pydantic_ai.messages import (
    BinaryImage,
    CompactionPart,
    FilePart,
    FinishReason,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ThinkingPart,
    ThinkingPartDelta,
    ToolAvailabilityDeltaEvent,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
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
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.ui import NativeEvent
from pydantic_ai.ui.claude_code import NDJSON_CONTENT_TYPE, ClaudeCodeEventStream
from pydantic_ai.usage import RunUsage, UsageLimits

from .conftest import IsInt, IsSameStr, IsStr, try_import

with try_import() as anthropic_imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.anyio

StreamedChunks = AsyncIterator[DeltaThinkingCalls | DeltaToolCalls | str]


def _timestamp() -> str:
    """Matcher for the UTC millisecond timestamps the CLI stamps on its non-`stream_event` lines."""
    return IsStr(regex=r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z')


def _event_stream(*, include_partial_messages: bool = False) -> ClaudeCodeEventStream[Any, Any]:
    """An event stream with the caller-supplied fields pinned, so its lines can be snapshotted.

    The ids and timestamps it generates stay nondeterministic — they're `uuid4`s and the wall
    clock — and are asserted with matchers.
    """
    return ClaudeCodeEventStream(
        session_id='session-1',
        model='function:stream',
        cwd='/workspace',
        include_partial_messages=include_partial_messages,
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
                'uuid': IsStr(),
                'model': 'function:stream',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': IsStr(regex=r'msg_.+'),
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'hello world'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 1,
                'duration_ms': IsInt(),
                'result': 'hello world',
                'session_id': 'session-1',
                'uuid': IsStr(),
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


async def test_defaults_generate_ids_and_disclose_no_working_directory():
    """Left to its defaults, the stream generates its own session id, uuids and UTC timestamps.

    `cwd` is the one field it will not fill in: an absolute server-side path says more about the
    machine than a consumer needs, so disclosing one is opt-in.
    """

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield 'hello'

    stream = ClaudeCodeEventStream()
    lines = await _lines(stream, _native_events(Agent(FunctionModel(stream_function=stream_function)), 'hi'))

    # No working directory is claimed (consumers read empty as absent), and no model, on either the
    # init record or the message.
    assert lines == snapshot(
        [
            {
                'type': 'system',
                'subtype': 'init',
                'cwd': '',
                'session_id': (session_id := IsSameStr()),
                'tools': [],
                'mcp_servers': [],
                'permissionMode': 'default',
                'slash_commands': [],
                'output_style': 'default',
                'uuid': IsStr(),
            },
            {
                'type': 'assistant',
                'message': {
                    'id': IsStr(regex=r'msg_.+'),
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'hello'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                },
                'parent_tool_use_id': None,
                'session_id': session_id,
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 1,
                'duration_ms': IsInt(),
                'result': 'hello',
                'session_id': session_id,
                'uuid': IsStr(),
                'usage': {
                    'input_tokens': 50,
                    'output_tokens': 1,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            },
        ]
    )

    init, assistant, result = lines
    assert init['session_id'] == stream.session_id
    assert len({init['uuid'], assistant['uuid'], result['uuid']}) == 3

    # Nor on the message the partial-messages mode announces up front.
    partial = ClaudeCodeEventStream(include_partial_messages=True)
    partial_lines = await _lines(partial, _native_events(Agent(FunctionModel(stream_function=stream_function)), 'hi'))
    message_start = next(line for line in partial_lines if line['type'] == 'stream_event')
    assert 'model' not in message_start['event']['message']


async def test_adjacent_text_parts_become_one_content_block():
    """A single answer split across adjacent text parts is emitted as one block, not one per part.

    Models that interleave citations end a text part and start another mid-answer. `stream-json`
    has no way to say "this block continues", so the run of parts is buffered into one block —
    otherwise the answer arrives as fragments and `result.result` reports only the last of them.

    Fed through `transform_stream` because a `FunctionModel` cannot express adjacent text parts:
    consecutive text deltas are merged into a single part.
    """
    first = TextPart('The answer is ')
    second = TextPart('42.')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=first)
        yield PartEndEvent(index=0, part=first, next_part_kind='text')
        yield PartStartEvent(index=1, part=second, previous_part_kind='text')
        yield PartEndEvent(index=1, part=second)

    lines = await _lines(_event_stream(include_partial_messages=True), events())

    assert [line['message']['content'] for line in lines if line['type'] == 'assistant'] == snapshot(
        [[{'type': 'text', 'text': 'The answer is 42.'}]]
    )
    assert lines[-1]['result'] == snapshot('The answer is 42.')
    # One block, so one `content_block_start`/`content_block_stop` pair, both at index 0.
    assert [
        (line['event']['type'], line['event'].get('index'))
        for line in lines
        if line['type'] == 'stream_event' and line['event']['type'].startswith('content_block')
    ] == snapshot(
        [
            ('content_block_start', 0),
            ('content_block_delta', 0),
            ('content_block_delta', 0),
            ('content_block_stop', 0),
        ]
    )


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
                'uuid': IsStr(),
                'model': 'function:stream',
            },
            {
                'type': 'assistant',
                'message': {
                    'id': IsStr(regex=r'msg_.+'),
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
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'user',
                'message': {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'sunny in Utrecht'}],
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'assistant',
                'message': {
                    'id': IsStr(regex=r'msg_.+'),
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'It is sunny in Utrecht.'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 2,
                'duration_ms': IsInt(),
                'result': 'It is sunny in Utrecht.',
                'session_id': 'session-1',
                'uuid': IsStr(),
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

    assert [line['message']['content'] for line in lines if line['type'] == 'user'] == snapshot(
        [
            [
                {
                    'type': 'tool_result',
                    'tool_use_id': 'call-1',
                    'content': """\
try something else

Fix the errors and try again.\
""",
                    'is_error': True,
                }
            ]
        ]
    )
    assert lines[-1]['result'] == snapshot('Gave up.')


def _tool_result_blocks(lines: list[Any]) -> list[Any]:
    return [line['message']['content'][0] for line in lines if line['type'] == 'user']


async def test_a_tool_call_left_pending_by_an_error_is_reported_as_failed():
    """A call cut off by a failing run is closed out as a `tool_result` marked `is_error`.

    gh-aw renders a `tool_result` without `is_error` as a success, so a call the run never
    completed has to say so — otherwise a broken run reports a green tool list.
    """
    part = ToolCallPart('run_job', '{}', tool_call_id='call-1')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=part)
        yield FunctionToolCallEvent(part=part)
        raise RuntimeError('kaboom')

    lines = await _lines(_event_stream(), events())

    assert _tool_result_blocks(lines) == snapshot(
        [
            {
                'type': 'tool_result',
                'tool_use_id': 'call-1',
                'content': 'Tool execution was interrupted by an error.',
                'is_error': True,
            }
        ]
    )


async def test_a_tool_call_cut_off_mid_arguments_re_emits_none_of_its_fragments():
    """A call the run failed part-way through closes out on exactly the fragments it already streamed.

    The completing fragment `_handle_tool_call_end` emits for arguments that never arrived as JSON
    text must not fire for a call that streamed some, or a client would concatenate its truncated
    JSON twice. The block reports the `INVALID_JSON` wrapper those arguments degrade to, while the
    fragments stay verbatim what the model sent.
    """
    part = ToolCallPart('get_weather', '{"city":', tool_call_id='call-1')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=part)
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='"Utre', tool_call_id='call-1'))
        raise RuntimeError('kaboom')

    lines = await _lines(_event_stream(include_partial_messages=True), events())

    assert [
        line['event']['delta']['partial_json']
        for line in lines
        if line['type'] == 'stream_event' and line['event'].get('delta', {}).get('type') == 'input_json_delta'
    ] == snapshot(['{"city":', '"Utre'])
    assert [line['message']['content'] for line in lines if line['type'] == 'assistant'] == snapshot(
        [[{'type': 'tool_use', 'id': 'call-1', 'name': 'get_weather', 'input': {'INVALID_JSON': '{"city":'}}]]
    )


async def test_a_tool_call_left_pending_by_a_cancellation_is_not_an_error():
    """Cancelling a run interrupts its pending calls; an interrupted call is not a failed one.

    The distinction is the whole point of the `outcome` discriminator: marking these `is_error`
    would render a run the caller stopped on purpose as a run that broke.
    """
    part = ToolCallPart('run_job', '{}', tool_call_id='call-1')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=part)
        yield FunctionToolCallEvent(part=part)
        raise RunCancelled('user stopped the run', messages=[])

    lines = await _lines(_event_stream(), events())

    assert _tool_result_blocks(lines) == snapshot(
        [
            {
                'type': 'tool_result',
                'tool_use_id': 'call-1',
                'content': 'The tool call was interrupted before a result was produced.',
            }
        ]
    )
    assert lines[-1] == snapshot(
        {
            'type': 'result',
            'subtype': 'success',
            'is_error': True,
            'terminal_reason': 'cancelled',
            'num_turns': 1,
            'duration_ms': IsInt(),
            'result': 'user stopped the run',
            'session_id': 'session-1',
            'uuid': IsStr(),
        }
    )
    # No `errors`: the run was stopped, not broken, and gh-aw renders `errors` as a failure report.
    assert 'errors' not in lines[-1]


async def test_a_denied_tool_call_is_not_an_error():
    """A call refused by an approval mechanism produced no result, but nothing about it failed."""
    part = ToolCallPart('delete_everything', '{}', tool_call_id='call-1')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)
        yield FunctionToolResultEvent(
            part=ToolReturnPart(
                tool_name='delete_everything',
                content='The tool call was denied.',
                tool_call_id='call-1',
                outcome='denied',
            )
        )

    lines = await _lines(_event_stream(), events())

    assert _tool_result_blocks(lines) == snapshot(
        [{'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'The tool call was denied.'}]
    )


async def test_thinking_run():
    """Thinking parts become `thinking` blocks, carrying their signature when the model signs them.

    Both blocks belong to the one model response, so both lines carry the same message id.
    """

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield {0: DeltaThinkingPart(content='Signed thought')}
        yield {0: DeltaThinkingPart(signature='sig-1')}
        yield 'Done thinking.'

    lines = await _run_lines(Agent(FunctionModel(stream_function=stream_function)), 'think')

    assert [line['message'] for line in lines if line['type'] == 'assistant'] == snapshot(
        [
            {
                'id': (msg_id := IsSameStr()),
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'thinking', 'thinking': 'Signed thought', 'signature': 'sig-1'}],
                'stop_reason': None,
                'stop_sequence': None,
                'model': 'function:stream',
            },
            {
                'id': msg_id,
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'text', 'text': 'Done thinking.'}],
                'stop_reason': None,
                'stop_sequence': None,
                'model': 'function:stream',
            },
        ]
    )


async def test_adjacent_thinking_parts_become_one_unsigned_block():
    """Adjacent thinking parts merge like adjacent text parts, and an unsigned block omits `signature`.

    An empty signature is a value Anthropic rejects, whereas an absent one is simply unsigned.

    Fed through `transform_stream` because a `FunctionModel` merges consecutive thinking deltas into
    one part, so adjacent thinking parts can't be expressed as a stream function.
    """
    first = ThinkingPart('I should ')
    second = ThinkingPart('look it up')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=first)
        yield PartEndEvent(index=0, part=first, next_part_kind='thinking')
        yield PartStartEvent(index=1, part=second, previous_part_kind='thinking')
        yield PartEndEvent(index=1, part=second)

    lines = await _lines(_event_stream(), events())

    assert [line['message']['content'] for line in lines if line['type'] == 'assistant'] == snapshot(
        [[{'type': 'thinking', 'thinking': 'I should look it up'}]]
    )


async def test_an_error_mid_run_of_adjacent_text_parts_still_reports_the_whole_buffer():
    """A run that fails between adjacent text parts closes out one block carrying everything buffered.

    A run of adjacent parts is the only time text no `assistant` line has reported yet is being held,
    so it's the only time a failure could lose an answer the model had already produced.
    """
    first = TextPart('The answer is ')
    second = TextPart('42')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=first)
        yield PartEndEvent(index=0, part=first, next_part_kind='text')
        yield PartStartEvent(index=1, part=second, previous_part_kind='text')
        raise RuntimeError('kaboom')

    lines = await _lines(_event_stream(), events())

    assert [line['message']['content'] for line in lines if line['type'] == 'assistant'] == snapshot(
        [[{'type': 'text', 'text': 'The answer is 42'}]]
    )
    # `result.result` reports the failure rather than the answer, which the `errors` block repeats.
    assert lines[-1]['result'] == snapshot('kaboom')


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

    assert _tool_result_blocks(lines) == snapshot(
        [
            {
                'type': 'tool_result',
                'tool_use_id': 'call-1',
                'content': [{'type': 'text', 'text': '["here is the screen","See file shot."]'}],
            }
        ]
    )


async def test_file_part_produces_no_record():
    """A model-generated `FilePart` reaches the consumer as nothing at all.

    `stream-json`'s assistant content blocks are only ever `text`, `thinking` or `tool_use`, so a
    generated file has no counterpart to map onto. Inventing a private block type would only be
    understood by a consumer we also wrote, which is the opposite of speaking the format, so the
    part is dropped and this test says so out loud.

    Dropping it must not leave a hole in the block indices: a client reassembling partial-messages
    mode addresses blocks by index, so the part after the dropped one has to be the next index
    rather than the one after it, which is why the sequence below carries a trailing text part.

    Fed through `transform_stream` because a model that emits a `FilePart` mid-response cannot be
    expressed as a `FunctionModel` stream function. `FilePart` gets no `PartEndEvent` — only parts
    that have deltas are ended — so the stream below is what a real response would produce.
    """
    text = TextPart('Here is the chart')
    caption = TextPart('Prices are up.')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=text)
        yield PartEndEvent(index=0, part=text, next_part_kind='file')
        yield PartStartEvent(
            index=1,
            part=FilePart(content=BinaryImage(data=b'fake-png', media_type='image/png')),
            previous_part_kind='text',
        )
        yield PartStartEvent(index=2, part=caption, previous_part_kind='file')
        yield PartEndEvent(index=2, part=caption)

    lines = await _lines(_event_stream(include_partial_messages=True), events())

    assert [line['message']['content'] for line in lines if line['type'] == 'assistant'] == snapshot(
        [[{'type': 'text', 'text': 'Here is the chart'}], [{'type': 'text', 'text': 'Prices are up.'}]]
    )
    assert [
        (line['event']['type'], line['event']['index'])
        for line in lines
        if line['type'] == 'stream_event' and line['event']['type'].startswith('content_block_')
    ] == snapshot(
        [
            ('content_block_start', 0),
            ('content_block_delta', 0),
            ('content_block_stop', 0),
            ('content_block_start', 1),
            ('content_block_delta', 1),
            ('content_block_stop', 1),
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
    # `result.result` reports the final *text*, and a structured output produced none: the answer is
    # the tool call above. gh-aw never reads the field, rendering the assistant lines instead.
    assert lines[-1]['result'] == snapshot('')


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
            'duration_ms': IsInt(),
            'result': 'Done.',
            'session_id': 'session-1',
            'uuid': IsStr(),
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


@pytest.mark.parametrize(
    'finish_reason,stop_reason',
    [
        ('stop', 'end_turn'),
        ('length', 'max_tokens'),
        ('tool_call', 'tool_use'),
        ('content_filter', 'refusal'),
        ('error', None),
    ],
)
async def test_every_finish_reason_maps_onto_anthropics_vocabulary(
    finish_reason: FinishReason, stop_reason: str | None
):
    """Each Pydantic AI finish reason reports the `stop_reason` Anthropic's own streams use.

    `'error'` is the one with no counterpart: the key is omitted rather than reported as `null`,
    because an errored run says so through `is_error` and `terminal_reason` instead.
    """

    async def events() -> AsyncIterator[NativeEvent]:
        result = AgentRunResult(output='Done.')
        result._state.message_history = [ModelResponse(parts=[], finish_reason=finish_reason)]  # pyright: ignore[reportPrivateUsage]
        yield AgentRunResultEvent(result=result)

    lines = await _lines(_event_stream(), events())

    assert lines[-1].get('stop_reason') == stop_reason
    assert ('stop_reason' in lines[-1]) is (stop_reason is not None)


async def test_a_run_halted_on_deferred_tools_is_not_reported_as_completed():
    """A run that stops awaiting approvals says so, rather than reading as a finished success.

    gh-aw renders the terminal `result` line as its whole Information section, so a halt reported as
    `'completed'` claims the agent answered when it is in fact waiting on the caller. Nothing failed
    either, so it stays `is_error: false` — the pause is named by `terminal_reason` alone.
    """

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield {0: DeltaToolCall(name='delete_everything', json_args='{}', tool_call_id='call-1')}

    agent = Agent(FunctionModel(stream_function=stream_function), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def delete_everything() -> str:
        return 'gone'  # pragma: no cover

    lines = await _run_lines(agent, 'delete it all')

    assert lines[-1] == snapshot(
        {
            'type': 'result',
            'subtype': 'success',
            'is_error': False,
            'terminal_reason': 'deferred_tool_requests',
            'num_turns': 1,
            'duration_ms': IsInt(),
            'result': '',
            'session_id': 'session-1',
            'uuid': IsStr(),
            'usage': {
                'input_tokens': 50,
                'output_tokens': 1,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
            },
        }
    )


async def test_tool_availability_delta_produces_no_record():
    """Tools becoming available mid-run reach the consumer as nothing at all.

    The siblings surface the delta so a live UI can refresh its tool list. `stream-json` has no
    such consumer: it's read by log parsers over a closed record vocabulary, and the only tool list
    the format has is the `init` record's, written before the run starts and never revised.
    """
    part = ToolCallPart('load_tools', '{}', tool_call_id='call-1')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)
        yield ToolAvailabilityDeltaEvent(part=ToolAvailabilityDeltaPart(tools_added=['deploy'], tool_call_id='call-1'))

    lines = await _lines(_event_stream(), events())

    assert [line['type'] for line in lines] == snapshot(['system', 'assistant', 'result'])


def _reassembled_tool_inputs(lines: list[Any]) -> list[tuple[Any, Any]]:
    """Pair each `tool_use` block's `input` with the JSON its `input_json_delta` fragments spell out.

    A client rebuilding a call's arguments from partial-messages mode only ever sees the fragments,
    so their concatenation has to parse back to exactly the whole-block `input`. A call with no
    arguments emits no fragments at all, which is how the CLI's own streams read.
    """
    pairs: list[tuple[Any, Any]] = []
    fragments: list[str] = []
    for line in lines:
        if line['type'] == 'stream_event':
            delta = line['event'].get('delta')
            if is_str_dict(delta) and delta.get('type') == 'input_json_delta':
                fragments.append(delta['partial_json'])
        elif line['type'] == 'assistant':
            block = line['message']['content'][0]
            if block['type'] == 'tool_use':
                pairs.append((block['input'], json.loads(''.join(fragments)) if fragments else {}))
            fragments = []
    return pairs


def _reassembled_thinking_signatures(lines: list[Any]) -> list[tuple[Any, Any]]:
    """Pair each `thinking` block's `signature` with what its `signature_delta` fragments spell out.

    The twin of `_reassembled_tool_inputs` for the other field partial-messages mode streams: a client
    only ever sees the fragments, so what they spell out has to be exactly the whole-block signature.
    An unsigned block emits none at all.
    """
    pairs: list[tuple[Any, Any]] = []
    fragments: list[str] = []
    for line in lines:
        if line['type'] == 'stream_event':
            delta = line['event'].get('delta')
            if is_str_dict(delta) and delta.get('type') == 'signature_delta':
                fragments.append(delta['signature'])
        elif line['type'] == 'assistant':
            block = line['message']['content'][0]
            if block['type'] == 'thinking':
                pairs.append((block.get('signature', ''), ''.join(fragments)))
            fragments = []
    return pairs


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
    # The fragments a client concatenates rebuild exactly the arguments the whole block reports.
    assert _reassembled_tool_inputs(lines) == snapshot([({'city': 'Utrecht'}, {'city': 'Utrecht'})])
    assert _reassembled_thinking_signatures(lines) == snapshot([('sig-1', 'sig-1')])
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
                'uuid': IsStr(),
                'model': 'function:stream',
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'message_start',
                    'message': {
                        'id': (first_message_id := IsSameStr()),
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
                'uuid': IsStr(),
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
                'uuid': IsStr(),
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
                'uuid': IsStr(),
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
                'uuid': IsStr(),
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
                'uuid': IsStr(),
            },
            {
                'type': 'assistant',
                'message': {
                    'id': first_message_id,
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'thinking', 'thinking': 'I should look it up', 'signature': 'sig-1'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_stop', 'index': 0},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
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
                'uuid': IsStr(),
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
                'uuid': IsStr(),
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
                'uuid': IsStr(),
            },
            {
                'type': 'assistant',
                'message': {
                    'id': first_message_id,
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
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_stop', 'index': 1},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'message_delta', 'delta': {'stop_reason': None, 'stop_sequence': None}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'message_stop'},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'user',
                'message': {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 'call-1', 'content': 'sunny in Utrecht'}],
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'stream_event',
                'event': {
                    'type': 'message_start',
                    'message': {
                        'id': (second_message_id := IsSameStr()),
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
                'uuid': IsStr(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'It is '}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'sunny.'}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'assistant',
                'message': {
                    'id': second_message_id,
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'text', 'text': 'It is sunny.'}],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'model': 'function:stream',
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'content_block_stop', 'index': 0},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'message_delta', 'delta': {'stop_reason': None, 'stop_sequence': None}},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'stream_event',
                'event': {'type': 'message_stop'},
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 2,
                'duration_ms': IsInt(),
                'result': 'It is sunny.',
                'session_id': 'session-1',
                'uuid': IsStr(),
                'usage': {
                    'input_tokens': 100,
                    'output_tokens': 15,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            },
        ]
    )


async def test_dict_tool_args_are_emitted_as_one_complete_fragment():
    """`dict` args reach the client as a single fragment carrying the whole encoded object.

    Unlike streamed `str` args, a `dict` delta is a merge rather than a continuation, so emitting
    one fragment per delta would concatenate into invalid JSON. The value is encoded the way
    Pydantic AI encodes tool arguments, which `json.dumps` alone cannot do — a `datetime` argument
    would raise instead of reaching the consumer.
    """
    args = {'when': datetime(2026, 8, 6, 12, tzinfo=timezone.utc)}
    start = NativeToolCallPart('log_at', args, tool_call_id='call-1')
    end = NativeToolCallPart('log_at', {**args, 'level': 'info'}, tool_call_id='call-1')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=start)
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'level': 'info'}, tool_call_id='call-1'))
        yield PartEndEvent(index=0, part=end)

    lines = await _lines(_event_stream(include_partial_messages=True), events())

    assert [
        line['event']['delta']['partial_json']
        for line in lines
        if line['type'] == 'stream_event' and line['event'].get('delta', {}).get('type') == 'input_json_delta'
    ] == snapshot(['{"when":"2026-08-06T12:00:00Z","level":"info"}'])
    assert _reassembled_tool_inputs(lines) == snapshot(
        [
            (
                {'when': '2026-08-06T12:00:00Z', 'level': 'info'},
                {'when': '2026-08-06T12:00:00Z', 'level': 'info'},
            )
        ]
    )


async def test_a_delta_carrying_no_arguments_emits_no_fragment():
    """A chunk that carries no arguments contributes nothing to the JSON a client reassembles.

    Providers that stream a tool call across chunks routinely send one with no arguments at all;
    emitting a fragment for it would inject a literal `null` into the middle of the JSON.
    """

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='noop', json_args='{}', tool_call_id='call-1')}
            yield {0: DeltaToolCall(tool_call_id='call-1')}
        else:
            yield 'Done.'

    agent = Agent(FunctionModel(stream_function=stream_function))

    @agent.tool_plain
    def noop() -> str:
        return 'ok'

    lines = await _run_lines(agent, 'go', include_partial_messages=True)

    assert [
        line['event']['delta']['partial_json']
        for line in lines
        if line['type'] == 'stream_event' and line['event'].get('delta', {}).get('type') == 'input_json_delta'
    ] == snapshot(['{}'])
    assert _reassembled_tool_inputs(lines) == snapshot([({}, {})])


async def test_a_thinking_block_emits_its_whole_signature_as_one_final_delta():
    """A block's signature is streamed once, as the last delta before the `assistant` line closing it.

    Upstream a signature replaces the one before it rather than extending it, so forwarding each
    `ThinkingPartDelta.signature_delta` would leave a client concatenating fragments that spell out
    something no block ever claimed: a run of adjacent signed parts would rebuild both signatures for
    a block signed with only the last, and a part re-signed mid-stream would rebuild the draft plus
    the final one. One `signature_delta` per block is also how the CLI's own streams read.

    Fed through `transform_stream` because a `FunctionModel` merges consecutive thinking deltas into
    one part, so adjacent thinking parts can't be expressed as a stream function.
    """
    first = ThinkingPart('step one ', signature='sig-a')
    second = ThinkingPart('step two', signature='sig-b')
    answer = TextPart('Done.')
    resigned = ThinkingPart('rethinking', signature='sig-final')

    async def events() -> AsyncIterator[NativeEvent]:
        yield PartStartEvent(index=0, part=first)
        yield PartDeltaEvent(index=0, delta=ThinkingPartDelta(signature_delta='sig-a'))
        yield PartEndEvent(index=0, part=first, next_part_kind='thinking')
        yield PartStartEvent(index=1, part=second, previous_part_kind='thinking')
        yield PartDeltaEvent(index=1, delta=ThinkingPartDelta(signature_delta='sig-b'))
        yield PartEndEvent(index=1, part=second)
        yield PartStartEvent(index=2, part=answer)
        yield PartEndEvent(index=2, part=answer)
        yield PartStartEvent(index=3, part=ThinkingPart('rethinking'))
        yield PartDeltaEvent(index=3, delta=ThinkingPartDelta(signature_delta='sig-draft'))
        yield PartDeltaEvent(index=3, delta=ThinkingPartDelta(signature_delta='sig-final'))
        yield PartEndEvent(index=3, part=resigned)

    lines = await _lines(_event_stream(include_partial_messages=True), events())

    assert _reassembled_thinking_signatures(lines) == snapshot([('sig-b', 'sig-b'), ('sig-final', 'sig-final')])
    # Each block's signature comes after its content, which is where the last delta before the
    # `assistant` record puts it.
    assert [
        line['event']['delta']['type']
        for line in lines
        if line['type'] == 'stream_event' and line['event']['type'] == 'content_block_delta'
    ] == snapshot(
        ['thinking_delta', 'thinking_delta', 'signature_delta', 'text_delta', 'thinking_delta', 'signature_delta']
    )


async def test_a_signature_carried_by_the_first_chunk_still_reaches_the_stream():
    """A signature the model sends up front is streamed too, though no delta ever carries it.

    Anthropic's redacted thinking arrives this way — the signature is set as the part is created, so
    a stream forwarding only `signature_delta`s would report the block's signature nowhere.
    """

    async def stream_function(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield {0: DeltaThinkingPart(content='reasoning', signature='sig-at-start')}
        yield 'Done.'

    lines = await _run_lines(
        Agent(FunctionModel(stream_function=stream_function)), 'think', include_partial_messages=True
    )

    assert _reassembled_thinking_signatures(lines) == snapshot([('sig-at-start', 'sig-at-start')])


@pytest.mark.vcr()
@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
async def test_a_real_anthropic_run_streams_thinking_and_a_tool_call(
    allow_model_requests: None, anthropic_api_key: str
):
    """The reassembly invariants hold on a real provider's events, not just hand-authored ones.

    Every other test in this file drives the stream from `FunctionModel` or a hand-written event
    sequence, which pins the mapping but takes the shape of the input on trust. This one runs
    extended thinking and a tool call against Anthropic, so the signature that reaches a
    `signature_delta` and the arguments a client rebuilds from `input_json_delta` fragments are the
    ones a provider actually produced.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    settings: AnthropicModelSettings = {'anthropic_thinking': {'type': 'enabled', 'budget_tokens': 1024}}
    agent = Agent(model, model_settings=settings)

    @agent.tool_plain
    def get_temperature(city: str) -> str:
        return '18°C'

    stream = ClaudeCodeEventStream[Any, Any](
        session_id='session-1', model='anthropic:claude-sonnet-4-5', include_partial_messages=True
    )
    lines = await _lines(stream, _native_events(agent, 'How warm is it in Utrecht? Use the tool, then answer.'))

    signature = next(
        block['signature']
        for line in lines
        if line['type'] == 'assistant'
        for block in line['message']['content']
        if block['type'] == 'thinking'
    )
    # A real provider signature — long, opaque, and reaching the client whole through exactly one
    # `signature_delta`, which is the invariant the hand-authored tests can only assume.
    assert signature == IsStr(min_length=100)
    assert _reassembled_thinking_signatures(lines) == [(signature, signature)]
    # Anthropic streams tool arguments as JSON text, so the fragments are the provider's own and
    # their concatenation still parses back to exactly the whole block's `input`.
    assert _reassembled_tool_inputs(lines) == snapshot([({'city': 'Utrecht'}, {'city': 'Utrecht'})])
    assert [
        (line['event']['type'], line['event'].get('content_block', line['event'].get('delta', {})).get('type'))
        if line['type'] == 'stream_event'
        else (line['type'], line.get('subtype'))
        for line in lines
    ] == snapshot(
        [
            ('system', 'init'),
            ('message_start', None),
            ('content_block_start', 'thinking'),
            ('content_block_delta', 'thinking_delta'),
            ('content_block_delta', 'thinking_delta'),
            ('content_block_delta', 'signature_delta'),
            ('assistant', None),
            ('content_block_stop', None),
            ('content_block_start', 'tool_use'),
            ('content_block_delta', 'input_json_delta'),
            ('content_block_delta', 'input_json_delta'),
            ('content_block_delta', 'input_json_delta'),
            ('assistant', None),
            ('content_block_stop', None),
            ('message_delta', None),
            ('message_stop', None),
            ('user', None),
            ('message_start', None),
            ('content_block_start', 'text'),
            ('content_block_delta', 'text_delta'),
            ('content_block_delta', 'text_delta'),
            ('content_block_delta', 'text_delta'),
            ('content_block_delta', 'text_delta'),
            ('content_block_delta', 'text_delta'),
            ('content_block_delta', 'text_delta'),
            ('assistant', None),
            ('content_block_stop', None),
            ('message_delta', None),
            ('message_stop', None),
            ('result', 'success'),
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
                'uuid': IsStr(),
                'model': 'function:stream',
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': True,
                'terminal_reason': 'error',
                'num_turns': 0,
                'duration_ms': IsInt(),
                'result': 'the model exploded',
                'session_id': 'session-1',
                'uuid': IsStr(),
                'errors': ['the model exploded'],
            },
        ]
    )


async def test_reusing_an_instance_reports_each_run_on_its_own_terms():
    """A second run through the same stream reports that run, not a residue of the first.

    Everything the stream accumulates — the failure, the turn count, the answer — is per-run state,
    and `num_turns` is what gh-aw compares against its max-turns budget.
    """

    async def failing(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        raise RuntimeError('the model exploded')
        yield  # Make this an async generator

    async def succeeding(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        yield 'all good'

    stream = _event_stream()
    first = await _lines(stream, _native_events(Agent(FunctionModel(stream_function=failing)), 'go'))
    second = await _lines(stream, _native_events(Agent(FunctionModel(stream_function=succeeding)), 'go'))

    assert (first[-1]['is_error'], first[-1]['num_turns']) == snapshot((True, 0))
    assert second[-1] == snapshot(
        {
            'type': 'result',
            'subtype': 'success',
            'is_error': False,
            'terminal_reason': 'completed',
            'num_turns': 1,
            'duration_ms': IsInt(),
            'result': 'all good',
            'session_id': 'session-1',
            'uuid': IsStr(),
            'usage': {
                'input_tokens': 50,
                'output_tokens': 2,
                'cache_creation_input_tokens': 0,
                'cache_read_input_tokens': 0,
            },
        }
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
                'uuid': IsStr(),
                'model': 'function:stream',
            },
            {
                'type': 'system',
                'subtype': 'compact_boundary',
                'compact_metadata': {'trigger': 'auto'},
                'session_id': 'session-1',
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'assistant',
                'message': {
                    'id': IsStr(regex=r'msg_.+'),
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
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'user',
                'message': {
                    'role': 'user',
                    'content': [{'type': 'tool_result', 'tool_use_id': 'ws-1', 'content': 'one result'}],
                },
                'parent_tool_use_id': None,
                'session_id': 'session-1',
                'uuid': IsStr(),
                'timestamp': _timestamp(),
            },
            {
                'type': 'result',
                'subtype': 'success',
                'is_error': False,
                'terminal_reason': 'completed',
                'num_turns': 1,
                'duration_ms': IsInt(),
                'result': '',
                'session_id': 'session-1',
                'uuid': IsStr(),
            },
        ]
    )


def _shape(value: dict[str, Any], label: str, shapes: dict[str, set[str]]) -> None:
    """Record the key set of every record, message, block, event and delta object in `value`."""
    shapes.setdefault(label, set()).update(value)
    for key, child in value.items():
        if key in ('message', 'event', 'delta', 'content_block'):
            assert is_str_dict(child)
            child_type = child.get('type')
            _shape(child, f'{key}:{child_type}' if child_type else key, shapes)
        elif key == 'content' and not isinstance(child, str):
            for block in child:
                assert is_str_dict(block)
                _shape(block, f'block:{block.get("type")}', shapes)


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
# message types. `result.errors` is the same kind of divergence: the CLI's own `success`-subtype
# failures carry no such key, but gh-aw's parser reads `lastEntry.errors` (in
# `log_parser_shared.cjs`) to render its Errors block, so a failed run reports one.
# Everything else we emit has to have been seen in a real stream.
UNOBSERVED_SHAPES = {'record:system:compact_boundary'}
UNOBSERVED_KEYS = {('block:tool_result', 'is_error'), ('record:result:success', 'errors')}


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

    async def failing(messages: list[ModelMessage], agent_info: AgentInfo) -> StreamedChunks:
        raise RuntimeError('the model exploded')
        yield  # Make this an async generator

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
    emitted += await _run_lines(Agent(FunctionModel(stream_function=failing)), 'go')
    emitted_shapes = _stream_shapes(emitted)

    unexpected = {
        (label, key)
        for label, keys in emitted_shapes.items()
        if label not in UNOBSERVED_SHAPES
        for key in keys - fixture_shapes.get(label, set())
    }
    assert unexpected == UNOBSERVED_KEYS
    # And the objects themselves: every kind of record, block and event we emit was seen in the CLI's
    # own output, so no consumer meets a shape the format doesn't have.
    assert set(emitted_shapes) - set(fixture_shapes) == UNOBSERVED_SHAPES


async def test_ghaw_parser_reads_our_stream(assets_path: Path):
    """gh-aw's own Claude log parser extracts turns, tokens and tool calls from our stream.

    This is the acceptance bar for "drop-in compatible": if their parser reads our output, an agent
    emitting it can run as gh-aw's `engine: claude` and keep the step-summary rendering and token
    metrics that third-party engines don't get.
    """
    if shutil.which('node') is None:  # pragma: lax no cover
        message = 'node is required to run the vendored gh-aw parser, the only compatibility oracle in the suite'
        # Skipping it on CI would be a green run that checked nothing, so there it's a failure.
        if os.getenv('CI'):
            pytest.fail(message)
        pytest.skip(message)

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

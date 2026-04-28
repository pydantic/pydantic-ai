"""Tests for the OpenAI Responses protocol integration."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from dirty_equals import IsFloat
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelMessage
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from .conftest import IsDatetime, IsSameStr, try_import

with try_import() as imports_successful:
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from pydantic_ai.ui.responses import ResponsesAdapter
    from pydantic_ai.ui.responses._event_stream import ResponsesEventStream

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai or starlette not installed'),
    pytest.mark.filterwarnings(
        'ignore:State was provided but `deps` of type `NoneType` does not implement the `StateHandler` protocol:UserWarning'
    ),
    pytest.mark.filterwarnings('ignore:Frontend system prompts were provided:UserWarning'),
]


async def _comprehensive_stream(
    messages: list[ModelMessage], agent_info: AgentInfo
) -> AsyncIterator[DeltaToolCalls | str]:
    """Stream that exercises text output and a frontend-tool call."""
    if not any(isinstance(part, ToolReturnPart) for m in messages for part in m.parts):
        yield 'Hello '
        yield 'there!'
        yield {0: DeltaToolCall(name='get_weather', json_args='{"location":"Paris"}', tool_call_id='call_1')}
    else:
        yield 'Final answer.'


def _comprehensive_request_body() -> bytes:
    """Comprehensive Responses API request: instructions + multi-part input + frontend tool + metadata."""
    return json.dumps(
        {
            'model': 'test',
            'stream': True,
            'instructions': 'Be concise.',
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_text', 'text': "What's the weather in Paris?"},
                    ],
                },
                {
                    'type': 'function_call',
                    'call_id': 'call_1',
                    'name': 'get_weather',
                    'arguments': '{"location":"Paris"}',
                },
                {
                    'type': 'function_call_output',
                    'call_id': 'call_1',
                    'output': '{"temperature":"sunny"}',
                },
            ],
            'tools': [
                {
                    'type': 'function',
                    'name': 'get_weather',
                    'description': 'Get the weather for a location',
                    'parameters': {
                        'type': 'object',
                        'properties': {'location': {'type': 'string'}},
                        'required': ['location'],
                    },
                    'strict': True,
                },
            ],
            'metadata': {'session_id': 'abc'},
            'tool_choice': 'auto',
            'parallel_tool_calls': True,
        }
    ).encode()


async def _collect_events(adapter: ResponsesAdapter[Any, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    async for chunk in adapter.encode_stream(adapter.run_stream()):
        if chunk.startswith('event: done'):
            continue
        for line in chunk.strip().split('\n'):
            if line.startswith('data: '):
                events.append(json.loads(line.removeprefix('data: ')))
    return events


async def test_comprehensive_run() -> None:
    """End-to-end Responses flow: parses request, runs agent, emits Responses-shaped SSE events."""
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    run_input = ResponsesAdapter.build_run_input(_comprehensive_request_body())
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)

    response_id = IsSameStr()
    item_id = IsSameStr()
    final_item_id = IsSameStr()

    assert events == snapshot(
        [
            {
                'type': 'response.created',
                'sequence_number': 0,
                'response': {
                    'id': response_id,
                    'created_at': IsFloat(),
                    'instructions': 'Be concise.',
                    'model': 'test',
                    'object': 'response',
                    'output': [],
                    'parallel_tool_calls': True,
                    'tool_choice': 'auto',
                    'tools': [
                        {
                            'type': 'function',
                            'name': 'get_weather',
                            'description': 'Get the weather for a location',
                            'parameters': {
                                'type': 'object',
                                'properties': {'location': {'type': 'string'}},
                                'required': ['location'],
                            },
                            'strict': True,
                        }
                    ],
                    'status': 'in_progress',
                    'usage': {
                        'input_tokens': 0,
                        'input_tokens_details': {'cached_tokens': 0},
                        'output_tokens': 0,
                        'output_tokens_details': {'reasoning_tokens': 0},
                        'total_tokens': 0,
                    },
                },
            },
            {
                'type': 'response.output_item.added',
                'sequence_number': 1,
                'output_index': 0,
                'item': {
                    'id': item_id,
                    'content': [],
                    'role': 'assistant',
                    'status': 'in_progress',
                    'type': 'message',
                },
            },
            {
                'type': 'response.content_part.added',
                'sequence_number': 2,
                'output_index': 0,
                'content_index': 0,
                'item_id': item_id,
                'part': {'annotations': [], 'text': '', 'type': 'output_text'},
            },
            {
                'content_index': 0,
                'delta': 'Final answer.',
                'item_id': final_item_id,
                'logprobs': [],
                'output_index': 0,
                'sequence_number': 3,
                'type': 'response.output_text.delta',
            },
            {
                'content_index': 0,
                'item_id': final_item_id,
                'logprobs': [],
                'output_index': 0,
                'sequence_number': 4,
                'text': 'Final answer.',
                'type': 'response.output_text.done',
            },
            {
                'content_index': 0,
                'item_id': final_item_id,
                'output_index': 0,
                'part': {'annotations': [], 'text': 'Final answer.', 'type': 'output_text'},
                'sequence_number': 5,
                'type': 'response.content_part.done',
            },
            {
                'item': {
                    'id': final_item_id,
                    'content': [{'annotations': [], 'text': 'Final answer.', 'type': 'output_text'}],
                    'role': 'assistant',
                    'status': 'completed',
                    'type': 'message',
                },
                'output_index': 0,
                'sequence_number': 6,
                'type': 'response.output_item.done',
            },
            {
                'response': {
                    'id': response_id,
                    'created_at': IsFloat(),
                    'instructions': 'Be concise.',
                    'model': 'test',
                    'object': 'response',
                    'output': [],
                    'parallel_tool_calls': True,
                    'tool_choice': 'auto',
                    'tools': [
                        {
                            'name': 'get_weather',
                            'parameters': {
                                'type': 'object',
                                'properties': {'location': {'type': 'string'}},
                                'required': ['location'],
                            },
                            'strict': True,
                            'type': 'function',
                            'description': 'Get the weather for a location',
                        }
                    ],
                    'status': 'completed',
                    'usage': {
                        'input_tokens': 50,
                        'input_tokens_details': {'cached_tokens': 0},
                        'output_tokens': 3,
                        'output_tokens_details': {'reasoning_tokens': 0},
                        'total_tokens': 53,
                    },
                },
                'sequence_number': 7,
                'type': 'response.completed',
            },
        ]
    )


async def test_load_messages_round_trip() -> None:
    """Loaded messages: instructions become system prompt; tool name is recovered from preceding `function_call`."""
    run_input = ResponsesAdapter.build_run_input(_comprehensive_request_body())
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    assert adapter.messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Be concise.', timestamp=IsDatetime()),
                    UserPromptPart(content=["What's the weather in Paris?"], timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='get_weather',
                        args='{"location":"Paris"}',
                        tool_call_id='call_1',
                    )
                ],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_weather',
                        content='{"temperature":"sunny"}',
                        tool_call_id='call_1',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
        ]
    )


async def test_streams_tool_call_as_function_call_output_item() -> None:
    """Frontend-tool calls emerge as `function_call` output items (the SDK literal must match)."""
    body = json.dumps(
        {
            'model': 'test',
            'stream': True,
            'input': 'What is the weather in Paris?',
            'tools': [
                {
                    'type': 'function',
                    'name': 'get_weather',
                    'parameters': {'type': 'object', 'properties': {}},
                }
            ],
        }
    ).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)

    function_call_items = [e['item'] for e in events if e.get('item', {}).get('type') == 'function_call']
    assert function_call_items
    assert function_call_items[0] == snapshot(
        {
            'arguments': '{"location":"Paris"}',
            'call_id': 'call_1',
            'name': 'get_weather',
            'type': 'function_call',
        }
    )


async def test_load_messages_function_output_without_call_raises() -> None:
    """A function_call_output that doesn't reference a preceding function_call must raise."""
    body = json.dumps(
        {
            'model': 'test',
            'stream': True,
            'input': [
                {'type': 'function_call_output', 'call_id': 'orphan', 'output': 'x'},
            ],
        }
    ).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)
    with pytest.raises(ValueError, match='unknown call_id'):
        _ = adapter.messages


async def test_state_from_metadata() -> None:
    """Request `metadata` is exposed as frontend state."""
    body = json.dumps({'model': 'test', 'stream': True, 'input': 'x', 'metadata': {'k': 'v'}}).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)
    assert adapter.state == {'k': 'v'}


async def test_user_image_data_uri() -> None:
    """A data-URI `input_image` is parsed into BinaryContent (not silently dropped)."""
    body = json.dumps(
        {
            'model': 'test',
            'stream': True,
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_image', 'image_url': 'data:image/png;base64,aGVsbG8='},
                    ],
                }
            ],
        }
    ).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    [request] = adapter.messages
    assert isinstance(request, ModelRequest)
    [user_part] = request.parts
    assert isinstance(user_part, UserPromptPart)
    [image] = user_part.content
    assert isinstance(image, BinaryContent)
    assert image.media_type == 'image/png'


async def test_user_image_url() -> None:
    """A non-data-URI `input_image` is parsed as ImageUrl."""
    body = json.dumps(
        {
            'model': 'test',
            'stream': True,
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [
                        {'type': 'input_image', 'image_url': 'https://example.com/cat.png'},
                    ],
                }
            ],
        }
    ).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    [request] = adapter.messages
    assert isinstance(request, ModelRequest)
    [user_part] = request.parts
    assert isinstance(user_part, UserPromptPart)
    [image] = user_part.content
    assert isinstance(image, ImageUrl)
    assert image.url == 'https://example.com/cat.png'


def test_to_responses_returns_app() -> None:
    """`Agent.to_responses()` returns a Starlette app mounting `POST /v1/responses`."""
    agent = Agent('test')
    app = agent.to_responses()
    assert isinstance(app, Starlette)
    paths_methods = {(getattr(r, 'path', ''), frozenset(getattr(r, 'methods', None) or [])) for r in app.routes}
    assert ('/v1/responses', frozenset({'POST'})) in paths_methods


def test_to_responses_app_streams() -> None:
    """The mounted endpoint accepts a Responses request and streams SSE chunks."""
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    app = agent.to_responses()

    with TestClient(app) as client:
        with client.stream(
            'POST',
            '/v1/responses',
            content=_comprehensive_request_body(),
            headers={'content-type': 'application/json'},
        ) as resp:
            assert resp.status_code == 200
            assert 'text/event-stream' in resp.headers['content-type']
            body = ''.join(resp.iter_text())

    event_types = re.findall(r'^event: (\S+)$', body, re.MULTILINE)
    assert event_types[0] == 'response.created'
    assert event_types[-2] == 'response.completed'
    assert event_types[-1] == 'done'
    assert 'response.output_text.delta' in event_types
    assert any(t.startswith('response.output_item') for t in event_types)


def test_build_run_input_rejects_non_object() -> None:
    """A request body that isn't a JSON object is rejected with a clear error."""
    with pytest.raises(ValueError, match='must be a JSON object'):
        ResponsesAdapter.build_run_input(b'[1, 2, 3]')


def test_load_messages_classmethod() -> None:
    """`ResponsesAdapter.load_messages` parses a flat sequence of input items into model messages."""
    items: list[dict[str, Any]] = [
        {'type': 'message', 'role': 'user', 'content': 'hi'},
        {'type': 'function_call', 'call_id': 'c1', 'name': 'echo', 'arguments': '{"x":1}'},
        {'type': 'function_call_output', 'call_id': 'c1', 'output': 'ok'},
    ]
    messages = ResponsesAdapter.load_messages(items)  # pyright: ignore[reportArgumentType]
    assert messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='hi', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[ToolCallPart(tool_name='echo', args='{"x":1}', tool_call_id='c1')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='echo', content='ok', tool_call_id='c1', timestamp=IsDatetime())]
            ),
        ]
    )


def _adapter_for(body: dict[str, Any]) -> ResponsesAdapter[None, str]:
    run_input = ResponsesAdapter.build_run_input(json.dumps(body).encode())
    agent = Agent(model=FunctionModel(stream_function=_comprehensive_stream))
    return ResponsesAdapter[None, str](agent=agent, run_input=run_input)


def test_load_messages_kitchen_sink() -> None:
    """One pass exercises every parsing branch: malformed entries skipped, well-formed ones parsed."""
    body: dict[str, Any] = {
        'model': 'test',
        'stream': True,
        'input': [
            'not_a_dict_item_skipped',
            {'type': 'message', 'role': 'system', 'content': 'sys-prompt'},
            {'type': 'message', 'role': 'developer', 'content': [{'type': 'text', 'text': 'dev-prompt'}]},
            {'type': 'message', 'role': 'user', 'content': 'bare-string'},
            {
                'type': 'message',
                'role': 'user',
                'content': [
                    'not_a_dict_part_skipped',
                    {'type': 'input_text', 'text': 'hello'},
                    {'type': 'input_image', 'image_url': 123},
                    {'type': 'input_image', 'image_url': 'https://example.com/cat.png'},
                    {'type': 'input_file', 'file_data': 'data:application/pdf;base64,JVBERg=='},
                    {'type': 'input_file', 'file_id': 'file-abc'},
                    {'type': 'unknown_part', 'value': 'x'},
                ],
            },
            {'role': 'user', 'content': 'implicit-message-no-type'},
            {'type': 'message', 'role': 'user', 'content': []},
            {'type': 'message', 'role': 'user', 'content': [{'type': 'unknown_part'}]},
            {'type': 'function_call', 'arguments': '{}'},
            {'type': 'function_call', 'name': 'echo', 'id': 'fc_id', 'arguments': '{"x":1}'},
            {'type': 'function_call_output', 'call_id': 'fc_id', 'output': 'ok'},
            {'type': 'function_call_output', 'call_id': 42, 'output': 'ignored'},
            {
                'type': 'message',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'assistant-text'}],
            },
        ],
    }
    adapter = _adapter_for(body)
    assert adapter.messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='sys-prompt', timestamp=IsDatetime()),
                    SystemPromptPart(content='dev-prompt', timestamp=IsDatetime()),
                    UserPromptPart(content='bare-string', timestamp=IsDatetime()),
                    UserPromptPart(
                        content=[
                            'hello',
                            ImageUrl(url='https://example.com/cat.png'),
                            BinaryContent(data=b'%PDF', media_type='application/pdf'),
                        ],
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(content='implicit-message-no-type', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='echo', args='{"x":1}', tool_call_id='fc_id')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='echo', content='ok', tool_call_id='fc_id', timestamp=IsDatetime())]
            ),
            ModelResponse(parts=[TextPart(content='assistant-text')], timestamp=IsDatetime()),
        ]
    )


def test_no_input_yields_no_messages() -> None:
    """A request with no `input` field produces no messages and no toolset."""
    adapter = _adapter_for({'model': 'test', 'stream': True})
    assert adapter.messages == []
    assert adapter.toolset is None


def test_malformed_tools_array_filters_out_invalid_entries() -> None:
    """Tool entries that aren't a `function`-typed dict with a string name are silently dropped."""
    body: dict[str, Any] = {
        'model': 'test',
        'stream': True,
        'input': 'hi',
        'tools': [
            'not_a_dict_skipped',
            {'type': 'web_search'},
            {'type': 'function'},
            {'type': 'function', 'name': 42},
            {'type': 'function', 'name': 'good_tool', 'parameters': 'not_a_dict'},
        ],
    }
    from pydantic_ai.toolsets import ExternalToolset

    adapter = _adapter_for(body)
    toolset = adapter.toolset
    assert isinstance(toolset, ExternalToolset)
    assert [t.name for t in toolset.tool_defs] == ['good_tool']
    assert toolset.tool_defs[0].parameters_json_schema == {}


@dataclass
class _CountingDeps:
    state: dict[str, Any] | None = None
    counter: int = 0


def test_state_handler_deps_are_replaced_per_request() -> None:
    """When `deps` implements `StateHandler`, each request gets a fresh copy via `replace`."""
    captured: list[int] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        captured.append(id(info))
        yield 'ok'

    agent = Agent(model=FunctionModel(stream_function=stream), deps_type=_CountingDeps)
    shared = _CountingDeps(counter=0)
    app = agent.to_responses(deps=shared)

    body = json.dumps({'model': 'test', 'stream': True, 'input': 'x'}).encode()
    with TestClient(app) as client:
        with client.stream('POST', '/v1/responses', content=body) as resp:
            assert resp.status_code == 200
            ''.join(resp.iter_text())
        with client.stream('POST', '/v1/responses', content=body) as resp:
            assert resp.status_code == 200
            ''.join(resp.iter_text())

    assert shared.counter == 0


async def test_error_path_emits_failed_response() -> None:
    """When the agent run raises, the stream emits a `response.completed` with status='failed'."""

    async def boom(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        raise RuntimeError('boom')
        yield  # Make this an async generator

    agent = Agent(model=FunctionModel(stream_function=boom))
    body = json.dumps({'model': 'test', 'stream': True, 'input': 'x'}).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)
    completed = next(e for e in events if e['type'] == 'response.completed')
    assert completed['response']['status'] == 'failed'


async def test_text_part_with_empty_content_emits_no_delta() -> None:
    """An empty text part at start and an empty delta in the middle should not yield delta events."""

    async def empties(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield ''
        yield 'hello'
        yield ''
        yield ' world'

    agent = Agent(model=FunctionModel(stream_function=empties))
    body = json.dumps({'model': 'test', 'stream': True, 'input': 'x'}).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)
    deltas = [e['delta'] for e in events if e['type'] == 'response.output_text.delta']
    assert deltas == ['hello', ' world']
    done = next(e for e in events if e['type'] == 'response.output_text.done')
    assert done['text'] == 'hello world'


async def test_tool_call_with_no_open_message_item() -> None:
    """A tool call before any text emits the function_call without first closing a message item."""

    async def tool_first(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if not any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            yield {0: DeltaToolCall(name='ping', json_args='{}', tool_call_id='c1')}
        else:
            yield 'done'

    agent = Agent(model=FunctionModel(stream_function=tool_first))
    body = json.dumps(
        {
            'model': 'test',
            'stream': True,
            'input': 'go',
            'tools': [{'type': 'function', 'name': 'ping', 'parameters': {'type': 'object'}}],
        }
    ).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)
    function_call_items = [e['item'] for e in events if e.get('item', {}).get('type') == 'function_call']
    assert function_call_items
    assert function_call_items[0]['name'] == 'ping'


async def test_text_then_tool_closes_open_message_item() -> None:
    """Text followed by a tool call closes the message item before emitting the function_call."""

    async def text_then_tool(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        if not any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            yield 'thinking...'
            yield {0: DeltaToolCall(name='ping', json_args='{}', tool_call_id='c1')}
        else:
            yield 'done'

    agent = Agent(model=FunctionModel(stream_function=text_then_tool))
    body = json.dumps(
        {
            'model': 'test',
            'stream': True,
            'input': 'go',
            'tools': [{'type': 'function', 'name': 'ping', 'parameters': {'type': 'object'}}],
        }
    ).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)
    event_types = [e['type'] for e in events]
    # Text item is emitted, closed (with text done), then tool call follows.
    text_done_idx = event_types.index('response.output_text.done')
    output_done_idx = event_types.index('response.output_item.done')
    function_call_idx = next(i for i, e in enumerate(events) if e.get('item', {}).get('type') == 'function_call')
    assert text_done_idx < output_done_idx < function_call_idx


def test_extract_text_content_skips_non_text_parts_in_lists() -> None:
    """`_extract_text_content` skips non-dict items, unknown types, and non-string text values."""
    body: dict[str, Any] = {
        'model': 'test',
        'stream': True,
        'input': [
            {
                'type': 'message',
                'role': 'system',
                'content': [
                    'not_a_dict_skipped',
                    {'type': 'unknown', 'text': 'ignored'},
                    {'type': 'input_text', 'text': 12345},
                    {'type': 'input_text', 'text': 'kept'},
                ],
            },
        ],
    }
    adapter = _adapter_for(body)
    [request] = adapter.messages
    assert isinstance(request, ModelRequest)
    [part] = request.parts
    assert isinstance(part, SystemPromptPart)
    assert part.content == 'kept'


def test_messages_with_empty_or_unparseable_content_drop_silently() -> None:
    """Edge cases that have no parseable text or content should produce no message parts."""
    body: dict[str, Any] = {
        'model': 'test',
        'stream': True,
        'input': [
            {'type': 'message', 'role': 'system', 'content': 42},
            {'type': 'message', 'role': 'system', 'content': []},
            {'type': 'message', 'role': 'user', 'content': None},
            {'type': 'message', 'role': 'assistant', 'content': None},
            {'type': 'message', 'role': 'assistant', 'content': []},
            {'type': 'message', 'role': 'function', 'content': 'unknown-role'},
            {'type': 'reasoning', 'summary': 'unknown-top-level-type'},
        ],
    }
    adapter = _adapter_for(body)
    assert adapter.messages == []


async def _collect_from_stream(
    event_stream: ResponsesEventStream[Any, Any], stream: AsyncIterator[Any]
) -> list[dict[str, Any]]:
    """Helper to feed a manual event generator through the event stream and collect SSE data frames."""
    events: list[dict[str, Any]] = []
    async for chunk in event_stream.encode_stream(event_stream.transform_stream(stream)):
        if chunk.startswith('event: done'):
            continue
        for line in chunk.strip().split('\n'):
            if line.startswith('data: '):
                events.append(json.loads(line.removeprefix('data: ')))
    return events


def _bare_run_input() -> dict[str, Any]:
    return {'model': 'test', 'stream': True, 'input': 'x'}


async def test_back_to_back_text_parts_keep_message_item_open() -> None:
    """Two TextParts back-to-back: the first PartEndEvent has `next_part_kind='text'`, so the
    message item stays open; the second PartStartEvent re-enters with state already set."""
    event_stream = ResponsesEventStream[None, str](run_input=_bare_run_input())  # pyright: ignore[reportArgumentType]

    async def parts() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=TextPart(content='Hello'))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=0, part=TextPart(content='Hello world'), next_part_kind='text')
        yield PartStartEvent(index=1, part=TextPart(content='Goodbye'), previous_part_kind='text')
        yield PartDeltaEvent(index=1, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=1, part=TextPart(content='Goodbye world'))

    events = await _collect_from_stream(event_stream, parts())
    deltas = [e['delta'] for e in events if e['type'] == 'response.output_text.delta']
    assert deltas == ['Hello', ' world', 'Goodbye', ' world']
    # Only one item.added/item.done pair — the back-to-back TextParts share the message item.
    item_added = [e for e in events if e['type'] == 'response.output_item.added']
    item_done = [e for e in events if e['type'] == 'response.output_item.done']
    assert len(item_added) == 1
    assert len(item_done) == 1
    done = next(e for e in events if e['type'] == 'response.output_text.done')
    assert done['text'] == 'Hello worldGoodbye world'


async def test_after_stream_closes_unclosed_message_item() -> None:
    """If the run ends mid-text without a PartEndEvent (e.g., crash), `after_stream` finalises the open item."""
    event_stream = ResponsesEventStream[None, str](run_input=_bare_run_input())  # pyright: ignore[reportArgumentType]

    async def truncated() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=TextPart(content='partial'))
        # Stream ends without PartEndEvent

    events = await _collect_from_stream(event_stream, truncated())
    event_types = [e['type'] for e in events]
    # after_stream still emits the closing events for the open message.
    assert 'response.output_text.done' in event_types
    assert event_types[-1] == 'response.completed'


async def test_tool_call_after_open_text_closes_item_first() -> None:
    """When a tool call follows an unclosed text part, the message item is closed before the function_call.

    This exercises the defensive close path in `handle_tool_call_start`: in normal operation
    `handle_text_end` closes the item, but if a tool call arrives without a preceding PartEndEvent,
    the tool-call handler must finalise the item itself.
    """
    event_stream = ResponsesEventStream[None, str](run_input=_bare_run_input())  # pyright: ignore[reportArgumentType]

    async def text_then_tool_no_end() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=TextPart(content='thinking'))
        yield PartStartEvent(index=1, part=ToolCallPart(tool_name='f', args='{}', tool_call_id='c1'))

    events = await _collect_from_stream(event_stream, text_then_tool_no_end())
    event_types = [e['type'] for e in events]
    text_done_idx = event_types.index('response.output_text.done')
    item_done_idx = next(
        i
        for i, e in enumerate(events)
        if e['type'] == 'response.output_item.done' and e['item'].get('type') == 'message'
    )
    function_call_idx = next(i for i, e in enumerate(events) if e.get('item', {}).get('type') == 'function_call')
    assert text_done_idx < item_done_idx < function_call_idx


async def test_tool_call_delta_is_a_no_op() -> None:
    """ToolCallPartDelta events must be accepted without crashing and without emitting deltas."""
    event_stream = ResponsesEventStream[None, str](run_input=_bare_run_input())  # pyright: ignore[reportArgumentType]

    async def deltas() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=ToolCallPart(tool_name='f', args='', tool_call_id='c1'))
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"x":1}', tool_call_id='c1'))
        yield PartEndEvent(index=0, part=ToolCallPart(tool_name='f', args='{"x":1}', tool_call_id='c1'))

    events = await _collect_from_stream(event_stream, deltas())
    event_types = [e['type'] for e in events]
    # No delta event surfaced; the tool call is emitted via item.added + item.done.
    assert 'response.function_call_arguments.delta' not in event_types
    item_added = [e for e in events if e['type'] == 'response.output_item.added']
    item_done = [e for e in events if e['type'] == 'response.output_item.done']
    assert [e['item']['type'] for e in item_added] == ['function_call']
    assert [e['item']['type'] for e in item_done] == ['function_call']

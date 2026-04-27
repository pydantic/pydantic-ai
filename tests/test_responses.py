"""Tests for the OpenAI Responses protocol integration."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
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
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from .conftest import IsDatetime, IsSameStr, try_import

with try_import() as imports_successful:
    from starlette.applications import Starlette
    from starlette.testclient import TestClient

    from pydantic_ai.ui.responses import ResponsesAdapter

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai or starlette not installed'),
    pytest.mark.filterwarnings(
        'ignore:State was provided but `deps` of type `NoneType` does not implement the `StateHandler` protocol:UserWarning'
    ),
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

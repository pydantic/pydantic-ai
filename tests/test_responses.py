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
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
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
    UploadedFile,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, BuiltinToolCallsReturns, DeltaToolCall, DeltaToolCalls, FunctionModel

from .conftest import IsDatetime, IsSameStr, try_import

with try_import() as imports_successful:
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.testclient import TestClient

    from pydantic_ai.ui.responses import ResponsesAdapter
    from pydantic_ai.ui.responses._event_stream import ResponsesEventStream
    from pydantic_ai.ui.responses.app import gateway

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


async def test_streams_frontend_tool_call_with_spec_event_sequence() -> None:
    """Frontend-tool calls emit the OpenAI spec sequence: item.added (empty args) → arguments.delta → arguments.done → item.done (full args)."""
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

    function_call_added = [
        e for e in events if e['type'] == 'response.output_item.added' and e['item'].get('type') == 'function_call'
    ]
    function_call_done = [
        e for e in events if e['type'] == 'response.output_item.done' and e['item'].get('type') == 'function_call'
    ]
    args_delta = [e for e in events if e['type'] == 'response.function_call_arguments.delta']
    args_done = [e for e in events if e['type'] == 'response.function_call_arguments.done']

    assert len(function_call_added) == 1
    assert function_call_added[0]['item'] == snapshot(
        {
            'id': IsSameStr(),
            'arguments': '',
            'call_id': 'call_1',
            'name': 'get_weather',
            'status': 'in_progress',
            'type': 'function_call',
        }
    )

    assert [e['delta'] for e in args_delta] == ['{"location":"Paris"}']
    assert len(args_done) == 1
    assert args_done[0]['arguments'] == '{"location":"Paris"}'
    assert args_done[0]['name'] == 'get_weather'

    assert len(function_call_done) == 1
    assert function_call_done[0]['item'] == snapshot(
        {
            'id': IsSameStr(),
            'arguments': '{"location":"Paris"}',
            'call_id': 'call_1',
            'name': 'get_weather',
            'status': 'completed',
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


async def test_input_file_with_file_id_passes_through_as_uploaded_file() -> None:
    """An `input_file` referencing a `file_id` is parsed as `UploadedFile(provider_name='openai', file_id=...)`."""
    body = json.dumps(
        {
            'model': 'test',
            'stream': True,
            'input': [
                {
                    'type': 'message',
                    'role': 'user',
                    'content': [{'type': 'input_file', 'file_id': 'file-abc123'}],
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
    [uploaded] = user_part.content
    assert uploaded == UploadedFile(file_id='file-abc123', provider_name='openai')


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
                    {'type': 'input_file'},
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
                            UploadedFile(file_id='file-abc', provider_name='openai'),
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
    # Frontend tool names exposed to the event stream for backend/frontend disambiguation.
    assert adapter.frontend_tool_names == frozenset({'good_tool'})


def testfrontend_tool_names_empty_when_no_tools() -> None:
    """Adapters without a `tools` array have an empty frontend-tool-names set, so all tool calls are backend."""
    adapter = _adapter_for({'model': 'test', 'stream': True, 'input': 'hi'})
    assert adapter.frontend_tool_names == frozenset()
    adapter_dict_tools = _adapter_for({'model': 'test', 'stream': True, 'input': 'hi', 'tools': 'not-a-list'})
    assert adapter_dict_tools.frontend_tool_names == frozenset()


@dataclass
class _CountingDeps:
    state: dict[str, Any] | None = None
    counter: int = 0


def test_state_handler_deps_are_replaced_per_request() -> None:
    """When `deps` implements `StateHandler`, each request gets a fresh copy via `replace`."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
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


async def test_frontend_tool_call_with_no_open_message_item() -> None:
    """A frontend tool call before any text emits the function_call without first closing a message item."""

    async def tool_first(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        # The agent stops on the frontend tool call (DeferredToolRequests), so a second turn
        # never fires — no `else` branch needed.
        yield {0: DeltaToolCall(name='ping', json_args='{}', tool_call_id='c1')}

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
    function_call_added = [
        e['item']
        for e in events
        if e['type'] == 'response.output_item.added' and e['item'].get('type') == 'function_call'
    ]
    assert len(function_call_added) == 1
    assert function_call_added[0]['name'] == 'ping'


async def test_text_then_frontend_tool_closes_open_message_item() -> None:
    """Text followed by a frontend tool call closes the message item before emitting the function_call."""

    async def text_then_tool(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
        yield 'thinking...'
        yield {0: DeltaToolCall(name='ping', json_args='{}', tool_call_id='c1')}

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
    text_done_idx = event_types.index('response.output_text.done')
    msg_done_idx = next(
        i
        for i, e in enumerate(events)
        if e['type'] == 'response.output_item.done' and e['item'].get('type') == 'message'
    )
    function_call_idx = next(
        i
        for i, e in enumerate(events)
        if e['type'] == 'response.output_item.added' and e['item'].get('type') == 'function_call'
    )
    assert text_done_idx < msg_done_idx < function_call_idx


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


def _bare_event_stream(*, frontend_tool_names: frozenset[str] = frozenset()) -> ResponsesEventStream[None, str]:
    return ResponsesEventStream[None, str](
        run_input=_bare_run_input(),  # pyright: ignore[reportArgumentType]
        frontend_tool_names=frontend_tool_names,
    )


async def test_back_to_back_text_parts_keep_message_item_open() -> None:
    """Two TextParts back-to-back share the message item via `followed_by_text=True`."""
    event_stream = _bare_event_stream()

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
    item_added = [e for e in events if e['type'] == 'response.output_item.added']
    item_done = [e for e in events if e['type'] == 'response.output_item.done']
    assert len(item_added) == 1
    assert len(item_done) == 1
    done = next(e for e in events if e['type'] == 'response.output_text.done')
    assert done['text'] == 'Hello worldGoodbye world'


async def test_after_stream_closes_unclosed_message_item() -> None:
    """If the run ends mid-text without a PartEndEvent (e.g., crash), `after_stream` finalises the open item."""
    event_stream = _bare_event_stream()

    async def truncated() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=TextPart(content='partial'))

    events = await _collect_from_stream(event_stream, truncated())
    event_types = [e['type'] for e in events]
    assert 'response.output_text.done' in event_types
    assert event_types[-1] == 'response.completed'


async def test_frontend_tool_call_after_open_text_closes_item_first() -> None:
    """A frontend tool call following an unclosed text part finalises the message before emitting the function_call."""
    event_stream = _bare_event_stream(frontend_tool_names=frozenset({'f'}))

    async def text_then_tool_no_end() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=TextPart(content='thinking'))
        yield PartStartEvent(index=1, part=ToolCallPart(tool_name='f', args='{}', tool_call_id='c1'))

    events = await _collect_from_stream(event_stream, text_then_tool_no_end())
    event_types = [e['type'] for e in events]
    text_done_idx = event_types.index('response.output_text.done')
    msg_done_idx = next(
        i
        for i, e in enumerate(events)
        if e['type'] == 'response.output_item.done' and e['item'].get('type') == 'message'
    )
    function_call_idx = next(
        i
        for i, e in enumerate(events)
        if e['type'] == 'response.output_item.added' and e['item'].get('type') == 'function_call'
    )
    assert text_done_idx < msg_done_idx < function_call_idx


async def test_tool_call_arguments_stream_as_deltas() -> None:
    """Args streamed via PartDelta produce `response.function_call_arguments.delta` events."""
    event_stream = _bare_event_stream(frontend_tool_names=frozenset({'f'}))

    async def deltas() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=ToolCallPart(tool_name='f', args='', tool_call_id='c1'))
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"x":', tool_call_id='c1'))
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='1}', tool_call_id='c1'))
        yield PartEndEvent(index=0, part=ToolCallPart(tool_name='f', args='{"x":1}', tool_call_id='c1'))

    events = await _collect_from_stream(event_stream, deltas())
    delta_events = [e for e in events if e['type'] == 'response.function_call_arguments.delta']
    assert [e['delta'] for e in delta_events] == ['{"x":', '1}']
    done_events = [e for e in events if e['type'] == 'response.function_call_arguments.done']
    assert len(done_events) == 1
    assert done_events[0]['arguments'] == '{"x":1}'


async def test_tool_call_delta_without_open_call_is_ignored() -> None:
    """Delta events for a tool_call_id with no preceding start are silently dropped."""
    event_stream = _bare_event_stream(frontend_tool_names=frozenset({'f'}))

    async def orphan_delta() -> AsyncIterator[Any]:
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{}', tool_call_id='unknown'))

    events = await _collect_from_stream(event_stream, orphan_delta())
    assert not any(e['type'].startswith('response.function_call_arguments') for e in events)


async def test_tool_call_delta_with_empty_text_is_skipped() -> None:
    """Empty/non-string `args_delta` shouldn't emit a delta event."""
    event_stream = _bare_event_stream(frontend_tool_names=frozenset({'f'}))

    async def parts() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=ToolCallPart(tool_name='f', args='', tool_call_id='c1'))
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='', tool_call_id='c1'))
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta={'k': 'v'}, tool_call_id='c1'))
        yield PartEndEvent(index=0, part=ToolCallPart(tool_name='f', args='', tool_call_id='c1'))

    events = await _collect_from_stream(event_stream, parts())
    delta_events = [e for e in events if e['type'] == 'response.function_call_arguments.delta']
    assert delta_events == []


async def test_tool_call_delta_without_tool_call_id_is_skipped() -> None:
    """A `ToolCallPartDelta` without a `tool_call_id` cannot be matched and is silently skipped."""
    event_stream = _bare_event_stream(frontend_tool_names=frozenset({'f'}))

    async def parts() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=ToolCallPart(tool_name='f', args='', tool_call_id='c1'))
        yield PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"x":1}', tool_call_id=None))
        yield PartEndEvent(index=0, part=ToolCallPart(tool_name='f', args='{"x":1}', tool_call_id='c1'))

    events = await _collect_from_stream(event_stream, parts())
    delta_events = [e for e in events if e['type'] == 'response.function_call_arguments.delta']
    assert delta_events == []


async def test_backend_tool_is_invisible_to_client() -> None:
    """A backend (agent-registered) tool runs server-side and emits no `function_call` events."""

    async def stream_with_backend_tool(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[DeltaToolCalls | str]:
        if not any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            yield {0: DeltaToolCall(name='lookup_user', json_args='{"user_id":"u_42"}', tool_call_id='b_1')}
        else:
            yield 'Alice is the answer.'

    agent = Agent(model=FunctionModel(stream_function=stream_with_backend_tool))

    @agent.tool_plain
    def lookup_user(user_id: str) -> str:
        return 'Alice'

    body = json.dumps({'model': 'test', 'stream': True, 'input': 'Who is u_42?'}).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)
    item_types = {e['item']['type'] for e in events if e.get('item')}
    assert 'function_call' not in item_types
    args_events = [e for e in events if e['type'].startswith('response.function_call_arguments')]
    assert args_events == []
    deltas = [e['delta'] for e in events if e['type'] == 'response.output_text.delta']
    assert deltas == ['Alice is the answer.']


async def test_backend_tool_end_event_is_silently_ignored() -> None:
    """The `tool_call_end` for a suppressed backend call must be a no-op (no orphan args.done)."""
    event_stream = _bare_event_stream(frontend_tool_names=frozenset())

    async def parts() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=ToolCallPart(tool_name='backend', args='{}', tool_call_id='c1'))
        yield PartEndEvent(index=0, part=ToolCallPart(tool_name='backend', args='{}', tool_call_id='c1'))

    events = await _collect_from_stream(event_stream, parts())
    args_done = [e for e in events if e['type'].startswith('response.function_call_arguments')]
    assert args_done == []
    item_added = [
        e for e in events if e['type'] == 'response.output_item.added' and e['item'].get('type') == 'function_call'
    ]
    assert item_added == []


@pytest.mark.parametrize(
    ('tool_kind', 'item_type', 'in_progress_event', 'completed_event'),
    [
        (
            'web_search',
            'web_search_call',
            'response.web_search_call.in_progress',
            'response.web_search_call.completed',
        ),
        (
            'code_execution',
            'code_interpreter_call',
            'response.code_interpreter_call.in_progress',
            'response.code_interpreter_call.completed',
        ),
        (
            'file_search',
            'file_search_call',
            'response.file_search_call.in_progress',
            'response.file_search_call.completed',
        ),
        (
            'image_generation',
            'image_generation_call',
            'response.image_generation_call.in_progress',
            'response.image_generation_call.completed',
        ),
    ],
)
async def test_builtin_tool_emits_dedicated_events(
    tool_kind: str, item_type: str, in_progress_event: str, completed_event: str
) -> None:
    """Each supported builtin tool kind maps to its dedicated OpenAI event family + output item type."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[BuiltinToolCallsReturns | str]:
        yield {
            0: BuiltinToolCallPart(
                tool_name=tool_kind,
                args='{}',
                tool_call_id='b_1',
                provider_name='function',
            )
        }
        yield {
            1: BuiltinToolReturnPart(
                tool_name=tool_kind,
                content={'ok': True},
                tool_call_id='b_1',
                provider_name='function',
            )
        }
        yield 'Done.'

    agent = Agent(model=FunctionModel(stream_function=stream))
    body = json.dumps({'model': 'test', 'stream': True, 'input': 'go'}).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)
    event_types = [e['type'] for e in events]
    assert in_progress_event in event_types
    assert completed_event in event_types
    item_types = {e['item']['type'] for e in events if e.get('item')}
    assert item_type in item_types
    assert 'function_call' not in item_types

    added = next(e for e in events if e['type'] == 'response.output_item.added' and e['item'].get('type') == item_type)
    done = next(e for e in events if e['type'] == 'response.output_item.done' and e['item'].get('type') == item_type)
    assert added['item']['status'] == 'in_progress'
    assert done['item']['status'] == 'completed'


async def test_builtin_tool_in_middle_of_text_closes_message_item() -> None:
    """Text → builtin tool transition closes the message item before emitting the builtin output item."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[BuiltinToolCallsReturns | str]:
        yield 'Searching... '
        yield {0: BuiltinToolCallPart(tool_name='web_search', args='{}', tool_call_id='ws_1', provider_name='function')}
        yield {
            1: BuiltinToolReturnPart(tool_name='web_search', content={}, tool_call_id='ws_1', provider_name='function')
        }
        yield 'Found nothing.'

    agent = Agent(model=FunctionModel(stream_function=stream))
    body = json.dumps({'model': 'test', 'stream': True, 'input': 'go'}).encode()
    run_input = ResponsesAdapter.build_run_input(body)
    adapter = ResponsesAdapter[None, str](agent=agent, run_input=run_input)

    events = await _collect_events(adapter)
    event_types = [e['type'] for e in events]
    text_done_idx = event_types.index('response.output_text.done')
    msg_done_idx = next(
        i
        for i, e in enumerate(events)
        if e['type'] == 'response.output_item.done' and e['item'].get('type') == 'message'
    )
    web_search_idx = next(
        i
        for i, e in enumerate(events)
        if e['type'] == 'response.output_item.added' and e['item'].get('type') == 'web_search_call'
    )
    assert text_done_idx < msg_done_idx < web_search_idx


async def test_builtin_tool_after_open_text_closes_item_first() -> None:
    """A builtin tool call following an unclosed text part finalises the message before emitting the builtin item."""
    event_stream = _bare_event_stream()

    async def parts() -> AsyncIterator[Any]:
        yield PartStartEvent(index=0, part=TextPart(content='thinking'))
        yield PartStartEvent(
            index=1,
            part=BuiltinToolCallPart(tool_name='web_search', args='{}', tool_call_id='ws_1', provider_name='function'),
        )

    events = await _collect_from_stream(event_stream, parts())
    event_types = [e['type'] for e in events]
    text_done_idx = event_types.index('response.output_text.done')
    web_search_idx = next(
        i
        for i, e in enumerate(events)
        if e['type'] == 'response.output_item.added' and e['item'].get('type') == 'web_search_call'
    )
    assert text_done_idx < web_search_idx


async def test_unknown_builtin_tool_kind_is_silently_suppressed() -> None:
    """A `BuiltinToolCallPart` with an unmapped kind is dropped — no events, no item."""
    event_stream = _bare_event_stream()

    async def parts() -> AsyncIterator[Any]:
        yield PartStartEvent(
            index=0,
            part=BuiltinToolCallPart(tool_name='unknown_kind', args='{}', tool_call_id='b_1', provider_name='function'),
        )
        yield PartEndEvent(
            index=0,
            part=BuiltinToolCallPart(tool_name='unknown_kind', args='{}', tool_call_id='b_1', provider_name='function'),
        )

    events = await _collect_from_stream(event_stream, parts())
    item_types = {e['item']['type'] for e in events if e.get('item')}
    # The unmapped kind produces no item.added/done at all.
    assert item_types == set()


async def test_builtin_tool_return_without_open_call_is_ignored() -> None:
    """A builtin return for an id that was never opened (e.g., unmapped kind) is silently dropped."""
    event_stream = _bare_event_stream()

    async def parts() -> AsyncIterator[Any]:
        yield PartStartEvent(
            index=0,
            part=BuiltinToolReturnPart(
                tool_name='web_search', content={}, tool_call_id='orphan', provider_name='function'
            ),
        )

    events = await _collect_from_stream(event_stream, parts())
    item_types = {e['item']['type'] for e in events if e.get('item')}
    assert 'web_search_call' not in item_types


@dataclass
class _Token:
    value: str


async def test_to_responses_deps_factory_called_per_request() -> None:
    """`deps_factory` is called per request with the actual Starlette `Request`."""
    captured: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'ok'

    async def factory(request: Request) -> _Token:
        token = request.headers.get('authorization', '').removeprefix('Bearer ')
        captured.append(token)
        return _Token(value=token)

    agent = Agent(model=FunctionModel(stream_function=stream), deps_type=_Token)
    app = agent.to_responses(deps=_Token(value=''), deps_factory=factory)

    body = json.dumps(_bare_run_input()).encode()
    with TestClient(app) as client:
        for token in ('t1', 't2'):
            with client.stream(
                'POST', '/v1/responses', content=body, headers={'Authorization': f'Bearer {token}'}
            ) as resp:
                assert resp.status_code == 200
                ''.join(resp.iter_text())

    assert captured == ['t1', 't2']


async def test_to_responses_deps_factory_sync_callable() -> None:
    """A synchronous `deps_factory` is supported alongside async."""
    captured: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'ok'

    def factory(request: Request) -> _Token:
        token = request.headers.get('x-tenant', '')
        captured.append(token)
        return _Token(value=token)

    agent = Agent(model=FunctionModel(stream_function=stream), deps_type=_Token)
    app = agent.to_responses(deps=_Token(value=''), deps_factory=factory)
    body = json.dumps(_bare_run_input()).encode()
    with TestClient(app) as client:
        with client.stream('POST', '/v1/responses', content=body, headers={'X-Tenant': 'acme'}) as resp:
            ''.join(resp.iter_text())
    assert captured == ['acme']


async def test_to_responses_deps_factory_takes_precedence_over_deps() -> None:
    """When both `deps` and `deps_factory` are passed, the factory wins per request."""
    captured: list[str] = []

    @dataclass
    class _Deps:
        token: str

    async def stream_with_capture(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        # The agent itself never sees deps; we just verify the factory was called.
        yield 'ok'

    def factory(request: Request) -> _Deps:
        token = request.headers.get('x-tenant', 'unknown')
        captured.append(token)
        return _Deps(token=token)

    agent = Agent(model=FunctionModel(stream_function=stream_with_capture), deps_type=_Deps)
    shared = _Deps(token='SHARED')
    app = agent.to_responses(deps=shared, deps_factory=factory)
    body = json.dumps(_bare_run_input()).encode()
    with TestClient(app) as client:
        with client.stream('POST', '/v1/responses', content=body, headers={'X-Tenant': 'acme'}) as resp:
            ''.join(resp.iter_text())
    assert captured == ['acme']


async def test_gateway_routes_by_model_field() -> None:
    """`gateway()` dispatches `POST /v1/responses` to the agent named by the request `model` field."""

    async def make_stream(marker: str):
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield marker

        return stream

    agent_a = Agent(model=FunctionModel(stream_function=await make_stream('SUPPORT_OUT')))
    agent_b = Agent(model=FunctionModel(stream_function=await make_stream('CODER_OUT')))
    app = gateway({'support': agent_a, 'coder': agent_b})

    with TestClient(app) as client:
        with client.stream(
            'POST', '/v1/responses', content=json.dumps({'model': 'support', 'stream': True, 'input': 'x'}).encode()
        ) as r:
            text_a = ''.join(r.iter_text())
        with client.stream(
            'POST', '/v1/responses', content=json.dumps({'model': 'coder', 'stream': True, 'input': 'x'}).encode()
        ) as r:
            text_b = ''.join(r.iter_text())

    assert 'SUPPORT_OUT' in text_a
    assert 'SUPPORT_OUT' not in text_b
    assert 'CODER_OUT' in text_b


async def test_gateway_unknown_model_returns_404() -> None:
    """Unknown `model` returns 404 with an OpenAI-shaped error envelope."""
    # The agent is never invoked here — gateway short-circuits with 404 before dispatching.
    app = gateway({'a': Agent('test')})
    body = json.dumps({'model': 'does-not-exist', 'stream': True, 'input': 'x'}).encode()
    with TestClient(app) as client:
        resp = client.post('/v1/responses', content=body)
    assert resp.status_code == 404
    assert resp.json() == snapshot(
        {
            'error': {
                'message': "The model 'does-not-exist' does not exist or you do not have access to it.",
                'type': 'invalid_request_error',
                'code': 'model_not_found',
            }
        }
    )


async def test_gateway_missing_model_field_returns_404() -> None:
    """A request body without a string `model` field is treated as unknown model."""
    app = gateway({'a': Agent('test')})
    body = json.dumps({'stream': True, 'input': 'x'}).encode()
    with TestClient(app) as client:
        resp = client.post('/v1/responses', content=body)
    assert resp.status_code == 404
    assert resp.json()['error']['code'] == 'model_not_found'


async def test_gateway_invalid_json_returns_400() -> None:
    """A non-JSON body is rejected with a 400 and an OpenAI-shaped envelope."""
    app = gateway({'a': Agent('test')})
    with TestClient(app) as client:
        resp = client.post('/v1/responses', content=b'not-json')
    assert resp.status_code == 400
    assert resp.json()['error']['code'] == 'invalid_json'


async def test_gateway_non_object_body_returns_400() -> None:
    """A JSON array (non-object) body is rejected with a 400."""
    app = gateway({'a': Agent('test')})
    with TestClient(app) as client:
        resp = client.post('/v1/responses', content=b'[1, 2, 3]')
    assert resp.status_code == 400
    assert resp.json()['error']['code'] == 'invalid_request'


async def test_gateway_models_list() -> None:
    """`GET /v1/models` returns the configured agents in OpenAI's models-list shape."""
    app = gateway({'support': Agent('test'), 'coder': Agent('test')})
    with TestClient(app) as client:
        resp = client.get('/v1/models')
    assert resp.status_code == 200
    payload = resp.json()
    assert payload['object'] == 'list'
    assert [m['id'] for m in payload['data']] == ['support', 'coder']
    assert all(m['object'] == 'model' for m in payload['data'])
    assert all(m['owned_by'] == 'pydantic-ai' for m in payload['data'])
    assert all(isinstance(m['created'], int) for m in payload['data'])


async def test_gateway_owned_by_override() -> None:
    """`owned_by` is configurable for org-specific listings."""
    app = gateway({'a': Agent('test')}, owned_by='acme-corp')
    with TestClient(app) as client:
        resp = client.get('/v1/models')
    assert all(m['owned_by'] == 'acme-corp' for m in resp.json()['data'])


async def test_gateway_with_deps_factory() -> None:
    """`gateway(deps_factory=...)` applies the factory to dispatched agents."""
    captured: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        yield 'ok'

    async def factory(request: Request) -> str:
        token = request.headers.get('x-tenant', '')
        captured.append(token)
        return token

    agent = Agent(model=FunctionModel(stream_function=stream), deps_type=str)
    app = gateway({'a': agent}, deps_factory=factory)
    body = json.dumps({'model': 'a', 'stream': True, 'input': 'x'}).encode()
    with TestClient(app) as client:
        with client.stream('POST', '/v1/responses', content=body, headers={'X-Tenant': 'acme'}) as resp:
            ''.join(resp.iter_text())
    assert captured == ['acme']


async def test_to_ag_ui_accepts_deps_factory() -> None:
    """`Agent.to_ag_ui` exposes `deps_factory` in parity with `to_responses`."""
    agent = Agent('test')
    app = agent.to_ag_ui(deps_factory=lambda _request: None)
    assert isinstance(app, Starlette)

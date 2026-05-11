"""Unit tests for `OpenResponsesModel`.

These tests exercise the raw-httpx / SSE-parsing surface that bypasses the OpenAI Python
SDK's closed pydantic union — covering response dispatch, streaming event handling,
request body construction, and message serialization (including the `pydantic_ai:*`
extension items). End-to-end layered-agent round-trip lives in
`tests/test_layered_agents.py`; this file targets branches and edge cases.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    AgentContextPart,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters

from ..conftest import try_import

with try_import() as imports_successful:
    import httpx
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    from pydantic_ai.models.openresponses import OpenResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai, starlette, or httpx_sse not installed'),
]


def _build_model(app: Starlette, *, api_key: str = 'test-key') -> tuple[OpenResponsesModel, httpx.AsyncClient]:
    """Build an `OpenResponsesModel` backed by an in-process Starlette app."""
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url='http://test')
    model = OpenResponsesModel(
        model_name='inner',
        provider=OpenAIProvider(api_key=api_key, base_url='http://test/v1', http_client=client),
    )
    return model, client


def _sse_body(events: list[tuple[str, dict[str, Any]]]) -> str:
    """Render a list of (event_type, payload) tuples as an SSE response body."""
    lines: list[str] = []
    for event_type, payload in events:
        lines.append(f'event: {event_type}')
        lines.append(f'data: {json.dumps(payload)}')
        lines.append('')
    lines.append('event: done')
    lines.append('data: ')
    lines.append('')
    return '\n'.join(lines)


async def test_request_dispatches_all_output_item_types(allow_model_requests: None) -> None:
    """Non-streaming `request()` parses every supported output item shape into ModelResponse parts."""

    async def handler(_request: Request) -> Response:
        return JSONResponse(
            {
                'id': 'resp_abc',
                'model': 'inner',
                'status': 'completed',
                'output': [
                    {'type': 'message', 'content': [{'type': 'output_text', 'text': 'hello world'}]},
                    {'type': 'function_call', 'call_id': 'f_1', 'name': 'frontend_tool', 'arguments': '{"x": 1}'},
                    {
                        'type': 'pydantic_ai:custom_tool_call',
                        'call_id': 'b_1',
                        'name': 'backend_tool',
                        'arguments': '{"y": 2}',
                    },
                    {'type': 'pydantic_ai:custom_tool_call_output', 'call_id': 'b_1', 'output': 'backend-result'},
                    {
                        'type': 'pydantic_ai:agent_context',
                        'from_agent': 'guardrail',
                        'role': 'context',
                        'content': 'session-context',
                    },
                    # Unknown item types are silently skipped.
                    {'type': 'unknown'},
                    # Non-dict items are skipped before the dispatch helper sees them.
                    'not-a-dict',
                    # Invalid extension item — missing required fields, gets ignored.
                    {'type': 'pydantic_ai:custom_tool_call', 'call_id': 'missing-name'},
                ],
            }
        )

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
        )
    finally:
        await client.aclose()

    text_parts = [p for p in response.parts if isinstance(p, TextPart)]
    tool_calls = [p for p in response.parts if isinstance(p, ToolCallPart)]
    builtin_calls = [p for p in response.parts if isinstance(p, BuiltinToolCallPart)]
    builtin_returns = [p for p in response.parts if isinstance(p, BuiltinToolReturnPart)]
    contexts = [p for p in response.parts if isinstance(p, AgentContextPart)]

    assert [p.content for p in text_parts] == ['hello world']
    assert [(p.tool_name, p.tool_call_id) for p in tool_calls] == [('frontend_tool', 'f_1')]
    assert [(p.tool_name, p.tool_call_id) for p in builtin_calls] == [('backend_tool', 'b_1')]
    assert [(p.tool_name, p.content) for p in builtin_returns] == [('backend_tool', 'backend-result')]
    assert [(p.from_agent, p.role, p.content) for p in contexts] == [('guardrail', 'context', 'session-context')]
    assert response.provider_response_id == 'resp_abc'
    assert response.finish_reason == 'stop'
    assert response.provider_details == {'finish_reason': 'completed'}


async def test_request_raises_model_http_error_on_4xx(allow_model_requests: None) -> None:
    async def handler(_request: Request) -> Response:
        return JSONResponse({'error': 'nope'}, status_code=400)

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        with pytest.raises(ModelHTTPError) as exc_info:
            await model.request([ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters())
    finally:
        await client.aclose()

    assert exc_info.value.status_code == 400


async def test_request_rejects_non_object_payload(allow_model_requests: None) -> None:
    async def handler(_request: Request) -> Response:
        return JSONResponse(['not', 'an', 'object'])

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        with pytest.raises(ValueError, match='non-object body'):
            await model.request([ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters())
    finally:
        await client.aclose()


async def test_request_stream_dispatches_all_event_types(allow_model_requests: None) -> None:
    """Streaming `request_stream()` dispatches every SSE event type via `_handle_sse`."""
    events: list[tuple[str, dict[str, Any]]] = [
        ('response.created', {'type': 'response.created', 'response': {'id': 'resp_xyz'}}),
        (
            'response.output_item.added',
            {
                'type': 'response.output_item.added',
                'item': {'type': 'message', 'id': 'msg_1'},
            },
        ),
        (
            'response.output_text.delta',
            {'type': 'response.output_text.delta', 'item_id': 'msg_1', 'delta': 'hi'},
        ),
        (
            'response.output_item.added',
            {
                'type': 'response.output_item.added',
                'item': {
                    'type': 'function_call',
                    'id': 'f_1',
                    'call_id': 'f_1',
                    'name': 'frontend_tool',
                    'arguments': '{"x": 1}',
                },
            },
        ),
        (
            'response.output_item.added',
            {
                'type': 'response.output_item.added',
                'item': {
                    'type': 'pydantic_ai:custom_tool_call',
                    'id': 'b_1',
                    'call_id': 'b_1',
                    'name': 'backend_tool',
                    'arguments': '{"y": 2}',
                },
            },
        ),
        (
            'response.output_item.added',
            {
                'type': 'response.output_item.added',
                'item': {
                    'type': 'pydantic_ai:agent_context',
                    'id': 'ctx_1',
                    'from_agent': 'guardrail',
                    'role': 'context',
                    'content': 'profile',
                },
            },
        ),
        (
            'response.output_item.done',
            {
                'type': 'response.output_item.done',
                'item': {
                    'type': 'pydantic_ai:custom_tool_call_output',
                    'call_id': 'b_1',
                    'output': 'backend-result',
                },
            },
        ),
        # `item` not a dict — skipped silently in both `_dispatch_item_added` and `_dispatch_item_done`.
        ('response.output_item.added', {'type': 'response.output_item.added', 'item': None}),
        ('response.output_item.done', {'type': 'response.output_item.done', 'item': None}),
        # Unknown event type — falls through every branch.
        ('response.other', {'type': 'response.other'}),
        # `response` missing on response.created — `is_str_dict` returns False, skipped.
        ('response.created', {'type': 'response.created'}),
        # `response.completed` with finish reason
        (
            'response.completed',
            {'type': 'response.completed', 'response': {'status': 'completed'}},
        ),
    ]

    async def handler(_request: Request) -> Response:
        return Response(_sse_body(events), media_type='text/event-stream')

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
        ) as stream:
            async for _ in stream:
                pass
            response = stream.get()
            cancel_errors = stream.get_stream_cancel_errors()
    finally:
        await client.aclose()

    assert response.provider_response_id == 'resp_xyz'
    assert response.finish_reason == 'stop'
    assert any(isinstance(p, TextPart) and p.content == 'hi' for p in response.parts)
    assert any(isinstance(p, ToolCallPart) and p.tool_name == 'frontend_tool' for p in response.parts)
    assert any(isinstance(p, BuiltinToolCallPart) and p.tool_name == 'backend_tool' for p in response.parts)
    assert any(isinstance(p, BuiltinToolReturnPart) and p.content == 'backend-result' for p in response.parts)
    assert any(isinstance(p, AgentContextPart) and p.from_agent == 'guardrail' for p in response.parts)
    assert cancel_errors == (httpx.StreamError, httpx.TransportError)


async def test_request_stream_raises_model_http_error_on_4xx(allow_model_requests: None) -> None:
    async def handler(_request: Request) -> Response:
        return Response('upstream broke', status_code=503)

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        with pytest.raises(ModelHTTPError) as exc_info:
            async with model.request_stream(
                [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
            ):
                pass
    finally:
        await client.aclose()

    assert exc_info.value.status_code == 503


async def test_build_request_body_carries_tools_instructions_and_settings() -> None:
    """Round-trip the request body construction so tools/instructions/settings reach the wire."""
    captured: dict[str, Any] = {}

    async def handler(request: Request) -> Response:
        captured.update(await request.json())
        captured['headers'] = dict(request.headers)
        return JSONResponse({'id': 'r', 'model': 'inner', 'status': 'completed', 'output': []})

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart(content='be terse'),
                SystemPromptPart(content='reply in english'),
                UserPromptPart(content='hi'),
            ]
        ),
    ]
    parameters = ModelRequestParameters()
    body = await model._build_request_body(  # pyright: ignore[reportPrivateUsage]
        messages,
        {'temperature': 0.1, 'top_p': 0.9, 'max_tokens': 256, 'extra_headers': {'X-Custom': 'yes'}},
        parameters,
        stream=False,
    )
    headers = model._build_headers({'extra_headers': {'X-Custom': 'yes'}}, sse=True)  # pyright: ignore[reportPrivateUsage]
    try:
        await client.aclose()
    finally:
        pass

    assert body['model'] == 'inner'
    assert body['instructions'] == 'be terse\n\nreply in english'
    assert body['temperature'] == 0.1
    assert body['top_p'] == 0.9
    assert body['max_output_tokens'] == 256
    assert headers['Authorization'] == 'Bearer test-key'
    assert headers['Accept'] == 'text/event-stream'
    assert headers['X-Custom'] == 'yes'


async def test_serialize_messages_covers_all_part_kinds() -> None:
    """`_serialize_messages` walks user/assistant/tool/context parts into wire items."""
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(content='hello'),
                UserPromptPart(content=['multi', 'part']),
                ToolReturnPart(tool_name='t', tool_call_id='c1', content='result'),
                SystemPromptPart(content='ignored at item layer'),
            ]
        ),
        ModelResponse(
            parts=[
                TextPart(content='reply'),
                ToolCallPart(tool_name='t', tool_call_id='c1', args='{"a":1}'),
                AgentContextPart(content='note', from_agent='outer', role='observation'),
            ]
        ),
    ]
    items = OpenResponsesModel._serialize_messages(messages)  # pyright: ignore[reportPrivateUsage]
    types = [it.get('type') or it.get('role') for it in items]
    assert types == ['user', 'user', 'function_call_output', 'assistant', 'function_call', 'pydantic_ai:agent_context']
    assert items[1]['content'] == 'multipart'
    assert items[2]['output'] == 'result'
    assert items[5]['from_agent'] == 'outer'


async def test_streamed_response_close_stream_closes_underlying_response(allow_model_requests: None) -> None:
    """`close_stream` is invoked by the framework on cancellation — verify it runs."""

    async def handler(_request: Request) -> Response:
        return Response(
            _sse_body([('response.created', {'type': 'response.created', 'response': {'id': 'r'}})]),
            media_type='text/event-stream',
        )

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
        ) as stream:
            # Pull one event so the stream is engaged.
            iterator = stream.__aiter__()
            try:
                await iterator.__anext__()
            except StopAsyncIteration:
                pass
            await stream.close_stream()
    finally:
        await client.aclose()


def test_system_property_returns_openresponses() -> None:
    """The `system` property surfaces the provider name used in `ModelResponse.provider_name`."""

    async def handler(_request: Request) -> Response:
        return JSONResponse({})

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, _client = _build_model(app)
    assert model.system == 'openresponses'


async def test_request_body_streams_flag_propagates() -> None:
    """`stream=True` flag flips the body field; `tools` are emitted when registered."""
    from pydantic_ai.tools import ToolDefinition

    async def handler(_request: Request) -> Response:
        return JSONResponse({})

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)

    parameters = ModelRequestParameters()
    parameters.tool_defs = {
        'lookup': ToolDefinition(
            name='lookup',
            description='look something up',
            parameters_json_schema={'type': 'object', 'properties': {}},
        )
    }
    body = await model._build_request_body(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[UserPromptPart(content='hi')])],
        None,
        parameters,
        stream=True,
    )
    await client.aclose()
    assert body['stream'] is True
    assert body['tools'] == [
        {
            'type': 'function',
            'name': 'lookup',
            'description': 'look something up',
            'parameters': {'type': 'object', 'properties': {}},
        }
    ]


async def test_agent_context_part_has_content() -> None:
    """`AgentContextPart.has_content` returns False for empty content."""
    assert AgentContextPart(content='x', from_agent='a').has_content() is True
    assert AgentContextPart(content='', from_agent='a').has_content() is False


async def test_request_stream_skips_malformed_responses(allow_model_requests: None) -> None:
    """`response.completed` without a parseable response body is silently skipped."""

    events: list[tuple[str, dict[str, Any]]] = [
        # Non-string id under response.created — falls past the isinstance check.
        ('response.created', {'type': 'response.created', 'response': {'id': 123}}),
        # Non-dict item under output_item.added — skipped.
        ('response.output_item.added', {'type': 'response.output_item.added', 'item': None}),
        # output_text.delta with non-string delta — skipped.
        ('response.output_text.delta', {'type': 'response.output_text.delta', 'item_id': 'x', 'delta': None}),
        # response.completed without a dict response — skipped.
        ('response.completed', {'type': 'response.completed'}),
        # response.completed with non-string status — skipped.
        ('response.completed', {'type': 'response.completed', 'response': {'status': 42}}),
    ]

    async def handler(_request: Request) -> Response:
        return Response(_sse_body(events), media_type='text/event-stream')

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
        ) as stream:
            async for _ in stream:
                pass
    finally:
        await client.aclose()


async def test_dispatch_item_added_skips_message_type() -> None:
    """Streaming `message` items are no-ops in `_dispatch_item_added` (handled via delta events)."""
    from pydantic_ai.models.openresponses import OpenResponsesStreamedResponse

    response = OpenResponsesStreamedResponse(
        model_request_parameters=ModelRequestParameters(),
        _model_name='inner',
        _provider_name='openresponses',
        _response=httpx.Response(200),
    )
    events: list[Any] = [ev async for ev in response._dispatch_item_added({'type': 'message', 'id': 'msg_1'})]  # pyright: ignore[reportPrivateUsage]
    assert events == []


async def test_iter_dicts_handles_non_list_value() -> None:
    """The `_iter_dicts` helper returns an empty list when the value isn't a list."""
    from pydantic_ai.models.openresponses import _iter_dicts  # pyright: ignore[reportPrivateUsage]

    assert _iter_dicts('not-a-list') == []
    assert _iter_dicts(None) == []
    assert _iter_dicts({'shape': 'dict'}) == []


async def test_collect_instructions_skips_non_request_messages() -> None:
    """Only `ModelRequest` messages contribute to the merged instructions string."""
    messages: list[ModelMessage] = [
        ModelResponse(parts=[TextPart(content='earlier reply')]),
        ModelRequest(parts=[SystemPromptPart(content='instructions A'), UserPromptPart(content='hi')]),
        ModelResponse(parts=[TextPart(content='another reply')]),
        ModelRequest(parts=[SystemPromptPart(content='instructions B')]),
    ]
    result = OpenResponsesModel._collect_instructions(messages, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    assert result == 'instructions A\n\ninstructions B'


async def test_request_stream_skips_malformed_dispatch_items(allow_model_requests: None) -> None:
    """Streaming `_dispatch_item_added` / `_dispatch_item_done` skip items with missing required fields."""

    events: list[tuple[str, dict[str, Any]]] = [
        # function_call missing call_id+name — `_dispatch_item_added` falls through to exit.
        (
            'response.output_item.added',
            {'type': 'response.output_item.added', 'item': {'type': 'function_call', 'id': 'f_x'}},
        ),
        # pydantic_ai:custom_tool_call missing name.
        (
            'response.output_item.added',
            {
                'type': 'response.output_item.added',
                'item': {'type': 'pydantic_ai:custom_tool_call', 'id': 'b_x', 'call_id': 'b_x'},
            },
        ),
        # pydantic_ai:agent_context with bogus role.
        (
            'response.output_item.added',
            {
                'type': 'response.output_item.added',
                'item': {
                    'type': 'pydantic_ai:agent_context',
                    'id': 'c_x',
                    'from_agent': 'g',
                    'role': 'bogus',
                    'content': 'x',
                },
            },
        ),
        # pydantic_ai:custom_tool_call_output without call_id.
        (
            'response.output_item.done',
            {
                'type': 'response.output_item.done',
                'item': {'type': 'pydantic_ai:custom_tool_call_output', 'output': 'orphan'},
            },
        ),
    ]

    async def handler(_request: Request) -> Response:
        return Response(_sse_body(events), media_type='text/event-stream')

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
        ) as stream:
            async for _ in stream:
                pass
            response = stream.get()
    finally:
        await client.aclose()
    # No malformed item produced a part.
    assert response.parts == []


async def test_serialize_messages_skips_unhandled_part_types() -> None:
    """`_serialize_messages` silently skips request/response parts it doesn't recognize."""
    from pydantic_ai.messages import RetryPromptPart, ThinkingPart

    messages: list[ModelMessage] = [
        ModelRequest(parts=[RetryPromptPart(content='retry me', tool_name='t', tool_call_id='c1')]),
        ModelResponse(parts=[ThinkingPart(content='internal monologue')]),
    ]
    items = OpenResponsesModel._serialize_messages(messages)  # pyright: ignore[reportPrivateUsage]
    assert items == []


async def test_request_silently_ignores_malformed_wire_items(allow_model_requests: None) -> None:
    """All `_dispatch_output_item` arms tolerate missing/wrong-typed fields by skipping the item."""

    async def handler(_request: Request) -> Response:
        return JSONResponse(
            {
                'model': 'inner',
                # No `status` field — exercises the `isinstance(status, str)` False branch.
                'output': [
                    # function_call missing call_id+name
                    {'type': 'function_call'},
                    # pydantic_ai:custom_tool_call missing both fields
                    {'type': 'pydantic_ai:custom_tool_call'},
                    # pydantic_ai:custom_tool_call_output without call_id
                    {'type': 'pydantic_ai:custom_tool_call_output', 'output': 'orphan'},
                    # pydantic_ai:agent_context with role not in the allowed set
                    {
                        'type': 'pydantic_ai:agent_context',
                        'from_agent': 'x',
                        'role': 'bogus',
                        'content': 'y',
                    },
                    # message item whose content chunks are non-output_text / non-str
                    {
                        'type': 'message',
                        'content': [
                            {'type': 'input_text', 'text': 'not output'},
                            {'type': 'output_text', 'text': 123},
                            'not-a-dict-chunk',
                        ],
                    },
                ],
            }
        )

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        response = await model.request(
            [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
        )
    finally:
        await client.aclose()
    # Every malformed item produced no part; the response is empty.
    assert response.parts == []
    assert response.finish_reason is None


async def test_request_body_omits_settings_when_values_are_none() -> None:
    """`_build_request_body` skips settings whose values are `None`."""

    async def handler(_request: Request) -> Response:
        return JSONResponse({'output': []})

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    from pydantic_ai.settings import ModelSettings

    settings: ModelSettings = {'top_p': 0.9}
    body = await model._build_request_body(  # pyright: ignore[reportPrivateUsage]
        [ModelRequest(parts=[UserPromptPart(content='hi')])],
        settings,
        ModelRequestParameters(),
        stream=False,
    )
    await client.aclose()
    # `temperature` not set → skipped via the `if value is not None:` False path.
    assert 'temperature' not in body
    assert body['top_p'] == 0.9


async def test_build_headers_handles_missing_api_key_and_non_string_extra_headers() -> None:
    """`_build_headers` omits `Authorization` when api_key is absent and skips non-string extra headers."""

    async def handler(_request: Request) -> Response:
        return JSONResponse({})

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url='http://test')
    model = OpenResponsesModel(
        model_name='inner',
        provider=OpenAIProvider(api_key='unset', base_url='http://test/v1', http_client=client),
    )
    # Strip the api_key so `getattr(..., 'api_key', None)` falsy path runs.
    model.client.api_key = ''
    headers = model._build_headers(  # pyright: ignore[reportPrivateUsage]
        {'extra_headers': {'X-Valid': 'yes', 42: 'bad-key', 'X-Bad': 7}},  # pyright: ignore[reportArgumentType]
    )
    await client.aclose()
    assert 'Authorization' not in headers
    assert headers['X-Valid'] == 'yes'
    assert 'X-Bad' not in headers
    assert 42 not in headers


async def test_streamed_response_handles_invalid_json_data(allow_model_requests: None) -> None:
    """SSE frames with non-JSON `data` payloads short-circuit cleanly."""

    body = 'event: response.created\ndata: not-json\n\nevent: done\ndata: \n\n'

    async def handler(_request: Request) -> Response:
        return Response(body, media_type='text/event-stream')

    app = Starlette(routes=[Route('/v1/responses', handler, methods=['POST'])])
    model, client = _build_model(app)
    try:
        async with model.request_stream(
            [ModelRequest(parts=[UserPromptPart(content='hi')])], None, ModelRequestParameters()
        ) as stream:
            async for _ in stream:
                pass
    finally:
        await client.aclose()

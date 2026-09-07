from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Literal, TypeAlias

import httpx2
import pytest

from pydantic_ai import Agent, ModelRequest
from pydantic_ai._utils import is_str_dict
from pydantic_ai.direct import model_request_stream_sync, model_request_sync

from .conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    _LoopBoundTransport: TypeAlias = tuple[httpx2.AsyncClient, list[tuple[bool, asyncio.AbstractEventLoop]]]

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


@pytest.fixture
def anthropic_loop_bound_transport() -> _LoopBoundTransport:
    requests: list[tuple[bool, asyncio.AbstractEventLoop]] = []

    class ResponseBody(httpx2.AsyncByteStream):
        def __init__(self, content: bytes) -> None:
            self.content = content

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield self.content

    class Transport(httpx2.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
            payload: object = json.loads(request.content)
            assert is_str_dict(payload)
            stream = payload.get('stream') is True
            requests.append((stream, asyncio.get_running_loop()))

            if stream:
                events: list[tuple[str, dict[str, object]]] = [
                    (
                        'message_start',
                        {
                            'type': 'message_start',
                            'message': {
                                'id': 'msg_stream',
                                'type': 'message',
                                'role': 'assistant',
                                'model': 'claude-test',
                                'content': [],
                                'stop_reason': None,
                                'stop_sequence': None,
                                'usage': {'input_tokens': 1, 'output_tokens': 0},
                            },
                        },
                    ),
                    (
                        'content_block_start',
                        {
                            'type': 'content_block_start',
                            'index': 0,
                            'content_block': {'type': 'text', 'text': ''},
                        },
                    ),
                    (
                        'content_block_delta',
                        {
                            'type': 'content_block_delta',
                            'index': 0,
                            'delta': {'type': 'text_delta', 'text': 'blue'},
                        },
                    ),
                    ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
                    (
                        'message_delta',
                        {
                            'type': 'message_delta',
                            'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                            'usage': {'output_tokens': 1},
                        },
                    ),
                    ('message_stop', {'type': 'message_stop'}),
                ]
                body = ''.join(f'event: {event}\ndata: {json.dumps(data)}\n\n' for event, data in events).encode()
                content_type = 'text/event-stream'
            else:
                body = json.dumps(
                    {
                        'id': 'msg_sync',
                        'type': 'message',
                        'role': 'assistant',
                        'model': 'claude-test',
                        'content': [{'type': 'text', 'text': 'green'}],
                        'stop_reason': 'end_turn',
                        'stop_sequence': None,
                        'usage': {'input_tokens': 1, 'output_tokens': 1},
                    }
                ).encode()
                content_type = 'application/json'

            return httpx2.Response(
                200,
                headers={'content-type': content_type},
                stream=ResponseBody(body),
                request=request,
            )

    http_client = httpx2.AsyncClient(transport=Transport())
    return http_client, requests


@pytest.mark.parametrize(
    ('api_surface', 'stream_first', 'use_history'),
    [
        pytest.param('agent', False, False, id='agent-run-sync-then-stream'),
        pytest.param('agent', False, True, id='agent-run-sync-then-stream-with-history'),
        pytest.param('agent', True, False, id='agent-stream-then-run-sync'),
        pytest.param('direct', False, False, id='direct-request-sync-then-stream'),
        pytest.param('direct', True, False, id='direct-stream-then-request-sync'),
    ],
)
@pytest.mark.parametrize('client_owner', ['provider', 'user'])
def test_sync_entry_points_keep_async_client_on_one_event_loop(
    allow_model_requests: None,
    anthropic_loop_bound_transport: _LoopBoundTransport,
    api_surface: Literal['agent', 'direct'],
    stream_first: bool,
    use_history: bool,
    client_owner: Literal['provider', 'user'],
) -> None:
    """A persistent transport exposes loop identity directly, which no HTTP recording can retain."""
    http_client, requests = anthropic_loop_bound_transport
    if client_owner == 'provider':
        provider = AnthropicProvider(api_key='test', http_client=http_client)
        client = provider.client
    else:
        client = AsyncAnthropic(api_key='test', http_client=http_client, max_retries=0)
        provider = AnthropicProvider(anthropic_client=client)
    client.max_retries = 0
    model = AnthropicModel('claude-test', provider=provider)

    try:
        if api_surface == 'agent':
            agent = Agent(model)
            if stream_first:
                with agent.run_stream_sync('first') as stream:
                    streamed_output = ''.join(stream.stream_text(debounce_by=None))
                run_output = agent.run_sync('second').output
            else:
                first = agent.run_sync('first')
                history = first.all_messages() if use_history else None
                run_output = first.output
                with agent.run_stream_sync('second', message_history=history) as stream:
                    streamed_output = ''.join(stream.stream_text(debounce_by=None))

            assert run_output == 'green'
            assert streamed_output == 'blue'
        else:
            messages = [ModelRequest.user_text_prompt('test')]
            if stream_first:
                with model_request_stream_sync(model, messages) as stream:
                    stream_events = list(stream)
                response = model_request_sync(model, messages)
            else:
                response = model_request_sync(model, messages)
                with model_request_stream_sync(model, messages) as stream:
                    stream_events = list(stream)

            assert response.parts
            assert stream_events
        assert len(requests) == 2
        assert [stream for stream, _loop in requests] == ([True, False] if stream_first else [False, True])
        assert requests[0][1] is requests[1][1]
    finally:
        asyncio.get_event_loop().run_until_complete(client.close())


@pytest.mark.anyio
async def test_async_run_and_stream_share_one_event_loop(
    allow_model_requests: None,
    anthropic_loop_bound_transport: _LoopBoundTransport,
) -> None:
    """Fully async requests use one event loop as a control for the synchronous bridge tests."""
    http_client, requests = anthropic_loop_bound_transport
    client = AsyncAnthropic(api_key='test', http_client=http_client, max_retries=0)
    model = AnthropicModel('claude-test', provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(model)

    try:
        first = await agent.run('first')
        async with agent.run_stream('second', message_history=first.all_messages()) as stream:
            streamed_output = ''.join([text async for text in stream.stream_text(debounce_by=None)])

        assert first.output == 'green'
        assert streamed_output == 'blue'
        assert len(requests) == 2
        assert [stream for stream, _loop in requests] == [False, True]
        assert requests[0][1] is requests[1][1]
    finally:
        await client.close()

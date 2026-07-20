from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Literal

import pytest

from pydantic_ai import Agent, ModelRequest
from pydantic_ai._utils import is_str_dict
from pydantic_ai.direct import model_request_stream_sync, model_request_sync

from .conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


@pytest.fixture
def anthropic_keepalive_server() -> Iterator[tuple[str, list[tuple[bool, int]]]]:
    requests: list[tuple[bool, int]] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_POST(self) -> None:
            size = int(self.headers.get('content-length', 0))
            payload: object = json.loads(self.rfile.read(size))
            assert is_str_dict(payload)
            stream = payload.get('stream') is True
            requests.append((stream, self.client_address[1]))

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

            self.send_response(200)
            self.send_header('content-type', content_type)
            self.send_header('content-length', str(len(body)))
            self.send_header('connection', 'keep-alive')
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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
    anthropic_keepalive_server: tuple[str, list[tuple[bool, int]]],
    api_surface: Literal['agent', 'direct'],
    stream_first: bool,
    use_history: bool,
    client_owner: Literal['provider', 'user'],
) -> None:
    """A real keep-alive connection is required because VCR does not retain asyncio transport state."""
    base_url, requests = anthropic_keepalive_server
    if client_owner == 'provider':
        provider = AnthropicProvider(api_key='test', base_url=base_url)
        client = provider.client
    else:
        client = AsyncAnthropic(api_key='test', base_url=base_url, max_retries=0)
        provider = AnthropicProvider(anthropic_client=client)
    client.max_retries = 0
    model = AnthropicModel('claude-test', provider=provider)

    try:
        if api_surface == 'agent':
            agent = Agent(model)
            if stream_first:
                with agent.run_stream_sync('first', model_settings={'timeout': 1}) as stream:
                    streamed_output = ''.join(stream.stream_text(debounce_by=None))
                run_output = agent.run_sync('second', model_settings={'timeout': 1}).output
            else:
                first = agent.run_sync('first', model_settings={'timeout': 1})
                history = first.all_messages() if use_history else None
                run_output = first.output
                with agent.run_stream_sync('second', message_history=history, model_settings={'timeout': 1}) as stream:
                    streamed_output = ''.join(stream.stream_text(debounce_by=None))

            assert run_output == 'green'
            assert streamed_output == 'blue'
        else:
            messages = [ModelRequest.user_text_prompt('test')]
            if stream_first:
                with model_request_stream_sync(model, messages, model_settings={'timeout': 1}) as stream:
                    stream_events = list(stream)
                response = model_request_sync(model, messages, model_settings={'timeout': 1})
            else:
                response = model_request_sync(model, messages, model_settings={'timeout': 1})
                with model_request_stream_sync(model, messages, model_settings={'timeout': 1}) as stream:
                    stream_events = list(stream)

            assert response.parts
            assert stream_events
        assert len(requests) == 2
        assert [stream for stream, _port in requests] == ([True, False] if stream_first else [False, True])
        assert requests[0][1] == requests[1][1]
    finally:
        asyncio.get_event_loop().run_until_complete(client.close())


@pytest.mark.anyio
async def test_async_run_and_stream_share_one_event_loop(
    allow_model_requests: None,
    anthropic_keepalive_server: tuple[str, list[tuple[bool, int]]],
) -> None:
    """Fully async requests retain their keep-alive connection when both use one event loop."""
    base_url, requests = anthropic_keepalive_server
    client = AsyncAnthropic(api_key='test', base_url=base_url, max_retries=0)
    model = AnthropicModel('claude-test', provider=AnthropicProvider(anthropic_client=client))
    agent = Agent(model)

    try:
        first = await agent.run('first', model_settings={'timeout': 1})
        async with agent.run_stream(
            'second', message_history=first.all_messages(), model_settings={'timeout': 1}
        ) as stream:
            streamed_output = ''.join([text async for text in stream.stream_text(debounce_by=None)])

        assert first.output == 'green'
        assert streamed_output == 'blue'
        assert len(requests) == 2
        assert requests == [(False, requests[0][1]), (True, requests[0][1])]
    finally:
        await client.close()

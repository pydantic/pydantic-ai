"""Shared Responses WebSocket frame/socket fixtures and an HTTP fallback for durable model units."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import anyio
import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

try:
    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
except ImportError:  # pragma: lax no cover
    pytest.skip('openai not installed', allow_module_level=True)

Frame = dict[str, JsonValue]
FRAME_ADAPTER = TypeAdapter(Frame)
_COMPLETED: Frame = {
    'id': 'resp_http',
    'model': 'gpt-4o',
    'object': 'response',
    'created_at': 1704067200,
    'status': 'completed',
    'output': [
        {
            'type': 'message',
            'id': 'msg_http',
            'role': 'assistant',
            'status': 'completed',
            'content': [{'type': 'output_text', 'text': 'from HTTP', 'annotations': [], 'logprobs': []}],
        }
    ],
    'parallel_tool_calls': True,
    'tool_choice': 'auto',
    'tools': [],
    'usage': {'input_tokens': 5, 'output_tokens': 1, 'total_tokens': 6},
}
_STREAM: list[Frame] = [
    {
        'type': 'response.created',
        'sequence_number': 0,
        'response': {**_COMPLETED, 'status': 'in_progress', 'output': []},
    },
    {
        'type': 'response.output_item.added',
        'sequence_number': 1,
        'output_index': 0,
        'item': {'type': 'message', 'id': 'msg_http', 'role': 'assistant', 'status': 'in_progress', 'content': []},
    },
    {
        'type': 'response.output_text.delta',
        'sequence_number': 2,
        'output_index': 0,
        'content_index': 0,
        'item_id': 'msg_http',
        'delta': 'from HTTP',
        'logprobs': [],
    },
    {'type': 'response.completed', 'sequence_number': 3, 'response': _COMPLETED},
]


def responses_reply(request: httpx2.Request) -> httpx2.Response:
    """Answer an HTTP Responses API request with `from HTTP`, streamed when it asks for a stream."""
    if not FRAME_ADAPTER.validate_json(request.content).get('stream'):
        return httpx2.Response(200, json=_COMPLETED)
    body = ''.join(f'event: {event["type"]}\ndata: {FRAME_ADAPTER.dump_json(event).decode()}\n\n' for event in _STREAM)
    return httpx2.Response(200, content=body.encode(), headers={'content-type': 'text/event-stream'})


def websocket_model(http_client: httpx2.AsyncClient) -> OpenAIResponsesModel:
    # The cassette fixture lets requests to localhost through to the mock transport.
    return OpenAIResponsesModel(
        'gpt-4o',
        provider=OpenAIProvider(base_url='http://localhost/v1', api_key='test', http_client=http_client),
        settings=OpenAIResponsesModelSettings(openai_transport='websocket'),
    )


def completed_response(response_id: str, *, store: bool | None = None, output: list[JsonValue] | None = None) -> Frame:
    response: Frame = {
        'id': response_id,
        'model': 'gpt-4o',
        'object': 'response',
        'created_at': 1704067200,
        'status': 'completed',
        'output': output or [],
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'usage': {'input_tokens': 5, 'output_tokens': 1, 'total_tokens': 6},
    }
    if store is not None:
        response['store'] = store
    return response


def text_events(text: str = 'ready', response_id: str = 'resp_test', *, store: bool | None = None) -> list[Frame]:
    response = completed_response(response_id, store=store)
    return [
        {'type': 'response.created', 'sequence_number': 0, 'response': {**response, 'status': 'in_progress'}},
        {
            'type': 'response.output_item.added',
            'sequence_number': 1,
            'output_index': 0,
            'item': {'type': 'message', 'id': 'msg_test', 'role': 'assistant', 'status': 'in_progress', 'content': []},
        },
        {
            'type': 'response.output_text.delta',
            'sequence_number': 2,
            'output_index': 0,
            'content_index': 0,
            'item_id': 'msg_test',
            'delta': text,
            'logprobs': [],
        },
        {'type': 'response.completed', 'sequence_number': 3, 'response': response},
    ]


def on_lane(events: list[Frame], stream_id: str) -> list[Frame]:
    return [{**event, 'stream_id': stream_id} for event in events]


@dataclass
class ScriptedSocket:
    """Control interruptions and concurrency beneath the real SDK event parser.

    Not a cassette: the tests using it need lanes interleaved on cue, transport failures, and provider errors at
    chosen points, which a live recording can't produce on demand. The recorded tests cover the real API's frames.
    """

    responses: deque[list[Frame]] = field(default_factory=lambda: deque([text_events()]))
    sent: list[Frame] = field(default_factory=list[Frame])
    incoming: deque[Frame | Exception] = field(default_factory=lambda: deque[Frame | Exception]())
    receiving: anyio.Event = field(default_factory=anyio.Event)
    waiting_for_input: anyio.Event = field(default_factory=anyio.Event)
    available: anyio.Event = field(default_factory=anyio.Event)
    sending: anyio.Event = field(default_factory=anyio.Event)
    send_gate: anyio.Event | None = None
    close_count: int = 0
    send_error: Exception | None = None

    async def send(self, data: str | bytes) -> None:
        self.sending.set()
        if self.send_error is not None:
            raise self.send_error
        if self.send_gate is not None:
            await self.send_gate.wait()
        frame = FRAME_ADAPTER.validate_json(data)
        self.sent.append(frame)
        if self.responses:
            events = self.responses.popleft()
            # The server echoes a request's `stream_id` on every event of its response.
            stream_id = frame.get('stream_id')
            self.push(*(on_lane(events, stream_id) if isinstance(stream_id, str) else events))

    def push(self, *events: Frame | Exception) -> None:
        self.incoming.extend(events)
        self.available.set()

    async def recv(self, *, decode: bool | None = None) -> str | bytes:
        self.receiving.set()
        while not self.incoming:
            if self.close_count:
                # HTTP-only durable fixtures import this module without the WebSocket extra.
                from websockets.exceptions import ConnectionClosedOK

                raise ConnectionClosedOK(None, None)
            self.available = anyio.Event()
            self.waiting_for_input.set()
            await self.available.wait()
        event = self.incoming.popleft()
        if isinstance(event, Exception):
            raise event
        raw = FRAME_ADAPTER.dump_json(event)
        return raw if decode is False else raw.decode()

    async def close(self, *, code: int = 1000, reason: str = '') -> None:
        self.close_count += 1
        self.available.set()

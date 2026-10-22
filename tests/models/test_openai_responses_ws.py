from __future__ import annotations

import asyncio
import hashlib
import hmac
import sys
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from copy import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, get_type_hints
from unittest.mock import Mock

import anyio
import httpx2
import pytest
from _pytest.fixtures import SubRequest
from httpx import Timeout
from pydantic import BaseModel, JsonValue, TypeAdapter
from typing_extensions import Unpack

from pydantic_ai import (
    Agent,
    CompactionPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RunContext,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import Hooks, SelectModel
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelRateLimitError,
    ModelServerError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.messages import ModelResponsePart
from pydantic_ai.models import AbstractModel, Model, ModelRequestContext, ModelRequestParameters, ModelSelectionContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import NativeOutput

from ..conftest import try_import
from ..durable_exec.test_durable_operation import JournalDurability
from ..realtime.ws_cassettes import (
    CassetteMessage,
    RealtimeCassette,
    RecordingWebSocket,
    ReplayWebSocket,
    realtime_cassette_plan,
)

with try_import() as imports_successful:
    import logfire
    from logfire.testing import CaptureLogfire
    from openai import APIError, AsyncAzureOpenAI, AsyncOpenAI
    from openai._base_client import platform_headers
    from openai.types.websocket_connection_options import WebSocketConnectionOptions
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.asyncio.server import ServerConnection, serve
    from websockets.datastructures import Headers
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidStatus
    from websockets.http11 import Response as HandshakeResponse

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')
READINESS_WAIT_TIMEOUT = 10
Frame = dict[str, JsonValue]
FRAME_ADAPTER = TypeAdapter(Frame)

with try_import() as bedrock_mantle_imports_successful:
    from pydantic_ai.models.bedrock_mantle import BedrockMantleResponsesModel
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider


@pytest.fixture
def anyio_backend() -> str:
    """The Responses WebSocket connector uses asyncio."""
    return 'asyncio'


@pytest.fixture(autouse=True)
def stable_platform_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid the SDK's macOS platform probe spawning a subprocess inside async tests.

    The SDK caches platform headers process-wide by the client's resolved platform, which is unset
    until its first HTTP request. Pinning the key keeps headers computed before and after that
    request equal, even when an earlier test cached the real platform for the unset key.
    """

    def unknown_platform_headers(version: str, *, platform: object) -> dict[str, str]:
        return platform_headers(version, platform='Unknown')

    monkeypatch.setattr('openai._base_client.get_platform', lambda: 'Unknown')
    monkeypatch.setattr('openai._base_client.platform_headers', unknown_platform_headers)


@dataclass
class RecordedResponses:
    model: OpenAIResponsesModel
    cassette: RealtimeCassette
    connections: list[str] = field(default_factory=list[str])
    cassettes: list[RealtimeCassette] = field(default_factory=list[RealtimeCassette])


@pytest.fixture
def recorded_responses(
    request: SubRequest, monkeypatch: pytest.MonkeyPatch, openai_api_key: str
) -> Iterator[RecordedResponses]:
    path = Path(__file__).parent / 'cassettes' / Path(__file__).stem / f'{request.node.name}.yaml'
    multiple_connections = request.node.name == 'test_cross_socket_continuation_replays_history'
    mode = request.config.getoption('record_mode')
    assert mode is None or isinstance(mode, str)
    plan = realtime_cassette_plan(cassette_exists=path.exists(), record_mode=mode)
    if plan == 'error_missing':  # pragma: no cover
        raise RuntimeError(f'Record the Responses WebSocket cassette with `--record-mode=rewrite`: {path}')
    cassette = RealtimeCassette.load(path) if plan == 'replay' else RealtimeCassette()
    recorded = RecordedResponses(
        OpenAIResponsesModel('gpt-6-astra', provider=OpenAIProvider(api_key=openai_api_key)), cassette
    )
    if multiple_connections and plan == 'replay':
        # This test sends one response on each socket. Keep both recordings in the
        # test-named cassette and split at the first terminal event for replay.
        boundary = next(
            index + 1
            for index, interaction in enumerate(cassette.interactions)
            if isinstance(interaction, CassetteMessage)
            and interaction.direction == 'received'
            and interaction.data.get('type') == 'response.completed'
        )
        recorded.cassettes = [
            RealtimeCassette(interactions=cassette.interactions[:boundary]),
            RealtimeCassette(interactions=cassette.interactions[boundary:]),
        ]

    async def connect(
        uri: str,
        *,
        additional_headers: Mapping[str, str],
        user_agent_header: str | None,
        **options: Unpack[WebSocketConnectionOptions],
    ) -> ReplayWebSocket | RecordingWebSocket:
        recorded.connections.append(uri)
        assert 'authorization' in {key.lower() for key in additional_headers}
        if multiple_connections:
            if plan == 'replay':
                connection_cassette = recorded.cassettes[len(recorded.connections) - 1]
            else:
                connection_cassette = RealtimeCassette()  # pragma: no cover
                recorded.cassettes.append(connection_cassette)  # pragma: no cover
        else:
            connection_cassette = cassette
        if plan == 'replay':
            return ReplayWebSocket(connection_cassette)
        # Only runs while recording.
        ws = await websocket_connect(  # pragma: no cover
            uri, additional_headers=additional_headers, user_agent_header=user_agent_header, **options
        )
        return RecordingWebSocket(ws, connection_cassette)  # pragma: no cover

    monkeypatch.setattr('openai.lib._websocket._WebSocketConnect', connect)
    try:
        yield recorded
    finally:
        if plan == 'record':  # pragma: no cover
            if multiple_connections:
                assert len(recorded.cassettes) == 2
                cassette.interactions = [
                    interaction
                    for connection_cassette in recorded.cassettes
                    for interaction in connection_cassette.interactions
                ]
                cassette.dump(path)
            elif cassette.interactions:
                cassette.dump(path)


class Weather(BaseModel):
    temperature: int


def test_connect_type_hints_are_resolvable():
    assert (
        get_type_hints(OpenAIResponsesModel.connect)['websocket_connection_options']
        == WebSocketConnectionOptions | None
    )


async def test_azure_connection_is_rejected_before_handshake(sockets: SocketHarness):
    async with AsyncAzureOpenAI(
        api_key='test',
        api_version='2024-12-01-preview',
        azure_endpoint='https://example.openai.azure.com/',
    ) as client:
        model = OpenAIResponsesModel('gpt-4o', provider=AzureProvider(openai_client=client))
        with pytest.raises(UserError, match='not supported with Azure OpenAI'):
            async with model.connect():
                pass
    assert sockets.connect_count == 0


async def test_tool_continuation(allow_model_requests: None, recorded_responses: RecordedResponses):
    """A real tool roundtrip reuses one socket and sends only the tool result on continuation."""
    calls: list[str] = []

    def get_weather(city: str) -> int:
        """Get the current temperature in Celsius for a city."""
        calls.append(city)
        return 18

    settings: OpenAIResponsesModelSettings = {
        'openai_previous_response_id': 'auto',
        'openai_store': False,
        'openai_reasoning_effort': 'low',
    }
    async with recorded_responses.model.connect() as connected:
        agent = Agent(connected, tools=[get_weather], output_type=NativeOutput(Weather), model_settings=settings)
        result = await agent.run('Call get_weather for Paris exactly once, then report its temperature.')
    assert calls == ['Paris']
    assert result.output == Weather(temperature=18)
    assert result.usage.input_tokens > 0
    assert result.usage.output_tokens > 0
    assert recorded_responses.connections == ['wss://api.openai.com/v1/responses']

    messages = result.all_messages()
    responses = [message for message in messages if isinstance(message, ModelResponse)]
    assert len(responses) == 2
    assert all(response.provider_response_id for response in responses)
    tool_call = next(part for part in responses[0].parts if isinstance(part, ToolCallPart))
    tool_return = next(
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )
    assert tool_call.tool_call_id == tool_return.tool_call_id
    sent = [
        event.data
        for event in recorded_responses.cassette.interactions
        if isinstance(event, CassetteMessage) and event.direction == 'sent'
    ]
    assert len(sent) == 2
    assert sent[0]['type'] == 'response.create'
    assert sent[0]['model'] == 'gpt-6-astra'
    assert sent[0]['input'] == [
        {'role': 'user', 'content': 'Call get_weather for Paris exactly once, then report its temperature.'}
    ]
    assert 'previous_response_id' not in sent[0]
    assert sent[1]['previous_response_id'] == responses[0].provider_response_id
    assert sent[1]['input'] == [{'type': 'function_call_output', 'call_id': tool_call.tool_call_id, 'output': '18'}]
    assert all(frame['store'] is False for frame in sent)
    assert all(
        frame['type'] == 'response.create' and 'stream' not in frame and 'background' not in frame for frame in sent
    )
    assert sent[0]['text']['format']['type'] == 'json_schema'


async def test_streaming_sequential_turns(allow_model_requests: None, recorded_responses: RecordedResponses):
    """Ordinary full-history streaming works across sequential turns on the same socket."""
    async with recorded_responses.model.connect() as connected:
        agent = Agent(connected)
        async with agent.run_stream('Reply with the word amber.') as first:
            chunks = [chunk async for chunk in first.stream_text(delta=True)]
            first_output = await first.get_output()
            history = first.all_messages()
        async with agent.run_stream('Reply with the word cobalt.', message_history=history) as second:
            second_output = await second.get_output()
            assert second.usage.output_tokens > 0
    assert chunks
    assert 'amber' in first_output.lower()
    assert 'cobalt' in second_output.lower()
    assert recorded_responses.connections == ['wss://api.openai.com/v1/responses']
    sent = [
        event.data
        for event in recorded_responses.cassette.interactions
        if isinstance(event, CassetteMessage) and event.direction == 'sent'
    ]
    assert len(sent) == 2
    assert sent[0]['type'] == 'response.create'
    assert sent[0]['model'] == 'gpt-6-astra'
    assert sent[0]['input'] == [{'role': 'user', 'content': 'Reply with the word amber.'}]
    assert all('previous_response_id' not in frame for frame in sent)
    assert len(sent[1]['input']) > len(sent[0]['input'])
    assert 'amber' in str(sent[1]['input']).lower()


async def test_overlapping_runs_continue_on_their_own_lanes(
    allow_model_requests: None, recorded_responses: RecordedResponses
):
    """A run that overlaps another takes a named lane, and each lane continues from its own store-false response."""
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'openai_store': False}
    async with recorded_responses.model.connect() as connected:
        agent = Agent(connected, model_settings=settings)
        async with agent.run_stream('Reply with the word amber.') as first:
            chunks = first.stream_text(delta=True)
            await anext(chunks)
            # The first response's stream is still open, so the default lane is busy.
            second = await agent.run('Reply with the word cobalt.')
            async for _ in chunks:
                pass
            first_history = first.all_messages()
        third = await agent.run('Reply with the word amber again.', message_history=first_history)
        fourth = await agent.run('Reply with the word cobalt again.', message_history=second.all_messages())

    assert 'cobalt' in second.output.lower()
    assert 'amber' in third.output.lower()
    assert 'cobalt' in fourth.output.lower()
    assert recorded_responses.connections == ['wss://api.openai.com/v1/responses']
    first_response = first_history[-1]
    second_response = second.all_messages()[-1]
    assert isinstance(first_response, ModelResponse) and isinstance(second_response, ModelResponse)
    frames = [event for event in recorded_responses.cassette.interactions if isinstance(event, CassetteMessage)]
    sent = [frame.data for frame in frames if frame.direction == 'sent']
    assert [(frame.get('stream_id'), frame.get('previous_response_id')) for frame in sent] == [
        (None, None),
        ('lane-1', None),
        (None, first_response.provider_response_id),
        ('lane-1', second_response.provider_response_id),
    ]
    assert sent[2]['input'] == [{'role': 'user', 'content': 'Reply with the word amber again.'}]
    # OpenAI echoes a named lane on every event of its responses and omits it on the default lane.
    received_lanes = {frame.data.get('stream_id') for frame in frames if frame.direction == 'received'}
    assert received_lanes == {None, 'lane-1'}


async def test_cross_socket_continuation_replays_history(
    allow_model_requests: None, recorded_responses: RecordedResponses
):
    """A store-false response ID from a closed socket cannot continue on another socket."""
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'openai_store': False}
    async with recorded_responses.model.connect() as first_model:
        first = await Agent(first_model, model_settings=settings).run('Reply with the word amber.')
    async with recorded_responses.model.connect() as second_model:
        second = await Agent(second_model, model_settings=settings).run(
            'Reply with the word cobalt.', message_history=first.all_messages()
        )

    assert 'amber' in first.output.lower()
    assert 'cobalt' in second.output.lower()
    assert len(recorded_responses.connections) == 2
    sent = [
        event.data
        for cassette in recorded_responses.cassettes
        for event in cassette.interactions
        if isinstance(event, CassetteMessage) and event.direction == 'sent'
    ]
    assert len(sent) == 2
    assert all(frame['type'] == 'response.create' and 'previous_response_id' not in frame for frame in sent)
    assert sent[0]['input'] == [{'role': 'user', 'content': 'Reply with the word amber.'}]
    assert len(sent[1]['input']) > len(sent[0]['input'])
    assert 'amber' in str(sent[1]['input']).lower()


def text_events(text: str = 'ready', response_id: str = 'resp_test') -> list[Frame]:
    response: Frame = {
        'id': response_id,
        'model': 'gpt-4o',
        'object': 'response',
        'created_at': 1704067200,
        'status': 'completed',
        'output': [],
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'usage': {'input_tokens': 5, 'output_tokens': 1, 'total_tokens': 6},
    }
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


def empty_completed_events(response_id: str) -> list[Frame]:
    response: Frame = {
        'id': response_id,
        'model': 'gpt-4o',
        'object': 'response',
        'created_at': 1704067200,
        'status': 'completed',
        'output': [],
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'usage': {'input_tokens': 2, 'output_tokens': 0, 'total_tokens': 2},
    }
    return [
        {'type': 'response.created', 'sequence_number': 0, 'response': {**response, 'status': 'in_progress'}},
        {'type': 'response.completed', 'sequence_number': 1, 'response': response},
    ]


def tool_call_events(response_id: str, call_id: str) -> list[Frame]:
    item: Frame = {
        'type': 'function_call',
        'id': 'fc_test',
        'status': 'completed',
        'arguments': '{}',
        'call_id': call_id,
        'name': 'step',
    }
    events = empty_completed_events(response_id)
    response = events[-1]['response']
    assert isinstance(response, dict)
    response['output'] = [item]
    events[-1]['response'] = response
    events[-1:-1] = [
        {
            'type': 'response.output_item.added',
            'sequence_number': 1,
            'output_index': 0,
            'item': {**item, 'status': 'in_progress', 'arguments': ''},
        },
        {'type': 'response.output_item.done', 'sequence_number': 2, 'output_index': 0, 'item': item},
    ]
    return events


def on_lane(events: list[Frame], stream_id: str) -> list[Frame]:
    return [{**event, 'stream_id': stream_id} for event in events]


@dataclass
class ScriptedSocket:
    """Control interruptions and concurrency beneath the real SDK event parser."""

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


@dataclass
class SocketHarness:
    pending: deque[ScriptedSocket] = field(default_factory=lambda: deque([ScriptedSocket()]))
    opened: list[ScriptedSocket] = field(default_factory=list[ScriptedSocket])
    urls: list[str] = field(default_factory=list[str])
    headers: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    authorization_headers: list[str] = field(default_factory=list[str])
    options: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    connecting: anyio.Event = field(default_factory=anyio.Event)
    connect_count: int = 0
    connect_gate: anyio.Event | None = None
    connect_error: Exception | None = None


@pytest.fixture
def sockets(monkeypatch: pytest.MonkeyPatch) -> SocketHarness:
    harness = SocketHarness()

    async def connect(uri: str, *, additional_headers: Mapping[str, str], **options: object) -> ScriptedSocket:
        harness.connect_count += 1
        harness.connecting.set()
        if harness.connect_error is not None:
            raise harness.connect_error
        if harness.connect_gate is not None:
            await harness.connect_gate.wait()
        harness.urls.append(uri)
        headers = {key.lower(): value for key, value in additional_headers.items()}
        authorization = headers.pop('authorization')
        assert authorization.startswith('Bearer ')
        harness.authorization_headers.append(authorization)
        harness.headers.append(headers)
        harness.options.append(options)
        socket = harness.pending.popleft()
        harness.opened.append(socket)
        return socket

    monkeypatch.setattr('openai.lib._websocket._WebSocketConnect', connect)
    return harness


@pytest.mark.parametrize('stream', [False, True])
async def test_connected_fallback_survives_primary_errors_across_tool_round_trips(
    allow_model_requests: None, sockets: SocketHarness, stream: bool
):
    """A provider error ends only its request, so fallback keeps switching on every request of a tool run."""
    first_socket = sockets.pending[0]
    unavailable: Frame = {
        'type': 'error',
        'status': 503,
        'error': {'type': 'server_error', 'code': 'unavailable', 'message': 'primary failed'},
    }
    first_socket.responses = deque([[unavailable], [unavailable]])
    second_socket = ScriptedSocket(
        responses=deque([tool_call_events('resp_tool', 'call_1'), text_events(response_id='resp_final')])
    )
    sockets.pending.append(second_socket)
    primary = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    secondary = OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key='test'))

    async with AsyncExitStack() as stack:
        connected_primary = await stack.enter_async_context(primary.connect())
        connected_secondary = await stack.enter_async_context(secondary.connect())
        agent = Agent(FallbackModel(connected_primary, connected_secondary))

        @agent.tool_plain
        def step() -> str:
            return 'done'

        if stream:
            async with agent.run_stream('hello') as result:
                assert await result.get_output() == 'ready'
        else:
            assert (await agent.run('hello')).output == 'ready'
        assert [socket.close_count for socket in sockets.opened] == [0, 0]

    assert sockets.opened == [first_socket, second_socket]
    assert [len(socket.sent) for socket in sockets.opened] == [2, 2]
    assert [socket.close_count for socket in sockets.opened] == [1, 1]


@pytest.mark.parametrize('owned_http_client', [False, True])
async def test_independent_lifetimes(
    allow_model_requests: None, sockets: SocketHarness, monkeypatch: pytest.MonkeyPatch, owned_http_client: bool
):
    """The connection context owns the socket; model and wrapper contexts borrow it."""
    requests: list[httpx2.Request] = []

    def http_handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        response = text_events()[-1]['response']
        return httpx2.Response(200, json=response)

    sockets.pending[0].responses.extend([text_events(response_id='resp_next')])
    sockets.pending.append(ScriptedSocket())
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        if owned_http_client:
            monkeypatch.setattr(
                'pydantic_ai.providers._openai_compatible.create_async_httpx2_client', lambda: http_client
            )
            provider = OpenAIProvider(api_key='test')
        else:
            provider = OpenAIProvider(api_key='test', http_client=http_client)
        source = OpenAIResponsesModel('gpt-4o', provider=provider)
        async with source.connect() as connected, source.connect() as independent:
            assert connected is not source and independent is not connected
            # A direct request avoids output validation: the HTTP stub only needs a complete response.
            await source.request([ModelRequest(parts=[UserPromptPart('hello')])], None, ModelRequestParameters())
            assert requests[0].url.path == '/v1/responses'
            async with Agent(WrapperModel(connected)) as agent:
                assert (await agent.run('hello')).output == 'ready'
            assert sockets.opened[0].close_count == 0
            assert not http_client.is_closed
            assert (await Agent(connected).run('again')).output == 'ready'
            assert (await Agent(independent).run('hello')).output == 'ready'
        assert [socket.close_count for socket in sockets.opened] == [1, 1]
        with pytest.raises(UserError, match='closed'):
            await Agent(connected).run('after close')
        assert len(sockets.opened[0].sent) == 2
        await source.request([ModelRequest(parts=[UserPromptPart('after close')])], None, ModelRequestParameters())
        assert not http_client.is_closed
        async with source:
            await source.request([ModelRequest(parts=[UserPromptPart('HTTP context')])], None, ModelRequestParameters())
        assert http_client.is_closed is owned_http_client


async def test_context_exit_interrupts_request(allow_model_requests: None, sockets: SocketHarness):
    socket = sockets.pending[0]
    socket.responses.clear()
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    before = asyncio.all_tasks()
    async with source, anyio.create_task_group() as tasks:
        async with source.connect() as connected:

            async def request() -> None:
                with pytest.raises(ModelAPIError, match='interrupted before completion'):
                    await Agent(connected).run('hello')

            tasks.start_soon(request)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await socket.receiving.wait()
        assert socket.close_count == 1
    assert asyncio.all_tasks() == before


async def test_concurrent_requests_share_the_socket_on_separate_lanes(
    allow_model_requests: None, sockets: SocketHarness
):
    """An overlapping run takes a named lane on the same socket, and the two responses' events can interleave."""
    socket = sockets.pending[0]
    socket.responses.clear()
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    results: dict[str, str] = {}

    async with source.connect() as connected:

        async def run(prompt: str) -> None:
            results[prompt] = (await Agent(connected).run(prompt)).output

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run, 'first')
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await socket.receiving.wait()
            socket.sending = anyio.Event()
            tasks.start_soon(run, 'second')
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await socket.sending.wait()
            first = text_events('first', 'resp_first')
            socket.push(*first[:2], *on_lane(text_events('second', 'resp_second'), 'lane-1'), *first[2:])

    assert results == {'first': 'first', 'second': 'second'}
    assert [frame.get('stream_id') for frame in socket.sent] == [None, 'lane-1']
    assert sockets.opened == [socket]
    assert socket.close_count == 1


async def test_alternating_conversations_continue_on_their_own_lanes(
    allow_model_requests: None, sockets: SocketHarness
):
    """A new run starts fresh on another lane, and each conversation keeps continuing from its own lane's cache."""
    socket = sockets.pending[0]
    socket.responses = deque(
        [
            text_events(response_id='resp_a1'),
            text_events(response_id='resp_b1'),
            text_events(response_id='resp_a2'),
            text_events(response_id='resp_b2'),
        ]
    )
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto'}
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'), settings=settings)

    async with source.connect() as connected:
        agent = Agent(connected)
        first = await agent.run('a1')
        second = await agent.run('b1')
        await agent.run('a2', message_history=first.all_messages())
        await agent.run('b2', message_history=second.all_messages())

    assert [(frame.get('stream_id'), frame.get('previous_response_id'), frame['input']) for frame in socket.sent] == [
        (None, None, [{'role': 'user', 'content': 'a1'}]),
        ('lane-1', None, [{'role': 'user', 'content': 'b1'}]),
        (None, 'resp_a1', [{'role': 'user', 'content': 'a2'}]),
        ('lane-1', 'resp_b1', [{'role': 'user', 'content': 'b2'}]),
    ]


async def test_lane_limit_reuses_least_recently_used_lane(
    allow_model_requests: None, sockets: SocketHarness, monkeypatch: pytest.MonkeyPatch
):
    """Past the named-lane limit, a request takes the least recently used lane and that lane's conversation replays."""
    monkeypatch.setattr('pydantic_ai.models._openai_responses_websocket.NAMED_LANE_LIMIT', 1)
    socket = sockets.pending[0]
    socket.responses = deque(
        [
            text_events(response_id='resp_a1'),
            text_events(response_id='resp_b1'),
            text_events(response_id='resp_c1'),
            text_events(response_id='resp_a2'),
        ]
    )
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto'}
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'), settings=settings)

    async with source.connect() as connected:
        agent = Agent(connected)
        first = await agent.run('a1')
        await agent.run('b1')
        await agent.run('c1')
        await agent.run('a2', message_history=first.all_messages())

    assert [(frame.get('stream_id'), frame.get('previous_response_id')) for frame in socket.sent] == [
        (None, None),
        ('lane-1', None),
        (None, None),
        ('lane-1', None),
    ]
    assert socket.sent[-1]['input'] == [
        {'role': 'user', 'content': 'a1'},
        {'role': 'assistant', 'content': 'ready'},
        {'role': 'user', 'content': 'a2'},
    ]


async def test_waiting_for_a_lane_honors_the_pool_timeout(
    allow_model_requests: None, sockets: SocketHarness, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr('pydantic_ai.models._openai_responses_websocket.NAMED_LANE_LIMIT', 0)
    socket = sockets.pending[0]
    socket.responses.clear()
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    results: list[str] = []

    async with source.connect() as connected:

        async def first_run() -> None:
            results.append((await Agent(connected).run('first')).output)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(first_run)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await socket.receiving.wait()
            with pytest.raises(ModelAPIError, match='No Responses WebSocket lane became free'):
                await Agent(connected, model_settings={'timeout': 0.01}).run('waiting')
            socket.push(*text_events('first', 'resp_first'))

    assert results == ['first']
    assert len(socket.sent) == 1


async def test_abandoned_response_holds_its_lane_until_drained(
    allow_model_requests: None, sockets: SocketHarness, monkeypatch: pytest.MonkeyPatch
):
    """A stream closed early keeps its lane until the reader discards the rest of that response."""
    monkeypatch.setattr('pydantic_ai.models._openai_responses_websocket.NAMED_LANE_LIMIT', 0)
    socket = sockets.pending[0]
    abandoned = text_events('abandoned', 'resp_abandoned')
    socket.responses = deque([abandoned[:3], text_events(response_id='resp_next')])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))

    async with source.connect() as connected:
        async with connected.request_stream(
            [ModelRequest.user_text_prompt('hello')], None, ModelRequestParameters()
        ) as response:
            async for _ in response:  # pragma: no branch
                break
        socket.push(abandoned[2], abandoned[3])
        assert (await Agent(connected).run('next')).output == 'ready'
        assert socket.close_count == 0

    assert [frame.get('stream_id') for frame in socket.sent] == [None, None]


async def test_closing_a_response_after_its_terminal_was_routed_frees_its_lane(
    allow_model_requests: None, sockets: SocketHarness
):
    """Another lane's reader can buffer a whole response, so closing it early still frees the lane."""
    socket = sockets.pending[0]
    first = text_events('first', 'resp_first')
    second = on_lane(text_events('second', 'resp_second'), 'lane-1')
    socket.responses = deque([first[:1], second[:1], text_events(response_id='resp_next')])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))

    async with source.connect() as connected:
        async with connected.request_stream([ModelRequest.user_text_prompt('first')], None, ModelRequestParameters()):
            async with connected.request_stream(
                [ModelRequest.user_text_prompt('second')], None, ModelRequestParameters()
            ) as response:
                socket.push(*first[1:], *second[1:])
                async for _ in response:
                    pass
                assert response.get().text == 'second'
        assert (await Agent(connected).run('next')).output == 'ready'

    assert [frame.get('stream_id') for frame in socket.sent] == [None, 'lane-1', None]


@pytest.mark.parametrize('error', ['unattributed', 'connection_limit'])
async def test_connection_scoped_error_fails_the_socket(allow_model_requests: None, sockets: SocketHarness, error: str):
    """An error no active lane owns, or the connection's lifetime limit, fails every later request over."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events(response_id='resp_first')])
    code = None if error == 'unattributed' else 'websocket_connection_limit_reached'
    failure: Frame = {'type': 'error', 'error': {'type': 'invalid_request_error', 'code': code, 'message': 'ended'}}
    if error == 'connection_limit':
        failure['status'] = 400
        socket.responses.append([failure])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))

    async with source.connect() as connected:
        await Agent(connected).run('first')
        if error == 'unattributed':
            # Without a `stream_id`, these belong to the idle default lane rather than the named lane waiting.
            socket.push(text_events()[2], failure)
        expected = 'ended' if code is None else f'{code}: ended'
        with pytest.raises(ModelAPIError, match=f'Responses WebSocket connection failed: {expected}$'):
            await Agent(connected).run('second')
        assert socket.close_count == 1
        with pytest.raises(ModelAPIError, match='Responses WebSocket connection failed'):
            await Agent(connected).run('third')

    assert len(socket.sent) == 2


async def test_cancelled_send_keeps_the_socket_for_other_lanes(allow_model_requests: None, sockets: SocketHarness):
    """A cancelled send may already have reached the server, so only its lane waits out that response."""
    socket = sockets.pending[0]
    socket.send_gate = anyio.Event()
    socket.responses.clear()
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))

    async with source.connect() as connected:

        async def cancelled_run() -> None:
            await Agent(connected).run('cancelled')

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(cancelled_run)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await socket.sending.wait()
            tasks.cancel_scope.cancel()
        socket.send_gate = None
        # The scripted socket drops the gated frame, but a real send would have written it: its response arrives.
        socket.push(*text_events('late', response_id='resp_late'))
        socket.responses.append(text_events(response_id='resp_next'))
        assert (await Agent(connected).run('next')).output == 'ready'
        socket.responses.append(text_events('again', response_id='resp_again'))
        assert (await Agent(connected).run('again')).output == 'again'
        assert socket.close_count == 0

    # Once the late response was discarded, the default lane was free again.
    assert [frame.get('stream_id') for frame in socket.sent] == ['lane-1', None]


async def test_parallel_nested_runs_share_the_parent_connection(allow_model_requests: None, sockets: SocketHarness):
    """A tool can start parallel runs on the connected model its own run is using, each on its own lane."""
    socket = sockets.pending[0]
    socket.responses = deque(
        [tool_call_events('resp_parent', 'call_1'), [], [], text_events('done', response_id='resp_final')]
    )
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))

    async with source.connect() as connected:
        agent = Agent(connected)
        worker = Agent(connected)

        @agent.tool_plain
        async def step() -> str:
            outputs: list[str] = []

            async def run(prompt: str) -> None:
                outputs.append((await worker.run(prompt)).output)

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(run, 'first')
                tasks.start_soon(run, 'second')
                # Both nested requests are in flight before either response arrives.
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    while len(socket.sent) < 3:
                        socket.sending = anyio.Event()
                        await socket.sending.wait()
                socket.push(
                    *on_lane(text_events(response_id='resp_first'), 'lane-1'),
                    *on_lane(text_events(response_id='resp_second'), 'lane-2'),
                )
            return ' '.join(outputs)

        assert (await agent.run('go')).output == 'done'

    assert [frame.get('stream_id') for frame in socket.sent] == [None, 'lane-1', 'lane-2', None]
    assert sockets.opened == [socket]


async def test_history_provenance_across_transports(allow_model_requests: None, sockets: SocketHarness):
    """Auto continuation reuses IDs only on the same live WebSocket."""
    socket_a = sockets.pending[0]
    socket_a.responses = deque([text_events('from socket A', 'resp_socket_a')])
    socket_b = ScriptedSocket(
        responses=deque([text_events('from socket B', 'resp_socket_b1'), text_events('again on B', 'resp_socket_b2')])
    )
    sockets.pending.append(socket_b)
    http_requests: list[httpx2.Request] = []

    def http_handler(request: httpx2.Request) -> httpx2.Response:
        http_requests.append(request)
        response = text_events('from HTTP', 'resp_http')[-1]['response']
        assert isinstance(response, dict)
        response['output'] = [
            {
                'type': 'message',
                'id': 'msg_http',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': 'from HTTP', 'annotations': [], 'logprobs': []}],
            }
        ]
        return httpx2.Response(200, json=response)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test', http_client=http_client))
        settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto'}

        async with source.connect() as connected_a:
            socket_a_result = await Agent(connected_a, model_settings=settings).run('from socket A')

        http_result = await Agent(source, model_settings=settings).run(
            'after socket A', message_history=socket_a_result.all_messages()
        )

        async with source.connect() as connected_b:
            http_history_result = await Agent(connected_b, model_settings=settings).run(
                'after HTTP', message_history=http_result.all_messages()
            )
            socket_a_history_result = await Agent(connected_b, model_settings=settings).run(
                'after socket A on B', message_history=socket_a_result.all_messages()
            )

    assert socket_a_result.output == 'from socket A'
    assert http_result.output == 'from HTTP'
    assert http_history_result.output == 'from socket B'
    assert socket_a_history_result.output == 'again on B'

    assert len(socket_a.sent) == 1
    assert socket_a.sent[0]['type'] == 'response.create'
    assert socket_a.sent[0]['model'] == 'gpt-4o'
    assert socket_a.sent[0]['input'] == [{'role': 'user', 'content': 'from socket A'}]
    assert 'previous_response_id' not in socket_a.sent[0]

    assert len(http_requests) == 1
    http_request = http_requests[0]
    assert http_request.url.path == '/v1/responses'
    http_body = FRAME_ADAPTER.validate_json(http_request.content)
    assert http_body['model'] == 'gpt-4o'
    assert http_body['input'] == [
        {'role': 'user', 'content': 'from socket A'},
        {'role': 'assistant', 'content': 'from socket A'},
        {'role': 'user', 'content': 'after socket A'},
    ]
    assert 'previous_response_id' not in http_body

    assert len(socket_b.sent) == 2
    assert [frame['type'] for frame in socket_b.sent] == ['response.create', 'response.create']
    assert [frame['model'] for frame in socket_b.sent] == ['gpt-4o', 'gpt-4o']
    assert socket_b.sent[0]['input'] == [
        {'role': 'user', 'content': 'from socket A'},
        {'role': 'assistant', 'content': 'from socket A'},
        {'role': 'user', 'content': 'after socket A'},
        {'role': 'assistant', 'content': 'from HTTP'},
        {'role': 'user', 'content': 'after HTTP'},
    ]
    assert socket_b.sent[1]['input'] == [
        {'role': 'user', 'content': 'from socket A'},
        {'role': 'assistant', 'content': 'from socket A'},
        {'role': 'user', 'content': 'after socket A on B'},
    ]
    assert all('previous_response_id' not in frame for frame in socket_b.sent)


async def test_older_socket_response_replays_history(
    allow_model_requests: None, sockets: SocketHarness, monkeypatch: pytest.MonkeyPatch
):
    """Once every lane is in use, an unrelated response can evict an earlier store-false parent from its lane."""
    monkeypatch.setattr('pydantic_ai.models._openai_responses_websocket.NAMED_LANE_LIMIT', 0)
    socket = sockets.pending[0]
    socket.responses = deque(
        [text_events('first answer', 'resp_first'), text_events('other answer', 'resp_other'), text_events('again')]
    )
    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'openai_store': False}

    async with model.connect() as connected:
        agent = Agent(connected, model_settings=settings)
        first = await agent.run('first question')
        await agent.run('unrelated question')
        again = await agent.run('follow up', message_history=first.all_messages())

    assert again.output == 'again'
    assert len(socket.sent) == 3
    assert 'previous_response_id' not in socket.sent[2]
    assert socket.sent[2]['input'] == [
        {'role': 'user', 'content': 'first question'},
        {'role': 'assistant', 'content': 'first answer'},
        {'role': 'user', 'content': 'follow up'},
    ]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('edit_history', [False, True])
async def test_same_socket_continuation_validates_history(
    allow_model_requests: None, sockets: SocketHarness, stream: bool, edit_history: bool
):
    """A copied connected model keeps its digest key and continues only the unchanged history."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events('first answer', 'resp_first'), text_events('again', 'resp_second')])
    key = b'conversation secret'
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'), history_digest_key=key)
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'openai_store': False}

    async with source.connect() as connected:
        agent = Agent(connected, model_settings=settings)
        if stream:
            async with agent.run_stream('secret') as first_stream:
                await first_stream.get_output()
                history = first_stream.all_messages()
        else:
            first = await agent.run('secret')
            history = first.all_messages()

        request, response = history
        assert isinstance(request, ModelRequest)
        assert isinstance(response, ModelResponse)
        digest = hmac.new(key, ModelMessagesTypeAdapter.dump_json([request], exclude_none=True), hashlib.sha256)
        part_adapter: TypeAdapter[ModelResponsePart] = TypeAdapter(ModelResponsePart)
        for part in response.parts:
            digest.update(b'\n' + part_adapter.dump_json(part, exclude_none=True))
        assert response.history_digest == digest.hexdigest()
        if edit_history:
            request.parts = [UserPromptPart('[redacted]')]

        copied = copy(connected)
        if stream:
            async with agent.run_stream('follow up', model=copied, message_history=history) as second_stream:
                assert await second_stream.get_output() == 'again'
        else:
            second = await agent.run('follow up', model=copied, message_history=history)
            assert second.output == 'again'

    assert sockets.opened == [socket]
    assert len(socket.sent) == 2
    assert all(frame['store'] is False for frame in socket.sent)
    if edit_history:
        assert 'previous_response_id' not in socket.sent[1]
        assert socket.sent[1]['input'] == [
            {'role': 'user', 'content': '[redacted]'},
            {'role': 'assistant', 'content': 'first answer'},
            {'role': 'user', 'content': 'follow up'},
        ]
    else:
        assert socket.sent[1]['previous_response_id'] == 'resp_first'
        assert socket.sent[1]['input'] == [{'role': 'user', 'content': 'follow up'}]


@pytest.mark.parametrize('stream', [False, True])
async def test_same_socket_continuation_replays_after_request_scoped_prompt(
    allow_model_requests: None, sockets: SocketHarness, stream: bool
):
    """The next request must replay history so the socket does not retain an expired prompt."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events('first answer', 'resp_first'), text_events('again')])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'openai_store': False}
    hooks: Hooks[None] = Hooks()

    @hooks.on.before_model_request
    async def add_prompt(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        if len(ctx.messages) == 1:
            ctx.enqueue(SystemPromptPart('Only for this request.', scope='request'))
        return request_context

    async with source.connect() as connected:
        agent = Agent(connected, model_settings=settings, capabilities=[hooks], deps_type=type(None))
        if stream:
            async with agent.run_stream('hello') as first_stream:
                await first_stream.get_output()
                history = first_stream.all_messages()
            async with agent.run_stream('follow up', message_history=history) as second_stream:
                assert await second_stream.get_output() == 'again'
        else:
            first = await agent.run('hello')
            second = await agent.run('follow up', message_history=first.all_messages())
            assert second.output == 'again'

    assert len(socket.sent) == 2
    assert socket.sent[0]['input'] == [
        {'role': 'user', 'content': 'hello'},
        {'role': 'system', 'content': 'Only for this request.'},
    ]
    assert 'previous_response_id' not in socket.sent[1]
    assert socket.sent[1]['input'] == [
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'first answer'},
        {'role': 'user', 'content': 'follow up'},
    ]


@pytest.mark.parametrize('destination', ['http', 'websocket'])
@pytest.mark.parametrize('edit_seed', [False, True])
async def test_concrete_seed_validates_history_across_transports(
    allow_model_requests: None, sockets: SocketHarness, destination: str, edit_seed: bool
):
    """An explicit stored seed must not restore history edited before a socket replay."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events('socket answer', 'resp_socket')])
    next_socket = ScriptedSocket(responses=deque([text_events('final answer', 'resp_final')]))
    sockets.pending.append(next_socket)
    http_bodies: list[Frame] = []

    def http_handler(request: httpx2.Request) -> httpx2.Response:
        http_bodies.append(FRAME_ADAPTER.validate_json(request.content))
        first = len(http_bodies) == 1
        text = 'seed answer' if first else 'final answer'
        response = text_events(text, 'resp_seed' if first else 'resp_final')[-1]['response']
        assert isinstance(response, dict)
        response['output'] = [
            {
                'type': 'message',
                'id': 'msg_http',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': text, 'annotations': []}],
            }
        ]
        return httpx2.Response(200, json=response)

    initial_settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'openai_store': True}
    seeded_settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'resp_seed', 'openai_store': False}
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        source = OpenAIResponsesModel(
            'gpt-4o',
            provider=OpenAIProvider(api_key='test', http_client=http_client),
            history_digest_key=b'conversation secret',
        )
        seed = await Agent(source, model_settings=initial_settings).run('secret')
        history = seed.all_messages()
        request = history[0]
        assert isinstance(request, ModelRequest)
        if edit_seed:
            request.parts = [UserPromptPart('[redacted]')]

        async with source.connect() as connected:
            replay = await Agent(connected, model_settings=seeded_settings).run('socket turn', message_history=history)
            if destination == 'http':
                final = await Agent(source, model_settings=seeded_settings).run(
                    'follow up', message_history=replay.all_messages()
                )
                final_frame = http_bodies[-1]
            else:
                async with source.connect() as other:
                    final = await Agent(other, model_settings=seeded_settings).run(
                        'follow up', message_history=replay.all_messages()
                    )
                final_frame = next_socket.sent[0]

    assert seed.output == 'seed answer'
    assert replay.output == 'socket answer'
    assert final.output == 'final answer'
    assert len(http_bodies) == (2 if destination == 'http' else 1)
    assert http_bodies[0]['store'] is True
    prefix: list[Frame] = [
        {'role': 'user', 'content': '[redacted]'},
        {'role': 'assistant', 'content': 'seed answer'},
    ]
    socket_turn: list[Frame] = [{'role': 'user', 'content': 'socket turn'}]
    assert socket.sent[0].get('previous_response_id') == (None if edit_seed else 'resp_seed')
    assert socket.sent[0]['input'] == [*(prefix if edit_seed else []), *socket_turn]
    assert final_frame.get('previous_response_id') == (None if edit_seed else 'resp_seed')
    assert final_frame['input'] == [
        *(prefix if edit_seed else []),
        *socket_turn,
        {'role': 'assistant', 'content': 'socket answer'},
        {'role': 'user', 'content': 'follow up'},
    ]


def test_compatible_response_is_not_a_history_boundary():
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    http_response = ModelResponse(
        parts=[TextPart('HTTP answer')], provider_name='openai', provider_response_id='resp_http'
    )
    assert source._history_boundary([http_response]) is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('seed_after_compaction', [False, True])
def test_concrete_seed_respects_compaction_when_transport_changes(seed_after_compaction: bool):
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    seed = ModelResponse(parts=[TextPart('seed')], provider_name='openai', provider_response_id='resp_seed')
    compaction = ModelResponse(
        parts=[CompactionPart(content='summary', provider_name='openai')],
        provider_name='openai',
        provider_details={'compaction': True},
    )
    socket_response = ModelResponse(
        parts=[TextPart('socket answer')],
        provider_name='openai',
        provider_response_id='resp_socket',
        provider_details={'transport': 'websocket'},
    )
    history: list[ModelRequest | ModelResponse] = [
        *([compaction, seed] if seed_after_compaction else [seed, compaction]),
        socket_response,
        ModelRequest.user_text_prompt('continue'),
    ]

    previous_id, new_messages = source._resolve_previous_response_id(  # pyright: ignore[reportPrivateUsage]
        'resp_seed', history
    )

    assert previous_id == ('resp_seed' if seed_after_compaction else None)
    assert new_messages == (history[2:] if seed_after_compaction else history)


async def test_connected_model_http_operations_replay_socket_history(
    allow_model_requests: None, sockets: SocketHarness
):
    socket = sockets.pending[0]
    socket.responses = deque([text_events('socket answer', 'resp_socket')])
    http_requests: list[httpx2.Request] = []

    def http_handler(request: httpx2.Request) -> httpx2.Response:
        http_requests.append(request)
        if request.url.path == '/v1/responses/input_tokens':
            return httpx2.Response(200, json={'input_tokens': 16, 'object': 'response.input_tokens'})
        return httpx2.Response(
            200,
            json={
                'id': 'resp_compact',
                'created_at': 0,
                'object': 'response.compaction',
                'output': [{'id': 'comp_1', 'encrypted_content': 'encrypted', 'type': 'compaction'}],
                'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2},
            },
        )

    settings: OpenAIResponsesModelSettings = {'openai_store': False, 'openai_previous_response_id': 'auto'}
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test', http_client=http_client))
        async with source.connect() as connected:
            first_request = ModelRequest.user_text_prompt('first')
            first_response = await connected.request([first_request], settings, ModelRequestParameters())
            assert (first_response.provider_details or {}).get('transport') == 'websocket'
            context = ModelRequestContext(
                model=connected,
                messages=[first_request, first_response, ModelRequest.user_text_prompt('next')],
                model_settings=settings,
                model_request_parameters=ModelRequestParameters(),
            )
            counted = await connected.count_tokens(context.messages, settings, context.model_request_parameters)
            compacted = await connected.compact_messages(context)
        with pytest.raises(UserError, match='WebSocket is closed'):
            await connected.count_tokens(context.messages, settings, context.model_request_parameters)
        with pytest.raises(UserError, match='WebSocket is closed'):
            await connected.compact_messages(context)

    assert len(socket.sent) == 1
    assert counted.input_tokens == 16
    assert compacted.provider_response_id == 'resp_compact'
    assert [request.url.path for request in http_requests] == [
        '/v1/responses/input_tokens',
        '/v1/responses/compact',
    ]
    for request in http_requests:
        body = FRAME_ADAPTER.validate_json(request.content)
        assert 'previous_response_id' not in body
        assert 'first' in str(body['input'])
        assert 'socket answer' in str(body['input'])
        assert 'next' in str(body['input'])
    assert socket.close_count == 1


@pytest.mark.parametrize(
    'previous_response_id,seed_in_history', [('auto', False), ('resp_seed', False), ('resp_seed', True)]
)
async def test_per_step_selection_switches_websocket_and_http(
    allow_model_requests: None, sockets: SocketHarness, previous_response_id: str, seed_in_history: bool
):
    """A selector can change transport inside one run without crossing cache-only IDs."""
    socket = sockets.pending[0]
    socket.responses = deque([tool_call_events('resp_socket', 'call_socket'), text_events('done', 'resp_final')])
    http_requests: list[httpx2.Request] = []

    def http_handler(request: httpx2.Request) -> httpx2.Response:
        http_requests.append(request)
        response = text_events(response_id='resp_http')[-1]['response']
        assert isinstance(response, dict)
        response['output'] = [
            {
                'type': 'function_call',
                'id': 'fc_http',
                'status': 'completed',
                'arguments': '{}',
                'call_id': 'call_http',
                'name': 'step',
            }
        ]
        return httpx2.Response(200, json=response)

    settings: OpenAIResponsesModelSettings = {
        'openai_previous_response_id': previous_response_id,
        'openai_store': False,
    }
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test', http_client=http_client))
        async with source.connect() as connected:
            selection_contexts: list[tuple[int, Model | None]] = []
            tool_models: list[AbstractModel] = []

            def select(ctx: ModelSelectionContext[object]) -> Model:
                selection_contexts.append((ctx.run_step, ctx.model))
                return connected if ctx.run_step in (1, 3) else source

            agent = Agent(source, deps_type=object, capabilities=[SelectModel(select)], model_settings=settings)

            @agent.tool
            def step(ctx: RunContext[object]) -> str:
                tool_models.append(ctx.model)
                return str(len(tool_models))

            history: list[ModelRequest | ModelResponse] = []
            if seed_in_history:
                history = [
                    ModelRequest(parts=[UserPromptPart('prior turn')]),
                    ModelResponse(
                        parts=[TextPart('prior answer')],
                        provider_name='openai',
                        provider_response_id='resp_seed',
                    ),
                ]
            result = await agent.run('start', message_history=history)

    assert result.output == 'done'
    assert selection_contexts == [(1, source), (2, connected), (3, source)]
    assert tool_models == [connected, source]
    assert len(socket.sent) == 2
    assert len(http_requests) == 1
    assert all(frame['type'] == 'response.create' and frame['model'] == 'gpt-4o' for frame in socket.sent)
    assert all(frame['store'] is False for frame in socket.sent)
    assert socket.sent[0]['input'] == [{'role': 'user', 'content': 'start'}]
    assert socket.sent[0].get('previous_response_id') == (None if previous_response_id == 'auto' else 'resp_seed')
    assert http_requests[0].url.path == '/v1/responses'
    http_body = FRAME_ADAPTER.validate_json(http_requests[0].content)
    assert http_body['model'] == 'gpt-4o'
    assert http_body['store'] is False
    replayed_socket_turn: list[Frame] = [
        {'role': 'user', 'content': 'start'},
        {'type': 'function_call', 'name': 'step', 'arguments': '{}', 'call_id': 'call_socket'},
        {'type': 'function_call_output', 'call_id': 'call_socket', 'output': '1'},
    ]
    assert http_body['input'] == replayed_socket_turn
    assert http_body.get('previous_response_id') == (None if previous_response_id == 'auto' else 'resp_seed')
    assert 'resp_socket' not in str(http_body)
    assert socket.sent[1].get('previous_response_id') == (None if previous_response_id == 'auto' else 'resp_seed')
    assert 'resp_http' not in str(socket.sent[1])
    final_input = socket.sent[1]['input']
    assert isinstance(final_input, list)
    assert final_input == [
        *replayed_socket_turn,
        {'type': 'function_call', 'name': 'step', 'arguments': '{}', 'call_id': 'call_http'},
        {'type': 'function_call_output', 'call_id': 'call_http', 'output': '2'},
    ]
    assert socket.close_count == 1


@pytest.mark.parametrize('previous_response_id', ['auto', 'resp_seed'])
async def test_per_step_selection_switches_distinct_sockets(
    allow_model_requests: None, sockets: SocketHarness, previous_response_id: str
):
    """A second same-name socket cannot inherit the first one's cache-only parent ID."""
    first_socket = sockets.pending[0]
    first_socket.responses = deque([tool_call_events('resp_first', 'call_first')])
    second_socket = ScriptedSocket(responses=deque([text_events('done', 'resp_second')]))
    sockets.pending.append(second_socket)
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    settings: OpenAIResponsesModelSettings = {
        'openai_previous_response_id': previous_response_id,
        'openai_store': False,
    }

    async with AsyncExitStack() as stack:
        first = await stack.enter_async_context(source.connect())
        second = await stack.enter_async_context(source.connect())
        selection_contexts: list[tuple[int, Model | None]] = []
        tool_models: list[AbstractModel] = []

        def select(ctx: ModelSelectionContext[object]) -> Model:
            selection_contexts.append((ctx.run_step, ctx.model))
            return first if ctx.run_step == 1 else second

        agent = Agent(source, deps_type=object, capabilities=[SelectModel(select)], model_settings=settings)

        @agent.tool
        def step(ctx: RunContext[object]) -> str:
            tool_models.append(ctx.model)
            return '1'

        result = await agent.run('start')

    assert result.output == 'done'
    assert selection_contexts == [(1, source), (2, first)]
    assert tool_models == [first]
    assert len(first_socket.sent) == len(second_socket.sent) == 1
    assert first_socket.sent[0]['input'] == [{'role': 'user', 'content': 'start'}]
    assert first_socket.sent[0].get('previous_response_id') == (None if previous_response_id == 'auto' else 'resp_seed')
    assert second_socket.sent[0]['input'] == [
        {'role': 'user', 'content': 'start'},
        {'type': 'function_call', 'name': 'step', 'arguments': '{}', 'call_id': 'call_first'},
        {'type': 'function_call_output', 'call_id': 'call_first', 'output': '1'},
    ]
    assert second_socket.sent[0].get('previous_response_id') == (
        None if previous_response_id == 'auto' else 'resp_seed'
    )
    assert 'resp_first' not in str(second_socket.sent[0])
    assert [first_socket.close_count, second_socket.close_count] == [1, 1]


async def test_model_selection_uses_connected_model_at_the_selected_precedence(
    allow_model_requests: None, sockets: SocketHarness, capfire: CaptureLogfire
):
    logfire.instrument_pydantic_ai()
    explicit_socket = sockets.pending[0]
    explicit_socket.responses = deque([text_events('explicit connected', 'resp_explicit')])
    static_socket = ScriptedSocket(responses=deque([text_events('static selected', 'resp_static')]))
    sockets.pending.append(static_socket)
    dependency_socket = ScriptedSocket(responses=deque([text_events('dependency selected', 'resp_deps')]))
    sockets.pending.append(dependency_socket)
    http_requests: list[httpx2.Request] = []

    def http_handler(request: httpx2.Request) -> httpx2.Response:
        http_requests.append(request)
        response = text_events('http selected', 'resp_http')[-1]['response']
        assert isinstance(response, dict)
        response['output'] = [
            {
                'type': 'message',
                'id': 'msg_http',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': 'http selected', 'annotations': [], 'logprobs': []}],
            }
        ]
        return httpx2.Response(200, json=response)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test', http_client=http_client))

        # A run with no default model can supply the connected model explicitly.
        explicit_agent = Agent(None)
        async with source.connect() as connected:
            with logfire.span('caller span'):
                explicit_result = await explicit_agent.run('explicit', model=connected)

        # A static capability can select the connected handle when the agent has no default.
        static_contexts: list[ModelSelectionContext[object]] = []
        async with source.connect() as connected:

            def select_static(ctx: ModelSelectionContext[object]) -> Model:
                static_contexts.append(ctx)
                return connected

            static_agent = Agent(None, capabilities=[SelectModel(select_static)])
            static_result = await static_agent.run('static')

        selection_contexts: list[ModelSelectionContext[str]] = []
        async with source.connect() as connected:

            def select_by_dependency(ctx: ModelSelectionContext[str]) -> Model:
                selection_contexts.append(ctx)
                return connected if ctx.deps == 'websocket' else source

            selecting_agent = Agent(None, deps_type=str, capabilities=[SelectModel(select_by_dependency)])
            websocket_result = await selecting_agent.run('select websocket', deps='websocket')
            http_result = await selecting_agent.run('select HTTP', deps='http')
            explicit_model_result = await selecting_agent.run(
                'explicit model wins', deps='websocket', model=TestModel(custom_output_text='explicit model')
            )

    assert explicit_result.output == 'explicit connected'
    assert static_result.output == 'static selected'
    assert websocket_result.output == 'dependency selected'
    assert http_result.output == 'http selected'
    assert explicit_model_result.output == 'explicit model'

    assert len(static_contexts) == 1
    assert static_contexts[0].model is None
    assert static_contexts[0].deps is None
    assert len(selection_contexts) == 2
    assert [(ctx.model, ctx.deps) for ctx in selection_contexts] == [(None, 'websocket'), (None, 'http')]

    assert [len(socket.sent) for socket in sockets.opened] == [1, 1, 1]
    assert [socket.close_count for socket in sockets.opened] == [1, 1, 1]
    assert [frame['model'] for socket in sockets.opened for frame in socket.sent] == ['gpt-4o'] * 3
    assert [frame['input'] for socket in sockets.opened for frame in socket.sent] == [
        [{'role': 'user', 'content': 'explicit'}],
        [{'role': 'user', 'content': 'static'}],
        [{'role': 'user', 'content': 'select websocket'}],
    ]

    assert len(http_requests) == 1
    http_request = http_requests[0]
    assert http_request.url.path == '/v1/responses'
    http_body = FRAME_ADAPTER.validate_json(http_request.content)
    assert http_body['model'] == 'gpt-4o'
    assert http_body['input'] == [{'role': 'user', 'content': 'select HTTP'}]
    logfire.force_flush()
    chat_transports = [
        span['attributes'].get('pydantic_ai.transport')
        for span in capfire.exporter.exported_spans_as_dict()
        if span['attributes'].get('gen_ai.operation.name') == 'chat'
        and span['attributes'].get('gen_ai.request.model') == 'gpt-4o'
    ]
    assert chat_transports == ['websocket', 'websocket', 'websocket', None]
    caller_spans = [span for span in capfire.exporter.exported_spans_as_dict() if span['name'] == 'caller span']
    assert len(caller_spans) == 1
    assert 'pydantic_ai.transport' not in caller_spans[0]['attributes']


async def test_streaming_transport_is_recorded_on_model_span(
    allow_model_requests: None, sockets: SocketHarness, capfire: CaptureLogfire
):
    logfire.instrument_pydantic_ai()
    sockets.pending[0].responses = deque([text_events('streamed')])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))

    async with source.connect() as connected:
        async with Agent(connected).run_stream('hello') as result:
            assert await result.get_output() == 'streamed'

    logfire.force_flush()
    chat_transports = [
        span['attributes'].get('pydantic_ai.transport')
        for span in capfire.exporter.exported_spans_as_dict()
        if span['attributes'].get('gen_ai.operation.name') == 'chat'
        and span['attributes'].get('gen_ai.request.model') == 'gpt-4o'
    ]
    assert chat_transports == ['websocket']


@pytest.mark.parametrize('wrap', ['none', 'wrapper', 'fallback'])
async def test_connected_model_is_rejected_before_durable_request(
    allow_model_requests: None, sockets: SocketHarness, wrap: str
):
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source.connect() as connected:
        model = (
            WrapperModel(connected)
            if wrap == 'wrapper'
            else FallbackModel(connected, source)
            if wrap == 'fallback'
            else connected
        )
        agent = Agent(model, name='responses', capabilities=[JournalDurability()])
        with pytest.raises(UserError, match='model with a live connection'):
            await agent.run('hello')
        assert sockets.opened[0].sent == []
    assert sockets.opened[0].close_count == 1


@asynccontextmanager
async def local_server(handler: Callable[[ServerConnection], Awaitable[None]]) -> AsyncGenerator[OpenAIResponsesModel]:
    """Serve `handler` from a real `websockets` server on localhost and yield a model pointed at it.

    The scripted socket replaces `websockets`, so it cannot pin the library's message size limit or flow control,
    and no recorded provider response exercises either: these tests run the real library on both ends instead.
    """
    async with serve(handler, '127.0.0.1', 0) as server:
        port: int = next(iter(server.sockets)).getsockname()[1]
        source = OpenAIResponsesModel(
            'gpt-4o', provider=OpenAIProvider(base_url=f'http://127.0.0.1:{port}/v1', api_key='test')
        )
        async with source:
            yield source


async def send_events(ws: ServerConnection, request: Frame, events: list[Frame]) -> None:
    stream_id = request.get('stream_id')
    for event in on_lane(events, stream_id) if isinstance(stream_id, str) else events:
        await ws.send(FRAME_ADAPTER.dump_json(event).decode())


async def test_event_over_one_mebibyte_keeps_the_socket_open(allow_model_requests: None):
    """An event larger than the `websockets` default message limit, like an image generation result, is received."""
    large = 'x' * (2**20 + 1)
    requests: list[Frame] = []

    async def handler(ws: ServerConnection) -> None:
        async for raw in ws:
            requests.append(FRAME_ADAPTER.validate_json(raw))
            text = large if len(requests) == 1 else 'ready'
            await send_events(ws, requests[-1], text_events(text, f'resp_{len(requests)}'))

    async with local_server(handler) as source, source.connect() as connected:
        assert (await Agent(connected).run('large')).output == large
        assert (await Agent(connected).run('next')).output == 'ready'
    assert len(requests) == 2


async def test_abandoned_stream_keeps_answering_keepalive_pings(allow_model_requests: None):
    """An abandoned response's unread events don't stop the idle socket from answering the server's pings.

    Once the client has abandoned the response, the server sends one keepalive ping, as a short `ping_interval`
    would, and records whether the pong arrives; without it, the server's `ping_timeout` would close the connection.
    Waiting on that pong rather than a timer keeps the test free of real waits when the socket answers.
    """
    requests: list[Frame] = []
    abandoned = anyio.Event()
    keepalive_answered: list[bool] = []
    keepalive_done = anyio.Event()

    async def handler(ws: ServerConnection) -> None:
        async for raw in ws:
            requests.append(FRAME_ADAPTER.validate_json(raw))
            if len(requests) > 1:
                await send_events(ws, requests[-1], text_events(response_id='resp_next'))
                continue
            events = text_events('word ', 'resp_abandoned')
            # More unread frames than the `websockets` default high-water mark of 16.
            await send_events(ws, requests[-1], [*events[:2], *[events[2]] * 50, events[3]])
            await abandoned.wait()
            pong = await ws.ping()
            with anyio.move_on_after(READINESS_WAIT_TIMEOUT / 2) as keepalive:
                await pong
            keepalive_answered.append(not keepalive.cancelled_caught)
            keepalive_done.set()

    async with local_server(handler) as source, source.connect() as connected:
        async with connected.request_stream(
            [ModelRequest.user_text_prompt('long answer')], None, ModelRequestParameters()
        ) as streamed:
            async for _ in streamed:  # pragma: no branch
                break
        abandoned.set()
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await keepalive_done.wait()
        assert keepalive_answered == [True]
        assert (await Agent(connected).run('next')).output == 'ready'


@pytest.mark.parametrize('interrupt', ['early_exit', 'explicit_close', 'cancel'])
async def test_interruption(allow_model_requests: None, sockets: SocketHarness, interrupt: str):
    socket = sockets.pending[0]
    socket.responses = deque([text_events()[:-1], text_events(response_id='resp_next')])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    before = asyncio.all_tasks()
    async with source.connect() as connected:
        if interrupt == 'early_exit':
            async with Agent(connected).run_stream('hello') as result:
                async for _ in result.stream_text(delta=True):  # pragma: no branch
                    break
        elif interrupt == 'explicit_close':
            async with connected.request_stream(
                [ModelRequest(parts=[UserPromptPart('hello')])], None, ModelRequestParameters()
            ) as response:
                await response.close_stream()
        else:
            started = anyio.Event()

            async def consume() -> None:
                async with Agent(connected).run_stream('hello') as result:
                    async for _ in result.stream_text(delta=True):
                        started.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(consume)
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await started.wait()
                tasks.cancel_scope.cancel()
        assert (await Agent(connected).run('reuse')).output == 'ready'
        assert socket.close_count == 0
    assert [frame.get('stream_id') for frame in socket.sent] == [None, 'lane-1']
    assert socket.close_count == 1
    if interrupt != 'early_exit':
        # Breaking from `stream_text()` schedules Python's async-generator finalizer separately.
        assert asyncio.all_tasks() == before


async def test_stream_cancel_during_receive(allow_model_requests: None, sockets: SocketHarness):
    """Cancelling a response interrupts its pending receive and leaves the socket open for other requests."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events()[:-1], text_events(response_id='resp_next')])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source.connect() as connected:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with connected.request_stream(
                [ModelRequest(parts=[UserPromptPart('hello')])], None, ModelRequestParameters()
            ) as response:

                async def consume() -> None:
                    async for _ in response:
                        pass

                async with anyio.create_task_group() as tasks:
                    tasks.start_soon(consume)
                    await socket.waiting_for_input.wait()
                    await response.cancel()

                assert response.cancelled
                assert response.get().state == 'interrupted'
                assert response.get().text == 'ready'
        assert socket.close_count == 0
        assert (await Agent(connected).run('next')).output == 'ready'
    assert socket.close_count == 1


@pytest.mark.parametrize(
    'failure',
    [
        'envelope',
        'status',
        'model_not_found',
        'failed',
        'steering',
        'disconnect',
        'send',
        'timeout',
    ],
)
@pytest.mark.parametrize('stream', [False, True])
async def test_errors(allow_model_requests: None, sockets: SocketHarness, failure: str, stream: bool):
    """WebSocket failures never trigger HTTP fallback or return successful partial output."""
    socket = sockets.pending[0]
    settings: OpenAIResponsesModelSettings = {}
    if failure in ('envelope', 'status', 'model_not_found'):
        error: Frame = {
            'type': 'error',
            'error': {
                'type': 'invalid_request_error',
                'code': 'model_not_found' if failure == 'model_not_found' else 'invalid_test',
                'message': 'test failure',
            },
        }
        if failure != 'envelope':
            error['status'] = 400
        socket.responses = deque([[error]])
    elif failure == 'failed':
        response = text_events()[-1]['response']
        assert isinstance(response, dict)
        response.update(status='failed', error={'code': 'server_error', 'message': 'test failure'})
        socket.responses = deque([[{'type': 'response.failed', 'sequence_number': 0, 'response': response}]])
    elif failure == 'steering':
        socket.responses = deque(
            [
                [
                    {
                        'type': 'response.steer.accepted',
                        'sequence_number': 0,
                        'steer': {'id': 'steer_test', 'previous_response_id': 'resp_test'},
                    }
                ]
            ]
        )
    elif failure == 'disconnect':
        socket.responses.clear()
        socket.push(*text_events()[:-1], ConnectionClosedError(None, None))
    elif failure == 'send':
        socket.send_error = OSError('test failure')
    else:
        socket.responses.clear()
        settings['timeout'] = 0.01
    model_name = 'gpt-5.2-proo' if failure == 'model_not_found' else 'gpt-4o'
    http_handler = Mock(return_value=httpx2.Response(401))
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler))
    source = OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key='test', http_client=http_client))
    async with http_client, source.connect() as connected:
        with pytest.raises(UnexpectedModelBehavior if failure == 'steering' else ModelAPIError) as raised:
            agent = Agent(connected, model_settings=settings)
            if stream:
                async with agent.run_stream('hello') as result:
                    await result.get_output()
            else:
                await agent.run('hello')
        transport_failure = failure in ('disconnect', 'send', 'timeout')
        assert socket.close_count == (1 if failure in ('disconnect', 'send') else 0)
        if failure in ('status', 'model_not_found'):
            assert isinstance(raised.value, ModelHTTPError)
            assert raised.value.status_code == 400
            assert raised.value.body == {
                'type': 'invalid_request_error',
                'code': 'model_not_found' if failure == 'model_not_found' else 'invalid_test',
                'message': 'test failure',
            }
            assert raised.value.suggested_model_id == ('openai:gpt-5.2-pro' if failure == 'model_not_found' else None)
        elif failure == 'envelope':
            assert 'invalid_test: test failure' in str(raised.value)
        elif failure == 'failed':
            assert 'server_error' in str(raised.value)
        if failure in ('disconnect', 'send'):
            with pytest.raises(ModelAPIError, match='WebSocket'):
                await Agent(connected).run('recover')
        elif not transport_failure:
            socket.responses = deque([text_events(response_id='resp_recovered')])
            assert (await Agent(connected).run('recover')).output == 'ready'

    http_handler.assert_not_called()


@pytest.mark.parametrize('stream', [False, True])
async def test_failed_response_without_error(allow_model_requests: None, sockets: SocketHarness, stream: bool):
    response = text_events()[-1]['response']
    assert isinstance(response, dict)
    response.update(status='failed', error=None)
    sockets.pending[0].responses = deque([[{'type': 'response.failed', 'sequence_number': 0, 'response': response}]])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    messages: list[ModelRequest | ModelResponse] = [ModelRequest.user_text_prompt('hello')]
    async with source.connect() as connected:
        if stream:
            async with connected.request_stream(messages, None, ModelRequestParameters()) as streamed:
                async for _ in streamed:
                    pass
                result = streamed.get()
        else:
            result = await connected.request(messages, None, ModelRequestParameters())

        assert result.parts == []
        assert result.finish_reason == 'error'
        assert (result.provider_details or {}).get('finish_reason') == 'failed'
        assert sockets.opened[0].close_count == 0
        sockets.opened[0].responses = deque([text_events(response_id='resp_retry')])
        retried = await connected.request([*messages, result], None, ModelRequestParameters())
        assert retried.text == 'ready'
    # A failed response cannot be continued from, so the retry sends the whole history.
    assert 'previous_response_id' not in sockets.opened[0].sent[1]


@pytest.mark.parametrize('error_code', ['content_filter', 'invalid_test'])
@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('api', ['agent', 'model'])
async def test_error_frame(
    allow_model_requests: None,
    sockets: SocketHarness,
    error_code: str,
    stream: bool,
    api: str,
):
    """A WebSocket error frame keeps its HTTP status and provider body."""
    error: Frame = {
        'type': 'invalid_request_error',
        'code': error_code,
        'message': 'The content was filtered.',
        'innererror': {
            'code': 'ResponsibleAIPolicyViolation',
            'content_filter_result': {'hate': {'filtered': True, 'severity': 'high'}},
        },
    }
    socket = sockets.pending[0]
    socket.responses = deque([[{'type': 'error', 'status': 400, 'error': error}]])
    http_handler = Mock(return_value=httpx2.Response(401))
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler))
    client = AsyncOpenAI(api_key='test', http_client=http_client)
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
    async with client, source.connect() as connected:
        with pytest.raises(ModelHTTPError) as raised:
            if api == 'agent':
                agent = Agent(connected)
                if stream:
                    async with agent.run_stream('hello'):
                        pytest.fail('The rejected request should not produce a stream')  # pragma: no cover
                else:
                    await agent.run('hello')
            else:
                messages: list[ModelRequest | ModelResponse] = [ModelRequest(parts=[UserPromptPart('hello')])]
                if stream:
                    async with connected.request_stream(messages, None, ModelRequestParameters()) as streamed:
                        async for _ in streamed:
                            pass
                else:
                    await connected.request(messages, None, ModelRequestParameters())
        assert raised.value.status_code == 400
        assert raised.value.body == error
        assert socket.close_count == 0
    assert socket.close_count == 1
    http_handler.assert_not_called()


@pytest.mark.parametrize(
    ('status', 'code', 'error_type', 'expected'),
    [
        (429, 'rate_limit_exceeded', 'requests', ModelRateLimitError),
        (400, 'context_length_exceeded', 'invalid_request_error', ModelContextWindowExceededError),
        (None, 'server_error', 'server_error', ModelServerError),
    ],
)
async def test_error_frame_raises_the_http_error_category(
    allow_model_requests: None,
    sockets: SocketHarness,
    status: int | None,
    code: str,
    error_type: str,
    expected: type[ModelAPIError],
):
    """An error frame raises the category the same error raises over HTTP, before or inside a stream."""
    frame: Frame = {'type': 'error', 'error': {'type': error_type, 'code': code, 'message': 'The request failed.'}}
    if status is not None:
        frame['status'] = status
    sockets.pending[0].responses = deque([[frame]])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source.connect() as connected:
        with pytest.raises(expected) as raised:
            await Agent(connected).run('hello')
    assert raised.value.provider_error_code == code
    assert raised.value.in_stream is (status is None)


async def test_incomplete_response(allow_model_requests: None, sockets: SocketHarness):
    events = text_events()
    terminal = events[-1]
    terminal['type'] = 'response.incomplete'
    response = terminal['response']
    assert isinstance(response, dict)
    response.update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
    sockets.pending[0].responses = deque([events])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source.connect() as connected:
        result = await connected.request(
            [ModelRequest(parts=[UserPromptPart('hello')])], None, ModelRequestParameters()
        )
        assert result.text == 'ready'
        assert result.finish_reason == 'length'
        assert (result.provider_details or {}).get('finish_reason') == 'max_output_tokens'
        assert sockets.opened[0].close_count == 0
        sockets.opened[0].responses = deque([text_events(response_id='resp_again')])
        again = await connected.request([ModelRequest(parts=[UserPromptPart('again')])], None, ModelRequestParameters())
        assert again.text == 'ready'


async def test_codex_connection_unsupported(sockets: SocketHarness):
    source = OpenAICodexModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    try:
        with pytest.raises(UserError, match='cannot use a persistent Responses WebSocket'):
            source.connect()
        assert not sockets.connecting.is_set()
        assert not source.client.is_closed()
    finally:
        await source.client.close()


@pytest.mark.parametrize('timeout_source', ['model', 'client'])
@pytest.mark.parametrize('connect_timeout', [20, None])
async def test_handshake_timeout_override(sockets: SocketHarness, timeout_source: str, connect_timeout: float | None):
    """Long and unlimited connect timeouts are not capped by the transport's ten-second default."""
    timeout = Timeout(30, connect=connect_timeout)
    settings: OpenAIResponsesModelSettings = {'timeout': timeout} if timeout_source == 'model' else {}
    client = AsyncOpenAI(
        api_key='test', timeout=httpx2.Timeout(30, connect=connect_timeout) if timeout_source == 'client' else 30
    )
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=client), settings=settings)
    async with client, source.connect():
        assert sockets.options[0]['open_timeout'] is None
    assert sockets.connect_count == 1
    assert sockets.opened[0].close_count == 1


@pytest.mark.parametrize(
    'options', [pytest.param({}, id='defaults'), pytest.param({'max_size': 1024, 'max_queue': 4}, id='caller')]
)
async def test_message_limits_default_to_unlimited(sockets: SocketHarness, options: WebSocketConnectionOptions):
    """The `websockets` message size and unread-frame limits are lifted unless the caller sets them."""
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source, source.connect(websocket_connection_options=options):
        pass
    assert {key: sockets.options[0][key] for key in ('max_size', 'max_queue')} == {
        'max_size': None,
        'max_queue': None,
        **options,
    }


@pytest.mark.parametrize('timeout_source', ['model', 'client'])
async def test_handshake_timeout(sockets: SocketHarness, timeout_source: str):
    """A stalled handshake honors the model's connect timeout without leaking a connection."""
    before = asyncio.all_tasks()
    sockets.connect_gate = anyio.Event()
    timeout = Timeout(10, connect=0.01)
    settings: OpenAIResponsesModelSettings = {'timeout': timeout} if timeout_source == 'model' else {}
    client = AsyncOpenAI(api_key='test', timeout=httpx2.Timeout(10, connect=0.01) if timeout_source == 'client' else 30)
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=client), settings=settings)
    async with client, source:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            with pytest.raises(ModelAPIError, match='WebSocket connection failed') as raised:
                async with source.connect():
                    pytest.fail('The stalled handshake should time out')  # pragma: no cover
        assert isinstance(raised.value.__cause__, TimeoutError)
        assert sockets.connecting.is_set()
        assert sockets.opened == []
        assert not source.client.is_closed()
    assert asyncio.all_tasks() == before


async def test_send_timeout(allow_model_requests: None, sockets: SocketHarness):
    """A stalled send honors the request's write timeout and invalidates the socket."""
    before = asyncio.all_tasks()
    socket = sockets.pending[0]
    socket.send_gate = anyio.Event()
    settings: OpenAIResponsesModelSettings = {'timeout': Timeout(10, write=0.01)}
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source, source.connect() as connected:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            with pytest.raises(ModelAPIError, match='WebSocket request failed') as raised:
                await Agent(connected, model_settings=settings).run('hello')
        assert isinstance(raised.value.__cause__, TimeoutError)
        assert socket.sending.is_set()
        assert socket.sent == []
        assert socket.close_count == 1
        with pytest.raises(ModelAPIError, match='WebSocket request failed'):
            await Agent(connected).run('cannot reuse')
    assert socket.close_count == 1
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('status_code', [200, 302, 401, None])
async def test_handshake_errors(sockets: SocketHarness, status_code: int | None):
    if status_code is not None:
        sockets.connect_error = InvalidStatus(
            HandshakeResponse(
                status_code,
                'Rejected',
                Headers([('X-Request-Id', 'first'), ('x-request-id', 'second')]),
                b'handshake rejected',
            )
        )
    else:
        sockets.connect_error = OSError('unreachable')
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    try:
        expected_error = ModelHTTPError if status_code is not None and status_code >= 400 else ModelAPIError
        with pytest.raises(expected_error) as raised:
            async with source.connect():
                pytest.fail('The handshake should fail')  # pragma: no cover
        assert type(raised.value) is expected_error
        assert sockets.connect_count == 1
        if isinstance(raised.value, ModelHTTPError):
            assert raised.value.status_code == status_code
            assert raised.value.body == 'handshake rejected'
            assert raised.value.headers == {'x-request-id': 'first, second'}
        assert not source.client.is_closed()
    finally:
        await source.client.close()


@pytest.mark.parametrize('auth_mode', ['sigv4', 'native-bearer'])
@pytest.mark.skipif(not bedrock_mantle_imports_successful(), reason='bedrock-mantle not installed')
async def test_bedrock_mantle_websocket_auth_requires_compatible_sdk_client(
    monkeypatch: pytest.MonkeyPatch, sockets: SocketHarness, auth_mode: Literal['sigv4', 'native-bearer']
):
    """Mantle's native auth modes fail before opening a socket with guidance for WebSocket auth."""
    monkeypatch.delenv('AWS_BEARER_TOKEN_BEDROCK', raising=False)
    if auth_mode == 'sigv4':
        provider = BedrockMantleProvider(
            region_name='us-east-1', aws_access_key_id='fake-access-key', aws_secret_access_key='fake-secret-key'
        )
    else:
        provider = BedrockMantleProvider(region_name='us-east-1', api_key='fake-bearer-token')
    source = BedrockMantleResponsesModel('openai.gpt-5.6-luna', provider=provider)

    async with source:
        with pytest.raises(UserError) as raised:
            async with source.connect():
                pytest.fail('The SDK should reject this authentication before opening a socket')  # pragma: no cover

        message = str(raised.value)
        assert message.startswith('The OpenAI SDK could not configure this WebSocket connection:')
        assert 'Use HTTP or supply an SDK client with WebSocket-compatible authentication.' in message
        assert sockets.connect_count == 0
        assert sockets.opened == []


@pytest.mark.skipif(not bedrock_mantle_imports_successful(), reason='bedrock-mantle not installed')
async def test_bedrock_mantle_websocket_accepts_compatible_bearer_client(sockets: SocketHarness):
    """A caller-supplied bearer client preserves the Mantle endpoint and closes its socket once."""
    client = AsyncOpenAI(api_key='fake-bearer-token', base_url='https://bedrock-mantle.us-east-1.api.aws/openai/v1')
    provider = BedrockMantleProvider(openai_client=client)
    source = BedrockMantleResponsesModel('openai.gpt-5.6-luna', provider=provider)
    socket = sockets.pending[0]

    async with client, source.connect():
        assert sockets.urls == ['wss://bedrock-mantle.us-east-1.api.aws/openai/v1/responses']
        assert sockets.authorization_headers == ['Bearer fake-bearer-token']
        assert socket.close_count == 0

    assert socket.close_count == 1


async def test_sdk_api_errors_during_websocket_connection_are_mapped(sockets: SocketHarness):
    request = httpx2.Request('GET', 'https://api.openai.com/v1/responses')
    sockets.connect_error = APIError('SDK API failure', request, body=None)
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    try:
        with pytest.raises(ModelAPIError, match='SDK API failure'):
            async with source.connect():
                pytest.fail('The SDK API error should be propagated')  # pragma: no cover
        assert sockets.connect_count == 1
    finally:
        await source.client.close()


async def test_missing_websocket_dependency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, 'pydantic_ai.models._openai_responses_websocket', None)
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    with pytest.raises(ImportError, match='openai-responses-websocket'):
        async with source.connect():
            pytest.fail('The optional dependency is required')  # pragma: no cover


@pytest.mark.parametrize('invalid', ['headers', 'mutated_headers', 'background', 'body', 'envelope', 'resume'])
async def test_incompatible_options(allow_model_requests: None, sockets: SocketHarness, invalid: str):
    settings: OpenAIResponsesModelSettings = {}
    history: list[ModelRequest | ModelResponse] = []
    if invalid == 'headers':
        settings['extra_headers'] = {'x-change': 'new'}
    elif invalid == 'background':
        settings['openai_background'] = True
    elif invalid == 'body':
        settings['extra_body'] = ['invalid']
    elif invalid == 'envelope':
        settings['extra_body'] = {'stream': True}
    elif invalid == 'resume':
        history.append(
            ModelResponse(
                parts=[TextPart('partial')],
                provider_name='openai',
                provider_response_id='resp_suspended',
                state='suspended',
            )
        )
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source.connect() as connected:
        if invalid == 'mutated_headers':
            assert connected.settings is not None
            assert 'extra_headers' in connected.settings
            connected.settings['extra_headers']['x-change'] = 'new'
        with pytest.raises(UserError):
            await Agent(connected, model_settings=settings).run(
                message_history=history or [ModelRequest(parts=[UserPromptPart('hello')])]
            )
        if invalid == 'resume':
            with pytest.raises(UserError, match='cannot be resumed over a WebSocket'):
                async with connected.request_stream(history, None, ModelRequestParameters()):
                    pytest.fail('A suspended response cannot produce a WebSocket stream')  # pragma: no cover
        assert sockets.opened[0].sent == []


@pytest.mark.parametrize(
    'request_headers,differing_headers',
    [
        pytest.param({}, '`openai-organization`, `x-tenant`', id='omitted-all'),
        pytest.param({'x-tenant': 'tenant-test'}, '`openai-organization`', id='omitted-organization'),
        pytest.param(
            {'openai-organization': 'other-org', 'x-tenant': 'other-tenant'},
            '`openai-organization`, `x-tenant`',
            id='changed-values',
        ),
        pytest.param(
            {'OPENAI-ORGANIZATION': 'org-test', 'x-TENANT': 'tenant-test'}, None, id='case-insensitive-equivalent'
        ),
        pytest.param(
            {'openai-organization': 'org-test', 'x-tenant': 'tenant-test', 'x-request-only': b'byte-header-value'},
            '`x-request-only`',
            id='bytes-value',
        ),
        pytest.param(
            {'openai-organization': 'org-test', 'x-tenant': 'tenant-test', 'x-request-only': 123},
            '`x-request-only`',
            id='int-value',
        ),
        pytest.param(
            {'openai-organization': 'org-test', 'x-tenant': 'tenant-test', 'Authorization': 'Bearer test'},
            '`authorization`',
            id='restated-client-credentials',
        ),
    ],
)
async def test_request_header_overrides(
    allow_model_requests: None,
    sockets: SocketHarness,
    request_headers: dict[str, str | bytes | int],
    differing_headers: str | None,
):
    settings: OpenAIResponsesModelSettings = {
        'extra_headers': {'OpenAI-Organization': 'org-test', 'X-Tenant': 'tenant-test'}
    }
    # Exercise malformed runtime settings outside the declared header type.
    request_settings: OpenAIResponsesModelSettings = {'extra_headers': request_headers}  # pyright: ignore[reportAssignmentType]
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'), settings=settings)
    async with source.connect() as connected:
        agent = Agent(connected, model_settings=request_settings)
        assert sockets.headers[0]['openai-organization'] == 'org-test'
        assert sockets.headers[0]['x-tenant'] == 'tenant-test'
        if differing_headers is None:
            assert (await agent.run('hello')).output == 'ready'
            assert len(sockets.opened[0].sent) == 1
        else:
            with pytest.raises(
                UserError, match='Request headers must match the open Responses WebSocket handshake'
            ) as raised:
                await agent.run('hello')
            assert f'Differing headers: {differing_headers}.' in str(raised.value)
            for value in ('org-test', 'tenant-test', *request_headers.values()):
                assert str(value) not in str(raised.value)
            assert sockets.opened[0].sent == []


@pytest.mark.parametrize(
    'request_headers',
    [
        pytest.param({}, id='inherited'),
        pytest.param({'X-Tenant': 'tenant-test'}, id='explicit'),
        pytest.param({'x-tenant': 'tenant-test'}, id='case-insensitive'),
    ],
)
async def test_client_default_headers(
    allow_model_requests: None, sockets: SocketHarness, request_headers: dict[str, str]
):
    client = AsyncOpenAI(api_key='test', default_headers={'X-Tenant': 'tenant-test'})
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
    request_settings: OpenAIResponsesModelSettings = {'extra_headers': request_headers}
    async with source.connect() as connected:
        assert sockets.headers[0]['x-tenant'] == 'tenant-test'
        assert (await Agent(connected, model_settings=request_settings).run('hello')).output == 'ready'
        assert len(sockets.opened[0].sent) == 1


async def test_connection_authorization_header(allow_model_requests: None, sockets: SocketHarness):
    """An `Authorization` header configured on the connection authenticates the handshake and matches requests."""
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='client-key'))
    request_settings: OpenAIResponsesModelSettings = {'extra_headers': {'Authorization': 'Bearer connection-key'}}
    async with source.connect(extra_headers={'Authorization': 'Bearer connection-key'}) as connected:
        assert (await Agent(connected, model_settings=request_settings).run('hello')).output == 'ready'
    assert sockets.authorization_headers == ['Bearer connection-key']


async def test_request_settings(allow_model_requests: None, sockets: SocketHarness):
    settings: OpenAIResponsesModelSettings = {
        'openai_service_tier': 'priority',
        'service_tier': 'flex',
        'openai_store': False,
        'temperature': 0.2,
        'extra_headers': {'x-test': 'default', 'User-Agent': 'custom-client'},
        'extra_body': {'temperature': 0.3, 'metadata': {'test': 'websocket'}},
    }
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'), settings=settings)
    options: WebSocketConnectionOptions = {'max_size': 128 * 1024}
    async with source.connect(extra_headers={'x-test': 'connected'}, websocket_connection_options=options) as connected:
        assert (await Agent(source).run('hello', model=connected)).output == 'ready'
    assert sockets.headers[0]['x-test'] == 'connected'
    assert sockets.headers[0]['user-agent'] == 'custom-client'
    assert sockets.options[0]['max_size'] == 128 * 1024
    assert sockets.urls == ['wss://api.openai.com/v1/responses']
    assert source.settings == settings
    sent = sockets.opened[0].sent[0]
    assert sent['service_tier'] == 'priority'
    assert sent['store'] is False
    assert sent['temperature'] == 0.3
    assert sent['metadata'] == {'test': 'websocket'}


async def test_refreshed_shared_client_credentials_keep_open_connection_usable(
    allow_model_requests: None, sockets: SocketHarness
):
    """A shared client's credential refresh for a new socket is not a request header change on an open one."""
    token = 'old-token'

    async def api_key() -> str:
        return token

    sockets.pending.append(ScriptedSocket())
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(openai_client=AsyncOpenAI(api_key=api_key)))
    async with source.connect() as first:
        token = 'refreshed-token'
        async with source.connect() as second:
            assert (await Agent(second).run('hello')).output == 'ready'
            assert (await Agent(first).run('hello')).output == 'ready'
    assert sockets.authorization_headers == ['Bearer old-token', 'Bearer refreshed-token']
    assert [len(socket.sent) for socket in sockets.opened] == [1, 1]

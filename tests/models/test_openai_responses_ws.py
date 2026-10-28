from __future__ import annotations

import asyncio
import sys
import threading
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from copy import copy
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Literal, NoReturn, get_type_hints
from unittest.mock import Mock

import anyio
import httpx2
import pytest
from _pytest.fixtures import SubRequest
from dirty_equals import IsPartialDict
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
    ModelConnectionError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelServerError,
    ModelTimeoutError,
    ModelUnavailableError,
    UnexpectedModelBehavior,
    UnsupportedSettingWarning,
    UserError,
)
from pydantic_ai.models import AbstractModel, Model, ModelRequestContext, ModelRequestParameters, ModelSelectionContext
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers import Provider
from pydantic_ai.providers.gateway import gateway_provider

from ..conftest import try_import
from ..durable_exec.test_durable_operation import (
    JournalDurability,
    _RegisteredDurability,  # pyright: ignore[reportPrivateUsage]
)
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
    from pydantic_ai.providers.openai_chatgpt import OpenAIChatGPTCredentials, OpenAIChatGPTProvider
    from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')
READINESS_WAIT_TIMEOUT = 10
Frame = dict[str, JsonValue]
FRAME_ADAPTER = TypeAdapter(Frame)
WEBSOCKET: OpenAIResponsesModelSettings = {'openai_transport': 'websocket'}

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
    provider: OpenAIProvider
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
    recorded = RecordedResponses(OpenAIProvider(api_key=openai_api_key), cassette)
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


def sent_frames(*cassettes: RealtimeCassette) -> list[Frame]:
    return [
        event.data
        for cassette in cassettes
        for event in cassette.interactions
        if isinstance(event, CassetteMessage) and event.direction == 'sent'
    ]


class Weather(BaseModel):
    temperature: int


def test_websocket_connection_options_type_hint_is_resolvable():
    assert (
        get_type_hints(OpenAIResponsesModel.__init__)['websocket_connection_options']
        == WebSocketConnectionOptions | None
    )


async def test_tool_continuation(allow_model_requests: None, recorded_responses: RecordedResponses):
    """A real tool roundtrip reuses the run's socket and sends only the tool result on continuation."""
    calls: list[str] = []

    def get_weather(city: str) -> int:
        """Get the current temperature in Celsius for a city."""
        calls.append(city)
        return 18

    settings: OpenAIResponsesModelSettings = {
        'openai_transport': 'websocket',
        'openai_previous_response_id': 'auto',
        'openai_store': False,
        'openai_reasoning_effort': 'low',
    }
    model = OpenAIResponsesModel('gpt-6-astra', provider=recorded_responses.provider)
    agent = Agent(model, tools=[get_weather], output_type=NativeOutput(Weather), model_settings=settings)
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
    # The socket echoes `store`, which decides where a response can be continued from.
    assert [
        ((response.provider_details or {}).get('transport'), (response.provider_details or {}).get('store'))
        for response in responses
    ] == [('websocket', False), ('websocket', False)]
    tool_call = next(part for part in responses[0].parts if isinstance(part, ToolCallPart))
    tool_return = next(
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    )
    assert tool_call.tool_call_id == tool_return.tool_call_id
    sent = sent_frames(recorded_responses.cassette)
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
    text_config = sent[0]['text']
    assert isinstance(text_config, dict)
    assert text_config['format'] == IsPartialDict(type='json_schema')


async def test_streaming_sequential_turns(allow_model_requests: None, recorded_responses: RecordedResponses):
    """Runs inside `async with agent:` share one socket, and a later turn continues from the stored response.

    The setting alone defaults `openai_previous_response_id` to `'auto'` on the socket, so the second turn sends only
    its own prompt. The request leaves `store` out, which the socket echoes back as `true`.
    """
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=recorded_responses.provider, settings=WEBSOCKET)
    agent = Agent(model)
    async with agent:
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
    first_response = history[-1]
    assert isinstance(first_response, ModelResponse)
    assert first_response.provider_details is not None
    assert first_response.provider_details['transport'] == 'websocket'
    assert first_response.provider_details['store'] is True
    sent = sent_frames(recorded_responses.cassette)
    assert len(sent) == 2
    assert sent[0]['type'] == 'response.create'
    assert sent[0]['model'] == 'gpt-6.1-sol'
    assert sent[0]['input'] == [{'role': 'user', 'content': 'Reply with the word amber.'}]
    assert 'previous_response_id' not in sent[0]
    assert sent[1]['previous_response_id'] == first_response.provider_response_id
    assert sent[1]['input'] == [{'role': 'user', 'content': 'Reply with the word cobalt.'}]
    assert all('store' not in frame for frame in sent)


async def test_overlapping_runs_continue_on_their_own_lanes(
    allow_model_requests: None, recorded_responses: RecordedResponses
):
    """A run that overlaps another takes a named lane, and each lane continues from its own store-false response.

    The second request goes out after the first response's first text delta and before its terminal event, while
    a long answer is still streaming, so the two responses are in flight on the socket at once.
    """
    settings: OpenAIResponsesModelSettings = {
        'openai_transport': 'websocket',
        'openai_previous_response_id': 'auto',
        'openai_store': False,
    }
    agent = Agent(OpenAIResponsesModel('gpt-6-astra', provider=recorded_responses.provider), model_settings=settings)
    async with agent:
        async with agent.run_stream(
            'Count from one to sixty in words, separated by spaces, then reply with the word amber.'
        ) as first:
            chunks = first.stream_text(delta=True, debounce_by=None)
            await anext(chunks)
            # The first response is still streaming, so the default lane is busy.
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
    first_completed = next(
        index
        for index, frame in enumerate(frames)
        if frame.direction == 'received' and frame.data['type'] == 'response.completed'
    )
    assert frames[first_completed].data.get('stream_id') is None
    # The second request went out, and its response started arriving, before the first response completed.
    assert any(
        frame.direction == 'sent' and frame.data.get('stream_id') == 'lane-1' for frame in frames[:first_completed]
    )
    assert any(
        frame.direction == 'received' and frame.data.get('stream_id') == 'lane-1' for frame in frames[:first_completed]
    )


async def test_cross_socket_continuation_replays_history(
    allow_model_requests: None, recorded_responses: RecordedResponses
):
    """Each run opens its own socket, and a store-false response from a closed socket isn't continued from."""
    settings: OpenAIResponsesModelSettings = {
        'openai_transport': 'websocket',
        'openai_previous_response_id': 'auto',
        'openai_store': False,
    }
    agent = Agent(OpenAIResponsesModel('gpt-6-astra', provider=recorded_responses.provider), model_settings=settings)
    first = await agent.run('Reply with the word amber.')
    second = await agent.run('Reply with the word cobalt.', message_history=first.all_messages())

    assert 'amber' in first.output.lower()
    assert 'cobalt' in second.output.lower()
    assert len(recorded_responses.connections) == 2
    sent = sent_frames(*recorded_responses.cassettes)
    assert len(sent) == 2
    assert all(frame['type'] == 'response.create' and 'previous_response_id' not in frame for frame in sent)
    assert sent[0]['input'] == [{'role': 'user', 'content': 'Reply with the word amber.'}]
    first_input, second_input = sent[0]['input'], sent[1]['input']
    assert isinstance(first_input, list) and isinstance(second_input, list)
    assert len(second_input) > len(first_input)
    assert 'amber' in str(second_input).lower()


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


def function_call_item(call_id: str, item_id: str = 'fc_test') -> Frame:
    return {
        'type': 'function_call',
        'id': item_id,
        'status': 'completed',
        'arguments': '{}',
        'call_id': call_id,
        'name': 'step',
    }


def tool_call_events(response_id: str, call_id: str) -> list[Frame]:
    item = function_call_item(call_id)
    response = completed_response(response_id, output=[item])
    return [
        {'type': 'response.created', 'sequence_number': 0, 'response': {**response, 'status': 'in_progress'}},
        {
            'type': 'response.output_item.added',
            'sequence_number': 1,
            'output_index': 0,
            'item': {**item, 'status': 'in_progress', 'arguments': ''},
        },
        {'type': 'response.output_item.done', 'sequence_number': 2, 'output_index': 0, 'item': item},
        {'type': 'response.completed', 'sequence_number': 3, 'response': response},
    ]


def on_lane(events: list[Frame], stream_id: str) -> list[Frame]:
    return [{**event, 'stream_id': stream_id} for event in events]


def http_reply(response_id: str, *output: Frame) -> httpx2.Response:
    """A complete Responses API reply to an HTTP request."""
    return httpx2.Response(200, json=completed_response(response_id, output=[*output]))


@dataclass
class HTTPResponses:
    """Answer HTTP Responses API requests with complete responses, recording each request."""

    replies: deque[httpx2.Response] = field(default_factory=deque[httpx2.Response])
    requests: list[httpx2.Request] = field(default_factory=list[httpx2.Request])
    client: httpx2.AsyncClient = field(init=False)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        if self.replies:
            return self.replies.popleft()
        return http_reply(
            'resp_http',
            {
                'type': 'message',
                'id': 'msg_http',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': 'from HTTP', 'annotations': [], 'logprobs': []}],
            },
        )

    @property
    def bodies(self) -> list[Frame]:
        return [FRAME_ADAPTER.validate_json(request.content) for request in self.requests]

    def provider(self) -> OpenAIProvider:
        return OpenAIProvider(api_key='test', http_client=self.client)


@pytest.fixture
async def http() -> AsyncIterator[HTTPResponses]:
    responses = HTTPResponses()
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(responses.handle)) as client:
        responses.client = client
        yield responses


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
    connect_count: int = 0
    connect_gate: anyio.Event | None = None
    connect_error: Exception | None = None


@pytest.fixture
def sockets(monkeypatch: pytest.MonkeyPatch) -> SocketHarness:
    """Replace the `websockets` connection with `ScriptedSocket`s, recording each handshake."""
    harness = SocketHarness()

    async def connect(uri: str, *, additional_headers: Mapping[str, str], **options: object) -> ScriptedSocket:
        harness.connect_count += 1
        if harness.connect_gate is not None and harness.connect_count == 1:
            await harness.connect_gate.wait()
        if harness.connect_error is not None:
            raise harness.connect_error
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


def websocket_model(
    provider: Provider[AsyncOpenAI] | None = None,
    *,
    settings: OpenAIResponsesModelSettings | None = None,
    profile: OpenAIModelProfile | None = None,
) -> OpenAIResponsesModel:
    """A model whose own settings ask for the Responses WebSocket."""
    return OpenAIResponsesModel(
        'gpt-4o',
        provider=provider or OpenAIProvider(api_key='test'),
        settings={**WEBSOCKET, **(settings or {})},
        profile=profile,
    )


def user_prompt(text: str) -> list[ModelRequest | ModelResponse]:
    return [ModelRequest.user_text_prompt(text)]


@pytest.mark.parametrize('scope', ['model', 'agent', 'run', 'unset'])
@pytest.mark.parametrize('owned_http_client', [False, True])
async def test_setting_opts_each_run_into_its_own_socket(
    allow_model_requests: None,
    sockets: SocketHarness,
    http: HTTPResponses,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    owned_http_client: bool,
):
    """The setting from the model, agent or run sends the run's requests over a socket the run opens and closes.

    The run doesn't enter the model, so a client the provider created stays open after it, as without the setting.
    """
    if owned_http_client:
        monkeypatch.setattr('pydantic_ai.providers._openai_compatible.create_async_httpx2_client', lambda: http.client)
        provider = OpenAIProvider(api_key='test')
    else:
        provider = http.provider()
    model = OpenAIResponsesModel('gpt-4o', provider=provider, settings=WEBSOCKET if scope == 'model' else None)
    agent = Agent(model, model_settings=WEBSOCKET if scope == 'agent' else None)
    result = await agent.run('hello', model_settings=WEBSOCKET if scope == 'run' else None)

    if scope == 'unset':
        assert result.output == 'from HTTP'
        assert sockets.connect_count == 0
        assert len(http.requests) == 1
    else:
        assert result.output == 'ready'
        socket = sockets.opened[0]
        assert [frame['input'] for frame in socket.sent] == [[{'role': 'user', 'content': 'hello'}]]
        assert socket.close_count == 1
        assert http.requests == []
    assert not http.client.is_closed


async def test_model_selection_paths_use_the_socket(
    allow_model_requests: None, sockets: SocketHarness, capfire: CaptureLogfire
):
    """`run(model=)`, a selection capability and a nested run on `ctx.model` each use the socket during their run.

    A nested run on the parent's model shares the parent's socket, on its own lane.
    """
    logfire.instrument_pydantic_ai()
    explicit_socket = sockets.pending[0]
    explicit_socket.responses = deque([text_events('explicit')])
    selected_socket = ScriptedSocket(responses=deque([text_events('selected')]))
    nested_socket = ScriptedSocket(
        responses=deque(
            [
                tool_call_events('resp_parent', 'call_1'),
                text_events('nested', 'resp_nested'),
                text_events('done', 'resp_done'),
            ]
        )
    )
    sockets.pending.extend([selected_socket, nested_socket])
    model = websocket_model()

    explicit = await Agent(None).run('explicit', model=model)
    selected = await Agent(None, capabilities=[SelectModel(lambda ctx: model)]).run('selected')

    parent = Agent(model)
    worker = Agent(None)

    @parent.tool
    async def step(ctx: RunContext[object]) -> str:
        assert isinstance(ctx.model, OpenAIResponsesModel)
        return (await worker.run('nested', model=ctx.model)).output

    nested = await parent.run('parent')

    assert (explicit.output, selected.output, nested.output) == ('explicit', 'selected', 'done')
    assert sockets.opened == [explicit_socket, selected_socket, nested_socket]
    assert [socket.close_count for socket in sockets.opened] == [1, 1, 1]
    assert [(frame.get('stream_id'), frame.get('previous_response_id')) for frame in nested_socket.sent] == [
        (None, None),
        ('lane-1', None),
        (None, 'resp_parent'),
    ]
    assert nested_socket.sent[1]['input'] == [{'role': 'user', 'content': 'nested'}]
    logfire.force_flush()
    chat_transports = [
        span['attributes'].get('pydantic_ai.transport')
        for span in capfire.exporter.exported_spans_as_dict()
        if span['attributes'].get('gen_ai.operation.name') == 'chat'
    ]
    assert chat_transports == ['websocket'] * 5


async def test_streaming_transport_is_recorded_on_model_span(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, capfire: CaptureLogfire
):
    """A socket request's `chat` span says so, and an HTTP request's carries no transport attribute."""
    logfire.instrument_pydantic_ai()
    sockets.pending[0].responses = deque([text_events('streamed')])
    agent = Agent(websocket_model(http.provider()))
    async with agent.run_stream('hello') as result:
        assert await result.get_output() == 'streamed'
    http_settings: OpenAIResponsesModelSettings = {'openai_transport': 'http'}
    with logfire.span('caller span'):
        await agent.run('over HTTP', model_settings=http_settings)

    logfire.force_flush()
    spans = capfire.exporter.exported_spans_as_dict()
    chat_transports = [
        span['attributes'].get('pydantic_ai.transport')
        for span in spans
        if span['attributes'].get('gen_ai.operation.name') == 'chat'
    ]
    assert chat_transports == ['websocket', None]
    caller_span = next(span for span in spans if span['name'] == 'caller span')
    assert 'pydantic_ai.transport' not in caller_span['attributes']


@pytest.mark.parametrize('entered', ['agent', 'model', 'wrapper'])
async def test_entering_keeps_the_socket_open_across_runs(
    allow_model_requests: None, sockets: SocketHarness, entered: str
):
    """`async with agent:` or `async with model:` keeps one socket for back-to-back runs, which continue on it."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events('first', 'resp_first'), text_events('second', 'resp_second')])
    model = websocket_model()
    agent = Agent(WrapperModel(model) if entered == 'wrapper' else model)
    scope: AbstractAsyncContextManager[object] = model if entered == 'model' else agent
    async with scope:
        first = await agent.run('first')
        assert socket.close_count == 0
        second = await agent.run('second', message_history=first.all_messages())
        assert socket.close_count == 0

    assert second.output == 'second'
    assert sockets.opened == [socket]
    assert socket.close_count == 1
    assert [(frame.get('previous_response_id'), frame['input']) for frame in socket.sent] == [
        (None, [{'role': 'user', 'content': 'first'}]),
        ('resp_first', [{'role': 'user', 'content': 'second'}]),
    ]


async def test_request_outliving_its_lease_keeps_the_socket_until_it_finishes(
    allow_model_requests: None, sockets: SocketHarness
):
    """A request started inside `async with model:` that is still running when the block exits keeps the socket."""
    socket = sockets.pending[0]
    socket.responses.clear()
    model = websocket_model()
    async with model:
        task = asyncio.create_task(model.request(user_prompt('hello'), None, ModelRequestParameters()))
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await socket.receiving.wait()
    assert socket.close_count == 0
    socket.push(*text_events('late'))
    assert (await task).text == 'late'
    assert socket.close_count == 1


@pytest.mark.parametrize('stream', [False, True])
async def test_request_without_a_lease_uses_http(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, stream: bool
):
    """A direct request outside a run or `async with model:` has nothing to keep a socket open for, so it uses HTTP."""
    model = websocket_model(http.provider())
    if stream:
        events = text_events('from HTTP', 'resp_http')
        body = ''.join(
            f'event: {event["type"]}\ndata: {FRAME_ADAPTER.dump_json(event).decode()}\n\n' for event in events
        )
        http.replies.append(httpx2.Response(200, content=body.encode(), headers={'content-type': 'text/event-stream'}))
        async with model.request_stream(user_prompt('hello'), None, ModelRequestParameters()) as streamed:
            async for _ in streamed:
                pass
        response = streamed.get()
        assert response.provider_details is not None
        assert 'transport' not in response.provider_details
        assert 'store' not in response.provider_details
    else:
        response = await model.request(user_prompt('hello'), None, ModelRequestParameters())
    assert response.text == 'from HTTP'
    assert sockets.connect_count == 0
    # The setting still records the response's history, since a later socket request continues from it by default.
    assert response.history_digest is not None


async def test_each_event_loop_uses_its_own_socket(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    """Each event loop opens and closes its own socket on the shared model."""
    socket = sockets.pending[0]
    second_socket = ScriptedSocket()
    sockets.pending.append(second_socket)

    def other_run() -> str:
        with asyncio.Runner():
            return Agent(model).run_sync('other loop').output

    model = websocket_model(http.provider())
    async with model:
        assert (await model.request(user_prompt('first'), None, ModelRequestParameters())).text == 'ready'
        other_loop = await anyio.to_thread.run_sync(other_run)
        assert socket.close_count == 0
    assert other_loop == 'ready'
    assert len(second_socket.sent) == 1
    assert second_socket.close_count == 1
    assert model._websocket.states == {}  # pyright: ignore[reportPrivateUsage]
    assert len(http.requests) == 0
    assert len(socket.sent) == 1
    assert socket.close_count == 1


async def test_run_on_another_event_loop_doesnt_wait_for_the_socket_this_loop_opens(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    """A handshake on one loop never blocks the other loop's socket."""
    gate = sockets.connect_gate = anyio.Event()
    sockets.pending.append(ScriptedSocket())
    model = websocket_model(http.provider())
    outputs: list[str] = []

    def other_run() -> None:
        with asyncio.Runner():
            outputs.append(Agent(model).run_sync('other loop').output)

    other_loop = threading.Thread(target=other_run, daemon=True)
    # The handshake outlasts the time the other run gets to finish.
    settings: OpenAIResponsesModelSettings = {'timeout': 6 * READINESS_WAIT_TIMEOUT}
    async with model:
        opening = asyncio.create_task(model.request(user_prompt('first'), settings, ModelRequestParameters()))
        await anyio.wait_all_tasks_blocked()
        assert sockets.connect_count == 1
        other_loop.start()
        await anyio.to_thread.run_sync(other_loop.join, READINESS_WAIT_TIMEOUT)
        waited = other_loop.is_alive()
        gate.set()
        assert (await opening).text == 'ready'
    assert not waited
    assert outputs == ['ready']
    assert sockets.connect_count == 2


async def test_reservation_from_another_event_loop_leaves_the_open_socket_alone(
    allow_model_requests: None, sockets: SocketHarness
):
    """A request on another thread's event loop can find no socket when it chooses one, and an open one by the time it
    reserves a lane, so the reservation checks the socket's event loop too and leaves the request to use HTTP.

    Two threads can't be interleaved on cue, so the test reserves directly, with a loop that never runs standing in for
    the other thread's.
    """

    async def connect() -> NoReturn:
        raise AssertionError('the open socket is never replaced')  # pragma: no cover

    model = websocket_model()
    holder = model._websocket  # pyright: ignore[reportPrivateUsage]
    other_loop = asyncio.new_event_loop()
    try:
        async with model:
            await model.request(user_prompt('hello'), None, ModelRequestParameters())
            state = holder.states[asyncio.get_running_loop()]
            socket = state.socket
            assert socket is not None
            assert await state.reserve(other_loop, connect, None, socket.headers) is None
    finally:
        other_loop.close()
    assert sockets.connect_count == 1


@pytest.mark.parametrize('lease_ends', [False, True])
async def test_request_waiting_for_another_to_open_the_socket(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, lease_ends: bool
):
    """A request that waits while another opens the socket takes a lane on it.

    When that handshake fails after the request's lease ended, the request uses HTTP rather than open a socket that
    would serve it alone.
    """
    socket = sockets.pending[0]
    socket.responses = deque([text_events('first'), text_events('second')])
    gate = sockets.connect_gate = anyio.Event()
    model = websocket_model(http.provider())
    async with model:
        opening = asyncio.create_task(model.request(user_prompt('first'), None, ModelRequestParameters()))
        waiting = asyncio.create_task(model.request(user_prompt('second'), None, ModelRequestParameters()))
        await anyio.wait_all_tasks_blocked()
        assert sockets.connect_count == 1
        if not lease_ends:
            gate.set()
            assert [(await opening).text, (await waiting).text] == ['first', 'second']
    if lease_ends:
        sockets.connect_error = OSError('unreachable')
        gate.set()
        with pytest.raises(ModelConnectionError, match='WebSocket connection failed: unreachable'):
            await opening
        assert (await waiting).text == 'from HTTP'
    assert sockets.connect_count == 1
    assert [frame.get('stream_id') for frame in socket.sent] == ([] if lease_ends else [None, 'lane-1'])


async def test_transport_failure_fails_every_request_on_the_socket(allow_model_requests: None, sockets: SocketHarness):
    """A send that fails ends the socket, so a request still reading from it fails too, and the next opens another."""
    socket = sockets.pending[0]
    socket.responses.clear()
    next_socket = ScriptedSocket()
    sockets.pending.append(next_socket)
    model = websocket_model()
    async with model:
        reading = asyncio.create_task(model.request(user_prompt('reading'), None, ModelRequestParameters()))
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await socket.waiting_for_input.wait()
        socket.send_error = OSError('test failure')
        with pytest.raises(ModelConnectionError, match='WebSocket request failed: test failure') as sending:
            await model.request(user_prompt('sending'), None, ModelRequestParameters())
        assert sending.value.phase == 'write'
        with pytest.raises(ModelConnectionError, match='WebSocket request failed: test failure') as read:
            await reading
        assert read.value.phase == 'read'
        assert (await model.request(user_prompt('again'), None, ModelRequestParameters())).text == 'ready'
    assert sockets.opened == [socket, next_socket]


@pytest.mark.parametrize(
    ('provider', 'stream_id_lanes'),
    [
        pytest.param(lambda: OpenAIProvider(api_key='test'), None, id='openai-host'),
        pytest.param(lambda: OpenAIProvider(base_url='https://api.x.ai/v1', api_key='test'), True, id='opted-in'),
    ],
)
async def test_concurrent_runs_share_the_socket_on_separate_lanes(
    allow_model_requests: None,
    sockets: SocketHarness,
    provider: Callable[[], Provider[AsyncOpenAI]],
    stream_id_lanes: bool | None,
):
    """Concurrent runs on one model share its socket, each on its own lane, and the two responses can interleave.

    Lanes run by default on OpenAI's own host, and elsewhere when the profile opts in.
    """
    socket = sockets.pending[0]
    socket.responses.clear()
    model = websocket_model(
        provider(),
        profile=None
        if stream_id_lanes is None
        else OpenAIModelProfile(openai_responses_supports_websocket_stream_id=stream_id_lanes),
    )
    # An explicit value is kept as given; unset stays unset, so the socket decides by host when it opens.
    assert model.profile.get('openai_responses_supports_websocket_stream_id') is stream_id_lanes
    results: dict[str, str] = {}

    async def run(prompt: str) -> None:
        results[prompt] = (await Agent(model).run(prompt)).output

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


@pytest.mark.parametrize('endpoint', ['other-host', 'other-websocket-host', 'opted-out', 'active-response-limit'])
async def test_request_without_a_free_lane_uses_http(
    allow_model_requests: None,
    sockets: SocketHarness,
    http: HTTPResponses,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
):
    """A request that finds no free lane goes over HTTP right away instead of waiting for one.

    Off OpenAI's WebSocket host, or when the profile opts out, the socket has only its default lane: a host without
    lanes doesn't tag its events with `stream_id`, so one request would read another's response. On OpenAI's host,
    a socket runs at most a fixed number of responses at once.
    """
    if endpoint == 'other-host':
        provider = OpenAIProvider(base_url='https://api.x.ai/v1', api_key='test', http_client=http.client)
    elif endpoint == 'other-websocket-host':
        client = AsyncOpenAI(api_key='test', websocket_base_url='wss://proxy.example/v1', http_client=http.client)
        provider = OpenAIProvider(openai_client=client)
    else:
        provider = http.provider()
    if endpoint == 'active-response-limit':
        monkeypatch.setattr('pydantic_ai.models._openai_responses_websocket.ACTIVE_RESPONSE_LIMIT', 1)
    socket = sockets.pending[0]
    socket.responses.clear()
    model = websocket_model(
        provider,
        profile=OpenAIModelProfile(openai_responses_supports_websocket_stream_id=False)
        if endpoint == 'opted-out'
        else None,
    )
    results: dict[str, str] = {}

    async def run(prompt: str) -> None:
        results[prompt] = (await Agent(model).run(prompt)).output

    async with model, anyio.create_task_group() as tasks:
        tasks.start_soon(run, 'first')
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await socket.receiving.wait()
        # The second run finishes over HTTP while the first still waits for its response on the socket.
        await run('second')
        assert results == {'second': 'from HTTP'}
        socket.push(*text_events('first'))

    assert results == {'first': 'first', 'second': 'from HTTP'}
    assert [frame.get('stream_id') for frame in socket.sent] == [None]
    assert [body['input'] for body in http.bodies] == [[{'role': 'user', 'content': 'second'}]]


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
    agent = Agent(websocket_model())
    async with agent:
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


@pytest.mark.parametrize('stream_id_lanes', [True, False])
async def test_abandoned_response_evicts_its_lane_response(
    allow_model_requests: None, sockets: SocketHarness, stream_id_lanes: bool
):
    """A response closed early evicts its lane's cached response, and holds the lane until the rest is discarded.

    With lanes, the next request takes another lane and its reads discard the rest. Without them, nothing would read
    the abandoned response, so the socket closes and the next request opens another.
    """
    socket = sockets.pending[0]
    abandoned = text_events('abandoned', 'resp_abandoned')
    socket.responses = deque([text_events(response_id='resp_first'), abandoned[:3]])
    next_socket = ScriptedSocket(responses=deque())
    sockets.pending.append(next_socket)
    after = next_socket if not stream_id_lanes else socket
    after.responses.extend([text_events(response_id='resp_next'), text_events(response_id='resp_again')])
    model = websocket_model(profile=OpenAIModelProfile(openai_responses_supports_websocket_stream_id=stream_id_lanes))

    async with model:
        first = await Agent(model).run('first')
        async with model.request_stream(
            [*first.all_messages(), ModelRequest.user_text_prompt('abandon')], None, ModelRequestParameters()
        ) as response:
            async for _ in response:  # pragma: no branch
                break
        assert socket.close_count == (0 if stream_id_lanes else 1)
        socket.push(*abandoned[3:])
        await Agent(model).run('continue', message_history=first.all_messages())
        # Once the abandoned response was discarded, its lane was free again.
        await Agent(model).run('again')

    assert [(frame.get('stream_id'), frame.get('previous_response_id')) for frame in socket.sent[:2]] == [
        (None, None),
        (None, 'resp_first'),
    ]
    continued = after.sent[-2]
    assert 'previous_response_id' not in continued
    assert continued['input'] == [
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': 'ready'},
        {'role': 'user', 'content': 'continue'},
    ]
    assert [frame.get('stream_id') for frame in after.sent[-2:]] == (
        ['lane-1', None] if stream_id_lanes else [None, None]
    )


async def test_error_frame_evicts_the_lane_response(allow_model_requests: None, sockets: SocketHarness):
    """A request rejected with an error frame evicts its lane's cached response, so continuing from it replays."""
    socket = sockets.pending[0]
    error: Frame = {
        'type': 'error',
        'status': 400,
        'error': {'type': 'invalid_request_error', 'code': 'invalid_test', 'message': 'rejected'},
    }
    socket.responses = deque([text_events(response_id='resp_first'), [error], text_events(response_id='resp_next')])
    agent = Agent(websocket_model())

    async with agent:
        first = await agent.run('first')
        with pytest.raises(ModelHTTPError):
            await agent.run('rejected', message_history=first.all_messages())
        await agent.run('continue', message_history=first.all_messages())

    assert [(frame.get('stream_id'), frame.get('previous_response_id')) for frame in socket.sent] == [
        (None, None),
        (None, 'resp_first'),
        (None, None),
    ]
    assert socket.sent[-1]['input'] == [
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': 'ready'},
        {'role': 'user', 'content': 'continue'},
    ]


async def test_closing_a_response_after_its_terminal_was_routed_frees_its_lane(
    allow_model_requests: None, sockets: SocketHarness
):
    """Another lane's reader can buffer a whole response, so closing it early still frees the lane."""
    socket = sockets.pending[0]
    first = text_events('first', 'resp_first')
    second = on_lane(text_events('second', 'resp_second'), 'lane-1')
    socket.responses = deque([first[:1], second[:1], text_events(response_id='resp_next')])
    model = websocket_model()

    async with model:
        async with model.request_stream(user_prompt('first'), None, ModelRequestParameters()):
            async with model.request_stream(user_prompt('second'), None, ModelRequestParameters()) as response:
                socket.push(*first[1:], *second[1:])
                async for _ in response:
                    pass
                assert response.get().text == 'second'
        assert (await Agent(model).run('next')).output == 'ready'

    assert [frame.get('stream_id') for frame in socket.sent] == [None, 'lane-1', None]


@pytest.mark.parametrize('error', ['unattributed', 'connection_limit'])
async def test_connection_scoped_error_fails_the_socket(allow_model_requests: None, sockets: SocketHarness, error: str):
    """An error no active lane owns, or the connection's lifetime limit, fails the socket, and the next run opens another."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events(response_id='resp_first')])
    code = None if error == 'unattributed' else 'websocket_connection_limit_reached'
    failure: Frame = {'type': 'error', 'error': {'type': 'invalid_request_error', 'code': code, 'message': 'ended'}}
    if error == 'connection_limit':
        failure['status'] = 400
        socket.responses.append([failure])
    next_socket = ScriptedSocket()
    sockets.pending.append(next_socket)
    agent = Agent(websocket_model())

    async with agent:
        await agent.run('first')
        if error == 'unattributed':
            # Without a `stream_id`, these belong to the idle default lane rather than the named lane waiting.
            socket.push(text_events()[2], failure)
        expected = 'ended' if code is None else f'{code}: ended'
        with pytest.raises(ModelConnectionError, match=f'Responses WebSocket connection failed: {expected}$') as raised:
            await agent.run('second')
        assert raised.value.phase == 'read'
        assert socket.close_count == 1
        assert (await agent.run('third')).output == 'ready'

    assert len(socket.sent) == 2
    assert [frame['input'] for frame in next_socket.sent] == [[{'role': 'user', 'content': 'third'}]]


async def test_cancelled_send_keeps_the_socket_for_other_lanes(allow_model_requests: None, sockets: SocketHarness):
    """A cancelled send may already have reached the server, so only its lane waits out that response."""
    socket = sockets.pending[0]
    socket.send_gate = anyio.Event()
    socket.responses.clear()
    model = websocket_model()

    async with model:

        async def cancelled_run() -> None:
            await Agent(model).run('cancelled')

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(cancelled_run)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await socket.sending.wait()
            tasks.cancel_scope.cancel()
        socket.send_gate = None
        # The scripted socket drops the gated frame, but a real send would have written it: its response arrives.
        socket.push(*text_events('late', response_id='resp_late'))
        socket.responses.append(text_events(response_id='resp_next'))
        assert (await Agent(model).run('next')).output == 'ready'
        socket.responses.append(text_events('again', response_id='resp_again'))
        assert (await Agent(model).run('again')).output == 'again'
        assert socket.close_count == 0

    # Once the late response was discarded, the default lane was free again.
    assert [frame.get('stream_id') for frame in socket.sent] == ['lane-1', None]


async def test_parallel_nested_runs_share_the_parent_socket(allow_model_requests: None, sockets: SocketHarness):
    """A tool can start parallel runs on its own run's model, which share that run's socket on their own lanes."""
    socket = sockets.pending[0]
    socket.responses = deque(
        [tool_call_events('resp_parent', 'call_1'), [], [], text_events('done', response_id='resp_final')]
    )
    model = websocket_model()
    agent = Agent(model)
    worker = Agent(model)

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
    assert socket.close_count == 1


@pytest.mark.parametrize('store', [True, False, None])
async def test_continuation_across_transports_follows_storage(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, store: bool | None
):
    """A response the socket stored continues anywhere; one it didn't, or didn't say, only on its own socket's lane.

    An HTTP response continues on a socket as it does over HTTP. Where a response can't be continued from, the
    history is replayed instead.
    """
    socket_a = sockets.pending[0]
    socket_a.responses = deque([text_events('from socket A', 'resp_socket_a', store=store)])
    socket_b = ScriptedSocket(
        responses=deque([text_events('from socket B', 'resp_socket_b'), text_events('again on B', 'resp_socket_b2')])
    )
    sockets.pending.append(socket_b)
    model = websocket_model(http.provider())
    over_http: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'openai_transport': 'http'}

    socket_result = await Agent(model).run('on socket A')
    # Continuing from stored history works as from the run's own: the response ID, `transport` and `store` survive.
    # The JSON is compared, since `provider_details['timestamp']` comes back as a string.
    dumped = ModelMessagesTypeAdapter.dump_json(socket_result.all_messages())
    history = ModelMessagesTypeAdapter.validate_json(dumped)
    assert ModelMessagesTypeAdapter.dump_json(history) == dumped
    http_result = await Agent(model, model_settings=over_http).run('over HTTP', message_history=history)
    agent_b = Agent(model)
    async with agent_b:
        socket_b_result = await agent_b.run('on socket B', message_history=history)
        after_http = await agent_b.run('after HTTP', message_history=http_result.all_messages())

    socket_response = history[-1]
    assert isinstance(socket_response, ModelResponse)
    assert socket_response.provider_response_id == 'resp_socket_a'
    assert socket_response.provider_details is not None
    assert socket_response.provider_details.get('transport') == 'websocket'
    assert socket_response.provider_details.get('store') is store
    assert (socket_b_result.output, after_http.output) == ('from socket B', 'again on B')

    replayed_socket_a: list[Frame] = [
        {'role': 'user', 'content': 'on socket A'},
        {'role': 'assistant', 'content': 'from socket A'},
    ]
    http_body = http.bodies[0]
    socket_b_frame, after_http_frame = socket_b.sent
    if store:
        assert http_body['previous_response_id'] == 'resp_socket_a'
        assert http_body['input'] == [{'role': 'user', 'content': 'over HTTP'}]
        assert socket_b_frame['previous_response_id'] == 'resp_socket_a'
        assert socket_b_frame['input'] == [{'role': 'user', 'content': 'on socket B'}]
    else:
        assert 'previous_response_id' not in http_body
        assert http_body['input'] == [*replayed_socket_a, {'role': 'user', 'content': 'over HTTP'}]
        assert 'previous_response_id' not in socket_b_frame
        assert socket_b_frame['input'] == [*replayed_socket_a, {'role': 'user', 'content': 'on socket B'}]
    assert after_http_frame['previous_response_id'] == 'resp_http'
    assert after_http_frame['input'] == [{'role': 'user', 'content': 'after HTTP'}]


@pytest.mark.parametrize('setting', ['conversation', 'previous_response'])
async def test_socket_requests_keep_an_explicit_server_side_state_setting(
    allow_model_requests: None, sockets: SocketHarness, setting: str
):
    """Socket requests default `openai_previous_response_id` to `'auto'` only while neither state setting is set.

    `openai_conversation_id` can't be combined with it, and a concrete response ID is kept as given.
    """
    settings: OpenAIResponsesModelSettings = (
        {'openai_conversation_id': 'conv_test'}
        if setting == 'conversation'
        else {'openai_previous_response_id': 'resp_seed'}
    )
    assert (await Agent(websocket_model()).run('hello', model_settings=settings)).output == 'ready'

    sent = sockets.opened[0].sent[0]
    if setting == 'conversation':
        assert sent['conversation'] == 'conv_test'
        assert 'previous_response_id' not in sent
    else:
        assert sent['previous_response_id'] == 'resp_seed'


async def test_per_step_selection_switches_websocket_and_http(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    """A selector can change transport inside one run; with `openai_store=False`, nothing is continued across it.

    The unstored socket turn is replayed over HTTP, and the HTTP turn, which OpenAI didn't store either, is replayed
    on the socket instead of chained from.
    """
    socket = sockets.pending[0]
    socket.responses = deque([tool_call_events('resp_socket', 'call_socket'), text_events('done', 'resp_final')])
    http.replies.append(http_reply('resp_http', function_call_item('call_http', 'fc_http')))
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'openai_store': False}
    http_model = OpenAIResponsesModel('gpt-4o', provider=http.provider())
    socket_model = websocket_model(http.provider())
    selection_contexts: list[tuple[int, Model | None]] = []

    def select(ctx: ModelSelectionContext[object]) -> Model:
        selection_contexts.append((ctx.run_step, ctx.model))
        return socket_model if ctx.run_step in (1, 3) else http_model

    agent = Agent(http_model, deps_type=object, capabilities=[SelectModel(select)], model_settings=settings)
    tool_models: list[AbstractModel] = []

    @agent.tool
    def step(ctx: RunContext[object]) -> str:
        tool_models.append(ctx.model)
        return str(len(tool_models))

    result = await agent.run('start')

    assert result.output == 'done'
    assert selection_contexts == [(1, http_model), (2, socket_model), (3, http_model)]
    assert tool_models == [socket_model, http_model]
    replayed_socket_turn: list[Frame] = [
        {'role': 'user', 'content': 'start'},
        {'type': 'function_call', 'name': 'step', 'arguments': '{}', 'call_id': 'call_socket'},
        {'type': 'function_call_output', 'call_id': 'call_socket', 'output': '1'},
    ]
    http_body = http.bodies[0]
    assert http_body['store'] is False
    assert http_body['input'] == replayed_socket_turn
    assert 'previous_response_id' not in http_body
    assert [(frame.get('stream_id'), frame.get('previous_response_id')) for frame in socket.sent] == [
        (None, None),
        ('lane-1', None),
    ]
    assert socket.sent[1]['input'] == [
        *replayed_socket_turn,
        {'type': 'function_call', 'name': 'step', 'arguments': '{}', 'call_id': 'call_http'},
        {'type': 'function_call_output', 'call_id': 'call_http', 'output': '2'},
    ]
    assert socket.close_count == 1


@pytest.mark.parametrize('history', ['http', 'stored_socket'])
async def test_unstored_socket_request_continues_only_from_its_lane(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, history: str
):
    """A socket request with `openai_store=False` replays a history its lane didn't cache.

    That covers an HTTP response, which may never have been stored, as in a Zero Data Retention history saved before
    the setting was turned on, and a response another socket stored. Its own lane's response is still continued from.
    """
    unstored: OpenAIResponsesModelSettings = {'openai_store': False}
    if history == 'http':
        earlier = await Agent(OpenAIResponsesModel('gpt-4o', provider=http.provider()), model_settings=unstored).run(
            'earlier'
        )
        earlier_text = 'from HTTP'
    else:
        sockets.pending[0].responses = deque([text_events('stored', 'resp_stored', store=True)])
        sockets.pending.append(ScriptedSocket())
        earlier = await Agent(websocket_model()).run('earlier')
        earlier_text = 'stored'
    socket = sockets.pending[0]
    socket.responses.append(text_events('again', 'resp_again'))
    agent = Agent(websocket_model(http.provider()), model_settings=unstored)
    async with agent:
        later = await agent.run('later', message_history=earlier.all_messages())
        await agent.run('again', message_history=later.all_messages())

    replayed, continued = socket.sent
    assert 'previous_response_id' not in replayed
    assert replayed['input'] == [
        {'role': 'user', 'content': 'earlier'},
        {'role': 'assistant', 'content': earlier_text},
        {'role': 'user', 'content': 'later'},
    ]
    assert continued['previous_response_id'] == 'resp_test'
    assert continued['input'] == [{'role': 'user', 'content': 'again'}]


async def test_history_edit_after_an_http_fallback_replays_on_the_socket(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    """A request that asked for the socket but went over HTTP records its history, like one on the socket.

    So once earlier messages are edited, the next socket request replays them rather than chain from it.
    """
    socket = sockets.pending[0]
    socket.responses.clear()
    model = websocket_model(
        http.provider(), profile=OpenAIModelProfile(openai_responses_supports_websocket_stream_id=False)
    )
    async with model:
        busy = asyncio.create_task(Agent(model).run('busy'))
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await socket.receiving.wait()
        # The socket's only lane is busy, so this request goes over HTTP.
        fallback = await Agent(model).run('secret')
        socket.push(*text_events('busy', 'resp_busy'))
        assert (await busy).output == 'busy'
        fallback_request, fallback_response = fallback.all_messages()
        assert isinstance(fallback_request, ModelRequest) and isinstance(fallback_response, ModelResponse)
        assert fallback_response.history_digest is not None
        fallback_request.parts = [UserPromptPart('[redacted]')]
        socket.responses.append(text_events('after', 'resp_after'))
        await Agent(model).run('after', message_history=fallback.all_messages())

    after = socket.sent[-1]
    assert 'previous_response_id' not in after
    assert after['input'] == [
        {'role': 'user', 'content': '[redacted]'},
        {'role': 'assistant', 'content': 'from HTTP'},
        {'role': 'user', 'content': 'after'},
    ]


async def test_per_step_selection_switches_distinct_sockets(allow_model_requests: None, sockets: SocketHarness):
    """Another model's socket can't continue from an unstored response cached on the first one's."""
    first_socket = sockets.pending[0]
    first_socket.responses = deque([tool_call_events('resp_first', 'call_first')])
    second_socket = ScriptedSocket(responses=deque([text_events('done', 'resp_second')]))
    sockets.pending.append(second_socket)
    first, second = websocket_model(), websocket_model()

    def select(ctx: ModelSelectionContext[object]) -> Model:
        return first if ctx.run_step == 1 else second

    agent = Agent(first, deps_type=object, capabilities=[SelectModel(select)])

    @agent.tool_plain
    def step() -> str:
        return '1'

    assert (await agent.run('start')).output == 'done'
    assert first_socket.sent[0]['input'] == [{'role': 'user', 'content': 'start'}]
    assert 'previous_response_id' not in second_socket.sent[0]
    assert second_socket.sent[0]['input'] == [
        {'role': 'user', 'content': 'start'},
        {'type': 'function_call', 'name': 'step', 'arguments': '{}', 'call_id': 'call_first'},
        {'type': 'function_call_output', 'call_id': 'call_first', 'output': '1'},
    ]
    assert [first_socket.close_count, second_socket.close_count] == [1, 1]


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('edit_history', [False, True])
async def test_same_socket_continuation_validates_history(
    allow_model_requests: None, sockets: SocketHarness, stream: bool, edit_history: bool
):
    """A copy of the model shares its socket and digest key, and continues only from unchanged history."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events('first answer', 'resp_first'), text_events('again', 'resp_second')])
    model = OpenAIResponsesModel(
        'gpt-4o', provider=OpenAIProvider(api_key='test'), settings=WEBSOCKET, history_digest_key=b'conversation secret'
    )
    unstored: OpenAIResponsesModelSettings = {'openai_store': False}
    agent = Agent(model, model_settings=unstored)

    async with model:
        if stream:
            async with agent.run_stream('secret') as first_stream:
                await first_stream.get_output()
                history = first_stream.all_messages()
        else:
            history = (await agent.run('secret')).all_messages()

        request, response = history
        assert isinstance(request, ModelRequest)
        assert isinstance(response, ModelResponse)
        assert response.history_digest is not None
        if edit_history:
            request.parts = [UserPromptPart('[redacted]')]

        copied = copy(model)
        if stream:
            async with agent.run_stream('follow up', model=copied, message_history=history) as second_stream:
                assert await second_stream.get_output() == 'again'
        else:
            assert (await agent.run('follow up', model=copied, message_history=history)).output == 'again'

    assert sockets.opened == [socket]
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
    hooks: Hooks[None] = Hooks()

    @hooks.on.before_model_request
    async def add_prompt(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        if len(ctx.messages) == 1:
            ctx.enqueue(SystemPromptPart('Only for this request.', scope='request'))
        return request_context

    agent = Agent(websocket_model(settings={'openai_store': False}), capabilities=[hooks], deps_type=type(None))
    async with agent:
        if stream:
            async with agent.run_stream('hello') as first_stream:
                await first_stream.get_output()
                history = first_stream.all_messages()
            async with agent.run_stream('follow up', message_history=history) as second_stream:
                assert await second_stream.get_output() == 'again'
        else:
            first = await agent.run('hello')
            assert (await agent.run('follow up', message_history=first.all_messages())).output == 'again'

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


@pytest.mark.parametrize('transport', ['http', 'websocket'])
@pytest.mark.parametrize('compaction_provider', ['openai', 'openai-codex', 'azure'])
async def test_compatible_compaction_stops_older_response_chaining(
    allow_model_requests: None,
    sockets: SocketHarness,
    http: HTTPResponses,
    transport: Literal['http', 'websocket'],
    compaction_provider: str,
):
    history: list[ModelRequest | ModelResponse] = [
        ModelRequest.user_text_prompt('before compaction'),
        ModelResponse(
            parts=[TextPart('old answer')],
            provider_name='openai',
            provider_response_id='resp_old',
            provider_details={'store': True},
        ),
        ModelResponse(
            parts=[
                CompactionPart(
                    id='cmp_1',
                    provider_name=compaction_provider,
                    provider_details={'encrypted_content': 'encrypted'},
                )
            ],
            provider_name=compaction_provider,
            provider_details={'compaction': True},
            canonical_system='azure' if compaction_provider == 'azure' else 'openai',
        ),
    ]
    settings = OpenAIResponsesModelSettings(openai_transport=transport, openai_previous_response_id='auto')
    model = OpenAIResponsesModel(
        'gpt-4o',
        provider=http.provider(),
        settings=settings,
        profile=OpenAIModelProfile(openai_responses_supports_prompt_cache_diagnostics=True),
    )
    await Agent(model).run('after compaction', message_history=history)

    body = sockets.opened[0].sent[0] if transport == 'websocket' else http.bodies[0]
    if compaction_provider == 'azure':
        assert body['previous_response_id'] == 'resp_old'
        assert body['prompt_cache_options'] == {'comparison_response_id': 'resp_old'}
        assert body['input'] == [{'role': 'user', 'content': 'after compaction'}]
    else:
        assert {key: body[key] for key in ('previous_response_id', 'prompt_cache_options') if key in body} == {}
        assert body['input'] == [
            {'id': 'cmp_1', 'encrypted_content': 'encrypted', 'type': 'compaction'},
            {'role': 'user', 'content': 'after compaction'},
        ]


async def test_token_counting_and_compaction_use_http(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    """Token counting and compaction go over HTTP, replay unstored socket history, and keep the SDK's retries."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events('socket answer', 'resp_socket', store=False)])
    http.replies.extend(
        [
            httpx2.Response(200, json={'input_tokens': 16, 'object': 'response.input_tokens'}),
            httpx2.Response(
                200,
                json={
                    'id': 'resp_compact',
                    'created_at': 0,
                    'object': 'response.compaction',
                    'output': [{'id': 'comp_1', 'encrypted_content': 'encrypted', 'type': 'compaction'}],
                    'usage': {'input_tokens': 1, 'output_tokens': 1, 'total_tokens': 2},
                },
            ),
        ]
    )
    model = websocket_model(http.provider())
    # Both operations go over HTTP, where the SDK honors `max_retries`, so they don't warn that it's ignored.
    settings: OpenAIResponsesModelSettings = {'openai_previous_response_id': 'auto', 'max_retries': 0}
    async with model:
        first_request = ModelRequest.user_text_prompt('first')
        first_response = await model.request([first_request], None, ModelRequestParameters())
        context = ModelRequestContext(
            model=model,
            messages=[first_request, first_response, ModelRequest.user_text_prompt('next')],
            model_settings=settings,
            model_request_parameters=ModelRequestParameters(),
        )
        counted = await model.count_tokens(context.messages, settings, context.model_request_parameters)
        compacted = await model.compact_messages(context)

    assert (first_response.provider_details or {}).get('transport') == 'websocket'
    assert len(socket.sent) == 1
    assert counted.input_tokens == 16
    assert compacted.provider_response_id == 'resp_compact'
    assert [request.url.path for request in http.requests] == ['/v1/responses/input_tokens', '/v1/responses/compact']
    for body in http.bodies:
        assert 'previous_response_id' not in body
        assert 'socket answer' in str(body['input'])


@pytest.mark.parametrize('durability', [JournalDurability, _RegisteredDurability])
async def test_durable_units_borrow_an_external_socket(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, durability: type[JournalDurability]
):
    """A registered model borrows a socket while an explicit context outlives its durable units.

    Engines whose units run a registered handler (Temporal, DBOS) and those that wrap a callback (Prefect) both mark
    the model unit.
    """
    agent = Agent(websocket_model(http.provider()), name='responses', capabilities=[durability()])
    async with agent:
        assert (await agent.run('hello')).output == 'ready'
    assert sockets.connect_count == 1
    assert len(http.requests) == 0


@asynccontextmanager
async def local_server(handler: Callable[[ServerConnection], Awaitable[None]]) -> AsyncGenerator[OpenAIResponsesModel]:
    """Serve `handler` from a real `websockets` server on localhost and yield an entered model pointed at it.

    The scripted socket replaces `websockets`, so it cannot pin the library's message size limit or flow control,
    and no recorded provider response exercises either: these tests run the real library on both ends instead.
    The server echoes `stream_id`, so the model runs lanes on it.
    """
    async with serve(handler, '127.0.0.1', 0) as server:
        port: int = next(iter(server.sockets)).getsockname()[1]
        model = websocket_model(
            OpenAIProvider(base_url=f'http://127.0.0.1:{port}/v1', api_key='test'),
            profile=OpenAIModelProfile(openai_responses_supports_websocket_stream_id=True),
        )
        async with model:
            yield model


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

    async with local_server(handler) as model:
        assert (await Agent(model).run('large')).output == large
        assert (await Agent(model).run('next')).output == 'ready'
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

    async with local_server(handler) as model:
        async with model.request_stream(user_prompt('long answer'), None, ModelRequestParameters()) as streamed:
            async for _ in streamed:  # pragma: no branch
                break
        abandoned.set()
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await keepalive_done.wait()
        assert keepalive_answered == [True]
        assert (await Agent(model).run('next')).output == 'ready'


@pytest.mark.parametrize('interrupt', ['early_exit', 'explicit_close', 'cancel'])
async def test_interruption(allow_model_requests: None, sockets: SocketHarness, interrupt: str):
    socket = sockets.pending[0]
    socket.responses = deque([text_events()[:-1], text_events(response_id='resp_next')])
    model = websocket_model()
    before = asyncio.all_tasks()
    async with model:
        if interrupt == 'early_exit':
            async with Agent(model).run_stream('hello') as result:
                async for _ in result.stream_text(delta=True):  # pragma: no branch
                    break
        elif interrupt == 'explicit_close':
            async with model.request_stream(user_prompt('hello'), None, ModelRequestParameters()) as response:
                await response.close_stream()
        else:
            started = anyio.Event()

            async def consume() -> None:
                async with Agent(model).run_stream('hello') as result:
                    async for _ in result.stream_text(delta=True):
                        started.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(consume)
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await started.wait()
                tasks.cancel_scope.cancel()
        assert (await Agent(model).run('reuse')).output == 'ready'
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
    model = websocket_model()
    async with model:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            async with model.request_stream(user_prompt('hello'), None, ModelRequestParameters()) as response:

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
        assert (await Agent(model).run('next')).output == 'ready'
    assert socket.close_count == 1


@pytest.mark.parametrize('stream', [False, True])
async def test_fallback_survives_primary_errors_across_tool_round_trips(
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
    agent = Agent(FallbackModel(websocket_model(settings={'max_retries': 0}), websocket_model()))

    @agent.tool_plain
    def step() -> str:
        return 'done'

    if stream:
        async with agent.run_stream('hello') as result:
            assert await result.get_output() == 'ready'
    else:
        assert (await agent.run('hello')).output == 'ready'

    assert sockets.opened == [first_socket, second_socket]
    assert [len(socket.sent) for socket in sockets.opened] == [2, 2]
    assert [socket.close_count for socket in sockets.opened] == [1, 1]


async def test_fallback_on_connection_errors_takes_over_from_a_failed_socket(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    """A transport failure mid-request raises `ModelConnectionError`, as over HTTP, so a fallback takes over.

    The next run opens a new socket instead of reusing the failed one.
    """
    socket = sockets.pending[0]
    socket.responses.clear()
    socket.push(ConnectionClosedError(None, None))
    next_socket = ScriptedSocket()
    sockets.pending.append(next_socket)
    model = websocket_model(http.provider())
    agent = Agent(FallbackModel(model, OpenAIResponsesModel('gpt-4o', provider=http.provider())))
    async with agent:
        assert (await agent.run('hello')).output == 'from HTTP'
        assert (await agent.run('again')).output == 'ready'

    assert len(socket.sent) == 1
    assert len(next_socket.sent) == 1
    assert len(http.requests) == 1


@pytest.mark.parametrize('retry_closed', [False, True])
async def test_socket_closed_while_idle_is_replaced_before_the_send(
    allow_model_requests: None, sockets: SocketHarness, retry_closed: bool
):
    """A socket the server closed while idle sent nothing, so the request is rebuilt for a new socket once.

    The new socket can't continue from the closed one's cache, so the history is replayed.
    """
    socket = sockets.pending[0]
    next_socket = ScriptedSocket()
    sockets.pending.append(next_socket)
    agent = Agent(websocket_model())
    async with agent:
        first = await agent.run('first')
        socket.send_error = ConnectionClosedOK(None, None)
        if retry_closed:
            next_socket.send_error = ConnectionClosedOK(None, None)
            with pytest.raises(ModelConnectionError, match='closed before the request was sent') as raised:
                await agent.run('second', message_history=first.all_messages())
            assert raised.value.phase == 'connect'
        else:
            assert (await agent.run('second', message_history=first.all_messages())).output == 'ready'

    assert len(socket.sent) == 1
    assert socket.close_count == 1
    assert sockets.opened == [socket, next_socket]
    if not retry_closed:
        assert 'previous_response_id' not in next_socket.sent[0]
        assert next_socket.sent[0]['input'] == [
            {'role': 'user', 'content': 'first'},
            {'role': 'assistant', 'content': 'ready'},
            {'role': 'user', 'content': 'second'},
        ]


async def test_socket_near_the_time_limit_is_replaced_once_its_responses_finish(
    allow_model_requests: None, sockets: SocketHarness
):
    """Near OpenAI's 60-minute limit, new requests open a new socket, and the old one closes once it drains."""
    socket = sockets.pending[0]
    socket.responses.clear()
    next_socket = ScriptedSocket()
    sockets.pending.append(next_socket)
    model = websocket_model()
    results: list[str] = []

    async def run(prompt: str) -> None:
        results.append((await Agent(model).run(prompt)).output)

    async with model, anyio.create_task_group() as tasks:
        tasks.start_soon(run, 'old')
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            await socket.receiving.wait()
        # Waiting out the limit isn't practical, so the socket is dated as if it had opened that long ago.
        holder_socket = model._websocket.states[asyncio.get_running_loop()].socket  # pyright: ignore[reportPrivateUsage]
        assert holder_socket is not None
        holder_socket.opened_at -= 60 * 60
        await run('new')
        assert socket.close_count == 0
        socket.push(*text_events('old'))

    assert results == ['ready', 'old']
    assert [len(socket.sent), len(next_socket.sent)] == [1, 1]
    assert [socket.close_count, next_socket.close_count] == [1, 1]


@dataclass(frozen=True)
class ErrorFrameCase:
    id: str
    status: int | None
    code: str
    error_type: str = 'invalid_request_error'
    expected: type[ModelAPIError] = ModelHTTPError
    model_name: str = 'gpt-4o'
    suggested_model_id: str | None = None


ERROR_FRAME_CASES = [
    ErrorFrameCase('envelope', status=None, code='invalid_test', expected=ModelAPIError),
    ErrorFrameCase('status', status=400, code='invalid_test'),
    ErrorFrameCase(
        'model_not_found',
        status=400,
        code='model_not_found',
        model_name='gpt-5.2-proo',
        suggested_model_id='openai:gpt-5.2-pro',
    ),
    ErrorFrameCase(
        'rate_limit', status=429, code='rate_limit_exceeded', error_type='requests', expected=ModelRateLimitError
    ),
    ErrorFrameCase(
        'context_window', status=400, code='context_length_exceeded', expected=ModelContextWindowExceededError
    ),
    ErrorFrameCase('in_stream', status=None, code='server_error', error_type='server_error', expected=ModelServerError),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in ERROR_FRAME_CASES])
@pytest.mark.parametrize('stream', [False, True])
async def test_error_frames(allow_model_requests: None, sockets: SocketHarness, case: ErrorFrameCase, stream: bool):
    """An error frame raises what the same error raises over HTTP, ends only its request, and never falls back to HTTP.

    One with a status was a rejection before the stream opened; one without it was an error in the stream.
    """
    socket = sockets.pending[0]
    error: Frame = {'type': case.error_type, 'code': case.code, 'message': 'test failure'}
    frame: Frame = {'type': 'error', 'error': error}
    if case.status is not None:
        frame['status'] = case.status
    socket.responses = deque([[frame], text_events(response_id='resp_recovered')])
    http_handler = Mock(return_value=httpx2.Response(401))
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        model = OpenAIResponsesModel(
            case.model_name,
            provider=OpenAIProvider(api_key='test', http_client=http_client),
            settings={**WEBSOCKET, 'max_retries': 0},
        )
        agent = Agent(model)
        async with agent:
            with pytest.raises(case.expected) as raised:
                if stream:
                    async with agent.run_stream('hello'):
                        pytest.fail('The rejected request should not produce a stream')  # pragma: no cover
                else:
                    await agent.run('hello')
            assert (await agent.run('recover')).output == 'ready'
            assert socket.close_count == 0

    assert raised.value.provider_error_code == case.code
    assert raised.value.in_stream is (case.status is None)
    if case.status is None:
        assert 'test failure' in str(raised.value)
    else:
        assert isinstance(raised.value, ModelHTTPError)
        assert raised.value.status_code == case.status
        assert raised.value.body == error
        assert raised.value.suggested_model_id == case.suggested_model_id
    http_handler.assert_not_called()


@pytest.mark.parametrize('failure', ['failed', 'steering', 'disconnect', 'send', 'timeout'])
@pytest.mark.parametrize('stream', [False, True])
async def test_errors(allow_model_requests: None, sockets: SocketHarness, failure: str, stream: bool):
    """A failure on the socket never falls back to HTTP or returns partial output; a transport failure ends the socket."""
    socket = sockets.pending[0]
    settings: OpenAIResponsesModelSettings = {}
    first_response: list[Frame] = []
    if failure == 'failed':
        response = completed_response('resp_test')
        response.update(status='failed', error={'code': 'server_error', 'message': 'test failure'})
        first_response = [{'type': 'response.failed', 'sequence_number': 0, 'response': response}]
    elif failure == 'steering':
        first_response = [
            {
                'type': 'response.steer.accepted',
                'sequence_number': 0,
                'steer': {'id': 'steer_test', 'previous_response_id': 'resp_test'},
            }
        ]
    elif failure == 'disconnect':
        socket.push(*text_events()[:-1], ConnectionClosedError(None, None))
    elif failure == 'send':
        socket.send_error = OSError('test failure')
    else:
        settings['timeout'] = 0.01
    transport_failure = failure in ('disconnect', 'send')
    socket.responses = deque([first_response, text_events(response_id='resp_recovered')])
    next_socket = ScriptedSocket()
    sockets.pending.append(next_socket)
    http_handler = Mock(return_value=httpx2.Response(401))
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        agent = Agent(websocket_model(OpenAIProvider(api_key='test', http_client=http_client)))
        async with agent:
            with pytest.raises(UnexpectedModelBehavior if failure == 'steering' else ModelAPIError) as raised:
                if stream:
                    async with agent.run_stream('hello', model_settings=settings) as result:
                        await result.get_output()
                else:
                    await agent.run('hello', model_settings=settings)
            assert socket.close_count == (1 if transport_failure else 0)
            if failure == 'failed':
                assert 'server_error' in str(raised.value)
            elif failure != 'steering':
                # The classes and phases the same failure raises over HTTP.
                assert type(raised.value) is (ModelTimeoutError if failure == 'timeout' else ModelConnectionError)
                assert isinstance(raised.value, ModelConnectionError)
                assert raised.value.phase == ('write' if failure == 'send' else 'read')
            assert (await agent.run('recover')).output == 'ready'

    # A failed socket is replaced; after a request-scoped failure, the next request uses the same one.
    assert sockets.opened == ([socket, next_socket] if transport_failure else [socket])
    http_handler.assert_not_called()


@pytest.mark.parametrize('stream', [False, True])
async def test_failed_response_without_error(allow_model_requests: None, sockets: SocketHarness, stream: bool):
    response = completed_response('resp_test')
    response.update(status='failed', error=None)
    sockets.pending[0].responses = deque([[{'type': 'response.failed', 'sequence_number': 0, 'response': response}]])
    model = websocket_model()
    messages = user_prompt('hello')
    async with model:
        if stream:
            async with model.request_stream(messages, None, ModelRequestParameters()) as streamed:
                async for _ in streamed:
                    pass
                result = streamed.get()
        else:
            result = await model.request(messages, None, ModelRequestParameters())

        assert result.parts == []
        assert result.finish_reason == 'error'
        assert (result.provider_details or {}).get('finish_reason') == 'failed'
        assert sockets.opened[0].close_count == 0
        sockets.opened[0].responses = deque([text_events(response_id='resp_retry')])
        retried = await model.request([*messages, result], None, ModelRequestParameters())
        assert retried.text == 'ready'
    # A failed response cannot be continued from, so the retry sends the whole history.
    assert 'previous_response_id' not in sockets.opened[0].sent[1]


async def test_incomplete_response(allow_model_requests: None, sockets: SocketHarness):
    events = text_events()
    terminal = events[-1]
    terminal['type'] = 'response.incomplete'
    response = terminal['response']
    assert isinstance(response, dict)
    response.update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
    sockets.pending[0].responses = deque([events, text_events(response_id='resp_again')])
    model = websocket_model()
    async with model:
        result = await model.request(user_prompt('hello'), None, ModelRequestParameters())
        assert result.text == 'ready'
        assert result.finish_reason == 'length'
        assert (result.provider_details or {}).get('finish_reason') == 'max_output_tokens'
        assert sockets.opened[0].close_count == 0
        assert (await model.request(user_prompt('again'), None, ModelRequestParameters())).text == 'ready'


@pytest.mark.parametrize(
    ('options', 'timeout_source', 'connect_timeout'),
    [
        pytest.param({}, 'model', 20, id='defaults-model-timeout'),
        pytest.param({}, 'client', None, id='defaults-unlimited-client-timeout'),
        pytest.param({'max_size': 1024, 'max_queue': 4}, 'model', None, id='caller'),
    ],
)
async def test_message_limits_default_to_unlimited(
    allow_model_requests: None,
    sockets: SocketHarness,
    options: WebSocketConnectionOptions,
    timeout_source: str,
    connect_timeout: float | None,
):
    """The `websockets` message size and unread-frame limits are lifted unless the caller sets them.

    The handshake deadline is the request's connect timeout, so long and unlimited ones aren't capped by the
    transport's ten-second default.
    """
    client = AsyncOpenAI(
        api_key='test', timeout=httpx2.Timeout(30, connect=connect_timeout) if timeout_source == 'client' else 30
    )
    settings: OpenAIResponsesModelSettings = (
        {'timeout': Timeout(30, connect=connect_timeout)} if timeout_source == 'model' else {}
    )
    model = OpenAIResponsesModel(
        'gpt-4o',
        provider=OpenAIProvider(openai_client=client),
        settings={**WEBSOCKET, **settings},
        websocket_connection_options=options,
    )
    async with client:
        assert (await Agent(model).run('hello')).output == 'ready'
    assert {key: sockets.options[0][key] for key in ('max_size', 'max_queue', 'open_timeout')} == {
        'max_size': None,
        'max_queue': None,
        **options,
        'open_timeout': None,
    }


@pytest.mark.parametrize('timeout_source', ['model', 'client'])
async def test_handshake_timeout(allow_model_requests: None, sockets: SocketHarness, timeout_source: str):
    """A stalled handshake honors the request's connect timeout without leaking a connection."""
    before = asyncio.all_tasks()
    sockets.connect_gate = anyio.Event()
    settings: OpenAIResponsesModelSettings = {'timeout': Timeout(10, connect=0.01)} if timeout_source == 'model' else {}
    client = AsyncOpenAI(api_key='test', timeout=httpx2.Timeout(10, connect=0.01) if timeout_source == 'client' else 30)
    model = OpenAIResponsesModel(
        'gpt-4o', provider=OpenAIProvider(openai_client=client), settings={**WEBSOCKET, **settings}
    )
    async with client, model:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            with pytest.raises(ModelTimeoutError, match='WebSocket connection timed out') as raised:
                await model.request(user_prompt('hello'), None, ModelRequestParameters())
        assert raised.value.phase == 'connect'
        assert isinstance(raised.value.__cause__, TimeoutError)
        assert sockets.connect_count == 1
        assert sockets.opened == []
        assert not model.client.is_closed()
    assert asyncio.all_tasks() == before


async def test_send_timeout(allow_model_requests: None, sockets: SocketHarness):
    """A stalled send honors the request's write timeout and fails the socket, so the next request opens another."""
    before = asyncio.all_tasks()
    socket = sockets.pending[0]
    socket.send_gate = anyio.Event()
    next_socket = ScriptedSocket()
    sockets.pending.append(next_socket)
    agent = Agent(websocket_model())
    async with agent:
        with anyio.fail_after(READINESS_WAIT_TIMEOUT):
            with pytest.raises(ModelTimeoutError, match='WebSocket request timed out') as raised:
                await agent.run('hello', model_settings={'timeout': Timeout(10, write=0.01)})
        assert raised.value.phase == 'write'
        assert isinstance(raised.value.__cause__, TimeoutError)
        assert socket.sent == []
        assert socket.close_count == 1
        assert (await agent.run('again')).output == 'ready'
    assert len(next_socket.sent) == 1
    assert asyncio.all_tasks() == before


@pytest.mark.parametrize('status_code', [200, 302, 401, None])
async def test_handshake_errors(allow_model_requests: None, sockets: SocketHarness, status_code: int | None):
    """A rejected handshake keeps its status, body and headers; an unreachable host is a connection error."""
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
    model = websocket_model()
    expected_error = (
        ModelConnectionError if status_code is None else ModelHTTPError if status_code >= 400 else ModelAPIError
    )
    async with model:
        with pytest.raises(expected_error) as raised:
            await model.request(user_prompt('hello'), None, ModelRequestParameters())
        assert sockets.connect_count == 1
        if isinstance(raised.value, ModelConnectionError):
            assert raised.value.phase == 'connect'
        if isinstance(raised.value, ModelHTTPError):
            assert raised.value.status_code == status_code
            assert raised.value.body == 'handshake rejected'
            assert raised.value.headers == {'x-request-id': 'first, second'}
        assert not model.client.is_closed()


@pytest.mark.parametrize(
    ('status_code', 'body', 'expected'),
    [
        pytest.param(
            429,
            b'{"error": {"message": "Slow down.", "type": "requests", "code": "rate_limit_exceeded"}}',
            ModelRateLimitError,
            id='rate_limit',
        ),
        pytest.param(
            429,
            b'{"error": {"message": "No credits.", "type": "insufficient_quota", "code": "insufficient_quota"}}',
            ModelQuotaExceededError,
            id='quota',
        ),
        pytest.param(503, b'Service unavailable', ModelUnavailableError, id='unavailable'),
        pytest.param(
            500, b'{"error": {"message": "Oops.", "type": "server_error", "code": null}}', ModelServerError, id='server'
        ),
    ],
)
async def test_handshake_rejection_raises_the_http_error_category(
    allow_model_requests: None, sockets: SocketHarness, status_code: int, body: bytes, expected: type[ModelAPIError]
):
    """A rejected handshake raises what the same response to an HTTP request raises."""
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: httpx2.Response(status_code, content=body))
    ) as http_client:
        client = AsyncOpenAI(api_key='test', http_client=http_client, max_retries=0)
        model = websocket_model(OpenAIProvider(openai_client=client))
        with pytest.raises(ModelHTTPError) as over_http:
            await model.request(user_prompt('hello'), None, ModelRequestParameters())
        sockets.connect_error = InvalidStatus(HandshakeResponse(status_code, 'Rejected', Headers(), body))
        async with model:
            with pytest.raises(ModelHTTPError) as over_websocket:
                await model.request(user_prompt('hello'), None, ModelRequestParameters())

    websocket_error, http_error = over_websocket.value, over_http.value
    assert isinstance(websocket_error, expected)
    assert (
        type(websocket_error),
        websocket_error.status_code,
        websocket_error.body,
        websocket_error.provider_error_code,
        websocket_error.provider_error_type,
    ) == (
        type(http_error),
        status_code,
        http_error.body,
        http_error.provider_error_code,
        http_error.provider_error_type,
    )


async def test_sdk_api_errors_during_websocket_connection_are_mapped(
    allow_model_requests: None, sockets: SocketHarness
):
    request = httpx2.Request('GET', 'https://api.openai.com/v1/responses')
    sockets.connect_error = APIError('SDK API failure', request, body=None)
    model = websocket_model()
    async with model:
        with pytest.raises(ModelAPIError, match='SDK API failure'):
            await model.request(user_prompt('hello'), None, ModelRequestParameters())
    assert sockets.connect_count == 1


OVERRIDABLE_NEVER_SOCKET_SETUPS = ['azure-provider', 'azure-client', 'gateway']
# OAuth credentials reach HTTP requests only, and Codex session headers vary per conversation, so `profile=` can't
# turn the socket on for these.
FIXED_NEVER_SOCKET_SETUPS = ['openai-chatgpt', 'openai-codex', 'openai-codex-model']
WEBSOCKET_PROFILE = OpenAIModelProfile(openai_responses_supports_websocket=True)


def never_socket_model(
    setup: str, client: httpx2.AsyncClient, profile: OpenAIModelProfile | None = None
) -> OpenAIResponsesModel:
    """A WebSocket-mode model whose profile turns `openai_responses_supports_websocket` off by default."""
    if setup == 'azure-provider':
        # The v1 endpoint gets a plain `AsyncOpenAI` client, so only `AzureProvider.model_profile` sets the flag.
        provider: Provider[AsyncOpenAI] = AzureProvider(
            azure_endpoint='https://example.openai.azure.com/openai/v1', api_key='test', http_client=client
        )
    elif setup == 'azure-client':
        azure_client = AsyncAzureOpenAI(
            api_key='test',
            api_version='2024-12-01-preview',
            azure_endpoint='https://example.openai.azure.com/',
            http_client=client,
        )
        provider = OpenAIProvider(openai_client=azure_client)
    elif setup == 'gateway':
        provider = gateway_provider(
            'openai', api_key='test', base_url='https://gateway.example/proxy', http_client=client
        )
    elif setup == 'openai-chatgpt':
        credentials = OpenAIChatGPTCredentials(
            client_id='client', access_token='access', refresh_token='refresh', scope='chatgpt.tokens.use.direct'
        )
        provider = OpenAIChatGPTProvider(credentials, http_client=client)
    elif setup == 'openai-codex':
        codex_credentials = OpenAICodexCredentials(access_token='access', refresh_token='refresh', account_id='a')
        provider = OpenAICodexProvider(codex_credentials, http_client=client)
    else:
        assert setup == 'openai-codex-model'
        return OpenAICodexModel(
            'gpt-4o', provider=OpenAIProvider(api_key='test', http_client=client), settings=WEBSOCKET, profile=profile
        )
    return websocket_model(provider, profile=profile)


@pytest.mark.parametrize(
    ('setup', 'profile'),
    [
        *(pytest.param(setup, None, id=setup) for setup in OVERRIDABLE_NEVER_SOCKET_SETUPS + FIXED_NEVER_SOCKET_SETUPS),
        *(pytest.param(setup, WEBSOCKET_PROFILE, id=f'{setup}-profile-on') for setup in FIXED_NEVER_SOCKET_SETUPS),
    ],
)
async def test_setup_that_can_never_use_a_socket_warns_and_uses_http(
    allow_model_requests: None, sockets: SocketHarness, setup: str, profile: OpenAIModelProfile | None
):
    """A model whose profile turns the socket off says so, and sends the request over HTTP, here rejected with a 400.

    For the OAuth providers and `OpenAICodexModel`, a `profile=` that turns the socket on doesn't change that.
    """
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(400))) as client:
        model = never_socket_model(setup, client, profile)
        async with model:
            with (
                pytest.warns(
                    UnsupportedSettingWarning,
                    match=(
                        r"`openai_transport='websocket'` setting .* `openai_responses_supports_websocket=False`.* "
                        r'`profile=OpenAIModelProfile\(openai_responses_supports_websocket=True\)`'
                    ),
                ),
                pytest.raises(ModelHTTPError),
            ):
                await model.request(user_prompt('hello'), None, ModelRequestParameters())
    assert sockets.connect_count == 0


@pytest.mark.parametrize('setup', OVERRIDABLE_NEVER_SOCKET_SETUPS)
async def test_profile_flag_overrides_a_setup_that_can_never_use_a_socket(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, setup: str
):
    """`openai_responses_supports_websocket=True` in `profile=` wins over the Azure and gateway defaults."""
    model = never_socket_model(setup, http.client, WEBSOCKET_PROFILE)
    assert (await Agent(model).run('hello')).output == 'ready'
    assert sockets.connect_count == 1
    assert http.requests == []


async def test_missing_websocket_dependency_warns_and_uses_http(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setitem(sys.modules, 'pydantic_ai.models._openai_responses_websocket', None)
    agent = Agent(websocket_model(http.provider()))
    with pytest.warns(UnsupportedSettingWarning, match=r'install `pydantic-ai-slim\[openai-responses-websocket\]`'):
        assert (await agent.run('hello')).output == 'from HTTP'
    assert sockets.connect_count == 0


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_other_async_backends_warn_and_use_http(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    """Only the asyncio event loop runs the socket, so a request under Trio uses HTTP, even inside `async with model:`."""
    model = websocket_model(http.provider())
    async with model:
        with pytest.warns(UnsupportedSettingWarning, match='Responses WebSockets need the asyncio event loop'):
            response = await model.request(user_prompt('hello'), None, ModelRequestParameters())
    assert response.text == 'from HTTP'
    assert sockets.connect_count == 0


@pytest.mark.parametrize(
    'auth_mode',
    [
        pytest.param(
            'sigv4',
            marks=pytest.mark.skipif(find_spec('botocore') is None, reason='SigV4 signing needs `openai[bedrock]`'),
        ),
        'native-bearer',
    ],
)
@pytest.mark.skipif(not bedrock_mantle_imports_successful(), reason='bedrock-mantle not installed')
async def test_bedrock_mantle_native_auth_warns_and_uses_http(
    allow_model_requests: None,
    monkeypatch: pytest.MonkeyPatch,
    sockets: SocketHarness,
    auth_mode: Literal['sigv4', 'native-bearer'],
):
    """The SDK can't authenticate a socket with Mantle's native auth, which it finds before connecting."""
    monkeypatch.delenv('AWS_BEARER_TOKEN_BEDROCK', raising=False)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda request: httpx2.Response(400))) as client:
        if auth_mode == 'sigv4':
            provider = BedrockMantleProvider(
                region_name='us-east-1',
                aws_access_key_id='fake-access-key',
                aws_secret_access_key='fake-secret-key',
                http_client=client,
            )
        else:
            provider = BedrockMantleProvider(region_name='us-east-1', api_key='fake-bearer-token', http_client=client)
        model = BedrockMantleResponsesModel('openai.gpt-5.6-luna', provider=provider, settings=WEBSOCKET)
        async with model:
            with (
                pytest.warns(UnsupportedSettingWarning, match='the OpenAI SDK could not configure the WebSocket'),
                pytest.raises(ModelHTTPError),
            ):
                await model.request(user_prompt('hello'), None, ModelRequestParameters())
    assert sockets.connect_count == 0


@pytest.mark.skipif(not bedrock_mantle_imports_successful(), reason='bedrock-mantle not installed')
async def test_bedrock_mantle_websocket_accepts_compatible_bearer_client(
    allow_model_requests: None, sockets: SocketHarness
):
    """A caller-supplied bearer client preserves the Mantle endpoint and closes its socket once."""
    client = AsyncOpenAI(api_key='fake-bearer-token', base_url='https://bedrock-mantle.us-east-1.api.aws/openai/v1')
    model = BedrockMantleResponsesModel(
        'openai.gpt-5.6-luna', provider=BedrockMantleProvider(openai_client=client), settings=WEBSOCKET
    )
    socket = sockets.pending[0]
    async with client:
        assert (await Agent(model).run('hello')).output == 'ready'
    assert sockets.urls == ['wss://bedrock-mantle.us-east-1.api.aws/openai/v1/responses']
    assert sockets.authorization_headers == ['Bearer fake-bearer-token']
    assert socket.close_count == 1


@pytest.mark.parametrize('request_kind', ['background', 'background_body', 'resume'])
async def test_background_and_resumed_requests_use_http(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses, request_kind: str
):
    """The socket can't run a background response or retrieve a suspended one, so those requests use HTTP."""
    settings: OpenAIResponsesModelSettings = (
        {'openai_background': True}
        if request_kind == 'background'
        else {'extra_body': {'background': True}}
        if request_kind == 'background_body'
        else {}
    )
    messages: list[ModelRequest | ModelResponse] = (
        [
            ModelRequest.user_text_prompt('hello'),
            ModelResponse(
                parts=[TextPart('partial')],
                provider_name='openai',
                provider_response_id='resp_suspended',
                state='suspended',
            ),
        ]
        if request_kind == 'resume'
        else user_prompt('hello')
    )
    model = websocket_model(http.provider())
    async with model:
        response = await model.request(messages, settings, ModelRequestParameters())
    assert response.text == 'from HTTP'
    assert sockets.connect_count == 0
    assert [request.method for request in http.requests] == (['GET'] if request_kind == 'resume' else ['POST'])


@pytest.mark.parametrize('extra_body', [['invalid'], {'stream': True}, {'stream_id': 'x'}, {'type': 'other'}])
async def test_socket_extra_body_cannot_change_the_frame(
    allow_model_requests: None, sockets: SocketHarness, extra_body: object
):
    # Exercise malformed runtime settings outside the declared type.
    settings: OpenAIResponsesModelSettings = {'extra_body': extra_body}
    model = websocket_model()
    async with model:
        with pytest.raises(UserError, match='Responses WebSocket `extra_body`'):
            await model.request(user_prompt('hello'), settings, ModelRequestParameters())
    assert sockets.opened[0].sent == []


@pytest.mark.parametrize(
    ('request_headers', 'over_socket'),
    [
        pytest.param({}, False, id='omitted-all'),
        pytest.param({'x-tenant': 'tenant-test'}, False, id='omitted-organization'),
        pytest.param({'openai-organization': 'other-org', 'x-tenant': 'other-tenant'}, False, id='changed-values'),
        pytest.param({'OPENAI-ORGANIZATION': 'org-test', 'x-TENANT': 'tenant-test'}, True, id='case-insensitive'),
        pytest.param(
            {'openai-organization': 'org-test', 'x-tenant': 'tenant-test', 'x-request-only': b'byte-header-value'},
            False,
            id='bytes-value',
        ),
        pytest.param(
            {'openai-organization': 'org-test', 'x-tenant': 'tenant-test', 'Authorization': 'Bearer test'},
            False,
            id='restated-client-credentials',
        ),
    ],
)
async def test_request_with_other_headers_than_the_handshake_uses_http(
    allow_model_requests: None,
    sockets: SocketHarness,
    http: HTTPResponses,
    request_headers: dict[str, str | bytes],
    over_socket: bool,
):
    """A request whose headers differ from the ones the socket opened with goes over HTTP, which sends them."""
    model = websocket_model(
        http.provider(), settings={'extra_headers': {'OpenAI-Organization': 'org-test', 'X-Tenant': 'tenant-test'}}
    )
    # Exercise malformed runtime settings outside the declared header type.
    request_settings: OpenAIResponsesModelSettings = {'extra_headers': request_headers}  # pyright: ignore[reportAssignmentType]
    agent = Agent(model)
    async with agent:
        assert (await agent.run('handshake')).output == 'ready'
        assert sockets.headers[0]['openai-organization'] == 'org-test'
        assert sockets.headers[0]['x-tenant'] == 'tenant-test'
        sockets.opened[0].responses.append(text_events())
        output = (await agent.run('hello', model_settings=request_settings)).output

    assert output == ('ready' if over_socket else 'from HTTP')
    assert len(sockets.opened[0].sent) == (2 if over_socket else 1)
    assert len(http.requests) == (0 if over_socket else 1)


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
    model = websocket_model(OpenAIProvider(openai_client=client), settings={'extra_headers': request_headers})
    assert (await Agent(model).run('hello')).output == 'ready'
    assert sockets.headers[0]['x-tenant'] == 'tenant-test'
    assert len(sockets.opened[0].sent) == 1


async def test_settings_authorization_header_authenticates_the_handshake(
    allow_model_requests: None, sockets: SocketHarness
):
    """An `Authorization` header in the settings authenticates the handshake instead of the client's key."""
    model = websocket_model(
        OpenAIProvider(api_key='client-key'), settings={'extra_headers': {'Authorization': 'Bearer settings-key'}}
    )
    assert (await Agent(model).run('hello')).output == 'ready'
    assert sockets.authorization_headers == ['Bearer settings-key']


async def test_max_retries_is_supported_over_the_socket(allow_model_requests: None, sockets: SocketHarness):
    model = websocket_model(settings={'max_retries': 0})
    assert (await Agent(model).run('hello')).output == 'ready'


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('store', [False, True])
async def test_websocket_mode_http_echo_controls_automatic_continuation(
    allow_model_requests: None, http: HTTPResponses, stream: bool, store: bool
):
    """HTTP fallback records storage and only omits retained input when that response is reachable."""
    model = websocket_model(http.provider())
    first_request = ModelRequest.user_text_prompt('first')
    if stream:
        events = text_events('answer', 'resp_seed', store=store)
        body = ''.join(f'event: {e["type"]}\ndata: {FRAME_ADAPTER.dump_json(e).decode()}\n\n' for e in events)
        http.replies.append(httpx2.Response(200, content=body.encode(), headers={'content-type': 'text/event-stream'}))
        async with model.request_stream([first_request], None, ModelRequestParameters()) as result:
            async for _ in result:
                pass
        first = result.get()
    else:
        response = completed_response('resp_seed', store=store)
        http.replies.append(httpx2.Response(200, json=response))
        first = await model.request([first_request], None, ModelRequestParameters())
    assert (first.provider_details or {}).get('store') is store
    await model.request([first_request, first, ModelRequest.user_text_prompt('second')], None, ModelRequestParameters())
    assert http.bodies[1].get('previous_response_id') == ('resp_seed' if store else None)
    assert ('first' in str(http.bodies[1]['input'])) is not store


async def test_socket_rejection_rebuilds_cached_history(
    allow_model_requests: None, sockets: SocketHarness, monkeypatch: pytest.MonkeyPatch
):
    socket = sockets.pending[0]
    error: Frame = {'type': 'error', 'status': 503, 'error': {'type': 'server_error', 'message': 'rejected'}}
    socket.responses = deque([text_events('answer', 'resp_seed', store=False), [error], text_events()])

    async def sleep(delay: float) -> None:
        pass

    monkeypatch.setattr(anyio, 'sleep', sleep)
    agent = Agent(websocket_model(settings={'openai_store': False, 'max_retries': 1}))
    async with agent:
        first = await agent.run('first')
        await agent.run('second', message_history=first.all_messages())
    assert socket.sent[1]['previous_response_id'] == 'resp_seed'
    assert 'previous_response_id' not in socket.sent[2]
    assert socket.sent[2]['input'] == [
        {'role': 'user', 'content': 'first'},
        {'role': 'assistant', 'content': 'answer'},
        {'role': 'user', 'content': 'second'},
    ]


async def test_socket_error_after_response_starts_is_not_retried(allow_model_requests: None, sockets: SocketHarness):
    socket = sockets.pending[0]
    error: Frame = {'type': 'error', 'status': 503, 'error': {'type': 'server_error', 'message': 'failed'}}
    socket.responses = deque([[text_events()[0], error]])
    with pytest.raises(ModelHTTPError):
        await Agent(websocket_model()).run('hello')
    assert len(socket.sent) == 1


async def test_model_exit_uses_the_captured_lease_kind(allow_model_requests: None, sockets: SocketHarness):
    from pydantic_ai.durable_exec._operation_backend import durable_unit_scope

    model = websocket_model()
    async with model:
        state = model._websocket.states[asyncio.get_running_loop()]  # pyright: ignore[reportPrivateUsage]
        assert state.external_leases == 1
        # Enter inside a durable unit and exit after its ambient flag has changed.
        async with AsyncExitStack() as nested:
            with durable_unit_scope():
                await nested.enter_async_context(model)
        assert state.leases == state.external_leases == 1
    assert model._websocket.states == {}  # pyright: ignore[reportPrivateUsage]


async def test_overlapping_model_contexts_release_their_own_lease(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    """An external owner exiting first cannot leave its ownership attached to another task's durable scope."""
    from pydantic_ai.durable_exec._operation_backend import durable_unit_scope

    model = websocket_model(http.provider())
    external_entered = anyio.Event()
    durable_entered = anyio.Event()
    external_exited = anyio.Event()

    async def external_owner() -> None:
        async with model:
            await model.request(user_prompt('first'), None, ModelRequestParameters())
            external_entered.set()
            await durable_entered.wait()
        external_exited.set()

    async def durable_owner() -> None:
        await external_entered.wait()
        with durable_unit_scope():
            async with model:
                durable_entered.set()
                await external_exited.wait()
                state = model._websocket.states[asyncio.get_running_loop()]  # pyright: ignore[reportPrivateUsage]
                assert state.external_leases == 0
                assert (await model.request(user_prompt('durable'), None, ModelRequestParameters())).text == 'from HTTP'

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(external_owner)
            tasks.start_soon(durable_owner)
    assert len(sockets.opened[0].sent) == 1
    assert sockets.opened[0].close_count == 1
    assert len(http.requests) == 1
    assert model._websocket.states == {}  # pyright: ignore[reportPrivateUsage]


async def test_overlapping_agents_keep_fallback_child_contexts(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    model = websocket_model(http.provider())
    sockets.pending[0].responses.append(text_events('second', 'resp_second'))
    fallback = FallbackModel(model)
    first_entered = anyio.Event()
    second_entered = anyio.Event()
    first_exited = anyio.Event()

    async def first_owner() -> None:
        async with Agent(fallback) as agent:
            await agent.run('first')
            first_entered.set()
            await second_entered.wait()
        first_exited.set()

    async def second_owner() -> None:
        await first_entered.wait()
        async with Agent(fallback) as agent:
            second_entered.set()
            await first_exited.wait()
            assert sockets.opened[0].close_count == 0
            await agent.run('second')

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(first_owner)
            tasks.start_soon(second_owner)
    assert len(sockets.opened[0].sent) == 2
    assert sockets.opened[0].close_count == 1
    assert model._websocket.states == {}  # pyright: ignore[reportPrivateUsage]


async def test_task_spawned_during_wrapper_entry_has_its_own_model_context(
    allow_model_requests: None, sockets: SocketHarness, http: HTTPResponses
):
    from pydantic_ai.durable_exec._operation_backend import durable_unit_scope

    model = websocket_model(http.provider())
    entered = anyio.Event()
    exited = anyio.Event()

    async def child() -> None:
        with durable_unit_scope():
            async with model:
                entered.set()
                await exited.wait()
                state = model._websocket.states[asyncio.get_running_loop()]  # pyright: ignore[reportPrivateUsage]
                assert state.external_leases == 0
                assert (await model.request(user_prompt('durable'), None, ModelRequestParameters())).text == 'from HTTP'

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with anyio.create_task_group() as tasks:

            class SpawnWrapper(WrapperModel):
                async def __aenter__(self) -> SpawnWrapper:
                    await super().__aenter__()
                    tasks.start_soon(child)
                    await entered.wait()
                    return self

            async with Agent(SpawnWrapper(model)) as agent:
                await agent.run('first')
            exited.set()
    assert len(sockets.opened[0].sent) == 1
    assert sockets.opened[0].close_count == 1
    assert len(http.requests) == 1
    assert model._websocket.states == {}  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('status', [408, 409, 429, 500, 503])
@pytest.mark.parametrize('max_retries', [0, 1, None])
@pytest.mark.parametrize('stream', [False, True])
async def test_early_rejection_retry_budget(
    allow_model_requests: None,
    sockets: SocketHarness,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    max_retries: int | None,
    stream: bool,
):
    socket = sockets.pending[0]
    error: Frame = {
        'type': 'error',
        'status': status,
        'error': {
            'type': 'server_error',
            'code': 'retry_test',
            'message': 'rejected',
            'headers': {'retry-after-ms': '250'},
        },
    }
    socket.responses = deque([[error], [error], text_events()])
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(anyio, 'sleep', sleep)
    settings: OpenAIResponsesModelSettings = {} if max_retries is None else {'max_retries': max_retries}
    agent = Agent(websocket_model(settings=settings))

    async def run() -> str:
        if stream:
            async with agent.run_stream('hello') as result:
                return await result.get_output()
        return (await agent.run('hello')).output

    if max_retries is None:
        assert await run() == 'ready'
    else:
        with pytest.raises(ModelHTTPError):
            await run()
    expected_retries = 2 if max_retries is None else max_retries
    assert len(socket.sent) == expected_retries + 1
    assert delays == [0.25] * expected_retries
    assert socket.close_count == 1


async def test_request_settings(allow_model_requests: None, sockets: SocketHarness):
    settings: OpenAIResponsesModelSettings = {
        'openai_transport': 'websocket',
        'openai_service_tier': 'priority',
        'service_tier': 'flex',
        'openai_store': False,
        'temperature': 0.2,
        'extra_headers': {'x-test': 'default', 'User-Agent': 'custom-client'},
        'extra_body': {'temperature': 0.3, 'metadata': {'test': 'websocket'}},
    }
    model = OpenAIResponsesModel(
        'gpt-4o',
        provider=OpenAIProvider(api_key='test'),
        settings=settings,
        websocket_connection_options={'max_size': 128 * 1024},
    )
    assert (await Agent(model).run('hello')).output == 'ready'
    assert sockets.headers[0]['x-test'] == 'default'
    assert sockets.headers[0]['user-agent'] == 'custom-client'
    assert sockets.options[0]['max_size'] == 128 * 1024
    assert sockets.urls == ['wss://api.openai.com/v1/responses']
    assert model.settings == settings
    sent = sockets.opened[0].sent[0]
    assert sent['service_tier'] == 'priority'
    assert sent['store'] is False
    assert sent['temperature'] == 0.3
    assert sent['metadata'] == {'test': 'websocket'}


async def test_refreshed_client_credentials_keep_the_open_socket_usable(
    allow_model_requests: None, sockets: SocketHarness
):
    """A shared client's credential refresh is not a header change, so requests keep using the open socket."""
    token = 'old-token'

    async def api_key() -> str:
        return token

    socket = sockets.pending[0]
    socket.responses.append(text_events())
    agent = Agent(websocket_model(OpenAIProvider(openai_client=AsyncOpenAI(api_key=api_key))))
    async with agent:
        assert (await agent.run('hello')).output == 'ready'
        token = 'refreshed-token'
        assert (await agent.run('again')).output == 'ready'
    assert sockets.authorization_headers == ['Bearer old-token']
    assert len(socket.sent) == 2

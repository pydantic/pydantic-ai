from __future__ import annotations

import asyncio
import sys
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import anyio
import httpx2
import pytest
from _pytest.fixtures import SubRequest
from httpx import Timeout
from pydantic import BaseModel, JsonValue, TypeAdapter
from typing_extensions import Unpack

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.agent.wrapper import WrapperAgent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import
from ..realtime.ws_cassettes import (
    CassetteMessage,
    RealtimeCassette,
    RecordingWebSocket,
    ReplayWebSocket,
    realtime_cassette_plan,
)

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types.websocket_connection_options import WebSocketConnectionOptions
    from websockets.asyncio.client import connect as websocket_connect
    from websockets.datastructures import Headers
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidStatus
    from websockets.http11 import Response as HandshakeResponse

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai / websockets not installed')
READINESS_WAIT_TIMEOUT = 10
Frame = dict[str, JsonValue]
FRAME_ADAPTER = TypeAdapter(Frame)


@pytest.fixture(autouse=True)
def stable_platform_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid the SDK's macOS platform probe spawning a subprocess inside async tests."""
    monkeypatch.setattr('openai._base_client.get_platform', lambda: 'Unknown')


@dataclass
class RecordedResponses:
    model: OpenAIResponsesModel
    cassette: RealtimeCassette
    connections: list[str] = field(default_factory=list[str])


@pytest.fixture
def recorded_responses(
    request: SubRequest, monkeypatch: pytest.MonkeyPatch, openai_api_key: str
) -> Iterator[RecordedResponses]:
    path = Path(__file__).parent / 'cassettes' / Path(__file__).stem / f'{request.node.name}.yaml'
    mode = request.config.getoption('record_mode')
    assert mode is None or isinstance(mode, str)
    plan = realtime_cassette_plan(cassette_exists=path.exists(), record_mode=mode)
    if plan == 'error_missing':  # pragma: no cover
        raise RuntimeError(f'Record the Responses WebSocket cassette with `--record-mode=rewrite`: {path}')
    cassette = RealtimeCassette.load(path) if plan == 'replay' else RealtimeCassette()
    recorded = RecordedResponses(
        OpenAIResponsesModel('gpt-6-astra', provider=OpenAIProvider(api_key=openai_api_key)), cassette
    )

    async def connect(
        uri: str,
        *,
        additional_headers: Mapping[str, str],
        user_agent_header: str | None,
        **options: Unpack[WebSocketConnectionOptions],
    ) -> ReplayWebSocket | RecordingWebSocket:
        recorded.connections.append(uri)
        assert 'authorization' in {key.lower() for key in additional_headers}
        if plan == 'replay':
            return ReplayWebSocket(cassette)
        # Only runs while recording.
        ws = await websocket_connect(  # pragma: no cover
            uri, additional_headers=additional_headers, user_agent_header=user_agent_header, **options
        )
        return RecordingWebSocket(ws, cassette)  # pragma: no cover

    monkeypatch.setattr('openai.lib._websocket._WebSocketConnect', connect)
    try:
        yield recorded
    finally:
        if plan == 'record' and cassette.interactions:  # pragma: no cover
            cassette.dump(path)


class Weather(BaseModel):
    temperature: int


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
        'openai_responses_service_tier': 'ultrafast',
        'openai_reasoning_effort': 'low',
    }
    agent = Agent(
        recorded_responses.model, tools=[get_weather], output_type=NativeOutput(Weather), model_settings=settings
    )
    async with agent.connect():
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
    assert 'previous_response_id' not in sent[0]
    assert sent[1]['previous_response_id'] == responses[0].provider_response_id
    assert sent[1]['input'] == [{'type': 'function_call_output', 'call_id': tool_call.tool_call_id, 'output': '18'}]
    assert all(frame['service_tier'] == 'ultrafast' and frame['store'] is False for frame in sent)
    assert all(
        frame['type'] == 'response.create' and 'stream' not in frame and 'background' not in frame for frame in sent
    )
    assert sent[0]['text']['format']['type'] == 'json_schema'


async def test_streaming_sequential_turns(allow_model_requests: None, recorded_responses: RecordedResponses):
    """Ordinary full-history streaming works across sequential turns on the same socket."""
    agent = Agent(recorded_responses.model)
    async with agent.connect():
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
    assert all('previous_response_id' not in frame for frame in sent)
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
        self.sent.append(FRAME_ADAPTER.validate_json(data))
        if self.responses:
            self.push(*self.responses.popleft())

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
    options: list[dict[str, object]] = field(default_factory=list[dict[str, object]])
    connecting: anyio.Event = field(default_factory=anyio.Event)
    connect_gate: anyio.Event | None = None
    connect_error: Exception | None = None


@pytest.fixture
def sockets(monkeypatch: pytest.MonkeyPatch) -> SocketHarness:
    harness = SocketHarness()

    async def connect(uri: str, *, additional_headers: Mapping[str, str], **options: object) -> ScriptedSocket:
        harness.connecting.set()
        if harness.connect_error is not None:
            raise harness.connect_error
        if harness.connect_gate is not None:
            await harness.connect_gate.wait()
        harness.urls.append(uri)
        headers = {key.lower(): value for key, value in additional_headers.items()}
        assert headers.pop('authorization').startswith('Bearer ')
        harness.headers.append(headers)
        harness.options.append(options)
        socket = harness.pending.popleft()
        harness.opened.append(socket)
        return socket

    monkeypatch.setattr('openai.lib._websocket._WebSocketConnect', connect)
    return harness


@pytest.mark.parametrize('defer_model_check', [False, True])
async def test_agent_connection_shorthand(
    allow_model_requests: None, sockets: SocketHarness, monkeypatch: pytest.MonkeyPatch, defer_model_check: bool
):
    """A shorthand-created agent uses its connection across ordinary and streamed runs."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    socket = sockets.pending[0]
    socket.responses.append(text_events('second', 'resp_second'))
    agent = Agent('openai-responses:gpt-4o', defer_model_check=defer_model_check)
    async with agent.connect() as connected_agent:
        assert connected_agent is agent
        first = await agent.run('first')
        assert first.output == 'ready'
        async with agent.run_stream('second', conversation=first.conversation) as second:
            assert await second.get_output() == 'second'
        assert socket.close_count == 0
    assert len(sockets.opened) == 1
    assert len(socket.sent) == 2
    assert socket.close_count == 1


async def test_agent_connection_headers(
    allow_model_requests: None, sockets: SocketHarness, monkeypatch: pytest.MonkeyPatch
):
    """Agent settings cannot silently change the headers of an already-open connection."""
    monkeypatch.setenv('OPENAI_API_KEY', 'test')
    settings: OpenAIResponsesModelSettings = {'extra_headers': {'x-tenant': 'tenant-test'}}
    agent = Agent('openai-responses:gpt-4o', model_settings=settings)
    async with agent.connect():
        assert 'x-tenant' not in sockets.headers[0]
        with pytest.raises(UserError, match='headers passed through agent or run `model_settings`') as raised:
            await agent.run('hello')
        assert 'Differing headers: `x-tenant`.' in str(raised.value)
        assert 'tenant-test' not in str(raised.value)
        assert sockets.opened[0].sent == []


@pytest.mark.parametrize('wrap_model', [False, True])
@pytest.mark.parametrize('wrap_agent', [False, True])
async def test_agent_connection_restores_http(
    allow_model_requests: None, sockets: SocketHarness, wrap_model: bool, wrap_agent: bool
):
    """Connection exit restores HTTP, preserves wrappers, and leaves the source client open."""
    requests: list[httpx2.Request] = []
    wrapper_requests: list[str] = []

    def http_handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        response = text_events()[-1]['response']
        assert isinstance(response, dict)
        response['output'] = [
            {
                'type': 'message',
                'id': 'msg_http',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': 'http', 'annotations': [], 'logprobs': []}],
            }
        ]
        return httpx2.Response(200, json=response)

    class RecordingWrapper(WrapperModel):
        async def request(
            self,
            messages: list[ModelRequest | ModelResponse],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            wrapper_requests.append('request')
            return await super().request(messages, model_settings, model_request_parameters)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(http_handler)) as http_client:
        source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test', http_client=http_client))
        agent: AbstractAgent[None, str] = Agent(RecordingWrapper(source) if wrap_model else source)
        if wrap_agent:
            agent = WrapperAgent(agent)
        assert (await agent.run('before connecting')).output == 'http'
        with pytest.raises(ValueError, match='leave connection'):
            async with agent.connect() as connected_agent:
                assert connected_agent is agent
                assert (await agent.run('connected')).output == 'ready'
                raise ValueError('leave connection')
        assert not http_client.is_closed
        assert (await agent.run('after connecting')).output == 'http'
    assert len(requests) == 2
    assert len(sockets.opened[0].sent) == 1
    assert sockets.opened[0].close_count == 1
    assert wrapper_requests == (['request'] * 3 if wrap_model else [])


async def test_agent_connection_nested_override(allow_model_requests: None, sockets: SocketHarness):
    """Nested connections restore the outer connection and then the original agent model."""
    outer = sockets.pending[0]
    outer.responses.append(text_events('outer again', 'resp_outer_again'))
    inner = ScriptedSocket(responses=deque([text_events('inner', 'resp_inner')]))
    sockets.pending.append(inner)
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    agent = Agent(TestModel(custom_output_text='original'))
    with agent.override(model=source):
        async with agent.connect():
            assert (await agent.run('outer')).output == 'ready'
            async with agent.connect():
                assert (await agent.run('inner')).output == 'inner'
            assert inner.close_count == 1 and outer.close_count == 0
            assert (await agent.run('outer again')).output == 'outer again'
    assert (await agent.run('original')).output == 'original'
    assert outer.close_count == 1
    assert [len(socket.sent) for socket in sockets.opened] == [2, 1]


async def test_agent_connections_are_task_local(allow_model_requests: None, sockets: SocketHarness):
    """Two tasks share an agent without sharing or replacing each other's connection."""
    first = sockets.pending[0]
    first.responses.clear()
    second = ScriptedSocket()
    sockets.pending.append(second)
    agent = Agent(OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test')))
    results: list[str] = []
    before = asyncio.all_tasks()

    async def first_run() -> None:
        async with agent.connect():
            results.append((await agent.run('first')).output)
            results.append((await agent.run('first again')).output)

    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(first_run)
            await first.receiving.wait()
            async with agent.connect():
                assert (await agent.run('second')).output == 'ready'
            assert second.close_count == 1 and first.close_count == 0
            first.responses.append(text_events('first again', 'resp_first_again'))
            first.push(*text_events('first', 'resp_first'))
    assert results == ['first', 'first again']
    assert [len(socket.sent) for socket in sockets.opened] == [2, 1]
    assert [socket.close_count for socket in sockets.opened] == [1, 1]
    assert asyncio.all_tasks() == before


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
        source = OpenAIResponsesModel(
            'gpt-4o',
            provider=provider,
            settings=OpenAIResponsesModelSettings(openai_responses_service_tier='ultrafast'),
        )
        async with source.connect() as connected, source.connect() as independent:
            assert connected is not source and independent is not connected
            # A direct request avoids output validation: the HTTP stub only needs a complete response.
            await source.request([ModelRequest(parts=[UserPromptPart('hello')])], None, ModelRequestParameters())
            assert requests[0].url.path == '/v1/responses'
            assert FRAME_ADAPTER.validate_json(requests[0].content)['service_tier'] == 'ultrafast'
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
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        async with source, anyio.create_task_group() as tasks:
            async with source.connect() as connected:

                async def request() -> None:
                    with pytest.raises(ModelAPIError, match='interrupted before completion'):
                        await Agent(connected).run('hello')

                tasks.start_soon(request)
                await socket.receiving.wait()
    assert socket.close_count == 1
    assert asyncio.all_tasks() == before


async def test_overlap(allow_model_requests: None, sockets: SocketHarness):
    socket = sockets.pending[0]
    socket.responses.clear()
    sockets.pending.append(ScriptedSocket())
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source.connect() as connected, source.connect() as independent:
        result: list[str] = []

        async def first_run() -> None:
            result.append((await Agent(connected).run('first')).output)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(first_run)
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await socket.receiving.wait()
            with pytest.raises(UserError, match='one active response'):
                await Agent(connected).run('overlapping')
            assert (await Agent(independent).run('independent')).output == 'ready'
            assert len(socket.sent) == 1
            socket.push(*text_events())
        assert result == ['ready']
        assert socket.close_count == 0


@pytest.mark.parametrize('interrupt', ['early_exit', 'explicit_close', 'cancel'])
async def test_interruption(allow_model_requests: None, sockets: SocketHarness, interrupt: str):
    socket = sockets.pending[0]
    socket.responses = deque([text_events()[:-1]])
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
        assert socket.close_count == 1
        with pytest.raises(UserError, match='closed'):
            await Agent(connected).run('cannot reuse')
    assert socket.close_count == 1
    assert asyncio.all_tasks() == before


async def test_stream_cancel_during_receive(allow_model_requests: None, sockets: SocketHarness):
    """Cancelling a blocked consumer closes the socket without leaking its transport error."""
    socket = sockets.pending[0]
    socket.responses = deque([text_events()[:-1]])
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
    assert socket.close_count == 1


@pytest.mark.parametrize(
    'failure', ['envelope', 'status', 'model_not_found', 'failed', 'steering', 'disconnect', 'send', 'timeout']
)
@pytest.mark.parametrize('stream', [False, True])
async def test_errors(allow_model_requests: None, sockets: SocketHarness, failure: str, stream: bool):
    """Server errors and interrupted transports are not mistaken for successful partial output."""
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
    source = OpenAIResponsesModel(model_name, provider=OpenAIProvider(api_key='test'))
    async with source.connect() as connected:
        with pytest.raises(UnexpectedModelBehavior if failure == 'steering' else ModelAPIError) as raised:
            agent = Agent(connected, model_settings=settings)
            if stream:
                async with agent.run_stream('hello') as result:
                    await result.get_output()
            else:
                await agent.run('hello')
        assert socket.close_count == (0 if failure == 'failed' else 1)
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
            socket.responses.append(text_events())
            assert (await Agent(connected).run('recover')).output == 'ready'


async def test_incomplete_response(allow_model_requests: None, sockets: SocketHarness):
    events = text_events()
    terminal = events[-1]
    terminal['type'] = 'response.incomplete'
    response = terminal['response']
    assert isinstance(response, dict)
    response.update(status='incomplete', incomplete_details={'reason': 'max_output_tokens'})
    sockets.pending[0].responses = deque([events, text_events()])
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    async with source.connect() as connected:
        result = await connected.request(
            [ModelRequest(parts=[UserPromptPart('hello')])], None, ModelRequestParameters()
        )
        assert result.finish_reason == 'length'
        assert result.provider_details is not None
        assert result.provider_details['finish_reason'] == 'max_output_tokens'
        assert sockets.opened[0].close_count == 0
        assert (await Agent(connected).run('next')).output == 'ready'


async def test_handshake_timeout(sockets: SocketHarness):
    """A stalled handshake honors the model's connect timeout without leaking a connection."""
    before = asyncio.all_tasks()
    sockets.connect_gate = anyio.Event()
    settings: OpenAIResponsesModelSettings = {'timeout': Timeout(10, connect=0.01)}
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'), settings=settings)
    async with source:
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
        with pytest.raises(UserError, match='closed'):
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
        if isinstance(raised.value, ModelHTTPError):
            assert raised.value.status_code == status_code
            assert raised.value.body == 'handshake rejected'
            assert raised.value.headers == {'x-request-id': 'first, second'}
        assert not source.client.is_closed()
    finally:
        await source.client.close()


async def test_codex_connection_unsupported(sockets: SocketHarness):
    source = OpenAICodexModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    try:
        with pytest.raises(UserError, match='`OpenAICodexModel` does not support explicit connections'):
            async with Agent(source).connect():
                pytest.fail('Codex connections require conversation-specific headers')  # pragma: no cover
        assert not sockets.connecting.is_set()
        assert not source.client.is_closed()
    finally:
        await source.client.close()


async def test_missing_websocket_dependency(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, 'pydantic_ai.models._openai_responses_websocket', None)
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    with pytest.raises(ImportError, match=r'Install `pydantic-ai-slim\[openai,realtime\]`'):
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
    ],
)
async def test_request_header_overrides(
    allow_model_requests: None, sockets: SocketHarness, request_headers: dict[str, str], differing_headers: str | None
):
    settings: OpenAIResponsesModelSettings = {
        'extra_headers': {'OpenAI-Organization': 'org-test', 'X-Tenant': 'tenant-test'}
    }
    request_settings: OpenAIResponsesModelSettings = {'extra_headers': request_headers}
    source = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'), settings=settings)
    agent = Agent(source, model_settings=request_settings)
    async with agent.connect():
        assert sockets.headers[0]['openai-organization'] == 'org-test'
        assert sockets.headers[0]['x-tenant'] == 'tenant-test'
        if differing_headers is None:
            assert (await agent.run('hello')).output == 'ready'
            assert len(sockets.opened[0].sent) == 1
        else:
            with pytest.raises(UserError, match='Request `extra_headers` must match') as raised:
                await agent.run('hello')
            assert f'Differing headers: {differing_headers}.' in str(raised.value)
            for value in ('org-test', 'tenant-test', *request_headers.values()):
                assert value not in str(raised.value)
            assert sockets.opened[0].sent == []


@pytest.mark.parametrize(
    'request_headers',
    [
        pytest.param({}, id='inherited'),
        pytest.param({'X-Tenant': 'tenant-test'}, id='explicit'),
        pytest.param({'x-tenant': 'tenant-test'}, id='case-insensitive'),
        pytest.param({'Authorization': 'Bearer test'}, id='authorization'),
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


async def test_request_settings(allow_model_requests: None, sockets: SocketHarness):
    settings: OpenAIResponsesModelSettings = {
        'openai_responses_service_tier': 'ultrafast',
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
    assert sent['service_tier'] == 'ultrafast'
    assert sent['store'] is False
    assert sent['temperature'] == 0.3
    assert sent['metadata'] == {'test': 'websocket'}

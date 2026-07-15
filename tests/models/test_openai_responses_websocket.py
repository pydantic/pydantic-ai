"""Tests for OpenAI Responses API WebSocket mode."""

from __future__ import annotations as _annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelAPIError, ModelHTTPError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.usage import RunUsage

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText
    from openai.types.websocket_connection_options import WebSocketConnectionOptions
    from websockets.datastructures import Headers
    from websockets.exceptions import ConnectionClosedOK, InvalidStatus, WebSocketException
    from websockets.http11 import Response

    from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import response_message
    from .websocket_cassettes import CassetteInteraction, ReplayConnect, ReplayWebSocket, WebSocketCassette

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai/websockets not installed'),
    pytest.mark.anyio,
]

_HELLO_PROMPT = 'Say "hello" and nothing else.'


@pytest.fixture
def openai_model(openai_api_key: str, allow_model_requests: None) -> OpenAIResponsesModel:
    return OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))


def _http_text_response(text: str = 'hello from HTTP') -> Any:
    return response_message(
        [
            ResponseOutputMessage(
                id='msg_http',
                content=[ResponseOutputText(text=text, type='output_text', annotations=[])],
                role='assistant',
                status='completed',
                type='message',
            )
        ]
    )


async def _assert_uses_http(model: OpenAIResponsesModel, prompt: str = 'Use HTTP.') -> None:
    create = AsyncMock(return_value=_http_text_response())
    with patch.object(model.client.responses, 'create', create):
        result = await Agent(model).run(prompt)

    assert result.output == 'hello from HTTP'
    create.assert_awaited_once()


def _cassette_path(name: str) -> Path:
    return Path(__file__).parent / 'cassettes' / Path(__file__).stem / name


def _load_cassette(name: str) -> WebSocketCassette:
    return WebSocketCassette.load(_cassette_path(name))


class _StubWebSocket:
    """Minimal stub that satisfies the SDK websocket interface."""

    def __init__(self, sent_event: asyncio.Event, send_exception: Exception | None = None) -> None:
        self._sent_event = sent_event
        self._send_exception = send_exception

    async def send(self, data: str | bytes) -> None:
        self._sent_event.set()
        if self._send_exception is not None:
            raise self._send_exception

    async def recv(self, *, decode: bool | None = False) -> bytes:
        await asyncio.sleep(3600)
        return b''  # pragma: no cover

    async def close(self, *, code: int = 1000, reason: str = '') -> None:
        pass

    @property
    def protocol(self) -> Any:
        return None  # pragma: no cover


@pytest.mark.ws_cassette
async def test_connect_lifecycle(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect() as connected_model:
        assert connected_model is openai_ws_model
        result = await agent.run(_HELLO_PROMPT)

    assert 'hello' in result.output.lower()
    await _assert_uses_http(openai_ws_model)


async def test_connect_parallel_separate_connections(openai_model: OpenAIResponsesModel) -> None:
    served_by: list[ReplayWebSocket] = []

    class TrackingReplayWebSocket(ReplayWebSocket):
        async def send(self, message: str | bytes) -> None:
            served_by.append(self)
            await super().send(message)

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(TrackingReplayWebSocket(_load_cassette('test_ws_simple_text_request.yaml')))

    async def run_one() -> str:
        async with openai_model.connect():
            return (await Agent(openai_model).run(_HELLO_PROMPT)).output

    with patch('websockets.asyncio.client.connect', fake_connect):
        outputs = await asyncio.gather(run_one(), run_one())

    assert all('hello' in output.lower() for output in outputs)
    assert len(served_by) == 2
    assert served_by[0] is not served_by[1]


async def test_no_connect_uses_http(openai_model: OpenAIResponsesModel) -> None:
    def fail_connect(*args: Any, **kwargs: Any) -> None:
        raise AssertionError('websocket connection should not be opened')  # pragma: no cover

    with patch('websockets.asyncio.client.connect', fail_connect):
        await _assert_uses_http(openai_model)


async def test_child_task_outliving_connect_uses_http(openai_model: OpenAIResponsesModel) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def run_after_context() -> str:
        started.set()
        await release.wait()
        return (await Agent(openai_model).run('Use HTTP.')).output

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(WebSocketCassette()))

    create = AsyncMock(return_value=_http_text_response())
    with (
        patch('websockets.asyncio.client.connect', fake_connect),
        patch.object(openai_model.client.responses, 'create', create),
    ):
        async with openai_model.connect():
            task = asyncio.create_task(run_after_context())
            await started.wait()

        release.set()
        assert await task == 'hello from HTTP'

    create.assert_awaited_once()


@pytest.mark.parametrize('stream', [False, True])
async def test_ws_background_mode_not_supported(openai_model: OpenAIResponsesModel, stream: bool) -> None:
    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(WebSocketCassette()))

    agent = Agent(
        openai_model,
        model_settings=OpenAIResponsesModelSettings(openai_background=True),
    )
    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            with pytest.raises(UserError, match='`openai_background` is not supported'):
                if stream:
                    async with agent.run_stream('Run in the background.'):
                        pass  # pragma: no cover
                else:
                    await agent.run('Run in the background.')


@pytest.mark.parametrize('stream', [False, True])
async def test_ws_suspended_response_resumes_over_http(openai_model: OpenAIResponsesModel, stream: bool) -> None:
    request = ModelRequest(parts=[UserPromptPart(content='Continue.')])
    suspended = ModelResponse(
        parts=[],
        model_name='gpt-4o-mini',
        provider_name='openai',
        provider_response_id='resp_background',
        state='suspended',
    )
    settings = OpenAIResponsesModelSettings(openai_background=True)
    retrieve = AsyncMock(return_value=_http_text_response('resumed over HTTP'))

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(WebSocketCassette()))

    with (
        patch('websockets.asyncio.client.connect', fake_connect),
        patch.object(openai_model, '_responses_retrieve', retrieve),
    ):
        async with openai_model.connect():
            if stream:
                async with openai_model.request_stream(
                    [request, suspended], settings, ModelRequestParameters()
                ) as response:
                    async for _ in response:
                        pass
                    assert response.get().parts
            else:
                response = await openai_model.request([request, suspended], settings, ModelRequestParameters())
                assert response.parts

    retrieve.assert_awaited_once_with('resp_background', settings)


async def test_ws_invalid_status_raises_model_http_error(openai_model: OpenAIResponsesModel) -> None:
    response = Response(429, 'Too Many Requests', Headers(), body=b'too many requests')

    def fake_connect(*args: Any, **kwargs: Any) -> Any:
        raise InvalidStatus(response)

    with patch('websockets.asyncio.client.connect', fake_connect):
        with pytest.raises(ModelHTTPError) as exc_info:
            async with openai_model.connect():
                pass  # pragma: no cover

    assert exc_info.value.status_code == 429


async def test_ws_os_error_handshake_raises_model_api_error(openai_model: OpenAIResponsesModel) -> None:
    def fake_connect(*args: Any, **kwargs: Any) -> Any:
        raise OSError('dns failure')

    with patch('websockets.asyncio.client.connect', fake_connect):
        with pytest.raises(ModelAPIError, match='connection failed'):
            async with openai_model.connect():
                pass  # pragma: no cover


async def test_connect_respects_allow_model_requests(openai_api_key: str) -> None:
    model = OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))

    with pytest.raises(RuntimeError, match='ALLOW_MODEL_REQUESTS is False'):
        async with model.connect():
            pass  # pragma: no cover


async def test_connect_forwards_handshake_options(openai_model: OpenAIResponsesModel) -> None:
    extra_query = {'trace': 'request-123'}
    extra_headers = {'X-Test-Header': 'header-value'}
    websocket_connection_options = WebSocketConnectionOptions(compression=None, max_size=1024)
    manager = ReplayConnect(ReplayWebSocket(WebSocketCassette()))

    with patch.object(openai_model.client.responses, 'connect', return_value=manager) as connect:
        async with openai_model.connect(
            extra_query=extra_query,
            extra_headers=extra_headers,
            websocket_connection_options=websocket_connection_options,
        ):
            pass

    connect.assert_called_once_with(
        extra_query=extra_query,
        extra_headers=extra_headers,
        websocket_connection_options=websocket_connection_options,
    )


async def test_connect_requires_websockets(openai_model: OpenAIResponsesModel) -> None:
    real_import = __import__

    def import_without_websockets(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == 'websockets':
            raise ImportError
        return real_import(name, *args, **kwargs)

    assert import_without_websockets('asyncio') is asyncio

    with patch('builtins.__import__', import_without_websockets):
        with pytest.raises(ImportError, match=r'pip install "openai\[realtime\]"'):
            async with openai_model.connect():
                pass  # pragma: no cover


async def test_ws_concurrent_requests_error(openai_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_model)
    sent = asyncio.Event()

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(_StubWebSocket(sent))

    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            first = asyncio.create_task(agent.run('Hello'))
            await sent.wait()
            with pytest.raises(UserError, match='already handling a request'):
                await agent.run('World')
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first


async def test_ws_send_connection_closed_raises_model_api_error(openai_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_model)

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(_StubWebSocket(asyncio.Event(), ConnectionClosedOK(None, None)))

    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            with pytest.raises(ModelAPIError, match='connection closed'):
                await agent.run('Hello')


async def test_ws_send_websocket_exception_raises_model_api_error(openai_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_model)

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(_StubWebSocket(asyncio.Event(), WebSocketException('boom')))

    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            with pytest.raises(ModelAPIError, match='WebSocket error'):
                await agent.run('Hello')


async def test_ws_concurrent_streaming_requests_error(openai_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_model)

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(_load_cassette('test_ws_streamed_text_request.yaml')))

    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            async with agent.run_stream(_HELLO_PROMPT):
                with pytest.raises(UserError, match='already handling a request'):
                    async with agent.run_stream(_HELLO_PROMPT):
                        pass  # pragma: no cover


@pytest.mark.ws_cassette
async def test_ws_simple_text_request(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        result = await agent.run(_HELLO_PROMPT)

    assert 'hello' in result.output.lower()
    assert result.usage == RunUsage(requests=1, input_tokens=15, output_tokens=3, details={'reasoning_tokens': 0})


async def test_ws_request_forwards_optional_parameters(openai_model: OpenAIResponsesModel) -> None:
    settings = OpenAIResponsesModelSettings(
        max_tokens=123,
        temperature=0.25,
        top_p=0.75,
        openai_service_tier='priority',
        openai_truncation='auto',
        openai_context_management=[{'type': 'compaction', 'compact_threshold': 2048}],
        openai_store=False,
        openai_user='user_123',
        openai_include_code_execution_outputs=True,
        openai_include_web_search_sources=True,
        openai_include_file_search_results=True,
        openai_logprobs=True,
        openai_top_logprobs=2,
        openai_prompt_cache_key='cache_key_123',
        openai_prompt_cache_retention='24h',
        openai_conversation_id='conv_123',
    )
    expected_request = CassetteInteraction(
        direction='sent',
        data={
            'type': 'response.create',
            'model': 'gpt-4o-mini',
            'input': [{'role': 'user', 'content': 'Forward every setting.'}],
            'conversation': 'conv_123',
            'truncation': 'auto',
            'context_management': [{'type': 'compaction', 'compact_threshold': 2048}],
            'max_output_tokens': 123,
            'temperature': 0.25,
            'top_p': 0.75,
            'service_tier': 'priority',
            'top_logprobs': 2,
            'store': False,
            'user': 'user_123',
            'include': [
                'code_interpreter_call.outputs',
                'web_search_call.action.sources',
                'file_search_call.results',
                'message.output_text.logprobs',
            ],
            'prompt_cache_key': 'cache_key_123',
            'prompt_cache_retention': '24h',
        },
    )
    simple = _load_cassette('test_ws_simple_text_request.yaml')
    cassette = WebSocketCassette(
        interactions=[expected_request, *(item for item in simple.interactions if item.direction != 'sent')]
    )

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(cassette))

    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            response = await openai_model.request(
                [ModelRequest(parts=[UserPromptPart(content='Forward every setting.')])],
                settings,
                ModelRequestParameters(),
            )

    assert response.parts


@pytest.mark.ws_cassette
async def test_ws_streamed_text_request(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        async with agent.run_stream(_HELLO_PROMPT) as result:
            output = await result.get_output()

    assert 'hello' in output.lower()


@pytest.mark.ws_cassette
async def test_ws_sequential_requests(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        result1 = await agent.run('Say "first"')
        result2 = await agent.run('Say "second"')

    assert result1.output == 'First.'
    assert result2.output == 'Second.'


@pytest.mark.ws_cassette
async def test_ws_sequential_streamed_requests(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        async with agent.run_stream('Say "first"') as result1:
            output1 = await result1.get_output()
        async with agent.run_stream('Say "second"') as result2:
            output2 = await result2.get_output()

    assert output1 == 'First.'
    assert output2 == 'Second.'


@pytest.mark.ws_cassette
async def test_ws_request_with_tools(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    @agent.tool_plain
    def get_temperature(city: str) -> str:
        """Get the current temperature for a city."""
        return f'25°C in {city}'

    async with openai_ws_model.connect():
        result = await agent.run('What is the temperature in Paris?')

    assert result.output
    tool_calls = [
        part
        for msg in result.all_messages()
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    assert len(tool_calls) >= 1
    assert tool_calls[0].tool_name == 'get_temperature'


async def test_ws_tool_continuation_uses_connection_local_state(openai_model: OpenAIResponsesModel) -> None:
    cassette = deepcopy(_load_cassette('test_ws_request_with_tools.yaml'))
    sent = [interaction for interaction in cassette.interactions if interaction.direction == 'sent']
    assert len(sent) == 2
    first_response_id = next(
        interaction.data['response']['id']
        for interaction in cassette.interactions
        if interaction.data.get('type') == 'response.completed'
    )

    sent[0].data['store'] = False
    tool_output = next(item for item in sent[1].data['input'] if item.get('type') == 'function_call_output')
    sent[1].data['input'] = [tool_output]
    sent[1].data['previous_response_id'] = first_response_id
    sent[1].data['store'] = False

    after_first_response = False
    for interaction in cassette.interactions:
        if interaction is sent[1]:
            after_first_response = True
        response = interaction.data.get('response')
        if isinstance(response, dict):
            response['store'] = False
            if after_first_response:
                response['previous_response_id'] = first_response_id

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(cassette))

    agent = Agent(
        openai_model,
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto', openai_store=False),
    )

    @agent.tool_plain
    def get_temperature(city: str) -> str:
        """Get the current temperature for a city."""
        return f'25°C in {city}'

    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            result = await agent.run('What is the temperature in Paris?')

    assert result.output


@pytest.mark.ws_cassette
async def test_ws_request_with_structured_output(openai_ws_model: OpenAIResponsesModel) -> None:
    class CityInfo(BaseModel):
        name: str
        country: str

    agent: Agent[None, CityInfo] = Agent(openai_ws_model, output_type=CityInfo)

    async with openai_ws_model.connect():
        result = await agent.run('Tell me about Paris.')

    assert isinstance(result.output, CityInfo)
    assert result.output.name


@pytest.mark.ws_cassette
async def test_ws_error_event_raises(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        with pytest.raises(ModelHTTPError) as exc_info:
            await agent.run('trigger an error')

    assert exc_info.value.status_code == 400
    assert exc_info.value.body == {
        'code': 'previous_response_not_found',
        'message': "Previous response with id 'resp_abc' not found.",
        'param': 'previous_response_id',
    }


@pytest.mark.ws_cassette
async def test_ws_error_event_raises_streaming(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        with pytest.raises(ModelAPIError, match='server had an error'):
            async with agent.run_stream('trigger an error'):
                pass  # pragma: no cover


async def test_ws_error_event_leaves_connection_usable(openai_model: OpenAIResponsesModel) -> None:
    error = _load_cassette('test_ws_error_event_raises.yaml')
    success = _load_cassette('test_ws_simple_text_request.yaml')
    cassette = WebSocketCassette(interactions=[*error.interactions, *success.interactions])

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(cassette))

    agent = Agent(openai_model)
    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            with pytest.raises(ModelHTTPError):
                await agent.run('trigger an error')
            result = await agent.run(_HELLO_PROMPT)

    assert 'hello' in result.output.lower()


async def test_ws_connection_limit_error_requires_new_context(openai_model: OpenAIResponsesModel) -> None:
    cassette = deepcopy(_load_cassette('test_ws_error_event_raises.yaml'))
    error = cassette.interactions[-1].data['error']
    error['code'] = 'websocket_connection_limit_reached'
    error['message'] = 'Create a new websocket connection to continue.'

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(cassette))

    agent = Agent(openai_model)
    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            with pytest.raises(ModelHTTPError):
                await agent.run('trigger an error')
            with pytest.raises(UserError, match=r'cancelled or failed|unknown state'):
                await agent.run(_HELLO_PROMPT)


@pytest.mark.parametrize('terminal_status', ['failed', 'incomplete'])
@pytest.mark.parametrize('stream', [False, True])
async def test_ws_terminal_response_leaves_connection_usable(
    openai_model: OpenAIResponsesModel, terminal_status: str, stream: bool
) -> None:
    first = deepcopy(_load_cassette('test_ws_simple_text_request.yaml'))
    terminal = next(item for item in reversed(first.interactions) if item.data.get('type') == 'response.completed')
    terminal.data['type'] = f'response.{terminal_status}'
    terminal.data['response']['status'] = terminal_status
    if terminal_status == 'failed':
        terminal.data['response']['error'] = {'code': 'server_error', 'message': 'The response failed.'}
    else:
        terminal.data['response']['incomplete_details'] = {'reason': 'max_output_tokens'}

    success = _load_cassette('test_ws_simple_text_request.yaml')
    cassette = WebSocketCassette(interactions=[*first.interactions, *success.interactions])

    def fake_connect(*args: Any, **kwargs: Any) -> ReplayConnect:
        return ReplayConnect(ReplayWebSocket(cassette))

    messages: list[ModelRequest | ModelResponse] = [ModelRequest(parts=[UserPromptPart(content=_HELLO_PROMPT)])]
    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            if stream:
                async with openai_model.request_stream(messages, None, ModelRequestParameters()) as first_response:
                    async for _ in first_response:
                        pass
                assert first_response.get().state == 'complete'
            else:
                first_response = await openai_model.request(messages, None, ModelRequestParameters())
                assert first_response.state == 'complete'

            second_response = await openai_model.request(messages, None, ModelRequestParameters())

    assert second_response.state == 'complete'


@pytest.mark.ws_cassette
async def test_ws_connection_closed_before_terminal(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        with pytest.raises(UnexpectedModelBehavior, match='closed before a terminal'):
            await agent.run('hello')


@pytest.mark.ws_cassette
async def test_ws_connection_closed_before_terminal_streaming(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        with pytest.raises(UnexpectedModelBehavior, match='closed before a terminal'):
            async with agent.run_stream('hello'):
                pass  # pragma: no cover


@pytest.mark.ws_cassette
async def test_ws_abnormal_close_raises_model_api_error(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        with pytest.raises(ModelAPIError, match='closed unexpectedly'):
            await agent.run('hello')
        with pytest.raises(UserError, match=r'cancelled or failed|unknown state'):
            await agent.run('hello again')


@pytest.mark.ws_cassette
async def test_ws_abnormal_close_streaming_raises_model_api_error(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        with pytest.raises(ModelAPIError, match='closed unexpectedly'):
            async with agent.run_stream('hello'):
                pass  # pragma: no cover


@pytest.mark.ws_cassette(allow_unconsumed=True)
async def test_ws_stream_cancel(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        async with agent.run_stream(_HELLO_PROMPT) as result:
            async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                break
            with pytest.raises(NotImplementedError, match='Stream cancellation is not supported in WebSocket mode'):
                await result.cancel()
            assert result.cancelled


@pytest.mark.ws_cassette(allow_unconsumed=True)
async def test_ws_poisoned_session_after_early_stream_exit(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        async with agent.run_stream(_HELLO_PROMPT) as result:
            async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                break

        with pytest.raises(UserError, match=r'cancelled or failed|unknown state'):
            await agent.run(_HELLO_PROMPT)


@pytest.mark.ws_cassette
async def test_ws_cross_model_isolation(openai_ws_model: OpenAIResponsesModel, openai_api_key: str) -> None:
    model_b = OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))
    agent_a: Agent[None, str] = Agent(openai_ws_model)
    agent_b: Agent[None, str] = Agent(model_b)

    create_b = AsyncMock(return_value=_http_text_response())
    with patch.object(model_b.client.responses, 'create', create_b):
        async with openai_ws_model.connect():
            result_b = await agent_b.run('Use HTTP.')
            result_a = await agent_a.run(_HELLO_PROMPT)

    assert result_b.output == 'hello from HTTP'
    assert 'hello' in result_a.output.lower()
    create_b.assert_awaited_once()

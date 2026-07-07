"""Tests for OpenAI Responses API WebSocket mode."""

from __future__ import annotations as _annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ModelAPIError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.usage import RunUsage

from ..conftest import try_import

with try_import() as imports_successful:
    import websockets  # noqa: F401  # pyright: ignore[reportUnusedImport]
    from openai.types.responses import ResponseOutputMessage, ResponseOutputText

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import response_message
    from .websocket_cassettes import ReplayWebSocket, WebSocketCassette

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


class _ReplayConnect:
    """Mimics `websockets.connect` as an awaitable and async context manager."""

    def __init__(self, ws: ReplayWebSocket):
        self._ws = ws

    def __await__(self) -> Any:
        async def _resolve() -> ReplayWebSocket:
            return self._ws

        return _resolve().__await__()

    async def __aenter__(self) -> ReplayWebSocket:
        return self._ws  # pragma: no cover

    async def __aexit__(self, *args: Any) -> None:
        pass


class _StubWebSocket:
    """Minimal stub that satisfies the SDK websocket interface."""

    def __init__(self, sent_event: asyncio.Event | None = None) -> None:
        self._sent_event = sent_event

    async def send(self, data: str | bytes) -> None:
        if self._sent_event is not None:
            self._sent_event.set()

    async def recv(self, *, decode: bool | None = False) -> bytes:
        await asyncio.sleep(3600)
        return b''  # pragma: no cover

    async def close(self, *, code: int = 1000, reason: str = '') -> None:
        pass

    @property
    def protocol(self) -> Any:
        return None  # pragma: no cover


class _StubConnect:
    """Mimics `websockets.connect` as an awaitable and async context manager."""

    def __init__(self, ws: _StubWebSocket):
        self._ws = ws

    def __await__(self) -> Any:
        async def _resolve() -> _StubWebSocket:
            return self._ws

        return _resolve().__await__()

    async def __aenter__(self) -> _StubWebSocket:
        return self._ws  # pragma: no cover

    async def __aexit__(self, *args: Any) -> None:
        pass


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

    def fake_connect(*args: Any, **kwargs: Any) -> _ReplayConnect:
        return _ReplayConnect(TrackingReplayWebSocket(_load_cassette('test_ws_simple_text_request.yaml')))

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
        raise AssertionError('websocket connection should not be opened')

    with patch('websockets.asyncio.client.connect', fail_connect):
        await _assert_uses_http(openai_model)


async def test_connect_respects_allow_model_requests(openai_api_key: str) -> None:
    model = OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))

    with pytest.raises(RuntimeError, match='ALLOW_MODEL_REQUESTS is False'):
        async with model.connect():
            pass  # pragma: no cover


async def test_ws_concurrent_requests_error(openai_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_model)
    sent = asyncio.Event()

    def fake_connect(*args: Any, **kwargs: Any) -> _StubConnect:
        return _StubConnect(_StubWebSocket(sent))

    with patch('websockets.asyncio.client.connect', fake_connect):
        async with openai_model.connect():
            first = asyncio.create_task(agent.run('Hello'))
            await sent.wait()
            with pytest.raises(UserError, match='already handling a request'):
                await agent.run('World')
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first


async def test_ws_concurrent_streaming_requests_error(openai_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_model)

    def fake_connect(*args: Any, **kwargs: Any) -> _ReplayConnect:
        return _ReplayConnect(ReplayWebSocket(_load_cassette('test_ws_streamed_text_request.yaml')))

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

    assert result1.output
    assert result2.output


@pytest.mark.ws_cassette
async def test_ws_sequential_streamed_requests(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        async with agent.run_stream(_HELLO_PROMPT) as result1:
            output1 = await result1.get_output()
        async with agent.run_stream(_HELLO_PROMPT) as result2:
            output2 = await result2.get_output()

    assert 'hello' in output1.lower()
    assert 'hello' in output2.lower()


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
        with pytest.raises(ModelAPIError, match='server had an error'):
            await agent.run('trigger an error')


@pytest.mark.ws_cassette
async def test_ws_error_event_raises_streaming(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        with pytest.raises(ModelAPIError, match='server had an error'):
            async with agent.run_stream('trigger an error'):
                pass  # pragma: no cover


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


@pytest.mark.ws_cassette
async def test_ws_abnormal_close_streaming_raises_model_api_error(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        with pytest.raises(ModelAPIError, match='closed unexpectedly'):
            async with agent.run_stream('hello'):
                pass  # pragma: no cover


@pytest.mark.ws_cassette
async def test_ws_stream_cancel(openai_ws_model: OpenAIResponsesModel) -> None:
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        async with agent.run_stream(_HELLO_PROMPT) as result:
            async for _ in result.stream_text(delta=True, debounce_by=None):  # pragma: no branch
                break
            with pytest.raises(NotImplementedError, match='Stream cancellation is not supported in WebSocket mode'):
                await result.cancel()
            assert result.cancelled


@pytest.mark.ws_cassette
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

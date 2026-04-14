"""Tests for OpenAI Responses API WebSocket mode.

These tests verify that `OpenAIResponsesModel.connect()` routes requests over WebSocket
while falling back to HTTP when no connection is active.
"""

from __future__ import annotations as _annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, ToolCallPart

from ..conftest import try_import

with try_import() as imports_successful:
    import websockets  # noqa: F401

    from pydantic_ai.models.openai import _WS_CONNECTION, OpenAIResponsesModel  # pyright: ignore[reportPrivateUsage]
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai/websockets not installed'),
    pytest.mark.anyio,
]


@pytest.fixture
def openai_model(openai_api_key: str, allow_model_requests: None) -> OpenAIResponsesModel:
    """Bare model for lifecycle tests — patches websockets.connect to yield a stub."""
    return OpenAIResponsesModel('gpt-4o-mini', provider=OpenAIProvider(api_key=openai_api_key))


class _StubWebSocket:
    """Minimal stub that satisfies AsyncResponsesConnection's websocket interface."""

    async def send(self, data: str | bytes) -> None:
        pass

    async def recv(self, *, decode: bool | None = None) -> bytes:
        await asyncio.sleep(3600)
        return b''  # pragma: no cover

    async def close(self, *, code: int = 1000, reason: str = '') -> None:
        pass

    @property
    def protocol(self) -> Any:
        return None


class _StubConnect:
    """Mimics websockets.asyncio.client.connect — awaitable (returns ws) and async context manager."""

    def __init__(self, *args: Any, **kwargs: Any):
        self._ws = _StubWebSocket()

    def __await__(self) -> Any:
        async def _resolve() -> _StubWebSocket:
            return self._ws

        return _resolve().__await__()

    async def __aenter__(self) -> _StubWebSocket:
        return self._ws

    async def __aexit__(self, *args: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_connect_lifecycle(openai_model: OpenAIResponsesModel) -> None:
    """ContextVar is set inside connect(), cleared after exit, and connect() yields the model."""
    assert _WS_CONNECTION.get() is None

    with patch('websockets.asyncio.client.connect', _StubConnect):
        async with openai_model.connect() as connected_model:
            assert connected_model is openai_model
            assert _WS_CONNECTION.get() is not None

    assert _WS_CONNECTION.get() is None


async def test_connect_parallel_separate_connections(openai_model: OpenAIResponsesModel) -> None:
    """Two connect() contexts in parallel via asyncio.gather get isolated ContextVar copies."""
    connections_seen: list[object] = []

    async def capture_connection(model: OpenAIResponsesModel) -> None:
        async with model.connect():
            conn = _WS_CONNECTION.get()
            connections_seen.append(id(conn))
            await asyncio.sleep(0.01)

    with patch('websockets.asyncio.client.connect', _StubConnect):
        await asyncio.gather(
            capture_connection(openai_model),
            capture_connection(openai_model),
        )
    assert len(connections_seen) == 2
    assert connections_seen[0] != connections_seen[1]


async def test_no_connect_uses_http(openai_model: OpenAIResponsesModel) -> None:
    """Without connect(), ContextVar is None — requests go through HTTP path."""
    assert _WS_CONNECTION.get() is None


async def test_ws_request_after_disconnect(openai_model: OpenAIResponsesModel) -> None:
    """After exiting connect(), ContextVar is cleared."""
    with patch('websockets.asyncio.client.connect', _StubConnect):
        async with openai_model.connect():
            pass

    assert _WS_CONNECTION.get() is None


async def test_ws_concurrent_requests_error(openai_model: OpenAIResponsesModel) -> None:
    """Parallel agent.run() on the same model in the same connect() raises RuntimeError."""
    agent: Agent[None, str] = Agent(openai_model)

    with patch('websockets.asyncio.client.connect', _StubConnect):
        async with openai_model.connect():
            with pytest.raises(RuntimeError, match='already handling a request'):
                await asyncio.gather(
                    agent.run('Hello'),
                    agent.run('World'),
                )


# ---------------------------------------------------------------------------
# Requests (cassette-recorded)
# ---------------------------------------------------------------------------


@pytest.mark.ws_cassette
async def test_ws_simple_text_request(openai_ws_model: OpenAIResponsesModel) -> None:
    """Simple text request over WebSocket returns a text response."""
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        result = await agent.run('Say "hello" and nothing else.')

    assert 'hello' in result.output.lower()


@pytest.mark.ws_cassette
async def test_ws_streamed_text_request(openai_ws_model: OpenAIResponsesModel) -> None:
    """Streamed text request over WebSocket produces streamed text."""
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        async with agent.run_stream('Say "hello" and nothing else.') as result:
            output = await result.get_output()

    assert 'hello' in output.lower()


@pytest.mark.ws_cassette
async def test_ws_sequential_requests(openai_ws_model: OpenAIResponsesModel) -> None:
    """Two sequential agent.run() calls in the same WS context both succeed."""
    agent: Agent[None, str] = Agent(openai_ws_model)

    async with openai_ws_model.connect():
        result1 = await agent.run('Say "first"')
        result2 = await agent.run('Say "second"')

    assert result1.output
    assert result2.output


@pytest.mark.ws_cassette
async def test_ws_request_with_tools(openai_ws_model: OpenAIResponsesModel) -> None:
    """Agent with tools processes tool calls correctly over WebSocket."""
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
    """Agent with output_type processes structured output correctly over WebSocket."""
    from pydantic import BaseModel

    class CityInfo(BaseModel):
        name: str
        country: str

    agent: Agent[None, CityInfo] = Agent(openai_ws_model, output_type=CityInfo)

    async with openai_ws_model.connect():
        result = await agent.run('Tell me about Paris.')

    assert isinstance(result.output, CityInfo)
    assert result.output.name

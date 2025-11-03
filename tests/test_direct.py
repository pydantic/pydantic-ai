import asyncio
import re
from contextlib import contextmanager
from datetime import timezone
from unittest.mock import AsyncMock, patch

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.direct import (
    StreamedResponseSync,
    _prepare_model,  # pyright: ignore[reportPrivateUsage]
    model_request,
    model_request_stream,
    model_request_stream_sync,
    model_request_sync,
)
from pydantic_ai.messages import (
    FinalResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsNow, IsStr

pytestmark = pytest.mark.anyio


async def test_model_request():
    model_response = await model_request('test', [ModelRequest.user_text_prompt('x')])
    assert model_response == snapshot(
        ModelResponse(
            parts=[TextPart(content='success (no tool calls)')],
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            usage=RequestUsage(input_tokens=51, output_tokens=4),
        )
    )


async def test_model_request_tool_call():
    model_response = await model_request(
        'test',
        [ModelRequest.user_text_prompt('x')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[ToolDefinition(name='tool_name', parameters_json_schema={'type': 'object'})],
            allow_text_output=False,
        ),
    )
    assert model_response == snapshot(
        ModelResponse(
            parts=[ToolCallPart(tool_name='tool_name', args={}, tool_call_id=IsStr(regex='pyd_ai_.*'))],
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            usage=RequestUsage(input_tokens=51, output_tokens=2),
        )
    )


def test_model_request_sync():
    model_response = model_request_sync('test', [ModelRequest.user_text_prompt('x')])
    assert model_response == snapshot(
        ModelResponse(
            parts=[TextPart(content='success (no tool calls)')],
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            usage=RequestUsage(input_tokens=51, output_tokens=4),
        )
    )


def test_model_request_stream_sync():
    with model_request_stream_sync('test', [ModelRequest.user_text_prompt('x')]) as stream:
        chunks = list(stream)
        assert chunks == snapshot(
            [
                PartStartEvent(index=0, part=TextPart(content='')),
                FinalResultEvent(tool_name=None, tool_call_id=None),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='success ')),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='(no ')),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='tool ')),
                PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='calls)')),
                PartEndEvent(index=0, part=TextPart(content='success (no tool calls)')),
            ]
        )
        assert stream.response == snapshot(
            ModelResponse(
                parts=[TextPart(content='success (no tool calls)')],
                usage=RequestUsage(input_tokens=51, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                provider_name='test',
            )
        )

        repr_str = repr(stream)
        assert 'TestStreamedResponse' in repr_str
        assert 'test' in repr_str


async def test_model_request_stream():
    async with model_request_stream('test', [ModelRequest.user_text_prompt('x')]) as stream:
        chunks = [chunk async for chunk in stream]
    assert chunks == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='success ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='(no ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='tool ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='calls)')),
            PartEndEvent(index=0, part=TextPart(content='success (no tool calls)')),
        ]
    )


def test_model_request_stream_sync_without_context_manager():
    """Test that accessing properties or iterating without context manager raises RuntimeError."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('x')]

    expected_error_msg = re.escape(
        'StreamedResponseSync must be used as a context manager. Use: `with model_request_stream_sync(...) as stream:`'
    )

    stream_cm = model_request_stream_sync('test', messages)

    stream_repr = repr(stream_cm)
    assert 'StreamedResponseSync' in stream_repr
    assert 'context_entered=False' in stream_repr

    with pytest.raises(RuntimeError, match=expected_error_msg):
        _ = stream_cm.model_name

    with pytest.raises(RuntimeError, match=expected_error_msg):
        _ = stream_cm.timestamp

    with pytest.raises(RuntimeError, match=expected_error_msg):
        stream_cm.get()

    with pytest.raises(RuntimeError, match=expected_error_msg):
        stream_cm.usage()

    with pytest.raises(RuntimeError, match=expected_error_msg):
        list(stream_cm)

    with pytest.raises(RuntimeError, match=expected_error_msg):
        for _ in stream_cm:
            break  # pragma: no cover


def test_model_request_stream_sync_exception_in_stream():
    """Test handling of exceptions raised during streaming."""
    async_stream_mock = AsyncMock()
    async_stream_mock.__aenter__ = AsyncMock(side_effect=ValueError('Stream error'))

    stream_sync = StreamedResponseSync(_async_stream_cm=async_stream_mock)

    with stream_sync:
        with pytest.raises(ValueError, match='Stream error'):
            list(stream_sync)


def test_model_request_stream_sync_timeout():
    """Test timeout when stream fails to initialize."""
    async_stream_mock = AsyncMock()

    async def slow_init():
        await asyncio.sleep(0.1)

    async_stream_mock.__aenter__ = AsyncMock(side_effect=slow_init)

    stream_sync = StreamedResponseSync(_async_stream_cm=async_stream_mock)

    with patch('pydantic_ai.direct.STREAM_INITIALIZATION_TIMEOUT', 0.01):
        with stream_sync:
            with pytest.raises(RuntimeError, match='Stream failed to initialize within timeout'):
                stream_sync.get()


def test_model_request_stream_sync_intermediate_get():
    """Test getting properties of StreamedResponse before consuming all events."""
    messages: list[ModelMessage] = [ModelRequest.user_text_prompt('x')]

    with model_request_stream_sync('test', messages) as stream:
        response = stream.get()
        assert response is not None

        usage = stream.usage()
        assert usage is not None


@contextmanager
def set_instrument_default(value: bool):
    """Context manager to temporarily set the default instrumentation value."""
    initial_value = Agent._instrument_default  # pyright: ignore[reportPrivateUsage]
    try:
        Agent._instrument_default = value  # pyright: ignore[reportPrivateUsage]
        yield
    finally:
        Agent._instrument_default = initial_value  # pyright: ignore[reportPrivateUsage]


def test_prepare_model():
    with set_instrument_default(False):
        model = _prepare_model('test', None)
        assert isinstance(model, TestModel)

        model = _prepare_model('test', True)
        assert isinstance(model, InstrumentedModel)

    with set_instrument_default(True):
        model = _prepare_model('test', None)
        assert isinstance(model, InstrumentedModel)

        model = _prepare_model('test', False)
        assert isinstance(model, TestModel)

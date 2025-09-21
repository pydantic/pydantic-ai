import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from datetime import timezone
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai._run_context import RunContext
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
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RequestUsage

from .conftest import IsNow, IsStr

pytestmark = pytest.mark.anyio


class RecordingTestModel(TestModel):
    def __init__(self, *, settings: ModelSettings | None = None):
        super().__init__(settings=settings)
        self.recorded_model_settings: ModelSettings | None = None

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.recorded_model_settings = model_settings
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        self.recorded_model_settings = model_settings
        async with super().request_stream(
            messages,
            model_settings,
            model_request_parameters,
            run_context=run_context,
        ) as stream:
            yield stream


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


async def test_model_request_merges_model_settings_from_instance():
    base_settings: ModelSettings = {'temperature': 0.3, 'seed': 7}
    override_settings: ModelSettings = {'temperature': 0.9, 'max_tokens': 20}
    model = RecordingTestModel(settings=base_settings)

    await model_request(
        model,
        [ModelRequest.user_text_prompt('x')],
        model_settings=override_settings,
        instrument=False,
    )

    assert model.recorded_model_settings == {'temperature': 0.9, 'seed': 7, 'max_tokens': 20}


async def test_model_request_string_model_uses_override_settings(monkeypatch: pytest.MonkeyPatch):
    override_settings: ModelSettings = {'temperature': 0.2, 'max_tokens': 5}
    recording_model = RecordingTestModel()

    def fake_infer_model(model_identifier: object) -> RecordingTestModel:
        assert model_identifier == 'test'
        return recording_model

    monkeypatch.setattr('pydantic_ai.direct.models.infer_model', fake_infer_model)

    await model_request(
        'test',
        [ModelRequest.user_text_prompt('x')],
        model_settings=override_settings,
        instrument=False,
    )

    assert recording_model.recorded_model_settings == override_settings


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
            ]
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
        ]
    )


async def test_model_request_stream_merges_model_settings_from_instance():
    base_settings: ModelSettings = {'temperature': 0.1, 'seed': 42}
    override_settings: ModelSettings = {'temperature': 0.5, 'max_tokens': 100}
    model = RecordingTestModel(settings=base_settings)

    async with model_request_stream(
        model,
        [ModelRequest.user_text_prompt('x')],
        model_settings=override_settings,
        instrument=False,
    ) as stream:
        _ = [chunk async for chunk in stream]

    assert model.recorded_model_settings == {'temperature': 0.5, 'seed': 42, 'max_tokens': 100}


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

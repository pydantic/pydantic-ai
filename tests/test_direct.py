from contextlib import contextmanager
from datetime import timezone

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.direct import (
    _prepare_model,  # pyright: ignore[reportPrivateUsage]
    model_request,
    model_request_stream,
    model_request_sync,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.instrumented import InstrumentedModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import Usage

from .conftest import IsNow, IsStr

pytestmark = pytest.mark.anyio


async def test_model_request():
    model_response = await model_request('test', [ModelRequest.user_text_prompt('x')])
    assert model_response == snapshot(
        ModelResponse(
            parts=[TextPart(content='success (no tool calls)')],
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            usage=Usage(requests=1, request_tokens=51, response_tokens=4, total_tokens=55),
        )
    )


async def test_model_request_tool_call():
    model_response = await model_request(
        'test',
        [ModelRequest.user_text_prompt('x')],
        model_request_parameters=ModelRequestParameters(
            function_tools=[
                ToolDefinition(name='tool_name', description='', parameters_json_schema={'type': 'object'})
            ],
            allow_text_output=False,
        ),
    )
    assert model_response == snapshot(
        ModelResponse(
            parts=[ToolCallPart(tool_name='tool_name', args={}, tool_call_id=IsStr(regex='pyd_ai_.*'))],
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            usage=Usage(requests=1, request_tokens=51, response_tokens=2, total_tokens=53),
        )
    )


def test_model_request_sync():
    model_response = model_request_sync('test', [ModelRequest.user_text_prompt('x')])
    assert model_response == snapshot(
        ModelResponse(
            parts=[TextPart(content='success (no tool calls)')],
            model_name='test',
            timestamp=IsNow(tz=timezone.utc),
            usage=Usage(requests=1, request_tokens=51, response_tokens=4, total_tokens=55),
        )
    )


async def test_model_request_stream():
    async with model_request_stream('test', [ModelRequest.user_text_prompt('x')]) as stream:
        chunks = [chunk async for chunk in stream]
    assert chunks == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='success ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='(no ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='tool ')),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='calls)')),
        ]
    )


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

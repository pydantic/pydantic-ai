from __future__ import annotations

from datetime import datetime, timezone

import pytest

from pydantic_ai.messages import (
    FinalResultEvent,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import ReplayStreamedResponse
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot


@pytest.fixture
def model_request_parameters() -> ModelRequestParameters:
    return ModelRequestParameters(
        function_tools=[],
        builtin_tools=[],
        output_mode='text',
        allow_text_output=True,
        output_tools=[],
        output_object=None,
    )


@pytest.fixture
def timestamp() -> datetime:
    return datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def usage() -> RequestUsage:
    return RequestUsage(input_tokens=10, output_tokens=20)


@pytest.fixture
def model_response(timestamp: datetime, usage: RequestUsage) -> ModelResponse:
    return ModelResponse(
        parts=[
            TextPart(content='Hello world'),
            ThinkingPart(content='Let me think about this'),
            ToolCallPart(tool_name='get_weather', args='{"city": "London"}', tool_call_id='call_1'),
        ],
        model_name='test-model',
        provider_name='test-provider',
        provider_url='https://test.example.com',
        timestamp=timestamp,
        usage=usage,
    )


async def test_replay_streamed_response_events(
    model_request_parameters: ModelRequestParameters, model_response: ModelResponse
) -> None:
    """Verify that ReplayStreamedResponse replays all part types as stream events."""
    stream = ReplayStreamedResponse(model_request_parameters, model_response)
    events = [event async for event in stream]

    assert events == snapshot(
        [
            PartStartEvent(index=0, part=TextPart(content='Hello world')),
            FinalResultEvent(tool_name=None, tool_call_id=None),
            PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='Hello world')),
            PartEndEvent(
                index=0,
                part=TextPart(content='Hello world'),
                next_part_kind='thinking',
            ),
            PartStartEvent(index=1, part=ThinkingPart(content='Let me think about this'), previous_part_kind='text'),
            PartDeltaEvent(index=1, delta=ThinkingPartDelta(content_delta='Let me think about this')),
            PartEndEvent(
                index=1,
                part=ThinkingPart(content='Let me think about this'),
                next_part_kind='tool-call',
            ),
            PartStartEvent(
                index=2,
                part=ToolCallPart(tool_name='get_weather', args='{"city": "London"}', tool_call_id='call_1'),
                previous_part_kind='thinking',
            ),
            PartDeltaEvent(index=2, delta=ToolCallPartDelta(args_delta='{"city": "London"}')),
            PartEndEvent(
                index=2,
                part=ToolCallPart(
                    tool_name='get_weather',
                    args='{"city": "London"}',
                    tool_call_id='call_1',
                ),
            ),
        ]
    )


async def test_replay_streamed_response_get(
    model_request_parameters: ModelRequestParameters, model_response: ModelResponse
) -> None:
    """Verify that get() returns the original ModelResponse."""
    stream = ReplayStreamedResponse(model_request_parameters, model_response)
    assert stream.get() is model_response


async def test_replay_streamed_response_usage(
    model_request_parameters: ModelRequestParameters,
    model_response: ModelResponse,
    usage: RequestUsage,
) -> None:
    """Verify that usage() delegates to the underlying response."""
    stream = ReplayStreamedResponse(model_request_parameters, model_response)
    assert stream.usage() == usage
    assert stream.usage().input_tokens == 10
    assert stream.usage().output_tokens == 20


async def test_replay_streamed_response_metadata(
    model_request_parameters: ModelRequestParameters,
    model_response: ModelResponse,
    timestamp: datetime,
) -> None:
    """Verify that metadata properties delegate to the underlying response."""
    stream = ReplayStreamedResponse(model_request_parameters, model_response)
    assert stream.model_name == 'test-model'
    assert stream.provider_name == 'test-provider'
    assert stream.provider_url == 'https://test.example.com'
    assert stream.timestamp == timestamp


async def test_replay_streamed_response_metadata_defaults(
    model_request_parameters: ModelRequestParameters,
) -> None:
    """Verify that model_name and provider_name return empty strings when the response has None."""
    response = ModelResponse(parts=[TextPart(content='hi')])
    stream = ReplayStreamedResponse(model_request_parameters, response)
    assert stream.model_name == ''
    assert stream.provider_name == ''
    assert stream.provider_url is None


async def test_replay_streamed_response_empty_text_part(
    model_request_parameters: ModelRequestParameters,
) -> None:
    """Verify that an empty TextPart emits a PartStartEvent but no PartDeltaEvent."""
    response = ModelResponse(parts=[TextPart(content='')])
    stream = ReplayStreamedResponse(model_request_parameters, response)
    events = [event async for event in stream]

    # Empty text: PartStartEvent, FinalResultEvent (from allow_text_output), PartEndEvent
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[1], FinalResultEvent)
    assert isinstance(events[2], PartEndEvent)


async def test_replay_streamed_response_empty_thinking_part(
    model_request_parameters: ModelRequestParameters,
) -> None:
    """Verify that an empty ThinkingPart emits a PartStartEvent but no PartDeltaEvent."""
    response = ModelResponse(parts=[ThinkingPart(content='')])
    stream = ReplayStreamedResponse(model_request_parameters, response)
    events = [event async for event in stream]

    assert len(events) == 2
    assert isinstance(events[0], PartStartEvent)
    assert isinstance(events[1], PartEndEvent)

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pytest
from tenacity import AsyncRetrying, RetryError, retry_if_exception_type, stop_after_attempt, wait_fixed
from throttled.asyncio import RateLimiterType, Throttled, rate_limiter, store

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseStreamEvent,
    PartDeltaEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.rate_limited import RateLimitedModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import Usage


class FailingModel(Model):
    """A model that fails a specified number of times before succeeding."""

    @property
    def system(self) -> str:
        return 'failing_system'

    @property
    def model_name(self) -> str:
        return 'failing_model'

    def __init__(self, fail_count: int = 2) -> None:
        self.fail_count = fail_count
        self.attempt_count = 0

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.attempt_count += 1
        if self.attempt_count <= self.fail_count:
            raise ValueError(f'Failing on attempt {self.attempt_count}')

        return ModelResponse(
            parts=[TextPart('success after retries')],
            usage=Usage(request_tokens=100, response_tokens=50),
            model_name=self.model_name,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        self.attempt_count += 1
        if self.attempt_count <= self.fail_count:
            raise ValueError(f'Failing on stream attempt {self.attempt_count}')

        yield SuccessResponseStream()


class SuccessResponseStream(StreamedResponse):
    def __init__(self) -> None:
        super().__init__()
        self._timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        self._usage = Usage(request_tokens=100, response_tokens=50)
        # First send a part start event
        text_part = TextPart('first part')
        yield PartStartEvent(index=0, part=text_part)
        # Then send a delta event
        delta = TextPartDelta(content_delta=' continued')
        yield PartDeltaEvent(index=0, delta=delta)

    @property
    def model_name(self) -> str:
        return 'success_model'

    @property
    def timestamp(self) -> datetime:
        return self._timestamp


class SimpleModel(Model):
    """A simple model that returns a fixed response."""

    @property
    def system(self) -> str:
        return 'simple_system'

    @property
    def model_name(self) -> str:
        return 'simple_model'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('simple response')],
            usage=Usage(request_tokens=10, response_tokens=20),
            model_name=self.model_name,
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> AsyncIterator[StreamedResponse]:
        yield SimpleResponseStream()


class SimpleResponseStream(StreamedResponse):
    def __init__(self) -> None:
        super().__init__()
        self._timestamp = datetime(2023, 1, 1, tzinfo=timezone.utc)

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        self._usage = Usage(request_tokens=10, response_tokens=20)
        # First send a part start event
        text_part = TextPart('stream part 1')
        yield PartStartEvent(index=0, part=text_part)
        # Then send a delta event for the same vendor part ID
        delta = TextPartDelta(content_delta=' and part 2')
        yield PartDeltaEvent(index=0, delta=delta)

    @property
    def model_name(self) -> str:
        return 'simple_model'

    @property
    def timestamp(self) -> datetime:
        return self._timestamp


pytestmark = [pytest.mark.anyio]


async def test_rate_limited_model_basic():
    """Test basic functionality of RateLimitedModel with a simple model with only retryer."""
    # Create a simple model wrapped with rate limiting but no retries
    simple_model = SimpleModel()
    retry_config = AsyncRetrying(
        stop=stop_after_attempt(1),  # No retries
        wait=wait_fixed(0.1),
    )

    rate_limited_model = RateLimitedModel(simple_model, retryer=retry_config)

    # Check properties are passed through
    assert rate_limited_model.system == 'simple_system'
    assert rate_limited_model.model_name == 'simple_model'

    # Check request works as expected
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart('system prompt'),
                UserPromptPart('user prompt'),
            ]
        ),
    ]

    response = await rate_limited_model.request(
        messages,
        model_settings=ModelSettings(temperature=0.7),
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    )

    # Check response is as expected
    assert response.model_name == 'simple_model'
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'simple response'
    assert response.usage.request_tokens == 10
    assert response.usage.response_tokens == 20


async def test_rate_limited_model_stream():
    """Test streaming functionality of RateLimitedModel."""
    simple_model = SimpleModel()
    retry_config = AsyncRetrying(
        stop=stop_after_attempt(1),  # No retries
        wait=wait_fixed(0.1),
    )

    rate_limited_model = RateLimitedModel(simple_model, retryer=retry_config)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('user prompt for streaming'),
            ]
        ),
    ]

    async with rate_limited_model.request_stream(
        messages,
        model_settings=ModelSettings(temperature=0.7),
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    ) as response_stream:
        events = [event async for event in response_stream]

        # Check events are as expected
        assert len(events) == 2
        assert isinstance(events[0], PartStartEvent)
        assert isinstance(events[0].part, TextPart)
        assert events[0].part.content == 'stream part 1'

        assert isinstance(events[1], PartDeltaEvent)
        assert isinstance(events[1].delta, TextPartDelta)
        assert events[1].delta.content_delta == ' and part 2'

        # Check usage
        response = response_stream.get()
        assert response.usage.request_tokens == 10
        assert response.usage.response_tokens == 20


async def test_rate_limited_model_retry():
    """Test retry functionality of RateLimitedModel with a failing model."""
    # Create a failing model that will succeed after 2 failures
    failing_model = FailingModel(fail_count=2)

    # Configure retrying to allow 3 attempts (original + 2 retries)
    retry_config = AsyncRetrying(
        retry=retry_if_exception_type(ValueError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
    )

    rate_limited_model = RateLimitedModel(failing_model, retryer=retry_config)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('user prompt that will eventually succeed'),
            ]
        ),
    ]

    # This should succeed after retries
    response = await rate_limited_model.request(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    )

    # Check response is as expected after retries
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'success after retries'
    assert failing_model.attempt_count == 3  # Original + 2 retries


async def test_rate_limited_model_stream_retry():
    """Test retry functionality of RateLimitedModel with streaming."""
    # Reset attempt counter and create a failing model that will succeed after 2 failures
    failing_model = FailingModel(fail_count=2)

    # Configure retrying to allow 3 attempts (original + 2 retries)
    retry_config = AsyncRetrying(
        retry=retry_if_exception_type(ValueError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
    )

    rate_limited_model = RateLimitedModel(failing_model, retryer=retry_config)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('user prompt for streaming with retries'),
            ]
        ),
    ]

    # This should succeed after retries
    async with rate_limited_model.request_stream(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    ) as response_stream:
        events = [event async for event in response_stream]

        # Check events exist
        assert len(events) > 0

        # Check retry count
        assert failing_model.attempt_count == 3  # Original + 2 retries


async def test_rate_limited_model_max_retries_exceeded():
    """Test behavior when max retries are exceeded."""
    # Create a failing model that will never succeed (fails 5 times)
    failing_model = FailingModel(fail_count=5)

    # Configure retrying to allow only 3 attempts (original + 2 retries)
    retry_config = AsyncRetrying(
        retry=retry_if_exception_type(ValueError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
    )

    rate_limited_model = RateLimitedModel(failing_model, retryer=retry_config)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('user prompt that will never succeed'),
            ]
        ),
    ]

    # This should fail after all retries are exhausted with a RetryError
    with pytest.raises(RetryError) as exc_info:
        await rate_limited_model.request(
            messages,
            model_settings=None,
            model_request_parameters=ModelRequestParameters(
                function_tools=[],
                allow_text_output=True,
                output_tools=[],
            ),
        )

    # The inner exception should be a ValueError
    assert 'ValueError' in str(exc_info.value)
    # And the failing model should have been called 3 times
    assert failing_model.attempt_count == 3  # Original + 2 retries


async def test_rate_limited_model_limiter_only():
    """Test RateLimitedModel with just a rate limiter (no retryer)."""
    # Create a simple model with rate limiting
    simple_model = SimpleModel()

    # Create a rate limiter - 10 requests per second to make tests fast
    throttle = Throttled(
        using=RateLimiterType.GCRA.value,
        quota=rate_limiter.per_sec(10, burst=10),
        store=store.MemoryStore(),
    )
    rate_limited_model = RateLimitedModel(simple_model, limiter=throttle)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('user prompt with rate limiting'),
            ]
        ),
    ]

    # Make a request
    response = await rate_limited_model.request(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    )

    # Check response is correct
    assert response.model_name == 'simple_model'
    assert len(response.parts) == 1
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'simple response'

    # Now test with streaming
    async with rate_limited_model.request_stream(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    ) as response_stream:
        events = [event async for event in response_stream]

        # Verify events
        assert len(events) == 2
        assert isinstance(events[0], PartStartEvent)
        assert isinstance(events[0].part, TextPart)
        assert events[0].part.content == 'stream part 1'


async def test_rate_limited_model_both_limiter_and_retryer():
    """Test RateLimitedModel with both a rate limiter and retryer."""
    # Create a failing model that succeeds after 2 retries
    failing_model = FailingModel(fail_count=2)

    # Configure the limiter and retryer
    throttle = Throttled(
        using=RateLimiterType.GCRA.value,
        quota=rate_limiter.per_sec(10, burst=10),
        store=store.MemoryStore(),
    )
    retry_config = AsyncRetrying(
        retry=retry_if_exception_type(ValueError),
        stop=stop_after_attempt(3),
        wait=wait_fixed(0.1),
    )

    rate_limited_model = RateLimitedModel(failing_model, limiter=throttle, retryer=retry_config)

    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('user prompt with rate limiting and retries'),
            ]
        ),
    ]

    # Make a request - should succeed after retries
    response = await rate_limited_model.request(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    )

    # Check response is correct
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'success after retries'

    # Verify the model was called the right number of times
    assert failing_model.attempt_count == 3

    # Reset for streaming test
    failing_model.attempt_count = 0

    # Test with streaming - should also succeed after retries
    async with rate_limited_model.request_stream(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    ) as response_stream:
        events = [event async for event in response_stream]

        # Verify events exist
        assert len(events) > 0

    # Verify the model was called the right number of times
    assert failing_model.attempt_count == 3


async def test_rate_limited_model_concurrent_requests():
    """Test RateLimitedModel with concurrent requests."""
    import time

    # Create several simple model instances
    simple_model = SimpleModel()

    # Create a real rate limiter that will allow 2 requests per second
    throttle = Throttled(
        using=RateLimiterType.GCRA.value,
        quota=rate_limiter.per_sec(2),  # 2 requests per second
        store=store.MemoryStore(),
    )

    rate_limited_model = RateLimitedModel(simple_model, limiter=throttle)

    # Create the message for all requests
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('concurrent request test'),
            ]
        ),
    ]

    # Make 5 sequential requests and measure time
    start_time = time.time()
    
    for i in range(5):
        response = await rate_limited_model.request(
            messages,
            model_settings=None,
            model_request_parameters=ModelRequestParameters(
                function_tools=[],
                allow_text_output=True,
                output_tools=[],
            ),
        )
        assert response.model_name == 'simple_model'
    
    total_time = time.time() - start_time
    
    # With 2 requests per second and 5 requests total, with GCRA algorithm:
    # - First 2 requests go immediately (burst allowed)
    # - Wait ~0.5s, next 2 requests go
    # - Wait ~0.5s, last request goes
    # Total should be at least 1 second
    assert total_time >= 1.0, f'Expected at least 1 second for 5 requests at 2/sec, but took {total_time}s'
    
    # But it shouldn't take too much longer (allow some margin for processing)
    assert total_time < 2.5, f'Expected less than 2.5 seconds for 5 requests at 2/sec, but took {total_time}s'


async def test_rate_limited_model_neither_limiter_nor_retryer():
    """Test RateLimitedModel with neither limiter nor retryer."""
    # Create a simple model
    simple_model = SimpleModel()

    # Create a rate limited model with no limiter or retryer
    rate_limited_model = RateLimitedModel(simple_model)

    # Check that it acts as a simple pass-through
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('simple pass-through test'),
            ]
        ),
    ]

    # Check normal request
    response = await rate_limited_model.request(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    )

    # Verify the response
    assert response.model_name == 'simple_model'
    assert isinstance(response.parts[0], TextPart)
    assert response.parts[0].content == 'simple response'

    # Check streaming request
    async with rate_limited_model.request_stream(
        messages,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(
            function_tools=[],
            allow_text_output=True,
            output_tools=[],
        ),
    ) as response_stream:
        events = [event async for event in response_stream]

        # Verify events
        assert len(events) == 2
        assert isinstance(events[0], PartStartEvent)
        assert isinstance(events[0].part, TextPart)
        assert events[0].part.content == 'stream part 1'

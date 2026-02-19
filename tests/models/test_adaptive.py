from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import pytest

from pydantic_ai import Agent, ModelHTTPError, ModelResponse, TextPart
from pydantic_ai.models.adaptive import AdaptiveContext, AdaptiveModel
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


def success_response(messages, agent_info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('success')])


def failure_response(messages, agent_info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=500, model_name='test-function-model', body={'error': 'test error'})


success_model = FunctionModel(success_response)
failure_model = FunctionModel(failure_response)


async def test_basic_success():
    """Test that adaptive model works with first model succeeding."""
    models = [success_model]

    def selector(ctx: AdaptiveContext) -> FunctionModel:
        # First attempt - use first model
        if not ctx.attempts:
            return models[0]
        raise RuntimeError('Should not retry on success')

    adaptive = AdaptiveModel(selector=selector)
    agent = Agent(model=adaptive)
    result = await agent.run('hello')
    assert result.output == 'success'


async def test_fallback_to_second_model():
    """Test fallback from failing model to success model."""
    models = [failure_model, success_model]

    def selector(ctx: AdaptiveContext) -> FunctionModel:
        if not ctx.attempts:
            # Try first model
            return models[0]
        elif len(ctx.attempts) == 1:
            # First failed, try second
            return models[1]
        raise RuntimeError('All models exhausted')

    adaptive = AdaptiveModel(selector=selector)
    agent = Agent(model=adaptive)
    result = await agent.run('hello')
    assert result.output == 'success'


async def test_retry_same_model():
    """Test retrying the same model."""
    call_count = 0

    def counting_response(messages, agent_info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ModelHTTPError(status_code=500, model_name='test', body={'error': 'first attempt'})
        return ModelResponse(parts=[TextPart('success on retry')])

    retry_model = FunctionModel(counting_response)
    models = [retry_model]

    def selector(ctx: AdaptiveContext) -> FunctionModel:
        if len(ctx.attempts) < 2:
            # Retry same model up to 2 times
            return models[0]
        raise RuntimeError('Max retries exceeded')

    adaptive = AdaptiveModel(selector=selector)
    agent = Agent(model=adaptive)
    result = await agent.run('hello')
    assert result.output == 'success on retry'
    assert call_count == 2


# test_max_attempts removed - max_attempts feature was removed from API
# Users should implement their own limits in the selector


async def test_selector_raises_exception():
    """Test that selector raising exception stops the loop."""

    def selector(ctx: AdaptiveContext) -> FunctionModel:
        raise RuntimeError('No suitable model')  # Immediately give up

    adaptive = AdaptiveModel(selector=selector)
    agent = Agent(model=adaptive)

    with pytest.raises(Exception) as exc_info:
        await agent.run('hello')

    assert 'selector raised an exception' in str(exc_info.value)


async def test_attempts_history():
    """Test that attempts history is correctly tracked."""
    attempts_log = []
    models = [failure_model, success_model]

    def selector(ctx: AdaptiveContext) -> FunctionModel:
        attempts_log.append(len(ctx.attempts))

        if not ctx.attempts:
            return models[0]  # Try failure model
        elif len(ctx.attempts) == 1:
            # Check first attempt was recorded
            assert ctx.attempts[0].model == models[0]
            assert ctx.attempts[0].exception is not None
            assert isinstance(ctx.attempts[0].exception, ModelHTTPError)
            assert ctx.attempts[0].duration.total_seconds() > 0
            return models[1]  # Try success model
        raise RuntimeError('All models exhausted')

    adaptive = AdaptiveModel(selector=selector)
    agent = Agent(model=adaptive)
    result = await agent.run('hello')

    assert result.output == 'success'
    assert attempts_log == [0, 1]  # Called with 0 attempts, then 1 attempt


async def test_async_selector():
    """Test that async selectors work."""
    models = [success_model]

    async def async_selector(ctx: AdaptiveContext) -> FunctionModel:
        await asyncio.sleep(0.01)  # Small async delay
        if not ctx.attempts:
            return models[0]
        raise RuntimeError('Should not retry')

    adaptive = AdaptiveModel(selector=async_selector)
    agent = Agent(model=adaptive)
    result = await agent.run('hello')
    assert result.output == 'success'


async def test_async_selector_with_backoff():
    """Test exponential backoff with async selector."""
    call_times = []

    def time_tracking_response(messages, agent_info: AgentInfo) -> ModelResponse:
        call_times.append(time.time())
        if len(call_times) < 3:
            raise ModelHTTPError(status_code=500, model_name='test', body={'error': 'retry'})
        return ModelResponse(parts=[TextPart('success')])

    retry_model = FunctionModel(time_tracking_response)
    models = [retry_model]

    async def backoff_selector(ctx: AdaptiveContext) -> FunctionModel:
        if ctx.attempts:
            # Exponential backoff: 0.01s, 0.02s
            delay = 0.01 * (2 ** (len(ctx.attempts) - 1))
            await asyncio.sleep(delay)

        if len(ctx.attempts) < 3:
            return models[0]
        raise RuntimeError('Max retries exceeded')

    adaptive = AdaptiveModel(selector=backoff_selector)
    agent = Agent(model=adaptive)
    result = await agent.run('hello')

    assert result.output == 'success'
    assert len(call_times) == 3

    # Verify backoff delays
    if len(call_times) >= 2:
        delay1 = call_times[1] - call_times[0]
        assert delay1 >= 0.01  # At least 10ms

    if len(call_times) >= 3:
        delay2 = call_times[2] - call_times[1]
        assert delay2 >= 0.02  # At least 20ms


async def test_context_has_correct_fields():
    """Test that AdaptiveContext has all expected fields."""
    models = [success_model]

    def selector(ctx: AdaptiveContext) -> FunctionModel:
        assert hasattr(ctx, 'state')
        assert hasattr(ctx, 'attempts')
        assert hasattr(ctx, 'messages')
        assert hasattr(ctx, 'model_settings')
        assert hasattr(ctx, 'model_request_parameters')

        # Fields that were removed
        assert not hasattr(ctx, 'models')  # Models are in closure now
        assert not hasattr(ctx, 'attempt_number')  # Use len(attempts) instead
        assert not hasattr(ctx, 'run_context')  # Replaced by state

        assert len(ctx.attempts) == 0  # First attempt
        assert len(ctx.messages) > 0

        return models[0]

    adaptive = AdaptiveModel(selector=selector)
    agent = Agent(model=adaptive)
    result = await agent.run('hello')
    assert result.output == 'success'


async def test_with_state():
    """Test that state works correctly with selector."""

    @dataclass
    class MyState:
        counter: int = 0

    models_list = []

    async def stream_response(messages, agent_info: AgentInfo) -> AsyncIterator[str]:
        yield 'success'

    stream_model = FunctionModel(stream_function=stream_response)
    models_list.append(stream_model)

    def selector(ctx: AdaptiveContext[MyState]) -> FunctionModel:
        # Access state
        assert isinstance(ctx.state, MyState)
        ctx.state.counter += 1

        if not ctx.attempts:
            return models_list[0]
        raise RuntimeError('Should not retry')

    from collections.abc import AsyncIterator

    my_state = MyState()
    adaptive = AdaptiveModel(selector=selector, state=my_state)
    agent = Agent(model=adaptive)

    async with agent.run_stream('hello') as result:
        text = await result.get_output()

    assert text == 'success'
    assert my_state.counter == 1  # Selector was called once


async def test_streaming():
    """Test that streaming works with adaptive model."""
    from collections.abc import AsyncIterator

    async def stream_response(messages, agent_info: AgentInfo) -> AsyncIterator[str]:
        yield 'hello '
        yield 'world'

    stream_model = FunctionModel(stream_function=stream_response)
    models = [stream_model]

    def selector(ctx: AdaptiveContext) -> FunctionModel:
        return models[0]

    adaptive = AdaptiveModel(selector=selector)
    agent = Agent(model=adaptive)

    async with agent.run_stream('test') as result:
        text = await result.get_output()
        assert text == 'hello world'


# test_empty_models_list removed - models parameter was removed from API


async def test_model_name():
    """Test that model_name is correctly formatted."""
    adaptive = AdaptiveModel(selector=lambda ctx: success_model)
    assert adaptive.model_name == 'adaptive'

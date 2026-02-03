"""Tests for concurrency limiting functionality."""

# pyright: reportPrivateUsage=false, reportAttributeAccessIssue=false, reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import asyncio
import importlib.util
from typing import TYPE_CHECKING, Any

import pytest

from pydantic_ai import Agent, ConcurrencyLimit, ConcurrencyLimiter, ConcurrencyLimitExceeded
from pydantic_ai.concurrency import get_concurrency_context
from pydantic_ai.models.test import TestModel

if TYPE_CHECKING:
    from logfire.testing import CaptureLogfire

logfire_installed = importlib.util.find_spec('logfire') is not None

pytestmark = pytest.mark.anyio


class TestConcurrencyLimiter:
    """Tests for the ConcurrencyLimiter class."""

    async def test_basic_acquisition(self):
        """Test that limiter limits concurrent access."""
        limiter = ConcurrencyLimiter(max_running=2)
        acquired: list[int] = []

        async def acquire_and_hold(id: int, hold_time: float):
            async with get_concurrency_context(limiter, 'test'):
                acquired.append(id)
                await asyncio.sleep(hold_time)

        # Start 3 tasks with limit of 2
        tasks = [asyncio.create_task(acquire_and_hold(i, 0.1)) for i in range(3)]
        await asyncio.sleep(0.05)
        assert len(acquired) == 2  # Only 2 can proceed
        await asyncio.gather(*tasks)
        assert len(acquired) == 3

    async def test_nowait_acquisition(self):
        """Test that immediate acquisition works."""
        limiter = ConcurrencyLimiter(max_running=10)
        # With high limit, should acquire immediately
        async with get_concurrency_context(limiter, 'test'):
            pass  # No waiting

    async def test_waiting_count_tracking(self):
        """Test that waiting_count is accurately tracked."""
        limiter = ConcurrencyLimiter(max_running=1)
        started = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with get_concurrency_context(limiter, 'test'):
                started.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await started.wait()

        # Now limiter is held, check waiting count as we add waiters
        assert limiter.waiting_count == 0

        async def waiter():
            async with get_concurrency_context(limiter, 'test'):
                pass

        waiter_tasks = [asyncio.create_task(waiter()) for _ in range(3)]
        await asyncio.sleep(0.01)
        assert limiter.waiting_count == 3

        release.set()
        await task
        await asyncio.gather(*waiter_tasks)
        assert limiter.waiting_count == 0

    async def test_backpressure_raises(self):
        """Test that exceeding max_queued raises ConcurrencyLimitExceeded."""
        limiter = ConcurrencyLimiter(max_running=1, max_queued=2)
        hold = asyncio.Event()

        async def holder():
            async with get_concurrency_context(limiter, 'test'):
                await hold.wait()

        # Fill the running slot
        task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        # Fill the queue (2 allowed)
        waiter1 = asyncio.create_task(holder())
        waiter2 = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        # This should raise - queue is full
        with pytest.raises(ConcurrencyLimitExceeded):
            async with get_concurrency_context(limiter, 'test'):
                pass

        hold.set()
        await asyncio.gather(task, waiter1, waiter2)

    async def test_backpressure_race_condition(self):
        """Test that max_queued is enforced atomically under concurrent load.

        This test verifies the fix for a race condition where multiple tasks could
        simultaneously pass the max_queued check before any of them actually started
        waiting on the limiter.
        """
        limiter = ConcurrencyLimiter(max_running=1, max_queued=1)
        hold = asyncio.Event()
        started = asyncio.Event()

        async def holder():
            async with get_concurrency_context(limiter, 'holder'):
                started.set()
                await hold.wait()

        # Fill the running slot and wait for it to be held
        task = asyncio.create_task(holder())
        await started.wait()

        # Now launch multiple tasks simultaneously that all try to queue.
        # With max_queued=1, exactly one should succeed in queuing.
        num_concurrent = 5
        results: list[str] = []
        barrier = asyncio.Barrier(num_concurrent)

        async def try_acquire(idx: int):
            # Use barrier to ensure all tasks try to acquire at the same time
            await barrier.wait()
            try:
                async with get_concurrency_context(limiter, f'task-{idx}'):
                    results.append(f'acquired-{idx}')
            except ConcurrencyLimitExceeded:
                results.append(f'rejected-{idx}')

        # Launch all tasks simultaneously
        tasks = [asyncio.create_task(try_acquire(i)) for i in range(num_concurrent)]
        await asyncio.sleep(0.1)  # Give tasks time to hit the barrier and try to acquire

        # Release the holder
        hold.set()
        await asyncio.gather(task, *tasks)

        # Verify: exactly one task should have been allowed to queue and acquire
        # The rest should have been rejected
        acquired = [r for r in results if r.startswith('acquired-')]
        rejected = [r for r in results if r.startswith('rejected-')]
        assert len(acquired) == 1, f'Expected exactly 1 acquired, got {len(acquired)}: {acquired}'
        assert len(rejected) == num_concurrent - 1, f'Expected {num_concurrent - 1} rejected, got {len(rejected)}'

    async def test_from_int_limit(self):
        """Test creating from simple int."""
        limiter = ConcurrencyLimiter.from_limit(5)
        assert limiter.max_running == 5
        assert limiter._max_queued is None

    async def test_from_limiter_config(self):
        """Test creating from ConcurrencyLimit."""
        config = ConcurrencyLimit(max_running=5, max_queued=10)
        limiter = ConcurrencyLimiter.from_limit(config)
        assert limiter.max_running == 5
        assert limiter._max_queued == 10

    async def test_properties(self):
        """Test the various properties of ConcurrencyLimiter."""
        limiter = ConcurrencyLimiter(max_running=5, name='test-limiter')
        assert limiter.max_running == 5
        assert limiter.running_count == 0
        assert limiter.available_count == 5
        assert limiter.waiting_count == 0
        assert limiter.name == 'test-limiter'

        # After acquiring one slot
        await limiter.acquire('test')
        assert limiter.running_count == 1
        assert limiter.available_count == 4
        limiter.release()
        assert limiter.running_count == 0
        assert limiter.available_count == 5


class TestGetConcurrencyContext:
    """Tests for the get_concurrency_context helper."""

    async def test_returns_context_when_provided(self):
        """Test that get_concurrency_context returns a working context."""
        limiter = ConcurrencyLimiter(max_running=1)
        async with get_concurrency_context(limiter, 'test'):
            pass  # Should acquire and release

    async def test_returns_null_context_when_none(self):
        """Test that get_concurrency_context returns a no-op context when None."""
        async with get_concurrency_context(None, 'test'):
            pass  # Should be a no-op


class TestAgentConcurrency:
    """Tests for agent-level concurrency limiting."""

    async def test_agent_concurrency_limit(self):
        """Test that agent respects max_concurrency."""
        agent = Agent(TestModel(), max_concurrency=2)
        running = 0
        max_running = 0
        lock = asyncio.Lock()

        @agent.tool_plain
        async def slow_tool() -> str:
            nonlocal running, max_running
            async with lock:
                running += 1
                max_running = max(max_running, running)
            await asyncio.sleep(0.1)
            async with lock:
                running -= 1
            return 'done'

        results = await asyncio.gather(
            *[agent.run('call slow_tool', model=TestModel(call_tools=['slow_tool'])) for _ in range(5)]
        )

        assert max_running <= 2
        assert len(results) == 5

    async def test_agent_concurrency_backpressure(self):
        """Test that agent raises when queue exceeds max_queued."""
        agent = Agent(TestModel(), max_concurrency=ConcurrencyLimit(max_running=1, max_queued=1))
        hold = asyncio.Event()

        @agent.tool_plain
        async def hold_tool() -> str:
            await hold.wait()
            return 'done'

        # Start 2 runs (1 running + 1 queued = at limit)
        task1 = asyncio.create_task(agent.run('x', model=TestModel(call_tools=['hold_tool'])))
        task2 = asyncio.create_task(agent.run('x', model=TestModel(call_tools=['hold_tool'])))
        await asyncio.sleep(0.05)

        # Third should raise
        with pytest.raises(ConcurrencyLimitExceeded):
            await agent.run('x', model=TestModel(call_tools=['hold_tool']))

        hold.set()
        await asyncio.gather(task1, task2)

    async def test_agent_no_limit_by_default(self):
        """Test that agents have no concurrency limit by default."""
        agent = Agent(TestModel())
        assert agent._concurrency_limiter is None

    async def test_agent_with_int_concurrency(self):
        """Test that agent accepts int for max_concurrency."""
        agent = Agent(TestModel(), max_concurrency=5)
        assert agent._concurrency_limiter is not None
        assert agent._concurrency_limiter.max_running == 5
        assert agent._concurrency_limiter._max_queued is None

    async def test_agent_with_limiter_concurrency(self):
        """Test that agent accepts ConcurrencyLimit for max_concurrency."""
        agent = Agent(TestModel(), max_concurrency=ConcurrencyLimit(max_running=5, max_queued=10))
        assert agent._concurrency_limiter is not None
        assert agent._concurrency_limiter.max_running == 5
        assert agent._concurrency_limiter._max_queued == 10


class TestConcurrencyLimitedModel:
    """Tests for the ConcurrencyLimitedModel wrapper."""

    async def test_basic_concurrency_limit(self):
        """Test that ConcurrencyLimitedModel limits concurrent requests."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        request_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        base_model = TestModel()
        original_request = TestModel.request.__get__(base_model)

        async def tracking_request(*args: Any, **kwargs: Any):
            nonlocal request_count, max_concurrent
            async with lock:
                request_count += 1
                max_concurrent = max(max_concurrent, request_count)
            try:
                await asyncio.sleep(0.1)  # Simulate slow request
                return await original_request(*args, **kwargs)
            finally:
                async with lock:
                    request_count -= 1

        base_model.request = tracking_request

        model = ConcurrencyLimitedModel(base_model, limiter=2)
        agent = Agent(model)

        await asyncio.gather(*[agent.run(f'prompt {i}') for i in range(5)])

        assert max_concurrent <= 2

    async def test_with_int_limiter(self):
        """Test ConcurrencyLimitedModel with int limiter."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        model = ConcurrencyLimitedModel(TestModel(), limiter=5)
        assert model._limiter.max_running == 5
        assert model._limiter._max_queued is None

    async def test_with_concurrency_limit(self):
        """Test ConcurrencyLimitedModel with ConcurrencyLimit."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        model = ConcurrencyLimitedModel(TestModel(), limiter=ConcurrencyLimit(max_running=5, max_queued=10))
        assert model._limiter.max_running == 5
        assert model._limiter._max_queued == 10

    async def test_with_shared_limiter(self):
        """Test ConcurrencyLimitedModel with shared ConcurrencyLimiter."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        shared_limiter = ConcurrencyLimiter(max_running=3, name='shared-pool')
        model1 = ConcurrencyLimitedModel(TestModel(), limiter=shared_limiter)
        model2 = ConcurrencyLimitedModel(TestModel(), limiter=shared_limiter)

        # Both models should share the same limiter
        assert model1._limiter is model2._limiter
        assert model1._limiter.name == 'shared-pool'

    async def test_shared_limiter_limits_across_models(self):
        """Test that shared limiter limits concurrent requests across multiple models."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        request_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        shared_limiter = ConcurrencyLimiter(max_running=2)

        def create_tracking_model():
            base_model = TestModel()
            original_request = TestModel.request.__get__(base_model)

            async def tracking_request(*args: Any, **kwargs: Any):
                nonlocal request_count, max_concurrent
                async with lock:
                    request_count += 1
                    max_concurrent = max(max_concurrent, request_count)
                try:
                    await asyncio.sleep(0.1)
                    return await original_request(*args, **kwargs)
                finally:
                    async with lock:
                        request_count -= 1

            base_model.request = tracking_request
            return ConcurrencyLimitedModel(base_model, limiter=shared_limiter)

        model1 = create_tracking_model()
        model2 = create_tracking_model()

        agent1 = Agent(model1)
        agent2 = Agent(model2)

        # Run 3 requests on each agent (6 total), but limit is 2
        await asyncio.gather(
            *[agent1.run(f'prompt {i}') for i in range(3)],
            *[agent2.run(f'prompt {i}') for i in range(3)],
        )

        # Should never exceed 2 concurrent requests across both models
        assert max_concurrent <= 2

    async def test_limit_model_concurrency_helper(self):
        """Test the limit_model_concurrency helper function."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel, limit_model_concurrency

        # With limiter
        model = limit_model_concurrency(TestModel(), limiter=5)
        assert isinstance(model, ConcurrencyLimitedModel)

        # Without limiter (returns original)
        base_model = TestModel()
        model = limit_model_concurrency(base_model, limiter=None)
        assert model is base_model

        # With model name string
        model = limit_model_concurrency('test', limiter=5)
        assert isinstance(model, ConcurrencyLimitedModel)

    async def test_model_properties_delegated(self):
        """Test that model properties are properly delegated to wrapped model."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        base_model = TestModel(model_name='custom-test')
        model = ConcurrencyLimitedModel(base_model, limiter=5)

        assert model.model_name == 'custom-test'
        assert model.system == 'test'


class TestAgentWithSharedLimiter:
    """Tests for agent with shared ConcurrencyLimiter."""

    async def test_agent_with_shared_limiter(self):
        """Test that agents can share a ConcurrencyLimiter."""
        shared_limiter = ConcurrencyLimiter(max_running=2)

        agent1 = Agent(TestModel(), max_concurrency=shared_limiter)
        agent2 = Agent(TestModel(), max_concurrency=shared_limiter)

        # Both agents should share the same limiter
        assert agent1._concurrency_limiter is agent2._concurrency_limiter


class TestConcurrencyLimiterName:
    """Tests for ConcurrencyLimiter name parameter."""

    async def test_limiter_with_name(self):
        """Test that limiter name is properly set and accessible."""
        limiter = ConcurrencyLimiter(max_running=5, name='my-limiter')
        assert limiter.name == 'my-limiter'

    async def test_limiter_without_name(self):
        """Test that limiter name is None by default."""
        limiter = ConcurrencyLimiter(max_running=5)
        assert limiter.name is None

    async def test_from_limit_with_name(self):
        """Test creating limiter from limit with name."""
        limiter = ConcurrencyLimiter.from_limit(5, name='my-limit')
        assert limiter.name == 'my-limit'
        assert limiter.max_running == 5

    @pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
    async def test_named_limiter_waiting_adds_limiter_name_attribute(self, capfire: CaptureLogfire):
        """Test that waiting with a named limiter adds limiter_name to span attributes."""
        limiter = ConcurrencyLimiter(max_running=1, name='test-pool')
        hold = asyncio.Event()

        async def holder():
            async with get_concurrency_context(limiter, 'test-source'):
                await hold.wait()

        # Start holder to occupy the slot
        task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        # Start a waiter - this will trigger the span with limiter_name attribute
        async def waiter():
            async with get_concurrency_context(limiter, 'test-source'):
                pass

        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)

        hold.set()
        await task
        await waiter_task

        # Verify span was created with the correct attributes
        spans = capfire.exporter.exported_spans_as_dict()
        assert len(spans) == 1
        span = spans[0]
        assert span['name'] == 'waiting for test-pool concurrency'
        attrs = span['attributes']
        assert attrs['source'] == 'test-source'
        assert attrs['limiter_name'] == 'test-pool'
        assert attrs['max_running'] == 1
        assert 'waiting_count' in attrs

    @pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
    async def test_unnamed_limiter_waiting_uses_source_in_span_name(self, capfire: CaptureLogfire):
        """Test that waiting without a limiter name uses source for span name."""
        limiter = ConcurrencyLimiter(max_running=1)  # No name
        hold = asyncio.Event()

        async def holder():
            async with get_concurrency_context(limiter, 'model:gpt-4'):
                await hold.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        async def waiter():
            async with get_concurrency_context(limiter, 'model:gpt-4'):
                pass

        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)

        hold.set()
        await task
        await waiter_task

        # Verify span uses source in name when limiter has no name
        spans = capfire.exporter.exported_spans_as_dict()
        assert len(spans) == 1
        span = spans[0]
        assert span['name'] == 'waiting for model:gpt-4 concurrency'
        attrs = span['attributes']
        assert attrs['source'] == 'model:gpt-4'
        assert 'limiter_name' not in attrs  # Should not be present when name is None
        assert attrs['max_running'] == 1

    @pytest.mark.skipif(not logfire_installed, reason='logfire not installed')
    async def test_limiter_with_max_queued_includes_attribute_in_span(self, capfire: CaptureLogfire):
        """Test that max_queued is included in span attributes when set."""
        limiter = ConcurrencyLimiter(max_running=1, max_queued=5, name='queued-pool')
        hold = asyncio.Event()

        async def holder():
            async with get_concurrency_context(limiter, 'test'):
                await hold.wait()

        task = asyncio.create_task(holder())
        await asyncio.sleep(0.01)

        async def waiter():
            async with get_concurrency_context(limiter, 'test'):
                pass

        waiter_task = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)

        hold.set()
        await task
        await waiter_task

        # Verify max_queued is in span attributes
        spans = capfire.exporter.exported_spans_as_dict()
        assert len(spans) == 1
        attrs = spans[0]['attributes']
        assert attrs['max_queued'] == 5


class TestConcurrencyLimiterWithTracer:
    """Tests for ConcurrencyLimiter with custom tracer."""

    async def test_custom_tracer_is_stored(self):
        """Test that custom tracer is stored and returned by _get_tracer."""
        from opentelemetry.trace import NoOpTracer

        custom_tracer = NoOpTracer()
        limiter = ConcurrencyLimiter(max_running=5, tracer=custom_tracer)

        # Verify the tracer is stored and returned
        assert limiter._get_tracer() is custom_tracer

    async def test_from_limit_with_tracer(self):
        """Test that from_limit passes tracer to the created limiter."""
        from opentelemetry.trace import NoOpTracer

        custom_tracer = NoOpTracer()
        limiter = ConcurrencyLimiter.from_limit(5, tracer=custom_tracer)
        assert limiter._get_tracer() is custom_tracer


class TestConcurrencyLimitedModelMethods:
    """Tests for ConcurrencyLimitedModel count_tokens and request_stream methods."""

    async def test_count_tokens(self):
        """Test that count_tokens delegates to wrapped model with concurrency limiting."""
        from unittest.mock import AsyncMock

        from pydantic_ai.models import ModelRequestParameters
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
        from pydantic_ai.usage import RequestUsage

        base_model = TestModel()
        # Mock count_tokens to return a value
        base_model.count_tokens = AsyncMock(return_value=RequestUsage())
        model = ConcurrencyLimitedModel(base_model, limiter=5)

        # count_tokens should delegate to wrapped model
        usage = await model.count_tokens([], None, ModelRequestParameters())
        assert usage is not None
        base_model.count_tokens.assert_called_once()

    async def test_request_stream(self):
        """Test that request_stream is called with concurrency limiting."""
        from pydantic_ai.models import ModelRequestParameters
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        base_model = TestModel()
        model = ConcurrencyLimitedModel(base_model, limiter=5)

        # request_stream should work
        async with model.request_stream([], None, ModelRequestParameters()) as stream:
            # Consume the stream
            async for _ in stream:
                pass

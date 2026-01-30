"""Tests for concurrency limiting functionality."""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pydantic_ai import Agent, ConcurrencyLimiter, ConcurrencyLimitExceeded
from pydantic_ai.concurrency import RecordingSemaphore, get_concurrency_context
from pydantic_ai.models.test import TestModel

pytestmark = pytest.mark.anyio


class TestRecordingSemaphore:
    """Tests for the RecordingSemaphore class."""

    async def test_basic_acquisition(self):
        """Test that semaphore limits concurrent access."""
        sem = RecordingSemaphore(max_running=2)
        acquired: list[int] = []

        async def acquire_and_hold(id: int, hold_time: float):
            async with get_concurrency_context(sem, 'test'):
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
        sem = RecordingSemaphore(max_running=10)
        # With high limit, should acquire immediately
        async with get_concurrency_context(sem, 'test'):
            pass  # No waiting

    async def test_waiting_count_tracking(self):
        """Test that waiting_count is accurately tracked."""
        sem = RecordingSemaphore(max_running=1)
        started = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with get_concurrency_context(sem, 'test'):
                started.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await started.wait()

        # Now sem is held, check waiting count as we add waiters
        assert sem.waiting_count == 0

        async def waiter():
            async with get_concurrency_context(sem, 'test'):
                pass

        waiter_tasks = [asyncio.create_task(waiter()) for _ in range(3)]
        await asyncio.sleep(0.01)
        assert sem.waiting_count == 3

        release.set()
        await task
        await asyncio.gather(*waiter_tasks)
        assert sem.waiting_count == 0

    async def test_backpressure_raises(self):
        """Test that exceeding max_queued raises ConcurrencyLimitExceeded."""
        sem = RecordingSemaphore(max_running=1, max_queued=2)
        hold = asyncio.Event()

        async def holder():
            async with get_concurrency_context(sem, 'test'):
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
            async with get_concurrency_context(sem, 'test'):
                pass

        hold.set()
        await asyncio.gather(task, waiter1, waiter2)

    async def test_from_int_limit(self):
        """Test creating from simple int."""
        sem = RecordingSemaphore.from_limit(5)
        assert sem._max_running == 5
        assert sem._max_queued is None

    async def test_from_limiter_config(self):
        """Test creating from ConcurrencyLimiter."""
        config = ConcurrencyLimiter(max_running=5, max_queued=10)
        sem = RecordingSemaphore.from_limit(config)
        assert sem._max_running == 5
        assert sem._max_queued == 10


class TestGetConcurrencyContext:
    """Tests for the get_concurrency_context helper."""

    async def test_returns_context_when_provided(self):
        """Test that get_concurrency_context returns a working context."""
        sem = RecordingSemaphore(max_running=1)
        async with get_concurrency_context(sem, 'test'):
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
        agent = Agent(TestModel(), max_concurrency=ConcurrencyLimiter(max_running=1, max_queued=1))
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
        assert agent._concurrency_limiter._max_running == 5
        assert agent._concurrency_limiter._max_queued is None

    async def test_agent_with_limiter_concurrency(self):
        """Test that agent accepts ConcurrencyLimiter for max_concurrency."""
        agent = Agent(TestModel(), max_concurrency=ConcurrencyLimiter(max_running=5, max_queued=10))
        assert agent._concurrency_limiter is not None
        assert agent._concurrency_limiter._max_running == 5
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
        assert model._semaphore._max_running == 5
        assert model._semaphore._max_queued is None

    async def test_with_concurrency_limiter(self):
        """Test ConcurrencyLimitedModel with ConcurrencyLimiter."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        model = ConcurrencyLimitedModel(TestModel(), limiter=ConcurrencyLimiter(max_running=5, max_queued=10))
        assert model._semaphore._max_running == 5
        assert model._semaphore._max_queued == 10

    async def test_with_shared_semaphore(self):
        """Test ConcurrencyLimitedModel with shared RecordingSemaphore."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        shared_semaphore = RecordingSemaphore(max_running=3, name='shared-pool')
        model1 = ConcurrencyLimitedModel(TestModel(), limiter=shared_semaphore)
        model2 = ConcurrencyLimitedModel(TestModel(), limiter=shared_semaphore)

        # Both models should share the same semaphore
        assert model1._semaphore is model2._semaphore
        assert model1._semaphore.name == 'shared-pool'

    async def test_shared_semaphore_limits_across_models(self):
        """Test that shared semaphore limits concurrent requests across multiple models."""
        from pydantic_ai.models.concurrency import ConcurrencyLimitedModel

        request_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        shared_semaphore = RecordingSemaphore(max_running=2)

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
            return ConcurrencyLimitedModel(base_model, limiter=shared_semaphore)

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


class TestAgentWithSharedSemaphore:
    """Tests for agent with shared RecordingSemaphore."""

    async def test_agent_with_shared_semaphore(self):
        """Test that agents can share a RecordingSemaphore."""
        shared_semaphore = RecordingSemaphore(max_running=2)

        agent1 = Agent(TestModel(), max_concurrency=shared_semaphore)
        agent2 = Agent(TestModel(), max_concurrency=shared_semaphore)

        # Both agents should share the same semaphore
        assert agent1._concurrency_limiter is agent2._concurrency_limiter


class TestRecordingSemaphoreName:
    """Tests for RecordingSemaphore name parameter."""

    async def test_semaphore_with_name(self):
        """Test that semaphore name is properly set and accessible."""
        sem = RecordingSemaphore(max_running=5, name='my-semaphore')
        assert sem.name == 'my-semaphore'

    async def test_semaphore_without_name(self):
        """Test that semaphore name is None by default."""
        sem = RecordingSemaphore(max_running=5)
        assert sem.name is None

    async def test_from_limit_with_name(self):
        """Test creating semaphore from limit with name."""
        sem = RecordingSemaphore.from_limit(5, name='my-limit')
        assert sem.name == 'my-limit'
        assert sem._max_running == 5

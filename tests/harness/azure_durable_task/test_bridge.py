from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task

    from pydantic_ai_harness.azure_durable_task import run_durable

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


@pytest.fixture
def orchestration_context(durable_worker: InMemoryWorker) -> MagicMock:
    context = MagicMock(spec=task.OrchestrationContext)
    context.current_utc_datetime = durable_worker.now
    return context


@pytest.mark.parametrize('finish_background', [False, True])
def test_returning_body_does_not_drain_rescheduling_tasks(
    orchestration_context: MagicMock, finish_background: bool
) -> None:
    iterations: list[int] = []
    cleanup: list[str] = []
    background_tasks: list[asyncio.Task[None]] = []
    started = asyncio.Event()
    budget = 8

    async def background() -> None:
        try:
            # A finite budget makes starvation fail instead of hanging the test.
            for index in range(budget):
                iterations.append(index)
                started.set()
                await asyncio.sleep(0)
        finally:
            await asyncio.sleep(0)
            cleanup.append('closed')

    async def body() -> str:
        pending = asyncio.create_task(background())
        background_tasks.append(pending)
        if finish_background:
            await pending
        else:
            await started.wait()
        return 'done'

    with pytest.raises(StopIteration) as completed:
        next(run_durable(orchestration_context, body))
    assert completed.value.value == 'done'
    assert len(background_tasks) == 1
    assert background_tasks[0].done()
    assert background_tasks[0].cancelled() is (not finish_background)
    assert cleanup == ['closed']
    if finish_background:
        assert len(iterations) == budget
    else:
        assert 0 < len(iterations) < budget
    orchestration_context.create_timer.assert_not_called()


@pytest.mark.parametrize('finish_callbacks', [False, True])
def test_returning_body_does_not_drain_rescheduling_callbacks(
    orchestration_context: MagicMock, finish_callbacks: bool
) -> None:
    iterations: list[int] = []
    loops: list[asyncio.AbstractEventLoop] = []
    budget = 8

    async def body() -> str:
        loop = asyncio.get_running_loop()
        loops.append(loop)
        finished = asyncio.Event()

        def repeat() -> None:
            iterations.append(len(iterations))
            if len(iterations) < budget:
                loop.call_soon(repeat)
            else:
                finished.set()

        loop.call_soon(repeat)
        if finish_callbacks:
            await finished.wait()
        return 'done'

    with pytest.raises(StopIteration) as completed:
        next(run_durable(orchestration_context, body))
    assert completed.value.value == 'done'
    assert len(loops) == 1
    assert loops[0].is_closed()
    if finish_callbacks:
        assert len(iterations) == budget
    else:
        assert 0 < len(iterations) < budget
    orchestration_context.create_timer.assert_not_called()

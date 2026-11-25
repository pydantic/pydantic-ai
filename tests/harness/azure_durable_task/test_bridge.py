from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from functools import partial
from unittest.mock import MagicMock

import pytest

from pydantic_ai import (
    Agent,
    RunContext,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError, OrchestrationStatus

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

    from .worker import (
        InMemoryWorker,
        completed,
        registered_agent,
        run_agent,
        schedule_agent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


@pytest.mark.parametrize('delay', [0.0000003, 0.000001, 0.0000013])
def test_durable_timer_never_precedes_asyncio_deadline(durable_worker: InMemoryWorker, delay: float) -> None:
    deadlines: list[float] = []

    def orchestrator(
        context: task.OrchestrationContext, payload: object
    ) -> Generator[task.Task[object], object, float]:
        async def body() -> float:
            loop = asyncio.get_running_loop()
            deadlines.append(loop.time() + delay)
            await asyncio.sleep(delay)
            return loop.time()

        return (yield from run_durable(context, body))

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.schedule(orchestrator)
    durable_worker.run_until_idle()
    assert len(durable_worker.timers) == 1
    deadline = durable_worker.timers[0][2].fireAt.ToDatetime().replace(tzinfo=UTC).timestamp()
    assert deadline >= deadlines[0]
    durable_worker.fire_timers()
    state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() >= deadlines[0]
    assert durable_worker.timers == []


@pytest.mark.parametrize('timeout', [False, True])
def test_first_completed_is_delivered_before_its_sibling(durable_worker: InMemoryWorker, timeout: bool) -> None:
    fast = Agent(TestModel(custom_output_text='fast'), name='fast', capabilities=[AzureDurableTaskDurability()])
    slow = Agent(TestModel(custom_output_text='slow'), name='slow', capabilities=[AzureDurableTaskDurability()])
    completed: list[str] = []

    def orchestrator(
        context: task.OrchestrationContext, payload: object
    ) -> Generator[task.Task[object], object, list[str]]:
        async def body() -> list[str]:
            async def first() -> str:
                async with asyncio.timeout(5 if timeout else None):
                    return (await fast.run('Hi')).output

            first_task = asyncio.create_task(first())
            second_task = asyncio.create_task(slow.run('Hi'))
            done, pending = await asyncio.wait([first_task, second_task], return_when=asyncio.FIRST_COMPLETED)
            assert first_task in done and second_task in pending
            completed.append(first_task.result())
            return [first_task.result(), (await second_task).output]

        return (yield from run_durable(context, body))

    with registered_agent(durable_worker, fast), registered_agent(durable_worker, slow):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.schedule(orchestrator)
        assert durable_worker.step()
        assert len(durable_worker.activity_queue) == 2
        assert durable_worker.step()
        assert durable_worker.step()
        assert completed == ['fast']
        if timeout:
            durable_worker.fire_timers()
        state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() == ['fast', 'slow']


def test_task_group_receives_failure_before_the_other_activity_finishes(durable_worker: InMemoryWorker) -> None:
    def failure(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('fast failure')

    fast = Agent(FunctionModel(failure), name='failed', capabilities=[AzureDurableTaskDurability()])
    slow = Agent(TestModel(), name='unrelated', capabilities=[AzureDurableTaskDurability()])

    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, None]:
        async def body() -> None:
            async with asyncio.TaskGroup() as group:
                group.create_task(fast.run('Hi'))
                group.create_task(slow.run('Hi'))

        return (yield from run_durable(context, body))

    with registered_agent(durable_worker, fast), registered_agent(durable_worker, slow):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.schedule(orchestrator)
        assert durable_worker.step()
        assert durable_worker.step()
        assert durable_worker.step()
        state = durable_worker.state(instance_id)
        assert state.runtime_status == OrchestrationStatus.FAILED
        assert durable_worker.activity_calls == ['failed__model.request']


@pytest.mark.parametrize('retain', [True, False])
@pytest.mark.parametrize('empty', [True, False])
def test_owned_loop_closes_async_generators(durable_worker: InMemoryWorker, retain: bool, empty: bool) -> None:
    cleanup: list[str] = []
    retained: list[AsyncGenerator[int]] = []

    async def values() -> AsyncGenerator[int]:
        try:
            if not empty:
                yield 1
        finally:
            cleanup.append('start')
            await asyncio.sleep(0)
            cleanup.append('end')

    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, str]:
        async def body() -> str:
            iterator = values()
            if retain:
                retained.append(iterator)
            async for _ in iterator:
                break
            return 'done'

        return (yield from run_durable(context, body))

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.schedule(orchestrator)
    state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() == 'done'
    assert cleanup == ['start', 'end']
    if retain:
        with pytest.raises(StopAsyncIteration):
            asyncio.run(anext(retained[0]))


@pytest.fixture
def orchestration_context(durable_worker: InMemoryWorker) -> MagicMock:
    context = MagicMock(spec=task.OrchestrationContext)
    context.instance_id = 'instance'
    context.current_utc_datetime = durable_worker.now
    context.call_activity.return_value = task.CompletableTask[object]()
    return context


@pytest.mark.parametrize('from_callback', [False, True])
def test_generator_cleanup_can_run_under_an_outer_event_loop(
    orchestration_context: MagicMock, from_callback: bool
) -> None:
    agent = Agent(TestModel(), name='closed', capabilities=[AzureDurableTaskDurability()])
    cleanup: list[str] = []

    async def body() -> None:
        try:
            await agent.run('Hi')
        finally:
            await asyncio.sleep(0)
            cleanup.append('closed')

    generator = run_durable(orchestration_context, body)
    next(generator)

    async def close() -> None:
        outer_loop = asyncio.get_running_loop()
        outer_task = asyncio.current_task()
        assert outer_task is not None
        if from_callback:
            outer_loop.call_soon(generator.close)
            await asyncio.sleep(0)
        else:
            generator.close()
        assert asyncio.get_running_loop() is outer_loop
        assert asyncio.current_task() is outer_task
        await asyncio.sleep(0)
        assert asyncio.current_task() is outer_task
        assert asyncio.all_tasks() == {outer_task}

    asyncio.run(close())
    assert cleanup == ['closed']


def test_cleanup_cannot_wait_for_more_durable_work(orchestration_context: MagicMock) -> None:
    agent = Agent(TestModel(), name='cleanup', capabilities=[AzureDurableTaskDurability()])
    cleaned: list[str] = []

    async def body() -> None:
        try:
            await agent.run('Hi')
        finally:
            try:
                await asyncio.sleep(1)
            finally:
                cleaned.append('cancelled')

    generator = run_durable(orchestration_context, body)
    next(generator)
    with pytest.raises(UserError, match='cleanup awaited durable work'):
        generator.close()
    assert cleaned == ['cancelled']


def test_running_loop_is_rejected_before_creating_the_body(orchestration_context: MagicMock) -> None:
    body = MagicMock(side_effect=partial(asyncio.sleep, 0))

    async def check() -> None:
        generator = run_durable(orchestration_context, body)
        with pytest.raises(UserError, match='not a running event loop'):
            next(generator)

    asyncio.run(check())
    body.assert_not_called()


def test_external_generator_failure_drains_agent_scope(orchestration_context: MagicMock) -> None:
    agent = Agent(TestModel(), name='interrupted', capabilities=[AzureDurableTaskDurability()])
    exited: list[bool] = []

    async def body() -> str:
        try:
            return (await agent.run('Hi')).output
        finally:
            exited.append(True)

    generator = run_durable(orchestration_context, body)
    next(generator)
    with pytest.raises(RuntimeError, match='caller interrupted'):
        generator.throw(RuntimeError('caller interrupted'))
    assert exited == [True]


def test_run_deadline_uses_the_orchestration_clock(durable_worker: InMemoryWorker) -> None:
    deadlines: list[datetime | None] = []
    agent = Agent(TestModel(custom_output_text='done'), name='deadline', capabilities=[AzureDurableTaskDurability()])

    @agent.tool
    def check_deadline(ctx: RunContext[object]) -> str:
        deadlines.append(ctx.deadline)
        return 'deadline preserved'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None, timeout=30).output == 'done'
    assert deadlines == [datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)]


def test_cancelled_callbacks_do_not_run(orchestration_context: MagicMock) -> None:
    errors: list[dict[str, object]] = []

    async def body() -> None:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, context: errors.append(context))
        loop.call_soon(pytest.fail, 'Cancelled callback ran').cancel()
        await asyncio.sleep(0)

    with pytest.raises(StopIteration):
        next(run_durable(orchestration_context, body))
    assert errors == []


def test_threadsafe_callback_is_not_scheduled_before_rejection(durable_worker: InMemoryWorker) -> None:
    scheduled: list[bool] = []
    agent = Agent(TestModel(), name='threadsafe', capabilities=[AzureDurableTaskDurability()])

    @agent.output_validator
    def validate(output: str) -> str:
        with pytest.raises(UserError, match='Threads and external I/O') as exc_info:
            asyncio.get_running_loop().call_soon_threadsafe(scheduled.append, True)
        raise exc_info.value

    with registered_agent(durable_worker, agent):
        handle = schedule_agent(durable_worker, agent, 'Hi', deps=None)
        with pytest.raises(OrchestrationFailedError, match='Threads and external I/O'):
            completed(durable_worker, handle)
    assert scheduled == []


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

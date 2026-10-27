from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from unittest.mock import MagicMock

import pytest

from pydantic_ai import Agent, CustomEvent, RunContext
from pydantic_ai.durable_exec import CapabilityOperationId, DurableOperation
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.azure_durable_task import (
        AzureDurableTaskDurability,
        AzureDurableTaskOperationBackend,
        run_durable,
    )

    from .worker import (
        EmptyCacheIdentity,
        IdentityTransport,
        InMemoryWorker,
        JsonResultCodec,
        completed,
        registered_agent,
        run_agent,
        schedule_agent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


def test_unbound_capability_cannot_register() -> None:
    with pytest.raises(UserError, match=r'Bind .* to an agent'):
        AzureDurableTaskDurability().get_durable_operation_backend()


def test_backend_outside_orchestration_and_invalid_native_policy(durable_worker: InMemoryWorker) -> None:
    async def echo(value: str) -> str:
        return value

    backend = AzureDurableTaskOperationBackend(agent_name='custom', activity_scope=lambda _: nullcontext())
    operation = backend.bind(
        DurableOperation(
            operation_id=CapabilityOperationId('custom', operation='echo'),
            handler=echo,
            parameter_transport=IdentityTransport[str](),
            cache_identity=EmptyCacheIdentity[str](),
            result_codec=JsonResultCodec[str](str),
            config_role='capability',
        )
    )
    assert asyncio.run(operation('ordinary')) == 'ordinary'

    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, str]:
        async def body() -> str:
            return await operation('invalid', config=object())

        return (yield from run_durable(context, body))

    durable_worker.add_orchestrator(orchestrator)
    instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
    state = durable_worker.wait(instance_id)
    assert state is not None
    with pytest.raises(OrchestrationFailedError, match='activity configuration must be'):
        state.raise_if_failed()


def test_activity_cannot_emit_live_events(durable_worker: InMemoryWorker) -> None:
    @dataclass
    class DurableTaskProgressEvent(CustomEvent):
        done: int

    agent = Agent(TestModel(), name='emit', capabilities=[AzureDurableTaskDurability()])

    @agent.tool
    async def emit(ctx: RunContext[object]) -> None:
        await ctx.emit(DurableTaskProgressEvent(done=1))

    with registered_agent(durable_worker, agent):
        with pytest.raises(OrchestrationFailedError, match='Emitting events from a tool'):
            run_agent(durable_worker, agent, 'Hi', deps=None)


def test_incompatible_persisted_tool_definition_is_rejected(durable_worker: InMemoryWorker) -> None:
    agent = Agent(TestModel(), name='removed', capabilities=[AzureDurableTaskDurability()])

    @agent.tool_plain
    def original() -> str:
        # The invalid persisted name must be rejected before this tool runs.
        return 'original'  # pragma: no cover

    with registered_agent(durable_worker, agent):
        run = schedule_agent(durable_worker, agent, 'Hi', deps=None)
        for _ in range(20):
            if not (durable_worker.activity_queue and durable_worker.activity_queue[0][1].endswith('.call__tool')):
                assert durable_worker.step()
        assert durable_worker.activity_queue and durable_worker.activity_queue[0][1].endswith('.call__tool')
        instance_id, name, task_id, encoded = durable_worker.activity_queue.popleft()
        assert encoded is not None
        payload = json.loads(encoded)
        payload['parameters']['name'] = 'removed'
        payload['parameters']['original_name'] = 'removed'
        payload['parameters']['tool_def']['name'] = 'removed'
        durable_worker.activity_queue.appendleft((instance_id, name, task_id, json.dumps(payload)))
        with pytest.raises(OrchestrationFailedError, match='is no longer available'):
            completed(durable_worker, run)


def test_running_loop_is_rejected_before_creating_the_body() -> None:
    context = MagicMock(spec=task.OrchestrationContext)
    body = MagicMock(side_effect=partial(asyncio.sleep, 0))

    async def check() -> None:
        generator = run_durable(context, body)
        with pytest.raises(UserError, match='not a running event loop'):
            next(generator)

    asyncio.run(check())
    body.assert_not_called()


def test_external_generator_failure_drains_agent_scope() -> None:
    context = MagicMock(spec=task.OrchestrationContext)
    context.instance_id = 'interrupted'
    context.current_utc_datetime = datetime(2026, 1, 1)
    context.call_activity.return_value = task.CompletableTask[object]()
    agent = Agent(TestModel(), name='interrupted', capabilities=[AzureDurableTaskDurability()])
    exited: list[bool] = []

    async def body() -> str:
        try:
            return (await agent.run('Hi')).output
        finally:
            exited.append(True)

    generator = run_durable(context, body)
    next(generator)
    with pytest.raises(RuntimeError, match='caller interrupted'):
        generator.throw(RuntimeError('caller interrupted'))
    assert exited == [True]


@pytest.mark.parametrize('durable', [True, False])
def test_deadlines_and_cancelled_callbacks(durable_worker: InMemoryWorker, durable: bool) -> None:
    agent = Agent(TestModel(custom_output_text='done'), name='deadline', capabilities=[AzureDurableTaskDurability()])

    @agent.tool
    def check_deadline(ctx: RunContext[object]) -> str:
        assert ctx.deadline is not None
        if durable:
            assert ctx.deadline == datetime(2026, 1, 1, 0, 0, 30, tzinfo=UTC)
        return 'deadline preserved'

    if not durable:
        assert agent.run_sync('Hi', timeout=30).output == 'done'
        return

    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, str]:
        async def body() -> str:
            asyncio.get_running_loop().call_soon(pytest.fail, 'Cancelled callback ran').cancel()
            await agent.run('Hi', timeout=30)
            return (await agent.run('Again', timeout=30)).output

        return (yield from run_durable(context, body))

    with registered_agent(durable_worker, agent):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
        state = durable_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() == 'done'


def test_threadsafe_callback_is_not_scheduled_before_rejection(durable_worker: InMemoryWorker) -> None:
    scheduled: list[bool] = []
    agent = Agent(TestModel(), name='threadsafe', capabilities=[AzureDurableTaskDurability()])

    @agent.output_validator
    def validate(output: str) -> str:
        asyncio.get_running_loop().call_soon_threadsafe(scheduled.append, True)
        # Thread-safe scheduling must be rejected before the validator returns.
        return output  # pragma: no cover

    with registered_agent(durable_worker, agent):
        handle = schedule_agent(durable_worker, agent, 'Hi', deps=None)
        with pytest.raises(OrchestrationFailedError, match='Threads and external I/O'):
            completed(durable_worker, handle)
    assert scheduled == []

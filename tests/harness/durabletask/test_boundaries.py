from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from pydantic_ai import Agent, CustomEvent, RunContext
from pydantic_ai.durable_exec import CapabilityOperationId
from pydantic_ai.durable_exec._operation import (
    DurableOperation,
    IdentityParameterTransport,
    NoCacheIdentity,
    TypedResultCodec,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.durabletask import (
        DurableTaskAgent,
        DurableTaskDurability,
        DurableTaskOperationBackend,
        run_durable,
    )

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


def test_unbound_capability_cannot_register() -> None:
    with pytest.raises(UserError, match=r'Bind .* to an agent'):
        DurableTaskDurability().get_durable_operation_backend()


def test_backend_outside_orchestration_and_invalid_native_policy(durable_worker: InMemoryWorker) -> None:
    @asynccontextmanager
    async def scope(value: JsonValue) -> AsyncGenerator[None]:
        yield

    async def echo(value: str) -> str:
        return value

    backend = DurableTaskOperationBackend(agent_name='custom', activity_scope=scope)
    operation = backend.bind(
        DurableOperation(
            operation_id=CapabilityOperationId('custom', operation='echo'),
            handler=echo,
            parameter_transport=IdentityParameterTransport[str](),
            cache_identity=NoCacheIdentity[str](),
            result_codec=TypedResultCodec[str](str),
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

    agent = Agent(TestModel(), name='emit', capabilities=[DurableTaskDurability()])

    @agent.tool
    async def emit(ctx: RunContext[object]) -> None:
        await ctx.emit(DurableTaskProgressEvent(done=1))

    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        with pytest.raises(OrchestrationFailedError, match='Emitting events inside a Durable Task activity'):
            durable.start(durable_worker.client, 'Hi').result()


def test_incompatible_persisted_tool_definition_is_rejected(durable_worker: InMemoryWorker) -> None:
    agent = Agent(TestModel(), name='removed', capabilities=[DurableTaskDurability()])

    @agent.tool_plain
    def original() -> str:
        return 'original'

    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        run = durable.start(durable_worker.client, 'Hi')
        for _ in range(20):
            if durable_worker.activity_queue and durable_worker.activity_queue[0][1].endswith('.call_tool'):
                break
            assert durable_worker.step()
        else:
            pytest.fail('No function tool activity was scheduled')
        instance_id, name, task_id, encoded = durable_worker.activity_queue.popleft()
        assert encoded is not None
        payload = json.loads(encoded)
        payload['parameters']['name'] = 'removed'
        payload['parameters']['original_name'] = 'removed'
        payload['parameters']['tool_def']['name'] = 'removed'
        durable_worker.activity_queue.appendleft((instance_id, name, task_id, json.dumps(payload)))
        with pytest.raises(OrchestrationFailedError, match='is no longer available'):
            run.result()


def test_running_loop_is_rejected_before_creating_the_body() -> None:
    context = MagicMock(spec=task.OrchestrationContext)
    called: list[bool] = []

    async def body() -> None:
        called.append(True)

    async def check() -> None:
        generator = run_durable(context, body)
        with pytest.raises(UserError, match='not a running event loop'):
            next(generator)

    asyncio.run(check())
    assert called == []


def test_external_generator_failure_drains_agent_scope() -> None:
    context = MagicMock(spec=task.OrchestrationContext)
    context.instance_id = 'interrupted'
    context.current_utc_datetime = datetime(2026, 1, 1)
    context.call_activity.return_value = task.CompletableTask[object]()
    agent = Agent(TestModel(), name='interrupted', capabilities=[DurableTaskDurability()])
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
    agent = Agent(TestModel(custom_output_text='done'), name='deadline', capabilities=[DurableTaskDurability()])

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

    with DurableTaskAgent(agent).register(durable_worker):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
        state = durable_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() == 'done'


def test_threadsafe_callback_is_not_scheduled_before_rejection(durable_worker: InMemoryWorker) -> None:
    scheduled: list[bool] = []
    agent = Agent(TestModel(), name='threadsafe', capabilities=[DurableTaskDurability()])

    @agent.output_validator
    def validate(output: str) -> str:
        asyncio.get_running_loop().call_soon_threadsafe(scheduled.append, True)
        return output

    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        handle = durable.start(durable_worker.client, 'Hi')
        with pytest.raises(OrchestrationFailedError, match='Threads and external I/O'):
            handle.result()
    assert scheduled == []

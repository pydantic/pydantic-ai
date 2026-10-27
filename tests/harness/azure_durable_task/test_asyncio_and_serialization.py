import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime, timedelta
from functools import partial
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import BaseModel, ConfigDict, Json

from pydantic_ai import Agent, Choice, Choices, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
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
def test_durable_timer_never_precedes_asyncio_deadline(durable_worker: 'InMemoryWorker', delay: float) -> None:
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
    instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
    durable_worker.run_until_idle()
    assert len(durable_worker.timers) == 1
    deadline = durable_worker.timers[0][2].fireAt.ToDatetime().replace(tzinfo=UTC).timestamp()
    assert deadline >= deadlines[0]
    durable_worker.fire_timers()
    state = durable_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() >= deadlines[0]
    assert durable_worker.timers == []


@pytest.mark.parametrize('non_utf8', [False, True])
def test_serialization_failure_is_terminal_with_native_retries(
    durable_worker: 'InMemoryWorker', non_utf8: bool
) -> None:
    effects: list[str] = []
    agent = Agent(
        TestModel(custom_output_text='done'),
        name='bad-json',
        capabilities=[
            AzureDurableTaskDurability(
                retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
            )
        ],
    )

    @agent.tool_plain
    def write() -> object:
        effects.append('written')
        return b'\xff' if non_utf8 else object()

    with registered_agent(durable_worker, agent):
        handle = schedule_agent(durable_worker, agent, 'Go', deps=None)
        durable_worker.run_until_idle()
        assert effects == ['written']
        assert durable_worker.timers == []
        with pytest.raises(OrchestrationFailedError, match='Durable Task serialization failed'):
            completed(durable_worker, handle)
    assert effects == ['written']


@pytest.mark.parametrize('timeout', [False, True])
def test_first_completed_is_delivered_before_its_sibling(durable_worker: 'InMemoryWorker', timeout: bool) -> None:
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
        instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
        assert durable_worker.step()
        assert len(durable_worker.activity_queue) == 2
        assert durable_worker.step()
        assert durable_worker.step()
        assert completed == ['fast']
        if timeout:
            durable_worker.fire_timers()
        state = durable_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() == ['fast', 'slow']


def test_task_group_receives_failure_before_the_other_activity_finishes(durable_worker: 'InMemoryWorker') -> None:
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
        instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
        assert durable_worker.step()
        assert durable_worker.step()
        assert durable_worker.step()
        state = durable_worker.state(instance_id)
        assert state is not None
        assert state.runtime_status == OrchestrationStatus.FAILED
        assert durable_worker.activity_calls == ['failed____model.request']


def test_typed_tool_arguments_are_restored_from_json(durable_worker: 'InMemoryWorker') -> None:
    class Payload(BaseModel):
        value: int
        timestamp: datetime
        identifier: UUID
        encoded: Json[list[int]]

    args = {
        'value': 42,
        'timestamp': '2026-01-01T00:00:00Z',
        'identifier': '12345678-1234-5678-1234-567812345678',
        'encoded': '[1,2]',
    }
    observed: list[int] = []
    validated: list[int] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart('consume', json.dumps(args))] if len(messages) == 1 else [TextPart('done')]
        )

    agent = Agent(FunctionModel(model), deps_type=int, name='typed-args', capabilities=[AzureDurableTaskDurability()])

    def validate(ctx: RunContext[int], payload: Payload) -> None:
        assert isinstance(payload, Payload)
        assert ctx.deps == 7
        validated.append(payload.value)

    @agent.tool_plain(args_validator=validate)
    def consume(payload: Payload) -> str:
        assert isinstance(payload, Payload)
        assert payload.timestamp.year == 2026
        assert payload.identifier.hex == '12345678123456781234567812345678'
        assert payload.encoded == [1, 2]
        observed.append(payload.value)
        return 'typed'

    assert agent.run_sync('Hi', deps=7).output == 'done'
    assert observed == [42]
    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=7).output == 'done'
    assert observed == [42, 42]
    assert validated == [42, 42]


def test_choice_action_output_and_partial_output(durable_worker: 'InMemoryWorker') -> None:
    choices = Choices({'answer': Choice(value=lambda: 42)})
    choice_agent = Agent(TestModel(), output_type=choices, name='choice', capabilities=[AzureDurableTaskDurability()])
    assert choice_agent.run_sync('Hi').output == 42
    with registered_agent(durable_worker, choice_agent):
        assert run_agent(durable_worker, choice_agent, 'Hi', deps=None).output == 42

    def output(prefix: str, value: str) -> str:
        return f'{prefix}:{value}'

    partial_agent = Agent(
        TestModel(custom_output_args={'value': 'x'}),
        output_type=partial(output, 'prefix'),
        name='partial',
        capabilities=[AzureDurableTaskDurability()],
    )
    assert partial_agent.run_sync('Hi').output == 'prefix:x'
    with registered_agent(durable_worker, partial_agent):
        assert run_agent(durable_worker, partial_agent, 'Hi', deps=None).output == 'prefix:x'


def test_strict_and_json_outputs_round_trip(durable_worker: 'InMemoryWorker') -> None:
    class StrictOutput(BaseModel):
        model_config = ConfigDict(strict=True)
        timestamp: datetime

    class JsonOutput(BaseModel):
        data: Json[list[int]]

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"timestamp":"2026-01-01T00:00:00Z"}')])

    strict = Agent(
        FunctionModel(model),
        output_type=StrictOutput,
        name='strict',
        capabilities=[AzureDurableTaskDurability()],
    )
    expected = strict.run_sync('Hi').output
    with registered_agent(durable_worker, strict):
        assert run_agent(durable_worker, strict, 'Hi', deps=None).output == expected

    json_agent = Agent(
        TestModel(custom_output_args={'data': '[1,2]'}),
        output_type=JsonOutput,
        name='json',
        capabilities=[AzureDurableTaskDurability()],
    )
    expected_json = json_agent.run_sync('Hi').output
    with registered_agent(durable_worker, json_agent):
        assert run_agent(durable_worker, json_agent, 'Hi', deps=None).output == expected_json


@pytest.mark.parametrize('retain', [True, False])
@pytest.mark.parametrize('empty', [True, False])
def test_owned_loop_closes_async_generators(durable_worker: 'InMemoryWorker', retain: bool, empty: bool) -> None:
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
    instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
    state = durable_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() == 'done'
    assert cleanup == ['start', 'end']
    if retain:
        with pytest.raises(StopAsyncIteration):
            asyncio.run(anext(retained[0]))


@pytest.mark.parametrize('from_callback', [False, True])
def test_generator_cleanup_can_run_under_an_outer_event_loop(from_callback: bool) -> None:
    context = MagicMock(spec=task.OrchestrationContext)
    context.current_utc_datetime = datetime(2026, 1, 1)
    context.call_activity.return_value = task.CompletableTask[object]()
    agent = Agent(TestModel(), name='closed', capabilities=[AzureDurableTaskDurability()])
    cleanup: list[str] = []

    async def body() -> None:
        try:
            await agent.run('Hi')
        finally:
            await asyncio.sleep(0)
            cleanup.append('closed')

    generator = run_durable(context, body)
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


def test_cleanup_cannot_wait_for_more_durable_work() -> None:
    context = MagicMock(spec=task.OrchestrationContext)
    context.current_utc_datetime = datetime(2026, 1, 1)
    context.call_activity.return_value = task.CompletableTask[object]()
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

    generator = run_durable(context, body)
    next(generator)
    with pytest.raises(UserError, match='cleanup awaited durable work'):
        generator.close()
    assert cleaned == ['cancelled']

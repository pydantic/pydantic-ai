import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from datetime import datetime, timedelta
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
from pydantic_ai.output import NativeOutput, OutputSpec, PromptedOutput, TextOutput, ToolOutput
from pydantic_ai.profiles import ModelProfile

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError, OrchestrationStatus

    from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability, run_durable

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


@pytest.mark.parametrize('non_utf8', [False, True])
def test_serialization_failure_is_terminal_with_native_retries(
    durable_worker: 'InMemoryWorker', non_utf8: bool
) -> None:
    effects: list[str] = []
    agent = Agent(
        TestModel(custom_output_text='done'),
        name='bad-json',
        capabilities=[
            DurableTaskDurability(
                retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
            )
        ],
    )

    @agent.tool_plain
    def write() -> object:
        effects.append('written')
        return b'\xff' if non_utf8 else object()

    with DurableTaskAgent(agent).register(durable_worker):
        handle = DurableTaskAgent(agent).start(durable_worker.client, 'Go')
        durable_worker.run_until_idle()
        assert effects == ['written']
        assert durable_worker.timers == []
        with pytest.raises(OrchestrationFailedError, match='Durable Task serialization failed'):
            handle.result()
    assert effects == ['written']


@pytest.mark.parametrize('timeout', [False, True])
def test_first_completed_is_delivered_before_its_sibling(durable_worker: 'InMemoryWorker', timeout: bool) -> None:
    fast = Agent(TestModel(custom_output_text='fast'), name='fast', capabilities=[DurableTaskDurability()])
    slow = Agent(TestModel(custom_output_text='slow'), name='slow', capabilities=[DurableTaskDurability()])
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

    with DurableTaskAgent(fast).register(durable_worker), DurableTaskAgent(slow).register(durable_worker):
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

    fast = Agent(FunctionModel(failure), name='failed', capabilities=[DurableTaskDurability()])
    slow = Agent(TestModel(), name='unrelated', capabilities=[DurableTaskDurability()])

    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, None]:
        async def body() -> None:
            async with asyncio.TaskGroup() as group:
                group.create_task(fast.run('Hi'))
                group.create_task(slow.run('Hi'))

        return (yield from run_durable(context, body))

    with DurableTaskAgent(fast).register(durable_worker), DurableTaskAgent(slow).register(durable_worker):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
        assert durable_worker.step()
        assert durable_worker.step()
        assert durable_worker.step()
        state = durable_worker.state(instance_id)
        assert state is not None
        assert state.runtime_status == OrchestrationStatus.FAILED
        assert durable_worker.activity_calls == ['failed__model.request']


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

    agent = Agent(FunctionModel(model), deps_type=int, name='typed-args', capabilities=[DurableTaskDurability()])

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
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        assert durable.start(durable_worker.client, 'Hi', deps_input=7).result().output == 'done'
    assert observed == [42, 42]
    assert validated == [42, 42]


def test_choice_action_output_and_partial_output(durable_worker: 'InMemoryWorker') -> None:
    choices = Choices({'answer': Choice(value=lambda: 42)})
    choice_agent = Agent(TestModel(), output_type=choices, name='choice', capabilities=[DurableTaskDurability()])
    assert choice_agent.run_sync('Hi').output == 42
    choice = DurableTaskAgent(choice_agent)
    with choice.register(durable_worker):
        assert choice.start(durable_worker.client, 'Hi').result().output == 42

    def output(prefix: str, value: str) -> str:
        return f'{prefix}:{value}'

    partial_agent = Agent(
        TestModel(custom_output_args={'value': 'x'}),
        output_type=partial(output, 'prefix'),
        name='partial',
        capabilities=[DurableTaskDurability()],
    )
    assert partial_agent.run_sync('Hi').output == 'prefix:x'
    durable = DurableTaskAgent(partial_agent)
    with durable.register(durable_worker):
        assert durable.start(durable_worker.client, 'Hi').result().output == 'prefix:x'


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
        capabilities=[DurableTaskDurability()],
    )
    expected = strict.run_sync('Hi').output
    durable_strict = DurableTaskAgent(strict)
    with durable_strict.register(durable_worker):
        assert durable_strict.start(durable_worker.client, 'Hi').result().output == expected

    json_agent = Agent(
        TestModel(custom_output_args={'data': '[1,2]'}),
        output_type=JsonOutput,
        name='json',
        capabilities=[DurableTaskDurability()],
    )
    expected_json = json_agent.run_sync('Hi').output
    durable_json = DurableTaskAgent(json_agent)
    with durable_json.register(durable_worker):
        assert durable_json.start(durable_worker.client, 'Hi').result().output == expected_json


@pytest.mark.parametrize('retain', [True, False])
def test_owned_loop_closes_async_generators(durable_worker: 'InMemoryWorker', retain: bool) -> None:
    cleanup: list[str] = []
    retained: list[AsyncGenerator[int]] = []

    async def values() -> AsyncGenerator[int]:
        try:
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


def test_generator_cleanup_can_run_under_an_outer_event_loop() -> None:
    context = MagicMock(spec=task.OrchestrationContext)
    context.current_utc_datetime = datetime(2026, 1, 1)
    context.call_activity.return_value = task.CompletableTask[object]()
    agent = Agent(TestModel(), name='closed', capabilities=[DurableTaskDurability()])
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
        generator.close()
        assert asyncio.get_running_loop() is outer_loop

    asyncio.run(close())
    assert cleanup == ['closed']


def test_cleanup_cannot_wait_for_more_durable_work() -> None:
    context = MagicMock(spec=task.OrchestrationContext)
    context.current_utc_datetime = datetime(2026, 1, 1)
    context.call_activity.return_value = task.CompletableTask[object]()
    agent = Agent(TestModel(), name='cleanup', capabilities=[DurableTaskDurability()])
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


@pytest.mark.parametrize('annotated', [True, False])
def test_choice_actions_need_annotations_for_structured_results(
    durable_worker: 'InMemoryWorker', annotated: bool
) -> None:
    class Output(BaseModel):
        value: int

    def action() -> Output:
        return Output(value=42)

    choices = Choices({'answer': Choice(value=action if annotated else lambda: Output(value=42))})
    agent = Agent(TestModel(), output_type=choices, name='choice-result', capabilities=[DurableTaskDurability()])
    assert agent.run_sync('Hi').output == Output(value=42)
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        run = durable.start(durable_worker.client, 'Hi')
        if annotated:
            assert run.result().output == Output(value=42)
        else:
            with pytest.raises(OrchestrationFailedError, match='serializ'):
                run.result()


@pytest.mark.parametrize('mode', ['native', 'prompted', 'tool', 'text'])
def test_output_markers_describe_returned_values(durable_worker: 'InMemoryWorker', mode: str) -> None:
    class Output(BaseModel):
        value: int

    def tool_output(value: int) -> Output:
        return Output(value=value)

    def text_output(text: str) -> Output:
        return Output(value=int(text))

    def structured(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('{"value":42}')])

    output_type: OutputSpec[Output]
    if mode == 'native':
        output_type = NativeOutput(Output)
    elif mode == 'prompted':
        output_type = PromptedOutput(Output)
    elif mode == 'tool':
        output_type = [ToolOutput(tool_output)]
    else:
        output_type = [TextOutput(text_output)]
    model = (
        FunctionModel(structured, profile=ModelProfile(supports_json_schema_output=True))
        if mode in {'native', 'prompted'}
        else TestModel(custom_output_text='42')
        if mode == 'text'
        else TestModel(custom_output_args={'value': 42})
    )
    agent = Agent(model, output_type=output_type, name='marked', capabilities=[DurableTaskDurability()])
    assert agent.run_sync('Hi').output == Output(value=42)
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        assert durable.start(durable_worker.client, 'Hi').result().output == Output(value=42)


@pytest.mark.parametrize('selected', ['typed', 'literal', 'constructor'])
def test_choices_restore_both_actions_and_values(durable_worker: 'InMemoryWorker', selected: str) -> None:
    class Output(BaseModel):
        value: int = 42

    def action() -> Output:
        return Output()

    choices = Choices(
        {
            'typed': Choice(value=action),
            'literal': Choice(value=7),
            'constructor': Choice(value=Output),
        }
    )
    agent = Agent(
        TestModel(custom_output_args=selected),
        output_type=choices,
        name='mixed',
        capabilities=[DurableTaskDurability()],
    )
    expected = 7 if selected == 'literal' else Output()
    assert agent.run_sync('Hi').output == expected
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        assert durable.start(durable_worker.client, 'Hi').result().output == expected

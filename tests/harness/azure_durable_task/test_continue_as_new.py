from __future__ import annotations

import asyncio
import json
from collections.abc import Generator
from pathlib import Path

import anyio
import pytest
from pydantic import JsonValue

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from ...conftest import try_import

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, ContinueAsNew, run_durable

    from .examples import load_generation_example
    from .worker import InMemoryWorker, completed, registered_agent

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


class TestContinueAsNew:
    @pytest.mark.parametrize('is_async', [False, True])
    @pytest.mark.parametrize('stream', [False, True])
    @pytest.mark.parametrize('continue_now', [False, True])
    def test_handler_can_continue_or_defer(
        self, durable_worker: InMemoryWorker, is_async: bool, stream: bool, continue_now: bool
    ) -> None:
        calls: list[str] = []
        cleaned: list[bool] = []
        steps: list[int] = []

        def checkpoint(ctx: RunContext[None]) -> None:
            steps.append(ctx.run_step)
            if ctx.run_step == 2:
                assert ctx.usage.requests == 1
                assert isinstance(ctx.messages[-1], ModelRequest)
                assert any(isinstance(part, ToolReturnPart) for part in ctx.messages[-1].parts)
                if continue_now:
                    raise ContinueAsNew({'requests': ctx.usage.requests, 'tool_results': len(calls)})

        async def async_checkpoint(ctx: RunContext[None]) -> None:
            await asyncio.sleep(0)
            checkpoint(ctx)

        agent = Agent(
            TestModel(custom_output_text='done'),
            name='callback',
            deps_type=type(None),
            capabilities=[AzureDurableTaskDurability(continue_as_new=async_checkpoint if is_async else checkpoint)],
        )

        @agent.tool_plain
        def effect() -> str:
            calls.append('once')
            return 'saved'

        def workflow(
            ctx: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, JsonValue]:
            async def body() -> str:
                try:
                    if stream:
                        async with agent.run_stream('Hi') as response:
                            return await response.get_output()
                    return (await agent.run('Hi')).output
                finally:
                    cleaned.append(True)

            try:
                return (yield from run_durable(ctx, body))
            except ContinueAsNew as continuation:
                assert cleaned
                return continuation.input

        durable_worker.add_orchestrator(workflow)
        with registered_agent(durable_worker, agent):
            result = completed(durable_worker, durable_worker.schedule(workflow))
        assert result == ({'requests': 1, 'tool_results': 1} if continue_now else 'done')
        assert calls == ['once']
        assert 1 in steps and 2 in steps
        assert cleaned
        assert durable_worker.timers == []

    @pytest.mark.parametrize('configured', [False, True])
    def test_outside_orchestration_is_passthrough(self, configured: bool) -> None:
        calls: list[int] = []

        def checkpoint(ctx: RunContext[None]) -> None:
            calls.append(ctx.run_step)  # pragma: no cover

        agent = Agent(
            TestModel(custom_output_text='ordinary'),
            name='outside',
            deps_type=type(None),
            capabilities=[AzureDurableTaskDurability(continue_as_new=checkpoint if configured else None)],
        )
        assert agent.run_sync('Hi').output == 'ordinary'
        assert calls == []

    @pytest.mark.parametrize('depth', [1, 2])
    def test_singleton_task_groups_unwind_before_handoff(self, durable_worker: InMemoryWorker, depth: int) -> None:
        cleaned: list[bool] = []

        def checkpoint(ctx: RunContext[None]) -> None:
            raise ContinueAsNew({'next': ctx.usage.requests})

        agent = Agent(
            TestModel(),
            name='grouped',
            deps_type=type(None),
            capabilities=[AzureDurableTaskDurability(continue_as_new=checkpoint)],
        )

        async def grouped() -> None:
            async with anyio.create_task_group():
                await agent.run('Hi')

        def workflow(
            ctx: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, JsonValue]:
            async def body() -> None:
                try:
                    if depth == 1:
                        await grouped()
                    else:
                        async with anyio.create_task_group():
                            await grouped()
                finally:
                    cleaned.append(True)

            with pytest.raises(ContinueAsNew) as continuation:
                yield from run_durable(ctx, body)
            assert cleaned
            return continuation.value.input

        durable_worker.add_orchestrator(workflow)
        with registered_agent(durable_worker, agent):
            assert completed(durable_worker, durable_worker.schedule(workflow)) == {'next': 0}
        assert cleaned

    def test_grouped_cleanup_failure_is_not_hidden(self, durable_worker: InMemoryWorker) -> None:
        def workflow(ctx: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, None]:
            async def body() -> None:
                raise BaseExceptionGroup('cleanup failed', [ContinueAsNew(None), RuntimeError('resource cleanup')])

            yield from run_durable(ctx, body)

        durable_worker.add_orchestrator(workflow)
        with pytest.raises(BaseExceptionGroup, match='cleanup failed') as failure:
            completed(durable_worker, durable_worker.schedule(workflow))
        assert len(failure.value.exceptions) == 2
        assert isinstance(failure.value.exceptions[0], ContinueAsNew)
        assert str(failure.value.exceptions[1]) == 'resource cleanup'

    def test_unrelated_singleton_group_is_not_unwrapped(self, durable_worker: InMemoryWorker) -> None:
        def workflow(ctx: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, None]:
            async def body() -> None:
                raise ExceptionGroup('application failure', [ValueError('not a continuation')])

            yield from run_durable(ctx, body)

        durable_worker.add_orchestrator(workflow)
        with pytest.raises(OrchestrationFailedError, match='application failure'):
            completed(durable_worker, durable_worker.schedule(workflow))

    def test_handler_failure_is_not_a_continuation(self, durable_worker: InMemoryWorker) -> None:
        def checkpoint(ctx: RunContext[None]) -> None:
            raise ValueError('checkpoint failed')

        agent = Agent(
            TestModel(),
            name='failed-callback',
            deps_type=type(None),
            capabilities=[AzureDurableTaskDurability(continue_as_new=checkpoint)],
        )

        def workflow(ctx: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, None]:
            yield from run_durable(ctx, lambda: agent.run('Hi'))

        durable_worker.add_orchestrator(workflow)
        with registered_agent(durable_worker, agent):
            with pytest.raises(OrchestrationFailedError, match='checkpoint failed'):
                completed(durable_worker, durable_worker.schedule(workflow))
        assert durable_worker.activity_calls == []

    def test_handler_can_await_a_durable_operation(self, durable_worker: InMemoryWorker) -> None:
        writes: list[int] = []

        class Archive(AbstractCapability[None]):
            id = 'archive'

            @durable_operation('save')
            async def save(self, ctx: RunContext[None], messages: int) -> str:
                writes.append(messages)
                return 'stored'

        async def checkpoint(ctx: RunContext[None]) -> None:
            result = await archive.save(ctx, len(ctx.messages))
            raise ContinueAsNew(result)

        archive = Archive()
        agent = Agent(
            TestModel(),
            name='archive',
            deps_type=type(None),
            capabilities=[archive, AzureDurableTaskDurability(continue_as_new=checkpoint)],
        )

        def workflow(
            ctx: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, JsonValue]:
            with pytest.raises(ContinueAsNew) as continuation:
                yield from run_durable(ctx, lambda: agent.run('Hi'))
            return continuation.value.input

        durable_worker.add_orchestrator(workflow)
        with registered_agent(durable_worker, agent):
            assert completed(durable_worker, durable_worker.schedule(workflow)) == 'stored'
        assert writes == [1]
        assert len(durable_worker.activity_calls) == 1

    def test_usage_budget_survives_application_continuation(self, durable_worker: InMemoryWorker) -> None:
        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            count = sum(
                isinstance(part, ToolReturnPart)
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
            return ModelResponse(parts=[ToolCallPart('effect', {}, tool_call_id=f'call-{count}')])

        example = load_generation_example(FunctionModel(model), agent_name='budget')
        calls: list[str] = []

        @example.agent.tool_plain
        def effect() -> str:
            calls.append('once')
            return 'saved'

        durable_worker.add_orchestrator(example.orchestrator)
        with registered_agent(durable_worker, example.agent):
            instance_id = durable_worker.schedule(
                example.orchestrator, input={'prompt': 'Hi', 'usage_limits': {'request_limit': 3}}
            )
            with pytest.raises(OrchestrationFailedError, match='request_limit'):
                completed(durable_worker, instance_id)
        assert calls == ['once', 'once', 'once']
        assert len(durable_worker.instances[instance_id].generations) == 1

    @pytest.mark.parametrize(
        ('fixture', 'commit'),
        [
            ('pr1-escaped-history.json', 'b6b07c333271e94dcd63eb99d6b7ceb3390a7b65'),
            ('pr1-main-history.json', '3eefe33b2398f8f28119b28c08313e7fd1956560'),
        ],
    )
    def test_final_pr1_histories_replay_identically(
        self, durable_worker: InMemoryWorker, fixture: str, commit: str
    ) -> None:
        recording = json.loads(Path(__file__).with_name(fixture).read_text(encoding='utf-8'))
        assert recording['commit'] == commit
        agent = Agent(
            TestModel(custom_output_text='replacement must not execute'),
            name='pr1-replay',
            capabilities=[AzureDurableTaskDurability()],
        )

        @agent.tool_plain
        async def baseline() -> str:
            pytest.fail('A recorded PR1 activity executed during replay')  # pragma: no cover

        def pr1_replay_workflow(
            ctx: task.OrchestrationContext, prompt: str
        ) -> Generator[task.Task[object], object, str]:
            result = yield from run_durable(ctx, lambda: agent.run(prompt))
            return result.output

        durable_worker.add_orchestrator(pr1_replay_workflow)
        with registered_agent(durable_worker, agent):
            actions = durable_worker.replay_history(recording)
        assert len(actions) == 1
        assert actions[0].completeOrchestration.result.value == recording['output']
        assert durable_worker.activity_calls == []

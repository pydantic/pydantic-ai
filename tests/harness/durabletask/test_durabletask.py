from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, AsyncIterator, Generator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

import pytest
from pydantic import BaseModel, JsonValue

from pydantic_ai import (
    Agent,
    ApprovalRequired,
    CallDeferred,
    DeferredToolRequests,
    ModelRetry,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, durable_operation
from pydantic_ai.durable_exec import RoleBasedOperationConfig
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, PartStartEvent, TextPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import DynamicToolset, FunctionToolset

from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability, run_durable

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


class _WorkerDependencies:
    def __init__(self, value: JsonValue = 42) -> None:
        self.value = value


class TestDurableTaskDurability:
    def test_happy_path(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(custom_output_text='hello'), name='support', capabilities=[DurableTaskDurability()])
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            handle = durable.start(durable_worker.client, 'Hi', instance_id='hello')
            result = handle.result()
        assert result.output == 'hello'
        assert result.run_id == 'hello:support:1'
        assert durable_worker.activity_calls == ['support__model.request']

    def test_outside_orchestration(self) -> None:
        agent = Agent(TestModel(custom_output_text='plain'), name='plain', capabilities=[DurableTaskDurability()])
        assert agent.run_sync('Hi').output == 'plain'

    def test_structured_output(self, durable_worker: InMemoryWorker) -> None:
        class Output(BaseModel):
            value: int

        agent = Agent(
            TestModel(custom_output_args={'value': 42}),
            output_type=Output,
            name='structured',
            capabilities=[DurableTaskDurability()],
        )
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            result = durable.start(durable_worker.client, 'Hi').result()
        assert isinstance(result.output, Output)
        assert result.output.value == 42
        assert result.all_messages()
        assert result.usage.requests == 1

    def test_callable_output(self, durable_worker: InMemoryWorker) -> None:
        def output(value: int) -> str:
            return f'value={value}'

        agent = Agent(
            TestModel(custom_output_args={'value': 42}),
            output_type=output,
            name='callable',
            capabilities=[DurableTaskDurability()],
        )
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            assert durable.start(durable_worker.client, 'Hi').result().output == 'value=42'

    def test_instrumented_agent(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='instrumented',
            capabilities=[Instrumentation(), DurableTaskDurability()],
        )
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            result = durable.start(durable_worker.client, 'Hi').result()
        assert result.output == 'done'
        assert result.usage.requests == 1

    def test_parallel_tools_are_scheduled_together(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []
        agent = Agent(TestModel(custom_output_text='done'), name='parallel', capabilities=[DurableTaskDurability()])

        @agent.tool_plain
        def first() -> str:
            calls.append('first')
            return 'one'

        @agent.tool_plain
        def second() -> str:
            calls.append('second')
            return 'two'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            handle = durable.start(durable_worker.client, 'Hi')
            assert durable_worker.step()  # Schedule the model.
            assert durable_worker.step()  # Execute the model.
            assert durable_worker.step()  # Schedule both argument validators, before either runs.
            assert len(durable_worker.activity_queue) == 2
            assert handle.result().output == 'done'
        assert calls == ['first', 'second']
        assert durable_worker.activity_calls.count('parallel__function_toolset__<agent>.call_tool') == 2

    def test_worker_dependencies_and_cleanup(self, durable_worker: InMemoryWorker) -> None:
        lifetimes: list[str] = []

        @asynccontextmanager
        async def deps_factory(value: JsonValue) -> AsyncGenerator[_WorkerDependencies]:
            lifetimes.append('enter')
            try:
                yield _WorkerDependencies(value)
            finally:
                lifetimes.append('exit')

        capability = DurableTaskDurability(deps_factory=deps_factory)
        agent = Agent(
            TestModel(custom_output_text='done'), deps_type=_WorkerDependencies, name='deps', capabilities=[capability]
        )
        values: list[JsonValue] = []

        @agent.tool
        def read(ctx: RunContext[_WorkerDependencies]) -> str:
            values.append(ctx.deps.value)
            assert ctx.agent is not None
            assert ctx.is_tool_available('read')
            return 'ok'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            assert durable.start(durable_worker.client, 'Hi', deps_input={'value': 42}).result().output == 'done'
        assert values == [{'value': 42}]
        assert lifetimes == ['enter', 'exit'] * len(durable_worker.activity_calls)

    def test_orchestrator_dependency_access_is_guarded(self, durable_worker: InMemoryWorker) -> None:
        @asynccontextmanager
        async def deps_factory(value: JsonValue) -> AsyncGenerator[_WorkerDependencies]:
            yield _WorkerDependencies()

        agent = Agent(
            TestModel(),
            deps_type=_WorkerDependencies,
            name='guard',
            capabilities=[DurableTaskDurability(deps_factory=deps_factory)],
        )

        @agent.instructions
        def instructions(ctx: RunContext[_WorkerDependencies]) -> str:
            return str(ctx.deps.value)

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            with pytest.raises(OrchestrationFailedError, match='only available inside a Durable Task activity'):
                durable.start(durable_worker.client, 'Hi').result()
        assert durable_worker.activity_calls == []

    @pytest.mark.parametrize('approved', [True, False])
    def test_approval(self, durable_worker: InMemoryWorker, approved: bool) -> None:
        calls: list[str] = []
        agent = Agent(
            TestModel(custom_output_text='done'),
            output_type=[str, DeferredToolRequests],
            name='approval',
            capabilities=[DurableTaskDurability()],
        )

        @agent.tool
        def write(ctx: RunContext[object]) -> str:
            if not ctx.tool_call_approved:
                raise ApprovalRequired()
            calls.append('written')
            return 'ok'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            handle = durable.start(durable_worker.client, 'Hi')
            durable_worker.run_until_idle()
            pending = handle.pending_tools()
            assert len(pending.approvals) == 1
            with pytest.raises(TimeoutError):
                handle.result(timeout=0)
            handle.approve(pending.approvals[0].tool_call_id, approved=approved)
            assert handle.result().output == 'done'
        assert calls == (['written'] if approved else [])

    def test_deferred_tool(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []
        agent = Agent(
            TestModel(custom_output_text='done'),
            output_type=[str, DeferredToolRequests],
            name='deferred',
            capabilities=[DurableTaskDurability()],
        )

        @agent.tool_plain
        def external() -> str:
            calls.append('requested')
            raise CallDeferred()

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            handle = durable.start(durable_worker.client, 'Hi')
            durable_worker.run_until_idle()
            pending = handle.pending_tools()
            assert len(pending.calls) == 1
            with pytest.raises(UserError, match='not awaiting approval'):
                handle.approve(pending.calls[0].tool_call_id)
            handle.resolve(pending.calls[0].tool_call_id, {'status': 'ok'})
            assert handle.result().output == 'done'
            with pytest.raises(UserError, match='no pending tool requests'):
                handle.pending_tools()
        assert calls == ['requested']

    @pytest.mark.parametrize('unicode_error', [False, True])
    def test_native_activity_retry(self, durable_worker: InMemoryWorker, unicode_error: bool) -> None:
        calls: list[str] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('request')
            if len(calls) == 1:
                if unicode_error:
                    raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid byte')
                raise RuntimeError('temporary failure')
            return ModelResponse(parts=[TextPart('recovered')])

        agent = Agent(
            FunctionModel(model),
            name='retry',
            capabilities=[
                DurableTaskDurability(
                    retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
                )
            ],
        )
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            handle = durable.start(durable_worker.client, 'Hi')
            durable_worker.run_until_idle()
            assert calls == ['request']
            assert len(durable_worker.timers) == 1
            durable_worker.fire_timers()
            assert handle.result().output == 'recovered'
        assert calls == ['request', 'request']

    def test_model_retry_is_control_flow_not_activity_failure(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []
        agent = Agent(TestModel(custom_output_text='done'), name='semantic', capabilities=[DurableTaskDurability()])

        @agent.tool_plain
        def retry_once() -> str:
            calls.append('called')
            if len(calls) == 1:
                raise ModelRetry('Try again')
            return 'ok'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            assert durable.start(durable_worker.client, 'Hi').result().output == 'done'
        assert calls == ['called', 'called']
        assert not durable_worker.timers

    def test_operation_names_are_compatibility_contract(self, durable_worker: InMemoryWorker) -> None:
        with try_import() as mcp_imports:
            from pydantic_ai.mcp import MCPToolset
        if not mcp_imports():
            pytest.skip('mcp not installed')
        from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

        provider = InMemoryProvider()

        class Compat(AbstractCapability[object]):
            id = 'compat'

            @durable_operation('operation')
            async def operation(self, ctx: RunContext[object]) -> None:
                pass

            def get_workspace(self, ctx: RunContext[object], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
                return provider.backend(ref)

        async def events(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        def function_tool() -> str:
            return 'ok'

        def dynamic(ctx: RunContext[object]) -> FunctionToolset[object]:
            return FunctionToolset([function_tool])

        agent = Agent(
            TestModel(),
            name='compat',
            toolsets=[
                FunctionToolset([function_tool], id='functions'),
                DynamicToolset(dynamic, id='dynamic'),
                MCPToolset[object]('http://unused.invalid/mcp', id='mcp'),
            ],
            capabilities=[
                Compat(),
                DurableTaskDurability(models={'registered': TestModel()}, event_stream_handler=events),
            ],
        )

        @agent.tool_plain
        def tool() -> str:
            return 'ok'

        capability = DurableTaskDurability.from_agent(agent)
        assert capability is not None
        names = {registration.__name__ for registration in capability.get_durable_operation_backend().registrations()}
        assert names == {
            'compat__function_toolset__<agent>.call_tool',
            'compat__function_toolset__<agent>.validate_args',
            'compat__function_toolset__functions.call_tool',
            'compat__function_toolset__functions.validate_args',
            'compat__dynamic_toolset__dynamic.get_tools',
            'compat__dynamic_toolset__dynamic.call_tool',
            'compat__dynamic_toolset__dynamic.validate_args',
            'compat__mcp_server__mcp.get_tools',
            'compat__mcp_server__mcp.get_instructions',
            'compat__mcp_server__mcp.call_tool',
            'compat__model.request',
            'compat__model.request_stream',
            'compat__model.compact_messages',
            'compat__model.cancel_suspended_response',
            'compat__capability__compat.operation',
            'compat__capability__workspace.call',
            'compat__event_stream_handler',
        }
        with DurableTaskAgent(agent).register(durable_worker):
            assert set(durable_worker._registry.orchestrators) == {'compat__agent'}  # pyright: ignore[reportPrivateUsage]

    def test_requires_attached_capability(self) -> None:
        with pytest.raises(UserError, match='Attach `DurableTaskDurability`'):
            DurableTaskAgent(Agent(TestModel(), name='missing'))

    def test_replay_on_a_new_worker_does_not_repeat_tools(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []

        def make_agent() -> DurableTaskAgent[object, str]:
            agent = Agent(TestModel(custom_output_text='done'), name='replay', capabilities=[DurableTaskDurability()])

            @agent.tool_plain
            def write() -> str:
                calls.append('written')
                return 'ok'

            return DurableTaskAgent(agent)

        durable = make_agent()
        with durable.register(durable_worker):
            handle = durable.start(durable_worker.client, 'Hi')
            while not calls:
                assert durable_worker.step()
        replacement = InMemoryWorker()
        replacement.instances = durable_worker.instances
        replacement.activity_queue = durable_worker.activity_queue
        new_durable = make_agent()
        with new_durable.register(replacement):
            state = replacement.wait(handle.instance_id)
        assert state is not None
        state.raise_if_failed()
        assert calls == ['written']
        assert replacement.activity_calls == ['replay__model.request']

    def test_json_dependencies(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'),
            deps_type=int,
            name='json-deps',
            capabilities=[DurableTaskDurability[int]()],
        )
        observed: list[int] = []

        @agent.tool
        def read(ctx: RunContext[int]) -> str:
            observed.append(ctx.deps)
            return 'ok'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            assert durable.start(durable_worker.client, 'Hi', deps_input=42).result().output == 'done'
        assert observed == [42]

    def test_inline_tool_has_no_activity(self, durable_worker: InMemoryWorker) -> None:
        config = RoleBasedOperationConfig[task.RetryPolicy | None](
            model=None,
            event=None,
            capability=None,
            tool=None,
            resolve_tool=lambda operation, tool, name: False if name == 'deterministic' else None,
        )
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='inline',
            capabilities=[DurableTaskDurability(operation_config=config)],
        )

        @agent.tool_plain
        def deterministic() -> int:
            return 42

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            assert durable.start(durable_worker.client, 'Hi').result().output == 'done'
        assert durable_worker.activity_calls == ['inline__model.request', 'inline__model.request']

    def test_tool_retry_policy(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []
        policy = task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
        config = RoleBasedOperationConfig[task.RetryPolicy | None](
            model=None,
            event=None,
            capability=None,
            tool=None,
            resolve_tool=lambda operation, tool, name: policy if name == 'transient' else None,
        )
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='tool-retry',
            capabilities=[DurableTaskDurability(operation_config=config)],
        )

        @agent.tool_plain
        def transient() -> str:
            calls.append('called')
            if len(calls) == 1:
                raise RuntimeError('retry me')
            return 'ok'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            handle = durable.start(durable_worker.client, 'Hi')
            durable_worker.run_until_idle()
            assert calls == ['called']
            assert len(durable_worker.timers) == 1
            durable_worker.fire_timers()
            assert handle.result().output == 'done'
        assert calls == ['called', 'called']

    def test_dynamic_toolset_is_rebuilt_in_activities(self, durable_worker: InMemoryWorker) -> None:
        observed: list[str] = []

        def build(ctx: RunContext[object]) -> FunctionToolset[object]:
            toolset = FunctionToolset[object]()

            @toolset.tool_plain
            def dynamic() -> str:
                observed.append('called')
                return 'ok'

            return toolset

        agent = Agent(
            TestModel(custom_output_text='done'),
            toolsets=[DynamicToolset(build, id='dynamic')],
            name='dynamic-agent',
            capabilities=[DurableTaskDurability()],
        )
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            assert durable.start(durable_worker.client, 'Hi').result().output == 'done'
        assert observed == ['called']
        assert 'dynamic-agent__dynamic_toolset__dynamic.get_tools' in durable_worker.activity_calls

    def test_async_timer_uses_durable_clock(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(custom_output_text='done'), name='timer', capabilities=[DurableTaskDurability()])
        durable = DurableTaskAgent(agent)
        lifetimes: list[str] = []

        def orchestrator(
            context: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, str]:
            async def body() -> str:
                lifetimes.append('enter')
                try:
                    await asyncio.sleep(5)
                    return (await agent.run('Hi')).output
                finally:
                    lifetimes.append('exit')

            return (yield from run_durable(context, body))

        with durable.register(durable_worker):
            durable_worker.add_orchestrator(orchestrator)
            instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
            durable_worker.run_until_idle()
            assert len(durable_worker.timers) == 1
            assert lifetimes == ['enter', 'exit']
            durable_worker.fire_timers()
            state = durable_worker.wait(instance_id)
        assert state is not None
        state.raise_if_failed()
        assert state.get_output() == 'done'
        assert lifetimes == ['enter', 'exit'] * 3

    def test_non_durable_wait_fails_instead_of_hanging(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(custom_output_text='done'), name='bad-wait', capabilities=[DurableTaskDurability()])

        @agent.output_validator
        async def block(output: str) -> str:
            await asyncio.Event().wait()
            return output

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            with pytest.raises(OrchestrationFailedError, match='awaited non-durable work'):
                durable.start(durable_worker.client, 'Hi').result()

    def test_live_context_fields_are_not_silently_defaulted(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='context', capabilities=[DurableTaskDurability()])

        @agent.tool
        def messages(ctx: RunContext[object]) -> str:
            return str(len(ctx.messages))

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            with pytest.raises(OrchestrationFailedError, match="'messages' is not available"):
                durable.start(durable_worker.client, 'Hi').result()

    def test_registration_lifetime(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='registration', capabilities=[DurableTaskDurability()])
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            handle = durable.start(durable_worker.client, 'Hi')
            with pytest.raises(UserError, match='already registered'):
                with durable.register(durable_worker):
                    pytest.fail('nested registration must fail')
        with pytest.raises(OrchestrationFailedError, match='before starting the worker'):
            handle.result()

    def test_conflicting_retry_configuration(self) -> None:
        with pytest.raises(UserError, match='not both'):
            DurableTaskDurability(
                retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2),
                operation_config=RoleBasedOperationConfig(model=None, event=None, capability=None, tool=None),
            )

    def test_capability_operations_are_activities(self, durable_worker: InMemoryWorker) -> None:
        observed: list[str] = []

        class Recorder(AbstractCapability[object]):
            id = 'recorder'

            async def before_run(self, ctx: RunContext[object]) -> None:
                await self.record(ctx, 'started')

            @durable_operation('record')
            async def record(self, ctx: RunContext[object], value: str) -> None:
                observed.append(value)
                assert ctx.agent is not None

        agent = Agent(
            TestModel(custom_output_text='done'), name='capability', capabilities=[Recorder(), DurableTaskDurability()]
        )
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            assert durable.start(durable_worker.client, 'Hi').result().output == 'done'
        assert observed == ['started']
        assert durable_worker.activity_calls == [
            'capability__capability__recorder.record',
            'capability__model.request',
        ]

    def test_streamed_model_and_event_handler(self, durable_worker: InMemoryWorker) -> None:
        observed: list[AgentStreamEvent] = []

        async def handler(ctx: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in events:
                observed.append(event)
                assert ctx.agent is not None

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='greet', json_args='{}')}
            else:
                yield 'hello'
                yield ' world'

        agent = Agent(
            FunctionModel(stream_function=stream),
            name='stream',
            capabilities=[DurableTaskDurability(event_stream_handler=handler)],
        )

        @agent.tool_plain
        def greet() -> str:
            return 'hello'

        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            assert durable.start(durable_worker.client, 'Hi').result().output == 'hello world'
        assert any(isinstance(event, PartStartEvent) for event in observed)
        assert 'stream__model.request_stream' in durable_worker.activity_calls
        assert 'stream__event_stream_handler' in durable_worker.activity_calls

    def test_cancel_suspended_response(self, durable_worker: InMemoryWorker) -> None:
        cancelled: list[str] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response.state)

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(message, ModelResponse) for message in messages):
                return ModelResponse(parts=[TextPart('partial')], state='suspended')
            raise RuntimeError('continuation failed')

        agent = Agent(CancellableModel(model), name='cancel', capabilities=[DurableTaskDurability()])
        durable = DurableTaskAgent(agent)
        with durable.register(durable_worker):
            with pytest.raises(OrchestrationFailedError, match='continuation failed'):
                durable.start(durable_worker.client, 'Hi').result()
        assert cancelled == ['suspended']
        assert 'cancel__model.cancel_suspended_response' in durable_worker.activity_calls

    def test_new_earlier_timer_does_not_wait_for_the_original_timer(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='timers', capabilities=[DurableTaskDurability()])
        durable = DurableTaskAgent(agent)

        def orchestrator(
            context: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, list[int]]:
            async def body() -> list[int]:
                completed: list[int] = []

                async def later() -> None:
                    await asyncio.sleep(10)
                    completed.append(10)

                async def earlier() -> None:
                    await agent.run('Hi')
                    await asyncio.sleep(1)
                    completed.append(1)

                await asyncio.gather(later(), earlier())
                return completed

            return (yield from run_durable(context, body))

        with durable.register(durable_worker):
            durable_worker.add_orchestrator(orchestrator)
            instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
            durable_worker.run_until_idle()
            assert len(durable_worker.timers) == 2
            durable_worker.fire_timers()
            assert durable_worker.now == datetime(2026, 1, 1, 0, 0, 1)
            durable_worker.run_until_idle()
            assert len(durable_worker.timers) == 1
            durable_worker.fire_timers()
            state = durable_worker.wait(instance_id)
        assert state is not None
        state.raise_if_failed()
        assert state.get_output() == [1, 10]

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterable, AsyncIterator, Generator
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product

import pytest
from pydantic import BaseModel, TypeAdapter

from pydantic_ai import (
    Agent,
    CallDeferred,
    CustomEvent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
    ToolApproved,
    ToolDenied,
)
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, durable_operation
from pydantic_ai.durable_exec import (
    CapabilityOperationId,
    DurableOperation,
    conversation_id_from_run_id,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartStartEvent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

    from .worker import (
        EmptyCacheIdentity,
        IdentityTransport,
        InMemoryWorker,
        JsonResultCodec,
        agent_orchestrator,
        completed,
        registered_agent,
        run_agent,
        schedule_agent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


@dataclass
class _DataclassDeps:
    tenant: str


class _ModelDeps(BaseModel):
    tenant: str


class TestAzureDurableTaskDurability:
    @pytest.mark.parametrize('decision', ['approve', 'deny', 'retry', 'case-sensitive'])
    def test_deferred_calls_resume_after_external_event(self, durable_worker: InMemoryWorker, decision: str) -> None:
        """Exercise SDK event replay and deferred control flow without a scheduler or model service."""
        requests = 0

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal requests
            requests += 1
            if requests == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart('approve_payment', {'amount': 1}, tool_call_id='call_X'),
                        ToolCallPart(
                            'approve_payment' if decision == 'case-sensitive' else 'external_lookup',
                            {'amount': 2} if decision == 'case-sensitive' else {},
                            tool_call_id='call_x',
                        ),
                    ]
                )
            if decision == 'retry' and requests == 2:
                assert any(
                    isinstance(part, ToolReturnPart) and part.outcome == 'retried' and 'Try again' in str(part.content)
                    for message in messages
                    for part in message.parts
                )
                return ModelResponse(parts=[ToolCallPart('external_lookup', {}, tool_call_id='call_x')])
            return ModelResponse(
                parts=[
                    TextPart(
                        '|'.join(
                            str(part.content)
                            for message in messages
                            for part in message.parts
                            if isinstance(part, ToolReturnPart)
                        )
                    )
                ]
            )

        agent = Agent(
            FunctionModel(model),
            name='review',
            output_type=[str, DeferredToolRequests],
            capabilities=[AzureDurableTaskDurability()],
        )

        @agent.tool_plain(requires_approval=True)
        def approve_payment(amount: int) -> str:
            return f'Paid {amount}'

        @agent.tool_plain
        def external_lookup() -> str:
            raise CallDeferred()

        def orchestrator(ctx: task.OrchestrationContext, prompt: str) -> Generator[task.Task[object], object, str]:
            result = yield from run_durable(ctx, lambda: agent.run(prompt))
            pause = 0
            while isinstance(result.output, DeferredToolRequests):
                ctx.set_custom_status({'event': f'review:{pause}'})
                payload = yield ctx.wait_for_external_event(f'review:{pause}')
                approvals = TypeAdapter(dict[str, bool]).validate_python(payload, strict=True)
                deferred = DeferredToolResults(approvals=dict(approvals))
                if deferred.approvals:
                    deferred.approvals['call_X'] = (
                        ToolDenied('Payment blocked')
                        if decision == 'deny'
                        else ToolApproved(override_args={'amount': 7})
                    )
                for call in result.output.calls:
                    deferred.calls[call.tool_call_id] = (
                        ModelRetry('Try again') if decision == 'retry' and pause == 0 else {'status': 'complete'}
                    )
                result = yield from run_durable(
                    ctx,
                    lambda: agent.run(
                        message_history=result.all_messages(),
                        deferred_tool_results=deferred,
                        usage=result.usage,
                        conversation_id=result.conversation_id,
                    ),
                )
                pause += 1
            ctx.set_custom_status(None)
            return result.output

        with registered_agent(durable_worker, agent):
            durable_worker.add_orchestrator(orchestrator)
            instance_id = durable_worker.schedule(orchestrator, input='Review')
            durable_worker.run_until_idle()
            state = durable_worker.state(instance_id)
            assert state.get_custom_status() == {'event': 'review:0'}
            approvals = {'call_X': True}
            if decision == 'case-sensitive':
                approvals['call_x'] = False
            durable_worker.raise_event(instance_id, 'REVIEW:0', data=approvals)
            if decision == 'retry':
                durable_worker.run_until_idle()
                state = durable_worker.state(instance_id)
                assert state.get_custom_status() == {'event': 'review:1'}
                durable_worker.raise_event(instance_id, 'review:0', data=approvals)
                durable_worker.run_until_idle()
                state = durable_worker.state(instance_id)
                assert state.get_custom_status() == {'event': 'review:1'}
                durable_worker.raise_event(instance_id, 'review:1', data={})
            output = completed(durable_worker, instance_id)
        assert isinstance(output, str)
        assert ('Paid 7' in output) is (decision != 'deny')
        assert ('Payment blocked' in output) is (decision == 'deny')
        assert ('Paid 2' not in output) and (
            'denied' in output if decision == 'case-sensitive' else 'complete' in output
        )
        assert requests == (3 if decision == 'retry' else 2)
        state = durable_worker.state(instance_id)
        assert state.get_custom_status() is None

    def test_happy_path(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(
            TestModel(custom_output_text='hello'), name='support', capabilities=[AzureDurableTaskDurability()]
        )
        with registered_agent(durable_worker, agent):
            result = run_agent(durable_worker, agent, 'Hi', instance_id='hello', deps=None)
        assert result.output == 'hello'
        assert durable_worker.activity_calls == ['support__model.request']

    @pytest.mark.parametrize('same_agent', [False, True])
    def test_multiple_runs_have_distinct_replay_stable_ids(
        self, durable_worker: InMemoryWorker, same_agent: bool
    ) -> None:
        first = Agent(TestModel(), name='first', capabilities=[AzureDurableTaskDurability()])
        second = first if same_agent else Agent(TestModel(), name='second', capabilities=[AzureDurableTaskDurability()])
        episodes: list[list[str]] = []

        def orchestrator(
            context: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, list[str]]:
            run_ids: list[str] = []
            episodes.append(run_ids)
            for agent in (first, second):
                result = yield from run_durable(context, lambda: agent.run('Hi'))
                assert result.conversation_id == conversation_id_from_run_id(result.run_id)
                run_ids.append(result.run_id)
            return run_ids

        durable_worker.add_orchestrator(orchestrator)
        with ExitStack() as stack:
            stack.enter_context(registered_agent(durable_worker, first))
            if not same_agent:
                stack.enter_context(registered_agent(durable_worker, second))
            instance_id = durable_worker.schedule(orchestrator)
            run_ids = completed(durable_worker, instance_id)
        assert isinstance(run_ids, list) and len(run_ids) == 2
        assert run_ids[0] != run_ids[1]
        assert episodes == [[], [run_ids[0]], run_ids]
        assert isinstance(run_ids[0], str) and run_ids[0].startswith(f'{instance_id}:first:')
        assert isinstance(run_ids[1], str) and run_ids[1].startswith(f'{instance_id}:{second.name}:')

        replacement = InMemoryWorker()
        replacement.instances = durable_worker.instances
        replacement.add_orchestrator(orchestrator)
        with ExitStack() as stack:
            stack.enter_context(registered_agent(replacement, first))
            if not same_agent:
                stack.enter_context(registered_agent(replacement, second))
            replacement.orchestration_turn(instance_id)
        assert episodes[-1] == run_ids
        assert replacement.activity_calls == []

    def test_run_ids_change_after_continue_as_new(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='continued', capabilities=[AzureDurableTaskDurability()])
        generations: dict[bool, set[str]] = {False: set(), True: set()}

        def orchestrator(
            context: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, str]:
            result = yield from run_durable(context, lambda: agent.run('Hi'))
            generations[bool(payload)].add(result.run_id)
            if not payload:
                yield from run_durable(context, lambda: asyncio.sleep(1))
                context.continue_as_new(True)
            return result.run_id

        durable_worker.add_orchestrator(orchestrator)
        with registered_agent(durable_worker, agent):
            instance_id = durable_worker.schedule(orchestrator)
            durable_worker.run_until_idle()
            assert len(generations[False]) == 1
            durable_worker.fire_timers()
            result = completed(durable_worker, instance_id)
        assert len(generations[True]) == 1
        assert generations[False].isdisjoint(generations[True])
        assert result in generations[True]

    def test_outside_orchestration(self) -> None:
        agent = Agent(TestModel(custom_output_text='plain'), name='plain', capabilities=[AzureDurableTaskDurability()])
        assert agent.run_sync('Hi').output == 'plain'

    def test_multiple_durability_capabilities_are_rejected(self) -> None:
        with pytest.raises(UserError, match='only one durable execution engine'):
            Agent(
                TestModel(), name='duplicate', capabilities=[AzureDurableTaskDurability(), AzureDurableTaskDurability()]
            )

    def test_structured_output(self, durable_worker: InMemoryWorker) -> None:
        class Output(BaseModel):
            value: int

        agent = Agent(
            TestModel(custom_output_args={'value': 42}),
            output_type=Output,
            name='structured',
            capabilities=[AzureDurableTaskDurability()],
        )
        with registered_agent(durable_worker, agent):
            result = run_agent(durable_worker, agent, 'Hi', deps=None)
        assert isinstance(result.output, Output)
        assert result.output.value == 42

    def test_callable_output(self, durable_worker: InMemoryWorker) -> None:
        def output(value: int) -> str:
            return f'value={value}'

        agent = Agent(
            TestModel(custom_output_args={'value': 42}),
            output_type=output,
            name='callable',
            capabilities=[AzureDurableTaskDurability()],
        )
        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'value=42'

    def test_instrumented_agent(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='instrumented',
            capabilities=[Instrumentation(), AzureDurableTaskDurability()],
        )
        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'

    def test_parallel_tools_are_scheduled_together(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []
        agent = Agent(
            TestModel(custom_output_text='done'), name='parallel', capabilities=[AzureDurableTaskDurability()]
        )

        @agent.tool_plain
        def first() -> str:
            calls.append('first')
            return 'one'

        @agent.tool_plain
        def second() -> str:
            calls.append('second')
            return 'two'

        with registered_agent(durable_worker, agent):
            handle = schedule_agent(durable_worker, agent, 'Hi', deps=None)
            assert durable_worker.step()  # Schedule the model.
            assert durable_worker.step()  # Execute the model.
            assert durable_worker.step()  # Schedule both argument validators, before either runs.
            assert len(durable_worker.activity_queue) == 2
            assert completed(durable_worker, handle) == 'done'
        assert calls == ['first', 'second']
        assert durable_worker.activity_calls.count('parallel__function_toolset__-x3C-agent-x3E-.call_tool') == 2

    def test_model_unicode_error_is_retried_not_a_serialization_failure(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls.append('request')
            if len(calls) == 1:
                raise UnicodeDecodeError('utf-8', b'\xff', 0, 1, 'invalid byte')
            return ModelResponse(parts=[TextPart('recovered')])

        agent = Agent(
            FunctionModel(model),
            name='retry',
            capabilities=[
                AzureDurableTaskDurability(
                    model_options=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
                )
            ],
        )
        with registered_agent(durable_worker, agent):
            handle = schedule_agent(durable_worker, agent, 'Hi', deps=None)
            durable_worker.run_until_idle()
            assert calls == ['request']
            assert len(durable_worker.timers) == 1
            durable_worker.fire_timers()
            assert completed(durable_worker, handle) == 'recovered'
        assert calls == ['request', 'request']

    def test_model_retry_is_control_flow_not_activity_failure(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []
        agent = Agent(
            TestModel(custom_output_text='done'), name='semantic', capabilities=[AzureDurableTaskDurability()]
        )

        @agent.tool_plain
        def retry_once() -> str:
            calls.append('called')
            if len(calls) == 1:
                raise ModelRetry('Try again')
            return 'ok'

        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
        assert calls == ['called', 'called']
        assert not durable_worker.timers

    def test_operation_names_are_compatibility_contract(self) -> None:
        with try_import() as mcp_imports:
            from pydantic_ai.mcp import MCPToolset
        if not mcp_imports():
            # Optional-extra probes can skip here; all-extras coverage does not.
            pytest.skip('mcp not installed')  # pragma: lax no cover

        provider = InMemoryProvider()

        # Name discovery inspects callback signatures without executing their bodies.
        class Compat(AbstractCapability[object]):
            id = 'compat'

            @durable_operation('operation')
            async def operation(self, ctx: RunContext[object]) -> None:
                pass

            def get_workspace(self, ctx: RunContext[object], *, ref: WorkspaceRef | None) -> WorkspaceBackend:
                return provider.backend(ref)  # pragma: no cover

        async def events(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass

        def function_tool() -> str:
            return 'ok'  # pragma: no cover

        def dynamic(ctx: RunContext[object]) -> FunctionToolset[object]:
            return FunctionToolset([function_tool])  # pragma: no cover

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
                AzureDurableTaskDurability(models={'registered': TestModel()}, event_stream_handler=events),
            ],
        )

        @agent.tool_plain
        def tool() -> str:
            return 'ok'  # pragma: no cover

        capability = AzureDurableTaskDurability.from_agent(agent)
        assert capability is not None
        names = {registration.__name__ for registration in capability.get_durable_operation_backend().registrations()}
        assert names == {
            'compat__function_toolset__-x3C-agent-x3E-.call_tool',
            'compat__function_toolset__-x3C-agent-x3E-.validate_args',
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

    def test_activity_name_escaping_preserves_safe_punctuation(self) -> None:
        all_names: set[str] = set()
        for name, escaped in [
            ('safe_Name.-09`+:', 'safe_Name.--09`+:'),
            ('<agent>', '-x3C-agent-x3E-'),
            ('_x3C_agent_x3E_', '_x3C_agent_x3E_'),
            ('-x3C-agent-x3E-', '--x3C--agent--x3E--'),
            ('has_xmarker', 'has_xmarker'),
            ('name/\u00e9', 'name-x2F--xE9-'),
            ('\U0001f600', '-x1F600-'),
            ('<AB', '-x3C-AB'),
            ('\u3cab', '-x3CAB-'),
        ]:
            agent = Agent(TestModel(), name=name, capabilities=[AzureDurableTaskDurability()])
            capability = AzureDurableTaskDurability.from_agent(agent)
            assert capability is not None
            names = {
                registration.__name__ for registration in capability.get_durable_operation_backend().registrations()
            }
            assert all(value.startswith(f'{escaped}__') for value in names)
            assert names.isdisjoint(all_names)
            all_names.update(names)

    def test_activity_name_escaping_is_injective(self) -> None:
        names: set[str] = set()
        for length in (1, 2):
            for characters in product('a_-x03C<\u00e9\U0001f600', repeat=length):
                agent = Agent(TestModel(), name=''.join(characters), capabilities=[AzureDurableTaskDurability()])
                capability = AzureDurableTaskDurability.from_agent(agent)
                assert capability is not None
                generated = {
                    activity.__name__ for activity in capability.get_durable_operation_backend().registrations()
                }
                assert all(re.fullmatch(r'[A-Za-z0-9_.`+:-]{1,100}', name) for name in generated)
                assert generated.isdisjoint(names)
                names.update(generated)

    def test_long_activity_names_are_a_compatibility_contract(self) -> None:
        agent = Agent(TestModel(), name='long-' + 'a' * 120, capabilities=[AzureDurableTaskDurability()])
        capability = AzureDurableTaskDurability.from_agent(agent)
        assert capability is not None
        names = {activity.__name__ for activity in capability.get_durable_operation_backend().registrations()}
        assert names == {
            f'long--{"a" * 77}-{suffix}'
            for suffix in [
                'ad5823062676cd02',
                'bd7afee1bdb903e9',
                '29a6f88c60af8609',
                '8b6d0ca92d02859d',
                '688888820165bdcf',
                '4a56fe7a73fcb1b6',
            ]
        }
        assert all(len(name) == 100 for name in names)

    def test_replay_on_a_new_worker_does_not_repeat_tools(self, durable_worker: InMemoryWorker) -> None:
        calls: list[str] = []

        def make_agent() -> Agent[object, str]:
            agent = Agent(
                TestModel(custom_output_text='done'), name='replay', capabilities=[AzureDurableTaskDurability()]
            )

            @agent.tool_plain
            def write() -> str:
                calls.append('written')
                return 'ok'

            return agent

        durable = make_agent()
        with registered_agent(durable_worker, durable):
            handle = schedule_agent(durable_worker, durable, 'Hi', deps=None)
            while not calls:
                assert durable_worker.step()
        replacement = InMemoryWorker()
        replacement.instances = durable_worker.instances
        replacement.activity_queue = durable_worker.activity_queue
        new_durable = make_agent()
        replacement.add_orchestrator(agent_orchestrator(new_durable, deps=None))
        with registered_agent(replacement, new_durable):
            state = replacement.wait(handle)
        state.raise_if_failed()
        assert calls == ['written']
        assert replacement.activity_calls == ['replay__model.request']

    @pytest.mark.parametrize('metadata', [False, True])
    def test_inline_tool_has_no_activity(self, durable_worker: InMemoryWorker, metadata: bool) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='inline',
            capabilities=[
                AzureDurableTaskDurability(
                    resolve_tool_options=None if metadata else lambda operation, tool, name: False
                )
            ],
        )
        observed: list[bool] = []
        durability = AzureDurableTaskDurability.from_agent(agent)
        assert durability is not None

        @agent.tool_plain(metadata={'azure_durable_task': False} if metadata else None)
        def deterministic() -> int:
            observed.append(durability.in_durable_context)
            return 42

        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
        assert durable_worker.activity_calls == ['inline__model.request', 'inline__model.request']
        assert observed
        assert all(observed)

    @pytest.mark.parametrize('resolution', ['inherit', 'override', 'disabled'])
    def test_resolver_sets_the_tool_retry_policy(self, durable_worker: InMemoryWorker, resolution: str) -> None:
        calls: list[str] = []
        policy = task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
        resolved = {
            'inherit': None,
            'override': policy,
            'disabled': task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=1),
        }[resolution]
        enabled = resolution != 'disabled'
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='tool-retry',
            capabilities=[
                AzureDurableTaskDurability(
                    tool_options=None if resolution == 'override' else policy,
                    resolve_tool_options=lambda operation, tool, name: resolved if name == 'transient' else None,
                )
            ],
        )

        @agent.tool_plain
        def transient() -> str:
            calls.append('called')
            if len(calls) == 1:
                raise RuntimeError('retry me')
            return 'ok'

        with registered_agent(durable_worker, agent):
            handle = schedule_agent(durable_worker, agent, 'Hi', deps=None)
            durable_worker.run_until_idle()
            assert calls == ['called']
            if enabled:
                assert len(durable_worker.timers) == 1
                durable_worker.fire_timers()
                assert completed(durable_worker, handle) == 'done'
            else:
                assert durable_worker.timers == []
                with pytest.raises(OrchestrationFailedError, match='retry me'):
                    completed(durable_worker, handle)
        assert calls == (['called', 'called'] if enabled else ['called'])

    @pytest.mark.parametrize('role', ['model', 'tool', 'event', 'capability'])
    def test_per_role_options(self, durable_worker: InMemoryWorker, role: str) -> None:
        attempts: dict[str, int] = {}

        def attempt(operation_role: str) -> None:
            attempts[operation_role] = attempts.get(operation_role, 0) + 1
            if operation_role == role and attempts[operation_role] == 1:
                raise RuntimeError(f'transient {role}')

        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            attempt('model')
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='lookup', json_args='{}', tool_call_id='call')}
            else:
                yield 'done'

        async def events(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                # Model stream events are delivered inside the model activity.
                if isinstance(event, FunctionToolCallEvent):
                    attempt('event')

        class Recorder(AbstractCapability[object]):
            id = 'recorder'

            async def before_run(self, ctx: RunContext[object]) -> None:
                await self.record(ctx)

            @durable_operation('record')
            async def record(self, ctx: RunContext[object]) -> None:
                attempt('capability')

        policy = task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
        agent = Agent(
            FunctionModel(stream_function=stream),
            name='role-options',
            capabilities=[
                Recorder(),
                AzureDurableTaskDurability(
                    model_options=policy if role == 'model' else None,
                    tool_options=policy if role == 'tool' else None,
                    event_options=policy if role == 'event' else None,
                    capability_options=policy if role == 'capability' else None,
                    event_stream_handler=events,
                ),
            ],
        )

        @agent.tool_plain
        def lookup() -> int:
            attempt('tool')
            return 42

        with registered_agent(durable_worker, agent):
            instance_id = schedule_agent(durable_worker, agent, 'Hi', deps=None)
            durable_worker.run_until_idle()
            assert attempts[role] == 1
            assert len(durable_worker.timers) == 1
            durable_worker.fire_timers()
            assert completed(durable_worker, instance_id) == 'done'
        assert attempts[role] >= 2
        assert set(attempts) == {'model', 'tool', 'event', 'capability'}
        assert durable_worker.timers == []

    @pytest.mark.parametrize('metadata', [False, True])
    def test_dynamic_tools_cannot_opt_out(self, durable_worker: InMemoryWorker, metadata: bool) -> None:
        def build(ctx: RunContext[object]) -> FunctionToolset[object]:
            toolset = FunctionToolset[object]()

            @toolset.tool_plain(metadata={'azure_durable_task': False} if metadata else None)
            def lookup() -> int:
                return 42  # pragma: no cover

            return toolset

        agent = Agent(
            TestModel(custom_output_text='done'),
            name='dynamic-opt-out',
            toolsets=[DynamicToolset(build, id='dynamic')],
            capabilities=[
                AzureDurableTaskDurability(
                    resolve_tool_options=None if metadata else lambda operation, tool, name: False
                )
            ],
        )
        with registered_agent(durable_worker, agent):
            with pytest.raises(
                OrchestrationFailedError,
                match=r"metadata=\{'azure_durable_task': False\}.* can opt only function tools"
                if metadata
                else '`resolve_tool_options` can opt only function tools',
            ):
                run_agent(durable_worker, agent, 'Hi', deps=None)
        assert not any(name.endswith('.call_tool') for name in durable_worker.activity_calls)

    def test_resolver_receives_original_function_name(self, durable_worker: InMemoryWorker) -> None:
        toolset = FunctionToolset[object](id='named')
        names: list[str] = []

        @toolset.tool_plain
        def lookup() -> int:
            return 42

        def options(operation: object, tool: object | None, name: str) -> None:
            if tool is not None:
                names.append(name)

        agent = Agent(
            TestModel(custom_output_text='done'),
            name='original-name',
            toolsets=[toolset.prefixed('visible')],
            capabilities=[AzureDurableTaskDurability(resolve_tool_options=options)],
        )
        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
        assert set(names) == {'lookup'}

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
            capabilities=[AzureDurableTaskDurability()],
        )
        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
        assert observed == ['called']
        assert 'dynamic--agent__dynamic_toolset__dynamic.get_tools' in durable_worker.activity_calls

    @pytest.mark.parametrize('resolver', [False, True])
    @pytest.mark.parametrize('kind', ['dynamic', 'mcp-tools', 'mcp-instructions'])
    def test_discovery_inherits_its_bound_retry_policy(
        self, durable_worker: InMemoryWorker, kind: str, resolver: bool
    ) -> None:
        attempts: list[str] = []

        def transient_discovery() -> None:
            attempts.append('discovery')
            if len(attempts) == 1:
                raise RuntimeError('transient discovery')

        toolset: AbstractToolset[object]
        if kind == 'dynamic':

            def build(ctx: RunContext[object]) -> FunctionToolset[object]:
                transient_discovery()
                return FunctionToolset()

            toolset = DynamicToolset(build, id='dynamic')
        else:
            with try_import() as mcp_imports:
                from fastmcp import FastMCP

                from pydantic_ai.mcp import MCPToolset
            if not mcp_imports():
                # Optional-extra runs may lack MCP; all-extras coverage exercises these cases.
                pytest.skip('mcp not installed')  # pragma: lax no cover

            class FlakyMCP(MCPToolset[object]):
                async def get_tools(self, ctx: RunContext[object]) -> dict[str, ToolsetTool[object]]:
                    if kind == 'mcp-tools':
                        transient_discovery()
                    return await super().get_tools(ctx)

                async def get_instructions(self, ctx: RunContext[object]) -> InstructionPart | None:
                    if kind == 'mcp-instructions':
                        transient_discovery()
                    return await super().get_instructions(ctx)

            server: FastMCP[None] = FastMCP(name='discovery', instructions='Use deterministic tools.')
            toolset = FlakyMCP(server, id='mcp', include_instructions=True)

        policy = task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
        durability = AzureDurableTaskDurability(
            tool_options=None if resolver else policy,
            resolve_tool_options=(lambda operation, tool, name: policy) if resolver else None,
        )
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='discovery-retry',
            toolsets=[toolset],
            capabilities=[durability],
        )
        with registered_agent(durable_worker, agent):
            instance_id = schedule_agent(durable_worker, agent, 'Hi', deps=None)
            durable_worker.run_until_idle()
            assert attempts == ['discovery']
            assert len(durable_worker.timers) == 1
            durable_worker.fire_timers()
            assert completed(durable_worker, instance_id) == 'done'
        assert attempts == ['discovery', 'discovery']

    def test_async_timer_uses_durable_clock(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(custom_output_text='done'), name='timer', capabilities=[AzureDurableTaskDurability()])

        def orchestrator(
            context: task.OrchestrationContext, payload: object
        ) -> Generator[task.Task[object], object, str]:
            async def body() -> str:
                await asyncio.sleep(5)
                return (await agent.run('Hi')).output

            return (yield from run_durable(context, body))

        with registered_agent(durable_worker, agent):
            durable_worker.add_orchestrator(orchestrator)
            instance_id = durable_worker.schedule(orchestrator)
            durable_worker.run_until_idle()
            assert len(durable_worker.timers) == 1
            assert durable_worker.activity_calls == []
            durable_worker.fire_timers()
            state = durable_worker.wait(instance_id)
        state.raise_if_failed()
        assert state.get_output() == 'done'

    def test_non_durable_wait_fails_instead_of_hanging(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'), name='bad-wait', capabilities=[AzureDurableTaskDurability()]
        )

        @agent.output_validator
        async def block(output: str) -> str:
            await asyncio.Event().wait()
            # The unresolved non-durable wait must fail before this return.
            return output  # pragma: no cover

        with registered_agent(durable_worker, agent):
            with pytest.raises(OrchestrationFailedError, match='awaited non-durable work'):
                run_agent(durable_worker, agent, 'Hi', deps=None)

    def test_live_context_fields_are_not_silently_defaulted(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='context', capabilities=[AzureDurableTaskDurability()])

        @agent.tool
        def messages(ctx: RunContext[object]) -> str:
            return str(len(ctx.messages))

        with registered_agent(durable_worker, agent):
            with pytest.raises(OrchestrationFailedError, match="'messages' is not available"):
                run_agent(durable_worker, agent, 'Hi', deps=None)

    def test_registration_lifetime(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='registration', capabilities=[AzureDurableTaskDurability()])
        with registered_agent(durable_worker, agent):
            handle = schedule_agent(durable_worker, agent, 'Hi', deps=None)
            with pytest.raises(UserError, match='already registered'), registered_agent(durable_worker, agent):
                pass
        with pytest.raises(OrchestrationFailedError, match='before starting the worker'):
            completed(durable_worker, handle)

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
            TestModel(custom_output_text='done'),
            name='capability',
            capabilities=[Recorder(), AzureDurableTaskDurability()],
        )
        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
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
            capabilities=[AzureDurableTaskDurability(event_stream_handler=handler)],
        )

        @agent.tool_plain
        def greet() -> str:
            return 'hello'

        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'hello world'
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

        agent = Agent(CancellableModel(model), name='cancel', capabilities=[AzureDurableTaskDurability()])
        with registered_agent(durable_worker, agent):
            with pytest.raises(OrchestrationFailedError, match='continuation failed'):
                run_agent(durable_worker, agent, 'Hi', deps=None)
        assert cancelled == ['suspended']
        assert 'cancel__model.cancel_suspended_response' in durable_worker.activity_calls

    def test_new_earlier_timer_does_not_wait_for_the_original_timer(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(TestModel(), name='timers', capabilities=[AzureDurableTaskDurability()])

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

        with registered_agent(durable_worker, agent):
            durable_worker.add_orchestrator(orchestrator)
            instance_id = durable_worker.schedule(orchestrator)
            durable_worker.run_until_idle()
            assert len(durable_worker.timers) == 2
            durable_worker.fire_timers()
            assert durable_worker.now == datetime(2026, 1, 1, 0, 0, 1)
            durable_worker.run_until_idle()
            assert len(durable_worker.timers) == 1
            durable_worker.fire_timers()
            state = durable_worker.wait(instance_id)
        state.raise_if_failed()
        assert state.get_output() == [1, 10]

    def test_user_error_is_terminal_under_native_retries(self, durable_worker: InMemoryWorker) -> None:
        effects: list[str] = []
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='terminal-user-error',
            capabilities=[
                AzureDurableTaskDurability(
                    tool_options=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
                )
            ],
        )

        @agent.tool_plain
        def submit() -> str:
            effects.append('written')
            raise UserError('The tool configuration is invalid.')

        with registered_agent(durable_worker, agent):
            run = schedule_agent(durable_worker, agent, 'Hi', deps=None)
            with pytest.raises(OrchestrationFailedError, match='The tool configuration is invalid'):
                completed(durable_worker, run)
        assert effects == ['written']
        assert durable_worker.timers == []

    @pytest.mark.parametrize('deps', [42, _DataclassDeps(tenant='acme'), _ModelDeps(tenant='acme')], ids=type)
    def test_dependencies_are_rebuilt_in_activities(self, durable_worker: InMemoryWorker, deps: object) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'),
            deps_type=type(deps),
            name='deps',
            capabilities=[AzureDurableTaskDurability()],
        )
        observed: list[object] = []

        @agent.tool
        def read(ctx: RunContext[object]) -> str:
            observed.append(ctx.deps)
            return 'ok'

        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=deps).output == 'done'
        assert observed == [deps]


def test_unbound_capability_cannot_register() -> None:
    with pytest.raises(UserError, match=r'Bind .* to an agent'):
        AzureDurableTaskDurability().get_durable_operation_backend()


def test_backend_outside_orchestration_and_invalid_native_policy(durable_worker: InMemoryWorker) -> None:
    async def echo(value: str) -> str:
        return value

    capability = AzureDurableTaskDurability.from_agent(
        Agent(TestModel(), name='custom', capabilities=[AzureDurableTaskDurability()])
    )
    assert capability is not None
    backend = capability.get_durable_operation_backend()
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
    instance_id = durable_worker.schedule(orchestrator)
    state = durable_worker.wait(instance_id)
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


def test_same_tool_calls_keep_replay_identity_when_completion_order_changes(durable_worker: InMemoryWorker) -> None:
    calls: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('lookup', {'key': 'left'}, tool_call_id='call-left'),
                    ToolCallPart('lookup', {'key': 'right'}, tool_call_id='call-right'),
                ]
            )
        results = {
            part.tool_call_id: part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart)
        }
        assert results == {'call-left': 'left-result', 'call-right': 'right-result'}
        return ModelResponse(parts=[TextPart('done')])

    def make_agent() -> Agent[object, str]:
        agent = Agent(FunctionModel(model), name='same-tool', capabilities=[AzureDurableTaskDurability()])

        @agent.tool_plain
        def lookup(key: str) -> str:
            calls.append(key)
            return f'{key}-result'

        return agent

    durable = make_agent()
    with registered_agent(durable_worker, durable):
        run = schedule_agent(durable_worker, durable, 'Hi', deps=None)
        for _ in range(20):
            if not (
                len(durable_worker.activity_queue) == 2
                and all(name.endswith('.call_tool') for _, name, _, _ in durable_worker.activity_queue)
            ):
                assert durable_worker.step()
        assert len(durable_worker.activity_queue) == 2
        assert all(name.endswith('.call_tool') for _, name, _, _ in durable_worker.activity_queue)
        durable_worker.activity_queue.reverse()
        assert durable_worker.step()
        assert calls == ['right']
        assert durable_worker.step()
    replacement = InMemoryWorker()
    replacement.instances = durable_worker.instances
    replacement.activity_queue = durable_worker.activity_queue
    replacement_agent = make_agent()
    replacement.add_orchestrator(agent_orchestrator(replacement_agent, deps=None))
    with registered_agent(replacement, replacement_agent):
        assert completed(replacement, run) == 'done'
    assert calls == ['right', 'left']
    assert replacement.activity_calls == [
        'same--tool__function_toolset__-x3C-agent-x3E-.call_tool',
        'same--tool__model.request',
    ]

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import AsyncIterable, AsyncIterator, Generator
from datetime import datetime, timedelta
from itertools import product
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
    ModelRetry,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability, Instrumentation, durable_operation
from pydantic_ai.durable_exec import (
    BaseDurabilityCapability,
    CapabilityOperationId,
    DurableOperation,
    RoleBasedOperationConfig,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    InstructionPart,
    ModelMessage,
    ModelResponse,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset, ToolsetTool

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


class TestAzureDurableTaskDurability:
    def test_only_existing_private_hooks_are_overridden(self) -> None:
        private_methods = {
            name for name in vars(AzureDurableTaskDurability) if name.startswith('_') and not name.startswith('__')
        }
        assert private_methods - {'_activity_scope', '_restore_context'} <= set(dir(BaseDurabilityCapability))
        assert {
            'get_default_run_id',
            'get_default_conversation_id',
            'get_run_start_time',
            'get_run_clock',
        } <= vars(AzureDurableTaskDurability).keys()

    def test_happy_path(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(
            TestModel(custom_output_text='hello'), name='support', capabilities=[AzureDurableTaskDurability()]
        )
        with registered_agent(durable_worker, agent):
            result = run_agent(durable_worker, agent, 'Hi', instance_id='hello', deps=None)
        assert result.output == 'hello'
        assert result.run_id == 'hello:support:1'
        assert durable_worker.activity_calls == ['support____model.request']

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
        assert result.all_messages()
        assert result.usage.requests == 1

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
            result = run_agent(durable_worker, agent, 'Hi', deps=None)
        assert result.output == 'done'
        assert result.usage.requests == 1

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
        assert durable_worker.activity_calls.count('parallel____function__toolset_____x3C_agent_x3E_.call__tool') == 2

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
                AzureDurableTaskDurability(
                    retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
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

    def test_operation_names_are_compatibility_contract(
        self, durable_worker: InMemoryWorker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with try_import() as mcp_imports:
            from pydantic_ai.mcp import MCPToolset
        if not mcp_imports():
            # Optional-extra probes can skip here; all-extras coverage does not.
            pytest.skip('mcp not installed')  # pragma: lax no cover
        from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

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
            'compat____function__toolset_____x3C_agent_x3E_.call__tool',
            'compat____function__toolset_____x3C_agent_x3E_.validate__args',
            'compat____function__toolset____functions.call__tool',
            'compat____function__toolset____functions.validate__args',
            'compat____dynamic__toolset____dynamic.get__tools',
            'compat____dynamic__toolset____dynamic.call__tool',
            'compat____dynamic__toolset____dynamic.validate__args',
            'compat____mcp__server____mcp.get__tools',
            'compat____mcp__server____mcp.get__instructions',
            'compat____mcp__server____mcp.call__tool',
            'compat____model.request',
            'compat____model.request__stream',
            'compat____model.compact__messages',
            'compat____model.cancel__suspended__response',
            'compat____capability____compat.operation',
            'compat____capability____workspace.call',
            'compat____event__stream__handler',
        }
        add_orchestrator = MagicMock()
        monkeypatch.setattr(durable_worker, 'add_orchestrator', add_orchestrator)
        with capability.register(durable_worker):
            add_orchestrator.assert_not_called()

    def test_activity_name_escaping_preserves_safe_punctuation(self) -> None:
        all_names: set[str] = set()
        for name, escaped in [
            ('safe_Name.-09`+:', 'safe__Name.-09`+:'),
            ('<agent>', '_x3C_agent_x3E_'),
            ('_x3C_agent_x3E_', '__x3C__agent__x3E__'),
            ('has_xmarker', 'has__xmarker'),
            ('name/\u00e9', 'name_x2F__xE9_'),
            ('\U0001f600', '_x1F600_'),
            ('<AB', '_x3C_AB'),
            ('\u3cab', '_x3CAB_'),
        ]:
            agent = Agent(TestModel(), name=name, capabilities=[AzureDurableTaskDurability()])
            capability = AzureDurableTaskDurability.from_agent(agent)
            assert capability is not None
            names = {
                registration.__name__ for registration in capability.get_durable_operation_backend().registrations()
            }
            assert names
            assert all(re.fullmatch(r'[a-zA-Z0-9_.`+:-]{1,100}', value) for value in names)
            assert all(value.startswith(f'{escaped}____') for value in names)
            assert names.isdisjoint(all_names)
            all_names.update(names)

    def test_activity_name_escaping_is_injective(self) -> None:
        names: set[str] = set()
        for length in (1, 2):
            for characters in product('a_x0<\u00e9\U0001f600', repeat=length):
                agent = Agent(TestModel(), name=''.join(characters), capabilities=[AzureDurableTaskDurability()])
                capability = AzureDurableTaskDurability.from_agent(agent)
                assert capability is not None
                generated = {
                    activity.__name__ for activity in capability.get_durable_operation_backend().registrations()
                }
                assert generated.isdisjoint(names)
                names.update(generated)

    @pytest.mark.parametrize('name', ['', 'x' * 101, 'bad/name', 'bad name', 'bad<name>', 'bad@name'])
    def test_invalid_activity_registration_is_rejected(self, name: str) -> None:
        agent = Agent(TestModel(), name='names', capabilities=[AzureDurableTaskDurability()])
        capability = AzureDurableTaskDurability.from_agent(agent)
        assert capability is not None
        operation = DurableOperation(
            operation_id=CapabilityOperationId('names', operation='sleep'),
            handler=asyncio.sleep,
            parameter_transport=IdentityTransport[float](),
            cache_identity=EmptyCacheIdentity[float](),
            result_codec=JsonResultCodec[None](type(None)),
            config_role='capability',
        )
        with pytest.raises(UserError, match='1-100 ASCII characters'):
            capability.get_durable_operation_backend().register(operation, name=name, config=None)

    def test_generated_activity_name_collision_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        digest = MagicMock()
        digest.hexdigest.return_value = '0' * 64
        monkeypatch.setattr(hashlib, 'sha256', MagicMock(return_value=digest))
        with pytest.raises(UserError, match='not unique after escaping'):
            Agent(TestModel(), name='long-' + 'a' * 120, capabilities=[AzureDurableTaskDurability()])

    def test_previously_colliding_toolset_names_are_distinct(self) -> None:
        agent = Agent(
            TestModel(),
            name='collision',
            toolsets=[FunctionToolset(id='<custom>'), FunctionToolset(id='_custom_')],
            capabilities=[AzureDurableTaskDurability()],
        )
        capability = AzureDurableTaskDurability.from_agent(agent)
        assert capability is not None
        names = {activity.__name__ for activity in capability.get_durable_operation_backend().registrations()}
        assert 'collision____function__toolset_____x3C_custom_x3E_.call__tool' in names
        assert 'collision____function__toolset______custom__.call__tool' in names

    def test_long_activity_names_are_a_compatibility_contract(self) -> None:
        agent = Agent(TestModel(), name='long-' + 'a' * 120, capabilities=[AzureDurableTaskDurability()])
        capability = AzureDurableTaskDurability.from_agent(agent)
        assert capability is not None
        names = {activity.__name__ for activity in capability.get_durable_operation_backend().registrations()}
        assert names == {
            f'long-{"a" * 78}-{suffix}'
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
        assert state is not None
        state.raise_if_failed()
        assert calls == ['written']
        assert replacement.activity_calls == ['replay____model.request']

    def test_json_dependencies(self, durable_worker: InMemoryWorker) -> None:
        agent = Agent(
            TestModel(custom_output_text='done'),
            deps_type=int,
            name='json-deps',
            capabilities=[AzureDurableTaskDurability[int]()],
        )
        observed: list[int] = []

        @agent.tool
        def read(ctx: RunContext[int]) -> str:
            observed.append(ctx.deps)
            return 'ok'

        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=42).output == 'done'
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
            capabilities=[AzureDurableTaskDurability(operation_config=config)],
        )

        @agent.tool_plain
        def deterministic() -> int:
            return 42

        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
        assert durable_worker.activity_calls == ['inline____model.request', 'inline____model.request']

    @pytest.mark.parametrize('enabled', [False, True])
    def test_tool_retry_policy(self, durable_worker: InMemoryWorker, enabled: bool) -> None:
        calls: list[str] = []
        policy = task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
        base_policy = None if enabled else policy
        tool_policy = policy if enabled else None
        config = RoleBasedOperationConfig[task.RetryPolicy | None](
            model=None,
            event=None,
            capability=None,
            tool=base_policy,
            resolve_tool=lambda operation, tool, name: tool_policy if name == 'transient' else base_policy,
        )
        agent = Agent(
            TestModel(custom_output_text='done'),
            name='tool-retry',
            capabilities=[AzureDurableTaskDurability(operation_config=config)],
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
        assert 'dynamic-agent____dynamic__toolset____dynamic.get__tools' in durable_worker.activity_calls

    @pytest.mark.parametrize('role_config', [False, True])
    @pytest.mark.parametrize('kind', ['dynamic', 'mcp-tools', 'mcp-instructions'])
    def test_discovery_inherits_its_bound_retry_policy(
        self, durable_worker: InMemoryWorker, kind: str, role_config: bool
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
        durability = (
            AzureDurableTaskDurability(
                operation_config=RoleBasedOperationConfig(model=None, event=None, capability=None, tool=policy)
            )
            if role_config
            else AzureDurableTaskDurability(retry_policy=policy)
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

        with registered_agent(durable_worker, agent):
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

    def test_conflicting_retry_configuration(self) -> None:
        with pytest.raises(UserError, match='not both'):
            AzureDurableTaskDurability(
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
            TestModel(custom_output_text='done'),
            name='capability',
            capabilities=[Recorder(), AzureDurableTaskDurability()],
        )
        with registered_agent(durable_worker, agent):
            assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
        assert observed == ['started']
        assert durable_worker.activity_calls == [
            'capability____capability____recorder.record',
            'capability____model.request',
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
        assert 'stream____model.request__stream' in durable_worker.activity_calls
        assert 'stream____event__stream__handler' in durable_worker.activity_calls

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
        assert 'cancel____model.cancel__suspended__response' in durable_worker.activity_calls

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

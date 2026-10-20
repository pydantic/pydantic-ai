from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any, Self

import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec import CapabilityOperationId
from pydantic_ai.durable_exec._operation import (
    DurableOperation,
    IdentityParameterTransport,
    NoCacheIdentity,
    TypedResultCodec,
)
from pydantic_ai.durable_exec._toolset import DispatchSnapshots
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import Workspace, WorkspaceBackend, WorkspaceRef, WorkspaceUnavailableError

from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.durabletask import (
        DurableTaskAgent,
        DurableTaskDurability,
        DurableTaskOperationBackend,
        run_durable,
    )
    from pydantic_ai_harness.durabletask._run_context import (
        _ContextSnapshot,  # pyright: ignore[reportPrivateUsage]
        restore_context,
    )

    from .worker import InMemoryWorker

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


def test_exhaustive_run_context_transport_policy() -> None:
    transported = {
        'run_id',
        'conversation_id',
        'metadata',
        'retries',
        'tool_call_id',
        'tool_name',
        'tool_call_approved',
        'tool_call_metadata',
        'retry',
        'max_retries',
        'run_step',
        'partial_output',
        'trace_include_content',
        'instrumentation_version',
        'usage',
        'usage_limits',
        'deadline',
        'loaded_capability_ids',
        'discovered_tool_names',
        '_anchored_evidence',
        'available_tool_names',
        'active_capability_ids',
        '_deferred_capability_ids',
        'capability_active',
        '_tool_name_path',
    }
    rebuilt = {'deps', 'agent', 'root_capability', 'workspace'}
    unavailable = {
        'model',
        'prompt',
        'messages',
        'validation_context',
        'tracer',
        'model_settings',
        'capabilities',
        '_capability',
        '_cancellation',
        '_event_stream_buffer',
        '_pending_immediate_dispatches',
        '_event_stream_replacements',
        '_mcp_tool_defs_cache',
        '_model_id',
    }
    reset = {'tool_manager', 'realtime_session', '_durable_operations', '_run_capabilities_by_id', '_run_held_toolsets'}
    assert set(RunContext.__dataclass_fields__ | DispatchSnapshots.__dataclass_fields__) == (
        transported | rebuilt | unavailable | reset | {'pending_messages'}
    )
    agent = Agent(TestModel(), name='context-policy')
    original = RunContext(
        deps={'tenant': 'a'},
        model=TestModel(),
        usage=RunUsage(requests=2, input_tokens=5),
        agent=agent,
        run_id='run',
        conversation_id='conversation',
        metadata={'tag': 'value'},
        retries={'lookup': 1},
        tool_call_id='call',
        tool_name='lookup',
        tool_call_approved=True,
        tool_call_metadata={'reviewer': 'a'},
        retry=1,
        max_retries=3,
        run_step=2,
        partial_output=True,
        trace_include_content=True,
        usage_limits=UsageLimits(request_limit=4),
        deadline=datetime(2026, 1, 1, tzinfo=UTC),
        loaded_capability_ids={'loaded'},
        discovered_tool_names={'lookup'},
        capability_active=True,
    )
    adapter = TypeAdapter(_ContextSnapshot)
    recorded = adapter.validate_json(adapter.dump_json(_ContextSnapshot.capture(original)))
    restored = restore_context(recorded, deps={'worker': 'built'}, agent=agent)
    for field in transported:
        assert getattr(restored, field) == getattr(original, field), field
    for field in unavailable:
        with pytest.raises(UserError, match=repr(field)):
            getattr(restored, field)
    for field in reset:
        assert getattr(restored, field) is None, field
    assert restored.deps == {'worker': 'built'}
    assert restored.agent is agent
    assert restored.root_capability is agent.root_capability
    with pytest.raises(UserError, match=r'[Ee]nqueu'):
        restored.enqueue('not transported')
    with pytest.raises(WorkspaceUnavailableError, match='No workspace is attached'):
        asyncio.run(restored.workspace.working_dir())
    copied = replace(restored, tool_name='other')
    assert copied.tool_name == 'other'
    assert copied.available_tool_names == restored.available_tool_names


@pytest.mark.parametrize('restore', [False, True])
def test_missing_workspace_errors_keep_their_types(restore: bool) -> None:
    provider = InMemoryProvider()

    class Workspaces(AbstractCapability[object]):
        def get_workspace(self, ctx: RunContext[object], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            return provider.backend(ref) if restore else None

    agent = Agent(TestModel(), capabilities=[Workspaces()])
    original = RunContext(deps=None, model=TestModel(), usage=RunUsage(), workspace=Workspace(provider.backend(None)))
    asyncio.run(original.workspace.working_dir())
    recorded = _ContextSnapshot.capture(original)
    provider.environments.clear()
    if not restore:
        with pytest.raises(UserError, match='No capability can restore workspace'):
            restore_context(recorded, deps=None, agent=agent)
    else:
        restored = restore_context(recorded, deps=None, agent=agent)
        with pytest.raises(WorkspaceUnavailableError, match='does not exist'):
            asyncio.run(restored.workspace.working_dir())


@pytest.mark.parametrize('stage', ['envelope', 'parameters', 'dependencies'])
def test_invalid_activity_input_is_terminal(durable_worker: InMemoryWorker, stage: str) -> None:
    agent = Agent(
        TestModel(),
        deps_type=str,
        name='invalid-wire',
        capabilities=[
            DurableTaskDurability(
                retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
            )
        ],
    )
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        run = durable.start(durable_worker.client, 'Hi', deps_input='valid')
        assert durable_worker.step()
        instance_id, name, task_id, encoded = durable_worker.activity_queue.popleft()
        assert encoded is not None
        payload = json.loads(encoded)
        if stage == 'envelope':
            payload = {}
        elif stage == 'parameters':
            payload['parameters'] = None
        else:
            payload['deps_input'] = {'not': 'a string'}
        durable_worker.activity_queue.append((instance_id, name, task_id, json.dumps(payload)))
        with pytest.raises(OrchestrationFailedError, match='Durable Task serialization failed'):
            run.result()
    assert durable_worker.timers == []
    assert durable_worker.activity_calls == ['invalid-wire__model.request']


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

    def make_agent() -> DurableTaskAgent[object, str]:
        agent = Agent(FunctionModel(model), name='same-tool', capabilities=[DurableTaskDurability()])

        @agent.tool_plain
        def lookup(key: str) -> str:
            calls.append(key)
            return f'{key}-result'

        return DurableTaskAgent(agent)

    durable = make_agent()
    with durable.register(durable_worker):
        run = durable.start(durable_worker.client, 'Hi')
        for _ in range(20):
            if len(durable_worker.activity_queue) == 2 and all(
                name.endswith('.call_tool') for _, name, _, _ in durable_worker.activity_queue
            ):
                break
            assert durable_worker.step()
        else:
            pytest.fail('The same-tool batch was not scheduled')
        durable_worker.activity_queue.reverse()
        assert durable_worker.step()
        assert calls == ['right']
        assert durable_worker.step()
    replacement = InMemoryWorker()
    replacement.instances = durable_worker.instances
    replacement.activity_queue = durable_worker.activity_queue
    replacement_agent = make_agent()
    with replacement_agent.register(replacement):
        assert replacement_agent.get_run(replacement.client, run.instance_id).result().output == 'done'
    assert calls == ['right', 'left']
    assert replacement.activity_calls == ['same-tool__function_toolset__<agent>.call_tool', 'same-tool__model.request']


@pytest.mark.parametrize('kind', ['function', 'mcp'])
@pytest.mark.parametrize('cancel', [False, True])
def test_activity_toolset_scope_closes_after_failure_or_cancellation(
    durable_worker: InMemoryWorker, kind: str, cancel: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    lifetimes: list[str] = []

    def fail() -> None:
        assert lifetimes[-1] == 'enter'
        if cancel:
            raise asyncio.CancelledError('cancel tool')
        raise RuntimeError('failed tool')

    toolset: AbstractToolset[object]
    if kind == 'function':

        class Functions(FunctionToolset[object]):
            async def __aenter__(self) -> Self:
                lifetimes.append('enter')
                return await super().__aenter__()

            async def __aexit__(self, *args: Any) -> None:
                try:
                    await super().__aexit__(*args)
                finally:
                    lifetimes.append('exit')

        toolset = Functions([fail], id='tools')
    else:
        with try_import() as mcp_imports:
            from fastmcp import FastMCP

            from pydantic_ai.mcp import MCPToolset
        if not mcp_imports():
            pytest.skip('mcp not installed')

        server: FastMCP[None] = FastMCP()
        server.tool(fail)

        class MCP(MCPToolset[object]):
            async def __aenter__(self) -> Self:
                entered = await super().__aenter__()
                lifetimes.append('enter')
                return entered

            async def __aexit__(self, *args: Any) -> bool | None:
                try:
                    return await super().__aexit__(*args)
                finally:
                    lifetimes.append('exit')

            async def call_tool(
                self, name: str, tool_args: dict[str, Any], ctx: RunContext[object], tool: ToolsetTool[object]
            ) -> Any:
                fail()

        toolset = MCP(server, id='tools')

    agent = Agent(TestModel(), name='lifetimes', toolsets=[toolset], capabilities=[DurableTaskDurability()])
    activity_turn = durable_worker.activity_turn

    def checked_activity_turn(instance_id: str, name: str, task_id: int, encoded: str | None) -> None:
        activity_turn(instance_id, name, task_id, encoded)
        assert lifetimes.count('enter') == lifetimes.count('exit')

    monkeypatch.setattr(durable_worker, 'activity_turn', checked_activity_turn)
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        with pytest.raises(OrchestrationFailedError) as error:
            durable.start(durable_worker.client, 'Hi').result()
    if not cancel:
        assert 'failed tool' in str(error.value)
    attempts = sum('__tools.' in name for name in durable_worker.activity_calls)
    assert attempts >= 1
    assert any(name.endswith('.call_tool') for name in durable_worker.activity_calls)
    depth = 0
    for event in lifetimes:
        depth += 1 if event == 'enter' else -1
        assert depth >= 0
    assert depth == 0
    assert lifetimes.count('enter') >= attempts


@pytest.mark.parametrize('kind', ['collision', 'legacy', 'codec-failure', 'json-failure'])
def test_activity_result_wire_contract(durable_worker: InMemoryWorker, kind: str) -> None:
    @asynccontextmanager
    async def scope(deps_input: JsonValue) -> AsyncGenerator[None]:
        yield

    value: JsonValue = (
        {'format': 'pydantic-ai-durabletask-result-v1', 'value': 'user data', 'serialization_error': 'not a failure'}
        if kind == 'collision'
        else {'answer': 42}
    )

    async def handler(params: dict[str, JsonValue]) -> object:
        return object() if kind.endswith('failure') else value

    backend = DurableTaskOperationBackend(agent_name='wire', activity_scope=scope)
    operation = backend.bind(
        DurableOperation(
            operation_id=CapabilityOperationId('wire', operation='echo'),
            handler=handler,
            parameter_transport=IdentityParameterTransport[dict[str, JsonValue]](),
            cache_identity=NoCacheIdentity[dict[str, JsonValue]](),
            result_codec=TypedResultCodec[object](object, mode='identity' if kind == 'json-failure' else 'json'),
            config_role='capability',
        )
    )

    def orchestrator(
        context: task.OrchestrationContext, payload: object
    ) -> Generator[task.Task[object], object, object]:
        async def body() -> object:
            return await operation({'parameters': None, 'deps_input': None})

        return (yield from run_durable(context, body))

    with backend.worker_scope(durable_worker):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
        assert durable_worker.step()
        assert durable_worker.step()
        event = durable_worker.instances[instance_id].new_events[0]
        assert event.HasField('taskCompleted')
        recorded = json.loads(event.taskCompleted.result.value)
        if kind == 'collision':
            assert recorded == snapshot(
                {
                    'format': 'pydantic-ai-durabletask-result-v1',
                    'value': {
                        'format': 'pydantic-ai-durabletask-result-v1',
                        'value': 'user data',
                        'serialization_error': 'not a failure',
                    },
                    'serialization_error': None,
                }
            )
        elif kind == 'legacy':
            event.taskCompleted.result.value = json.dumps(recorded['value'])
        else:
            assert recorded['serialization_error'].startswith('Durable Task serialization failed:')
        state = durable_worker.wait(instance_id)
    assert state is not None
    if kind.endswith('failure'):
        with pytest.raises(OrchestrationFailedError, match='Durable Task serialization failed'):
            state.raise_if_failed()
    else:
        state.raise_if_failed()
        assert state.get_output() == value

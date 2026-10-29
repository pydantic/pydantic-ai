import asyncio
import json
import sys
from collections.abc import Generator
from dataclasses import replace
from datetime import datetime, timedelta
from functools import partial
from importlib import import_module
from typing import Any, Self
from uuid import UUID

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, ConfigDict, Json, JsonValue

from pydantic_ai import (
    Agent,
    Choice,
    Choices,
    RunContext,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.durable_exec import (
    CapabilityOperationId,
    DurableOperation,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolKind,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    import pydantic_ai_harness
    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

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


def test_import_without_mcp(monkeypatch: pytest.MonkeyPatch) -> None:
    package = 'pydantic_ai_harness.azure_durable_task'
    monkeypatch.delattr(pydantic_ai_harness, 'azure_durable_task', raising=False)
    for name in [name for name in sys.modules if name == package or name.startswith(f'{package}.')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, 'pydantic_ai.mcp', None)

    durability_type = import_module(package).AzureDurableTaskDurability
    durability = durability_type.from_agent(Agent(TestModel(), name='no-mcp', capabilities=[durability_type()]))
    assert durability is not None
    assert durability.get_durable_operation_backend().registrations()


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
                tool_options=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
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

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=7).output == 'done'
    assert observed == [42]
    assert validated == [42]


def test_choice_action_output_and_partial_output(durable_worker: 'InMemoryWorker') -> None:
    choices = Choices({'answer': Choice(value=lambda: 42)})
    choice_agent = Agent(TestModel(), output_type=choices, name='choice', capabilities=[AzureDurableTaskDurability()])
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


def test_incompatible_persisted_tool_definition_is_rejected(durable_worker: 'InMemoryWorker') -> None:
    agent = Agent(TestModel(), name='removed', capabilities=[AzureDurableTaskDurability()])

    @agent.tool_plain
    def original() -> str:
        # The invalid persisted name must be rejected before this tool runs.
        return 'original'  # pragma: no cover

    with registered_agent(durable_worker, agent):
        run = schedule_agent(durable_worker, agent, 'Hi', deps=None)
        for _ in range(20):
            if not (durable_worker.activity_queue and durable_worker.activity_queue[0][1].endswith('.call_tool')):
                assert durable_worker.step()
        assert durable_worker.activity_queue and durable_worker.activity_queue[0][1].endswith('.call_tool')
        instance_id, name, task_id, encoded = durable_worker.activity_queue.popleft()
        assert encoded is not None
        payload = json.loads(encoded)
        payload['parameters']['name'] = 'removed'
        payload['parameters']['original_name'] = 'removed'
        payload['parameters']['tool_def']['name'] = 'removed'
        durable_worker.activity_queue.appendleft((instance_id, name, task_id, json.dumps(payload)))
        with pytest.raises(OrchestrationFailedError, match='is no longer available'):
            completed(durable_worker, run)


def test_replaced_activity_context_preserves_tool_kind_snapshot(durable_worker: 'InMemoryWorker') -> None:
    lookup_kind = ToolKind('azure_activity_context', 'lookup')
    names: list[str | None] = []
    agent = Agent(
        TestModel(custom_output_text='done'),
        name='context-copy',
        capabilities=[AzureDurableTaskDurability()],
    )

    @agent.tool(tool_kind=lookup_kind)
    async def lookup(ctx: RunContext[object]) -> str:
        copied = replace(ctx, tool_call_id='copied-call')
        assert copied is not ctx
        assert copied.tool_call_id == 'copied-call'
        assert ctx.tool_name_for_kind(lookup_kind) == 'lookup'
        names.append(copied.tool_name_for_kind(lookup_kind))
        assert copied.tool_name_for_kind('azure_activity_context.missing') is None
        return 'found'

    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert names == ['lookup']
    assert 'context--copy__function_toolset__-x3C-agent-x3E-.call_tool' in durable_worker.activity_calls


@pytest.mark.parametrize('stage', ['envelope', 'parameters', 'dependencies'])
def test_invalid_activity_input_is_terminal(durable_worker: 'InMemoryWorker', stage: str) -> None:
    agent = Agent(
        TestModel(),
        deps_type=str,
        name='invalid-wire',
        capabilities=[
            AzureDurableTaskDurability(
                tool_options=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
            )
        ],
    )
    with registered_agent(durable_worker, agent):
        run = schedule_agent(durable_worker, agent, 'Hi', deps='valid')
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
            completed(durable_worker, run)
    assert durable_worker.timers == []
    assert durable_worker.activity_calls == ['invalid--wire__model.request']


@pytest.mark.parametrize('kind', ['function', 'mcp'])
@pytest.mark.parametrize('cancel', [False, True])
def test_activity_toolset_scope_closes_after_failure_or_cancellation(
    durable_worker: 'InMemoryWorker', kind: str, cancel: bool, monkeypatch: pytest.MonkeyPatch
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
            # Optional-extra probes can skip here; all-extras coverage does not.
            pytest.skip('mcp not installed')  # pragma: lax no cover

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

    agent = Agent(TestModel(), name='lifetimes', toolsets=[toolset], capabilities=[AzureDurableTaskDurability()])
    activity_turn = durable_worker.activity_turn

    def checked_activity_turn(instance_id: str, name: str, task_id: int, encoded: str | None) -> None:
        activity_turn(instance_id, name, task_id, encoded)
        assert lifetimes.count('enter') == lifetimes.count('exit')

    monkeypatch.setattr(durable_worker, 'activity_turn', checked_activity_turn)
    with registered_agent(durable_worker, agent):
        with pytest.raises(OrchestrationFailedError) as error:
            run_agent(durable_worker, agent, 'Hi', deps=None)
    if not cancel:
        assert 'failed tool' in str(error.value)
    assert 'enter' in lifetimes


@pytest.mark.parametrize('kind', ['collision', 'value', 'missing-format', 'codec-failure', 'json-failure'])
def test_activity_result_wire_contract(durable_worker: 'InMemoryWorker', kind: str) -> None:
    value: JsonValue = (
        {'format': 'pydantic-ai-durabletask-result-v1', 'value': 'user data', 'error': 'not a failure'}
        if kind == 'collision'
        else {'answer': 42}
    )

    async def handler(params: dict[str, JsonValue]) -> object:
        return object() if kind.endswith('failure') else value

    capability = AzureDurableTaskDurability.from_agent(
        Agent(TestModel(), name='wire', capabilities=[AzureDurableTaskDurability()])
    )
    assert capability is not None
    backend = capability.get_durable_operation_backend()
    operation = backend.bind(
        DurableOperation(
            operation_id=CapabilityOperationId('wire', operation='echo'),
            handler=handler,
            parameter_transport=IdentityTransport[dict[str, JsonValue]](),
            cache_identity=EmptyCacheIdentity[dict[str, JsonValue]](),
            result_codec=JsonResultCodec[object](object, mode='identity' if kind == 'json-failure' else 'json'),
            config_role='capability',
        )
    )

    def orchestrator(
        context: task.OrchestrationContext, payload: object
    ) -> Generator[task.Task[object], object, object]:
        async def body() -> object:
            return await operation({'parameters': None, 'deps_input': None})

        return (yield from run_durable(context, body))

    with capability.register(durable_worker):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.schedule(orchestrator)
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
                        'error': 'not a failure',
                    },
                    'error': None,
                }
            )
        elif kind.endswith('failure'):
            assert recorded['error'].startswith('Durable Task serialization failed:')
        else:
            assert recorded['value'] == value
            if kind == 'missing-format':
                del recorded['format']
                event.taskCompleted.result.value = json.dumps(recorded)
        state = durable_worker.wait(instance_id)
    if kind == 'missing-format':
        with pytest.raises(OrchestrationFailedError, match='Field required'):
            state.raise_if_failed()
    elif kind.endswith('failure'):
        with pytest.raises(OrchestrationFailedError, match='Durable Task serialization failed'):
            state.raise_if_failed()
    else:
        state.raise_if_failed()
        assert state.get_output() == value


def test_model_compaction_restores_the_model(durable_worker: 'InMemoryWorker') -> None:
    class CompactModel(TestModel):
        async def compact_messages(
            self, request_context: ModelRequestContext, *, instructions: str | None = None
        ) -> ModelResponse:
            assert request_context.model is self
            assert request_context.model_settings == {'temperature': 0.2}
            assert instructions == 'Keep decisions'
            return ModelResponse(parts=[TextPart('compacted')])

    model = CompactModel()
    agent = Agent(model, name='compact', capabilities=[AzureDurableTaskDurability()])
    capability = AzureDurableTaskDurability.from_agent(agent)
    assert capability is not None

    def orchestrator(context: task.OrchestrationContext, payload: object) -> Generator[task.Task[object], object, bool]:
        async def body() -> bool:
            request = ModelRequestContext(
                model=model,
                messages=[],
                model_settings={'temperature': 0.2},
                model_request_parameters=ModelRequestParameters(),
            )

            async def handler(context: ModelRequestContext) -> ModelResponse:
                return await context.model.compact_messages(context, instructions='Keep decisions')

            result = await capability.wrap_model_request(
                RunContext(deps=None, model=model, usage=RunUsage()), request_context=request, handler=handler
            )
            return result.parts == [TextPart('compacted')]

        return (yield from run_durable(context, body))

    with registered_agent(durable_worker, agent):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.schedule(orchestrator)
        state = durable_worker.wait(instance_id)
    state.raise_if_failed()
    assert state.get_output() is True
    assert durable_worker.activity_calls == ['compact__model.compact_messages']


@pytest.mark.parametrize('restore', [True, False])
def test_workspace_is_reconstructed_in_activities(durable_worker: 'InMemoryWorker', restore: bool) -> None:
    provider = InMemoryProvider()
    durability = AzureDurableTaskDurability[object]()

    class Workspaces(AbstractCapability[object]):
        def get_workspace(self, ctx: RunContext[object], *, ref: WorkspaceRef | None) -> WorkspaceBackend | None:
            return provider.backend(ref) if ref is None or restore or durability.in_durable_context else None

        async def before_run(self, ctx: RunContext[object]) -> None:
            await ctx.workspace.write_text('hook.txt', 'written in hook')

    agent = Agent(
        TestModel(custom_output_text='done'),
        name='workspace',
        capabilities=[Workspaces(), durability],
    )

    @agent.tool
    async def read(ctx: RunContext[object]) -> str:
        assert await ctx.workspace.read_text('hook.txt') == 'written in hook'
        return 'read in activity'

    with registered_agent(durable_worker, agent):
        run = schedule_agent(durable_worker, agent, 'Hi', deps=None)
        if not restore:
            with pytest.raises(OrchestrationFailedError, match='No capability can supply the workspace'):
                completed(durable_worker, run)
            return
        assert completed(durable_worker, run) == 'done'
    assert 'workspace__capability__workspace.call' in durable_worker.activity_calls
    assert len(provider.environments) == 1


def test_mcp_discovery_instructions_and_calls(durable_worker: 'InMemoryWorker') -> None:
    with try_import() as mcp_imports:
        from fastmcp import FastMCP

        from pydantic_ai.mcp import MCPToolset
    if not mcp_imports():
        # Optional-extra probes can skip here; all-extras coverage does not.
        pytest.skip('mcp not installed')  # pragma: lax no cover

    server: FastMCP[None] = FastMCP(name='calc', instructions='Add numbers.')
    calls: list[int] = []

    @server.tool
    def add(a: int, b: int) -> int:
        calls.append(a + b)
        return a + b

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('add', {'a': 2, 'b': 3})])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model),
        name='mcp',
        toolsets=[MCPToolset[object](server, id='calc', include_instructions=True)],
        capabilities=[AzureDurableTaskDurability()],
    )
    with registered_agent(durable_worker, agent):
        assert run_agent(durable_worker, agent, 'Hi', deps=None).output == 'done'
    assert calls == [5]
    assert durable_worker.activity_calls == [
        'mcp__mcp_server__calc.get_tools',
        'mcp__mcp_server__calc.get_instructions',
        'mcp__model.request',
        'mcp__mcp_server__calc.call_tool',
        'mcp__mcp_server__calc.get_instructions',
        'mcp__model.request',
    ]

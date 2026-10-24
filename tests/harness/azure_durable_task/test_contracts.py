from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta
from typing import Any, Self

import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec import CapabilityOperationId, DurableOperation
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolKind,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool

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
        agent_orchestrator,
        completed,
        registered_agent,
        run_agent,
        schedule_agent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


def test_replaced_activity_context_preserves_tool_kind_snapshot(durable_worker: InMemoryWorker) -> None:
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
    assert 'context-copy____function__toolset_____x3C_agent_x3E_.call__tool' in durable_worker.activity_calls


@pytest.mark.parametrize('stage', ['envelope', 'parameters', 'dependencies'])
def test_invalid_activity_input_is_terminal(durable_worker: InMemoryWorker, stage: str) -> None:
    agent = Agent(
        TestModel(),
        deps_type=str,
        name='invalid-wire',
        capabilities=[
            AzureDurableTaskDurability(
                retry_policy=task.RetryPolicy(first_retry_interval=timedelta(seconds=1), max_number_of_attempts=2)
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
    assert durable_worker.activity_calls == ['invalid-wire____model.request']


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
                and all(name.endswith('.call__tool') for _, name, _, _ in durable_worker.activity_queue)
            ):
                assert durable_worker.step()
        assert len(durable_worker.activity_queue) == 2
        assert all(name.endswith('.call__tool') for _, name, _, _ in durable_worker.activity_queue)
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
        'same-tool____function__toolset_____x3C_agent_x3E_.call__tool',
        'same-tool____model.request',
    ]


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
    attempts = sum('____tools.' in name for name in durable_worker.activity_calls)
    assert attempts >= 1
    assert any(name.endswith('.call__tool') for name in durable_worker.activity_calls)
    depth = 0
    for event in lifetimes:
        depth += 1 if event == 'enter' else -1
        assert depth >= 0
    assert depth == 0
    assert lifetimes.count('enter') >= attempts


@pytest.mark.parametrize('kind', ['collision', 'value', 'missing-format', 'codec-failure', 'json-failure'])
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

    backend = AzureDurableTaskOperationBackend(agent_name='wire', activity_scope=scope)
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
        elif kind.endswith('failure'):
            assert recorded['serialization_error'].startswith('Durable Task serialization failed:')
        else:
            assert recorded['value'] == value
            if kind == 'missing-format':
                del recorded['format']
                event.taskCompleted.result.value = json.dumps(recorded)
        state = durable_worker.wait(instance_id)
    assert state is not None
    if kind == 'missing-format':
        with pytest.raises(OrchestrationFailedError, match='Field required'):
            state.raise_if_failed()
    elif kind.endswith('failure'):
        with pytest.raises(OrchestrationFailedError, match='Durable Task serialization failed'):
            state.raise_if_failed()
    else:
        state.raise_if_failed()
        assert state.get_output() == value

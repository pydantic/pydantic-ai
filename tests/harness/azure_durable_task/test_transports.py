from __future__ import annotations

from collections.abc import Generator

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from ...conftest import try_import
from ...workspace_fakes import InMemoryProvider

with try_import() as imports_successful:
    from durabletask import task
    from durabletask.client import OrchestrationFailedError

    from pydantic_ai_harness.azure_durable_task import AzureDurableTaskDurability, run_durable

    from .worker import (
        InMemoryWorker,
        completed,
        registered_agent,
        run_agent,
        schedule_agent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='durabletask not installed')


def test_model_compaction_restores_the_model(durable_worker: InMemoryWorker) -> None:
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
        instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
        state = durable_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() is True
    assert durable_worker.activity_calls == ['compact____model.compact__messages']


@pytest.mark.parametrize('restore', [True, False])
def test_workspace_is_reconstructed_in_activities(durable_worker: InMemoryWorker, restore: bool) -> None:
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
    assert 'workspace____capability____workspace.call' in durable_worker.activity_calls
    assert len(provider.environments) == 1


def test_mcp_discovery_instructions_and_calls(durable_worker: InMemoryWorker) -> None:
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
        'mcp____mcp__server____calc.get__tools',
        'mcp____mcp__server____calc.get__instructions',
        'mcp____model.request',
        'mcp____mcp__server____calc.call__tool',
        'mcp____mcp__server____calc.get__instructions',
        'mcp____model.request',
    ]

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

    from pydantic_ai_harness.durabletask import DurableTaskAgent, DurableTaskDurability, run_durable

    from .worker import InMemoryWorker

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
    agent = Agent(model, name='compact', capabilities=[DurableTaskDurability()])
    durable = DurableTaskAgent(agent)
    capability = DurableTaskDurability.from_agent(agent)
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

    with durable.register(durable_worker):
        durable_worker.add_orchestrator(orchestrator)
        instance_id = durable_worker.client.schedule_new_orchestration(orchestrator)
        state = durable_worker.wait(instance_id)
    assert state is not None
    state.raise_if_failed()
    assert state.get_output() is True
    assert durable_worker.activity_calls == ['compact__model.compact_messages']


@pytest.mark.parametrize('restore', [True, False])
def test_workspace_is_reconstructed_in_activities(durable_worker: InMemoryWorker, restore: bool) -> None:
    provider = InMemoryProvider()
    durability = DurableTaskDurability[object]()

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

    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        run = durable.start(durable_worker.client, 'Hi')
        if not restore:
            with pytest.raises(OrchestrationFailedError, match='No capability can restore workspace'):
                run.result()
            return
        assert run.result().output == 'done'
    assert 'workspace__capability__workspace.call' in durable_worker.activity_calls
    assert len(provider.environments) == 1


def test_mcp_discovery_instructions_and_calls(durable_worker: InMemoryWorker) -> None:
    with try_import() as mcp_imports:
        from fastmcp import FastMCP

        from pydantic_ai.mcp import MCPToolset
    if not mcp_imports():
        pytest.skip('mcp not installed')

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
        capabilities=[DurableTaskDurability()],
    )
    durable = DurableTaskAgent(agent)
    with durable.register(durable_worker):
        assert durable.start(durable_worker.client, 'Hi').result().output == 'done'
    assert calls == [5]
    assert durable_worker.activity_calls == [
        'mcp__mcp_server__calc.get_tools',
        'mcp__mcp_server__calc.get_instructions',
        'mcp__model.request',
        'mcp__mcp_server__calc.call_tool',
        'mcp__mcp_server__calc.get_instructions',
        'mcp__model.request',
    ]

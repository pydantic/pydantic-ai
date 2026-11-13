"""Spills under `TemporalDurability`: the spill and its read-back run in activities, with and without a storage."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AgentCapability, LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness import CapabilityStorage, FileSystem, ToolOutputLimits
from pydantic_ai_harness.tool_output_limits import READ_TOOL_NAME, Band, Spill
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]

PAYLOAD = '\n'.join(f'line {i}' for i in range(500))


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if not returns:
        return ModelResponse(parts=[ToolCallPart('_big_tool', {})])
    if len(returns) == 1:
        assert returns[0].metadata is not None
        handle = returns[0].metadata['overflow_handle']
        return ModelResponse(parts=[ToolCallPart(READ_TOOL_NAME, {'handle': handle, 'limit': 2})])
    return ModelResponse(parts=[TextPart('done')])


def _big_tool() -> str:
    return PAYLOAD


@workflow.defn
class SpillWorkflow(PydanticAIWorkflow):
    agent: Agent[None, str]

    @workflow.run
    async def run(self) -> list[str]:
        result = await self.agent.run('go')
        return [str(p.content) for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]


@pytest.fixture(scope='module')
async def temporal_env() -> AsyncIterator[WorkflowEnvironment]:
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        yield env


async def _run(env: WorkflowEnvironment, *capabilities: AgentCapability[None]) -> list[str]:
    all_capabilities: list[AgentCapability[None]] = [
        *capabilities,
        ToolOutputLimits(bands=[Band(over=100, action=Spill())]),
        TemporalDurability(),
    ]
    agent = Agent(
        FunctionModel(_model),
        name='capability_storage',
        deps_type=type(None),
        tools=[_big_tool],
        capabilities=all_capabilities,
    )
    SpillWorkflow.agent = agent
    SpillWorkflow.__pydantic_ai_agents__ = [agent]
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='capability-storage', workflows=[SpillWorkflow], workflow_runner=runner):
        return await client.execute_workflow(
            SpillWorkflow.run,
            id=uuid4().hex,
            task_queue='capability-storage',
            execution_timeout=timedelta(seconds=60),
        )


def _spill(root: Path) -> str:
    [spill] = (root / '.pydantic-ai-harness' / 'tool-output').glob('*/*')
    return spill.read_text(encoding='utf-8')


class TestCapabilityStorageUnderTemporal:
    async def test_spills_go_to_storage_inside_a_workflow(
        self, temporal_env: WorkflowEnvironment, tmp_path: Path
    ) -> None:
        work = tmp_path / 'work'
        storage = tmp_path / 'storage'
        work.mkdir()
        storage.mkdir()

        spilled, read = await _run(
            temporal_env, LocalWorkspace(work), FileSystem(), CapabilityStorage(LocalWorkspaceBackend(storage))
        )

        assert _spill(storage) == PAYLOAD
        assert f'{READ_TOOL_NAME}(handle=' in spilled
        assert read.endswith('line 0\nline 1')
        assert list(work.iterdir()) == []

    async def test_spills_go_to_the_run_workspace_without_a_storage(
        self, temporal_env: WorkflowEnvironment, tmp_path: Path
    ) -> None:
        _, read = await _run(temporal_env, LocalWorkspace(tmp_path))

        assert _spill(tmp_path) == PAYLOAD
        assert read.endswith('line 0\nline 1')

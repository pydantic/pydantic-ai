"""Temporal integration of Shell's run-scoped cwd."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.shell import Shell
from tests.harness._temporal import skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), skip_temporal_sandbox_on_314]


async def test_completed_run_removes_cwd_state(tmp_path: Path) -> None:
    (tmp_path / 'a').mkdir()
    agent = _agent(tmp_path, 'a')
    result = await agent.run('start')
    assert result.output == 'done'
    state_dir = tmp_path / '.pydantic-ai-harness/shell/run-state'
    assert not state_dir.exists() or not list(state_dir.iterdir())


@workflow.defn
class ShellWorkflow(PydanticAIWorkflow):
    agents: dict[str, Agent[None, str]]

    @workflow.run
    async def run(self, key: str) -> list[str]:
        result = await self.agents[key].run('start')
        return [
            str(part.content)
            for message in result.all_messages()
            for part in message.parts
            if part.part_kind == 'tool-return'
        ]


def _agent(root: Path, key: str) -> Agent[None, str]:
    def model(messages: object, info: object) -> ModelResponse:
        count = sum(p.part_kind == 'tool-return' for m in messages for p in m.parts)  # type: ignore[attr-defined]
        if count < 2:
            return ModelResponse(
                parts=[ToolCallPart('run_command', {'command': f'cd {key} && pwd' if count == 0 else 'pwd'})]
            )
        return ModelResponse(parts=[TextPart('done')])

    return Agent(
        FunctionModel(model),
        name=f'shell_{key}',
        capabilities=[LocalWorkspace(root), Shell(persist_cwd=True), TemporalDurability()],
    )


async def test_concurrent_temporal_workflows_keep_separate_cwd(tmp_path: Path) -> None:
    for key in ('a', 'b'):
        (tmp_path / key).mkdir()
    agents = {key: _agent(tmp_path, key) for key in ('a', 'b')}
    ShellWorkflow.agents = agents
    ShellWorkflow.__pydantic_ai_agents__ = list(agents.values())
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        runner = SandboxedWorkflowRunner(
            restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types')
        )
        async with Worker(client, task_queue='shell-cwd', workflows=[ShellWorkflow], workflow_runner=runner):
            results = await asyncio.gather(
                *(
                    client.execute_workflow(
                        ShellWorkflow.run,
                        key,
                        id=uuid4().hex,
                        task_queue='shell-cwd',
                        execution_timeout=timedelta(seconds=120),
                    )
                    for key in ('a', 'b')
                )
            )
    for key, outputs in zip(('a', 'b'), results):
        assert len(outputs) == 2
        assert all(str(tmp_path / key) in output for output in outputs)

"""Temporal replay of a veto followed by a background launch."""

from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path
from uuid import uuid4

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileSystem
from pydantic_ai_harness.shell import Shell
from tests.harness.conftest import skip_temporal_sandbox_on_314

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), skip_temporal_sandbox_on_314]

_vetoes: list[str] = []


def _model(messages: list[ModelMessage], info: object) -> ModelResponse:
    root = str(messages[0].parts[0].content)  # type: ignore[union-attr]
    returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
    if len(returns) == 0:
        part = ToolCallPart('write_file', {'path': root + '/blocked.txt', 'content': 'denied'})
    elif len(returns) == 1:
        part = ToolCallPart('start_command', {'command': f'echo once >> {root}/launches.txt'})
    elif len(returns) == 2:
        ids = re.findall(r'ID: ([0-9a-f]{32})', str(returns[-1].content))
        part = ToolCallPart('check_command', {'command_id': ids[-1] if ids else 'missing'})
    else:
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[part])


async def _stream(messages: list[ModelMessage], info: object):
    for index, part in enumerate(_model(messages, info).parts):
        if isinstance(part, TextPart):
            yield part.content
        else:
            assert isinstance(part, ToolCallPart)
            yield {index: DeltaToolCall(name=part.tool_name, json_args=json.dumps(part.args))}


@workflow.defn
class ReplayWorkflow(PydanticAIWorkflow):
    agent: Agent[None, str]

    @workflow.run
    async def run(self, root: str) -> list[str]:
        result = await self.agent.run(root)
        return [str(p.content) for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]


async def test_temporal_history_replays_veto_and_background_job_once(tmp_path: Path) -> None:
    _vetoes.clear()
    workflow_id = uuid4().hex
    agent = Agent(
        FunctionModel(_model, stream_function=_stream),
        name='temporal_replay_harness',
        capabilities=[LocalWorkspace(tmp_path), FileSystem(), Shell(), TemporalDurability()],
    )

    @agent.on_event(FileChangeRequestEvent)
    async def veto(ctx: RunContext[None], event: FileChangeRequestEvent) -> None:
        _vetoes.append(event.path)
        event.cancel('denied')

    ReplayWorkflow.agent = agent
    ReplayWorkflow.__pydantic_ai_agents__ = [agent]
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types')
    )
    async with await WorkflowEnvironment.start_local() as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(client, task_queue='replay-harness', workflows=[ReplayWorkflow], workflow_runner=runner):
            result = await client.execute_workflow(
                ReplayWorkflow.run,
                str(tmp_path),
                id=workflow_id,
                task_queue='replay-harness',
                execution_timeout=timedelta(seconds=120),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()
    assert 'denied' in result[0]
    assert 'ID: ' in result[1]
    assert '[status: ' in result[2]
    assert _vetoes == ['blocked.txt']
    assert not (tmp_path / 'blocked.txt').exists()
    # The launch runs in the background and may still be starting when the workflow ends.
    launches = tmp_path / 'launches.txt'
    with anyio.fail_after(30):
        while not (launches.exists() and launches.read_text()):
            await anyio.sleep(0.01)  # pragma: lax no cover
    assert launches.read_text().splitlines() == ['once']
    await Replayer(workflows=[ReplayWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner).replay_workflow(
        history
    )
    # Replayer re-runs workflow-side listeners; an external listener must deduplicate its own effects.
    assert _vetoes == ['blocked.txt', 'blocked.txt']
    assert (tmp_path / 'launches.txt').read_text().splitlines() == ['once']

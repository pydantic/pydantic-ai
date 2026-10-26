"""Temporal replay of a `Memory` run whose snapshot compaction drops mid-run.

Recovering a Temporal workflow replays its history: activities, such as `Memory`'s snapshot load, are not run
again, and the workflow must issue the same commands. The run here records a snapshot, has compaction drop it
during the tool loop, and puts the same snapshot back. Replaying it after the store changed must neither read the
store nor diverge.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.memory import InMemoryStore, Memory, MemoryFile
from tests.harness._temporal import skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [pytest.mark.temporal, pytest.mark.xdist_group(name='harness-temporal'), skip_temporal_sandbox_on_314]


class _CountingStore(InMemoryStore):
    """Counts reads, so a replay that read the store again shows."""

    reads = 0

    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        _CountingStore.reads += 1
        return await super().read(path, max_chars=max_chars)


_store = _CountingStore({'main/MEMORY.md': '- version A'})
_snapshots_sent: list[list[str]] = []
"""The memory snapshots each model request carried, recorded where the model runs."""


def _snapshots(messages: list[ModelMessage]) -> list[str]:
    return [
        item.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
        if isinstance(item, TextContent) and item.content.startswith('<memory>\n')
    ]


def _model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    _snapshots_sent.append(_snapshots(messages))
    if len(_snapshots_sent) < 5:
        return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id=f'call-{len(_snapshots_sent)}')])
    return ModelResponse(parts=[TextPart('done')])


def lookup() -> str:
    """Look something up."""
    return 'found'


agent = Agent(
    FunctionModel(_model),
    name='memory_replay',
    tools=[lookup],
    capabilities=[
        SlidingWindowCompaction(max_messages=4, keep_messages=2, preserve_first_user_message=False),
        Memory(store=_store),
        TemporalDurability(),
    ],
)


@workflow.defn
class MemoryReplayWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def test_replay_restores_the_recorded_snapshot_without_reading_the_store() -> None:
    _snapshots_sent.clear()
    workflow_id = uuid4().hex
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types')
    )
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        async with Worker(client, task_queue='memory-replay', workflows=[MemoryReplayWorkflow], workflow_runner=runner):
            output = await client.execute_workflow(
                MemoryReplayWorkflow.run,
                'what do I like?',
                id=workflow_id,
                task_queue='memory-replay',
                execution_timeout=timedelta(seconds=120),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert output == 'done'
    # Compaction drops the snapshot during the tool loop; every request still sends exactly the one recorded.
    assert len(_snapshots_sent) == 5
    [snapshot] = _snapshots_sent[0]
    assert '- version A' in snapshot
    assert all(sent == [snapshot] for sent in _snapshots_sent)

    current = await _store.read('main/MEMORY.md', max_chars=100)
    assert current is not None
    await _store.write('main/MEMORY.md', '- version B', expected_version=current.version)
    reads = _CountingStore.reads
    await Replayer(
        workflows=[MemoryReplayWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner
    ).replay_workflow(history)
    # Recovery replays the recorded snapshot load instead of reading the changed store.
    assert _CountingStore.reads == reads

from __future__ import annotations

import sys
import uuid
from datetime import timedelta

import anyio
import pytest

from pydantic_ai import Agent, EndStrategy, FunctionToolResultEvent, Tool
from pydantic_ai.models.test import TestModel

try:
    from temporalio import activity, workflow
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip('Temporal sandbox is incompatible with Python 3.14', allow_module_level=True)

pytestmark = pytest.mark.xdist_group(name='temporal-tool-completion')

REPLAY_TIMEOUT = 30


async def first() -> str:
    return 'first'


async def second() -> str:
    return 'second'


_tools = [Tool(first, metadata={'temporal': False}), Tool(second, metadata={'temporal': False})]
_graceful_agent = Agent(
    TestModel(custom_output_text='done'),
    name='completion_graceful',
    tools=_tools,
    capabilities=[TemporalDurability()],
)
_exhaustive_agent = Agent(
    TestModel(custom_output_text='done'),
    name='completion_exhaustive',
    tools=_tools,
    end_strategy='exhaustive',
    capabilities=[TemporalDurability()],
)
_legacy_agent = Agent(
    TestModel(call_tools=[], custom_output_text='legacy'),
    name='completion_legacy',
    capabilities=[TemporalDurability()],
)


@activity.defn
async def record_first() -> None:
    pass


@activity.defn
async def record_second() -> None:
    pass


@workflow.defn(failure_exception_types=[Exception])
class ToolCompletionOrderWorkflow:
    @workflow.run
    async def run(self, end_strategy: EndStrategy) -> list[str]:
        agent = _graceful_agent if end_strategy == 'graceful' else _exhaustive_agent
        results: list[str] = []
        async with agent.run_stream_events('go') as events:
            async for event in events:
                if isinstance(event, FunctionToolResultEvent):
                    name = event.part.tool_name
                    results.append(name)
                    # Different activity types make a changed event order replay-visible.
                    await workflow.execute_activity('record_' + name, start_to_close_timeout=timedelta(seconds=10))
        return results


@workflow.defn
class LegacyCompletionWaitWorkflow:
    @workflow.run
    async def run(self) -> str:
        return (await _legacy_agent.run('go')).output


@pytest.mark.parametrize('end_strategy', ['graceful', 'exhaustive'])
async def test_simultaneous_tool_completion_commands_replay(client: Client, end_strategy: EndStrategy) -> None:
    """Result events drive workflow commands, so their simultaneous-completion order must replay."""
    task_queue = 'tool-completion-order'
    async with Worker(
        client,
        task_queue=task_queue,
        workflows=[ToolCompletionOrderWorkflow],
        activities=[record_first, record_second],
        plugins=[AgentPlugin(_graceful_agent), AgentPlugin(_exhaustive_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            ToolCompletionOrderWorkflow.run,
            end_strategy,
            id=f'tool-completion-{uuid.uuid4()}',
            task_queue=task_queue,
        )
        assert await handle.result() == ['first', 'second']
        history = await handle.fetch_history()

    with anyio.fail_after(REPLAY_TIMEOUT):
        await Replayer(
            workflows=[ToolCompletionOrderWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        ).replay_workflow(history)


async def test_history_without_tool_completion_patch_replays(client: Client, monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker upgrade keeps histories without the new marker on the legacy waiting path."""
    original_patched = workflow.patched

    def patched(patch_id: str) -> bool:
        if patch_id == 'pydantic_ai:deterministic_tool_completion':
            return False
        return original_patched(patch_id)

    task_queue = 'legacy-tool-completion'
    with monkeypatch.context() as recording:
        recording.setattr(workflow, 'patched', patched)
        async with Worker(
            client,
            task_queue=task_queue,
            workflows=[LegacyCompletionWaitWorkflow],
            plugins=[AgentPlugin(_legacy_agent)],
            workflow_runner=UnsandboxedWorkflowRunner(),
        ):
            handle = await client.start_workflow(
                LegacyCompletionWaitWorkflow.run,
                id=f'legacy-tool-completion-{uuid.uuid4()}',
                task_queue=task_queue,
            )
            assert await handle.result() == 'legacy'
            history = await handle.fetch_history()

    with anyio.fail_after(REPLAY_TIMEOUT):
        await Replayer(
            workflows=[LegacyCompletionWaitWorkflow],
            workflow_runner=UnsandboxedWorkflowRunner(),
            data_converter=pydantic_data_converter,
        ).replay_workflow(history)

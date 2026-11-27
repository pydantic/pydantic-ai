"""A summarizer nested in `TieredCompaction` or `FallbackCompaction` runs in an activity under Temporal (#14743).

Before, the nested summary call ran in the workflow sandbox, which rejects model I/O.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import uuid4

import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import CompactionPart, ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytest.importorskip('temporalio')

from temporalio import workflow
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.compaction import ClearToolResults, FallbackCompaction, SummarizingCompaction, TieredCompaction
from tests.harness._temporal import ignore_source_reads_left_open, skip_temporal_sandbox_on_314
from tests.temporal_utils import temporal_dev_server_cache_dir

pytestmark = [
    pytest.mark.temporal,
    pytest.mark.xdist_group(name='harness-temporal'),
    skip_temporal_sandbox_on_314,
    ignore_source_reads_left_open,
]

summary_calls: list[str] = []


def _summarizer(label: str) -> SummarizingCompaction[None]:
    def summarize(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        summary_calls.append(label)
        return ModelResponse(parts=[TextPart(f'{label} summary')])

    return SummarizingCompaction(
        model=FunctionModel(summarize), max_messages=1, keep_messages=1, preserve_first_user_message=False
    )


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('done')])


tiered_agent = Agent(
    FunctionModel(_respond),
    name='temporal_nested_tiered',
    deps_type=type(None),
    capabilities=[
        TieredCompaction(tiers=[ClearToolResults(), _summarizer('tier')], target_tokens=1),
        TemporalDurability[None](),
    ],
)
fallback_agent = Agent(
    FunctionModel(_respond),
    name='temporal_nested_fallback',
    deps_type=type(None),
    capabilities=[FallbackCompaction([_summarizer('chain')], max_tokens=1), TemporalDurability[None]()],
)


def _history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('response')]),
        ModelRequest(parts=[UserPromptPart('second')]),
        ModelResponse(parts=[TextPart('response')]),
    ]


def _summaries(messages: list[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, CompactionPart) and part.content is not None
    ]


@workflow.defn
class NestedCompactionWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [tiered_agent, fallback_agent]

    @workflow.run
    async def run(self) -> list[str]:
        tiered = await tiered_agent.run('continue', message_history=_history())
        fallback = await fallback_agent.run('continue', message_history=_history())
        return _summaries(tiered.all_messages()) + _summaries(fallback.all_messages())


async def test_nested_summarizers_run_in_their_composite_activity_and_survive_replay() -> None:
    runner = SandboxedWorkflowRunner(
        restrictions=SandboxRestrictions.default.with_passthrough_modules(__name__, 'annotated_types', 'coverage')
    )
    summary_calls.clear()
    async with await WorkflowEnvironment.start_local(download_dest_dir=temporal_dev_server_cache_dir()) as env:  # pyright: ignore[reportUnknownMemberType]
        client = await Client.connect(env.client.service_client.config.target_host, plugins=[PydanticAIPlugin()])
        workflow_id = uuid4().hex
        async with Worker(
            client, task_queue='nested-compaction', workflows=[NestedCompactionWorkflow], workflow_runner=runner
        ):
            summaries = await client.execute_workflow(
                NestedCompactionWorkflow.run,
                id=workflow_id,
                task_queue='nested-compaction',
                execution_timeout=timedelta(seconds=60),
            )
        history = await client.get_workflow_handle(workflow_id).fetch_history()

    assert summaries == ['tier summary', 'chain summary']
    assert summary_calls == ['tier', 'chain']
    activities = [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in history.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ]
    assert [name for name in activities if '__capability__' in name] == [
        'agent__temporal_nested_tiered__capability__tiered_compaction__compact',
        'agent__temporal_nested_fallback__capability__fallback_compaction__compact',
    ]
    replayer = Replayer(workflows=[NestedCompactionWorkflow], plugins=[PydanticAIPlugin()], workflow_runner=runner)
    replay = await replayer.replay_workflow(history, raise_on_replay_failure=False)
    assert replay.replay_failure is None
    assert summary_calls == ['tier', 'chain'], 'the replay asked a summarizer again'

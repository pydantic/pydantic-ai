"""Compaction under Temporal: the `compact_history` hooks run in workflow code, and a replay reproduces the run.

In a module of its own: a Temporal workflow imports its module again in the sandbox for every run.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import AbstractCapability, CompactionRequest, CompactionResult
from pydantic_ai.exceptions import ModelContextWindowExceededError
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.worker import Replayer, Worker

    from pydantic_ai.durable_exec.temporal import (
        AgentPlugin,
        PydanticAIPlugin,
        TemporalAgent,  # pyright: ignore[reportDeprecated]
        TemporalDurability,
    )

    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

pytestmark = [pytest.mark.xdist_group(name='temporal-compaction')]


def _model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # The first request overflows; once a context-overflow compaction is in the history, the next one fits.
    summaries = [p.content for m in messages for p in m.parts if isinstance(p, CompactionPart)]
    if summaries == ['threshold summary']:
        raise ModelContextWindowExceededError('m', 'context_length_exceeded')
    return ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=10, output_tokens=2))


class _RequestCompaction(AbstractCapability[Any]):
    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        ctx.request_compaction(reason='threshold')
        return request_context


class _Summarize(AbstractCapability[Any]):
    async def compact_history(self, ctx: RunContext[Any], request: CompactionRequest) -> CompactionResult | None:
        part = CompactionPart(content=f'{request.reason} summary')
        return CompactionResult(
            messages=[ModelRequest(parts=[part]), request.messages[-1]],
            part=part,
            handler='summarize',
            usage=RequestUsage(input_tokens=100, output_tokens=5),
        )


_agent = Agent(
    FunctionModel(_model_fn, model_name='m'),
    name='temporal_compaction',
    capabilities=[_RequestCompaction(), _Summarize(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class CompactionWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> tuple[list[str], int, int]:
        result = await _agent.run(prompt)
        usage = result.usage
        return [record.reason for record in result.compactions], usage.input_tokens, usage.output_tokens


async def test_compaction_runs_in_the_workflow_and_replays(client: Client):
    """A hook's requested compaction and a context overflow's compaction run in workflow code, and replay.

    The requested compaction's attempt overflows, so it's discarded and the overflow's compaction is kept; both are
    billed.

    The compaction's priced timestamp comes from the history, not the clock (see
    `test_compaction_usage_is_priced_at_a_timestamp_from_the_history`), so the replay commits the same record.
    """
    async with Worker(client, task_queue=TASK_QUEUE, workflows=[CompactionWorkflow], plugins=[AgentPlugin(_agent)]):
        handle = await client.start_workflow(
            CompactionWorkflow.run,
            args=['Hello'],
            id=CompactionWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
        handlers, input_tokens, output_tokens = await handle.result()
        history = await handle.fetch_history()
    # Both compactions are billed (100 + 5 each), next to the one model request that answered (10 + 2).
    assert (handlers, input_tokens, output_tokens) == (['context_overflow'], 210, 12)

    await Replayer(workflows=[CompactionWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(history)


def _answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # A request here would mean the compaction-only run continued the conversation.
    raise AssertionError('`compact_history` sends no model request')  # pragma: no cover


# Instructions give a run continuing the history something to send, which a regular run then would.
_legacy_agent = TemporalAgent(  # pyright: ignore[reportDeprecated]
    Agent(
        FunctionModel(_answer, model_name='m'),
        name='temporal_legacy_compact_history',
        instructions='Be brief.',
        capabilities=[_Summarize()],
    ),
    activity_config=BASE_ACTIVITY_CONFIG,
)


@workflow.defn
class LegacyCompactHistoryWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> tuple[list[str], list[str]]:
        result = await _legacy_agent.compact_history(
            [ModelRequest(parts=[UserPromptPart(prompt)]), ModelResponse(parts=[TextPart('Hello!')])]
        )
        summaries = [p.content or '' for m in result.all_messages() for p in m.parts if isinstance(p, CompactionPart)]
        return [record.reason for record in result.compactions], summaries


async def test_temporal_agent_compact_history_runs_in_the_workflow_and_replays(client: Client):
    """`TemporalAgent.compact_history()` runs in the workflow that calls it, and a replay compacts again.

    Replay re-executes the workflow code, `compact_history()` call included, so the run is compaction-only again.
    """
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[LegacyCompactHistoryWorkflow],
        plugins=[AgentPlugin(_legacy_agent)],
    ):
        handle = await client.start_workflow(
            LegacyCompactHistoryWorkflow.run,
            args=['Hello'],
            id=LegacyCompactHistoryWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
        assert await handle.result() == (['manual'], ['manual summary'])
        history = await handle.fetch_history()

    await Replayer(workflows=[LegacyCompactHistoryWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(history)

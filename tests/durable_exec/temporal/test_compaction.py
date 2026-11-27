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
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import AbstractCapability, CompactionRequest, CompactionResult, NativeCompaction
from pydantic_ai.exceptions import ModelContextWindowExceededError
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.usage import RequestUsage

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.worker import Replayer, Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin, TemporalDurability

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


_ENCRYPTED = {'encrypted_content': 'gAAAAB-opaque-compaction', 'type': 'compaction'}
"""What a provider's encrypted compaction item carries, which only the provider can read."""


class _NativeCompactingModel(FunctionModel):
    """A model whose on-demand compaction returns an encrypted `CompactionPart`, like OpenAI's."""

    def __init__(self) -> None:
        super().__init__(
            self._respond,
            model_name='native',
            profile=ModelProfile(supports_compaction_on_demand=True, compaction_instructions='replaces_prompt'),
        )

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Answered from the compacted history, so the part sent back is the one the compaction activity returned.
        [part] = [p for m in messages for p in m.parts if isinstance(p, CompactionPart)]
        assert part.provider_details == _ENCRYPTED
        return ModelResponse(parts=[TextPart('done')], usage=RequestUsage(input_tokens=10, output_tokens=2))

    async def compact_messages(
        self, request_context: ModelRequestContext, *, instructions: str | None = None
    ) -> ModelResponse:
        part = CompactionPart(id='cmp_1', provider_name=self.system, provider_details=dict(_ENCRYPTED))
        return ModelResponse(
            parts=[part], usage=RequestUsage(input_tokens=100, output_tokens=5), model_name=self.model_name
        )


_native_agent = Agent(
    _NativeCompactingModel(),
    name='temporal_native_compaction',
    capabilities=[NativeCompaction(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class NativeCompactionWorkflow:
    @workflow.run
    async def run(self, history: str) -> tuple[list[str], str]:
        result = await _native_agent.run(
            'And now?', message_history=ModelMessagesTypeAdapter.validate_json(history), compact=True
        )
        return [record.handler for record in result.compactions], result.all_messages_json().decode()


async def test_native_compaction_part_crosses_an_activity_and_replays(client: Client):
    """`compact=True` with `NativeCompaction` makes the model's compaction an activity, and the run replays.

    The encrypted `CompactionPart` the compaction activity returns crosses the activity payload intact, is sent back
    to the model in the next activity, and is kept in the history the workflow returns. Replaying the recorded history
    reproduces the run: the part comes from the recorded activity result, not a new compaction call.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hello')]),
        ModelResponse(parts=[TextPart('Hi')], model_name='native'),
    ]
    async with Worker(
        client, task_queue=TASK_QUEUE, workflows=[NativeCompactionWorkflow], plugins=[AgentPlugin(_native_agent)]
    ):
        handle = await client.start_workflow(
            NativeCompactionWorkflow.run,
            args=[ModelMessagesTypeAdapter.dump_json(history).decode()],
            id=NativeCompactionWorkflow.__name__,
            task_queue=TASK_QUEUE,
            execution_timeout=timedelta(seconds=30),
        )
        handlers, messages_json = await handle.result()
        recorded = await handle.fetch_history()

    assert handlers == ['native']
    # The compaction is an activity of its own, before the request it serves.
    assert [
        event.activity_task_scheduled_event_attributes.activity_type.name
        for event in recorded.events
        if event.HasField('activity_task_scheduled_event_attributes')
    ] == [
        'agent__temporal_native_compaction__model_compact_messages',
        'agent__temporal_native_compaction__model_request',
    ]
    messages = ModelMessagesTypeAdapter.validate_json(messages_json)
    [part] = [p for m in messages for p in m.parts if isinstance(p, CompactionPart)]
    assert part == CompactionPart(id='cmp_1', provider_name='function', provider_details=_ENCRYPTED)

    await Replayer(workflows=[NativeCompactionWorkflow], plugins=[PydanticAIPlugin()]).replay_workflow(recorded)

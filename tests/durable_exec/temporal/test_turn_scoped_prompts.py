"""Turn-scoped system prompts under Temporal.

A turn-scoped `SystemPromptPart` stays in the message history, so the history records exactly what
each model request was built from, and the request a workflow hands its model activity is
reproducible from that history alone. This runs a workflow whose capability adds one to every
request, checks what the model activity was sent, then replays the workflow's history to show the
run is deterministic.
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RunContext,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

with workflow.unsafe.imports_passed_through():
    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE

pytestmark = [pytest.mark.xdist_group(name='temporal-durability')]


@dataclass
class TurnReminder(AbstractCapability[Any]):
    """Adds a turn-scoped reminder to every model request, from workflow code."""

    requests: int = field(default=0, init=False)

    async def for_run(self, ctx: RunContext[Any]) -> TurnReminder:
        return TurnReminder()

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.requests += 1
        reminder = ModelRequest(parts=[SystemPromptPart(f'Reminder {self.requests}.', scope='turn')])
        ctx.messages.append(reminder)
        request_context.messages = [*request_context.messages, reminder]
        return request_context


_seen_by_model: list[list[str]] = []


def _model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # Runs in the model activity, on the request the workflow prepared for it.
    _seen_by_model.append(
        [
            str(part.content)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and str(part.content).startswith('<system>')
        ]
    )
    if len(_seen_by_model) == 1:
        return ModelResponse(parts=[ToolCallPart('look_up', {'city': 'Paris'})])
    return ModelResponse(parts=[TextPart('Paris is cloudy.')])


def look_up(city: str) -> str:
    return f'{city}: 18C, cloudy'


_agent = Agent(
    FunctionModel(_model_fn),
    name='turn_scoped_prompts',
    tools=[look_up],
    capabilities=[TurnReminder(), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)


@workflow.defn
class TurnScopedPromptsWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _agent.run(prompt)
        return result.all_messages_json().decode()


async def test_turn_scoped_prompts_replay_deterministically(client: Client):
    """The history records every turn-scoped prompt, the activity got only the current ones, and replay agrees."""
    _seen_by_model.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TurnScopedPromptsWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            TurnScopedPromptsWorkflow.run,
            args=['Weather in Paris?'],
            id=f'{TurnScopedPromptsWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        messages = ModelMessagesTypeAdapter.validate_json(await handle.result())
        history = await handle.fetch_history()

    assert [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.scope == 'turn'
    ] == ['Reminder 1.', 'Reminder 2.']
    # `FunctionModel` can't clear a turn-scoped prompt itself, so each request carried only its own.
    assert _seen_by_model == [
        ['<system>Reminder 1.</system>'],
        ['<system>Reminder 2.</system>'],
    ]

    replay = await Replayer(
        workflows=[TurnScopedPromptsWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(history)
    assert replay.replay_failure is None

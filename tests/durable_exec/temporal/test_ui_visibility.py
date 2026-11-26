"""A `'private'` tool's `ui_visibility` stamp under Temporal.

The model activity stamps the call it streams from the tool definitions the workflow hands it, so the stamp has to
cross the activity boundary on the events and stay in the history the workflow returns, for the UI event streams and
`dump_messages` to keep the tool from the client. The workflow's own history then replays, as the stamping adds no
command.
"""

from __future__ import annotations

import json
import sys
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessage, ModelMessagesTypeAdapter, RunContext
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.contrib.pydantic import pydantic_data_converter
    from temporalio.worker import Replayer, UnsandboxedWorkflowRunner, Worker

    from pydantic_ai.durable_exec.temporal import AgentPlugin, TemporalDurability
except ImportError:  # pragma: lax no cover
    pytest.skip('temporal not installed', allow_module_level=True)

try:
    from pydantic_ai.ui.ag_ui import AGUIAdapter, AGUIEventStream
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter, VercelAIEventStream
except ImportError:  # pragma: lax no cover
    pytest.skip('ag-ui-protocol not installed', allow_module_level=True)

if sys.version_info >= (3, 14):  # pragma: lax no cover
    pytest.skip(
        'temporalio sandbox is incompatible with Python 3.14: '
        'sandbox module state accumulates across validation cycles causing import failures after ~22 workflows '
        '(remove when https://github.com/temporalio/sdk-python/issues/1326 closes)',
        allow_module_level=True,
    )

with workflow.unsafe.imports_passed_through():
    from ._shared import BASE_ACTIVITY_CONFIG, TASK_QUEUE

pytestmark = [pytest.mark.xdist_group(name='temporal-ui-visibility')]

SECRET = 'internal-margin-42'

_handled: list[AgentStreamEvent] = []
"""The events the workflow's event stream handler saw."""


async def _keep_events(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for event in stream:
        _handled.append(event)


async def _model_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        yield 'Your order is on its way.'
    else:
        yield {0: DeltaToolCall(name='lookup_margin', json_args='{"order_id": "o_1"}', tool_call_id='call_0')}


def lookup_margin(order_id: str) -> str:
    return SECRET


_agent = Agent(
    FunctionModel(stream_function=_model_fn),
    name='ui_visibility',
    deps_type=type(None),
    capabilities=[ProcessEventStream(_keep_events), TemporalDurability(activity_config=BASE_ACTIVITY_CONFIG)],
)
_agent.tool_plain(ui_visibility='private')(lookup_margin)


@workflow.defn
class PrivateToolWorkflow:
    @workflow.run
    async def run(self, prompt: str) -> str:
        result = await _agent.run(prompt)
        return result.all_messages_json().decode()


async def _events(events: Sequence[AgentStreamEvent]) -> AsyncIterator[AgentStreamEvent]:
    for event in events:
        yield event


async def test_private_tool_stamp_crosses_the_activity_boundary_and_replays(client: Client):
    """The handler's events and the returned history carry the stamp, the UI sends nothing of the tool, and replay agrees."""
    _handled.clear()
    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[PrivateToolWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
    ):
        handle = await client.start_workflow(
            PrivateToolWorkflow.run,
            args=['Where is my order?'],
            id=f'{PrivateToolWorkflow.__name__}-{uuid.uuid4()}',
            task_queue=TASK_QUEUE,
        )
        messages = ModelMessagesTypeAdapter.validate_json(await handle.result())
        history = await handle.fetch_history()
    handled = list(_handled)

    assert [
        event.part.ui_visibility
        for event in handled
        if isinstance(event, PartStartEvent) and isinstance(event.part, ToolCallPart)
    ] == ['private']
    assert [event.part.ui_visibility for event in handled if isinstance(event, FunctionToolResultEvent)] == ['private']
    assert [
        (part.part_kind, part.ui_visibility)
        for message in messages
        for part in message.parts
        if isinstance(part, (ToolCallPart, ToolReturnPart))
    ] == [('tool-call', 'private'), ('tool-return', 'private')]

    ag_ui_stream = AGUIEventStream[None, Any]()
    vercel_stream = VercelAIEventStream[None, Any](sdk_version=6)
    wires = [
        ''.join([ag_ui_stream.encode_event(e) async for e in ag_ui_stream.transform_stream(_events(handled))]),
        ''.join([vercel_stream.encode_event(e) async for e in vercel_stream.transform_stream(_events(handled))]),
        json.dumps([m.model_dump(mode='json', by_alias=True) for m in AGUIAdapter.dump_messages(messages)]),
        json.dumps([m.model_dump(mode='json', by_alias=True) for m in VercelAIAdapter.dump_messages(messages)]),
    ]
    for wire in wires:
        assert 'Your order is on its way.' in wire
        assert 'lookup_margin' not in wire
        assert SECRET not in wire

    replay = await Replayer(
        workflows=[PrivateToolWorkflow],
        plugins=[AgentPlugin(_agent)],
        workflow_runner=UnsandboxedWorkflowRunner(),
        data_converter=pydantic_data_converter,
    ).replay_workflow(history)
    assert replay.replay_failure is None

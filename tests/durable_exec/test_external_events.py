"""Tests for waiting on an external event in a durable run, and for `DurableApprovals`, which waits for a decision.

No core engine implements the wait yet, so these install an in-memory `ExternalEventWaiter` with
`external_event_scope`, the way an engine does for each durable agent run, and drive `DurableApprovals`
through `Agent.run`. The AWS Lambda implementation, with suspension and replay, is tested in
`tests/harness/aws_lambda/test_aws_lambda_approvals.py`.
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DurableApprovals, HandleDeferredToolCalls
from pydantic_ai.durable_exec import (
    ExternalEventError,
    ExternalEventWaiter,
    durable_unit_scope,
    external_event_scope,
    wait_for_external_event,
)
from pydantic_ai.exceptions import CallDeferred, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults


class RecordingWaiter(ExternalEventWaiter):
    """Announces a fixed token, then returns `outcome` as the payload, or raises it."""

    def __init__(self, outcome: JsonValue | ExternalEventError) -> None:
        self.outcome = outcome
        self.waits: list[tuple[str, timedelta | None]] = []

    async def wait(
        self, name: str, *, announce: Callable[[str], Awaitable[None]], timeout: timedelta | None
    ) -> JsonValue:
        self.waits.append((name, timeout))
        await announce('token-1')
        if isinstance(self.outcome, ExternalEventError):
            raise self.outcome
        return self.outcome


def delete_and_look_up(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call an approval-required tool and an external one, then report what came back."""
    if len(messages) == 1:
        return ModelResponse(
            parts=[
                ToolCallPart('delete', {'path': 'a.txt'}, tool_call_id='call-1'),
                ToolCallPart('look_up', {}, tool_call_id='call-2'),
            ]
        )
    return ModelResponse(parts=[TextPart('done')])


def build_agent(
    capabilities: list[Any], deleted: list[str], *, output_type: Any = str
) -> tuple[Agent[None, Any], list[tuple[str, list[str], list[str]]]]:
    announced: list[tuple[str, list[str], list[str]]] = []

    async def announce(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
        announced.append(
            (token, [call.tool_call_id for call in requests.approvals], [call.tool_call_id for call in requests.calls])
        )

    agent = Agent(
        FunctionModel(delete_and_look_up),
        deps_type=type(None),
        output_type=output_type,
        capabilities=[DurableApprovals[None](announce, timeout=timedelta(hours=1)), *capabilities],
    )

    @agent.tool_plain(requires_approval=True)
    def delete(path: str) -> str:
        deleted.append(path)
        return f'deleted {path}'

    @agent.tool_plain
    def look_up() -> str:
        raise CallDeferred

    return agent, announced


def tool_returns(messages: list[ModelMessage]) -> list[tuple[str, object, str]]:
    return [
        (part.tool_call_id, part.content, part.outcome)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


async def test_decision_approves_with_override_args_and_supplies_external_result() -> None:
    deleted: list[str] = []
    agent, announced = build_agent([], deleted)
    waiter = RecordingWaiter(
        {
            'approvals': {'call-1': {'kind': 'tool-approved', 'override_args': {'path': 'b.txt'}}},
            'calls': {'call-2': {'answer': 42}},
        }
    )

    with external_event_scope(waiter):
        result = await agent.run('clean up')

    assert result.output == 'done'
    assert waiter.waits == [('deferred_tool_calls', timedelta(hours=1))]
    assert announced == [('token-1', ['call-1'], ['call-2'])]
    assert deleted == ['b.txt']
    assert tool_returns(result.all_messages()) == snapshot(
        [('call-1', 'deleted b.txt', 'success'), ('call-2', {'answer': 42}, 'success')]
    )


async def test_decision_denies_with_and_without_a_message() -> None:
    deleted: list[str] = []
    agent, _ = build_agent([], deleted)
    waiter = RecordingWaiter(
        {
            'approvals': {'call-1': {'kind': 'tool-denied', 'message': 'Not that file.'}},
            'calls': {'call-2': {'kind': 'tool-failed', 'message': 'Lookup service is down.'}},
        }
    )

    with external_event_scope(waiter):
        result = await agent.run('clean up')

    assert deleted == []
    assert tool_returns(result.all_messages()) == snapshot(
        [('call-1', 'Not that file.', 'denied'), ('call-2', 'Lookup service is down.', 'failed')]
    )

    with external_event_scope(RecordingWaiter({'approvals': {'call-1': False}, 'calls': {'call-2': 'x'}})):
        result = await agent.run('clean up')

    assert tool_returns(result.all_messages())[0] == ('call-1', 'The tool call was denied.', 'denied')


async def test_decision_metadata_reaches_the_approved_tool() -> None:
    seen: list[dict[str, Any] | None] = []
    agent = Agent(
        FunctionModel(delete_and_look_up),
        deps_type=type(None),
        capabilities=[DurableApprovals[None](lambda ctx, requests, token: None)],
    )

    @agent.tool(requires_approval=True)
    def delete(ctx: RunContext[None], path: str) -> str:
        seen.append(ctx.tool_call_metadata)
        return 'deleted'

    @agent.tool_plain
    def look_up() -> str:
        return 'found'

    payload: JsonValue = {'approvals': {'call-1': True}, 'metadata': {'call-1': {'approved_by': 'alice'}}}
    with external_event_scope(RecordingWaiter(payload)):
        await agent.run('clean up')

    assert seen == [{'approved_by': 'alice'}]


async def test_no_decision_denies_approvals_and_fails_external_calls() -> None:
    deleted: list[str] = []
    agent, _ = build_agent([], deleted)

    with external_event_scope(RecordingWaiter(ExternalEventError('Callback timed out'))):
        result = await agent.run('clean up')

    assert deleted == []
    assert tool_returns(result.all_messages()) == snapshot(
        [
            ('call-1', 'No decision on this tool call arrived: Callback timed out', 'denied'),
            ('call-2', 'No decision on this tool call arrived: Callback timed out', 'failed'),
        ]
    )


async def test_partial_decision_leaves_the_rest_pending() -> None:
    deleted: list[str] = []
    agent, _ = build_agent([], deleted, output_type=[str, DeferredToolRequests])

    with external_event_scope(RecordingWaiter({'approvals': {'call-1': True}})):
        result = await agent.run('clean up')

    assert deleted == ['a.txt']
    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_call_id for call in result.output.calls] == ['call-2']
    assert result.output.approvals == []


@pytest.mark.parametrize(
    ('payload', 'error'),
    [
        pytest.param(
            {'approvals': {'call-1': 'yes'}},
            r'(?s)The decision sent for the pending deferred tool calls is not valid: .*approvals\.call-1',
            id='not-deferred-tool-results',
        ),
        pytest.param(
            {'approvals': {'call-0': True}},
            r"not valid: `approvals` contains tool call IDs not in this `DeferredToolRequests.approvals`: \['call-0'\]",
            id='stale-tool-call-id',
        ),
        pytest.param(
            {'approvals': {'call-1': True}, 'metadata': {'call-0': {'approved_by': 'alice'}}},
            r"not valid: `metadata` contains tool call IDs that are not pending: \['call-0'\]",
            id='metadata-for-a-call-that-is-not-pending',
        ),
    ],
)
async def test_invalid_decision_fails_the_run(payload: JsonValue, error: str) -> None:
    deleted: list[str] = []
    agent, _ = build_agent([], deleted)

    with external_event_scope(RecordingWaiter(payload)), pytest.raises(UserError, match=error):
        await agent.run('clean up')

    assert deleted == []


async def test_other_handlers_resolve_calls_before_the_wait() -> None:
    """`DurableApprovals` is innermost, so a handler listed after it is still asked first."""
    deleted: list[str] = []

    def resolve_external(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(calls={call.tool_call_id: 'from handler' for call in requests.calls})

    agent, announced = build_agent([HandleDeferredToolCalls(resolve_external)], deleted)

    with external_event_scope(RecordingWaiter({'approvals': {'call-1': True}})):
        result = await agent.run('clean up')

    assert announced == [('token-1', ['call-1'], [])]
    assert tool_returns(result.all_messages()) == snapshot(
        [('call-1', 'deleted a.txt', 'success'), ('call-2', 'from handler', 'success')]
    )


async def test_without_a_durable_engine_the_run_fails() -> None:
    agent, _ = build_agent([], [])

    with pytest.raises(UserError, match=r'can only be called inside a durable agent run on an engine that supports'):
        await agent.run('clean up')


async def test_a_sync_announce_runs_in_a_worker_thread() -> None:
    """Announcing is typically blocking I/O (Slack, email, a database row), so a sync `announce` runs off the event loop."""
    announced: list[tuple[str, int]] = []
    waiter = RecordingWaiter({'answer': 'yes'})

    with external_event_scope(waiter):
        payload = await wait_for_external_event(
            'review', announce=lambda token: announced.append((token, threading.get_ident()))
        )

    assert payload == {'answer': 'yes'}
    [(token, thread_id)] = announced
    assert token == 'token-1'
    assert thread_id != threading.get_ident()
    assert waiter.waits == [('review', None)]


async def test_wait_rejects_a_timeout_that_is_not_positive() -> None:
    with pytest.raises(UserError, match=r'`timeout` must be positive, got datetime.timedelta\(0\)'):
        await wait_for_external_event('review', announce=lambda token: None, timeout=timedelta(0))
    with pytest.raises(UserError, match=r'`timeout` must be positive, got datetime.timedelta\(days=-1'):
        DurableApprovals[None](lambda ctx, requests, token: None, timeout=timedelta(seconds=-1))


async def test_wait_inside_a_durable_unit_is_rejected() -> None:
    """A tool call runs in a durable unit whose result the engine records, so a wait can't be recorded inside it."""
    waiter = RecordingWaiter({'answer': 'yes'})

    with external_event_scope(waiter), durable_unit_scope():
        with pytest.raises(UserError, match=r'was called from inside a durable unit, such as a tool call'):
            await wait_for_external_event('review', announce=lambda token: None)

    assert waiter.waits == []


def test_not_spec_serializable() -> None:
    assert DurableApprovals.get_serialization_name() is None

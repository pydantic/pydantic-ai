"""Tests for waiting on a decision inside an AWS Lambda durable execution, with `DurableApprovals`.

The run creates a Lambda callback, announces its ID in a step, and suspends on the callback's
result. A test resolves the callback on the fake context the way `SendDurableExecutionCallbackSuccess`
would, and the next invocation replays the journal: the model is not called again, the
announcement is not repeated, and the wait returns the decision.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable, AsyncIterator
from datetime import timedelta
from typing import Any

import pytest
from aws_durable_execution_sdk_python.config import StepSemantics
from aws_durable_execution_sdk_python.exceptions import SuspendExecution
from inline_snapshot import snapshot

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import DurableApprovals
from pydantic_ai.durable_exec import wait_for_external_event
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai_harness.aws_lambda import AWSLambdaDurability, run_durable

from .conftest import FakeDurableContext


def delete_then_report(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('delete', {'path': 'a.txt'}, tool_call_id='call-1')])
    returned = next(part for part in messages[-1].parts if isinstance(part, ToolReturnPart))
    return ModelResponse(parts=[TextPart(f'{returned.outcome}: {returned.content}')])


class Approvals:
    """An agent whose `delete` tool needs approval, and a record of what was announced and deleted."""

    def __init__(self, *, timeout: timedelta | None = timedelta(minutes=30), **durability: Any) -> None:
        self.announced: list[tuple[str, list[str]]] = []
        self.deleted: list[str] = []
        self.model_calls = 0

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.model_calls += 1
            return delete_then_report(messages, info)

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            for part in model_fn(messages, info).parts:
                if isinstance(part, TextPart):
                    yield part.content
                else:
                    assert isinstance(part, ToolCallPart)
                    yield {
                        0: DeltaToolCall(
                            name=part.tool_name, json_args=part.args_as_json_str(), tool_call_id=part.tool_call_id
                        )
                    }

        async def announce(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
            self.announced.append((token, [call.tool_call_id for call in requests.approvals]))

        self.agent = Agent(
            FunctionModel(model_fn, stream_function=stream_fn),
            name='a',
            deps_type=type(None),
            capabilities=[AWSLambdaDurability[None](**durability), DurableApprovals[None](announce, timeout=timeout)],
        )

        @self.agent.tool_plain(requires_approval=True)
        def delete(path: str) -> str:
            self.deleted.append(path)
            return f'deleted {path}'

    def suspend(self) -> FakeDurableContext:
        """Run the first invocation, which suspends waiting for the decision."""
        ctx = FakeDurableContext()
        with pytest.raises(SuspendExecution):
            run_durable(lambda: self.agent.run('clean up'), context=ctx)
        return ctx


def test_the_run_suspends_until_the_decision_and_resumes_from_the_journal() -> None:
    approvals = Approvals()

    first = approvals.suspend()

    assert first.step_names == snapshot(
        ['a__model.request', 'a__external_event.deferred_tool_calls', 'a__external_event.deferred_tool_calls.announce']
    )
    assert approvals.announced == [('callback-1', ['call-1'])]
    assert approvals.deleted == []
    callback = first.operations[1]
    assert callback.callback_config is not None
    assert callback.callback_config.timeout_seconds == 1800

    first.complete_callback(
        'callback-1',
        json.dumps({'approvals': {'call-1': {'kind': 'tool-approved', 'override_args': {'path': 'b.txt'}}}}),
    )
    resumed = FakeDurableContext(journal=first.operations)
    result = run_durable(lambda: approvals.agent.run('clean up'), context=resumed)

    assert result.output == 'success: deleted b.txt'
    assert approvals.deleted == ['b.txt']
    assert approvals.announced == [('callback-1', ['call-1'])], 'replay does not announce again'
    assert approvals.model_calls == 2, 'the first model response came from its checkpoint'
    assert resumed.invoked == ['a__function_toolset__<agent>.call_tool:delete', 'a__model.request']

    replayed = FakeDurableContext(journal=resumed.operations)
    assert run_durable(lambda: approvals.agent.run('clean up'), context=replayed).output == 'success: deleted b.txt'
    assert replayed.invoked == []
    assert approvals.deleted == ['b.txt']


def test_a_denial_reaches_the_model_without_running_the_tool() -> None:
    approvals = Approvals()
    first = approvals.suspend()

    first.complete_callback('callback-1', '{"approvals": {"call-1": {"kind": "tool-denied", "message": "Not today."}}}')
    result = run_durable(lambda: approvals.agent.run('clean up'), context=FakeDurableContext(journal=first.operations))

    assert result.output == 'denied: Not today.'
    assert approvals.deleted == []


def test_a_timed_out_callback_denies_the_call() -> None:
    approvals = Approvals()
    first = approvals.suspend()

    first.fail_callback('callback-1', 'Callback timed out')
    result = run_durable(lambda: approvals.agent.run('clean up'), context=FakeDurableContext(journal=first.operations))

    assert result.output == 'denied: No decision on this tool call arrived: Callback timed out'
    assert approvals.deleted == []


@pytest.mark.parametrize(
    ('result', 'error'),
    [
        pytest.param(
            'approve',
            r"The result sent to AWS Lambda callback 'a__external_event.deferred_tool_calls' is not valid JSON",
            id='not-json',
        ),
        pytest.param(None, r'The decision sent for the pending deferred tool calls is not valid', id='no-result'),
    ],
)
def test_a_result_that_is_not_a_decision_fails_the_run(result: str | None, error: str) -> None:
    approvals = Approvals()
    first = approvals.suspend()

    first.complete_callback('callback-1', result)
    with pytest.raises(UserError, match=error):
        run_durable(lambda: approvals.agent.run('clean up'), context=FakeDurableContext(journal=first.operations))

    assert approvals.deleted == []


def test_no_timeout_and_the_base_step_config_reach_the_callback() -> None:
    approvals = Approvals(timeout=None, step_config={'step_semantics': StepSemantics.AT_MOST_ONCE_PER_RETRY})

    first = approvals.suspend()

    callback, announce = first.operations[1], first.operations[2]
    assert callback.callback_config is not None
    assert callback.callback_config.timeout_seconds == 0
    assert announce.config is not None
    assert announce.config.step_semantics == StepSemantics.AT_MOST_ONCE_PER_RETRY


def test_a_sub_second_timeout_rounds_up_to_a_whole_second() -> None:
    first = Approvals(timeout=timedelta(milliseconds=1500)).suspend()

    callback = first.operations[1]
    assert callback.callback_config is not None
    assert callback.callback_config.timeout_seconds == 2


def test_outside_a_durable_handler_the_wait_is_rejected() -> None:
    approvals = Approvals()

    with pytest.raises(UserError, match=r'can only be called inside a durable agent run'):
        approvals.agent.run_sync('clean up')


def test_waiting_inside_a_tool_step_is_rejected() -> None:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('ask', {})])

    agent = Agent(FunctionModel(model_fn), name='a', capabilities=[AWSLambdaDurability()])

    @agent.tool_plain
    async def ask() -> str:
        await wait_for_external_event('question', announce=lambda token: None)
        raise AssertionError('unreachable')  # pragma: no cover

    with pytest.raises(UserError, match=r'was called from inside a durable unit, such as a tool call'):
        run_durable(lambda: agent.run('go'), context=FakeDurableContext())


def test_waiting_inside_the_announce_step_is_rejected() -> None:
    async def announce(ctx: RunContext[None], requests: DeferredToolRequests, token: str) -> None:
        await wait_for_external_event('nested', announce=lambda nested_token: None)

    agent = Agent(
        FunctionModel(delete_then_report),
        name='a',
        deps_type=type(None),
        capabilities=[AWSLambdaDurability[None](), DurableApprovals[None](announce)],
    )

    @agent.tool_plain(requires_approval=True)
    def delete(path: str) -> str:
        raise AssertionError('unreachable')  # pragma: no cover

    with pytest.raises(UserError, match=r'was called from inside a durable unit, such as a tool call or an `announce`'):
        run_durable(lambda: agent.run('clean up'), context=FakeDurableContext())


def test_event_stream_handler_steps_and_the_wait_replay_in_order() -> None:
    """The `DeferredToolRequestsEvent` handler step and the wait share the bridge's queue, so replay meets them in the same order."""
    events: list[str] = []

    async def handle_events(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(type(event).__name__)

    approvals = Approvals(event_stream_handler=handle_events)
    first = approvals.suspend()
    names = first.step_names
    assert names == snapshot(
        [
            'a__model.request_stream',
            'a__event_stream_handler',
            'a__event_stream_handler',
            'a__external_event.deferred_tool_calls',
            'a__external_event.deferred_tool_calls.announce',
        ]
    )
    assert 'DeferredToolRequestsEvent' in events

    [(callback_id, _)] = approvals.announced
    first.complete_callback(callback_id, '{"approvals": {"call-1": true}}')
    resumed = FakeDurableContext(journal=first.operations)
    result = run_durable(lambda: approvals.agent.run('clean up'), context=resumed)

    assert result.output == 'success: deleted a.txt'
    assert resumed.step_names[: len(names)] == names
    assert approvals.deleted == ['a.txt']

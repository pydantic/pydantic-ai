"""Tests for what a deferred tool call waits for (`DeferredToolRequests.kinds`) and who answered it (`DeferredToolResults.answered_by`)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, HandleDeferredToolCalls
from pydantic_ai.capabilities.abstract import ValidatedToolArgs
from pydantic_ai.exceptions import CallDeferred, ModelRetry, ToolFailed
from pydantic_ai.messages import (
    AgentStreamEvent,
    DeferredToolResultsEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition


def ask_look_up_and_delete(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    # Every run ends with a call still pending, so the model is only asked once.
    return ModelResponse(
        parts=[
            ToolCallPart('ask', {}, tool_call_id='question'),
            ToolCallPart('look_up', {}, tool_call_id='external'),
            ToolCallPart('delete', {}, tool_call_id='approval'),
        ]
    )


async def stream_ask_look_up_and_delete(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
    for index, part in enumerate(ask_look_up_and_delete(messages, info).parts):
        assert isinstance(part, ToolCallPart)
        yield {index: DeltaToolCall(name=part.tool_name, json_args='{}', tool_call_id=part.tool_call_id)}


def deferring_agent(**kwargs: Any) -> Agent[None, str | DeferredToolRequests]:
    agent = Agent(
        FunctionModel(ask_look_up_and_delete, stream_function=stream_ask_look_up_and_delete),
        output_type=[str, DeferredToolRequests],
        deps_type=type(None),
        **kwargs,
    )

    @agent.tool_plain
    def ask() -> str:
        raise CallDeferred(metadata={'topic': 'scope'}, kind='question')

    @agent.tool_plain
    def look_up() -> str:
        raise CallDeferred

    @agent.tool_plain(requires_approval=True)
    def delete() -> str:
        raise AssertionError('not approved')  # pragma: no cover

    return agent


@pytest.mark.parametrize('end_strategy', ['early', 'graceful', 'exhaustive'])
async def test_a_question_is_marked_in_the_requests(end_strategy: Any) -> None:
    result = await deferring_agent(end_strategy=end_strategy).run('go')

    requests = result.output
    assert isinstance(requests, DeferredToolRequests)
    assert requests.kinds == {'question': 'question'}
    assert [call.tool_call_id for call in requests.calls] == ['question', 'external'], (
        'in the order the model made them'
    )
    assert {call.tool_call_id: requests.kind(call.tool_call_id) for call in [*requests.calls, *requests.approvals]} == {
        'question': 'question',
        'external': 'external',
        'approval': 'approval',
    }
    assert requests.metadata == {'question': {'topic': 'scope'}}
    with pytest.raises(ValueError, match="No pending deferred tool call has ID 'missing'"):
        requests.kind('missing')
    assert result.conversation.deferred_tool_requests == requests


async def test_answered_calls_leave_the_kinds_and_answerers_reach_the_results_event() -> None:
    """A handler answers the question and says who answered; the rest stay pending with their kinds."""
    events: list[AgentStreamEvent] = []

    def answer_question(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(calls={'question': 'everything'}, answered_by={'question': 'alice'})

    async def record(ctx: RunContext[None], stream: Any) -> None:
        async for event in stream:
            events.append(event)

    agent = deferring_agent(capabilities=[HandleDeferredToolCalls(answer_question)])
    result = await agent.run('go', event_stream_handler=record)

    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.kinds == {}
    assert [call.tool_call_id for call in result.output.calls] == ['external']
    [results_event] = [event for event in events if isinstance(event, DeferredToolResultsEvent)]
    assert results_event.results.answered_by == {'question': 'alice'}


def test_answerers_for_calls_that_are_not_pending_are_dropped() -> None:
    requests = DeferredToolRequests(calls=[ToolCallPart('look_up', {}, tool_call_id='a')])
    results = DeferredToolResults(calls={'a': 1}, answered_by={'a': 'alice', 'b': 'bob'})
    merged = DeferredToolResults()
    merged.update(results)

    assert merged.answered_by == {'a': 'alice', 'b': 'bob'}
    assert requests.remaining(results) is None
    assert TypeAdapter(DeferredToolResults).validate_json(
        TypeAdapter(DeferredToolResults).dump_json(results)
    ).answered_by == snapshot({'a': 'alice', 'b': 'bob'})


async def test_a_question_deferred_by_an_args_validator_is_marked() -> None:
    def ask_for_scope(ctx: RunContext[None]) -> None:
        raise CallDeferred(kind='question')

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart('delete', {}, tool_call_id='approval')])

    agent = Agent(FunctionModel(model), output_type=[str, DeferredToolRequests], deps_type=type(None))

    @agent.tool_plain(requires_approval=True, args_validator=ask_for_scope)
    def delete() -> str:
        raise AssertionError('not approved')  # pragma: no cover

    result = await agent.run('go')

    assert isinstance(result.output, DeferredToolRequests)
    assert result.output.kind('approval') == 'question'


async def test_an_approved_call_that_asks_a_question_stays_pending_as_one() -> None:
    """A call a handler resolves can defer again when it runs; it joins the calls still pending with its kind."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('look_up', {}, tool_call_id='external'),
                ToolCallPart('delete', {}, tool_call_id='approval'),
            ]
        )

    def approve_delete(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(approvals={'approval': True})

    agent = Agent(
        FunctionModel(model),
        output_type=[str, DeferredToolRequests],
        deps_type=type(None),
        capabilities=[HandleDeferredToolCalls(approve_delete)],
    )

    @agent.tool_plain
    def look_up() -> str:
        raise CallDeferred

    @agent.tool_plain(requires_approval=True)
    def delete() -> str:
        raise CallDeferred(kind='question')

    result = await agent.run('go')

    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_call_id for call in result.output.calls] == ['external', 'approval']
    assert result.output.kinds == {'approval': 'question'}


async def test_an_answer_given_during_the_run_passes_the_tools_result_hooks() -> None:
    """A question the tool raised itself, answered by a handler, passes `after_tool_execute` as the tool's own result would."""
    seen: list[Any] = []

    class Redact(AbstractCapability[None]):
        async def after_tool_execute(
            self,
            ctx: RunContext[None],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: ValidatedToolArgs,
            result: Any,
        ) -> Any:
            seen.append(result)
            return result.replace('hunter2', '[redacted]') if isinstance(result, str) else result

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('ask', {}, tool_call_id='question'),
                    ToolCallPart('look_up', {}, tool_call_id='external'),
                ]
            )
        return ModelResponse(
            parts=[
                TextPart(
                    ' | '.join(str(part.content) for part in messages[-1].parts if isinstance(part, ToolReturnPart))
                )
            ]
        )

    def answer(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(calls={'question': 'my password is hunter2', 'external': 'hunter2 found'})

    agent = Agent(
        FunctionModel(model),
        deps_type=type(None),
        capabilities=[Redact(), HandleDeferredToolCalls(answer)],
    )

    @agent.tool_plain
    def ask() -> str:
        raise CallDeferred(kind='question')

    @agent.tool_plain
    def look_up() -> str:
        raise CallDeferred

    result = await agent.run('go')

    assert result.output == 'my password is [redacted] | hunter2 found'
    assert seen == ['my password is hunter2'], "an external result is not the tool's own, so it is passed on as is"


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        (ModelRetry('Answer in English.'), 'Answer in English.'),
        (ToolFailed('The answer was blocked.'), 'The answer was blocked.'),
    ],
)
async def test_a_result_hook_can_reject_an_answer_given_during_the_run(error: Exception, expected: str) -> None:
    class Block(AbstractCapability[None]):
        async def after_tool_execute(
            self,
            ctx: RunContext[None],
            *,
            call: ToolCallPart,
            tool_def: ToolDefinition,
            args: ValidatedToolArgs,
            result: Any,
        ) -> Any:
            raise error

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('ask', {}, tool_call_id='question')])
        return ModelResponse(parts=[TextPart(str([getattr(part, 'content', None) for part in messages[-1].parts]))])

    def answer(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return requests.build_results(calls={'question': ToolReturn('nein', content='notes')})

    agent = Agent(FunctionModel(model), deps_type=type(None), capabilities=[Block(), HandleDeferredToolCalls(answer)])

    @agent.tool_plain
    def ask() -> str:
        raise CallDeferred(kind='question')

    assert expected in (await agent.run('go')).output

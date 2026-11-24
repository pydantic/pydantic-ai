"""A tool call deferred while a run ends on a final result still gets a result in the history.

Providers like OpenAI and Anthropic reject a request with a tool call that has no result, so every call the
final response made needs one, whether the run deferred it by its tool's kind or the tool deferred it while
running. `FunctionModel` makes the deferring call and the output call in one response, which is what's under
test, not anything a provider decides.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, ApprovalRequired, CallDeferred, EndStrategy
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.run import AgentRunResultEvent

DEFERRED_RESULT = 'Tool call deferred - the run ended on a final result before its result was collected.'
SKIPPED_RESULT = 'Tool not executed - a final result was already processed.'

Deferral = Literal['approval_required', 'call_deferred', 'requires_approval']
Route = Literal['run', 'run_stream', 'run_stream_events']


class Summary(BaseModel):
    text: str


CALLS = [
    ToolCallPart('delete_file', {'path': 'a.txt'}, tool_call_id='call_delete'),
    ToolCallPart('final_result', {'text': 'done'}, tool_call_id='call_output'),
]


def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """One response with a call to a tool that defers, and the output tool."""
    return ModelResponse(parts=list(CALLS))


async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
    for index, call in enumerate(CALLS):
        yield {index: DeltaToolCall(call.tool_name, call.args_as_json_str(), tool_call_id=call.tool_call_id)}


def deferring_agent(end_strategy: EndStrategy, deferral: Deferral) -> Agent[None, Summary]:
    agent = Agent(FunctionModel(respond, stream_function=stream), output_type=Summary, end_strategy=end_strategy)

    # With `requires_approval=True`, the call is deferred by its tool's kind, so the tool never runs.
    @agent.tool_plain(requires_approval=deferral == 'requires_approval')
    def delete_file(path: str) -> str:
        if deferral == 'approval_required':
            raise ApprovalRequired
        raise CallDeferred

    return agent


async def run_agent(
    agent: Agent[None, Summary], route: Route
) -> tuple[Summary, list[ModelMessage], list[AgentStreamEvent]]:
    events: list[AgentStreamEvent] = []
    if route == 'run':
        result = await agent.run('Clean up.')
        return result.output, result.all_messages(), events
    if route == 'run_stream':
        async with agent.run_stream('Clean up.') as streamed:
            output = await streamed.get_output()
        return output, streamed.all_messages(), events
    async with agent.run_stream_events('Clean up.') as stream_events:
        async for event in stream_events:
            if isinstance(event, AgentRunResultEvent):
                return event.result.output, event.result.all_messages(), events
            events.append(event)
    raise AssertionError('The run ended without a result event.')  # pragma: no cover


def delete_file_results(messages: list[ModelMessage]) -> list[str]:
    return [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart) and part.tool_call_id == 'call_delete'
    ]


@pytest.mark.parametrize('route', ['run', 'run_stream', 'run_stream_events'])
@pytest.mark.parametrize('deferral', ['approval_required', 'call_deferred'])
@pytest.mark.parametrize('end_strategy', ['graceful', 'exhaustive'])
async def test_call_deferred_while_running_gets_a_result_when_the_run_ends_on_output(
    end_strategy: EndStrategy, deferral: Deferral, route: Route
):
    """The call ran before it deferred, so its result doesn't say it wasn't executed, only that it wasn't collected."""
    output, messages, events = await run_agent(deferring_agent(end_strategy, deferral), route)

    assert output == Summary(text='done')
    assert delete_file_results(messages) == [DEFERRED_RESULT]
    if route == 'run_stream_events':
        call_events = [
            e for e in events if isinstance(e, FunctionToolCallEvent) and e.part.tool_call_id == 'call_delete'
        ]
        result_events = [
            e for e in events if isinstance(e, FunctionToolResultEvent) and e.tool_call_id == 'call_delete'
        ]
        assert len(call_events) == 1
        assert [e.part.content for e in result_events] == [DEFERRED_RESULT]


# `run_stream()` raises `UserError` for a call deferred by its tool's kind next to an output it committed, which is
# tracked separately from the calls deferred while running that this module is about.
@pytest.mark.parametrize('route', ['run', 'run_stream_events'])
@pytest.mark.parametrize('end_strategy', ['graceful', 'exhaustive'])
async def test_call_deferred_by_its_tool_kind_is_not_executed_when_the_run_ends_on_output(
    end_strategy: EndStrategy, route: Route
):
    """A call that needs approval by its tool's kind never ran, so it keeps the 'not executed' result."""
    output, messages, _ = await run_agent(deferring_agent(end_strategy, 'requires_approval'), route)

    assert output == Summary(text='done')
    assert delete_file_results(messages) == [SKIPPED_RESULT]


async def test_calls_deferred_while_running_get_results_in_the_response_order():
    """Calls deferred different ways land in different groups, but their results follow the order the model called them."""

    def respond_with_both(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # `CallDeferred` is grouped after `ApprovalRequired`, so this order differs from the grouping.
        return ModelResponse(
            parts=[
                ToolCallPart('start_export', {}, tool_call_id='call_export'),
                ToolCallPart('delete_file', {'path': 'a.txt'}, tool_call_id='call_delete'),
                ToolCallPart('final_result', {'text': 'done'}, tool_call_id='call_output'),
            ]
        )

    agent = Agent(FunctionModel(respond_with_both), output_type=Summary)

    @agent.tool_plain
    def start_export() -> str:
        raise CallDeferred

    @agent.tool_plain
    def delete_file(path: str) -> str:
        raise ApprovalRequired

    result = await agent.run('Export, then clean up.')

    last_request = result.all_messages()[-1]
    assert isinstance(last_request, ModelRequest)
    assert [(part.tool_call_id, part.content) for part in last_request.parts if isinstance(part, ToolReturnPart)] == [
        ('call_output', 'Final result processed.'),
        ('call_export', DEFERRED_RESULT),
        ('call_delete', DEFERRED_RESULT),
    ]

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import CallDeferred
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults
from pydantic_ai_harness import DenyUnanswered
from pydantic_ai_harness.ask_user import AskUser
from pydantic_ai_harness.deny_unanswered import DEFAULT_QUESTION_MESSAGE

QUESTION = [{'header': 'Check', 'question': 'Proceed?', 'options': [{'label': 'Yes'}, {'label': 'No'}]}]


def _call_then_echo(tool_name: str, args: dict[str, object], call_id: str, final: str):
    """Model that makes one scripted tool call, then finishes with the tool result."""

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(content=final)])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id=call_id)])

    return model


def _tool_returns(messages: list[ModelMessage]) -> list[ToolReturnPart]:
    return [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]


def _approval_agent(ran: list[int], *, deny: bool) -> Agent[None, str | DeferredToolRequests]:
    """An agent whose model calls `danger`, which needs approval, recording each run of its body in `ran`."""
    agent = Agent(
        FunctionModel(_call_then_echo('danger', {'x': 1}, 'c1', 'done')),
        capabilities=[DenyUnanswered()] if deny else [],
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain(requires_approval=True)
    def danger(x: int) -> str:
        ran.append(x)
        return 'ran'

    return agent


async def test_deny_unanswered_answers_approval_required_tool_in_run():
    """A held approval is denied in-run with the capability's message; the tool body never runs."""
    ran: list[int] = []
    result = await _approval_agent(ran, deny=True).run('do it')
    assert result.output == 'done'
    assert ran == []
    assert any(
        'No one was available to approve it' in str(part.content) for part in _tool_returns(result.all_messages())
    )


@pytest.mark.parametrize(
    ('question_message', 'expected'),
    [
        pytest.param(None, DEFAULT_QUESTION_MESSAGE, id='default'),
        pytest.param('Nobody is on duty; decide yourself.', 'Nobody is on duty; decide yourself.', id='custom'),
    ],
)
async def test_deny_unanswered_answers_ask_user_question_in_run(question_message: str | None, expected: str):
    """An unanswered `AskUser` question gets `question_message` as its tool result, and the run finishes."""
    agent = Agent(
        FunctionModel(_call_then_echo('ask_user_question', {'questions': QUESTION}, 'q1', 'done')),
        capabilities=[
            AskUser(answerer=None),
            DenyUnanswered() if question_message is None else DenyUnanswered(question_message=question_message),
        ],
        output_type=[str, DeferredToolRequests],
    )

    result = await agent.run('go')
    assert result.output == 'done'
    assert [(part.tool_name, part.content) for part in _tool_returns(result.all_messages())] == [
        ('ask_user_question', expected)
    ]


async def test_deny_unanswered_leaves_other_deferred_calls_deferred():
    """A deferred call that is neither an approval nor an `AskUser` question is left for its own handler."""
    agent = Agent(
        FunctionModel(_call_then_echo('wait_for_job', {}, 'j1', 'unreached')),
        capabilities=[DenyUnanswered()],
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain
    def wait_for_job() -> str:
        raise CallDeferred

    result = await agent.run('start the job')
    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_call_id for call in result.output.calls] == ['j1']
    assert result.output.approvals == []


async def test_without_capability_calls_stay_deferred():
    """Without `DenyUnanswered` the run pauses on `DeferredToolRequests`, and the caller's approval runs the tool."""
    ran: list[int] = []
    agent = _approval_agent(ran, deny=False)

    result = await agent.run('do it')
    assert isinstance(result.output, DeferredToolRequests)
    assert [call.tool_call_id for call in result.output.approvals] == ['c1']
    assert ran == []

    resumed = await agent.run(
        message_history=result.all_messages(), deferred_tool_results=DeferredToolResults(approvals={'c1': True})
    )
    assert resumed.output == 'done'
    assert ran == [1]

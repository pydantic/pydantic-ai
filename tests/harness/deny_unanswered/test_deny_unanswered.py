import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai_harness import DenyUnanswered
from pydantic_ai_harness.ask_user import AskUser

QUESTION = [{'header': 'Check', 'question': 'Proceed?', 'options': ['Yes', 'No']}]


def anyio_backend():
    return 'asyncio'


pytestmark = pytest.mark.anyio


def _call_then_echo(tool_name: str, args: dict[str, object], call_id: str, final: str):
    """Model that makes one scripted tool call, then finishes with the tool result."""

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(content=final)])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id=call_id)])

    return model


async def test_deny_unanswered_answers_approval_required_tool_in_run():
    """A held approval is denied in-run with the capability's message; the tool body never runs."""
    agent = Agent(
        FunctionModel(_call_then_echo('danger', {'x': 1}, 'c1', 'done')),
        capabilities=[DenyUnanswered()],
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain(requires_approval=True)
    def danger(x: int) -> str:
        raise AssertionError('the tool body must not run')

    result = await agent.run('do it')
    assert result.output == 'done'
    tool_returns = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    assert any('No one was available to approve it' in str(part.content) for part in tool_returns)


async def test_deny_unanswered_answers_ask_user_question_in_run():
    """An unanswered `AskUser` question is resolved through the answerer plumbing as cancelled."""
    agent = Agent(
        FunctionModel(_call_then_echo('ask_user_question', {'questions': QUESTION}, 'q1', 'done')),
        capabilities=[AskUser(answerer=None), DenyUnanswered()],
        output_type=[str, DeferredToolRequests],
    )

    result = await agent.run('go')
    assert result.output == 'done'
    tool_returns = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ToolReturnPart)
    ]
    assert any(part.tool_name == 'ask_user_question' for part in tool_returns)


async def test_without_capability_calls_stay_deferred():
    """Without `DenyUnanswered` the run still pauses on `DeferredToolRequests`."""
    agent = Agent(
        FunctionModel(_call_then_echo('danger', {'x': 1}, 'c1', 'unreached')),
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool_plain(requires_approval=True)
    def danger(x: int) -> str:
        raise AssertionError('the tool body must not run')

    result = await agent.run('do it')
    assert isinstance(result.output, DeferredToolRequests)

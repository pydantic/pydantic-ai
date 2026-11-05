import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import DenyUnanswered, SubAgents
from pydantic_ai_harness.ask_user import AskUser
from pydantic_ai_harness.subagents import SubAgent

QUESTION = [{'header': 'Check', 'question': 'Proceed?', 'options': [{'label': 'Yes'}, {'label': 'No'}]}]


def anyio_backend():
    return 'asyncio'


pytestmark = pytest.mark.anyio


def _call_then_echo(tool_name: str, args: dict[str, object], call_id: str, final: str, observed: list[str]):
    """Model that makes one scripted tool call, then finishes with the tool result."""

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                observed.append(str(part.content))
                return ModelResponse(parts=[TextPart(content=final)])
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args, tool_call_id=call_id)])

    return model


async def test_deny_unanswered_reaches_subagents_children_by_default():
    """A delegation child denies its own unanswered question because the parent run carries the capability."""
    observed: list[str] = []
    child = Agent(
        FunctionModel(_call_then_echo('ask_user_question', {'questions': QUESTION}, 'q1', 'child done', observed)),
        name='child',
        capabilities=[AskUser(answerer=None)],
    )

    async def parent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(content='parent done')])
        delegate = next(tool for tool in info.function_tools if 'delegate' in tool.name)
        return ModelResponse(
            parts=[
                ToolCallPart(tool_name=delegate.name, args={'agent_name': 'child', 'task': 'ask'}, tool_call_id='d1')
            ]
        )

    parent = Agent(
        FunctionModel(parent_model),
        capabilities=[DenyUnanswered(), SubAgents(agents=[SubAgent(child)])],
    )

    result = await parent.run('delegate')
    assert result.output == 'parent done'
    assert result.output == 'parent done'
    assert any('declined to answer' in content for content in observed)

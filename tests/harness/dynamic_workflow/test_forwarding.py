import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import DenyUnanswered, DynamicWorkflow
from pydantic_ai_harness.ask_user import AskUser

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


async def test_deny_unanswered_reaches_dynamic_workflow_children_by_default():
    """A workflow child run denies its own unanswered question because the parent run carries the capability."""
    observed: list[str] = []
    child = Agent(
        FunctionModel(_call_then_echo('ask_user_question', {'questions': QUESTION}, 'q1', 'child done', observed)),
        name='child',
        capabilities=[AskUser(answerer=None)],
    )

    async def parent_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for part in messages[-1].parts:
            if isinstance(part, ToolReturnPart):
                return ModelResponse(parts=[TextPart(content='workflow done')])
        run_workflow = next(tool for tool in info.function_tools if 'workflow' in tool.name)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=run_workflow.name, args={'code': 'r = await child(task="ask")'}, tool_call_id='w1'
                )
            ]
        )

    parent = Agent(
        FunctionModel(parent_model),
        capabilities=[DenyUnanswered(), DynamicWorkflow(agents=[child])],
    )

    result = await parent.run('run it')
    assert result.output == 'workflow done'
    assert any('declined to answer' in content for content in observed)

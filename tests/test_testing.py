import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.testing import assert_tool_call, assert_tool_call_sequence, tool_calls


def test_assert_tool_call_checks_final_response_tool_args() -> None:
    response = ModelResponse(parts=[ToolCallPart('search', {'q': 'agents'}, tool_call_id='call-1')])

    assert tool_calls(response) == response.tool_calls
    assert assert_tool_call(response, 'search', args={'q': 'agents'}) is response.tool_calls[0]
    assert assert_tool_call_sequence(response, ['search']) == response.tool_calls


def test_tool_call_helpers_can_scan_agent_run_history() -> None:
    def model_logic(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('lookup', {'key': 'weather'}, tool_call_id='call-1')])

    agent = Agent(FunctionModel(model_logic))

    @agent.tool_plain
    def lookup(key: str) -> str:
        return f'{key}: sunny'

    result = agent.run_sync('check weather')

    assert result.response.tool_calls == []
    with pytest.raises(AssertionError, match='No tool calls found'):
        assert_tool_call_sequence(result, ['lookup'])
    assert [call.tool_name for call in tool_calls(result, across_history=True)] == ['lookup']
    assert_tool_call(result, 'lookup', args={'key': 'weather'}, across_history=True)


def test_assert_tool_call_reports_available_calls() -> None:
    response = ModelResponse(parts=[ToolCallPart('search', {'q': 'agents'}, tool_call_id='call-1')])

    with pytest.raises(AssertionError, match="Expected tool call 'lookup'"):
        assert_tool_call(response, 'lookup')


def test_assert_tool_call_checks_args() -> None:
    response = ModelResponse(parts=[ToolCallPart('search', {'q': 'agents'}, tool_call_id='call-1')])

    with pytest.raises(AssertionError, match='with args'):
        assert_tool_call(response, 'search', args={'q': 'tools'})

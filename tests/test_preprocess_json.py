"""Tests for the preprocess_json feature."""

import re

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent, ModelResponse, NativeOutput, PromptedOutput, ToolOutput
from pydantic_ai.messages import ModelMessage, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

pytestmark = pytest.mark.anyio


class MyModel(BaseModel):
    name: str
    value: int


def fix_json(json_str: str) -> str:
    """Simple preprocessor that fixes trailing commas and unquoted keys."""
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'(\w+)\s*:', r'"\1":', json_str)
    return json_str


async def fix_json_async(json_str: str) -> str:
    """Async version of the preprocessor."""
    return fix_json(json_str)


def failing_preprocessor(json_str: str) -> str:
    """A preprocessor that always raises an error."""
    raise ValueError('Preprocessing failed!')


# =============================================================================
# ToolOutput Tests
# =============================================================================


async def test_tool_output_preprocess_response_sync():
    """Test that sync preprocess_json works with ToolOutput."""

    def return_malformed_json(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        # For BaseModel types, the JSON should NOT have a "response" wrapper
        malformed_json = '{name: "test", value: 42,}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, malformed_json)])

    agent = Agent(
        FunctionModel(return_malformed_json),
        output_type=ToolOutput(MyModel, preprocess_json=fix_json),
    )

    result = await agent.run('Test')
    assert result.output == snapshot(MyModel(name='test', value=42))


async def test_tool_output_preprocess_response_async():
    """Test that async preprocess_json works with ToolOutput."""

    def return_malformed_json(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        # For BaseModel types, the JSON should NOT have a "response" wrapper
        malformed_json = '{name: "test", value: 123,}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, malformed_json)])

    agent = Agent(
        FunctionModel(return_malformed_json),
        output_type=ToolOutput(MyModel, preprocess_json=fix_json_async),
    )

    result = await agent.run('Test')
    assert result.output == snapshot(MyModel(name='test', value=123))


async def test_tool_output_no_preprocess_with_valid_json():
    """Test that valid JSON works without preprocessing."""

    def return_valid_json(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        # For BaseModel types, the JSON should NOT have a "response" wrapper
        valid_json = '{"name": "test", "value": 999}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, valid_json)])

    agent = Agent(
        FunctionModel(return_valid_json),
        output_type=ToolOutput(MyModel, preprocess_json=fix_json),
    )

    result = await agent.run('Test')
    assert result.output == snapshot(MyModel(name='test', value=999))


# =============================================================================
# NativeOutput Tests
# =============================================================================


async def test_native_output_preprocess_response():
    """Test that preprocess_json works with NativeOutput."""

    def return_malformed_json(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        malformed_json = '{name: "native_test", value: 42,}'
        return ModelResponse(parts=[TextPart(content=malformed_json)])

    agent = Agent(
        FunctionModel(return_malformed_json),
        output_type=NativeOutput(MyModel, preprocess_json=fix_json),
    )

    result = await agent.run('Test')
    assert result.output == snapshot(MyModel(name='native_test', value=42))


# =============================================================================
# PromptedOutput Tests
# =============================================================================


async def test_prompted_output_preprocess_response():
    """Test that preprocess_json works with PromptedOutput."""

    def return_malformed_json(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        malformed_json = '{name: "prompted_test", value: 100,}'
        return ModelResponse(parts=[TextPart(content=malformed_json)])

    agent = Agent(
        FunctionModel(return_malformed_json),
        output_type=PromptedOutput(MyModel, preprocess_json=fix_json),
    )

    result = await agent.run('Test')
    assert result.output == snapshot(MyModel(name='prompted_test', value=100))


# =============================================================================
# Tool Decorator Tests
# =============================================================================


async def test_tool_preprocess_response():
    """Test that preprocess_json works with @agent.tool_plain decorator."""
    call_count = 0

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1 and info.function_tools:
            malformed_args = '{name: "tool_test", value: 42,}'
            return ModelResponse(parts=[ToolCallPart('get_data', malformed_args)])
        return ModelResponse(parts=[TextPart(content='tool_test: 42')])

    agent: Agent[None, str] = Agent(FunctionModel(call_tool))

    @agent.tool_plain(preprocess_json=fix_json)
    def get_data(name: str, value: int) -> str:
        return f'{name}: {value}'

    result = await agent.run('Test')
    assert result.output == snapshot('tool_test: 42')


async def test_tool_preprocess_response_async():
    """Test that async preprocess_json works with @agent.tool_plain decorator."""
    call_count = 0

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1 and info.function_tools:
            malformed_args = '{name: "async_test", value: 99,}'
            return ModelResponse(parts=[ToolCallPart('get_data', malformed_args)])
        return ModelResponse(parts=[TextPart(content='async_test: 99')])

    agent: Agent[None, str] = Agent(FunctionModel(call_tool))

    @agent.tool_plain(preprocess_json=fix_json_async)
    def get_data(name: str, value: int) -> str:
        return f'{name}: {value}'

    result = await agent.run('Test')
    assert result.output == snapshot('async_test: 99')


# =============================================================================
# Error Handling Tests
# =============================================================================


async def test_preprocess_response_error_triggers_retry():
    """Test that preprocessing errors trigger retry with proper error message."""
    from pydantic_ai import UnexpectedModelBehavior

    call_count = 0

    def return_json_with_fallback(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        assert info.output_tools is not None
        # For BaseModel types, the JSON should NOT have a "response" wrapper
        if call_count <= 2:
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": "first", "value": 1}')])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, '{"name": "retry", "value": 1}')])

    agent = Agent(
        FunctionModel(return_json_with_fallback),
        output_type=ToolOutput(MyModel, preprocess_json=failing_preprocessor),
    )

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await agent.run('Test')

    assert 'Exceeded maximum retries' in str(exc_info.value)


async def test_tool_preprocess_error_triggers_retry():
    """Test that tool preprocessing errors trigger retry."""
    from pydantic_ai import UnexpectedModelBehavior

    call_count = 0

    def call_tool(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count <= 2 and info.function_tools:
            return ModelResponse(parts=[ToolCallPart('get_data', '{"name": "test"}')])
        return ModelResponse(parts=[TextPart(content='done')])

    agent: Agent[None, str] = Agent(FunctionModel(call_tool))

    @agent.tool_plain(preprocess_json=failing_preprocessor)
    def get_data(name: str) -> str:
        return name

    with pytest.raises(UnexpectedModelBehavior) as exc_info:
        await agent.run('Test')

    assert 'exceeded max retries' in str(exc_info.value)


# =============================================================================
# Dict Data Bypass Tests
# =============================================================================


async def test_dict_data_bypasses_preprocessing():
    """Test that dict data (not string) bypasses preprocessing."""

    def return_dict_args(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        # For BaseModel types, the dict should NOT have a "response" wrapper
        dict_args = {'name': 'dict_test', 'value': 777}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, dict_args)])

    def should_not_be_called(json_str: str) -> str:
        raise AssertionError('Preprocessor should not be called for dict data')

    agent = Agent(
        FunctionModel(return_dict_args),
        output_type=ToolOutput(MyModel, preprocess_json=should_not_be_called),
    )

    result = await agent.run('Test')
    assert result.output == snapshot(MyModel(name='dict_test', value=777))


import pytest
from typing import Annotated
from pydantic import AfterValidator
from pydantic_core import ValidationError
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.exceptions import ToolOutputValidationError, ModelRetry
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart

async def test_schema_validation_failure():
    def my_tool(ctx: RunContext[None], x: int) -> int:
        return "not an int"

    agent = Agent(FunctionModel(lambda ms, info: ModelResponse(parts=[ToolCallPart("my_tool", {"x": 1}, "1")])), tools=[Tool(my_tool, validate_return=True)])

    with pytest.raises(ToolOutputValidationError) as exc_info:
        await agent.run("call my_tool")
    
    assert exc_info.value.tool_name == "my_tool"
    assert "integer" in str(exc_info.value.validation_error)

async def test_schema_validation_success():
    def my_tool(ctx: RunContext[None], x: int) -> int:
        return x * 2

    called = False
    async def call_tool_model(messages, info):
        nonlocal called
        if not called:
            called = True
            return ModelResponse(parts=[ToolCallPart("my_tool", {"x": 21}, "1")])
        else:
            return ModelResponse(parts=[TextPart("Done")])

    agent = Agent(FunctionModel(call_tool_model), tools=[Tool(my_tool, validate_return=True)])
    result = await agent.run("call my_tool")
    
    found_tool_return = False
    for msg in result.all_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                    assert part.content == 42
                    found_tool_return = True
    assert found_tool_return

def sanitize_output(v: str) -> str:
    return v.strip().upper()

async def test_custom_validator_sanitization():
    def my_tool(ctx: RunContext[None]) -> Annotated[str, AfterValidator(sanitize_output)]:
        return "  hello  "

    called = False
    async def call_tool_model(messages, info):
        nonlocal called
        if not called:
            called = True
            return ModelResponse(parts=[ToolCallPart("my_tool", {}, "1")])
        else:
            return ModelResponse(parts=[TextPart("Done")])

    agent = Agent(FunctionModel(call_tool_model), tools=[Tool(my_tool, validate_return=True)])
    result = await agent.run("call my_tool")
    
    for msg in result.all_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                    assert part.content == "HELLO"

async def test_async_custom_validator():
    async def async_validator(ctx: RunContext[None], v: int) -> int:
        return v + 1

    def my_tool(ctx: RunContext[None], x: int) -> int:
        return x

    called = False
    async def call_tool_model(messages, info):
        nonlocal called
        if not called:
            called = True
            return ModelResponse(parts=[ToolCallPart("my_tool", {"x": 41}, "1")])
        else:
            return ModelResponse(parts=[TextPart("Done")])

    tool = Tool(my_tool, validate_return=True, result_validator=async_validator)
    agent = Agent(FunctionModel(call_tool_model), tools=[tool])
    result = await agent.run("call my_tool")
    
    for msg in result.all_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'part_kind') and part.part_kind == 'tool-return':
                    assert part.content == 42

async def test_model_retry_from_validator():
    def retry_validator(ctx: RunContext[None], v: int) -> int:
        if v < 100:
            raise ModelRetry("Value too low, try higher")
        return v

    def my_tool(ctx: RunContext[None], x: int) -> int:
        return x

    responses = [
        ModelResponse(parts=[ToolCallPart("my_tool", {"x": 50}, "1")]),
        ModelResponse(parts=[ToolCallPart("my_tool", {"x": 150}, "2")]),
        ModelResponse(parts=[TextPart("Done")])
    ]
    response_idx = 0
    async def call_tool_model(messages, info):
        nonlocal response_idx
        res = responses[response_idx]
        response_idx += 1
        return res

    tool = Tool(my_tool, validate_return=True, result_validator=retry_validator)
    agent = Agent(FunctionModel(call_tool_model), tools=[tool])
    result = await agent.run("call my_tool")
    
    assert response_idx == 3
    # Check that the first tool return was a retry prompt
    retry_found = False
    for msg in result.all_messages():
        if hasattr(msg, 'parts'):
            for part in msg.parts:
                if hasattr(part, 'part_kind') and part.part_kind == 'retry-prompt':
                    assert "Value too low" in part.content
                    retry_found = True
    assert retry_found

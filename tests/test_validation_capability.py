
import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import EnforceToolReturnValidation
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
from pydantic_ai.exceptions import ToolOutputValidationError

async def test_enforce_validation_capability():
    """Verify that the capability correctly enables validation for all tools."""
    def my_tool(ctx: RunContext[None], x: int) -> int:
        return "not an int"

    called = False
    async def call_tool_model(messages, info):
        nonlocal called
        if not called:
            called = True
            return ModelResponse(parts=[ToolCallPart("my_tool", {"x": 1}, "1")])
        else:
            return ModelResponse(parts=[TextPart("Done")])

    agent = Agent(
        FunctionModel(call_tool_model), 
        tools=[my_tool], 
        capabilities=[EnforceToolReturnValidation()]
    )

    # Correct pytest usage: raising the error here is the EXPECTED behavior
    with pytest.raises(ToolOutputValidationError) as exc_info:
        await agent.run("call my_tool")
    
    assert exc_info.value.tool_name == "my_tool"
    assert "integer" in str(exc_info.value.validation_error)

async def test_enforce_validation_capability_selective():
    """Verify that the capability can selectively enable validation."""
    def tool_a(ctx: RunContext[None]) -> int:
        return "not an int"
    
    def tool_b(ctx: RunContext[None]) -> int:
        return "not an int"

    called = 0
    async def call_tool_model(messages, info):
        nonlocal called
        if called == 0:
            called += 1
            return ModelResponse(parts=[ToolCallPart("tool_a", {}, "1")])
        elif called == 1:
            called += 1
            return ModelResponse(parts=[ToolCallPart("tool_b", {}, "2")])
        else:
            return ModelResponse(parts=[TextPart("Done")])

    # Only validate tool_a
    agent = Agent(
        FunctionModel(call_tool_model), 
        tools=[tool_a, tool_b], 
        capabilities=[EnforceToolReturnValidation(tools=['tool_a'])]
    )

    # Calling tool_a should fail
    with pytest.raises(ToolOutputValidationError):
        await agent.run("call tool_a")
    
    # Reset and call tool_b (should succeed because it's NOT being validated)
    called = 1
    result = await agent.run("call tool_b")
    assert "not an int" in str(result.all_messages())

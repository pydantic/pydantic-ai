#!/usr/bin/env python3
"""Test script to verify that FunctionModel now supports NativeOutput by default."""

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.output import NativeOutput


class TestOutput(BaseModel):
    message: str
    value: int


def simple_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart(content='{"message": "Hello World", "value": 42}')])


def main():
    # Create a FunctionModel without specifying a profile
    # This should now work with NativeOutput due to the default profile
    model = FunctionModel(simple_function)

    # Create an agent with NativeOutput
    agent = Agent(model, output_type=NativeOutput(TestOutput))

    # This should work without errors now
    result = agent.run_sync("Test")

    print(f"Success! Result: {result.output}")
    print(f"Result type: {type(result.output)}")
    assert isinstance(result.output, TestOutput)
    assert result.output.message == "Hello World"
    assert result.output.value == 42

    print("✅ FunctionModel now supports NativeOutput by default!")


if __name__ == "__main__":
    main()

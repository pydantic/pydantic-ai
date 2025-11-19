from dataclasses import dataclass

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, ValidationInfo, field_validator

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelResponse,
    NativeOutput,
    PromptedOutput,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolOutput,
)
from pydantic_ai._output import OutputSpec
from pydantic_ai.models.function import AgentInfo, FunctionModel


class Value(BaseModel):
    x: int

    @field_validator('x')
    def increment_value(cls, value: int, info: ValidationInfo):
        return value + (info.context or 0)


@dataclass
class Deps:
    increment: int


@pytest.mark.parametrize(
    'output_type',
    [
        Value,
        ToolOutput(Value),
        NativeOutput(Value),
        PromptedOutput(Value),
    ],
    ids=[
        'Value',
        'ToolOutput(Value)',
        'NativeOutput(Value)',
        'PromptedOutput(Value)',
    ],
)
def test_agent_output_with_validation_context(output_type: OutputSpec[Value]):
    """Test that the output is validated using the validation context"""

    def mock_llm(_: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        if isinstance(output_type, ToolOutput):
            return ModelResponse(parts=[ToolCallPart(tool_name='final_result', args={'x': 0})])
        else:
            text = Value(x=0).model_dump_json()
            return ModelResponse(parts=[TextPart(content=text)])

    agent = Agent(
        FunctionModel(mock_llm),
        output_type=output_type,
        deps_type=Deps,
        validation_context=lambda ctx: ctx.deps.increment,
    )

    result = agent.run_sync('', deps=Deps(increment=10))
    assert result.output.x == snapshot(10)


def test_agent_tool_call_with_validation_context():
    """Test that the argument passed to the tool call is validated using the validation context."""

    agent = Agent(
        'test',
        deps_type=Deps,
        validation_context=lambda ctx: ctx.deps.increment,
    )

    @agent.tool
    def get_value(ctx: RunContext[Deps], v: Value) -> int:
        # NOTE: The test agent calls this tool with Value(x=0) which should then have been influenced by the validation context through the `increment_value` field validator
        assert v.x == ctx.deps.increment
        return v.x

    result = agent.run_sync('', deps=Deps(increment=10))
    assert result.output == snapshot('{"get_value":10}')


def test_agent_output_function_with_validation_context():
    """Test that the argument passed to the output function is validated using the validation context."""

    def get_value(v: Value) -> int:
        return v.x

    agent = Agent(
        'test',
        output_type=get_value,
        deps_type=Deps,
        validation_context=lambda ctx: ctx.deps.increment,
    )

    result = agent.run_sync('', deps=Deps(increment=10))
    assert result.output == snapshot(10)


def test_agent_output_validator_with_validation_context():
    """Test that the argument passed to the output validator is validated using the validation context."""

    agent = Agent(
        'test',
        output_type=Value,
        deps_type=Deps,
        validation_context=lambda ctx: ctx.deps.increment,
    )

    @agent.output_validator
    def identity(ctx: RunContext[Deps], v: Value) -> Value:
        return v

    result = agent.run_sync('', deps=Deps(increment=10))
    assert result.output.x == snapshot(10)

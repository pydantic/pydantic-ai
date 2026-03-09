from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ValidationInfo, field_validator

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeOutput,
    PromptedOutput,
    RunContext,
    TextPart,
    ToolCallPart,
    ToolOutput,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai._output import OutputSpec
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr


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
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(tool_name='get_value', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__get_value')
                ],
                usage=RequestUsage(input_tokens=50, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='get_value',
                        content=10,
                        tool_call_id='pyd_ai_tool_call_id__get_value',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"get_value":10}')],
                usage=RequestUsage(input_tokens=51, output_tokens=7),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


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
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__final_result'
                    )
                ],
                usage=RequestUsage(input_tokens=50, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


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
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__final_result'
                    )
                ],
                usage=RequestUsage(input_tokens=50, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )


def test_agent_output_validator_with_intermediary_deps_change_and_validation_context():
    """Test that the validation context is updated as run dependencies are mutated."""

    agent = Agent(
        'test',
        output_type=Value,
        deps_type=Deps,
        validation_context=lambda ctx: ctx.deps.increment,
    )

    @agent.tool
    def bump_increment(ctx: RunContext[Deps]):
        assert ctx.validation_context == snapshot(10)  # validation ctx was first computed using the original deps
        ctx.deps.increment += 5  # update the deps

    @agent.output_validator
    def identity(ctx: RunContext[Deps], v: Value) -> Value:
        assert ctx.validation_context == snapshot(15)  # validation ctx was re-computed after deps update from tool call

        return v

    result = agent.run_sync('', deps=Deps(increment=10))
    assert result.output.x == snapshot(15)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='bump_increment', args={}, tool_call_id='pyd_ai_tool_call_id__bump_increment'
                    )
                ],
                usage=RequestUsage(input_tokens=50, output_tokens=2),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='bump_increment',
                        content=None,
                        tool_call_id='pyd_ai_tool_call_id__bump_increment',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='final_result', args={'x': 0}, tool_call_id='pyd_ai_tool_call_id__final_result'
                    )
                ],
                usage=RequestUsage(input_tokens=51, output_tokens=6),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='final_result',
                        content='Final result processed.',
                        tool_call_id='pyd_ai_tool_call_id__final_result',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )

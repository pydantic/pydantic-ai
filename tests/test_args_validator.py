"""Tests for the args_validator parameter on tools."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.exceptions import ApprovalRequired, ModelRetry
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    RetryPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.function import FunctionToolset


class _ExhaustiveTestValidOutput(BaseModel):
    value: str


class _ExhaustiveTestInvalidOutput(BaseModel):
    value: str
    count: int


def test_args_validator_success():
    """Test that args_validator runs before FunctionToolCallEvent and sets args_valid=True."""
    validator_called = False

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_called
        validator_called = True

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    assert validator_called


@pytest.mark.anyio
async def test_args_validator_failure():
    """Test that failed validation emits args_valid=False, retries with error message, then succeeds."""
    validator_calls = 0

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_calls
        validator_calls += 1
        if validator_calls == 1:
            raise ModelRetry('Validation failed: x must be positive')

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator, retries=2)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    events: list[Any] = []
    async for event in agent.run_stream_events('call add_numbers with x=1 and y=2', deps=42):
        events.append(event)

    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent) and e.part.tool_name == 'add_numbers']

    # First call: validation fails → args_valid=False; retry: validation passes → args_valid=True
    args_valid_values = [e.args_valid for e in tool_call_events]
    assert args_valid_values[0] is False
    assert args_valid_values[1] is True

    # Verify retry prompt contains the validation error message
    result_events = [e for e in events if isinstance(e, FunctionToolResultEvent)]
    retry_results = [e for e in result_events if isinstance(e.result, RetryPromptPart)]
    assert retry_results
    assert 'x must be positive' in str(retry_results[0].result.content)


def test_args_validator_not_configured():
    """Test that tools work without a custom args_validator."""
    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)


@pytest.mark.anyio
async def test_args_validator_async():
    """Test async validator functions work correctly."""
    validator_called = False

    async def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_called
        validator_called = True

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator)
    async def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    await agent.run('call add_numbers with x=1 and y=2', deps=42)

    assert validator_called


def test_args_validator_with_deps():
    """Test that validator uses RunContext.deps."""
    deps_value = None

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal deps_value
        deps_value = ctx.deps

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    assert deps_value == 42


def test_args_validator_tool_direct():
    """Test via Tool() direct instantiation."""
    validator_called = False

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_called
        validator_called = True

    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    tool = Tool(add_numbers, args_validator=my_validator)

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
        tools=[tool],
    )

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    assert validator_called


def test_args_validator_toolset():
    """Test via FunctionToolset."""
    validator_called = False

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_called
        validator_called = True

    toolset = FunctionToolset[int]()

    @toolset.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
        toolsets=[toolset],
    )

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    assert validator_called


@pytest.mark.anyio
async def test_args_validator_event_args_valid_field():
    """Test that FunctionToolCallEvent has args_valid field set correctly."""

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        pass  # Always succeeds

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    events: list[Any] = []
    async for event in agent.run_stream_events('call add_numbers with x=1 and y=2', deps=42):
        events.append(event)

    # Find FunctionToolCallEvent
    tool_call_events: list[FunctionToolCallEvent] = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    assert len(tool_call_events) >= 1

    # Check args_valid is True for the tool with validator
    add_number_events = [e for e in tool_call_events if e.part.tool_name == 'add_numbers']
    assert add_number_events, 'Should have events for add_numbers'
    for event in add_number_events:
        assert event.args_valid is True


@pytest.mark.anyio
async def test_args_validator_event_args_valid_no_custom_validator():
    """Test that args_valid=True when no custom validator but schema validation passes."""
    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    events: list[Any] = []
    async for event in agent.run_stream_events('call add_numbers with x=1 and y=2', deps=42):
        events.append(event)

    # Find FunctionToolCallEvent
    tool_call_events: list[FunctionToolCallEvent] = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    assert len(tool_call_events) >= 1

    # args_valid should be True when schema validation passes (no custom validator)
    add_number_events = [e for e in tool_call_events if e.part.tool_name == 'add_numbers']
    assert add_number_events, 'Should have events for add_numbers'
    for event in add_number_events:
        assert event.args_valid is True


def test_args_validator_tool_plain():
    """Test args_validator with tool_plain decorator."""
    validator_called = False

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_called
        validator_called = True

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool_plain(args_validator=my_validator)
    def add_numbers(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    assert validator_called


@pytest.mark.anyio
async def test_schema_validation_failure_args_valid_false():
    """Test that args_valid=False when Pydantic schema validation fails (no custom validator)."""
    from collections.abc import AsyncIterator

    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

    def return_invalid_args(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # pragma: no cover
        """Return a tool call with invalid arguments (wrong type)."""
        return ModelResponse(parts=[ToolCallPart(tool_name='add_numbers', args={'x': 'not_an_int', 'y': 2})])

    async def stream_invalid_args(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[DeltaToolCalls]:
        """Stream a tool call with invalid arguments."""
        yield {0: DeltaToolCall(name='add_numbers')}
        yield {0: DeltaToolCall(json_args='{"x": "not_an_int", "y": 2}')}

    agent = Agent(FunctionModel(return_invalid_args, stream_function=stream_invalid_args), deps_type=int)

    @agent.tool
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:  # pragma: no cover
        """Add two numbers."""
        return x + y

    events: list[Any] = []
    # The model always returns invalid args, so eventually max retries will be exceeded
    # We collect events until that happens
    try:
        async for event in agent.run_stream_events('call add_numbers', deps=42):  # pragma: no branch
            events.append(event)
    except UnexpectedModelBehavior:
        pass  # Expected when max retries exceeded

    # Find FunctionToolCallEvent
    tool_call_events: list[FunctionToolCallEvent] = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    assert len(tool_call_events) >= 1

    # The first event should have args_valid=False due to schema validation failure
    first_event = tool_call_events[0]
    assert first_event.part.tool_name == 'add_numbers'
    assert first_event.args_valid is False


def test_args_validator_max_retries_exceeded():
    """Test that UnexpectedModelBehavior is raised when validator always fails and max retries is exceeded."""
    from pydantic_ai.exceptions import UnexpectedModelBehavior

    def always_fail_validator(ctx: RunContext[int], x: int, y: int) -> None:
        raise ModelRetry('Always fails')

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=always_fail_validator, retries=2)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:  # pragma: no cover
        """Add two numbers."""
        return x + y

    with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries'):
        agent.run_sync('call add_numbers with x=1 and y=2', deps=42)


def test_args_validator_tool_from_schema():
    """Test Tool.from_schema() with args_validator parameter."""
    validator_called = False

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_called
        validator_called = True

    def add_numbers(ctx: RunContext[int], **kwargs: Any) -> int:
        """Add two numbers."""
        return kwargs['x'] + kwargs['y']

    json_schema = {
        'type': 'object',
        'properties': {
            'x': {'type': 'integer'},
            'y': {'type': 'integer'},
        },
        'required': ['x', 'y'],
    }

    tool = Tool.from_schema(
        add_numbers,
        name='add_numbers',
        description='Add two numbers',
        json_schema=json_schema,
        takes_ctx=True,
        args_validator=my_validator,
    )

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
        tools=[tool],
    )

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    assert validator_called


def test_args_validator_with_prepare():
    """Test that args_validator works together with prepare function."""
    from pydantic_ai.tools import ToolDefinition

    validator_called = False
    prepare_called = False

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_called
        validator_called = True

    async def my_prepare(ctx: RunContext[int], tool_def: ToolDefinition) -> ToolDefinition:
        nonlocal prepare_called
        prepare_called = True
        return tool_def

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator, prepare=my_prepare)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    assert prepare_called
    assert validator_called


def test_args_validator_multiple_tools():
    """Test that multiple tools can have different validators that work independently."""
    add_validator_calls = 0
    multiply_validator_calls = 0

    def add_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal add_validator_calls
        add_validator_calls += 1

    def multiply_validator(ctx: RunContext[int], a: int, b: int) -> None:
        nonlocal multiply_validator_calls
        multiply_validator_calls += 1

    agent = Agent(
        TestModel(call_tools=['add_numbers', 'multiply_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=add_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    @agent.tool(args_validator=multiply_validator)
    def multiply_numbers(ctx: RunContext[int], a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    agent.run_sync('call both tools', deps=42)

    # Both validators should have been called at least once
    # (TestModel may call tools multiple times depending on agent flow)
    assert add_validator_calls >= 1
    assert multiply_validator_calls >= 1
    # Ensure correct validator was called for correct tool (no cross-calling)
    # This is implicitly verified by no exceptions being raised


def test_args_validator_context_tool_name():
    """Test that validator can access tool_name from RunContext."""
    captured_tool_name = None

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal captured_tool_name
        captured_tool_name = ctx.tool_name

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    assert captured_tool_name == 'add_numbers'


def test_args_validator_context_retry():
    """Test that validator can access retry count from RunContext."""
    retry_values: list[int] = []

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        retry_values.append(ctx.retry)
        if len(retry_values) == 1:
            raise ModelRetry('First attempt fails')

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator, retries=2)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    # Validator is called once per tool call attempt (not doubled by re-validation during execution).
    # First attempt: retry=0, raises ModelRetry → validation fails → retry prompt sent to model
    # Second attempt: retry=1, succeeds → tool executes
    assert retry_values == [0, 1]


def test_exhaustive_strategy_second_output_schema_validation_fails():
    """Test exhaustive strategy when first output succeeds and second fails schema validation."""
    from pydantic_ai._output import ToolOutput
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    output_tools_called: list[str] = []

    def process_first(output: _ExhaustiveTestValidOutput) -> _ExhaustiveTestValidOutput:
        output_tools_called.append('first')
        return output

    def process_second(output: _ExhaustiveTestInvalidOutput) -> _ExhaustiveTestInvalidOutput:  # pragma: no cover
        output_tools_called.append('second')
        return output

    def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(
            parts=[
                # First output: valid schema
                ToolCallPart('first_output', {'value': 'valid'}),
                # Second output: invalid schema (count should be int, not string)
                ToolCallPart('second_output', {'value': 'invalid', 'count': 'not_an_int'}),
            ],
        )

    agent = Agent(
        FunctionModel(return_model),
        output_type=[
            ToolOutput(process_first, name='first_output'),
            ToolOutput(process_second, name='second_output'),
        ],
        end_strategy='exhaustive',
    )

    result = agent.run_sync('test exhaustive with schema validation failure')

    # Verify the result came from the first output tool
    assert isinstance(result.output, _ExhaustiveTestValidOutput)
    assert result.output.value == 'valid'

    # Only the first output tool should have been called
    # (second failed schema validation before its processor could run)
    assert output_tools_called == ['first']


def test_exhaustive_strategy_second_output_max_retries_exceeded():
    """Test exhaustive strategy when first output succeeds and second exceeds max retries."""
    from pydantic_ai._output import ToolOutput
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    output_tools_called: list[str] = []

    def process_first(output: _ExhaustiveTestValidOutput) -> _ExhaustiveTestValidOutput:
        output_tools_called.append('first')
        return output

    def process_second(output: _ExhaustiveTestInvalidOutput) -> _ExhaustiveTestInvalidOutput:  # pragma: no cover
        output_tools_called.append('second')
        return output

    def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(
            parts=[
                # First output: valid schema
                ToolCallPart('first_output', {'value': 'valid'}),
                # Second output: invalid schema (count should be int, not string)
                # Will fail schema validation and with output_retries=0, will exceed max retries
                ToolCallPart('second_output', {'value': 'invalid', 'count': 'not_an_int'}),
            ],
        )

    agent = Agent(
        FunctionModel(return_model),
        output_type=[
            ToolOutput(process_first, name='first_output'),
            ToolOutput(process_second, name='second_output'),
        ],
        end_strategy='exhaustive',
        output_retries=0,  # Immediately exceed max retries on first validation failure
    )

    result = agent.run_sync('test exhaustive with max retries exceeded')

    # Verify the result came from the first output tool
    assert isinstance(result.output, _ExhaustiveTestValidOutput)
    assert result.output.value == 'valid'

    # Only the first output tool should have been called
    assert output_tools_called == ['first']


@pytest.mark.anyio
async def test_external_tool_validation_failure():
    """Test that external tool validation failure is handled correctly."""
    from collections.abc import AsyncIterator
    from dataclasses import replace as dataclass_replace

    from pydantic_core import SchemaValidator, core_schema

    from pydantic_ai._output import DeferredToolRequests
    from pydantic_ai._run_context import RunContext as RC
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
    from pydantic_ai.tools import ToolDefinition
    from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool

    # Create a schema validator that requires 'value' field of type string
    strict_schema = core_schema.typed_dict_schema(
        {
            'value': core_schema.typed_dict_field(core_schema.str_schema(), required=True),
        },
    )
    STRICT_VALIDATOR = SchemaValidator(schema=strict_schema)

    class StrictExternalToolset(AbstractToolset[None]):
        """A custom external toolset with strict validation."""

        tool_defs: list[ToolDefinition]

        def __init__(self, tool_defs: list[ToolDefinition]):
            self.tool_defs = tool_defs

        @property
        def id(self) -> str | None:  # pragma: no cover
            return 'strict_external'

        async def get_tools(self, ctx: RC[None]) -> dict[str, ToolsetTool[None]]:
            return {
                tool_def.name: ToolsetTool(
                    toolset=self,
                    tool_def=dataclass_replace(tool_def, kind='external'),
                    max_retries=1,  # Allow 1 retry so validation failure doesn't raise immediately
                    args_validator=STRICT_VALIDATOR,
                )
                for tool_def in self.tool_defs
            }

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RC[None], tool: ToolsetTool[None]
        ) -> Any:  # pragma: no cover
            raise NotImplementedError('External tools cannot be called directly')

    async def stream_function(_: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        # Call external tool with invalid arguments (missing required 'value' field)
        yield {0: DeltaToolCall(name='external_tool', json_args='{"wrong_field": 123}')}

    # Create an external tool definition
    external_tool_def = ToolDefinition(
        name='external_tool',
        parameters_json_schema={
            'type': 'object',
            'properties': {'value': {'type': 'string'}},
            'required': ['value'],
        },
    )

    agent: Agent[None, str | DeferredToolRequests] = Agent(
        FunctionModel(stream_function=stream_function),
        output_type=[str, DeferredToolRequests],
        toolsets=[StrictExternalToolset(tool_defs=[external_tool_def])],
    )

    events: list[Any] = []
    # The model returns invalid args, which should trigger validation failure
    try:
        async for event in agent.run_stream_events('test external tool validation'):  # pragma: no branch
            events.append(event)
    except Exception:
        pass  # May fail due to validation errors

    # Find FunctionToolCallEvent
    tool_call_events: list[FunctionToolCallEvent] = [e for e in events if isinstance(e, FunctionToolCallEvent)]

    # Should have at least one tool call event
    assert tool_call_events, 'Should have at least one tool call event'
    # The event should have args_valid=False due to schema validation failure
    external_events = [e for e in tool_call_events if e.part.tool_name == 'external_tool']
    assert external_events, 'Should have external tool events'
    assert external_events[0].args_valid is False


def test_args_validator_not_double_called_for_approved_tools():
    """Test that args_validator is called exactly once for ToolApproved deferred tool calls.

    When a tool requires approval and is later re-run with ToolApproved, the validator
    should only be called once (during handle_call with approved=True), not twice
    (once in upfront validation with approved=False, then again in handle_call).
    """
    from pydantic_ai._output import DeferredToolRequests
    from pydantic_ai.tools import DeferredToolResults, ToolApproved

    validator_calls: list[tuple[int, bool]] = []

    def my_validator(ctx: RunContext[int], x: int) -> None:
        validator_calls.append((ctx.retry, ctx.tool_call_approved))

    agent = Agent(
        TestModel(),
        deps_type=int,
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext[int], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return x * 42

    # First run: tool requires approval, gets deferred
    result = agent.run_sync('Hello', deps=42)
    assert isinstance(result.output, DeferredToolRequests)
    assert len(result.output.approvals) == 1
    tool_call_id = result.output.approvals[0].tool_call_id

    # Validator should have been called once during the first run (for the initial tool call)
    assert len(validator_calls) == 1
    assert validator_calls[0] == (0, False)  # retry=0, approved=False

    # Second run: re-run with ToolApproved
    validator_calls.clear()
    messages = result.all_messages()
    result = agent.run_sync(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={tool_call_id: ToolApproved()}),
        deps=42,
    )

    # Validator should have been called exactly once with approved=True
    assert len(validator_calls) == 1
    assert validator_calls[0] == (0, True)  # retry=0, approved=True


def test_early_strategy_second_output_max_retries_exceeded():
    """Test early strategy when first output succeeds and second exceeds max retries."""
    from pydantic_ai._output import ToolOutput
    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    def process_first(output: _ExhaustiveTestValidOutput) -> _ExhaustiveTestValidOutput:
        return output

    def process_second(output: _ExhaustiveTestInvalidOutput) -> _ExhaustiveTestInvalidOutput:  # pragma: no cover
        return output

    def return_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        return ModelResponse(
            parts=[
                # First output: valid schema
                ToolCallPart('first_output', {'value': 'valid'}),
                # Second output: invalid schema (count should be int, not string)
                # Will fail schema validation and with output_retries=0, will exceed max retries
                ToolCallPart('second_output', {'value': 'invalid', 'count': 'not_an_int'}),
            ],
        )

    agent = Agent(
        FunctionModel(return_model),
        output_type=[
            ToolOutput(process_first, name='first_output'),
            ToolOutput(process_second, name='second_output'),
        ],
        end_strategy='early',
        output_retries=0,  # Immediately exceed max retries on first validation failure
    )

    result = agent.run_sync('test early with max retries exceeded')

    # Verify the result came from the first output tool
    assert isinstance(result.output, _ExhaustiveTestValidOutput)
    assert result.output.value == 'valid'


@pytest.mark.anyio
async def test_args_validator_run_stream_event_handler():
    """Test that args_valid is correctly set on FunctionToolCallEvent when using run_stream()."""
    from collections.abc import AsyncIterable

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        pass  # Always succeeds

    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    events: list[AgentStreamEvent] = []

    async def handler(ctx: RunContext[int], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    async with agent.run_stream('call add_numbers', deps=42, event_stream_handler=handler) as result:
        await result.get_output()

    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent)]
    assert tool_call_events
    for event in tool_call_events:
        assert event.args_valid is True


@pytest.mark.anyio
async def test_event_ordering_call_before_result():
    """Test that FunctionToolCallEvent is emitted before FunctionToolResultEvent for each tool call."""

    def my_validator(ctx: RunContext[None], x: int) -> None:
        pass

    agent = Agent(TestModel(call_tools=['my_tool']))

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext[None], x: int) -> int:
        """A tool."""
        return x * 2

    events: list[Any] = []
    async for event in agent.run_stream_events('test'):
        events.append(event)

    # For each tool call ID, verify that FunctionToolCallEvent appears before FunctionToolResultEvent
    call_ids_seen: set[str] = set()
    result_ids_seen: set[str] = set()
    for event in events:
        if isinstance(event, FunctionToolCallEvent):
            call_ids_seen.add(event.tool_call_id)
            # Should not have seen a result for this ID yet
            assert event.tool_call_id not in result_ids_seen, (
                f'FunctionToolResultEvent for {event.tool_call_id} appeared before FunctionToolCallEvent'
            )
        elif isinstance(event, FunctionToolResultEvent):
            result_id = event.result.tool_call_id
            result_ids_seen.add(result_id)
            # Should have already seen a call event for this ID
            assert result_id in call_ids_seen, (
                f'FunctionToolResultEvent for {result_id} appeared without prior FunctionToolCallEvent'
            )

    # Ensure we actually saw both event types
    assert call_ids_seen
    assert result_ids_seen


@pytest.mark.anyio
async def test_args_valid_none_for_presupplied_tool_approved():
    """Test that args_valid=None when re-running with ToolApproved (validation deferred to execution)."""
    from pydantic_ai._output import DeferredToolRequests
    from pydantic_ai.tools import DeferredToolResults, ToolApproved

    def my_validator(ctx: RunContext[int], x: int) -> None:
        pass

    agent = Agent(
        TestModel(),
        deps_type=int,
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext[int], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return x * 42

    # First run: tool requires approval
    result = await agent.run('Hello', deps=42)
    assert isinstance(result.output, DeferredToolRequests)
    tool_call_id = result.output.approvals[0].tool_call_id

    # Second run with ToolApproved: collect events
    messages = result.all_messages()
    events: list[Any] = []
    async for event in agent.run_stream_events(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={tool_call_id: ToolApproved()}),
        deps=42,
    ):
        events.append(event)

    # The FunctionToolCallEvent for the pre-supplied result should have args_valid=None
    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent) and e.part.tool_name == 'my_tool']
    assert tool_call_events
    assert tool_call_events[0].args_valid is None


@pytest.mark.anyio
async def test_args_valid_none_for_tool_denied():
    """Test that args_valid=None for ToolDenied and the denial message appears in the result event."""
    from pydantic_ai._output import DeferredToolRequests
    from pydantic_ai.tools import DeferredToolResults, ToolDenied

    def my_validator(ctx: RunContext[int], x: int) -> None:
        pass

    agent = Agent(
        TestModel(),
        deps_type=int,
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext[int], x: int) -> int:
        if not ctx.tool_call_approved:
            raise ApprovalRequired()
        return x  # pragma: no cover

    # First run: tool requires approval
    result = await agent.run('Hello', deps=42)
    assert isinstance(result.output, DeferredToolRequests)
    tool_call_id = result.output.approvals[0].tool_call_id

    # Second run with ToolDenied
    messages = result.all_messages()
    events: list[Any] = []
    async for event in agent.run_stream_events(
        message_history=messages,
        deferred_tool_results=DeferredToolResults(approvals={tool_call_id: ToolDenied('User denied this tool call')}),
        deps=42,
    ):
        events.append(event)

    # FunctionToolCallEvent should have args_valid=None (pre-supplied result, no upfront validation)
    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent) and e.part.tool_name == 'my_tool']
    assert tool_call_events
    assert tool_call_events[0].args_valid is None

    # FunctionToolResultEvent should contain the denial message
    result_events = [e for e in events if isinstance(e, FunctionToolResultEvent) and e.result.tool_name == 'my_tool']
    assert result_events
    assert result_events[0].result.content == 'User denied this tool call'


@pytest.mark.anyio
async def test_deferred_tool_validation_event_in_stream():
    """Test that deferred (requires_approval) tools emit FunctionToolCallEvent with correct args_valid."""
    from pydantic_ai._output import DeferredToolRequests

    def my_validator(ctx: RunContext[None], x: int) -> None:
        pass

    agent = Agent(
        TestModel(),
        output_type=[str, DeferredToolRequests],
    )

    @agent.tool(args_validator=my_validator)
    def my_tool(ctx: RunContext[None], x: int) -> int:
        raise ApprovalRequired()

    # Collect events during the first run (where tool will be deferred)
    events: list[Any] = []
    async for event in agent.run_stream_events('test'):
        events.append(event)

    # The tool call should have been validated before deferral
    tool_call_events = [e for e in events if isinstance(e, FunctionToolCallEvent) and e.part.tool_name == 'my_tool']
    assert tool_call_events
    # TestModel generates valid args (x=0 by default), so validation passes
    assert tool_call_events[0].args_valid is True

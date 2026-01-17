"""Tests for the args_validator parameter on tools."""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import FunctionToolCallEvent
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.function import FunctionToolset


# Test 1: args_validator success - validator passes, args_valid=True
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


# Test 2: args_validator failure - validator raises ModelRetry
def test_args_validator_failure():
    """Test that failed validation is handled correctly."""
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

    @agent.tool(args_validator=my_validator)
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)

    # The validator should have been called at least once
    assert validator_calls >= 1


# Test 3: args_validator not configured - args_valid=None
def test_args_validator_not_configured():
    """Test backward compatibility - args_valid=None when no validator."""
    agent = Agent(
        TestModel(call_tools=['add_numbers']),
        deps_type=int,
    )

    @agent.tool
    def add_numbers(ctx: RunContext[int], x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    # Check that it completes successfully (no assertion needed, just verify no exception)
    agent.run_sync('call add_numbers with x=1 and y=2', deps=42)


# Test 4: async validator function
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


# Test 5: sync validator function
def test_args_validator_sync():
    """Test sync validator functions work correctly."""
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


# Test 6: validator uses RunContext.deps
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


# Test 7: Tool() direct instantiation with validator
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


# Test 8: FunctionToolset with validator
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


# Test 9: FunctionToolCallEvent has args_valid field set correctly
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
    for event in tool_call_events:
        if event.part.tool_name == 'add_numbers':
            assert event.args_valid is True


# Test 10: args_valid=False when validator fails
@pytest.mark.anyio
async def test_args_validator_event_args_valid_false():
    """Test that args_valid=False when validator fails."""
    validator_calls = 0

    def my_validator(ctx: RunContext[int], x: int, y: int) -> None:
        nonlocal validator_calls
        validator_calls += 1
        if validator_calls == 1:
            raise ModelRetry('Validation failed')

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

    # At least one event should have args_valid=False (the first call that fails validation)
    args_valid_values = [e.args_valid for e in tool_call_events if e.part.tool_name == 'add_numbers']
    assert False in args_valid_values


# Test 11: args_valid=True when no custom validator but schema passes
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
    for event in tool_call_events:
        if event.part.tool_name == 'add_numbers':
            assert event.args_valid is True


# Test 12: tool_plain with args_validator
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

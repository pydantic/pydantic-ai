"""Tests for issue #5111: after_tool_execute hook fires for internal output tools.

The bug: Unscoped `after_tool_execute` hooks fire for internal output tools (e.g. `final_result`).
When the hook returns `ToolReturn`, the output tool code path doesn't unwrap it (unlike function
tools at `_agent_graph.py:1759`), so `result.output` becomes a `ToolReturn` instead of the
expected output type.

The fix: Skip hooks entirely for output tools (kind='output') to maintain consistency with
the rest of the architecture where output tools are handled separately from function tools.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition


class MyOutput(BaseModel):
    answer: str


pytestmark = [pytest.mark.anyio]


async def test_after_tool_execute_does_not_fire_for_output_tools():
    """Test that after_tool_execute hook does NOT fire for output tools.

    This is the minimal reproducer from issue #5111. Before the fix,
    result.output would be a ToolReturn instead of MyOutput.
    """
    hook_calls: list[str] = []

    hooks: Hooks = Hooks()

    @hooks.on.after_tool_execute
    async def wrap_result(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> ToolReturn:
        hook_calls.append(call.tool_name)
        return ToolReturn(return_value=result, content='Extra context for the model')

    agent = Agent(
        TestModel(),
        output_type=MyOutput,
        capabilities=[hooks],
    )

    result = await agent.run('Say hello')

    # The output should be MyOutput, not ToolReturn
    assert isinstance(result.output, MyOutput), (
        f'Expected MyOutput, got {type(result.output).__name__}. '
        f'The after_tool_execute hook incorrectly fired for output tool.'
    )

    # The hook should NOT have been called for the output tool
    assert 'final_result' not in hook_calls, (
        f'after_tool_execute hook was called for output tool: {hook_calls}'
    )


async def test_after_tool_execute_still_fires_for_function_tools():
    """Test that after_tool_execute hook still fires for function tools."""
    hook_calls: list[str] = []

    hooks: Hooks = Hooks()

    @hooks.on.after_tool_execute
    async def wrap_result(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> ToolReturn:
        hook_calls.append(call.tool_name)
        return ToolReturn(return_value=result, content='Extra context for the model')

    agent = Agent(
        TestModel(),
        output_type=str,
        capabilities=[hooks],
    )

    @agent.tool_plain
    def greet(name: str) -> str:
        """Greet someone by name."""
        return f'Hello, {name}!'

    result = await agent.run('Greet Alice')

    # The hook SHOULD have been called for the function tool
    assert 'greet' in hook_calls, (
        f'after_tool_execute hook was NOT called for function tool: {hook_calls}'
    )


async def test_before_tool_execute_does_not_fire_for_output_tools():
    """Test that before_tool_execute hook does NOT fire for output tools."""
    hook_calls: list[str] = []

    hooks: Hooks = Hooks()

    @hooks.on.before_tool_execute
    async def before_execute(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        hook_calls.append(call.tool_name)
        return args

    agent = Agent(
        TestModel(),
        output_type=MyOutput,
        capabilities=[hooks],
    )

    result = await agent.run('Say hello')

    assert isinstance(result.output, MyOutput)

    # The hook should NOT have been called for the output tool
    assert 'final_result' not in hook_calls, (
        f'before_tool_execute hook was called for output tool: {hook_calls}'
    )


async def test_wrap_tool_execute_does_not_fire_for_output_tools():
    """Test that wrap_tool_execute hook does NOT fire for output tools."""
    hook_calls: list[str] = []

    hooks: Hooks = Hooks()

    @hooks.on.tool_execute  # This is the wrap form of tool_execute hook
    async def wrap_execute(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler,
    ):
        hook_calls.append(call.tool_name)
        return await handler(args)

    agent = Agent(
        TestModel(),
        output_type=MyOutput,
        capabilities=[hooks],
    )

    result = await agent.run('Say hello')

    assert isinstance(result.output, MyOutput)

    # The hook should NOT have been called for the output tool
    assert 'final_result' not in hook_calls, (
        f'wrap_tool_execute hook was called for output tool: {hook_calls}'
    )


async def test_validate_hooks_do_not_fire_for_output_tools():
    """Test that tool_validate hooks do NOT fire for output tools."""
    hook_calls: list[str] = []

    hooks: Hooks = Hooks()

    @hooks.on.before_tool_validate
    async def before_validate(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: str | dict[str, Any],
    ) -> str | dict[str, Any]:
        hook_calls.append(f'before_validate:{call.tool_name}')
        return args

    @hooks.on.after_tool_validate
    async def after_validate(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        hook_calls.append(f'after_validate:{call.tool_name}')
        return args

    agent = Agent(
        TestModel(),
        output_type=MyOutput,
        capabilities=[hooks],
    )

    result = await agent.run('Say hello')

    assert isinstance(result.output, MyOutput)

    # Neither validate hook should have been called for the output tool
    output_tool_hooks = [h for h in hook_calls if 'final_result' in h]
    assert not output_tool_hooks, (
        f'Validate hooks were called for output tool: {output_tool_hooks}'
    )


async def test_hooks_fire_for_function_tools_not_output_tools():
    """Test that hooks fire for function tools but not for output tools in the same agent."""
    hook_calls: list[str] = []

    hooks: Hooks = Hooks()

    @hooks.on.after_tool_execute
    async def after_execute(
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        hook_calls.append(f'after_execute:{call.tool_name}')
        return result

    agent = Agent(
        TestModel(),
        output_type=MyOutput,
        capabilities=[hooks],
    )

    @agent.tool_plain
    def get_greeting() -> str:
        """Get a greeting message."""
        return 'Hello!'

    result = await agent.run('Get a greeting and return it')

    assert isinstance(result.output, MyOutput)

    # Function tool hook should have fired
    function_hooks = [h for h in hook_calls if 'get_greeting' in h]
    assert function_hooks, 'Hook should have fired for function tool'

    # Output tool hook should NOT have fired
    output_hooks = [h for h in hook_calls if 'final_result' in h]
    assert not output_hooks, f'Hook should NOT have fired for output tool: {output_hooks}'

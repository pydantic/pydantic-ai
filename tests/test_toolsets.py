from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, TypeVar
from unittest.mock import AsyncMock

import pytest
from inline_snapshot import snapshot
from typing_extensions import Self

from pydantic_ai import (
    AbstractToolset,
    CombinedToolset,
    FilteredToolset,
    FunctionToolset,
    PrefixedToolset,
    PreparedToolset,
    ToolCallPart,
    ToolsetTool,
    WrapperToolset,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio

T = TypeVar('T')


def build_run_context(deps: T, run_step: int = 0) -> RunContext[T]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
    )


async def test_function_toolset():
    @dataclass
    class PrefixDeps:
        prefix: str | None = None

    toolset = FunctionToolset[PrefixDeps]()

    async def prepare_add_prefix(ctx: RunContext[PrefixDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
        if ctx.deps.prefix is None:
            return tool_def

        return replace(tool_def, name=f'{ctx.deps.prefix}_{tool_def.name}')

    @toolset.tool(prepare=prepare_add_prefix)
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    no_prefix_context = build_run_context(PrefixDeps())
    no_prefix_toolset = await ToolManager[PrefixDeps](toolset).for_run_step(no_prefix_context)
    assert no_prefix_toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='add',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
                description='Add two numbers',
            )
        ]
    )
    assert await no_prefix_toolset.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2})) == 3

    foo_context = build_run_context(PrefixDeps(prefix='foo'))
    foo_toolset = await ToolManager[PrefixDeps](toolset).for_run_step(foo_context)
    assert foo_toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='foo_add',
                description='Add two numbers',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            )
        ]
    )
    assert await foo_toolset.handle_call(ToolCallPart(tool_name='foo_add', args={'a': 1, 'b': 2})) == 3

    @toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b  # pragma: lax no cover

    bar_context = build_run_context(PrefixDeps(prefix='bar'))
    bar_toolset = await ToolManager[PrefixDeps](toolset).for_run_step(bar_context)
    assert bar_toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='bar_add',
                description='Add two numbers',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='subtract',
                description='Subtract two numbers',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
        ]
    )
    assert await bar_toolset.handle_call(ToolCallPart(tool_name='bar_add', args={'a': 1, 'b': 2})) == 3


async def test_function_toolset_with_defaults():
    defaults_toolset = FunctionToolset[None](require_parameter_descriptions=True)

    with pytest.raises(
        UserError,
        match=re.escape('Missing parameter descriptions for'),
    ):

        @defaults_toolset.tool
        def add(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b  # pragma: no cover


async def test_function_toolset_with_defaults_overridden():
    defaults_toolset = FunctionToolset[None](require_parameter_descriptions=True)

    @defaults_toolset.tool(require_parameter_descriptions=False)
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b  # pragma: no cover


async def test_prepared_toolset_user_error_add_new_tools():
    """Test that PreparedToolset raises UserError when prepare function tries to add new tools."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b  # pragma: no cover

    async def prepare_add_new_tool(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # Try to add a new tool that wasn't in the original set
        new_tool = ToolDefinition(
            name='new_tool',
            description='A new tool',
            parameters_json_schema={
                'additionalProperties': False,
                'properties': {'x': {'type': 'integer'}},
                'required': ['x'],
                'type': 'object',
            },
        )
        return tool_defs + [new_tool]

    prepared_toolset = PreparedToolset(base_toolset, prepare_add_new_tool)

    with pytest.raises(
        UserError,
        match=re.escape(
            'Prepare function cannot add or rename tools. Use `FunctionToolset.add_function()` or `RenamedToolset` instead.'
        ),
    ):
        await ToolManager[None](prepared_toolset).for_run_step(context)


async def test_prepared_toolset_user_error_change_tool_names():
    """Test that PreparedToolset raises UserError when prepare function tries to change tool names."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b  # pragma: no cover

    @base_toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b  # pragma: no cover

    async def prepare_change_names(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # Try to change the name of an existing tool
        modified_tool_defs: list[ToolDefinition] = []
        for tool_def in tool_defs:
            if tool_def.name == 'add':
                modified_tool_defs.append(replace(tool_def, name='modified_add'))
            else:
                modified_tool_defs.append(tool_def)
        return modified_tool_defs

    prepared_toolset = PreparedToolset(base_toolset, prepare_change_names)

    with pytest.raises(
        UserError,
        match=re.escape(
            'Prepare function cannot add or rename tools. Use `FunctionToolset.add_function()` or `RenamedToolset` instead.'
        ),
    ):
        await ToolManager[None](prepared_toolset).for_run_step(context)


async def test_comprehensive_toolset_composition():
    """Test that all toolsets can be composed together and work correctly."""

    @dataclass
    class TestDeps:
        user_role: str = 'user'
        enable_advanced: bool = True

    # Create first FunctionToolset with basic math operations
    math_toolset = FunctionToolset[TestDeps]()

    @math_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @math_toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b  # pragma: no cover

    @math_toolset.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b  # pragma: no cover

    # Create second FunctionToolset with string operations
    string_toolset = FunctionToolset[TestDeps]()

    @string_toolset.tool
    def concat(s1: str, s2: str) -> str:
        """Concatenate two strings"""
        return s1 + s2

    @string_toolset.tool
    def uppercase(text: str) -> str:
        """Convert text to uppercase"""
        return text.upper()  # pragma: no cover

    @string_toolset.tool
    def reverse(text: str) -> str:
        """Reverse a string"""
        return text[::-1]  # pragma: no cover

    # Create third FunctionToolset with advanced operations
    advanced_toolset = FunctionToolset[TestDeps]()

    @advanced_toolset.tool
    def power(base: int, exponent: int) -> int:
        """Calculate base raised to the power of exponent"""
        return base**exponent  # pragma: no cover

    # Step 1: Prefix each FunctionToolset individually
    prefixed_math = PrefixedToolset(math_toolset, 'math')
    prefixed_string = PrefixedToolset(string_toolset, 'str')
    prefixed_advanced = PrefixedToolset(advanced_toolset, 'adv')

    # Step 2: Combine the prefixed toolsets
    combined_prefixed_toolset = CombinedToolset([prefixed_math, prefixed_string, prefixed_advanced])

    # Step 3: Filter tools based on user role and advanced flag, now using prefixed names
    def filter_tools(ctx: RunContext[TestDeps], tool_def: ToolDefinition) -> bool:
        # Only allow advanced tools if enable_advanced is True
        if tool_def.name.startswith('adv_') and not ctx.deps.enable_advanced:
            return False
        # Only allow string operations for admin users (simulating role-based access)
        if tool_def.name.startswith('str_') and ctx.deps.user_role != 'admin':
            return False
        return True

    filtered_toolset = FilteredToolset[TestDeps](combined_prefixed_toolset, filter_tools)

    # Step 4: Apply prepared toolset to modify descriptions (add user role annotation)
    async def prepare_add_context(ctx: RunContext[TestDeps], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # Annotate each tool description with the user role
        role = ctx.deps.user_role
        return [replace(td, description=f'{td.description} (role: {role})') for td in tool_defs]

    prepared_toolset = PreparedToolset(filtered_toolset, prepare_add_context)

    # Step 5: Test the fully composed toolset
    # Test with regular user context
    regular_deps = TestDeps(user_role='user', enable_advanced=True)
    regular_context = build_run_context(regular_deps)
    final_toolset = await ToolManager[TestDeps](prepared_toolset).for_run_step(regular_context)
    # Tool definitions should have role annotation
    assert final_toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='math_add',
                description='Add two numbers (role: user)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='math_subtract',
                description='Subtract two numbers (role: user)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='math_multiply',
                description='Multiply two numbers (role: user)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='adv_power',
                description='Calculate base raised to the power of exponent (role: user)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'base': {'type': 'integer'}, 'exponent': {'type': 'integer'}},
                    'required': ['base', 'exponent'],
                    'type': 'object',
                },
            ),
        ]
    )
    # Call a tool and check result
    result = await final_toolset.handle_call(ToolCallPart(tool_name='math_add', args={'a': 5, 'b': 3}))
    assert result == 8

    # Test with admin user context (should have string tools)
    admin_deps = TestDeps(user_role='admin', enable_advanced=True)
    admin_context = build_run_context(admin_deps)
    admin_final_toolset = await ToolManager[TestDeps](prepared_toolset).for_run_step(admin_context)
    assert admin_final_toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='math_add',
                description='Add two numbers (role: admin)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='math_subtract',
                description='Subtract two numbers (role: admin)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='math_multiply',
                description='Multiply two numbers (role: admin)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='str_concat',
                description='Concatenate two strings (role: admin)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'s1': {'type': 'string'}, 's2': {'type': 'string'}},
                    'required': ['s1', 's2'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='str_uppercase',
                description='Convert text to uppercase (role: admin)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'text': {'type': 'string'}},
                    'required': ['text'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='str_reverse',
                description='Reverse a string (role: admin)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'text': {'type': 'string'}},
                    'required': ['text'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='adv_power',
                description='Calculate base raised to the power of exponent (role: admin)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'base': {'type': 'integer'}, 'exponent': {'type': 'integer'}},
                    'required': ['base', 'exponent'],
                    'type': 'object',
                },
            ),
        ]
    )
    result = await admin_final_toolset.handle_call(
        ToolCallPart(tool_name='str_concat', args={'s1': 'Hello', 's2': 'World'})
    )
    assert result == 'HelloWorld'

    # Test with advanced features disabled
    basic_deps = TestDeps(user_role='user', enable_advanced=False)
    basic_context = build_run_context(basic_deps)
    basic_final_toolset = await ToolManager[TestDeps](prepared_toolset).for_run_step(basic_context)
    assert basic_final_toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='math_add',
                description='Add two numbers (role: user)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='math_subtract',
                description='Subtract two numbers (role: user)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
            ToolDefinition(
                name='math_multiply',
                description='Multiply two numbers (role: user)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'a': {'type': 'integer'}, 'b': {'type': 'integer'}},
                    'required': ['a', 'b'],
                    'type': 'object',
                },
            ),
        ]
    )


async def test_context_manager():
    try:
        from pydantic_ai.mcp import MCPServerStdio
    except ImportError:  # pragma: lax no cover
        pytest.skip('mcp is not installed')

    server1 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    server2 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    toolset = CombinedToolset([server1, PrefixedToolset(server2, 'prefix')])

    async with toolset:
        assert server1.is_running
        assert server2.is_running

        async with toolset:
            assert server1.is_running
            assert server2.is_running


class InitializationError(Exception):
    pass


async def test_context_manager_failed_initialization():
    """Test if MCP servers stop if any MCP server fails to initialize."""
    try:
        from pydantic_ai.mcp import MCPServerStdio
    except ImportError:  # pragma: lax no cover
        pytest.skip('mcp is not installed')

    server1 = MCPServerStdio('python', ['-m', 'tests.mcp_server'])
    server2 = AsyncMock()
    server2.__aenter__.side_effect = InitializationError

    toolset = CombinedToolset([server1, server2])

    with pytest.raises(InitializationError):
        async with toolset:
            pass

    assert server1.is_running is False


async def test_tool_manager_reuse_self():
    """Test the retry logic with failed_tools and for_run_step method."""

    run_context = build_run_context(None, run_step=1)

    tool_manager = await ToolManager[None](FunctionToolset()).for_run_step(run_context)

    same_tool_manager = await tool_manager.for_run_step(ctx=run_context)

    assert tool_manager is same_tool_manager

    step_2_context = build_run_context(None, run_step=2)

    updated_tool_manager = await tool_manager.for_run_step(ctx=step_2_context)

    assert tool_manager != updated_tool_manager


async def test_tool_manager_retry_logic():
    """Test the retry logic with failed_tools and for_run_step method."""

    @dataclass
    class TestDeps:
        pass

    # Create a toolset with tools that can fail
    toolset = FunctionToolset[TestDeps](max_retries=2)
    call_count: defaultdict[str, int] = defaultdict(int)

    @toolset.tool
    def failing_tool(x: int) -> int:
        """A tool that always fails"""
        call_count['failing_tool'] += 1
        raise ModelRetry('This tool always fails')

    @toolset.tool
    def other_tool(x: int) -> int:
        """A tool that works"""
        call_count['other_tool'] += 1
        return x * 2

    # Create initial context and tool manager
    initial_context = build_run_context(TestDeps())
    tool_manager = await ToolManager[TestDeps](toolset).for_run_step(initial_context)

    # Initially no failed tools
    assert tool_manager.failed_tools == set()
    assert initial_context.retries == {}

    # Call the failing tool - should add to failed_tools
    with pytest.raises(ToolRetryError):
        await tool_manager.handle_call(ToolCallPart(tool_name='failing_tool', args={'x': 1}))

    assert tool_manager.failed_tools == {'failing_tool'}
    assert call_count['failing_tool'] == 1

    # Call the working tool - should not add to failed_tools
    result = await tool_manager.handle_call(ToolCallPart(tool_name='other_tool', args={'x': 3}))
    assert result == 6
    assert tool_manager.failed_tools == {'failing_tool'}  # unchanged
    assert call_count['other_tool'] == 1

    # Test for_run_step - should create new tool manager with updated retry counts
    new_context = build_run_context(TestDeps(), run_step=1)
    new_tool_manager = await tool_manager.for_run_step(new_context)

    # The new tool manager should have retry count for the failed tool
    assert new_tool_manager.ctx is not None
    assert new_tool_manager.ctx.retries == {'failing_tool': 1}
    assert new_tool_manager.failed_tools == set()  # reset for new run step

    # Call the failing tool again in the new manager - should have retry=1
    with pytest.raises(ToolRetryError):
        await new_tool_manager.handle_call(ToolCallPart(tool_name='failing_tool', args={'x': 1}))

    # Call the failing tool another time in the new manager
    with pytest.raises(ToolRetryError):
        await new_tool_manager.handle_call(ToolCallPart(tool_name='failing_tool', args={'x': 1}))

    # Call the failing tool a third time in the new manager
    with pytest.raises(ToolRetryError):
        await new_tool_manager.handle_call(ToolCallPart(tool_name='failing_tool', args={'x': 1}))

    assert new_tool_manager.failed_tools == {'failing_tool'}
    assert call_count['failing_tool'] == 4

    # Create another run step
    another_context = build_run_context(TestDeps(), run_step=2)
    another_tool_manager = await new_tool_manager.for_run_step(another_context)

    # Should now have retry count of 2 for failing_tool
    assert another_tool_manager.ctx is not None
    assert another_tool_manager.ctx.retries == {'failing_tool': 2}
    assert another_tool_manager.failed_tools == set()

    # Call the failing tool _again_, now we should finally hit the limit
    with pytest.raises(UnexpectedModelBehavior, match="Tool 'failing_tool' exceeded max retries count of 2"):
        await another_tool_manager.handle_call(ToolCallPart(tool_name='failing_tool', args={'x': 1}))


async def test_tool_manager_multiple_failed_tools():
    """Test retry logic when multiple tools fail in the same run step."""

    @dataclass
    class TestDeps:
        pass

    toolset = FunctionToolset[TestDeps]()

    @toolset.tool
    def tool_a(x: int) -> int:
        """Tool A that fails"""
        raise ModelRetry('Tool A fails')

    @toolset.tool
    def tool_b(x: int) -> int:
        """Tool B that fails"""
        raise ModelRetry('Tool B fails')

    @toolset.tool
    def tool_c(x: int) -> int:
        """Tool C that works"""
        return x * 3

    # Create tool manager
    context = build_run_context(TestDeps())
    tool_manager = await ToolManager[TestDeps](toolset).for_run_step(context)

    # Call tool_a - should fail and be added to failed_tools
    with pytest.raises(ToolRetryError):
        await tool_manager.handle_call(ToolCallPart(tool_name='tool_a', args={'x': 1}))
    assert tool_manager.failed_tools == {'tool_a'}

    # Call tool_b - should also fail and be added to failed_tools
    with pytest.raises(ToolRetryError):
        await tool_manager.handle_call(ToolCallPart(tool_name='tool_b', args={'x': 1}))
    assert tool_manager.failed_tools == {'tool_a', 'tool_b'}

    # Call tool_c - should succeed and not be added to failed_tools
    result = await tool_manager.handle_call(ToolCallPart(tool_name='tool_c', args={'x': 2}))
    assert result == 6
    assert tool_manager.failed_tools == {'tool_a', 'tool_b'}  # unchanged

    # Create next run step - should have retry counts for both failed tools
    new_context = build_run_context(TestDeps(), run_step=1)
    new_tool_manager = await tool_manager.for_run_step(new_context)

    assert new_tool_manager.ctx is not None
    assert new_tool_manager.ctx.retries == {'tool_a': 1, 'tool_b': 1}
    assert new_tool_manager.failed_tools == set()  # reset for new run step


async def test_tool_manager_sequential_tool_call():
    toolset = FunctionToolset[None]()

    @toolset.tool(sequential=True)
    def tool_a(x: int) -> int: ...  # pragma: no cover

    @toolset.tool(sequential=False)
    def tool_b(x: int) -> int: ...  # pragma: no cover

    tool_manager = ToolManager[None](toolset)

    prepared_tool_manager = await tool_manager.for_run_step(build_run_context(None))

    assert prepared_tool_manager.should_call_sequentially([ToolCallPart(tool_name='tool_a', args={'x': 1})])
    assert not prepared_tool_manager.should_call_sequentially([ToolCallPart(tool_name='tool_b', args={'x': 1})])

    assert prepared_tool_manager.should_call_sequentially(
        [ToolCallPart(tool_name='tool_a', args={'x': 1}), ToolCallPart(tool_name='tool_b', args={'x': 1})]
    )
    assert prepared_tool_manager.should_call_sequentially(
        [ToolCallPart(tool_name='tool_b', args={'x': 1}), ToolCallPart(tool_name='tool_a', args={'x': 1})]
    )


async def test_visit_and_replace():
    toolset1 = FunctionToolset(id='toolset1')
    toolset2 = FunctionToolset(id='toolset2')

    active_dynamic_toolset = DynamicToolset(toolset_func=lambda ctx: toolset2)
    await active_dynamic_toolset.get_tools(build_run_context(None))
    assert active_dynamic_toolset._toolset is toolset2  # pyright: ignore[reportPrivateUsage]

    inactive_dynamic_toolset = DynamicToolset(toolset_func=lambda ctx: FunctionToolset())

    toolset = CombinedToolset(
        [
            WrapperToolset(toolset1),
            active_dynamic_toolset,
            inactive_dynamic_toolset,
        ]
    )
    visited_toolset = toolset.visit_and_replace(lambda toolset: WrapperToolset(toolset))
    assert visited_toolset == CombinedToolset(
        [
            WrapperToolset(WrapperToolset(toolset1)),
            DynamicToolset(
                toolset_func=active_dynamic_toolset.toolset_func,
                per_run_step=active_dynamic_toolset.per_run_step,
                _toolset=WrapperToolset(toolset2),
                _run_step=active_dynamic_toolset._run_step,  # pyright: ignore[reportPrivateUsage]
            ),
            WrapperToolset(inactive_dynamic_toolset),
        ]
    )


async def test_dynamic_toolset():
    class EnterableToolset(AbstractToolset[None]):
        entered_count = 0
        exited_count = 0

        @property
        def id(self) -> str | None:
            return None  # pragma: no cover

        @property
        def depth_count(self) -> int:
            return self.entered_count - self.exited_count

        async def __aenter__(self) -> Self:
            self.entered_count += 1
            return self

        async def __aexit__(self, *args: Any) -> bool | None:
            self.exited_count += 1
            return None

        async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
            return {}

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
        ) -> Any:
            return None  # pragma: no cover

    def toolset_factory(ctx: RunContext[None]) -> AbstractToolset[None]:
        return EnterableToolset()

    toolset = DynamicToolset[None](toolset_func=toolset_factory)

    def get_inner_toolset(toolset: DynamicToolset[None] | None) -> EnterableToolset | None:
        assert toolset is not None
        inner_toolset = toolset._toolset  # pyright: ignore[reportPrivateUsage]
        assert isinstance(inner_toolset, EnterableToolset) or inner_toolset is None
        return inner_toolset

    run_context = build_run_context(None)

    async with toolset:
        assert not toolset._toolset  # pyright: ignore[reportPrivateUsage]

        # Test that calling get_tools initializes the toolset
        tools = await toolset.get_tools(run_context)

        assert (inner_toolset := get_inner_toolset(toolset))
        assert inner_toolset.depth_count == 1

        # Test that the visitor applies when the toolset is initialized
        def initialized_visitor(toolset: AbstractToolset[None]) -> None:
            assert toolset is inner_toolset

        toolset.apply(initialized_visitor)

    assert get_inner_toolset(toolset) is None

    def uninitialized_visitor(visited_toolset: AbstractToolset[None]) -> None:
        assert visited_toolset is toolset

    toolset.apply(uninitialized_visitor)

    assert tools == {}


async def test_dynamic_toolset_empty():
    def no_toolset_func(ctx: RunContext[None]) -> None:
        return None

    toolset = DynamicToolset[None](toolset_func=no_toolset_func)

    run_context = build_run_context(None)

    tools = await toolset.get_tools(run_context)

    assert tools == {}

    async with toolset:
        assert toolset._toolset is None  # pyright: ignore[reportPrivateUsage]

        tools = await toolset.get_tools(run_context)

        assert tools == {}

        assert toolset._toolset is None  # pyright: ignore[reportPrivateUsage]

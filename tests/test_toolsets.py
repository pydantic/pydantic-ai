from __future__ import annotations

import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import timezone
from typing import Any, TypeVar
from unittest.mock import AsyncMock

import anyio
import pytest
from pydantic import ValidationError
from typing_extensions import Self

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup as BaseExceptionGroup  # pragma: lax no cover
else:
    BaseExceptionGroup = BaseExceptionGroup  # pragma: lax no cover

from pydantic_ai import (
    AbstractToolset,
    Agent,
    CombinedToolset,
    FilteredToolset,
    FunctionToolset,
    PrefixedToolset,
    PreparedToolset,
    ToolCallPart,
    ToolsetTool,
    WrapperToolset,
    capture_run_messages,
)
from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ModelRetry, ToolRetryError, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import (
    InstructionPart,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import Tool, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.usage import RequestUsage, RunUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsNow, IsStr

pytestmark = pytest.mark.anyio

T = TypeVar('T')


def build_run_context(deps: T, run_step: int = 0, max_retries: int = 0) -> RunContext[T]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=run_step,
        max_retries=max_retries,
    )


class MockToolsetWithInstructions(AbstractToolset[Any]):
    """A test toolset that returns custom instructions."""

    def __init__(self, instructions: str | None = None, id: str | None = None):
        self.custom_instructions = instructions
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id

    async def get_instructions(self, ctx: RunContext[Any]) -> str | None:
        return self.custom_instructions

    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        return {}

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> Any:
        return None


async def test_mock_toolset_with_instructions_interface():
    """Test that MockToolsetWithInstructions correctly implements AbstractToolset interface."""
    toolset = MockToolsetWithInstructions(instructions='test instructions', id='my-id')
    ctx = build_run_context(None)

    assert toolset.id == 'my-id'
    assert await toolset.get_tools(ctx) == {}
    assert await toolset.call_tool('any_tool', {}, ctx, None) is None  # type: ignore[arg-type]


async def test_function_toolset_instructions():
    """Test that FunctionToolset returns None for instructions by default."""
    toolset = FunctionToolset[None]()
    ctx = build_run_context(None)
    instructions = await toolset.get_instructions(ctx)
    assert instructions is None


async def test_function_toolset():
    @dataclass
    class PrefixDeps:
        prefix: str | None = None

    toolset = FunctionToolset[PrefixDeps]()

    async def prepare_add_prefix(ctx: RunContext[PrefixDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
        if ctx.deps.prefix is None:
            return tool_def

        return replace(tool_def, name=f'{ctx.deps.prefix}_{tool_def.name}')

    @toolset.tool_plain(prepare=prepare_add_prefix)
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
            )
        ]
    )
    assert await foo_toolset.handle_call(ToolCallPart(tool_name='foo_add', args={'a': 1, 'b': 2})) == 3

    @toolset.tool_plain
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
            ),
        ]
    )
    assert await bar_toolset.handle_call(ToolCallPart(tool_name='bar_add', args={'a': 1, 'b': 2})) == 3


async def test_toolset_tool_function_signature_property():
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def add(a: int, b: int) -> int:
        return a + b

    managed_toolset = await ToolManager[None](toolset).for_run_step(build_run_context(None))
    assert managed_toolset.tools is not None

    td = managed_toolset.tools['add'].tool_def
    sig = td.function_signature
    assert sig is not None
    assert list(sig.params) == ['a', 'b']
    assert td.render_signature('...') == snapshot("""\
def add(*, a: int, b: int) -> int:
    ...\
""")

    assert await managed_toolset.handle_call(ToolCallPart(tool_name='add', args={'a': 1, 'b': 2})) == 3


async def test_function_toolset_with_defaults():
    defaults_toolset = FunctionToolset[None](require_parameter_descriptions=True)

    with pytest.raises(
        UserError,
        match=re.escape('Missing parameter descriptions for'),
    ):

        @defaults_toolset.tool_plain
        def add(a: int, b: int) -> int:
            """Add two numbers"""
            return a + b  # pragma: no cover


async def test_abstract_toolset_instructions_default():
    """Test that the default instructions method returns None."""
    toolset = MockToolsetWithInstructions(instructions=None)
    ctx = build_run_context(None)
    instructions = await toolset.get_instructions(ctx)
    assert instructions is None


async def test_abstract_toolset_instructions_custom():
    """Test that instructions can return custom instructions."""
    custom_instructions = 'Use these tools carefully and always validate inputs.'
    toolset = MockToolsetWithInstructions(instructions=custom_instructions)
    ctx = build_run_context(None)
    instructions = await toolset.get_instructions(ctx)
    assert instructions == custom_instructions


async def test_abstract_toolset_instructions_empty_string():
    """Test that instructions can return an empty string."""
    toolset = MockToolsetWithInstructions(instructions='')
    ctx = build_run_context(None)
    instructions = await toolset.get_instructions(ctx)
    assert instructions == ''


async def test_function_toolset_with_defaults_overridden():
    defaults_toolset = FunctionToolset[None](require_parameter_descriptions=True)

    @defaults_toolset.tool_plain(require_parameter_descriptions=False)
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b  # pragma: no cover


async def test_prepared_toolset_sync_prepare_func():
    """`PreparedToolset` accepts a synchronous prepare function (no await needed)."""
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b  # pragma: no cover

    def prepare_keep_first(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        return tool_defs[:1]

    prepared_toolset = PreparedToolset(base_toolset, prepare_keep_first)

    tools = await prepared_toolset.get_tools(build_run_context(None))
    assert list(tools.keys()) == ['add']


async def test_prepared_toolset_user_error_add_new_tools():
    """Test that PreparedToolset raises UserError when prepare function tries to add new tools."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool_plain
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

    @base_toolset.tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b  # pragma: no cover

    @base_toolset.tool_plain
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

    @math_toolset.tool_plain
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @math_toolset.tool_plain
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b  # pragma: no cover

    @math_toolset.tool_plain
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b  # pragma: no cover

    # Create second FunctionToolset with string operations
    string_toolset = FunctionToolset[TestDeps]()

    @string_toolset.tool_plain
    def concat(s1: str, s2: str) -> str:
        """Concatenate two strings"""
        return s1 + s2

    @string_toolset.tool_plain
    def uppercase(text: str) -> str:
        """Convert text to uppercase"""
        return text.upper()  # pragma: no cover

    @string_toolset.tool_plain
    def reverse(text: str) -> str:
        """Reverse a string"""
        return text[::-1]  # pragma: no cover

    # Create third FunctionToolset with advanced operations
    advanced_toolset = FunctionToolset[TestDeps]()

    @advanced_toolset.tool_plain
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'string'},
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
                return_schema={'type': 'string'},
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
                return_schema={'type': 'string'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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
                return_schema={'type': 'integer'},
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

    @toolset.tool_plain
    def failing_tool(x: int) -> int:
        """A tool that always fails"""
        call_count['failing_tool'] += 1
        raise ModelRetry('This tool always fails')

    @toolset.tool_plain
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


async def test_handle_call_wrap_validation_errors_false():
    """`handle_call(wrap_validation_errors=False)` propagates raw errors and leaves retry-budget state untouched.

    Used by sandboxed callers (e.g. code-mode dispatch) that want validation and
    `ModelRetry` failures to surface at the sandbox `await` site as the original
    exception type, without consuming the agent's retry budget for the wrapping call.
    Mirrors the `wrap_validation_errors` flag on the output-tool methods.
    """

    toolset = FunctionToolset[None](max_retries=2)

    @toolset.tool_plain
    def needs_int(x: int) -> int:
        return x * 2

    @toolset.tool_plain
    def retrying() -> int:
        raise ModelRetry('please retry')

    tool_manager = await ToolManager[None](toolset).for_run_step(build_run_context(None))

    # Sanity: a valid call still works in raw mode (no path differences for happy paths).
    assert (
        await tool_manager.handle_call(
            ToolCallPart(tool_name='needs_int', args={'x': 5}),
            wrap_validation_errors=False,
        )
        == 10
    )

    # Pydantic ValidationError on bad args propagates raw, not as ToolRetryError.
    with pytest.raises(ValidationError):
        await tool_manager.handle_call(
            ToolCallPart(tool_name='needs_int', args={'x': 'not an int'}),
            wrap_validation_errors=False,
        )
    assert tool_manager.failed_tools == set()

    # ModelRetry from the tool body propagates raw too.
    with pytest.raises(ModelRetry, match='please retry'):
        await tool_manager.handle_call(
            ToolCallPart(tool_name='retrying', args={}),
            wrap_validation_errors=False,
        )
    assert tool_manager.failed_tools == set()

    # Default (wrap=True) still wraps as ToolRetryError and tracks failed tools.
    with pytest.raises(ToolRetryError):
        await tool_manager.handle_call(ToolCallPart(tool_name='needs_int', args={'x': 'not an int'}))
    assert tool_manager.failed_tools == {'needs_int'}

    with pytest.raises(ToolRetryError):
        await tool_manager.handle_call(ToolCallPart(tool_name='retrying', args={}))
    assert tool_manager.failed_tools == {'needs_int', 'retrying'}


async def test_toolset_max_retries_inherits_from_agent():
    """Agent(retries=...) should propagate to user-provided toolsets that don't set max_retries explicitly."""
    attempts: list[int] = []
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def always_fails(x: int) -> int:
        """A tool that always fails."""
        attempts.append(x)
        raise ModelRetry('Always fails')

    agent = Agent('test', toolsets=[toolset], tool_retries=0, output_retries=0)

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 0'):
            await agent.run('call always_fails', model=TestModel())

    # retries=0 means the tool is called once and then fails immediately.
    assert len(attempts) == 1
    assert messages == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='call always_fails', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='always_fails', args={'x': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_toolset_explicit_max_retries_overrides_agent():
    """FunctionToolset(max_retries=X) should take precedence over Agent(retries=Y)."""
    toolset = FunctionToolset[None](max_retries=2)
    attempts: list[int] = []

    @toolset.tool_plain
    def always_fails(x: int) -> int:
        """A tool that always fails."""
        attempts.append(x)
        raise ModelRetry('Always fails')

    agent = Agent('test', toolsets=[toolset], tool_retries=0, output_retries=0)

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 2'):
            await agent.run('call always_fails', model=TestModel())

    # Initial call + 2 retries = 3 attempts.
    assert len(attempts) == 3
    assert [type(m).__name__ for m in messages] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    retry_parts = [p for m in messages for p in getattr(m, 'parts', []) if isinstance(p, RetryPromptPart)]
    assert [p.content for p in retry_parts] == snapshot(['Always fails', 'Always fails'])


async def test_tool_explicit_retries_overrides_toolset_and_agent():
    """Tool(retries=X) should take precedence over both toolset and agent defaults."""
    attempts: list[int] = []

    def always_fails(x: int) -> int:
        """A tool that always fails."""
        attempts.append(x)
        raise ModelRetry('Always fails')

    toolset = FunctionToolset[None](tools=[Tool(always_fails, max_retries=3)])
    agent = Agent('test', toolsets=[toolset], tool_retries=0, output_retries=0)

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 3'):
            await agent.run('call always_fails', model=TestModel())

    # Initial call + 3 retries = 4 attempts.
    assert len(attempts) == 4
    retry_parts = [p for m in messages for p in getattr(m, 'parts', []) if isinstance(p, RetryPromptPart)]
    assert [p.content for p in retry_parts] == snapshot(['Always fails', 'Always fails', 'Always fails'])


async def test_prepare_function_sees_agent_max_retries():
    """Prepare functions should see the agent's default max_retries on ctx when the toolset doesn't set one."""
    captured_max_retries: list[int] = []
    captured_last_attempt: list[bool] = []

    async def capture_prepare(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition:
        captured_max_retries.append(ctx.max_retries)
        captured_last_attempt.append(ctx.last_attempt)
        return tool_def

    toolset = FunctionToolset[None]()

    @toolset.tool_plain(prepare=capture_prepare)
    def my_tool(x: int) -> int:
        """A tool."""
        return x

    agent = Agent('test', toolsets=[toolset], tool_retries=3, output_retries=3)
    result = await agent.run('call my_tool', model=TestModel())

    assert captured_max_retries[0] == 3
    assert captured_last_attempt[0] is False
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='call my_tool', timestamp=IsDatetime())],
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={'x': 0}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=52, output_tokens=4),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content=0,
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='{"my_tool":0}')],
                usage=RequestUsage(input_tokens=53, output_tokens=7),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_toolset_tool_max_retries_none_uses_tool_retries_not_output_retries():
    """When a user toolset leaves `max_retries=None` and `retries != output_retries`, the fallback
    must resolve to the agent's **tool** retry count, not the output retry count.
    Regression: `ctx.max_retries` previously carried `max_output_retries` during `get_tools`."""
    attempts: list[int] = []
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def always_fails(x: int) -> int:
        """A tool that always fails."""
        attempts.append(x)
        raise ModelRetry('Always fails')

    agent = Agent('test', toolsets=[toolset], tool_retries=1, output_retries=5)

    with capture_run_messages() as messages:
        with pytest.raises(UnexpectedModelBehavior, match='exceeded max retries count of 1'):
            await agent.run('call always_fails', model=TestModel())

    # retries=1 means initial call + 1 retry = 2 attempts, not 6 (which would be output_retries).
    assert len(attempts) == 2
    assert [type(m).__name__ for m in messages] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )
    retry_parts = [p for m in messages for p in getattr(m, 'parts', []) if isinstance(p, RetryPromptPart)]
    assert [p.content for p in retry_parts] == snapshot(['Always fails'])


async def test_prepare_function_sees_tool_retries_not_output_retries():
    """Prepare functions must see the agent's **tool** retry count on `ctx.max_retries`,
    not the output retry count. Regression for a non-output toolset previously receiving the
    output-tool preparation context."""
    captured: list[int] = []

    async def capture_prepare(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition:
        captured.append(ctx.max_retries)
        return tool_def

    toolset = FunctionToolset[None]()

    @toolset.tool_plain(prepare=capture_prepare)
    def my_tool(x: int) -> int:
        """A tool."""
        return x

    agent = Agent('test', toolsets=[toolset], tool_retries=1, output_retries=5)
    result = await agent.run('call my_tool', model=TestModel())

    assert captured[0] == 1
    assert [type(m).__name__ for m in result.all_messages()] == snapshot(
        ['ModelRequest', 'ModelResponse', 'ModelRequest', 'ModelResponse']
    )


async def test_tool_manager_multiple_failed_tools():
    """Test retry logic when multiple tools fail in the same run step."""

    @dataclass
    class TestDeps:
        pass

    toolset = FunctionToolset[TestDeps]()

    @toolset.tool_plain
    def tool_a(x: int) -> int:
        """Tool A that fails"""
        raise ModelRetry('Tool A fails')

    @toolset.tool_plain
    def tool_b(x: int) -> int:
        """Tool B that fails"""
        raise ModelRetry('Tool B fails')

    @toolset.tool_plain
    def tool_c(x: int) -> int:
        """Tool C that works"""
        return x * 3

    # Create tool manager with max_retries=1, matching what _agent_graph.py sets in a real run
    context = build_run_context(TestDeps(), max_retries=1)
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
    new_context = build_run_context(TestDeps(), run_step=1, max_retries=1)
    new_tool_manager = await tool_manager.for_run_step(new_context)

    assert new_tool_manager.ctx is not None
    assert new_tool_manager.ctx.retries == {'tool_a': 1, 'tool_b': 1}
    assert new_tool_manager.failed_tools == set()  # reset for new run step


async def test_tool_manager_sequential_tool_call():
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(sequential=True)
    def tool_a(x: int) -> int: ...  # pragma: no cover

    @toolset.tool_plain(sequential=False)
    def tool_b(x: int) -> int: ...  # pragma: no cover

    tool_manager = ToolManager[None](toolset)

    prepared_tool_manager = await tool_manager.for_run_step(build_run_context(None))

    assert (
        prepared_tool_manager.get_parallel_execution_mode([ToolCallPart(tool_name='tool_a', args={'x': 1})])
        == 'sequential'
    )
    assert (
        not prepared_tool_manager.get_parallel_execution_mode([ToolCallPart(tool_name='tool_b', args={'x': 1})])
        == 'sequential'
    )

    assert (
        prepared_tool_manager.get_parallel_execution_mode(
            [ToolCallPart(tool_name='tool_a', args={'x': 1}), ToolCallPart(tool_name='tool_b', args={'x': 1})]
        )
        == 'sequential'
    )
    assert (
        prepared_tool_manager.get_parallel_execution_mode(
            [ToolCallPart(tool_name='tool_b', args={'x': 1}), ToolCallPart(tool_name='tool_a', args={'x': 1})]
        )
        == 'sequential'
    )


async def test_visit_and_replace():
    toolset1 = FunctionToolset(id='toolset1')
    toolset2 = FunctionToolset(id='toolset2')

    run_ctx = build_run_context(None)

    active_dynamic_toolset = DynamicToolset(toolset_func=lambda ctx: toolset2)
    active_dynamic_toolset = await active_dynamic_toolset.for_run(run_ctx)
    assert isinstance(active_dynamic_toolset, DynamicToolset)
    # for_run with per_run_step=True defers factory evaluation; for_run_step evaluates in-place
    await active_dynamic_toolset.for_run_step(run_ctx)
    assert active_dynamic_toolset._toolset is not None  # pyright: ignore[reportPrivateUsage]
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

    expected_dynamic = DynamicToolset(
        toolset_func=active_dynamic_toolset.toolset_func,
        per_run_step=active_dynamic_toolset.per_run_step,
        id=active_dynamic_toolset._id,  # pyright: ignore[reportPrivateUsage]
    )
    expected_dynamic._toolset = WrapperToolset(toolset2)  # pyright: ignore[reportPrivateUsage]

    assert visited_toolset == CombinedToolset(
        [
            WrapperToolset(WrapperToolset(toolset1)),
            expected_dynamic,
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

    original_toolset = DynamicToolset[None](toolset_func=toolset_factory)

    run_context = build_run_context(None)

    def get_inner_toolset(toolset: DynamicToolset[None] | None) -> EnterableToolset | None:
        assert toolset is not None
        inner_toolset = toolset._toolset  # pyright: ignore[reportPrivateUsage]
        assert isinstance(inner_toolset, EnterableToolset) or inner_toolset is None
        return inner_toolset

    # for_run returns a new per-run copy; per_run_step=True defers factory evaluation
    toolset = await original_toolset.for_run(run_context)
    assert isinstance(toolset, DynamicToolset)
    assert toolset is not original_toolset
    assert toolset._toolset is None  # pyright: ignore[reportPrivateUsage]

    async with toolset:
        # for_run_step evaluates the factory and manages transitions in-place
        step_toolset = await toolset.for_run_step(run_context)
        assert step_toolset is toolset  # returns self after in-place update

        assert (inner_toolset := get_inner_toolset(toolset))
        assert inner_toolset.depth_count == 1

        tools = await toolset.get_tools(run_context)

        # Test that the visitor applies when the toolset is initialized
        def initialized_visitor(visited_toolset: AbstractToolset[None]) -> None:
            assert visited_toolset is inner_toolset

        toolset.apply(initialized_visitor)

    assert get_inner_toolset(toolset) is None

    def uninitialized_visitor(visited_toolset: AbstractToolset[None]) -> None:
        assert visited_toolset is original_toolset

    original_toolset.apply(uninitialized_visitor)

    assert tools == {}


async def test_dynamic_toolset_enter_failure_does_not_exit_unentered_toolset():
    """If the inner toolset's __aenter__ raises, DynamicToolset.__aexit__ must not
    try to exit a toolset that was never entered.

    Reproduces https://github.com/pydantic/pydantic-ai/issues/3542: a per-run-step
    factory produces a fresh toolset each step; if __aenter__ on the new one fails,
    the old logic still stored it and the outer context manager then called
    __aexit__ on an unentered toolset (MCPServer raised "__aexit__ called more
    times than __aenter__").
    """

    class FlakyToolset(AbstractToolset[None]):
        enter_count = 0
        exit_count = 0
        fail_on_enter = False

        @property
        def id(self) -> str | None:
            return None  # pragma: no cover

        async def __aenter__(self) -> Self:
            if self.fail_on_enter:
                raise RuntimeError('enter failed')
            self.enter_count += 1
            return self

        async def __aexit__(self, *args: Any) -> bool | None:
            self.exit_count += 1
            return None

        async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
            return {}  # pragma: no cover

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
        ) -> Any:
            return None  # pragma: no cover

    first = FlakyToolset()
    second = FlakyToolset()
    second.fail_on_enter = True
    produced = iter([first, second])

    def factory(ctx: RunContext[None]) -> AbstractToolset[None]:
        return next(produced)

    dynamic = DynamicToolset[None](toolset_func=factory)
    run_context = build_run_context(None)

    toolset = await dynamic.for_run(run_context)
    assert isinstance(toolset, DynamicToolset)
    async with toolset:
        await toolset.for_run_step(run_context)
        assert first.enter_count == 1
        with pytest.raises(RuntimeError, match='enter failed'):
            await toolset.for_run_step(run_context)
        # After the failed transition, _toolset should be None so __aexit__ is a no-op.
        assert toolset._toolset is None  # pyright: ignore[reportPrivateUsage]

    # Old toolset was exited exactly once during the transition; the failed one
    # was never entered so must never be exited.
    assert first.exit_count == 1
    assert second.exit_count == 0


async def test_dynamic_toolset_aenter_failure_does_not_exit_unentered_toolset():
    """If the initial outer __aenter__ fails, __aexit__ must not try to exit it."""

    class FailingEnterToolset(AbstractToolset[None]):
        exit_count = 0

        @property
        def id(self) -> str | None:
            return None  # pragma: no cover

        async def __aenter__(self) -> Self:
            raise RuntimeError('enter failed')

        async def __aexit__(self, *args: Any) -> bool | None:  # pragma: no cover
            self.exit_count += 1
            return None

        async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
            return {}  # pragma: no cover

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
        ) -> Any:
            return None  # pragma: no cover

    inner = FailingEnterToolset()
    dynamic = DynamicToolset[None](toolset_func=lambda ctx: inner, per_run_step=False)
    run_context = build_run_context(None)

    toolset = await dynamic.for_run(run_context)
    assert isinstance(toolset, DynamicToolset)
    with pytest.raises(RuntimeError, match='enter failed'):
        async with toolset:
            pass  # pragma: no cover

    assert toolset._toolset is None  # pyright: ignore[reportPrivateUsage]
    assert inner.exit_count == 0


async def test_dynamic_toolset_old_aexit_failure_does_not_store_new_toolset():
    """If the old toolset's __aexit__ raises during a per-run-step transition,
    the new toolset must not be stored (and thus not exited) since it was never entered.
    """

    class FailingExitToolset(AbstractToolset[None]):
        entered = False
        exited = False

        @property
        def id(self) -> str | None:
            return None  # pragma: no cover

        async def __aenter__(self) -> Self:
            self.entered = True
            return self

        async def __aexit__(self, *args: Any) -> bool | None:
            self.exited = True
            raise RuntimeError('exit failed')

        async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
            return {}  # pragma: no cover

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
        ) -> Any:
            return None  # pragma: no cover

    class TrackedToolset(AbstractToolset[None]):
        entered = False
        exited = False

        @property
        def id(self) -> str | None:
            return None  # pragma: no cover

        async def __aenter__(self) -> Self:  # pragma: no cover
            self.entered = True
            return self

        async def __aexit__(self, *args: Any) -> bool | None:  # pragma: no cover
            self.exited = True
            return None

        async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
            return {}  # pragma: no cover

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
        ) -> Any:
            return None  # pragma: no cover

    first = FailingExitToolset()
    second = TrackedToolset()
    produced = iter([first, second])

    def factory(ctx: RunContext[None]) -> AbstractToolset[None]:
        return next(produced)

    dynamic = DynamicToolset[None](toolset_func=factory)
    run_context = build_run_context(None)

    toolset = await dynamic.for_run(run_context)
    # Suppress the transition failure so we can inspect state afterwards; the
    # outer __aexit__ must not then try to exit the never-entered second toolset.
    with pytest.raises(RuntimeError, match='exit failed'):
        async with toolset:
            await toolset.for_run_step(run_context)
            assert first.entered
            await toolset.for_run_step(run_context)  # raises on old.__aexit__
            pass  # pragma: no cover

    assert first.exited
    assert not second.entered
    assert not second.exited


async def test_dynamic_toolset_empty():
    def no_toolset_func(ctx: RunContext[None]) -> None:
        return None  # pragma: no cover

    original_toolset = DynamicToolset[None](toolset_func=no_toolset_func)

    run_context = build_run_context(None)

    # for_run evaluates the factory; factory returns None so _toolset stays None
    toolset = await original_toolset.for_run(run_context)
    assert isinstance(toolset, DynamicToolset)
    assert toolset._toolset is None  # pyright: ignore[reportPrivateUsage]

    async with toolset:
        tools = await toolset.get_tools(run_context)

        assert tools == {}

        assert toolset._toolset is None  # pyright: ignore[reportPrivateUsage]


def test_dynamic_toolset_id():
    """Test that DynamicToolset can have an id set."""

    def toolset_func(ctx: RunContext[None]) -> FunctionToolset[None]:
        return FunctionToolset()  # pragma: no cover

    # No id by default
    toolset_no_id = DynamicToolset[None](toolset_func=toolset_func)
    assert toolset_no_id.id is None

    # Explicit id
    toolset_with_id = DynamicToolset[None](toolset_func=toolset_func, id='my_dynamic_toolset')
    assert toolset_with_id.id == 'my_dynamic_toolset'


async def test_wrapper_toolsets_delegate_instructions():
    """Test that wrapper toolsets properly delegate instructions calls."""
    base_instructions = 'Follow the base toolset instructions carefully.'
    base_toolset = MockToolsetWithInstructions(instructions=base_instructions)
    ctx = build_run_context(None)

    # Test PrefixedToolset delegation
    prefixed_toolset = base_toolset.prefixed('test')
    assert await prefixed_toolset.get_instructions(ctx) == base_instructions

    # Test FilteredToolset delegation
    def allow_all_filter(ctx: RunContext[Any], tool_def: ToolDefinition) -> bool:
        return True

    assert allow_all_filter(ctx, ToolDefinition(name='test', description='', parameters_json_schema={})) is True
    filtered_toolset = base_toolset.filtered(allow_all_filter)
    assert await filtered_toolset.get_instructions(ctx) == base_instructions

    # Test RenamedToolset delegation
    rename_map = {'old_name': 'new_name'}
    renamed_toolset = base_toolset.renamed(rename_map)
    assert await renamed_toolset.get_instructions(ctx) == base_instructions

    # Test ApprovalRequiredToolset delegation
    approval_toolset = base_toolset.approval_required()
    assert await approval_toolset.get_instructions(ctx) == base_instructions

    # Test PreparedToolset delegation
    async def prepare_func(ctx: RunContext[Any], tools: list[ToolDefinition]) -> list[ToolDefinition]:
        return tools

    assert await prepare_func(ctx, []) == []
    prepared_toolset = base_toolset.prepared(prepare_func)
    assert await prepared_toolset.get_instructions(ctx) == base_instructions


async def test_combined_toolset_instructions():
    """Test that CombinedToolset aggregates instructions from all contained toolsets."""
    instructions1 = 'Instructions from toolset 1.'
    instructions2 = 'Instructions from toolset 2.'

    toolset1 = MockToolsetWithInstructions(instructions=instructions1, id='toolset1')
    toolset2 = MockToolsetWithInstructions(instructions=instructions2, id='toolset2')
    toolset3 = MockToolsetWithInstructions(instructions=None, id='toolset3')  # No instructions

    combined = CombinedToolset([toolset1, toolset2, toolset3])
    ctx = build_run_context(None)

    # CombinedToolset aggregates non-None instructions from all contained toolsets as a list
    instructions = await combined.get_instructions(ctx)
    assert instructions == ['Instructions from toolset 1.', 'Instructions from toolset 2.']


async def test_combined_toolset_instructions_all_none():
    """Test that CombinedToolset returns None when all toolsets have no instructions."""
    toolset1 = MockToolsetWithInstructions(instructions=None, id='toolset1')
    toolset2 = MockToolsetWithInstructions(instructions=None, id='toolset2')

    combined = CombinedToolset([toolset1, toolset2])
    ctx = build_run_context(None)

    instructions = await combined.get_instructions(ctx)
    assert instructions is None


async def test_combined_toolset_instructions_empty():
    """Test that CombinedToolset returns None when no toolsets are provided."""
    combined = CombinedToolset([])
    ctx = build_run_context(None)

    instructions = await combined.get_instructions(ctx)
    assert instructions is None


def test_agent_toolset_decorator_id():
    """Test that @agent.toolset decorator requires explicit id or defaults to None."""
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel())

    @agent.toolset
    def my_tools(ctx: RunContext[None]) -> FunctionToolset[None]:
        return FunctionToolset()  # pragma: no cover

    @agent.toolset(id='custom_id')
    def other_tools(ctx: RunContext[None]) -> FunctionToolset[None]:
        return FunctionToolset()  # pragma: no cover

    # The toolsets are DynamicToolsets with None or explicit ids
    toolsets = agent.toolsets
    assert len(toolsets) == 3  # FunctionToolset for agent tools + 2 dynamic toolsets

    # First is the agent's own FunctionToolset
    assert isinstance(toolsets[0], FunctionToolset)

    # Second toolset without explicit id should have None
    assert isinstance(toolsets[1], DynamicToolset)
    assert toolsets[1].id is None

    # Third toolset should have explicit id
    assert isinstance(toolsets[2], DynamicToolset)
    assert toolsets[2].id == 'custom_id'


async def test_function_toolset_get_instructions_string():
    """FunctionToolset with a string instruction returns it via get_instructions."""
    toolset = FunctionToolset(instructions='Always use tool X for math.')

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result == [InstructionPart(content='Always use tool X for math.', dynamic=False)]


async def test_function_toolset_get_instructions_function():
    """FunctionToolset with a function instruction calls it via get_instructions."""
    toolset = FunctionToolset(instructions=lambda: 'Use search for lookups.')

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result == [InstructionPart(content='Use search for lookups.', dynamic=True)]


async def test_function_toolset_get_instructions_with_ctx():
    """FunctionToolset instruction function can access RunContext."""

    def my_instructions(ctx: RunContext[str]) -> str:
        return f'Deps are: {ctx.deps}'

    toolset = FunctionToolset[str](instructions=my_instructions)

    ctx = build_run_context('hello')
    result = await toolset.get_instructions(ctx)
    assert result == [InstructionPart(content='Deps are: hello', dynamic=True)]


async def test_function_toolset_get_instructions_async():
    """FunctionToolset with an async instruction function works."""

    async def my_instructions() -> str:
        return 'Async instructions here.'

    toolset = FunctionToolset(instructions=my_instructions)

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result == [InstructionPart(content='Async instructions here.', dynamic=True)]


async def test_function_toolset_get_instructions_multiple():
    """FunctionToolset with a sequence of instructions returns them as a list."""
    toolset = FunctionToolset(instructions=['First instruction.', lambda: 'Second instruction.'])

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result == [
        InstructionPart(content='First instruction.', dynamic=False),
        InstructionPart(content='Second instruction.', dynamic=True),
    ]


async def test_function_toolset_get_instructions_none_default():
    """FunctionToolset without instructions returns None."""
    toolset = FunctionToolset()

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result is None


async def test_function_toolset_instructions_decorator():
    """The @toolset.instructions decorator registers instruction functions."""
    toolset = FunctionToolset()

    @toolset.instructions
    def my_instructions() -> str:
        return 'Use tool Y for data processing.'

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result == [InstructionPart(content='Use tool Y for data processing.', dynamic=True)]


async def test_function_toolset_instructions_decorator_with_ctx():
    """The @toolset.instructions decorator works with RunContext."""
    toolset = FunctionToolset[int]()

    @toolset.instructions
    def my_instructions(ctx: RunContext[int]) -> str:
        return f'Dep value: {ctx.deps}'

    ctx = build_run_context(42)
    result = await toolset.get_instructions(ctx)
    assert result == [InstructionPart(content='Dep value: 42', dynamic=True)]


async def test_function_toolset_instructions_decorator_combined_with_constructor():
    """Constructor instructions and decorator instructions are combined."""
    toolset = FunctionToolset(instructions='From constructor.')

    @toolset.instructions
    def extra() -> str:
        return 'From decorator.'

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result == [
        InstructionPart(content='From constructor.', dynamic=False),
        InstructionPart(content='From decorator.', dynamic=True),
    ]


async def test_function_toolset_instructions_none_filtered():
    """Instructions returning None are excluded."""
    toolset = FunctionToolset(instructions=[lambda: None, 'Only this.'])

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result == [InstructionPart(content='Only this.', dynamic=False)]


async def test_function_toolset_empty_string_instructions():
    """Empty string instructions are filtered out, returning None."""
    toolset = FunctionToolset(instructions='')

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result is None


async def test_function_toolset_whitespace_only_instructions():
    """Whitespace-only instructions are filtered out, returning None."""
    toolset = FunctionToolset(instructions='   \n\n  ')

    ctx = build_run_context(None)
    result = await toolset.get_instructions(ctx)
    assert result is None


async def test_wrapper_toolset_passes_through_instructions():
    """WrapperToolset delegates get_instructions to wrapped toolset."""
    inner = FunctionToolset(instructions='Inner instructions.')
    wrapped = inner.prefixed('my')

    ctx = build_run_context(None)
    result = await wrapped.get_instructions(ctx)
    assert result == [InstructionPart(content='Inner instructions.', dynamic=False)]


async def test_combined_toolset_aggregates_instructions():
    """CombinedToolset gathers instructions from all children."""
    ts1 = FunctionToolset(instructions='Toolset 1 instructions.')
    ts2 = FunctionToolset(instructions='Toolset 2 instructions.')
    combined = CombinedToolset([ts1, ts2])

    ctx = build_run_context(None)
    result = await combined.get_instructions(ctx)
    assert result == [
        InstructionPart(content='Toolset 1 instructions.', dynamic=False),
        InstructionPart(content='Toolset 2 instructions.', dynamic=False),
    ]


async def test_combined_toolset_skips_none_instructions():
    """CombinedToolset skips toolsets that return None for instructions."""
    ts1 = FunctionToolset(instructions='Only from ts1.')
    ts2 = FunctionToolset()  # No instructions
    combined = CombinedToolset([ts1, ts2])

    ctx = build_run_context(None)
    result = await combined.get_instructions(ctx)
    assert result == [InstructionPart(content='Only from ts1.', dynamic=False)]


async def test_combined_toolset_all_none_returns_none():
    """CombinedToolset returns None when all children return None."""
    ts1 = FunctionToolset()
    ts2 = FunctionToolset()
    combined = CombinedToolset([ts1, ts2])

    ctx = build_run_context(None)
    result = await combined.get_instructions(ctx)
    assert result is None


async def test_combined_toolset_with_nested_list_instructions():
    """CombinedToolset flattens list[str] results from child CombinedToolsets (covers combined.py list branch)."""
    ts1 = FunctionToolset(instructions='Instruction A.')
    ts2 = FunctionToolset(instructions='Instruction B.')
    inner = CombinedToolset([ts1, ts2])  # returns list[str]

    ts3 = FunctionToolset(instructions='Instruction C.')
    outer = CombinedToolset([inner, ts3])
    ctx = build_run_context(None)

    result = await outer.get_instructions(ctx)
    assert result == [
        InstructionPart(content='Instruction A.', dynamic=False),
        InstructionPart(content='Instruction B.', dynamic=False),
        InstructionPart(content='Instruction C.', dynamic=False),
    ]


async def test_combined_toolset_cancels_siblings_on_get_tools_failure():
    """When one child's get_tools fails, siblings are cancelled instead of leaking as orphan tasks."""
    sibling_completed = False

    class FailingToolset(WrapperToolset[Any]):
        async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
            raise RuntimeError('boom')

    class SlowToolset(WrapperToolset[Any]):
        async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
            nonlocal sibling_completed
            await anyio.sleep(0.1)
            sibling_completed = True  # pragma: no cover
            return await self.wrapped.get_tools(ctx)  # pragma: no cover

    inner = FunctionToolset[None]()
    combined = CombinedToolset([FailingToolset(inner), SlowToolset(inner)])
    ctx = build_run_context(None)

    with pytest.raises(RuntimeError, match='boom'):
        await combined.get_tools(ctx)

    await anyio.sleep(0.2)
    assert sibling_completed is False


async def test_combined_toolset_get_tools_preserves_exception_cause():
    """Unwrapping the single-failure exception must preserve the original `__cause__` chain."""
    original = ValueError('underlying')

    class FailingToolset(WrapperToolset[Any]):
        async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
            raise RuntimeError('wrapper') from original

    inner = FunctionToolset[None]()
    combined = CombinedToolset([FailingToolset(inner)])
    ctx = build_run_context(None)

    with pytest.raises(RuntimeError, match='wrapper') as exc_info:
        await combined.get_tools(ctx)

    assert exc_info.value.__cause__ is original


async def test_combined_toolset_get_tools_raises_group_on_multiple_failures():
    """When multiple children fail concurrently, their errors surface as an ExceptionGroup."""

    @dataclass
    class RaisingToolset(WrapperToolset[Any]):
        message: str = ''

        async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
            await anyio.sleep(0)
            raise RuntimeError(self.message)

    inner = FunctionToolset[None]()
    combined = CombinedToolset(
        [RaisingToolset(wrapped=inner, message='first'), RaisingToolset(wrapped=inner, message='second')]
    )
    ctx = build_run_context(None)

    with pytest.raises(BaseExceptionGroup) as exc_info:
        await combined.get_tools(ctx)

    messages = {str(e) for e in exc_info.value.exceptions}
    assert messages == {'first', 'second'}


async def test_dynamic_toolset_instructions_before_resolution():
    """DynamicToolset returns None for instructions before get_tools resolves it."""
    dynamic = DynamicToolset(lambda ctx: FunctionToolset(instructions='Dynamic instructions.'))

    ctx = build_run_context(None)
    # Before get_tools is called, _toolset is None
    result = await dynamic.get_instructions(ctx)
    assert result is None


async def test_dynamic_toolset_instructions_after_resolution():
    """DynamicToolset delegates instructions after for_run_step resolves it."""

    def make_toolset(ctx: RunContext[None]) -> FunctionToolset[None]:
        return FunctionToolset[None](instructions='Dynamic instructions.')

    dynamic = DynamicToolset(make_toolset)

    ctx = build_run_context(None)
    # for_run_step triggers factory resolution for per_run_step=True
    await dynamic.for_run_step(ctx)
    result = await dynamic.get_instructions(ctx)
    assert result == [InstructionPart(content='Dynamic instructions.', dynamic=False)]


async def test_toolset_instructions_in_agent():
    """Toolset instructions appear in the model request when added to an agent."""
    from pydantic_ai import Agent

    toolset = FunctionToolset(instructions='Always use my_tool correctly.')

    @toolset.tool_plain
    def my_tool() -> str:
        """A simple tool."""
        return 'done'

    agent = Agent(TestModel(), toolsets=[toolset])
    result = await agent.run('Hello')
    first_message = result.all_messages()[0]
    assert first_message.instructions == 'Always use my_tool correctly.'  # type: ignore[union-attr]


async def test_dynamic_toolset_instructions_on_first_request():
    """Instructions from a DynamicToolset are present on the very first model request."""
    from pydantic_ai import Agent

    def make_toolset(ctx: RunContext[None]) -> FunctionToolset[None]:
        ts = FunctionToolset[None](instructions='Dynamic tool instructions.')

        @ts.tool_plain
        def my_dynamic_tool() -> str:
            """A tool inside the dynamic toolset."""
            return 'done'

        return ts

    agent = Agent(TestModel(), toolsets=[DynamicToolset(make_toolset)])
    result = await agent.run('Hello')
    first_message = result.all_messages()[0]
    assert first_message.instructions == 'Dynamic tool instructions.'  # type: ignore[union-attr]


async def test_resume_without_prompt_dynamic_toolset_instructions_resolve_once_for_request_step():
    """Resuming from a trailing ModelResponse should resolve dynamic toolsets only once for the next request step."""
    run_steps: list[int] = []

    def make_toolset(ctx: RunContext[None]) -> FunctionToolset[None]:
        run_steps.append(ctx.run_step)
        return FunctionToolset[None](instructions=f'Dynamic instructions at step {ctx.run_step}.')

    agent = Agent(TestModel(custom_output_text='done'), toolsets=[DynamicToolset(make_toolset)])
    result = await agent.run(message_history=[ModelResponse(parts=[TextPart(content='previous')])])

    # The resume pre-check and request preparation should use the same run_step.
    assert run_steps == [1]

    requests = [m for m in result.all_messages() if isinstance(m, ModelRequest)]
    assert requests[-1].instructions == 'Dynamic instructions at step 1.'


async def test_resume_without_prompt_dynamic_toolset_with_tool_calls_resolve_once_for_request_step():
    """Resuming from a trailing ModelResponse with ToolCallParts exercises the _handle_tool_calls path.

    This is the more common resume scenario and ensures dynamic toolsets are resolved only once even
    when the code path goes through _handle_tool_calls (which calls for_run_step).
    """
    run_steps: list[int] = []

    def make_toolset(ctx: RunContext[None]) -> FunctionToolset[None]:
        run_steps.append(ctx.run_step)
        toolset = FunctionToolset[None](instructions=f'Dynamic instructions at step {ctx.run_step}.')

        @toolset.tool_plain
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f'Hello {name}!'

        return toolset

    agent = Agent(TestModel(custom_output_text='done'), toolsets=[DynamicToolset(make_toolset)])
    result = await agent.run(
        message_history=[ModelResponse(parts=[ToolCallPart(tool_name='greet', args={'name': 'Alice'})])]
    )

    # The toolset factory is evaluated multiple times:
    # - step 1: UserPromptNode pre-check (aligned with upcoming request step)
    # - step 0: CallToolsNode._handle_tool_calls (current run_step before increment)
    # - step 1: ModelRequestNode._prepare_request (after run_step increment)
    # The important thing is that the first and last evaluations use step 1.
    assert run_steps[0] == 1
    assert run_steps[-1] == 1

    requests = [m for m in result.all_messages() if isinstance(m, ModelRequest)]
    # The last request should have instructions prepared at step 1
    assert requests[-1].instructions == 'Dynamic instructions at step 1.'


async def test_toolset_instructions_combined_with_agent_instructions():
    """Toolset instructions are appended after agent-level instructions."""
    from pydantic_ai import Agent

    toolset = FunctionToolset(instructions='Use search for lookups.')

    @toolset.tool_plain
    def search() -> str:
        """Search for information."""
        return 'results'

    agent = Agent(TestModel(), instructions='You are a helpful assistant.', toolsets=[toolset])
    result = await agent.run('Hello')
    first_message = result.all_messages()[0]
    assert first_message.instructions == 'You are a helpful assistant.\n\nUse search for lookups.'  # type: ignore[union-attr]


async def test_multiple_toolset_instructions_in_agent():
    """Multiple toolsets with instructions are all included."""
    from pydantic_ai import Agent

    ts1 = FunctionToolset(instructions='Use calculator for math.')

    @ts1.tool_plain
    def calculator() -> str:
        """Evaluate a math expression."""
        return '4'

    ts2 = FunctionToolset(instructions='Use search for lookups.')

    @ts2.tool_plain
    def search() -> str:
        """Search for information."""
        return 'results'

    agent = Agent(TestModel(), toolsets=[ts1, ts2])
    result = await agent.run('Hello')
    first_message = result.all_messages()[0]
    assert first_message.instructions == 'Use calculator for math.\n\nUse search for lookups.'  # type: ignore[union-attr]


async def test_toolset_instructions_alone_satisfy_validation():
    """Toolset instructions alone (no user prompt, no agent instructions, no history) are sufficient to run."""
    from pydantic_ai import Agent

    toolset = FunctionToolset(instructions='Always use my_tool correctly.')

    @toolset.tool_plain
    def my_tool() -> str:
        """A simple tool."""
        return 'done'

    agent = Agent(TestModel(), toolsets=[toolset])
    result = await agent.run()
    first_message = result.all_messages()[0]
    assert first_message.instructions == 'Always use my_tool correctly.'  # type: ignore[union-attr]
    assert first_message.parts == []


async def test_no_input_raises_without_toolset_instructions():
    """Without any prompt, instructions, or history, the agent raises UserError."""
    from pydantic_ai import Agent

    agent = Agent(TestModel())
    with pytest.raises(UserError, match='No message history, user prompt, or instructions provided'):
        await agent.run()


def test_tool_without_runctx_raises_warning():
    toolset = FunctionToolset()
    with pytest.warns(
        DeprecationWarning, match='Passing a function without `RunContext` to `FunctionToolset.tool\\(\\)`'
    ):

        @toolset.tool  # type: ignore[arg-type]  # pragma: no cover
        def add(x: int):
            return x + 1

        @toolset.tool(retries=2)  # type: ignore[arg-type]  # pragma: no cover
        def sub(x: int):
            return x - 1

    assert 'add' in toolset.tools
    assert 'sub' in toolset.tools


class StatefulToolset(AbstractToolset[None]):
    """A custom stateful toolset for testing for_run/for_run_step."""

    def __init__(self, *, call_count: int = 0, id: str | None = 'stateful'):
        self.call_count = call_count
        self._id = id

    @property
    def id(self) -> str | None:
        return self._id  # pragma: no cover

    async def for_run(self, ctx: RunContext[None]) -> AbstractToolset[None]:
        return StatefulToolset(call_count=0, id=self._id)

    async def for_run_step(self, ctx: RunContext[None]) -> AbstractToolset[None]:
        return StatefulToolset(call_count=self.call_count + 1, id=self._id)

    async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
        return {}  # pragma: no cover

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
    ) -> Any:
        self.call_count += 1  # pragma: no cover
        return self.call_count  # pragma: no cover


async def test_for_run_returns_fresh_instance():
    """Custom stateful toolset with for_run returning fresh instance."""
    original = StatefulToolset(call_count=5)
    ctx = build_run_context(None)

    run_toolset = await original.for_run(ctx)
    assert run_toolset is not original
    assert isinstance(run_toolset, StatefulToolset)
    assert run_toolset.call_count == 0
    assert original.call_count == 5  # original unchanged


async def test_for_run_step_default_returns_self():
    """Default for_run_step returns self for toolsets that don't override it."""
    toolset = FunctionToolset()
    ctx = build_run_context(None)

    step_toolset = await toolset.for_run_step(ctx)
    assert step_toolset is toolset


async def test_for_run_step_returns_new_instance():
    """StatefulToolset.for_run_step returns a new instance with bumped step counter."""
    toolset = StatefulToolset(call_count=3)
    ctx = build_run_context(None)

    step_toolset = await toolset.for_run_step(ctx)
    assert step_toolset is not toolset
    assert isinstance(step_toolset, StatefulToolset)
    assert step_toolset.call_count == 4
    assert toolset.call_count == 3  # original unchanged


async def test_wrapper_propagates_for_run():
    """Wrapper toolsets correctly propagate for_run to the wrapped toolset."""
    inner = StatefulToolset(call_count=10)
    wrapper = WrapperToolset(inner)
    ctx = build_run_context(None)

    run_wrapper = await wrapper.for_run(ctx)
    assert run_wrapper is not wrapper  # different because inner changed
    assert isinstance(run_wrapper, WrapperToolset)
    inner_after = run_wrapper.wrapped
    assert isinstance(inner_after, StatefulToolset)
    assert inner_after.call_count == 0  # fresh


async def test_wrapper_propagates_for_run_no_change():
    """Wrapper returns self when wrapped toolset returns self from for_run."""
    inner = FunctionToolset()  # FunctionToolset.for_run returns self
    wrapper = WrapperToolset(inner)
    ctx = build_run_context(None)

    run_wrapper = await wrapper.for_run(ctx)
    assert run_wrapper is wrapper


async def test_combined_propagates_for_run():
    """CombinedToolset propagates for_run to all children."""
    stateful = StatefulToolset(call_count=7)
    static = FunctionToolset()
    combined = CombinedToolset([stateful, static])
    ctx = build_run_context(None)

    run_combined = await combined.for_run(ctx)
    assert run_combined is not combined
    assert isinstance(run_combined, CombinedToolset)
    assert isinstance(run_combined.toolsets[0], StatefulToolset)
    assert run_combined.toolsets[0].call_count == 0
    assert run_combined.toolsets[1] is static  # unchanged


async def test_combined_for_run_always_fresh():
    """CombinedToolset.for_run always returns a new instance for per-run isolation."""
    static1 = FunctionToolset(id='a')
    static2 = FunctionToolset(id='b')
    combined = CombinedToolset([static1, static2])
    ctx = build_run_context(None)

    run_combined = await combined.for_run(ctx)
    assert run_combined is not combined
    assert isinstance(run_combined, CombinedToolset)
    # Children are unchanged (their for_run returns self)
    assert run_combined.toolsets[0] is static1
    assert run_combined.toolsets[1] is static2


async def test_wrapper_propagates_for_run_step_no_change():
    """Wrapper returns self when wrapped toolset returns self from for_run_step."""
    inner = FunctionToolset()  # FunctionToolset.for_run_step returns self
    wrapper = WrapperToolset(inner)
    ctx = build_run_context(None)

    step_wrapper = await wrapper.for_run_step(ctx)
    assert step_wrapper is wrapper


async def test_wrapper_propagates_for_run_step():
    """Wrapper creates new wrapper when wrapped toolset returns new instance from for_run_step."""
    inner = StatefulToolset(call_count=10)
    wrapper = WrapperToolset(inner)
    ctx = build_run_context(None)

    step_wrapper = await wrapper.for_run_step(ctx)
    assert step_wrapper is not wrapper
    assert isinstance(step_wrapper, WrapperToolset)
    inner_after = step_wrapper.wrapped
    assert isinstance(inner_after, StatefulToolset)
    assert inner_after.call_count == 11  # bumped by for_run_step


async def test_combined_propagates_for_run_step_no_change():
    """CombinedToolset returns self when no children change from for_run_step."""
    static1 = FunctionToolset(id='a')
    static2 = FunctionToolset(id='b')
    combined = CombinedToolset([static1, static2])
    ctx = build_run_context(None)

    step_combined = await combined.for_run_step(ctx)
    assert step_combined is combined


async def test_combined_propagates_for_run_step():
    """CombinedToolset creates new combined when a child returns new instance from for_run_step."""
    stateful = StatefulToolset(call_count=7)
    static = FunctionToolset()
    combined = CombinedToolset([stateful, static])
    ctx = build_run_context(None)

    step_combined = await combined.for_run_step(ctx)
    assert step_combined is not combined
    assert isinstance(step_combined, CombinedToolset)
    assert isinstance(step_combined.toolsets[0], StatefulToolset)
    assert step_combined.toolsets[0].call_count == 8  # bumped by for_run_step
    assert step_combined.toolsets[1] is static  # unchanged


async def test_dynamic_toolset_for_run_step_manages_transitions():
    """DynamicToolset with per_run_step=True manages internal transitions via for_run_step."""
    call_count = 0

    def factory(ctx: RunContext[None]) -> FunctionToolset[None]:
        nonlocal call_count
        call_count += 1
        return FunctionToolset(id=f'step-{call_count}')

    original = DynamicToolset[None](toolset_func=factory, per_run_step=True)
    ctx = build_run_context(None)

    # for_run creates a fresh copy without evaluating factory
    run_toolset = await original.for_run(ctx)
    assert isinstance(run_toolset, DynamicToolset)
    assert run_toolset._toolset is None  # pyright: ignore[reportPrivateUsage]
    assert call_count == 0

    async with run_toolset:
        # for_run_step evaluates the factory
        step1 = await run_toolset.for_run_step(ctx)
        assert step1 is run_toolset  # returns self after in-place update
        assert call_count == 1
        assert run_toolset._toolset is not None  # pyright: ignore[reportPrivateUsage]

        # Second for_run_step re-evaluates (new toolset each time)
        step2 = await run_toolset.for_run_step(ctx)
        assert step2 is run_toolset
        assert call_count == 2

    assert original._toolset is None  # pyright: ignore[reportPrivateUsage]


async def test_dynamic_toolset_for_run_step_same_instance_skips_transition():
    """DynamicToolset skips transition when factory returns the same instance."""
    stable_toolset = FunctionToolset[None](id='stable')

    def factory(ctx: RunContext[None]) -> FunctionToolset[None]:
        return stable_toolset

    original = DynamicToolset[None](toolset_func=factory, per_run_step=True)
    ctx = build_run_context(None)

    run_toolset = await original.for_run(ctx)
    assert isinstance(run_toolset, DynamicToolset)

    async with run_toolset:
        # First step: evaluates factory, sets _toolset
        step1 = await run_toolset.for_run_step(ctx)
        assert step1 is run_toolset
        assert run_toolset._toolset is stable_toolset  # pyright: ignore[reportPrivateUsage]

        # Second step: factory returns same instance, early return without transition
        step2 = await run_toolset.for_run_step(ctx)
        assert step2 is run_toolset
        assert run_toolset._toolset is stable_toolset  # pyright: ignore[reportPrivateUsage]


async def test_dynamic_toolset_for_run_step_factory_returns_none():
    """DynamicToolset handles factory returning None after previously returning a toolset."""
    call_count = 0

    def factory(ctx: RunContext[None]) -> FunctionToolset[None] | None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return FunctionToolset[None](id='first')
        return None

    original = DynamicToolset[None](toolset_func=factory, per_run_step=True)
    ctx = build_run_context(None)

    run_toolset = await original.for_run(ctx)
    assert isinstance(run_toolset, DynamicToolset)

    async with run_toolset:
        # First step: factory returns a toolset
        await run_toolset.for_run_step(ctx)
        assert run_toolset._toolset is not None  # pyright: ignore[reportPrivateUsage]

        # Second step: factory returns None — old toolset exited, new is None
        await run_toolset.for_run_step(ctx)
        assert run_toolset._toolset is None  # pyright: ignore[reportPrivateUsage]

        # Tools should be empty when _toolset is None
        tools = await run_toolset.get_tools(ctx)
        assert tools == {}


async def test_dynamic_toolset_per_run_step_false_for_run_evaluates():
    """DynamicToolset with per_run_step=False evaluates factory in for_run."""
    call_count = 0

    def factory(ctx: RunContext[None]) -> FunctionToolset[None]:
        nonlocal call_count
        call_count += 1
        return FunctionToolset()

    original = DynamicToolset[None](toolset_func=factory, per_run_step=False)
    ctx = build_run_context(None)

    run_toolset = await original.for_run(ctx)
    assert isinstance(run_toolset, DynamicToolset)
    assert call_count == 1
    assert run_toolset._toolset is not None  # pyright: ignore[reportPrivateUsage]

    # for_run_step returns self (no re-evaluation)
    step_toolset = await run_toolset.for_run_step(ctx)
    assert step_toolset is run_toolset
    assert call_count == 1


async def test_concurrent_runs_dont_share_state():
    """Multiple concurrent runs don't share state on stateful toolsets."""
    import asyncio

    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    call_counts: list[int] = []

    class CountingToolset(AbstractToolset[None]):
        def __init__(self) -> None:
            self.count = 0

        @property
        def id(self) -> str | None:
            return 'counting'  # pragma: no cover

        async def for_run(self, ctx: RunContext[None]) -> AbstractToolset[None]:
            return CountingToolset()

        async def get_tools(self, ctx: RunContext[None]) -> dict[str, ToolsetTool[None]]:
            self.count += 1
            call_counts.append(self.count)
            return {}

        async def call_tool(
            self, name: str, tool_args: dict[str, Any], ctx: RunContext[None], tool: ToolsetTool[None]
        ) -> Any:
            pass  # pragma: no cover

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('Done')])

    agent = Agent(FunctionModel(respond), toolsets=[CountingToolset()])

    # Run two concurrent agent runs
    results = await asyncio.gather(agent.run('Hello'), agent.run('World'))

    assert results[0].output == 'Done'
    assert results[1].output == 'Done'
    # Each run should have its own count (1), not share state (1, 2)
    assert all(c == 1 for c in call_counts)


def test_include_return_schemas_toolset():
    """IncludeReturnSchemasToolset sets include_return_schema=True on wrapped tools."""

    def my_tool(x: int) -> int:
        return x

    toolset = FunctionToolset(tools=[my_tool])
    test_model = TestModel()
    agent = Agent(test_model, toolsets=[toolset.include_return_schemas()])
    agent.run_sync('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    td = next(td for td in params.function_tools if td.name == 'my_tool')
    assert td.include_return_schema is True
    assert 'Return schema' in (td.description or '')


def test_set_metadata_toolset():
    """SetMetadataToolset merges metadata onto all wrapped tools."""

    def my_tool(x: int) -> int:
        return x

    toolset = FunctionToolset(tools=[my_tool])
    test_model = TestModel()
    agent = Agent(test_model, toolsets=[toolset.with_metadata(code_mode=True, priority=1)])
    agent.run_sync('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    td = next(td for td in params.function_tools if td.name == 'my_tool')
    assert td.metadata is not None
    assert td.metadata['code_mode'] is True
    assert td.metadata['priority'] == 1


async def test_filtered_toolset_async_filter():
    """FilteredToolset supports async filter functions."""

    def tool_a(x: int) -> int:
        return x

    def tool_b(x: str) -> str:
        return x  # pragma: no cover

    async def async_filter(ctx: RunContext, td: ToolDefinition) -> bool:
        return td.name == 'tool_a'

    toolset = FunctionToolset(tools=[tool_a, tool_b])
    test_model = TestModel()
    agent = Agent(test_model, toolsets=[toolset.filtered(async_filter)])
    await agent.run('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    tool_names = [td.name for td in params.function_tools]
    assert tool_names == ['tool_a']


def test_set_tool_metadata_capability():
    """SetToolMetadata capability merges metadata onto selected tools."""
    from pydantic_ai.capabilities import SetToolMetadata

    def tool_a(x: int) -> int:
        return x

    def tool_b(x: str) -> str:
        return x

    test_model = TestModel()
    agent = Agent(
        test_model,
        tools=[tool_a, tool_b],
        capabilities=[SetToolMetadata(tools=['tool_a'], code_mode=True)],
    )
    agent.run_sync('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    td_a = next(td for td in params.function_tools if td.name == 'tool_a')
    td_b = next(td for td in params.function_tools if td.name == 'tool_b')
    assert td_a.metadata is not None
    assert td_a.metadata['code_mode'] is True
    # tool_b should not have the metadata
    assert td_b.metadata is None or 'code_mode' not in td_b.metadata


def test_set_tool_metadata_capability_with_async_selector():
    """SetToolMetadata with async callable selector."""
    from pydantic_ai.capabilities import SetToolMetadata

    def tool_a(x: int) -> int:
        return x

    def tool_b(x: str) -> str:
        return x

    async def only_tool_a(ctx: RunContext, td: ToolDefinition) -> bool:
        return td.name == 'tool_a'

    test_model = TestModel()
    agent = Agent(
        test_model,
        tools=[tool_a, tool_b],
        capabilities=[SetToolMetadata(tools=only_tool_a, tagged=True)],
    )
    agent.run_sync('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    td_a = next(td for td in params.function_tools if td.name == 'tool_a')
    td_b = next(td for td in params.function_tools if td.name == 'tool_b')
    assert td_a.metadata is not None
    assert td_a.metadata['tagged'] is True
    assert td_b.metadata is None or 'tagged' not in td_b.metadata


def test_set_tool_metadata_capability_with_bare_string_selector():
    """SetToolMetadata with a bare string selector matches by exact name, not substring."""
    from pydantic_ai.capabilities import SetToolMetadata

    def my_tool(x: int) -> int:
        return x

    def my(x: str) -> str:
        return x

    test_model = TestModel()
    agent = Agent(
        test_model,
        tools=[my_tool, my],
        capabilities=[SetToolMetadata(tools='my_tool', tagged=True)],
    )
    agent.run_sync('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    td_my_tool = next(td for td in params.function_tools if td.name == 'my_tool')
    td_my = next(td for td in params.function_tools if td.name == 'my')
    assert td_my_tool.metadata is not None
    assert td_my_tool.metadata['tagged'] is True
    # 'my' should NOT match — bare string does exact match, not substring
    assert td_my.metadata is None or 'tagged' not in td_my.metadata


def test_set_tool_metadata_capability_with_sync_callable_selector():
    """SetToolMetadata with sync callable selector."""
    from pydantic_ai.capabilities import SetToolMetadata

    def tool_a(x: int) -> int:
        return x

    def tool_b(x: str) -> str:
        return x

    test_model = TestModel()
    agent = Agent(
        test_model,
        tools=[tool_a, tool_b],
        capabilities=[SetToolMetadata(tools=lambda ctx, td: td.name == 'tool_a', flagged=True)],
    )
    agent.run_sync('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    td_a = next(td for td in params.function_tools if td.name == 'tool_a')
    td_b = next(td for td in params.function_tools if td.name == 'tool_b')
    assert td_a.metadata is not None
    assert td_a.metadata['flagged'] is True
    assert td_b.metadata is None or 'flagged' not in td_b.metadata


def test_set_tool_metadata_capability_with_nested_dict_selector():
    """SetToolMetadata with nested dict selector exercises deep metadata matching."""
    from pydantic_ai.capabilities import SetToolMetadata
    from pydantic_ai.tools import Tool

    def tool_a(x: int) -> int:
        return x

    def tool_b(x: str) -> str:
        return x

    def tool_c(x: float) -> float:
        return x

    test_model = TestModel()
    agent = Agent(
        test_model,
        tools=[
            Tool(tool_a, metadata={'config': {'env': 'prod', 'region': 'us'}}),
            Tool(tool_b, metadata={'config': {'env': 'staging'}}),
            Tool(tool_c, metadata={'other': 'value'}),  # missing 'config' key entirely
        ],
        capabilities=[SetToolMetadata(tools={'config': {'env': 'prod'}}, verified=True)],
    )
    agent.run_sync('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    td_a = next(td for td in params.function_tools if td.name == 'tool_a')
    td_b = next(td for td in params.function_tools if td.name == 'tool_b')
    td_c = next(td for td in params.function_tools if td.name == 'tool_c')
    # tool_a matches: config.env == 'prod' (deep inclusion, extra 'region' key is fine)
    assert td_a.metadata is not None
    assert td_a.metadata['verified'] is True
    # tool_b doesn't match: config.env == 'staging'
    assert td_b.metadata is not None
    assert 'verified' not in td_b.metadata
    # tool_c doesn't match: 'config' key missing entirely
    assert td_c.metadata is not None
    assert 'verified' not in td_c.metadata


def test_set_tool_metadata_capability_with_dict_selector():
    """SetToolMetadata with dict selector matches tools by metadata."""
    from pydantic_ai.capabilities import SetToolMetadata
    from pydantic_ai.tools import Tool

    def tool_a(x: int) -> int:
        return x

    def tool_b(x: str) -> str:
        return x

    test_model = TestModel()
    agent = Agent(
        test_model,
        tools=[
            Tool(tool_a, metadata={'env': 'prod'}),
            Tool(tool_b, metadata={'env': 'staging'}),
        ],
        capabilities=[SetToolMetadata(tools={'env': 'prod'}, audited=True)],
    )
    agent.run_sync('test')

    params = test_model.last_model_request_parameters
    assert params is not None
    td_a = next(td for td in params.function_tools if td.name == 'tool_a')
    td_b = next(td for td in params.function_tools if td.name == 'tool_b')
    # tool_a matched the dict selector, gets audited=True merged
    assert td_a.metadata is not None
    assert td_a.metadata['audited'] is True
    assert td_a.metadata['env'] == 'prod'
    # tool_b didn't match
    assert td_b.metadata is not None
    assert 'audited' not in td_b.metadata


async def test_custom_toolset_returning_plain_str_instructions():
    """A custom AbstractToolset returning a plain str from get_instructions is treated as dynamic."""
    from pydantic_ai import Agent

    class PlainStrInstructionsToolset(FunctionToolset[None]):
        """A toolset that overrides get_instructions to return a plain str instead of InstructionPart."""

        async def get_instructions(self, ctx: RunContext[None]) -> str | None:  # type: ignore[override]
            return 'Custom toolset instruction.'

    agent = Agent(TestModel(), toolsets=[PlainStrInstructionsToolset()])
    result = await agent.run('Hello')
    first_message = result.all_messages()[0]
    assert first_message.instructions == 'Custom toolset instruction.'  # type: ignore[union-attr]


async def test_toolset_empty_instructions_filtered():
    """Empty and whitespace-only instructions from toolsets are filtered out."""
    from pydantic_ai import Agent
    from pydantic_ai.messages import InstructionPart

    class EmptyInstructionsToolset(FunctionToolset[None]):
        async def get_instructions(self, ctx: RunContext[None]) -> list[str | InstructionPart] | None:  # type: ignore[override]
            return [
                '',
                '   ',
                InstructionPart(content='', dynamic=True),
                InstructionPart(content='  ', dynamic=False),
                'valid instruction',
                InstructionPart(content='another valid', dynamic=True),
            ]

    agent = Agent(TestModel(), toolsets=[EmptyInstructionsToolset()])
    result = await agent.run('Hello')
    first_message = result.all_messages()[0]
    assert first_message.instructions == 'valid instruction\n\nanother valid'  # type: ignore[union-attr]

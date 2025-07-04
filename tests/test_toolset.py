from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass, field, replace
from typing import Any, Callable, TypeVar

import pytest
from inline_snapshot import snapshot

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets.combined import CombinedToolset
from pydantic_ai.toolsets.filtered import FilteredToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.toolsets.prefixed import PrefixedToolset
from pydantic_ai.toolsets.prepared import PreparedToolset
from pydantic_ai.toolsets.processed import ProcessedToolset
from pydantic_ai.usage import Usage

pytestmark = pytest.mark.anyio

T = TypeVar('T')


def build_run_context(deps: T) -> RunContext[T]:
    return RunContext(
        deps=deps,
        model=TestModel(),
        usage=Usage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


async def test_function_toolset_prepare_for_run():
    @dataclass
    class PrefixDeps:
        prefix: str | None = None

    context = build_run_context(PrefixDeps())
    toolset = FunctionToolset[PrefixDeps]()

    async def prepare_add_prefix(ctx: RunContext[PrefixDeps], tool_def: ToolDefinition) -> ToolDefinition | None:
        if ctx.deps.prefix is None:
            return tool_def

        return replace(tool_def, name=f'{ctx.deps.prefix}_{tool_def.name}')

    @toolset.tool(prepare=prepare_add_prefix)
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    assert toolset.tool_names == snapshot(['add'])
    assert toolset.tool_defs == snapshot(
        [
            ToolDefinition(
                name='add',
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
    assert await toolset.call_tool(context, 'add', {'a': 1, 'b': 2}) == 3

    no_prefix_context = build_run_context(PrefixDeps())
    no_prefix_toolset = await toolset.prepare_for_run(no_prefix_context)
    assert no_prefix_toolset.tool_names == toolset.tool_names
    assert no_prefix_toolset.tool_defs == toolset.tool_defs
    assert await no_prefix_toolset.call_tool(no_prefix_context, 'add', {'a': 1, 'b': 2}) == 3

    foo_context = build_run_context(PrefixDeps(prefix='foo'))
    foo_toolset = await toolset.prepare_for_run(foo_context)
    assert foo_toolset.tool_names == snapshot(['foo_add'])
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
    assert await foo_toolset.call_tool(foo_context, 'foo_add', {'a': 1, 'b': 2}) == 3

    @toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b  # pragma: lax no cover

    assert foo_toolset.tool_names == snapshot(['foo_add'])

    bar_context = build_run_context(PrefixDeps(prefix='bar'))
    bar_toolset = await toolset.prepare_for_run(bar_context)
    assert bar_toolset.tool_names == snapshot(['bar_add', 'subtract'])
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
    assert await bar_toolset.call_tool(bar_context, 'bar_add', {'a': 1, 'b': 2}) == 3

    bar_foo_toolset = await foo_toolset.prepare_for_run(bar_context)
    assert bar_foo_toolset == bar_toolset


async def test_prepared_toolset_user_error_add_new_tools():
    """Test that PreparedToolset raises UserError when prepare function tries to add new tools."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

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

    with pytest.raises(UserError, match='Prepare function is not allowed to change tool names or add new tools.'):
        await prepared_toolset.prepare_for_run(context)


async def test_prepared_toolset_user_error_change_tool_names():
    """Test that PreparedToolset raises UserError when prepare function tries to change tool names."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

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

    with pytest.raises(UserError, match='Prepare function is not allowed to change tool names or add new tools.'):
        await prepared_toolset.prepare_for_run(context)


async def test_prepared_toolset_allows_removing_tools():
    """Test that PreparedToolset allows removing tools from the original set."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @base_toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b

    @base_toolset.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    async def prepare_remove_tools(ctx: RunContext[None], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # Remove the 'subtract' tool, keep 'add' and 'multiply'
        return [tool_def for tool_def in tool_defs if tool_def.name != 'subtract']

    prepared_toolset = PreparedToolset(base_toolset, prepare_remove_tools)

    # This should not raise an error
    run_toolset = await prepared_toolset.prepare_for_run(context)

    # Verify that only 'add' and 'multiply' tools are available
    assert set(run_toolset.tool_names) == {'add', 'multiply'}
    assert len(run_toolset.tool_defs) == 2

    # Verify that the tools still work
    assert await run_toolset.call_tool(context, 'add', {'a': 5, 'b': 3}) == 8
    assert await run_toolset.call_tool(context, 'multiply', {'a': 4, 'b': 2}) == 8


async def test_prefixed_toolset_tool_defs():
    """Test that PrefixedToolset correctly prefixes tool definitions."""
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @base_toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b

    prefixed_toolset = PrefixedToolset(base_toolset, 'math')

    # Check that tool names are prefixed
    assert prefixed_toolset.tool_names == ['math_add', 'math_subtract']

    # Check that tool definitions have prefixed names
    tool_defs = prefixed_toolset.tool_defs
    assert len(tool_defs) == 2

    add_def = next(td for td in tool_defs if td.name == 'math_add')
    subtract_def = next(td for td in tool_defs if td.name == 'math_subtract')

    assert add_def.name == 'math_add'
    assert add_def.description == 'Add two numbers'
    assert subtract_def.name == 'math_subtract'
    assert subtract_def.description == 'Subtract two numbers'


async def test_prefixed_toolset_call_tools():
    """Test that PrefixedToolset correctly calls tools with prefixed names."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @base_toolset.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    prefixed_toolset = PrefixedToolset(base_toolset, 'calc')

    # Test calling tools with prefixed names
    result = await prefixed_toolset.call_tool(context, 'calc_add', {'a': 5, 'b': 3})
    assert result == 8

    result = await prefixed_toolset.call_tool(context, 'calc_multiply', {'a': 4, 'b': 2})
    assert result == 8


async def test_prefixed_toolset_prepare_for_run():
    """Test that PrefixedToolset correctly prepares for run with prefixed tools."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    prefixed_toolset = PrefixedToolset(base_toolset, 'test')

    # Prepare for run
    run_toolset = await prefixed_toolset.prepare_for_run(context)

    # Verify that the run toolset has prefixed tools
    assert run_toolset.tool_names == ['test_add']
    assert len(run_toolset.tool_defs) == 1
    assert run_toolset.tool_defs[0].name == 'test_add'

    # Verify that the tool still works
    result = await run_toolset.call_tool(context, 'test_add', {'a': 10, 'b': 5})
    assert result == 15


async def test_prefixed_toolset_error_invalid_prefix():
    """Test that PrefixedToolset raises ValueError for tool names that don't start with the prefix."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    prefixed_toolset = PrefixedToolset(base_toolset, 'math')

    # Test calling with wrong prefix
    with pytest.raises(ValueError, match="Tool name 'wrong_add' does not start with prefix 'math_'"):
        await prefixed_toolset.call_tool(context, 'wrong_add', {'a': 1, 'b': 2})

    # Test calling with no prefix
    with pytest.raises(ValueError, match="Tool name 'add' does not start with prefix 'math_'"):
        await prefixed_toolset.call_tool(context, 'add', {'a': 1, 'b': 2})

    # Test calling with partial prefix
    with pytest.raises(ValueError, match="Tool name 'mat_add' does not start with prefix 'math_'"):
        await prefixed_toolset.call_tool(context, 'mat_add', {'a': 1, 'b': 2})


async def test_prefixed_toolset_empty_prefix():
    """Test that PrefixedToolset works correctly with an empty prefix."""
    context = build_run_context(None)
    base_toolset = FunctionToolset[None]()

    @base_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    prefixed_toolset = PrefixedToolset(base_toolset, '')

    # Check that tool names have empty prefix (just underscore)
    assert prefixed_toolset.tool_names == ['_add']

    # Test calling the tool
    result = await prefixed_toolset.call_tool(context, '_add', {'a': 3, 'b': 4})
    assert result == 7

    # Test error for wrong name
    with pytest.raises(ValueError, match="Tool name 'add' does not start with prefix '_'"):
        await prefixed_toolset.call_tool(context, 'add', {'a': 1, 'b': 2})


async def test_comprehensive_toolset_composition():  # noqa: C901
    """Test that all toolsets can be composed together and work correctly."""

    @dataclass
    class TestDeps:
        user_role: str = 'user'
        enable_advanced: bool = True
        log_calls: bool = False
        log: list[str] = field(default_factory=list)

    # Create first FunctionToolset with basic math operations
    math_toolset = FunctionToolset[TestDeps]()

    @math_toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b

    @math_toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers"""
        return a - b

    @math_toolset.tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b

    # Create second FunctionToolset with string operations
    string_toolset = FunctionToolset[TestDeps]()

    @string_toolset.tool
    def concat(s1: str, s2: str) -> str:
        """Concatenate two strings"""
        return s1 + s2

    @string_toolset.tool
    def uppercase(text: str) -> str:
        """Convert text to uppercase"""
        return text.upper()

    @string_toolset.tool
    def reverse(text: str) -> str:
        """Reverse a string"""
        return text[::-1]

    # Create third FunctionToolset with advanced operations
    advanced_toolset = FunctionToolset[TestDeps]()

    @advanced_toolset.tool
    def power(base: int, exponent: int) -> int:
        """Calculate base raised to the power of exponent"""
        return base**exponent

    @advanced_toolset.tool
    def factorial(n: int) -> int:
        """Calculate factorial of n"""

        def _fact(x: int) -> int:
            if x <= 1:
                return 1
            return x * _fact(x - 1)

        return _fact(n)

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

    filtered_toolset = FilteredToolset(combined_prefixed_toolset, filter_tools)

    # Step 4: Apply prepared toolset to modify descriptions (add user role annotation)
    async def prepare_add_context(ctx: RunContext[TestDeps], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        # Annotate each tool description with the user role
        role = ctx.deps.user_role
        return [replace(td, description=f'{td.description} (role: {role})') for td in tool_defs]

    prepared_toolset = PreparedToolset(filtered_toolset, prepare_add_context)

    # Step 5: Apply processed toolset to add logging (store on deps.log, optionally wrap result)
    async def process_with_logging(
        ctx: RunContext[TestDeps],
        call_tool_func: Callable[[str, dict[str, Any], Any], Awaitable[Any]],
        name: str,
        tool_args: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if ctx.deps.log_calls:
            ctx.deps.log.append(f'Calling tool: {name} with args: {tool_args}')
        result = await call_tool_func(name, tool_args, *args, **kwargs)
        if ctx.deps.log_calls:
            ctx.deps.log.append(f'Tool {name} returned: {result}')
            # For demonstration, wrap the result in a dict if logging is enabled
            return {'result': result}
        return result

    processed_toolset = ProcessedToolset(prepared_toolset, process_with_logging)

    # Step 6: Test the fully composed toolset
    # Test with regular user context (log_calls=False)
    regular_deps = TestDeps(user_role='user', enable_advanced=True, log_calls=False)
    regular_context = build_run_context(regular_deps)
    final_toolset = await processed_toolset.prepare_for_run(regular_context)
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
            ToolDefinition(
                name='adv_factorial',
                description='Calculate factorial of n (role: user)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'n': {'type': 'integer'}},
                    'required': ['n'],
                    'type': 'object',
                },
            ),
        ]
    )
    # Call a tool and check result
    result = await final_toolset.call_tool(regular_context, 'math_add', {'a': 5, 'b': 3})
    assert result == 8

    # Test with admin user context (log_calls=False, should have string tools)
    admin_deps = TestDeps(user_role='admin', enable_advanced=True, log_calls=False)
    admin_context = build_run_context(admin_deps)
    admin_final_toolset = await processed_toolset.prepare_for_run(admin_context)
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
            ToolDefinition(
                name='adv_factorial',
                description='Calculate factorial of n (role: admin)',
                parameters_json_schema={
                    'additionalProperties': False,
                    'properties': {'n': {'type': 'integer'}},
                    'required': ['n'],
                    'type': 'object',
                },
            ),
        ]
    )
    result = await admin_final_toolset.call_tool(admin_context, 'str_concat', {'s1': 'Hello', 's2': 'World'})
    assert result == 'HelloWorld'

    # Test with logging enabled (log_calls=True, result should be wrapped)
    logging_deps = TestDeps(user_role='admin', enable_advanced=True, log_calls=True)
    logging_context = build_run_context(logging_deps)
    logging_final_toolset = await processed_toolset.prepare_for_run(logging_context)
    result = await logging_final_toolset.call_tool(logging_context, 'math_add', {'a': 10, 'b': 20})
    assert result == {'result': 30}
    assert logging_deps.log == ["Calling tool: math_add with args: {'a': 10, 'b': 20}", 'Tool math_add returned: 30']

    # Test with advanced features disabled (log_calls=False)
    basic_deps = TestDeps(user_role='user', enable_advanced=False, log_calls=False)
    basic_context = build_run_context(basic_deps)
    basic_final_toolset = await processed_toolset.prepare_for_run(basic_context)
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

    # Test prepare_for_run idempotency
    # toolset.prepare_for_run(ctx1).prepare_for_run(ctx2) == toolset.prepare_for_run(ctx2)
    ctx1 = build_run_context(TestDeps(user_role='user', enable_advanced=True, log_calls=False))
    ctx2 = build_run_context(TestDeps(user_role='admin', enable_advanced=True, log_calls=False))
    toolset_once = await processed_toolset.prepare_for_run(ctx2)
    toolset_twice = await (await processed_toolset.prepare_for_run(ctx1)).prepare_for_run(ctx2)
    assert toolset_once == toolset_twice

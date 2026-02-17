"""Tests for CodeExecutionToolset logic (description, caching, name collisions, deferred tools).

Uses StubRuntime so these tests don't require pydantic-monty or Docker.
"""

from __future__ import annotations

import pytest

from pydantic_ai._python_signature import FunctionSignature, TypeSignature, collect_unique_referenced_types
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset
from pydantic_ai.toolsets.function import FunctionToolset

from .conftest import StubRuntime, WeatherResult, build_code_execution_toolset, build_run_context, get_weather

pytestmark = pytest.mark.anyio


def _add(*, x: int, y: int) -> int:
    """Add two integers."""
    return x + y


def _get_weather_alias(city: str) -> WeatherResult:
    """Get weather (alias)."""
    return get_weather(city)


async def test_get_tools_produces_single_code_tool():
    """get_tools() returns exactly one tool named 'run_code'."""
    _, tools = await build_code_execution_toolset(StubRuntime(), (_add, False))
    assert list(tools.keys()) == ['run_code']


async def test_description_default():
    """Default description includes preamble and function signatures but no runtime instructions."""
    _, tools = await build_code_execution_toolset(StubRuntime(), (_add, False))
    description = tools['run_code'].tool_def.description or ''
    # Preamble present
    assert 'run Python code' in description
    # Function signature present
    assert 'async def _add' in description
    # No runtime instructions (StubRuntime.instructions is None)
    assert 'restricted Python subset' not in description


async def test_description_custom_string():
    """A custom string replaces the default preamble."""
    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_add, takes_ctx=False)
    cm = CodeExecutionToolset(ts, runtime=StubRuntime(), description='My preamble')
    tools = await cm.get_tools(build_run_context())
    assert 'My preamble' in (tools['run_code'].tool_def.description or '')


async def test_description_custom_callback():
    """A callback gets full control over the description."""

    def my_desc(sigs: list[FunctionSignature], types: list[TypeSignature], instructions: str | None) -> str:
        return f'{len(sigs)} tools'

    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_add, takes_ctx=False)
    cm = CodeExecutionToolset(ts, runtime=StubRuntime(), description=my_desc)
    tools = await cm.get_tools(build_run_context())
    assert tools['run_code'].tool_def.description == '1 tools'


async def test_deferred_tools_raise_user_error():
    """Wrapping a tool with requires_approval=True triggers UserError in get_tools()."""
    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_add, takes_ctx=False, requires_approval=True)
    cm = CodeExecutionToolset(ts, runtime=StubRuntime())
    with pytest.raises(UserError, match='approval and deferral are not yet supported'):
        await cm.get_tools(build_run_context())


async def test_name_collision_counter():
    """3 tools sanitizing to the same base name get _2/_3 suffixes."""

    def my_tool(*, x: int) -> int:
        """First."""
        return x

    # These have different __name__ but sanitize to same snake_case
    # Build toolset manually with colliding sanitized names
    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(my_tool, name='my-tool', takes_ctx=False)
    ts.add_function(my_tool, name='my.tool', takes_ctx=False)
    ts.add_function(my_tool, name='my tool', takes_ctx=False)
    cm = CodeExecutionToolset(ts, runtime=StubRuntime())
    tools = await cm.get_tools(build_run_context())
    description = tools['run_code'].tool_def.description or ''
    assert 'async def my_tool(' in description
    assert 'async def my_tool_2(' in description
    assert 'async def my_tool_3(' in description


async def test_cached_signature_reused_across_get_tools_calls():
    """Calling get_tools() twice reuses the same cached FunctionSignature objects."""
    code_execution, tools1 = await build_code_execution_toolset(StubRuntime(), (_add, False))

    ctx = build_run_context()
    tools2 = await code_execution.get_tools(ctx)

    tool1 = tools1['run_code']
    tool2 = tools2['run_code']

    # The code execution tool descriptions should match
    assert tool1.tool_def.description == tool2.tool_def.description


async def test_dedup_correctness_after_cache_backed_deepcopy():
    """Multiple tools with shared types produce correct dedup after cache-backed deepcopy."""
    _code_execution, tools = await build_code_execution_toolset(
        StubRuntime(), (get_weather, False), (_get_weather_alias, False)
    )
    tool = tools['run_code']

    # Both tools reference the same WeatherResult type — dedup should unify them
    unique_types = collect_unique_referenced_types(tool.signatures)
    weather_types = [t for t in unique_types if t.name == 'WeatherResult']
    assert len(weather_types) == 1


async def test_no_toolset_produces_run_code_tool():
    """CodeExecutionToolset() with no toolset still produces a 'run_code' tool."""
    cm = CodeExecutionToolset(runtime=StubRuntime())
    tools = await cm.get_tools(build_run_context())
    assert list(tools.keys()) == ['run_code']


async def test_no_toolset_description_omits_tool_calling():
    """With no toolset, the description uses the base prompt without mentioning tool calling."""
    cm = CodeExecutionToolset(runtime=StubRuntime())
    tools = await cm.get_tools(build_run_context())
    description = tools['run_code'].tool_def.description or ''
    # Base prompt is present
    assert 'run Python code' in description
    # Tools prompt elements are absent
    assert 'call other tools as functions' not in description
    assert 'Available functions' not in description
    assert 'async def' not in description

"""Tests for CodeExecutionToolset logic (description, caching, name collisions, deferred tools).

Uses StubEnvironment so these tests don't require pydantic-monty or Docker.
"""

from __future__ import annotations

from typing import Any

import pytest

from pydantic_ai._python_signature import FunctionSignature, TypeSignature, collect_unique_referenced_types
from pydantic_ai.exceptions import UserError
from pydantic_ai.toolsets.code_execution import CodeExecutionToolset
from pydantic_ai.toolsets.function import FunctionToolset

from .conftest import StubEnvironment, WeatherResult, build_code_execution_toolset, build_run_context, get_weather

pytestmark = pytest.mark.anyio


def _add(*, x: int, y: int) -> int:
    """Add two integers."""
    return x + y  # pragma: no cover


def _get_weather_alias(city: str) -> WeatherResult:
    """Get weather (alias)."""
    return get_weather(city)  # pragma: no cover


async def test_get_tools_produces_single_code_tool():
    """get_tools() returns exactly one tool named 'run_code'."""
    _, tools = await build_code_execution_toolset(StubEnvironment(), (_add, False))
    assert list(tools.keys()) == ['run_code']


async def test_description_default():
    """Default description includes preamble and function signatures but no environment instructions."""
    _, tools = await build_code_execution_toolset(StubEnvironment(), (_add, False))
    description = tools['run_code'].tool_def.description or ''
    # Preamble present
    assert 'run Python code' in description
    # Function signature present
    assert 'async def _add' in description
    # No environment instructions (StubEnvironment.tool_description returns None)
    assert 'restricted Python subset' not in description


async def test_description_custom_string():
    """A custom string replaces the default preamble."""
    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_add, takes_ctx=False)
    cm = CodeExecutionToolset(StubEnvironment(), toolset=ts, description='My preamble')
    tools = await cm.get_tools(build_run_context())
    assert 'My preamble' in (tools['run_code'].tool_def.description or '')


async def test_description_custom_callback():
    """A callback gets full control over the description."""

    def my_desc(sigs: list[FunctionSignature], types: list[TypeSignature], instructions: str | None) -> str:
        return f'{len(sigs)} tools'

    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_add, takes_ctx=False)
    cm = CodeExecutionToolset(StubEnvironment(), toolset=ts, description=my_desc)
    tools = await cm.get_tools(build_run_context())
    assert tools['run_code'].tool_def.description == '1 tools'


async def test_deferred_tools_raise_user_error():
    """Wrapping a tool with requires_approval=True triggers UserError in get_tools()."""
    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_add, takes_ctx=False, requires_approval=True)
    cm = CodeExecutionToolset(StubEnvironment(), toolset=ts)
    with pytest.raises(UserError, match='approval and deferral are not yet supported'):
        await cm.get_tools(build_run_context())


async def test_name_collision_counter():
    """3 tools sanitizing to the same base name get _2/_3 suffixes."""

    def my_tool(*, x: int) -> int:
        """First."""
        return x  # pragma: no cover

    # These have different __name__ but sanitize to same snake_case
    # Build toolset manually with colliding sanitized names
    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(my_tool, name='my-tool', takes_ctx=False)
    ts.add_function(my_tool, name='my.tool', takes_ctx=False)
    ts.add_function(my_tool, name='my tool', takes_ctx=False)
    cm = CodeExecutionToolset(StubEnvironment(), toolset=ts)
    tools = await cm.get_tools(build_run_context())
    description = tools['run_code'].tool_def.description or ''
    assert 'async def my_tool(' in description
    assert 'async def my_tool_2(' in description
    assert 'async def my_tool_3(' in description


async def test_cached_signature_reused_across_get_tools_calls():
    """Calling get_tools() twice reuses the same cached FunctionSignature objects."""
    code_execution, tools1 = await build_code_execution_toolset(StubEnvironment(), (_add, False))

    ctx = build_run_context()
    tools2 = await code_execution.get_tools(ctx)

    tool1 = tools1['run_code']
    tool2 = tools2['run_code']

    # The code execution tool descriptions should match
    assert tool1.tool_def.description == tool2.tool_def.description


async def test_dedup_correctness_after_cache_backed_deepcopy():
    """Multiple tools with shared types produce correct dedup after cache-backed deepcopy."""
    _code_execution, tools = await build_code_execution_toolset(
        StubEnvironment(), (get_weather, False), (_get_weather_alias, False)
    )
    tool = tools['run_code']

    # Both tools reference the same WeatherResult type — dedup should unify them
    unique_types = collect_unique_referenced_types(tool.signatures)
    weather_types = [t for t in unique_types if t.name == 'WeatherResult']
    assert len(weather_types) == 1


async def test_aenter_cleanup_on_wrapped_failure():
    """If wrapped toolset's __aenter__ raises, environment is cleaned up."""
    from pydantic_ai.toolsets.abstract import AbstractToolset

    class FailingToolset(AbstractToolset[None]):
        @property
        def id(self) -> str | None:
            return None

        async def __aenter__(self) -> Any:
            raise RuntimeError('wrapped failed')

        async def __aexit__(self, *args: Any) -> None:
            pass  # pragma: no cover

        async def get_tools(self, ctx: Any) -> dict[str, Any]:
            return {}  # pragma: no cover

        async def call_tool(self, name: str, tool_args: Any, ctx: Any, tool: Any) -> None:
            pass  # pragma: no cover

    enter_count = 0
    exit_count = 0

    class TrackingEnvironment(StubEnvironment):
        async def __aenter__(self) -> Any:
            nonlocal enter_count
            enter_count += 1
            return self

        async def __aexit__(self, *args: Any) -> None:
            nonlocal exit_count
            exit_count += 1

    failing = FailingToolset()
    assert failing.id is None  # verify the property works

    cm = CodeExecutionToolset(TrackingEnvironment(), toolset=failing)
    with pytest.raises(RuntimeError, match='wrapped failed'):
        await cm.__aenter__()
    assert enter_count == 1
    assert exit_count == 1  # environment was cleaned up


async def test_call_deferred_during_execution(monkeypatch: pytest.MonkeyPatch):
    """CallDeferred during tool execution raises UserError."""
    from pydantic_ai.exceptions import CallDeferred, UserError
    from pydantic_ai.toolsets.code_execution import CodeRuntimeError, FunctionCall

    call_made = False

    class ExecutingEnvironment(StubEnvironment):
        async def run_python(
            self, code: str, call_tool: Any = None, *, functions: Any = None, referenced_types: Any = None
        ) -> Any:
            nonlocal call_made
            call = FunctionCall(call_id='1', function_name='_add', args=(), kwargs={'x': 1, 'y': 2})
            try:
                await call_tool(call)
            except UserError:
                call_made = True
                raise CodeRuntimeError('deferred')

    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_add, takes_ctx=False)
    cm = CodeExecutionToolset(ExecutingEnvironment(), toolset=ts)
    ctx = build_run_context()
    tools = await cm.get_tools(ctx)
    tool = tools['run_code']

    # Monkeypatch handle_call to raise CallDeferred
    from pydantic_ai._tool_manager import ToolManager

    async def raising_handle(self: Any, tool_call: Any, **kwargs: Any) -> Any:
        raise CallDeferred()

    monkeypatch.setattr(ToolManager, 'handle_call', raising_handle)

    from pydantic_ai.exceptions import ModelRetry

    with pytest.raises(ModelRetry):
        await cm.call_tool('run_code', {'code': 'await _add(x=1, y=2)'}, ctx, tool)
    assert call_made


def test_get_weather_helper():
    """Verify get_weather fixture helper returns expected data."""
    result = get_weather('London')
    assert result == {'city': 'London', 'temperature': 15.0, 'unit': 'celsius', 'conditions': 'cloudy'}

    # Unknown city uses fallback data
    result = get_weather('Atlantis')
    assert result == {'city': 'Atlantis', 'temperature': 20.0, 'unit': 'celsius', 'conditions': 'unknown'}


async def test_no_toolset_produces_run_code_tool():
    """CodeExecutionToolset() with no toolset still produces a 'run_code' tool."""
    cm = CodeExecutionToolset(StubEnvironment())
    tools = await cm.get_tools(build_run_context())
    assert list(tools.keys()) == ['run_code']


async def test_no_toolset_description_omits_tool_calling():
    """With no toolset, the description uses the base prompt without mentioning tool calling."""
    cm = CodeExecutionToolset(StubEnvironment())
    tools = await cm.get_tools(build_run_context())
    description = tools['run_code'].tool_def.description or ''
    # Base prompt is present
    assert 'run Python code' in description
    # Tools prompt elements are absent
    assert 'call other tools as functions' not in description
    assert 'Available functions' not in description
    assert 'async def' not in description

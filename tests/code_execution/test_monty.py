"""Tests for Monty runtime type checking integration."""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict

try:
    from pydantic_monty import Monty

    from pydantic_ai.environments.monty import MontyEnvironment
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic-monty is not installed', allow_module_level=True)

from pydantic_ai._tool_manager import _parallel_execution_mode_ctx_var  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import Tool

from .conftest import build_code_execution_toolset, run_code_with_tools

pytestmark = pytest.mark.anyio


def add(*, x: int, y: int) -> int:
    """Add two integers."""
    return x + y


async def test_type_error_caught_at_validation_not_type_check():
    """With type checking disabled for REPL mode, type errors are caught at Pydantic validation time.

    The tool callback validates arguments via Pydantic, so passing strings where ints are
    expected raises a ValidationError wrapped as a runtime CodeRuntimeError → ModelRetry.
    """
    with pytest.raises(ModelRetry, match='Runtime error in generated code'):
        await run_code_with_tools('await add(x="hello", y="world")', MontyEnvironment(), (add, False))


async def test_generated_signatures_are_valid_python():
    """Generated signatures must be valid Python that Monty can parse and type check."""
    _, tools = await build_code_execution_toolset(MontyEnvironment(), (add, False))

    tool = tools['run_code']
    env = MontyEnvironment()
    prefix = env._build_type_check_prefix(tool.signatures, tool.referenced_types)  # pyright: ignore[reportPrivateUsage]

    # `...` and `pass` are not valid for Monty/ty type checking — ty is intentionally
    # stricter than pyright here. See https://github.com/astral-sh/ty/issues/1922
    assert prefix == snapshot('''\
import asyncio
from typing import Any, TypedDict, NotRequired, Literal

async def add(*, x: int, y: int) -> int:
    """Add two integers."""
    raise NotImplementedError()\
''')
    # Verify Monty can parse and type check code using this prefix
    m = Monty('add(x=1, y=2)', type_check=True, type_check_stubs=prefix)  # Should not raise


async def test_signatures_use_ellipsis_monty_converts_for_type_check():
    """Signatures use '...' body; Monty converts to 'raise NotImplementedError()' for type checking."""
    _code_execution, tools = await build_code_execution_toolset(MontyEnvironment(), (add, False))

    tool = tools['run_code']

    # LLM-facing description should have '...'
    description = tool.tool_def.description or ''
    assert '...' in description
    assert 'raise NotImplementedError()' not in description

    # But when Monty builds the type-check prefix, it converts to 'raise NotImplementedError()'
    env = MontyEnvironment()
    prefix = env._build_type_check_prefix(tool.signatures, tool.referenced_types)  # pyright: ignore[reportPrivateUsage]
    assert 'raise NotImplementedError()' in prefix
    assert '    ...' not in prefix


# --- Types and tools for test_full_description_snapshot ---


class _Tag(TypedDict):
    """A key-value tag."""

    key: str
    value: str


class _Resource(TypedDict):
    name: str
    tags: list[_Tag]
    metadata: NotRequired[dict[str, str]]
    """Extra metadata."""
    parent_id: int | None


def _find_resources(*, query: str, limit: int = 10) -> list[_Resource]:
    """Find resources matching a query."""
    return []  # pragma: no cover


# TODO (Douwe): this doesn't actually work :)
# Both have __name__ == 'Item' to test dedup when different tools share a type name
class _SearchItem(TypedDict):
    name: str
    price: float


class _LookupItem(TypedDict):
    id: int
    category: str


def _search_items(*, query: str) -> list[_SearchItem]:
    """Search for items by name."""
    return []  # pragma: no cover


def _get_item(*, item_id: int) -> list[_LookupItem]:
    """Get items by ID."""
    return []  # pragma: no cover


def _tag_resource(*, resource_name: str, tag: _Tag) -> bool:
    """Add a tag to a resource."""
    return True  # pragma: no cover


async def test_full_description_snapshot():
    """Snapshot the full run_code description with shared types, conflicts, and nesting."""
    _, tools = await build_code_execution_toolset(
        MontyEnvironment(),
        (_find_resources, False),
        (_search_items, False),
        (_get_item, False),
        (_tag_resource, False),
    )
    description = tools['run_code'].tool_def.description
    assert description == snapshot('''\

Use this tool to run Python code that can call other tools as functions.

You can use it to:
- filter tool return data to save context,
- perform complex operations that would take many model calls using standard tool calling, or
- pass the result of one tool to another without it entering your context window.

Execution model:
- This is a REPL session — state persists across calls. Variables, functions, and imports defined in previous calls are available in subsequent calls. You can split work across multiple calls and build on earlier results.
- If a previous call failed, the state from earlier *successful* calls is still intact — you only need to fix the failed snippet, not rewrite everything from scratch.
- You can create new functions for convenience.
- This tool is for calling and chaining tools programmatically — don't use it just to format or print your final analysis. Write your report as regular text in your response.

Session management:
- Set `restart: true` to clear all accumulated state and start a fresh session. You can combine it with `code` to reset and run in one call, or use it alone to just reset.
- Use restart when your session state is corrupted or you want a completely clean slate.



The runtime uses a restricted Python subset:
- you cannot use the standard library except builtin functions and the following modules: `sys`, `typing`, `asyncio`
- this means `collections`, `json`, `re`, `math`, `datetime`, `itertools`, `functools`, etc. are NOT available — use plain dicts, lists, and builtins instead
- you cannot use third party libraries
- you cannot define classes
- `sorted()` and `.sort()` do not support keyword arguments (`key=`, `reverse=`) and cannot sort lists of tuples — only sort flat lists of numbers or strings. If you need a custom sort order, build the output list manually (e.g. find max in a loop)
- chained subscript assignment like `x[a][b] = val` is NOT supported — read into a local variable, modify it, then assign back: `inner = x[a]; inner[b] = val; x[a] = inner`
- set operators (`|`, `&`, `-`, `^`) are not supported — use `set.update()`, `set.add()`, or loop to combine sets

State persists across calls — variables and functions defined in previous calls are available in subsequent calls.

The last expression evaluated is the return value.

Parallelism: use `asyncio.gather` to fire multiple calls at the same time instead of awaiting each one sequentially:

    # GOOD — parallel (all calls fire at once):
    results = await asyncio.gather(
        get_data(id=1),
        get_data(id=2),
        get_data(id=3),
    )

    # BAD — sequential (each call waits before the next starts):
    r1 = await get_data(id=1)
    r2 = await get_data(id=2)
    r3 = await get_data(id=3)


```python

# Available types:

class _Tag(TypedDict):
    """A key-value tag."""
    key: str
    value: str

class _Resource(TypedDict):
    name: str
    tags: list[_Tag]
    metadata: NotRequired[dict[str, str]]
    parent_id: int | None

class _SearchItem(TypedDict):
    name: str
    price: float

class _LookupItem(TypedDict):
    id: int
    category: str

# Available functions:

async def _find_resources(*, query: str, limit: int = 10) -> list[_Resource]:
    """Find resources matching a query."""
    ...

async def _search_items(*, query: str) -> list[_SearchItem]:
    """Search for items by name."""
    ...

async def _get_item(*, item_id: int) -> list[_LookupItem]:
    """Get items by ID."""
    ...

async def _tag_resource(*, resource_name: str, tag: _Tag) -> bool:
    """Add a tag to a resource."""
    ...

```\
''')


async def test_monty_runtime_error_raises_model_retry():
    """MontyRuntimeError during execution is surfaced as ModelRetry."""
    with pytest.raises(ModelRetry, match='Runtime error in generated code'):
        await run_code_with_tools('1 / 0', MontyEnvironment(), (add, False))


async def test_monty_syntax_error_message():
    """Monty syntax errors include a descriptive message for the LLM."""
    with pytest.raises(ModelRetry) as exc_info:
        await run_code_with_tools('def while invalid syntax', MontyEnvironment())

    assert str(exc_info.value) == snapshot("""\
Syntax error in generated code:
Expected an identifier, but found a keyword `while` that cannot be used here at byte range 4..9\
""")


class UserModel(BaseModel):
    name: str
    age: int


async def test_monty_normalizes_tool_results_to_json_compatible():
    """Tool results fed to Monty should be JSON-compatible (dicts, not BaseModels).

    Without normalization, Monty receives the raw Python object (e.g. a Pydantic
    BaseModel), but driver-based runtimes serialize results over JSON and would
    receive a plain dict. This inconsistency means code that works on one
    runtime could break on another.

    The fix: normalize all results to JSON-compatible form (via
    tool_return_ta.dump_python(mode='json')) before feeding them to Monty,
    matching what driver-based runtimes already do.

    This test exposes the issue by passing a tool result to a second tool that
    observes the actual Python type on the host side.
    """

    def get_user(id: int) -> UserModel:
        """Get a user by ID."""
        return UserModel(name='Alice', age=30)

    received_types: list[str] = []

    def inspect_type(data: Any) -> str:
        """Record the Python type of the received data."""
        received_types.append(type(data).__name__)
        return type(data).__name__

    code = 'user = await get_user(id=1)\nawait inspect_type(data=user)'

    result = await run_code_with_tools(
        code,
        MontyEnvironment(),
        (get_user, False),
        (inspect_type, False),
    )

    # After normalization, the second tool should receive a dict, not a UserModel.
    # This guarantees consistent behavior across runtimes.
    assert result == 'dict'
    assert received_types == ['dict']


async def test_build_type_check_prefix_empty_lists():
    """Empty signatures/types produces just the typing import line."""
    env = MontyEnvironment()
    prefix = env._build_type_check_prefix([], [])  # pyright: ignore[reportPrivateUsage]
    assert prefix == 'import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal'


# --- Sequential tool tests ---


def sync_add(*, x: int, y: int) -> int:
    """Add two integers synchronously."""
    return x + y


async def test_sequential_tool_renders_as_def():
    """A sequential tool renders as `def` (not `async def`) in description and type-check prefix."""
    _, tools = await build_code_execution_toolset(
        MontyEnvironment(),
        Tool(sync_add, takes_ctx=False, sequential=True),
    )
    tool = tools['run_code']

    # Description shows `def`, not `async def`
    description = tool.tool_def.description or ''
    assert 'def sync_add(' in description
    assert 'async def sync_add(' not in description

    # Type-check prefix also uses `def`
    env = MontyEnvironment()
    prefix = env._build_type_check_prefix(tool.signatures, tool.referenced_types)  # pyright: ignore[reportPrivateUsage]
    assert 'def sync_add(' in prefix
    assert 'async def sync_add(' not in prefix


async def test_sequential_tool_execution():
    """Sequential tool is called as `result = my_tool(x=1)` (no `await`)."""
    result = await run_code_with_tools(
        'sync_add(x=3, y=4)',
        MontyEnvironment(),
        Tool(sync_add, takes_ctx=False, sequential=True),
    )
    assert result == 7


async def test_mixed_sync_async_execution():
    """Both async (fire-then-await) and sync tools work in the same code block."""
    code = 'f = add(x=1, y=2)\nsum_result = sync_add(x=10, y=20)\nawaited = await f\n[awaited, sum_result]'
    result = await run_code_with_tools(
        code,
        MontyEnvironment(),
        (add, False),
        Tool(sync_add, takes_ctx=False, sequential=True),
    )
    assert result == [3, 30]


async def test_sequential_drain_behavior():
    """Firing an async tool then calling a sequential tool drains the async task first."""
    call_order: list[str] = []

    def fire_tool(*, name: str) -> str:
        """An async tool."""
        call_order.append(f'async:{name}')
        return f'async:{name}'

    def seq_tool(*, name: str) -> str:
        """A sequential tool."""
        call_order.append(f'seq:{name}')
        return f'seq:{name}'

    code = 'f = fire_tool(name="first")\nresult = seq_tool(name="second")\nawaited = await f\n[awaited, result]'
    result = await run_code_with_tools(
        code,
        MontyEnvironment(),
        (fire_tool, False),
        Tool(seq_tool, takes_ctx=False, sequential=True),
    )
    assert result == ['async:first', 'seq:second']
    # The async tool must complete before the sequential tool starts
    assert call_order == ['async:first', 'seq:second']


async def test_await_on_sync_tool_is_type_error():
    """`await sync_tool()` raises ModelRetry (runtime error since type checking is disabled for REPL mode)."""
    with pytest.raises(ModelRetry, match='Runtime error in generated code'):
        await run_code_with_tools(
            'await sync_add(x=1, y=2)',
            MontyEnvironment(),
            Tool(sync_add, takes_ctx=False, sequential=True),
        )


async def test_global_sequential_mode():
    """Setting _parallel_execution_mode_ctx_var to 'sequential' makes all tools render as `def`."""
    token = _parallel_execution_mode_ctx_var.set('sequential')
    try:
        _, tools = await build_code_execution_toolset(
            MontyEnvironment(),
            (add, False),
        )
        tool = tools['run_code']
        description = tool.tool_def.description or ''
        assert 'def add(' in description
        assert 'async def add(' not in description
    finally:
        _parallel_execution_mode_ctx_var.reset(token)


async def test_global_parallel_ordered_events_mode():
    """Setting _parallel_execution_mode_ctx_var to 'parallel_ordered_events' makes all tools render as `def`."""
    token = _parallel_execution_mode_ctx_var.set('parallel_ordered_events')
    try:
        _, tools = await build_code_execution_toolset(
            MontyEnvironment(),
            (add, False),
        )
        tool = tools['run_code']
        description = tool.tool_def.description or ''
        assert 'def add(' in description
        assert 'async def add(' not in description
    finally:
        _parallel_execution_mode_ctx_var.reset(token)


async def test_description_no_all_functions_are_async():
    """The prompt no longer says 'All functions are async'."""
    _, tools = await build_code_execution_toolset(MontyEnvironment(), (add, False))
    description = tools['run_code'].tool_def.description or ''
    assert 'All functions are async' not in description


async def test_pending_tasks_cancelled_on_runtime_error():
    """Async tasks fired before a runtime error are cancelled in the finally block."""
    code = 'f = add(x=1, y=2)\n1 / 0'
    with pytest.raises(ModelRetry, match='Runtime error in generated code'):
        await run_code_with_tools(code, MontyEnvironment(), (add, False))


# --- Direct run_python tests ---


async def test_run_python_success():
    """run_python returns the last expression."""
    env = MontyEnvironment()
    result = await env.run_python('"hello"')
    assert result == 'hello'


async def test_run_python_syntax_error():
    """run_python raises CodeSyntaxError for invalid syntax."""
    from pydantic_ai.toolsets.code_execution._abstract import CodeSyntaxError

    env = MontyEnvironment()
    with pytest.raises(CodeSyntaxError):
        await env.run_python('def while')


async def test_run_python_type_annotation_not_enforced_at_runtime():
    """With type checking disabled for REPL mode, type annotations are not enforced."""
    env = MontyEnvironment()
    # x: int = "hello" is accepted — Monty's runtime doesn't enforce annotations
    result = await env.run_python('x: int = "hello"\nx')
    assert result == 'hello'


async def test_run_python_runtime_error():
    """run_python raises CodeRuntimeError for runtime errors."""
    from pydantic_ai.toolsets.code_execution._abstract import CodeRuntimeError

    env = MontyEnvironment()
    with pytest.raises(CodeRuntimeError):
        await env.run_python('1 / 0')


async def test_run_python_with_functions_default_params():
    """run_python_with_functions works with default functions/referenced_types."""
    from unittest.mock import AsyncMock

    env = MontyEnvironment()
    result = await env.run_python_with_functions(
        '"hello"',
        function_callback=AsyncMock(),
    )
    assert result == 'hello'


# --- Reset and type_check tests ---


async def test_reset_clears_repl_state():
    """reset() discards REPL state so prior variables are no longer available."""
    env = MontyEnvironment()
    await env.run_python('x = 42')
    result = await env.run_python('x')
    assert result == 42

    env.reset()

    from pydantic_ai.toolsets.code_execution._abstract import CodeRuntimeError

    with pytest.raises(CodeRuntimeError, match='NameError'):
        await env.run_python('x')


async def test_reset_allows_fresh_start():
    """After reset(), new code executes in a clean environment."""
    env = MontyEnvironment()
    await env.run_python('x = "old"')
    env.reset()
    await env.run_python('x = "new"')
    result = await env.run_python('x')
    assert result == 'new'


async def test_type_check_catches_type_error():
    """type_check() raises CodeTypingError for type mismatches."""
    from pydantic_ai.toolsets.code_execution._abstract import CodeTypingError

    env = MontyEnvironment()
    with pytest.raises(CodeTypingError):
        env.type_check('x: int = "hello"')


async def test_type_check_catches_syntax_error():
    """type_check() raises CodeSyntaxError for invalid syntax."""
    from pydantic_ai.toolsets.code_execution._abstract import CodeSyntaxError

    env = MontyEnvironment()
    with pytest.raises(CodeSyntaxError):
        env.type_check('def while')


async def test_type_check_with_function_stubs():
    """type_check() validates calls against provided function signatures."""
    env = MontyEnvironment()
    _, tools = await build_code_execution_toolset(env, (add, False))
    tool = tools['run_code']

    # Valid call — should not raise
    env.type_check('await add(x=1, y=2)', signatures=tool.signatures, referenced_types=tool.referenced_types)


async def test_type_check_valid_code_passes():
    """type_check() does not raise for valid code."""
    env = MontyEnvironment()
    env.type_check('x: int = 42')  # Should not raise


async def test_repl_state_persists_across_calls():
    """REPL state persists — variables survive between calls."""
    env = MontyEnvironment()
    await env.run_python('count = 0')
    await env.run_python('count = count + 1')
    await env.run_python('count = count + 1')
    result = await env.run_python('count')
    assert result == 2


# --- Print output tests ---


async def test_print_only_returns_string():
    """print() with no expression result returns the printed text."""
    env = MontyEnvironment()
    result = await env.run_python('print("hello")')
    assert result == 'hello'


async def test_print_multiple_lines():
    """Multiple print() calls are concatenated."""
    env = MontyEnvironment()
    result = await env.run_python('print("line 1")\nprint("line 2")')
    assert result == 'line 1\nline 2'


async def test_output_only_preserves_structure():
    """Expression result without prints is returned as-is (structured)."""
    env = MontyEnvironment()
    result = await env.run_python('[1, 2, 3]')
    assert result == [1, 2, 3]


async def test_print_and_output_returns_dict():
    """print() combined with an expression result returns a dict with both."""
    env = MontyEnvironment()
    result = await env.run_python('print("debug")\n[1, 2, 3]')
    assert result == {'stdout': 'debug', 'result': [1, 2, 3]}


async def test_print_and_dict_output_returns_dict():
    """print() combined with a dict expression preserves the dict structure."""
    env = MontyEnvironment()
    result = await env.run_python('print("info")\n{"key": "value"}')
    assert result == {'stdout': 'info', 'result': {'key': 'value'}}


async def test_print_and_none_output_returns_string():
    """print() with None expression result returns just the printed text."""
    env = MontyEnvironment()
    result = await env.run_python('print("hello")\nNone')
    assert result == 'hello'


async def test_prints_included_in_runtime_error():
    """Print output before a runtime error is included in the error message."""
    from pydantic_ai.toolsets.code_execution._abstract import CodeRuntimeError

    env = MontyEnvironment()
    with pytest.raises(CodeRuntimeError) as exc_info:
        await env.run_python('print("debug info")\n1 / 0')
    assert 'debug info' in exc_info.value.message
    assert 'ZeroDivisionError' in exc_info.value.message
    assert '[stdout before error]' in exc_info.value.message


async def test_prints_not_in_error_when_no_prints():
    """Error messages without prior prints don't have the stdout wrapper."""
    from pydantic_ai.toolsets.code_execution._abstract import CodeRuntimeError

    env = MontyEnvironment()
    with pytest.raises(CodeRuntimeError) as exc_info:
        await env.run_python('1 / 0')
    assert 'stdout before error' not in exc_info.value.message


async def test_print_with_function_calls():
    """Print output is captured alongside external function calls."""
    env = MontyEnvironment()
    result = await run_code_with_tools(
        'print("before call")\nr = await add(x=1, y=2)\nprint("after call")\nr',
        env,
        (add, False),
    )
    assert result == {'stdout': 'before call\nafter call', 'result': 3}


async def test_print_after_function_call():
    """Prints after an external function call returns are still captured."""
    env = MontyEnvironment()
    result = await run_code_with_tools(
        'r = await add(x=10, y=20)\nprint(r)',
        env,
        (add, False),
    )
    assert result == '30'


async def test_prints_included_in_runtime_error_with_functions():
    """Print output before a runtime error with functions is included in the error message."""
    env = MontyEnvironment()
    with pytest.raises(ModelRetry, match='debug') as exc_info:
        await run_code_with_tools(
            'print("debug")\n1 / 0',
            env,
            (add, False),
        )
    assert 'debug' in str(exc_info.value)

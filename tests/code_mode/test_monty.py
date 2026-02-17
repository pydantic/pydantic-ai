"""Tests for Monty runtime type checking integration."""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict

try:
    from pydantic_monty import Monty

    from pydantic_ai.runtime.monty import MontyRuntime
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic-monty is not installed', allow_module_level=True)

from pydantic_ai._tool_manager import _parallel_execution_mode_ctx_var  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import Tool

from .conftest import build_code_mode_toolset, run_code_with_tools

pytestmark = pytest.mark.anyio


def add(*, x: int, y: int) -> int:
    """Add two integers."""
    return x + y


async def test_type_error_raises_model_retry():
    """Type errors in generated code raise ModelRetry so the LLM can fix them."""
    with pytest.raises(ModelRetry) as exc_info:
        await run_code_with_tools('add(x="hello", y="world")', MontyRuntime(), (add, False))

    assert str(exc_info.value) == snapshot("""\
Type error in generated code:
main.py:2:5: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["hello"]`
main.py:2:16: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["world"]`
""")


async def test_generated_signatures_are_valid_python():
    """Generated signatures must be valid Python that Monty can parse and type check."""
    _, tools = await build_code_mode_toolset(MontyRuntime(), (add, False))

    tool = tools['run_code_with_tools']
    runtime = MontyRuntime()
    prefix = runtime._build_type_check_prefix(tool.signatures, tool.referenced_types)  # pyright: ignore[reportPrivateUsage]

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
    m = Monty('add(x=1, y=2)', external_functions=['add'])
    m.type_check(prefix_code=prefix)  # Should not raise


async def test_signatures_use_ellipsis_monty_converts_for_type_check():
    """Signatures use '...' body; Monty converts to 'raise NotImplementedError()' for type checking."""
    _code_mode, tools = await build_code_mode_toolset(MontyRuntime(), (add, False))

    tool = tools['run_code_with_tools']

    # LLM-facing description should have '...'
    description = tool.tool_def.description or ''
    assert '...' in description
    assert 'raise NotImplementedError()' not in description

    # But when Monty builds the type-check prefix, it converts to 'raise NotImplementedError()'
    runtime = MontyRuntime()
    prefix = runtime._build_type_check_prefix(tool.signatures, tool.referenced_types)  # pyright: ignore[reportPrivateUsage]
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
    """Snapshot the full run_code_with_tools description with shared types, conflicts, and nesting."""
    _, tools = await build_code_mode_toolset(
        MontyRuntime(),
        (_find_resources, False),
        (_search_items, False),
        (_get_item, False),
        (_tag_resource, False),
    )
    description = tools['run_code_with_tools'].tool_def.description
    assert description == snapshot('''\

Use this tool to run Python code that can call other tools as functions, also known as "code mode" or "programmatic tool calling".

You can use it to:
- filter tool return data to save context,
- perform complex operations that would take many model calls using standard tool calling, or
- pass the result of one tool to another without it entering your context window.

Execution model:
- Each call to this tool runs in a completely isolated environment — no variables, results, or state persist between calls. You MUST do all your work (fetching data, processing it, and producing the final result) in a single code block. Do not split work across multiple calls expecting to use earlier results.
- If a previous call failed, you must rewrite the entire program from scratch — you cannot reference variables or results from a failed attempt.
- You can create new functions for convenience.
- This tool is for calling and chaining tools programmatically — don't use it just to format or print your final analysis. Write your report as regular text in your response.



The runtime uses a restricted Python subset:
- you cannot use the standard library except builtin functions and the following modules: `sys`, `typing`, `asyncio`
- this means `collections`, `json`, `re`, `math`, `datetime`, `itertools`, `functools`, etc. are NOT available — use plain dicts, lists, and builtins instead
- you cannot use third party libraries
- you cannot define classes
- `sorted()` and `.sort()` do not support keyword arguments (`key=`, `reverse=`) and cannot sort lists of tuples — only sort flat lists of numbers or strings. If you need a custom sort order, build the output list manually (e.g. find max in a loop)
- chained subscript assignment like `x[a][b] = val` is NOT supported — read into a local variable, modify it, then assign back: `inner = x[a]; inner[b] = val; x[a] = inner`
- set operators (`|`, `&`, `-`, `^`) are not supported — use `set.update()`, `set.add()`, or loop to combine sets

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
        await run_code_with_tools('1 / 0', MontyRuntime(), (add, False))


async def test_monty_syntax_error_message():
    """Monty syntax errors include a descriptive message for the LLM."""
    with pytest.raises(ModelRetry) as exc_info:
        await run_code_with_tools('def while invalid syntax', MontyRuntime())

    assert str(exc_info.value) == snapshot("""\
Syntax error in generated code:
Expected an identifier, but found a keyword `while` that cannot be used here at byte range 19..24\
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
        MontyRuntime(),
        (get_user, False),
        (inspect_type, False),
    )

    # After normalization, the second tool should receive a dict, not a UserModel.
    # This guarantees consistent behavior across runtimes.
    assert result == 'dict'
    assert received_types == ['dict']


async def test_build_type_check_prefix_empty_lists():
    """Empty signatures/types produces just the typing import line."""
    runtime = MontyRuntime()
    prefix = runtime._build_type_check_prefix([], [])  # pyright: ignore[reportPrivateUsage]
    assert prefix == 'import asyncio\nfrom typing import Any, TypedDict, NotRequired, Literal'


# --- Sequential tool tests ---


def sync_add(*, x: int, y: int) -> int:
    """Add two integers synchronously."""
    return x + y


async def test_sequential_tool_renders_as_def():
    """A sequential tool renders as `def` (not `async def`) in description and type-check prefix."""
    _, tools = await build_code_mode_toolset(
        MontyRuntime(),
        Tool(sync_add, takes_ctx=False, sequential=True),
    )
    tool = tools['run_code_with_tools']

    # Description shows `def`, not `async def`
    description = tool.tool_def.description or ''
    assert 'def sync_add(' in description
    assert 'async def sync_add(' not in description

    # Type-check prefix also uses `def`
    runtime = MontyRuntime()
    prefix = runtime._build_type_check_prefix(tool.signatures, tool.referenced_types)  # pyright: ignore[reportPrivateUsage]
    assert 'def sync_add(' in prefix
    assert 'async def sync_add(' not in prefix


async def test_sequential_tool_execution():
    """Sequential tool is called as `result = my_tool(x=1)` (no `await`)."""
    result = await run_code_with_tools(
        'sync_add(x=3, y=4)',
        MontyRuntime(),
        Tool(sync_add, takes_ctx=False, sequential=True),
    )
    assert result == 7


async def test_mixed_sync_async_execution():
    """Both async (fire-then-await) and sync tools work in the same code block."""
    code = 'f = add(x=1, y=2)\nsum_result = sync_add(x=10, y=20)\nawaited = await f\n[awaited, sum_result]'
    result = await run_code_with_tools(
        code,
        MontyRuntime(),
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
        MontyRuntime(),
        (fire_tool, False),
        Tool(seq_tool, takes_ctx=False, sequential=True),
    )
    assert result == ['async:first', 'seq:second']
    # The async tool must complete before the sequential tool starts
    assert call_order == ['async:first', 'seq:second']


async def test_await_on_sync_tool_is_type_error():
    """`await sync_tool()` raises ModelRetry from Monty's type checker."""
    with pytest.raises(ModelRetry, match='Type error in generated code'):
        await run_code_with_tools(
            'await sync_add(x=1, y=2)',
            MontyRuntime(),
            Tool(sync_add, takes_ctx=False, sequential=True),
        )


async def test_global_sequential_mode():
    """Setting _parallel_execution_mode_ctx_var to 'sequential' makes all tools render as `def`."""
    token = _parallel_execution_mode_ctx_var.set('sequential')
    try:
        _, tools = await build_code_mode_toolset(
            MontyRuntime(),
            (add, False),
        )
        tool = tools['run_code_with_tools']
        description = tool.tool_def.description or ''
        assert 'def add(' in description
        assert 'async def add(' not in description
    finally:
        _parallel_execution_mode_ctx_var.reset(token)


async def test_global_parallel_ordered_events_mode():
    """Setting _parallel_execution_mode_ctx_var to 'parallel_ordered_events' makes all tools render as `def`."""
    token = _parallel_execution_mode_ctx_var.set('parallel_ordered_events')
    try:
        _, tools = await build_code_mode_toolset(
            MontyRuntime(),
            (add, False),
        )
        tool = tools['run_code_with_tools']
        description = tool.tool_def.description or ''
        assert 'def add(' in description
        assert 'async def add(' not in description
    finally:
        _parallel_execution_mode_ctx_var.reset(token)


async def test_description_no_all_functions_are_async():
    """The prompt no longer says 'All functions are async'."""
    _, tools = await build_code_mode_toolset(MontyRuntime(), (add, False))
    description = tools['run_code_with_tools'].tool_def.description or ''
    assert 'All functions are async' not in description


async def test_pending_tasks_cancelled_on_runtime_error():
    """Async tasks fired before a runtime error are cancelled in the finally block."""
    code = 'f = add(x=1, y=2)\n1 / 0'
    with pytest.raises(ModelRetry, match='Runtime error in generated code'):
        await run_code_with_tools(code, MontyRuntime(), (add, False))

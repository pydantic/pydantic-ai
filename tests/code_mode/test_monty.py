"""Tests for Monty runtime type checking integration."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from typing_extensions import NotRequired, TypedDict

try:
    from pydantic_monty import Monty

    from pydantic_ai.runtime.monty import MontyRuntime
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic-monty is not installed', allow_module_level=True)

from pydantic_ai._python_signature import FunctionSignature, TypeSignature
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

from .conftest import build_code_mode_toolset, build_run_context, run_code_with_tools

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
main.py:1:5: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["hello"]`
main.py:1:16: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["world"]`
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
    return []


# Both have __name__ == 'Item' to test dedup when different tools share a type name
class _SearchItem(TypedDict):
    name: str
    price: float


class _LookupItem(TypedDict):
    id: int
    category: str


def _search_items(*, query: str) -> list[_SearchItem]:
    """Search for items by name."""
    return []


def _get_item(*, item_id: int) -> list[_LookupItem]:
    """Get items by ID."""
    return []


def _tag_resource(*, resource_name: str, tag: _Tag) -> bool:
    """Add a tag to a resource."""
    return True


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
- Each call to this tool runs in a completely isolated environment — variables don't persist between calls
- If a previous call failed, you must rewrite the entire program from scratch — you cannot reference variables or results from a failed attempt
- All functions are async. You can create new functions for convenience.
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

To run independent calls concurrently, fire them first, then `await`, or use `asyncio.gather`:
```python test="skip" lint="skip"
# starts immediately:
items_future = get_items()
users_future = get_users()

# wait for results:
items = await items_future
users = await users_future

# or equivalently:
import asyncio
items, users = await asyncio.gather(items_future, users_future)
```


```python

# Available types:

class _Tag(TypedDict):
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


# =============================================================================
# CodeModeToolset description customization
# =============================================================================


def _tool_alpha(*, x: int) -> int:
    """Tool alpha."""
    return x


async def test_custom_description_string():
    """A custom string replaces the default preamble."""
    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_tool_alpha, takes_ctx=False)
    cm = CodeModeToolset(ts, runtime=MontyRuntime(), description='My preamble')
    tools = await cm.get_tools(build_run_context())
    assert 'My preamble' in (tools['run_code_with_tools'].tool_def.description or '')


async def test_custom_description_callback():
    """A callback gets full control over the description."""

    def my_desc(sigs: list[FunctionSignature], types: list[TypeSignature], instructions: str | None) -> str:
        return f'{len(sigs)} tools'

    ts: FunctionToolset[None] = FunctionToolset()
    ts.add_function(_tool_alpha, takes_ctx=False)
    cm = CodeModeToolset(ts, runtime=MontyRuntime(), description=my_desc)
    tools = await cm.get_tools(build_run_context())
    assert tools['run_code_with_tools'].tool_def.description == '1 tools'

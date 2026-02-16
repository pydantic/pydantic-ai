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

from pydantic_ai.exceptions import ModelRetry

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

Use `run_code_with_tools` to write Python code that calls the available functions. You can make a single call or combine multiple steps in a script — use your judgment based on the task.

Execution model:
- Each `run_code_with_tools` call runs in an isolated environment — variables don't persist between calls
- Functions are async — call with `await`, e.g. `result = await get_items()`
- To run independent calls concurrently, fire them first, then await:
  ```python
  future_a = get_items()    # starts immediately
  future_b = get_users()    # starts immediately
  items = await future_a    # wait for results
  users = await future_b
  ```
- The last expression evaluated is the return value
- Return raw data when it answers the question directly; extract or transform when needed


Syntax note: the runtime uses a restricted Python subset.
- Imports are not available — use the provided functions and builtins (len, sum, str, etc.) or define your own helpers.

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

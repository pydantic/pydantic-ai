"""Tests for code mode type checking and error handling."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from pydantic_monty import Monty

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.runtime.monty import MontyRuntime, _build_type_check_prefix  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

from .conftest import build_run_context

pytestmark = pytest.mark.anyio


def add(x: int, y: int) -> int:
    """Add two integers."""
    return x + y


async def test_type_error_raises_model_retry():
    """Type errors in generated code raise ModelRetry so the LLM can fix them."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset, runtime=MontyRuntime())
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    run_code_tool = tools['run_code']

    with pytest.raises(ModelRetry) as exc_info:
        await code_mode.call_tool(
            'run_code',
            {'code': 'add("hello", "world")'},  # Strings instead of ints
            ctx,
            run_code_tool,
        )

    assert str(exc_info.value) == snapshot("""\
Type error in generated code:
main.py:1:5: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["hello"]`
main.py:1:14: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["world"]`
""")


async def test_syntax_error_raises_model_retry():
    """Syntax errors in generated code raise ModelRetry so the LLM can fix them."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset, runtime=MontyRuntime())
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    run_code_tool = tools['run_code']

    with pytest.raises(ModelRetry) as exc_info:
        await code_mode.call_tool(
            'run_code',
            {'code': 'def while invalid syntax'},
            ctx,
            run_code_tool,
        )

    assert str(exc_info.value) == snapshot("""\
Syntax error in generated code:
Expected an identifier, but found a keyword `while` that cannot be used here at byte range 4..9\
""")


async def test_valid_code_executes_successfully():
    """Valid code passes type checking and executes normally (sync tool `add` is awaited)."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset, runtime=MontyRuntime())
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    run_code_tool = tools['run_code']

    result = await code_mode.call_tool(
        'run_code',
        {'code': 'await add(x=1, y=2)'},
        ctx,
        run_code_tool,
    )

    assert result == 3


async def test_generated_signatures_are_valid_python():
    """Generated signatures must be valid Python that Monty can parse and type check."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset, runtime=MontyRuntime())
    ctx = build_run_context()
    await code_mode.get_tools(ctx)

    # Build the prefix that will be used for type checking
    assert code_mode._cached_signatures is not None  # pyright: ignore[reportPrivateUsage]
    prefix = _build_type_check_prefix(code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]

    # `...` and `pass` are not valid for Monty/ty type checking - ty is intentionally
    # stricter than pyright here. See https://github.com/astral-sh/ty/issues/1922
    # They may add a way to disable this check in the future.
    assert prefix == snapshot('''\
from typing import Any, TypedDict, NotRequired, Literal

async def add(x: int, y: int) -> int:
    """Add two integers."""
    raise NotImplementedError()\
''')
    # Verify Monty can parse and type check code using this prefix
    # If our signature generation is broken, this will raise MontySyntaxError
    m = Monty('add(1, 2)', external_functions=['add'])
    m.type_check(prefix_code=prefix)  # Should not raise


async def test_signatures_use_ellipsis_monty_converts_for_type_check():
    """Signatures use '...' body; Monty converts to 'raise NotImplementedError()' for type checking."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset, runtime=MontyRuntime())
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    # LLM-facing description should have '...'
    description = tools['run_code'].tool_def.description or ''
    assert '...' in description
    assert 'raise NotImplementedError()' not in description

    # Cached signatures should also have '...' (not NotImplementedError)
    assert code_mode._cached_signatures is not None  # pyright: ignore[reportPrivateUsage]
    assert all('...' in sig for sig in code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]
    assert not any('raise NotImplementedError()' in sig for sig in code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]

    # But when Monty builds the type-check prefix, it converts to 'raise NotImplementedError()'
    prefix = _build_type_check_prefix(code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]
    assert 'raise NotImplementedError()' in prefix
    assert '    ...' not in prefix

    assert description == snapshot('''\
Use run_code to write Python code that solves the task. Rather than calling tools individually, prefer writing a script that combines multiple steps.

Execution context:
- Each run_code call is isolated — variables do not persist between calls
- Plan your solution before writing code

CRITICAL Syntax restrictions (the runtime uses a restricted Python subset):
- No imports - use only the provided functions and builtins (len, sum, str, etc.) or write your own functions.

How to write effective code:
- External functions return coroutines — use `await` to get results
- For sequential execution: `items = await get_items()`
- For parallel execution of independent calls, fire first then await:
  ```python
  future_items = get_items()   # Fire (no await)
  future_users = get_users()   # Fire (no await)
  items = await future_items   # Both execute in parallel
  users = await future_users
  ```
- Use keyword arguments (e.g., `get_user(id=123)` not `get_user(123)`)
- Use for loops to handle multiple items
- The last expression evaluated becomes the return value

Available functions:

```python
async def add(x: int, y: int) -> int:
    """Add two integers."""
    ...
```

Example — parallel fetching, then sequential processing:
```python
# Parallel: fire independent calls first (no await yet)
future_items = get_items(category="electronics")
future_users = get_users(status="active")

# Await results — both calls execute in parallel
items = await future_items
users = await future_users

# Sequential: process items (each depends on previous result)
results = []
total = 0
for item in items:
    details = await get_item_details(id=item["id"])
    if details["status"] == "active":
        total = total + details["price"]
        results.append({"name": item["name"], "price": details["price"]})

{"total": total, "count": len(results), "items": results, "user_count": len(users)}
```\
''')

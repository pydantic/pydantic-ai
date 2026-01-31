"""Tests for code mode type checking and error handling."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot
from pydantic_monty import Monty

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.runtime.monty import _build_type_check_prefix  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


def build_run_context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        prompt=None,
        messages=[],
        run_step=0,
    )


def add(x: int, y: int) -> int:
    """Add two integers."""
    return x + y


async def test_type_error_raises_model_retry():
    """Type errors in generated code raise ModelRetry so the LLM can fix them."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset)
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

    code_mode = CodeModeToolset(wrapped=toolset)
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
    """Valid code passes type checking and executes normally."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset)
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


async def test_await_on_sync_tool_executes_successfully():
    """Sync tools can be awaited in code mode — signatures are always async def."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset)
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

    code_mode = CodeModeToolset(wrapped=toolset)
    ctx = build_run_context()
    await code_mode.get_tools(ctx)

    # Build the prefix that will be used for type checking
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


async def test_llm_sees_ellipsis_but_type_check_has_not_implemented():
    """LLM-facing signatures use '...' but type-check prefix uses 'raise NotImplementedError()'."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset)
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    # LLM-facing description should have '...'
    description = tools['run_code'].tool_def.description or ''
    assert '...' in description
    assert 'raise NotImplementedError()' not in description

    # Type-check prefix should still have 'raise NotImplementedError()'
    assert any('raise NotImplementedError()' in sig for sig in code_mode._cached_signatures)  # pyright: ignore[reportPrivateUsage]

    assert description == snapshot('''\
ALWAYS use run_code to solve the ENTIRE task in a single code block. Do not call tools individually - write one comprehensive Python script that fetches all data, processes it, and returns the complete answer.

CRITICAL execution model:
- Solve the COMPLETE problem in ONE run_code call - not partial solutions
- Each run_code call is ISOLATED - variables do NOT persist between calls
- Plan your entire solution before writing code, then implement it all at once


CRITICAL Syntax restrictions (the runtime uses a restricted Python subset):
- No imports - use only the provided functions and builtins (len, sum, str, etc.) or write your own functions.

How to write effective code:
- ALWAYS use `await` when calling external functions (e.g., `items = await get_items()`)
- ALWAYS use keyword arguments when calling functions (e.g., `get_user(id=123)` not `get_user(123)`)
- Use for loops to handle multiple items
- NEVER return raw tool results - always extract/filter to only what you need
- The last expression evaluated becomes the return value - make it a processed summary, not raw data

Available functions:

```python
async def add(x: int, y: int) -> int:
    """Add two integers."""
    ...
```

Example - fetching, filtering, and summarizing in one execution:
```python
# Fetch data
items = await get_items(category="electronics")

# Process immediately - extract only needed fields
results = []
total = 0
for item in items:
    details = await get_item_details(id=item["id"])
    if details["status"] == "active":
        total = total + details["price"]
        results.append({"name": item["name"], "price": details["price"]})

# Return processed summary, NOT raw data
{"total": total, "count": len(results), "items": results}
```\
''')

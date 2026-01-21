"""Tests for code mode type checking and error handling."""

from __future__ import annotations

import monty
import pytest
from inline_snapshot import snapshot

from pydantic_ai._run_context import RunContext
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets.code_mode import (
    CodeModeToolset,
    _build_type_check_prefix,  # pyright: ignore[reportPrivateUsage]
)
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
main.py:6:5: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["hello"]`
main.py:6:14: error[invalid-argument-type] Argument to function `add` is incorrect: Expected `int`, found `Literal["world"]`
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
        {'code': 'add(1, 2)'},
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

def add(x: int, y: int) -> int:
    """Add two integers."""
    raise NotImplementedError()\
''')
    # Verify Monty can parse and type check code using this prefix
    # If our signature generation is broken, this will raise MontySyntaxError
    m = monty.Monty('add(1, 2)', external_functions=['add'])
    m.type_check(prefix_code=prefix)  # Should not raise


async def test_signature_includes_raise_not_implemented():
    """Generated signatures include raise NotImplementedError() for Monty type checking."""
    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)

    code_mode = CodeModeToolset(wrapped=toolset)
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    description = tools['run_code'].tool_def.description or ''
    assert description == snapshot('''\
You should consider writing Python code to accomplish multiple tasks in one go instead of using multiple tools one by one.

How to:
- ALWAYS use keyword arguments when calling functions (e.g., `get_user(id=123)` not `get_user(123)`)
- Use for loops to handle multiple items (e.g., for each user, fetch their orders and aggregate)
- The last expression evaluated becomes the return value - make it the final answer

Syntax restrictions (the runtime uses a restricted Python subset):
- No imports - use only the provided functions and builtins (len, sum, str, etc.)
- No while loops - use for loops instead
- No comprehensions (list/dict/set) or generator expressions - use explicit for loops
- No lambdas - define logic inline
- No tuple unpacking (e.g., `a, b = 1, 2`) - assign variables separately
- No list index assignment (e.g., `lst[0] = x`) - use list.append() to build lists
- No string methods (.join, .split, .upper, etc.) - return data structures, not formatted strings

What DOES work:
- Dict assignment: `d["key"] = value`
- Dict methods: `.get()`, `.keys()`, `.values()`, `.items()`
- List methods: `.append()`
- F-strings: `f"value is {x}"`
- Builtins: `len()`, `sum()`, `str()`, `list()`, `range()`

Available functions:

```python
def add(x: int, y: int) -> int:
    """Add two integers."""
    raise NotImplementedError()
```

Example - completing a full aggregation task in one execution:
```python
items = get_items(category="electronics")
results = []
total = 0

for item in items:
    details = get_item_details(id=item["id"])
    if details["status"] == "active":
        total += details["price"]
        results.append({"name": item["name"], "price": details["price"]})

{"total": total, "count": len(results), "items": results}
```\
''')

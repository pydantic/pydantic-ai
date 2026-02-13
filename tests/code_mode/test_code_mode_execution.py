"""Test basic code mode execution: tool calls, parallelism, and error handling.

Parameterized across all CodeRuntime implementations (Monty, stdio subprocess).
"""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.runtime.abstract import CodeRuntime

try:
    from pydantic_ai.runtime.monty import MontyRuntime
except ImportError:  # pragma: lax no cover
    pytest.skip('pydantic-monty is not installed', allow_module_level=True)

from .conftest import run_code_with_tools

pytestmark = pytest.mark.anyio


async def test_simple_execution(code_runtime: CodeRuntime):
    """Valid code + tool call executes and returns result."""

    def add(x: int, y: int) -> int:
        return x + y

    result = await run_code_with_tools('await add(x=1, y=2)', code_runtime, (add, False))
    assert result == 3


async def test_parallel_execution(code_runtime: CodeRuntime):
    """Fire-then-await runs tools in parallel."""

    def slow_op(name: str) -> str:
        return f'done:{name}'

    code = 'f1 = slow_op(name="a")\nf2 = slow_op(name="b")\nr1 = await f1\nr2 = await f2\n[r1, r2]'
    result = await run_code_with_tools(code, code_runtime, (slow_op, False))
    assert result == ['done:a', 'done:b']


async def test_no_function_calls(code_runtime: CodeRuntime):
    """Code without function calls executes locally and returns result."""
    result = await run_code_with_tools('1 + 2', code_runtime)
    assert result == 3


async def test_syntax_error_raises_model_retry(code_runtime: CodeRuntime):
    """Syntax errors raise ModelRetry so the LLM can fix them."""
    with pytest.raises(ModelRetry):
        await run_code_with_tools('def while invalid', code_runtime)


async def test_monty_syntax_error_message():
    """Monty syntax errors include a descriptive message for the LLM."""
    with pytest.raises(ModelRetry) as exc_info:
        await run_code_with_tools('def while invalid syntax', MontyRuntime())

    assert str(exc_info.value) == snapshot("""\
Syntax error in generated code:
Expected an identifier, but found a keyword `while` that cannot be used here at byte range 4..9\
""")


async def test_runtime_error_raises_model_retry(code_runtime: CodeRuntime):
    """Runtime exceptions raise ModelRetry so the LLM can fix them."""
    with pytest.raises(ModelRetry):
        await run_code_with_tools('1 / 0', code_runtime)


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


async def test_tool_exception_propagates(code_runtime: CodeRuntime):
    """Tool exceptions propagate and crash the run, consistent with normal tool execution."""

    def failing_tool() -> str:
        raise ValueError('tool bug')

    with pytest.raises(ValueError, match='tool bug'):
        await run_code_with_tools('await failing_tool()', code_runtime, (failing_tool, False))


async def test_execution_timeout_raises_model_retry(code_runtime: CodeRuntime):
    """Execution timeout raises ModelRetry so the LLM is informed."""
    code_runtime.execution_timeout = 1.0
    with pytest.raises(ModelRetry, match='timed out'):
        await run_code_with_tools('while True: pass', code_runtime)

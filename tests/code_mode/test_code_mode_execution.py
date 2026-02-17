"""Test basic code mode execution: tool calls, parallelism, and error handling.

Parameterized across all CodeRuntime implementations (Monty, Docker).
"""

from __future__ import annotations

import pytest

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.runtime.abstract import CodeRuntime

from .conftest import run_code_with_tools

pytestmark = pytest.mark.anyio


async def test_simple_execution(code_runtime: CodeRuntime):
    """Valid code + tool call executes and returns result."""

    def add(x: int, y: int) -> int:
        return x + y  # pragma: no cover

    result = await run_code_with_tools('await add(x=1, y=2)', code_runtime, (add, False))
    assert result == 3


async def test_parallel_execution(code_runtime: CodeRuntime):
    """Fire-then-await runs tools in parallel."""

    def slow_op(name: str) -> str:
        return f'done:{name}'

    code = 'f1 = slow_op(name="a")\nf2 = slow_op(name="b")\nr1 = await f1\nr2 = await f2\n[r1, r2]'
    result = await run_code_with_tools(code, code_runtime, (slow_op, False))
    assert result == ['done:a', 'done:b']


async def test_parallel_execution_gather(code_runtime: CodeRuntime):
    """asyncio.gather runs tools in parallel."""

    def slow_op(name: str) -> str:
        return f'done:{name}'

    code = 'import asyncio\nresults = await asyncio.gather(slow_op(name="a"), slow_op(name="b"))\nlist(results)'
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


async def test_runtime_error_raises_model_retry(code_runtime: CodeRuntime):
    """Runtime exceptions raise ModelRetry so the LLM can fix them."""
    with pytest.raises(ModelRetry):
        await run_code_with_tools('1 / 0', code_runtime)


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


async def test_positional_args_raise_model_retry(code_runtime: CodeRuntime):
    """Positional arguments in code mode tool calls raise ModelRetry.

    Monty catches this at type-check time (too many positional arguments);
    Docker catches it at call_tool_callback time (positional args not supported).
    Either way the LLM gets a ModelRetry.
    """

    def add(x: int, y: int) -> int:
        return x + y  # pragma: no cover

    with pytest.raises(ModelRetry):
        await run_code_with_tools('await add(1, 2)', code_runtime, (add, False))

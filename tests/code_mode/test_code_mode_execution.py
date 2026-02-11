"""Test basic code mode execution: tool calls, parallelism, and error handling.

Parameterized across all CodeRuntime implementations (Monty, stdio subprocess).
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


async def test_runtime_error_raises_model_retry(code_runtime: CodeRuntime):
    """Runtime exceptions raise ModelRetry so the LLM can fix them."""
    with pytest.raises(ModelRetry):
        await run_code_with_tools('1 / 0', code_runtime)


async def test_positional_args(code_runtime: CodeRuntime):
    """Positional args are passed through to tools correctly."""

    def add(x: int, y: int) -> int:
        return x + y

    result = await run_code_with_tools('await add(1, 2)', code_runtime, (add, False))
    assert result == 3

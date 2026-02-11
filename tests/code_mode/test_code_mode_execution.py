"""Test basic code mode execution: tool calls, parallelism, and error handling.

Parameterized across all CodeRuntime implementations (Monty, stdio subprocess).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.runtime.abstract import CodeRuntime
from pydantic_ai.runtime.monty import MontyRuntime

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


class UserModel(BaseModel):
    name: str
    age: int


async def test_monty_normalizes_tool_results_to_json_compatible():
    """Tool results fed to Monty should be JSON-compatible (dicts, not BaseModels).

    Without normalization, Monty receives the raw Python object (e.g. a Pydantic
    BaseModel) on a fresh run, but would receive a plain dict on a checkpoint
    resume (after JSON round-trip). This means code that works on a fresh run
    could break on resume, or vice versa.

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
    # This guarantees consistent behavior between fresh runs and checkpoint resumes.
    assert result == 'dict'
    assert received_types == ['dict']

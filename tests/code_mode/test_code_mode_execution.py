"""Test basic code mode execution: tool calls, parallelism, and error handling.

Parameterized across all CodeRuntime implementations (Monty, stdio subprocess).
"""

from __future__ import annotations

import pytest

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.runtime.abstract import CodeRuntime
from pydantic_ai.toolsets.code_mode import CodeModeToolset
from pydantic_ai.toolsets.function import FunctionToolset

from .conftest import build_run_context

pytestmark = pytest.mark.anyio


async def test_simple_execution(code_runtime: CodeRuntime):
    """Valid code + tool call executes and returns result."""

    def add(x: int, y: int) -> int:
        return x + y

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(add, takes_ctx=False)
    code_mode = CodeModeToolset(wrapped=toolset, runtime=code_runtime)
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    result = await code_mode.call_tool('run_code', {'code': 'await add(x=1, y=2)'}, ctx, tools['run_code'])
    assert result == 3


async def test_parallel_execution(code_runtime: CodeRuntime):
    """Fire-then-await runs tools in parallel."""
    call_order: list[str] = []

    def slow_op(name: str) -> str:
        call_order.append(name)
        return f'done:{name}'

    toolset: FunctionToolset[None] = FunctionToolset()
    toolset.add_function(slow_op, takes_ctx=False)
    code_mode = CodeModeToolset(wrapped=toolset, runtime=code_runtime)
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    code = 'f1 = slow_op(name="a")\nf2 = slow_op(name="b")\nr1 = await f1\nr2 = await f2\n[r1, r2]'
    result = await code_mode.call_tool('run_code', {'code': code}, ctx, tools['run_code'])
    assert result == ['done:a', 'done:b']


async def test_no_function_calls(code_runtime: CodeRuntime):
    """Code without function calls executes locally and returns result."""
    toolset: FunctionToolset[None] = FunctionToolset()
    code_mode = CodeModeToolset(wrapped=toolset, runtime=code_runtime)
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    result = await code_mode.call_tool('run_code', {'code': '1 + 2'}, ctx, tools['run_code'])
    assert result == 3


async def test_syntax_error_raises_model_retry(code_runtime: CodeRuntime):
    """Syntax errors raise ModelRetry so the LLM can fix them."""
    toolset: FunctionToolset[None] = FunctionToolset()
    code_mode = CodeModeToolset(wrapped=toolset, runtime=code_runtime)
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    with pytest.raises(ModelRetry):
        await code_mode.call_tool('run_code', {'code': 'def while invalid'}, ctx, tools['run_code'])


async def test_runtime_error_raises_model_retry(code_runtime: CodeRuntime):
    """Runtime exceptions raise ModelRetry so the LLM can fix them."""
    toolset: FunctionToolset[None] = FunctionToolset()
    code_mode = CodeModeToolset(wrapped=toolset, runtime=code_runtime)
    ctx = build_run_context()
    tools = await code_mode.get_tools(ctx)

    with pytest.raises(ModelRetry):
        await code_mode.call_tool('run_code', {'code': '1 / 0'}, ctx, tools['run_code'])

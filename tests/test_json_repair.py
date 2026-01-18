"""Tests for JSON repair functionality in tool argument parsing.

This module tests that when `fast_json_repair` is available, broken JSON
passed as tool arguments will be automatically repaired before validation.
"""

from __future__ import annotations

import pytest

from pydantic_ai import FunctionToolset, ToolCallPart
from pydantic_ai._run_context import RunContext
from pydantic_ai._tool_manager import ToolManager
from pydantic_ai.models.test import TestModel
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


async def test_json_repair_fixes_broken_json():
    """Test that broken JSON is repaired when fast_json_repair is available."""
    pytest.importorskip('fast_json_repair')

    toolset = FunctionToolset[None]()

    @toolset.tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    context = build_run_context()
    tool_manager = await ToolManager[None](toolset).for_run_step(context)

    # Broken JSON: missing closing brace
    broken_json = '{"a": 1, "b": 2'

    result = await tool_manager.handle_call(
        ToolCallPart(tool_name='add', args=broken_json),
    )
    assert result == 3


async def test_json_repair_fixes_trailing_comma():
    """Test that JSON with trailing comma is repaired."""
    pytest.importorskip('fast_json_repair')

    toolset = FunctionToolset[None]()

    @toolset.tool
    def greet(name: str) -> str:
        """Greet someone."""
        return f'Hello, {name}!'

    context = build_run_context()
    tool_manager = await ToolManager[None](toolset).for_run_step(context)

    # Broken JSON: trailing comma
    broken_json = '{"name": "World",}'

    result = await tool_manager.handle_call(
        ToolCallPart(tool_name='greet', args=broken_json),
    )
    assert result == 'Hello, World!'


async def test_json_repair_fixes_single_quotes():
    """Test that JSON with single quotes is repaired."""
    pytest.importorskip('fast_json_repair')

    toolset = FunctionToolset[None]()

    @toolset.tool
    def echo(message: str) -> str:
        """Echo a message."""
        return message

    context = build_run_context()
    tool_manager = await ToolManager[None](toolset).for_run_step(context)

    # Broken JSON: single quotes instead of double quotes
    broken_json = "{'message': 'hello'}"

    result = await tool_manager.handle_call(
        ToolCallPart(tool_name='echo', args=broken_json),
    )
    assert result == 'hello'


async def test_valid_json_still_works():
    """Test that valid JSON continues to work normally."""
    toolset = FunctionToolset[None]()

    @toolset.tool
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    context = build_run_context()
    tool_manager = await ToolManager[None](toolset).for_run_step(context)

    # Valid JSON should work regardless of fast_json_repair being installed
    valid_json = '{"x": 5, "y": 4}'

    result = await tool_manager.handle_call(
        ToolCallPart(tool_name='multiply', args=valid_json),
    )
    assert result == 20


async def test_dict_args_still_work():
    """Test that dict args (not string JSON) continue to work normally."""
    toolset = FunctionToolset[None]()

    @toolset.tool
    def subtract(a: int, b: int) -> int:
        """Subtract two numbers."""
        return a - b

    context = build_run_context()
    tool_manager = await ToolManager[None](toolset).for_run_step(context)

    # Dict args (not string JSON) should work normally
    result = await tool_manager.handle_call(
        ToolCallPart(tool_name='subtract', args={'a': 10, 'b': 3}),
    )
    assert result == 7


async def test_repair_doesnt_help_validation_error():
    """Test that validation errors are raised when repair doesn't help.

    When JSON is syntactically valid but semantically wrong (e.g., wrong types),
    repair won't change it, so the original validation error should be raised.
    """
    pytest.importorskip('fast_json_repair')

    from pydantic_ai.exceptions import ToolRetryError

    toolset = FunctionToolset[None]()

    @toolset.tool
    def typed_func(count: int) -> int:
        """A function that expects an int."""
        return count * 2

    context = build_run_context()
    tool_manager = await ToolManager[None](toolset).for_run_step(context)

    # Valid JSON syntax, but wrong type - repair won't change this
    invalid_type_json = '{"count": "not_an_integer"}'

    with pytest.raises(ToolRetryError):
        await tool_manager.handle_call(
            ToolCallPart(tool_name='typed_func', args=invalid_type_json),
        )

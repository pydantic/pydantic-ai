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


async def test_repair_doesnt_help_when_json_unchanged():
    """Test that validation errors are raised when repair doesn't change the JSON.

    When JSON is syntactically valid but semantically wrong (e.g., wrong types),
    repair won't change it, so the original validation error should be raised.
    """
    pytest.importorskip('fast_json_repair')

    from pydantic_ai.exceptions import ToolRetryError

    toolset = FunctionToolset[None]()

    @toolset.tool
    def typed_func(count: int) -> int:  # pragma: no cover
        """A function that expects an int."""
        return count * 2  # Never called - we're testing validation failure

    context = build_run_context()
    tool_manager = await ToolManager[None](toolset).for_run_step(context)

    # Valid JSON syntax, but wrong type - repair won't change this
    invalid_type_json = '{"count": "not_an_integer"}'

    with pytest.raises(ToolRetryError):
        await tool_manager.handle_call(
            ToolCallPart(tool_name='typed_func', args=invalid_type_json),
        )


async def test_repair_changes_json_but_still_fails_validation():
    """Test that original error is raised when repair changes JSON but validation still fails.

    When JSON is malformed AND has wrong types, repair will fix the syntax but
    validation will still fail. We should re-raise the original error.
    """
    pytest.importorskip('fast_json_repair')

    from pydantic_ai.exceptions import ToolRetryError

    toolset = FunctionToolset[None]()

    @toolset.tool
    def typed_func(count: int) -> int:  # pragma: no cover
        """A function that expects an int."""
        return count * 2  # Never called - we're testing validation failure

    context = build_run_context()
    tool_manager = await ToolManager[None](toolset).for_run_step(context)

    # Malformed JSON (missing closing brace) AND wrong type
    # Repair will fix the brace, but validation still fails due to wrong type
    malformed_and_wrong_type = '{"count": "not_an_integer"'

    with pytest.raises(ToolRetryError):
        await tool_manager.handle_call(
            ToolCallPart(tool_name='typed_func', args=malformed_and_wrong_type),
        )


def test_object_output_processor_repair():
    """Test that ObjectOutputProcessor.validate also repairs JSON."""
    pytest.importorskip('fast_json_repair')

    from pydantic import BaseModel

    from pydantic_ai._output import ObjectOutputProcessor

    class MyOutput(BaseModel):
        name: str
        age: int

    processor = ObjectOutputProcessor(MyOutput)

    # Broken JSON: missing closing brace
    broken_json = '{"name": "Alice", "age": 30'

    result = processor.validate(broken_json)
    # validate returns a model instance when passed a BaseModel
    assert isinstance(result, MyOutput)
    assert result.name == 'Alice'
    assert result.age == 30


def test_object_output_processor_repair_trailing_comma():
    """Test that ObjectOutputProcessor.validate repairs trailing commas."""
    pytest.importorskip('fast_json_repair')

    from pydantic import BaseModel

    from pydantic_ai._output import ObjectOutputProcessor

    class MyOutput(BaseModel):
        value: str

    processor = ObjectOutputProcessor(MyOutput)

    # Broken JSON: trailing comma
    broken_json = '{"value": "test",}'

    result = processor.validate(broken_json)
    # validate returns a model instance when passed a BaseModel
    assert isinstance(result, MyOutput)
    assert result.value == 'test'


def test_object_output_processor_repair_partial_json():
    """Test that ObjectOutputProcessor.validate repairs partial JSON during streaming.

    This tests that JSON repair works with allow_partial=True, which is used during
    streaming. The repair should fix syntax errors (like single quotes) while
    preserving partial string values.
    """
    pytest.importorskip('fast_json_repair')

    from pydantic import BaseModel

    from pydantic_ai._output import ObjectOutputProcessor

    class Whale(BaseModel):
        name: str
        description: str | None = None

    processor = ObjectOutputProcessor(Whale)

    # Malformed partial JSON: single quotes + incomplete string value
    # This simulates streaming where the model outputs broken JSON mid-stream
    malformed_partial = '{"name": \'blue whale\', "description": "The blue whale is the lar'

    # Without repair, this would fail because Pydantic can't handle single quotes
    # With repair, the single quotes are fixed AND the partial string is preserved
    result = processor.validate(malformed_partial, allow_partial=True)

    assert isinstance(result, Whale)
    assert result.name == 'blue whale'
    # The partial description should be preserved (not truncated to None)
    assert result.description == 'The blue whale is the lar'


def test_validate_json_with_repair_partial():
    """Test that validate_json_with_repair works with allow_partial=True.

    This tests the streaming scenario where tool arguments arrive with syntax
    errors but need to be validated incrementally. Previously, repair was skipped
    for partial validation, but now it should work.
    """
    pytest.importorskip('fast_json_repair')

    from pydantic import BaseModel, TypeAdapter

    from pydantic_ai._utils import validate_json_with_repair

    class WhaleArgs(BaseModel):
        name: str
        description: str | None = None

    adapter = TypeAdapter(WhaleArgs)
    validator = adapter.validator

    # Malformed partial JSON with single quotes - simulates Claude's fine-grained
    # tool streaming which can produce invalid JSON mid-stream
    malformed_partial = '{"name": \'orca\', "description": "Also known as killer'

    # Validate as partial (streaming mode) - this should now work with repair!
    result = validate_json_with_repair(
        validator=validator,
        json_str=malformed_partial,
        allow_partial=True,
        validation_context=None,
    )

    assert result.name == 'orca'
    assert result.description == 'Also known as killer'

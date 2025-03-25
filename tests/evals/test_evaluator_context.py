from __future__ import annotations as _annotations

from typing import Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.otel._errors import SpanTreeRecordingError
    from pydantic_evals.otel.span_tree import SpanTree

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


def test_evaluator_context_basic():
    """Test basic EvaluatorContext functionality."""
    # Create a mock span tree
    span_tree = SpanTree()

    # Create a context with all fields populated
    ctx = EvaluatorContext(
        name='test_case',
        inputs={'input': 'value'},
        metadata={'meta': 'data'},
        expected_output={'expected': 'output'},
        output={'actual': 'output'},
        duration=1.0,
        _span_tree=span_tree,
        attributes={'attr': 'value'},
        metrics={'metric': 1.0},
    )

    # Test basic attribute access
    assert ctx.name == 'test_case'
    assert ctx.inputs == {'input': 'value'}
    assert ctx.metadata == {'meta': 'data'}
    assert ctx.expected_output == {'expected': 'output'}
    assert ctx.output == {'actual': 'output'}
    assert ctx.duration == 1.0
    assert ctx.attributes == {'attr': 'value'}
    assert ctx.metrics == {'metric': 1.0}

    # Test span_tree property
    assert ctx.span_tree == span_tree


def test_evaluator_context_span_tree_error():
    """Test EvaluatorContext with SpanTreeRecordingError."""

    ctx = EvaluatorContext(
        name='test',
        inputs={},
        metadata=None,
        expected_output=None,
        output={},
        duration=0.0,
        _span_tree=SpanTreeRecordingError('Test error'),
        attributes={},
        metrics={},
    )

    # Test that accessing span_tree raises the error
    with pytest.raises(SpanTreeRecordingError) as exc_info:
        _ = ctx.span_tree

    assert str(exc_info.value) == 'Test error'


def test_evaluator_context_with_custom_types():
    """Test EvaluatorContext with custom generic types."""

    class CustomInput:
        def __init__(self, value: str):
            self.value = value

    class CustomOutput:
        def __init__(self, result: int):
            self.result = result

    class CustomMetadata:
        def __init__(self, info: Any):
            self.info = info

    # Create context with custom types
    ctx = EvaluatorContext[CustomInput, CustomOutput, CustomMetadata](
        name='test',
        inputs=CustomInput('test_input'),
        metadata=CustomMetadata({'info': 'test'}),
        expected_output=CustomOutput(42),
        output=CustomOutput(42),
        duration=1.0,
        _span_tree=SpanTreeRecordingError('Test error'),
        attributes={},
        metrics={},
    )

    assert isinstance(ctx.inputs, CustomInput)
    assert ctx.inputs.value == 'test_input'
    assert isinstance(ctx.output, CustomOutput)
    assert ctx.output.result == 42
    assert isinstance(ctx.metadata, CustomMetadata)
    assert ctx.metadata.info == {'info': 'test'}

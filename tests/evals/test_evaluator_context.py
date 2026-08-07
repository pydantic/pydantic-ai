from __future__ import annotations as _annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.otel._errors import SpanTreeRecordingError
    from pydantic_evals.otel.span_tree import SpanNode, SpanTree

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
        inputs=dict[str, Any](),
        metadata=None,
        expected_output=None,
        output=dict[str, Any](),
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


def test_span_tree_recording_error_serde():
    """Test that SpanTreeRecordingError round-trips in message field through JSON serialization."""
    adapter = TypeAdapter(SpanTreeRecordingError)

    error = SpanTreeRecordingError('test error message')

    json_bytes = adapter.dump_json(error)
    restored = adapter.validate_json(json_bytes)

    assert isinstance(restored, SpanTreeRecordingError)
    assert restored.message == 'test error message'


def test_span_tree_recording_error_serde_json_shape():
    """Test the JSON shape produced by SpanTreeRecordingError serialization."""
    adapter = TypeAdapter(SpanTreeRecordingError)

    error = SpanTreeRecordingError('some message')
    data = adapter.dump_python(error, mode='json')

    assert data == {'message': 'some message', 'kind': 'span-tree-recording-error'}


def test_evaluator_context_serde_with_span_tree():
    """Test EvaluatorContext round-trips through `TypeAdapter(EvaluatorContext)` when _span_tree is a populated SpanTree.

    The tree is built with `add_spans` rather than `SpanTree(roots=[...])`: `__post_init__` rebuilds the
    tree from `nodes_by_id`, so passing `roots` directly would leave the parent/child path untested.
    """
    adapter = TypeAdapter(EvaluatorContext)
    root = SpanNode(
        name='root',
        trace_id=1,
        span_id=1,
        parent_span_id=None,
        start_timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_timestamp=datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        attributes={'key': 'value'},
    )
    child = SpanNode(
        name='child',
        trace_id=1,
        span_id=2,
        parent_span_id=1,
        start_timestamp=datetime(2025, 1, 1, 0, 0, 0, 500_000, tzinfo=timezone.utc),
        end_timestamp=datetime(2025, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
        attributes={},
    )
    span_tree = SpanTree()
    span_tree.add_spans([root, child])

    ctx = EvaluatorContext(
        name='test_case',
        inputs='hello',
        metadata=None,
        expected_output=None,
        output='world',
        duration=0.1,
        _span_tree=span_tree,
        attributes={},
        metrics={},
    )

    assert adapter.dump_python(ctx, mode='json') == snapshot(
        {
            'name': 'test_case',
            'inputs': 'hello',
            'metadata': None,
            'expected_output': None,
            'output': 'world',
            'duration': 0.1,
            '_span_tree': {
                'roots': [
                    {
                        'name': 'root',
                        'trace_id': 1,
                        'span_id': 1,
                        'parent_span_id': None,
                        'start_timestamp': '2025-01-01T00:00:00Z',
                        'end_timestamp': '2025-01-01T00:00:01Z',
                        'attributes': {'key': 'value'},
                        'status': 'unset',
                    }
                ],
                'nodes_by_id': {
                    '00000000000000000000000000000001:0000000000000001': {
                        'name': 'root',
                        'trace_id': 1,
                        'span_id': 1,
                        'parent_span_id': None,
                        'start_timestamp': '2025-01-01T00:00:00Z',
                        'end_timestamp': '2025-01-01T00:00:01Z',
                        'attributes': {'key': 'value'},
                        'status': 'unset',
                    },
                    '00000000000000000000000000000001:0000000000000002': {
                        'name': 'child',
                        'trace_id': 1,
                        'span_id': 2,
                        'parent_span_id': 1,
                        'start_timestamp': '2025-01-01T00:00:00.500000Z',
                        'end_timestamp': '2025-01-01T00:00:01Z',
                        'attributes': {},
                        'status': 'unset',
                    },
                },
            },
            'attributes': {},
            'metrics': {},
        }
    )

    json_bytes = adapter.dump_json(ctx)
    restored = adapter.validate_json(json_bytes)

    assert isinstance(restored.span_tree, SpanTree)
    assert [node.name for node in restored.span_tree.roots] == ['root']
    assert [node.name for node in restored.span_tree.roots[0].children] == ['child']


def test_evaluator_context_serde_with_error():
    """Test EvaluatorContext round-trips through `TypeAdapter(EvaluatorContext)` when _span_tree is an error."""
    adapter = TypeAdapter(EvaluatorContext)
    ctx = EvaluatorContext(
        name='test_case',
        inputs={'key': 'value'},
        metadata={'meta': 1},
        expected_output='expected',
        output='actual',
        duration=2.5,
        _span_tree=SpanTreeRecordingError('spans not recorded'),
        attributes={'attr': 'val'},
        metrics={'m': 42},
    )

    json_bytes = adapter.dump_json(ctx)
    restored = adapter.validate_json(json_bytes)

    assert restored.name == ctx.name
    assert restored.inputs == ctx.inputs
    assert restored.metadata == ctx.metadata
    assert restored.expected_output == ctx.expected_output
    assert restored.output == ctx.output
    assert restored.duration == ctx.duration
    assert restored.attributes == ctx.attributes
    assert restored.metrics == ctx.metrics
    with pytest.raises(SpanTreeRecordingError, match='spans not recorded'):
        _ = restored.span_tree

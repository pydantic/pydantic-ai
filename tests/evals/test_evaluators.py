from typing import Any

import pytest
from inline_snapshot import snapshot
from logfire.testing import CaptureLogfire
from pydantic import BaseModel

from pydantic_evals.evaluators.common import (
    contains,
    equals,
    equals_expected,
    is_instance,
    llm_judge,
    max_duration,
    python,
    span_query,
)
from pydantic_evals.evaluators.context import EvaluatorContext
from pydantic_evals.evaluators.spec import Evaluator, EvaluatorResult, EvaluatorSpec
from pydantic_evals.otel.span_tree import SpanTree

pytestmark = pytest.mark.anyio


class TaskInput(BaseModel):
    query: str


class TaskOutput(BaseModel):
    answer: str


class TaskMetadata(BaseModel):
    difficulty: str = 'easy'


@pytest.fixture
def test_context() -> EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]:
    return EvaluatorContext[TaskInput, TaskOutput, TaskMetadata](
        name='test_case',
        inputs=TaskInput(query='What is 2+2?'),
        output=TaskOutput(answer='4'),
        expected_output=TaskOutput(answer='4'),
        metadata=TaskMetadata(difficulty='easy'),
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )


async def test_evaluator_spec_initialization():
    """Test initializing EvaluatorSpec."""
    # Simple form with just a name
    spec1 = EvaluatorSpec(call='my_evaluator')
    assert spec1.call == 'my_evaluator'
    assert spec1.args == []
    assert spec1.kwargs == {}

    # Form with args
    spec2 = EvaluatorSpec(call='my_evaluator', args=['arg1', 'arg2'])
    assert spec2.call == 'my_evaluator'
    assert spec2.args == ['arg1', 'arg2']
    assert spec2.kwargs == {}

    # Form with kwargs
    spec3 = EvaluatorSpec(call='my_evaluator', kwargs={'key1': 'value1', 'key2': 'value2'})
    assert spec3.call == 'my_evaluator'
    assert spec3.args == []
    assert spec3.kwargs == {'key1': 'value1', 'key2': 'value2'}

    # Full form
    spec4 = EvaluatorSpec(call='my_evaluator', args=['arg1'], kwargs={'key1': 'value1'})
    assert spec4.call == 'my_evaluator'
    assert spec4.args == ['arg1']
    assert spec4.kwargs == {'key1': 'value1'}


async def test_evaluator_spec_serialization():
    """Test serializing EvaluatorSpec."""
    # Create a spec
    spec = EvaluatorSpec(call='my_evaluator', args=['arg1'], kwargs={'key1': 'value1'})

    # Serialize it
    serialized = spec.model_dump()

    # Check the serialized form
    assert serialized['call'] == 'my_evaluator'
    assert serialized['args'] == ['arg1']
    assert serialized['kwargs'] == {'key1': 'value1'}

    # Deserialize it
    deserialized = EvaluatorSpec.model_validate(serialized)

    # Check the deserialized form
    assert deserialized.call == 'my_evaluator'
    assert deserialized.args == ['arg1']
    assert deserialized.kwargs == {'key1': 'value1'}


async def test_evaluator_from_function():
    """Test creating an Evaluator from a function."""

    async def test_evaluator_func(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        return {'result': 'test'}

    evaluator = Evaluator[TaskInput, TaskOutput, TaskMetadata].from_function(test_evaluator_func)

    assert evaluator.function is not None
    assert evaluator.spec.call == 'test_evaluator_func'


async def test_evaluator_call(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test calling an Evaluator."""

    async def test_evaluator_func(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        assert ctx.inputs.query == 'What is 2+2?'
        assert ctx.output.answer == '4'
        assert ctx.expected_output and ctx.expected_output.answer == '4'
        assert ctx.metadata and ctx.metadata.difficulty == 'easy'
        return {'result': 'passed'}

    evaluator = Evaluator[TaskInput, TaskOutput, TaskMetadata].from_function(test_evaluator_func)

    results = await evaluator.execute(test_context)
    # results is a list of SourcedEvaluatorOutput, so we need to get the value from it
    assert len(results) == 1
    assert results[0].value == 'passed'


async def test_is_instance_evaluator():
    """Test the is_instance evaluator."""
    # Create a context with the correct object typing for is_instance
    object_context = EvaluatorContext[object, object, object](
        name='test_case',
        inputs=TaskInput(query='What is 2+2?'),
        output=TaskOutput(answer='4'),
        expected_output=None,
        metadata=None,
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    # Test with matching types
    result = await is_instance(object_context, type_name='TaskOutput')
    assert isinstance(result, EvaluatorResult)
    assert result.value is True

    # Test with non-matching types
    class DifferentOutput(BaseModel):
        different_field: str

    # Create a context with DifferentOutput
    diff_context = EvaluatorContext[object, object, object](
        name='mismatch_case',
        inputs=TaskInput(query='What is 2+2?'),
        output=DifferentOutput(different_field='not an answer'),
        expected_output=None,
        metadata=None,
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    result = await is_instance(diff_context, type_name='TaskInput')
    assert isinstance(result, EvaluatorResult)
    assert result.value is False


async def test_llm_judge_evaluator():
    """Test the llm_judge evaluator."""
    # We can't easily test this without mocking the LLM, so we'll just check that it's importable
    assert callable(llm_judge)


async def test_custom_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test a custom evaluator."""

    async def custom_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        # Check if the answer is correct based on expected output
        is_correct = ctx.output.answer == ctx.expected_output.answer if ctx.expected_output else False

        # Use metadata if available
        difficulty = ctx.metadata.difficulty if ctx.metadata else 'unknown'

        return {
            'is_correct': is_correct,
            'difficulty': difficulty,
        }

    result = await custom_evaluator(test_context)
    assert result['is_correct'] is True
    assert result['difficulty'] == 'easy'


async def test_evaluator_error_handling(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test error handling in evaluators."""

    async def failing_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        raise ValueError('Simulated error')

    evaluator = Evaluator[TaskInput, TaskOutput, TaskMetadata].from_function(failing_evaluator)

    # When called directly, it should raise an error
    with pytest.raises(ValueError, match='Simulated error'):
        await evaluator.execute(test_context)


async def test_evaluator_with_null_values():
    """Test evaluator with null expected_output and metadata."""
    context = EvaluatorContext[TaskInput, TaskOutput, TaskMetadata](
        name=None,
        inputs=TaskInput(query='What is 2+2?'),
        output=TaskOutput(answer='4'),
        expected_output=None,
        metadata=None,
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    async def test_evaluator(ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> dict[str, Any]:
        return {
            'has_expected_output': ctx.expected_output is not None,
            'has_metadata': ctx.metadata is not None,
        }

    result = await test_evaluator(context)
    assert result['has_expected_output'] is False
    assert result['has_metadata'] is False


async def test_equals_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test the equals evaluator."""
    # Test with matching value
    result = await equals(test_context, TaskOutput(answer='4'))
    assert result is True

    # Test with non-matching value
    result = await equals(test_context, TaskOutput(answer='5'))
    assert result is False

    # Test with completely different type
    result = await equals(test_context, 'not a TaskOutput')
    assert result is False


async def test_equals_expected_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test the equals_expected evaluator."""
    # Test with matching expected output (already set in test_context)
    result = await equals_expected(test_context)
    assert 'equals_expected' in result
    assert result['equals_expected'] is True

    # Test with non-matching expected output
    context_with_different_expected = EvaluatorContext[TaskInput, TaskOutput, TaskMetadata](
        name='test_case',
        inputs=TaskInput(query='What is 2+2?'),
        output=TaskOutput(answer='4'),
        expected_output=TaskOutput(answer='5'),  # Different expected output
        metadata=TaskMetadata(difficulty='easy'),
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )
    result = await equals_expected(context_with_different_expected)
    assert 'equals_expected' in result
    assert result['equals_expected'] is False

    # Test with no expected output
    context_with_no_expected = EvaluatorContext[TaskInput, TaskOutput, TaskMetadata](
        name='test_case',
        inputs=TaskInput(query='What is 2+2?'),
        output=TaskOutput(answer='4'),
        expected_output=None,  # No expected output
        metadata=TaskMetadata(difficulty='easy'),
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )
    result = await equals_expected(context_with_no_expected)
    assert result == {}  # Should return empty dict when no expected output


async def test_contains_evaluator():
    """Test the contains evaluator."""
    # Test with string output
    string_context = EvaluatorContext[object, str, object](
        name='string_test',
        inputs="What's in the box?",
        output='There is a cat in the box',
        expected_output=None,
        metadata=None,
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    # String contains - case sensitive
    result = await contains(string_context, 'cat in the')
    assert result.value is True
    assert result.reason is None

    # String doesn't contain
    result = await contains(string_context, 'dog')
    assert result.value is False
    assert result.reason is not None

    # Case sensitivity
    result = await contains(string_context, 'CAT', case_sensitive=True)
    assert result.value is False

    result = await contains(string_context, 'CAT', case_sensitive=False)
    assert result.value is True

    # Test with list output
    list_context = EvaluatorContext[object, list[int], object](
        name='list_test',
        inputs='List items',
        output=[1, 2, 3, 4, 5],
        expected_output=None,
        metadata=None,
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    # List contains
    result = await contains(list_context, 3)
    assert result.value is True

    # List doesn't contain
    result = await contains(list_context, 6)
    assert result.value is False

    # Test with dict output
    dict_context = EvaluatorContext[object, dict[str, str], object](
        name='dict_test',
        inputs='Dict items',
        output={'key1': 'value1', 'key2': 'value2'},
        expected_output=None,
        metadata=None,
        duration=0.1,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )

    # Dict contains key
    result = await contains(dict_context, 'key1')
    assert result.value is True

    # Dict contains subset
    result = await contains(dict_context, {'key1': 'value1'})
    assert result.value is True

    # Dict doesn't contain key-value pair
    result = await contains(dict_context, {'key1': 'wrong_value'})
    assert result.value is False


async def test_max_duration_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test the max_duration evaluator."""
    from datetime import timedelta

    # Test with duration under the maximum (using float seconds)
    result = await max_duration(test_context, 0.2)  # test_context has duration=0.1
    assert result is True

    # Test with duration over the maximum
    result = await max_duration(test_context, 0.05)
    assert result is False

    # Test with timedelta
    result = await max_duration(test_context, timedelta(milliseconds=200))
    assert result is True

    result = await max_duration(test_context, timedelta(milliseconds=50))
    assert result is False


async def test_span_query_evaluator(
    capfire: CaptureLogfire,
):
    """Test the span_query evaluator."""
    import logfire

    from pydantic_evals.otel._context_in_memory_span_exporter_with_deps import context_subtree
    from pydantic_evals.otel.span_tree import SpanQuery

    # Create a span tree with a known structure
    with context_subtree() as tree:
        with logfire.span('root_span'):
            with logfire.span('child_span', type='important'):
                pass

    # Create a context with this span tree
    context = EvaluatorContext[object, object, object](
        name='span_test',
        inputs=None,
        output=None,
        expected_output=None,
        metadata=None,
        duration=0.1,
        _span_tree=tree,
        attributes={},
        metrics={},
    )

    # Test positive case: query that matches
    query: SpanQuery = {'name_equals': 'child_span', 'has_attributes': {'type': 'important'}}
    result = await span_query(context, query)
    assert result is True

    # Test negative case: query that doesn't match
    query = {'name_equals': 'non_existent_span'}
    result = await span_query(context, query)
    assert result is False


async def test_python_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test the python evaluator."""
    # Test with a simple condition
    assert await python(test_context, "ctx.output.answer == '4'") == snapshot(True)

    # Test type sensitivity
    assert await python(test_context, 'ctx.output.answer == 4') == snapshot(False)

    # Test with a named condition
    assert await python(test_context, "{'correct_answer': ctx.output.answer == '4'}") == snapshot(
        {'correct_answer': True}
    )

    # Test with a condition that returns false
    assert await python(test_context, "ctx.output.answer == '5'") == snapshot(False)

    # Test with a condition that accesses context properties
    assert await python(test_context, "ctx.output.answer == '4' and ctx.metadata.difficulty == 'easy'") == snapshot(
        True
    )

    # Test reason rendering for strings
    assert await python(test_context, 'ctx.output.answer') == snapshot('4')

    # Test with a condition that returns a dict
    assert await python(
        test_context, "{'is_correct': ctx.output.answer == '4', 'is_easy': ctx.metadata.difficulty == 'easy'}"
    ) == snapshot({'is_correct': True, 'is_easy': True})

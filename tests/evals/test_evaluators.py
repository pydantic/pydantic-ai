from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import Any, cast

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, TypeAdapter
from pydantic_core import to_jsonable_python

from pydantic_ai import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ..conftest import IsStr, try_import

with try_import() as imports_successful:
    import logfire
    from logfire.testing import CaptureLogfire

    from pydantic_evals.evaluators._run_evaluator import run_evaluator
    from pydantic_evals.evaluators.common import (
        Contains,
        Equals,
        EqualsExpected,
        HasMatchingSpan,
        IsInstance,
        LLMJudge,
        MaxDuration,
    )
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.evaluators.evaluator import (
        EvaluationReason,
        EvaluationResult,
        Evaluator,
        EvaluatorFailure,
        EvaluatorOutput,
    )
    from pydantic_evals.evaluators.spec import EvaluatorSpec
    from pydantic_evals.otel._context_in_memory_span_exporter import context_subtree
    from pydantic_evals.otel.span_tree import SpanQuery, SpanTree

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


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
    spec1 = EvaluatorSpec(name='MyEvaluator', arguments=None)
    assert spec1.name == 'MyEvaluator'
    assert spec1.args == ()
    assert spec1.kwargs == {}

    # Form with args - using a tuple with a single element containing a tuple
    args_tuple = cast(tuple[Any], (('arg1', 'arg2'),))
    spec2 = EvaluatorSpec(name='MyEvaluator', arguments=args_tuple)
    assert spec2.name == 'MyEvaluator'
    assert len(spec2.args) == 1
    assert spec2.args[0] == ('arg1', 'arg2')
    assert spec2.kwargs == {}

    # Form with kwargs
    spec3 = EvaluatorSpec(name='MyEvaluator', arguments={'key1': 'value1', 'key2': 'value2'})
    assert spec3.name == 'MyEvaluator'
    assert spec3.args == ()
    assert spec3.kwargs == {'key1': 'value1', 'key2': 'value2'}


async def test_evaluator_spec_serialization():
    """Test serializing EvaluatorSpec."""
    # Create a spec
    spec = EvaluatorSpec(name='MyEvaluator', arguments={'key1': 'value1'})

    adapter = TypeAdapter(EvaluatorSpec)
    assert adapter.dump_python(spec) == snapshot({'name': 'MyEvaluator', 'arguments': {'key1': 'value1'}})
    assert adapter.dump_python(spec, context={'use_short_form': True}) == snapshot({'MyEvaluator': {'key1': 'value1'}})

    # Test string serialization
    spec_simple = EvaluatorSpec(name='MyEvaluator', arguments=None)
    assert adapter.dump_python(spec_simple) == snapshot({'name': 'MyEvaluator', 'arguments': None})
    assert adapter.dump_python(spec_simple, context={'use_short_form': True}) == snapshot('MyEvaluator')

    # Test single arg serialization
    single_arg = cast(tuple[Any], ('value1',))
    spec_single_arg = EvaluatorSpec(name='MyEvaluator', arguments=single_arg)
    assert adapter.dump_python(spec_single_arg) == snapshot({'name': 'MyEvaluator', 'arguments': ('value1',)})
    assert adapter.dump_python(spec_single_arg, context={'use_short_form': True}) == snapshot({'MyEvaluator': 'value1'})


async def test_llm_judge_serialization():
    # Ensure models are serialized based on their system + name when used with LLMJudge

    class MyModel(Model):
        async def request(
            self,
            messages: list[ModelMessage],
            model_settings: ModelSettings | None,
            model_request_parameters: ModelRequestParameters,
        ) -> ModelResponse:
            raise NotImplementedError

        @property
        def model_name(self) -> str:
            return 'my-model'

        @property
        def system(self) -> str:
            return 'my-system'

    adapter = TypeAdapter(Evaluator)

    assert adapter.dump_python(LLMJudge(rubric='my rubric', model=MyModel())) == {
        'name': 'LLMJudge',
        'arguments': {'model': 'my-system:my-model', 'rubric': 'my rubric'},
    }


async def test_evaluator_call(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test calling an Evaluator."""

    @dataclass
    class ExampleEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        """A test evaluator for testing purposes."""

        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> EvaluatorOutput:
            assert ctx.inputs.query == 'What is 2+2?'
            assert ctx.output.answer == '4'
            assert ctx.expected_output and ctx.expected_output.answer == '4'
            assert ctx.metadata and ctx.metadata.difficulty == 'easy'
            return {'result': 'passed'}

    evaluator = ExampleEvaluator()
    results = await run_evaluator(evaluator, test_context)

    assert not isinstance(results, EvaluatorFailure)
    assert len(results) == 1
    first_result = results[0]
    assert isinstance(first_result, EvaluationResult)
    assert first_result.name == 'result'
    assert first_result.value == 'passed'
    assert first_result.reason is None
    assert first_result.source == EvaluatorSpec(name='ExampleEvaluator', arguments=None)


async def test_is_instance_evaluator():
    """Test the IsInstance evaluator."""
    # Create a context with the correct object typing for IsInstance
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
    evaluator = IsInstance(type_name='TaskOutput')
    result = evaluator.evaluate(object_context)
    assert isinstance(result, EvaluationReason)
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

    result = evaluator.evaluate(diff_context)
    assert isinstance(result, EvaluationReason)
    assert result.value is False


async def test_custom_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test a custom evaluator."""

    @dataclass
    class CustomEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> EvaluatorOutput:
            # Check if the answer is correct based on expected output
            is_correct = ctx.output.answer == ctx.expected_output.answer if ctx.expected_output else False

            # Use metadata if available
            difficulty = ctx.metadata.difficulty if ctx.metadata else 'unknown'

            return {
                'is_correct': is_correct,
                'difficulty': difficulty,
            }

    evaluator = CustomEvaluator()
    result = evaluator.evaluate(test_context)
    assert result == snapshot({'difficulty': 'easy', 'is_correct': True})


async def test_custom_evaluator_name(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    @dataclass
    class CustomNameFieldEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        result: int
        evaluation_name: str

        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> EvaluatorOutput:
            return self.result

    evaluator = CustomNameFieldEvaluator(result=123, evaluation_name='abc')

    assert to_jsonable_python(await run_evaluator(evaluator, test_context)) == snapshot(
        [
            {
                'name': 'abc',
                'reason': None,
                'source': {'arguments': {'evaluation_name': 'abc', 'result': 123}, 'name': 'CustomNameFieldEvaluator'},
                'value': 123,
            }
        ]
    )

    @dataclass
    class CustomNamePropertyEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        result: int
        my_name: str

        @property
        def evaluation_name(self) -> str:
            return f'hello {self.my_name}'

        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> EvaluatorOutput:
            return self.result

    evaluator = CustomNamePropertyEvaluator(result=123, my_name='marcelo')

    assert to_jsonable_python(await run_evaluator(evaluator, test_context)) == snapshot(
        [
            {
                'name': 'hello marcelo',
                'reason': None,
                'source': {'arguments': {'my_name': 'marcelo', 'result': 123}, 'name': 'CustomNamePropertyEvaluator'},
                'value': 123,
            }
        ]
    )


async def test_evaluator_error_handling(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test error handling in evaluators."""

    @dataclass
    class FailingEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> EvaluatorOutput:
            raise ValueError('Simulated error')

    evaluator = FailingEvaluator()

    # When called directly, it should raise an error
    result = await run_evaluator(evaluator, test_context)
    assert result == EvaluatorFailure(
        name='FailingEvaluator',
        error_message='ValueError: Simulated error',
        error_stacktrace=IsStr(),
        source=FailingEvaluator().as_spec(),
    )


async def test_evaluator_with_null_values():
    """Test evaluator with null expected_output and metadata."""

    @dataclass
    class NullValueEvaluator(Evaluator[TaskInput, TaskOutput, TaskMetadata]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]) -> EvaluatorOutput:
            return {
                'has_expected_output': ctx.expected_output is not None,
                'has_metadata': ctx.metadata is not None,
            }

    evaluator = NullValueEvaluator()
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

    result = evaluator.evaluate(context)
    assert isinstance(result, dict)
    assert result['has_expected_output'] is False
    assert result['has_metadata'] is False


async def test_equals_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test the equals evaluator."""
    # Test with matching value
    evaluator = Equals(value=TaskOutput(answer='4'))
    result = evaluator.evaluate(test_context)
    assert result is True

    # Test with non-matching value
    evaluator = Equals(value=TaskOutput(answer='5'))
    result = evaluator.evaluate(test_context)
    assert result is False

    # Test with completely different type
    evaluator = Equals(value='not a TaskOutput')
    result = evaluator.evaluate(test_context)
    assert result is False


async def test_equals_expected_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test the equals_expected evaluator."""
    # Test with matching expected output (already set in test_context)
    evaluator = EqualsExpected()
    result = evaluator.evaluate(test_context)
    assert result is True

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
    result = evaluator.evaluate(context_with_different_expected)
    assert result is False

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
    result = evaluator.evaluate(context_with_no_expected)
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
    evaluator = Contains(value='cat in the')
    assert evaluator.evaluate(string_context) == snapshot(EvaluationReason(value=True))

    # String doesn't contain
    evaluator = Contains(value='dog')
    assert evaluator.evaluate(string_context) == snapshot(
        EvaluationReason(
            value=False,
            reason="Output string 'There is a cat in the box' does not contain expected string 'dog'",
        )
    )

    # Very long strings don't get included in reason
    evaluator = Contains(value='a' * 1000)
    assert evaluator.evaluate(string_context) == snapshot(
        EvaluationReason(
            value=False,
            reason="Output string 'There is a cat in the box' does not contain expected string 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa...aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        )
    )

    # Case sensitivity
    evaluator = Contains(value='CAT', case_sensitive=True)
    assert evaluator.evaluate(string_context) == snapshot(
        EvaluationReason(
            value=False,
            reason="Output string 'There is a cat in the box' does not contain expected string 'CAT'",
        )
    )

    evaluator = Contains(value='CAT', case_sensitive=False)
    assert evaluator.evaluate(string_context) == snapshot(EvaluationReason(value=True))

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
    evaluator = Contains(value=3)
    assert evaluator.evaluate(list_context) == snapshot(EvaluationReason(value=True))

    # List doesn't contain
    evaluator = Contains(value=6)
    assert evaluator.evaluate(list_context) == snapshot(
        EvaluationReason(value=False, reason='Output [1, 2, 3, 4, 5] does not contain provided value')
    )

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
    evaluator = Contains(value='key1')
    assert evaluator.evaluate(dict_context) == snapshot(EvaluationReason(value=True))

    # Dict contains subset
    evaluator = Contains(value={'key1': 'value1'})
    assert evaluator.evaluate(dict_context) == snapshot(EvaluationReason(value=True))

    # Dict doesn't contain key-value pair
    evaluator = Contains(value={'key1': 'wrong_value'})
    assert evaluator.evaluate(dict_context) == snapshot(
        EvaluationReason(
            value=False,
            reason="Output dictionary has different value for key 'key1': 'value1' != 'wrong_value'",
        )
    )

    # Dict doesn't contain key
    evaluator = Contains(value='key3')
    assert evaluator.evaluate(dict_context) == snapshot(
        EvaluationReason(
            value=False,
            reason="Output {'key1': 'value1', 'key2': 'value2'} does not contain provided value as a key",
        )
    )

    # Very long keys are truncated
    evaluator = Contains(value={'key1' * 500: 'wrong_value'})
    assert evaluator.evaluate(dict_context) == snapshot(
        EvaluationReason(
            value=False,
            reason="Output dictionary does not contain expected key 'key1key1key1ke...y1key1key1key1'",
        )
    )

    evaluator = Contains(value={'key1': 'wrong_value_' * 500})
    assert evaluator.evaluate(dict_context) == snapshot(
        EvaluationReason(
            value=False,
            reason="Output dictionary has different value for key 'key1': 'value1' != 'wrong_value_wrong_value_wrong_value_wrong_value_w..._wrong_value_wrong_value_wrong_value_wrong_value_'",
        )
    )


async def test_max_duration_evaluator(test_context: EvaluatorContext[TaskInput, TaskOutput, TaskMetadata]):
    """Test the max_duration evaluator."""
    from datetime import timedelta

    # Test with duration under the maximum (using float seconds)
    evaluator = MaxDuration(seconds=0.2)  # test_context has duration=0.1
    result = evaluator.evaluate(test_context)
    assert result is True

    # Test with duration over the maximum
    evaluator = MaxDuration(seconds=0.05)
    result = evaluator.evaluate(test_context)
    assert result is False

    # Test with timedelta
    evaluator = MaxDuration(seconds=timedelta(milliseconds=200))
    result = evaluator.evaluate(test_context)
    assert result is True

    evaluator = MaxDuration(seconds=timedelta(milliseconds=50))
    result = evaluator.evaluate(test_context)
    assert result is False


async def test_span_query_evaluator(
    capfire: CaptureLogfire,
):
    """Test the span_query evaluator."""

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
    evaluator = HasMatchingSpan(query=query)
    result = evaluator.evaluate(context)
    assert result is True

    # Test negative case: query that doesn't match
    query = {'name_equals': 'non_existent_span'}
    evaluator = HasMatchingSpan(query=query)
    result = evaluator.evaluate(context)
    assert result is False


async def test_import_errors():
    with pytest.raises(
        ImportError,
        match='The `Python` evaluator has been removed for security reasons. See https://github.com/pydantic/pydantic-ai/pull/2808 for more details and a workaround.',
    ):
        from pydantic_evals.evaluators import Python  # pyright: ignore[reportUnusedImport]

    with pytest.raises(
        ImportError,
        match='The `Python` evaluator has been removed for security reasons. See https://github.com/pydantic/pydantic-ai/pull/2808 for more details and a workaround.',
    ):
        from pydantic_evals.evaluators.common import Python  # pyright: ignore[reportUnusedImport] # noqa: F401

    with pytest.raises(
        ImportError,
        match="cannot import name 'Foo' from 'pydantic_evals.evaluators'",
    ):
        from pydantic_evals.evaluators import Foo  # pyright: ignore[reportUnusedImport]

    with pytest.raises(
        ImportError,
        match="cannot import name 'Foo' from 'pydantic_evals.evaluators.common'",
    ):
        from pydantic_evals.evaluators.common import Foo  # pyright: ignore[reportUnusedImport] # noqa: F401

    with pytest.raises(
        AttributeError,
        match="module 'pydantic_evals.evaluators' has no attribute 'Foo'",
    ):
        import pydantic_evals.evaluators as _evaluators

        _evaluators.Foo

    with pytest.raises(
        AttributeError,
        match="module 'pydantic_evals.evaluators.common' has no attribute 'Foo'",
    ):
        import pydantic_evals.evaluators.common as _common

        _common.Foo

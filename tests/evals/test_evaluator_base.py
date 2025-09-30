from __future__ import annotations as _annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals.evaluators._run_evaluator import run_evaluator
    from pydantic_evals.evaluators.context import EvaluatorContext
    from pydantic_evals.evaluators.evaluator import (
        EvaluationReason,
        EvaluationResult,
        Evaluator,
        EvaluatorFailure,
    )
    from pydantic_evals.otel._errors import SpanTreeRecordingError

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'), pytest.mark.anyio]


def test_evaluation_reason():
    """Test EvaluationReason class."""
    # Test with value only
    reason = EvaluationReason(value=True)
    assert reason.value is True
    assert reason.reason is None

    # Test with value and reason
    reason = EvaluationReason(value=42, reason='Perfect score')
    assert reason.value == 42
    assert reason.reason == 'Perfect score'

    # Test with string value
    reason = EvaluationReason(value='pass', reason='Test passed')
    assert reason.value == 'pass'
    assert reason.reason == 'Test passed'


def test_evaluation_result():
    """Test EvaluationResult class."""

    @dataclass
    class DummyEvaluator(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    evaluator = DummyEvaluator()

    # Test basic result
    result = EvaluationResult(name='test', value=True, reason='Success', source=evaluator.as_spec())
    assert result.name == 'test'
    assert result.value is True
    assert result.reason == 'Success'
    assert result.source == evaluator.as_spec()

    # Test downcast with matching type
    downcast = result.downcast(bool)
    assert downcast is not None
    assert downcast.value is True

    # Test downcast with non-matching type
    downcast = result.downcast(int)
    assert downcast is None

    # Test downcast with multiple types
    downcast = result.downcast(int, bool)
    assert downcast is not None
    assert downcast.value is True


def test_strict_abc_meta():
    """Test _StrictABCMeta metaclass."""
    # Test that abstract methods must be implemented
    with pytest.raises(TypeError) as exc_info:

        @dataclass
        class InvalidEvaluator(Evaluator[Any, Any, Any]):  # pyright: ignore[reportUnusedClass]
            pass

    assert 'must implement all abstract methods' in str(exc_info.value)
    assert 'evaluate' in str(exc_info.value)


if TYPE_CHECKING or imports_successful():  # pragma: no branch

    @dataclass
    class SimpleEvaluator(Evaluator[Any, Any, Any]):
        value: Any = True
        reason: str | None = None

        def evaluate(self, ctx: EvaluatorContext) -> bool | EvaluationReason:
            if self.reason is not None:
                return EvaluationReason(value=self.value, reason=self.reason)
            return self.value

    @dataclass
    class AsyncEvaluator(Evaluator[Any, Any, Any]):
        value: Any = True
        delay: float = 0.1

        async def evaluate(self, ctx: EvaluatorContext) -> bool:
            await asyncio.sleep(self.delay)
            return self.value

    @dataclass
    class MultiEvaluator(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> dict[str, bool]:
            return {'test1': True, 'test2': False}


async def test_evaluator_sync():
    """Test synchronous evaluator execution."""
    ctx = EvaluatorContext(
        name='test',
        inputs={},
        metadata=None,
        expected_output=None,
        output={},
        duration=0.0,
        _span_tree=SpanTreeRecordingError('did not record spans'),
        attributes={},
        metrics={},
    )

    # Test simple boolean result
    evaluator = SimpleEvaluator()
    result = evaluator.evaluate_sync(ctx)
    assert result is True

    # Test with reason
    evaluator = SimpleEvaluator(value=False, reason='Failed')
    result = evaluator.evaluate_sync(ctx)
    assert isinstance(result, EvaluationReason)
    assert result.value is False
    assert result.reason == 'Failed'

    # Test with dictionary result
    evaluator = MultiEvaluator()
    result = evaluator.evaluate_sync(ctx)
    assert isinstance(result, dict)
    assert result['test1'] is True
    assert result['test2'] is False


async def test_evaluator_async():
    """Test asynchronous evaluator execution."""
    ctx = EvaluatorContext(
        name='test',
        inputs={},
        metadata=None,
        expected_output=None,
        output={},
        duration=0.0,
        _span_tree=SpanTreeRecordingError('did not record spans'),
        attributes={},
        metrics={},
    )

    # Test async evaluator
    evaluator = AsyncEvaluator()
    result = await evaluator.evaluate_async(ctx)
    assert result is True

    # Test sync evaluator with async execution
    evaluator = SimpleEvaluator()
    result = await evaluator.evaluate_async(ctx)
    assert result is True


async def test_evaluation_name():
    """Test evaluator name method."""
    evaluator = SimpleEvaluator()
    assert evaluator.get_serialization_name() == 'SimpleEvaluator'
    assert evaluator.get_default_evaluation_name() == 'SimpleEvaluator'


async def test_evaluator_serialization():
    """Test evaluator serialization."""

    @dataclass
    class ExampleEvaluator(Evaluator[Any, Any, Any]):
        value: int = 42
        optional: str | None = None
        default_value: bool = False
        default_factory_value: list[int] = field(default_factory=list)

        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    # Test with default values
    evaluator = ExampleEvaluator()
    adapter = TypeAdapter(Evaluator)
    assert adapter.dump_python(evaluator, context=None) == snapshot({'arguments': None, 'name': 'ExampleEvaluator'})
    assert adapter.dump_python(evaluator, context={'use_short_form': True}) == snapshot('ExampleEvaluator')

    # Test with a single non-default value
    evaluator = ExampleEvaluator(value=100)
    assert adapter.dump_python(evaluator, context=None) == snapshot({'arguments': [100], 'name': 'ExampleEvaluator'})
    assert adapter.dump_python(evaluator, context={'use_short_form': True}) == snapshot({'ExampleEvaluator': 100})

    # Test with multiple non-default values
    evaluator = ExampleEvaluator(value=100, optional='test', default_value=True)
    assert adapter.dump_python(evaluator, context=None) == snapshot(
        {'arguments': {'default_value': True, 'optional': 'test', 'value': 100}, 'name': 'ExampleEvaluator'}
    )
    assert adapter.dump_python(evaluator, context={'use_short_form': True}) == snapshot(
        {'ExampleEvaluator': {'default_value': True, 'optional': 'test', 'value': 100}}
    )

    # Test with no arguments
    @dataclass
    class NoArgsEvaluator(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    evaluator = NoArgsEvaluator()
    assert adapter.dump_python(evaluator, context=None) == snapshot({'arguments': None, 'name': 'NoArgsEvaluator'})
    assert adapter.dump_python(evaluator, context={'use_short_form': True}) == snapshot('NoArgsEvaluator')


async def test_run_evaluator():
    """Test run_evaluator function."""
    ctx = EvaluatorContext(
        name='test',
        inputs={},
        metadata=None,
        expected_output=None,
        output={},
        duration=0.0,
        _span_tree=SpanTreeRecordingError('did not record spans'),
        attributes={},
        metrics={},
    )

    # Test with simple boolean result
    evaluator = SimpleEvaluator()
    results = await run_evaluator(evaluator, ctx)
    adapter = TypeAdapter[Sequence[EvaluationResult] | EvaluatorFailure](Sequence[EvaluationResult] | EvaluatorFailure)
    assert adapter.dump_python(results) == snapshot(
        [
            {
                'name': 'SimpleEvaluator',
                'reason': None,
                'source': {'arguments': None, 'name': 'SimpleEvaluator'},
                'value': True,
            }
        ]
    )

    # Test with reason
    evaluator = SimpleEvaluator(value=False, reason='Failed')
    results = await run_evaluator(evaluator, ctx)
    assert adapter.dump_python(results) == snapshot(
        [
            {
                'name': 'SimpleEvaluator',
                'reason': 'Failed',
                'source': {'arguments': {'reason': 'Failed', 'value': False}, 'name': 'SimpleEvaluator'},
                'value': False,
            }
        ]
    )

    # Test with dictionary result
    evaluator = MultiEvaluator()
    results = await run_evaluator(evaluator, ctx)
    assert adapter.dump_python(results) == snapshot(
        [
            {'name': 'test1', 'reason': None, 'source': {'arguments': None, 'name': 'MultiEvaluator'}, 'value': True},
            {'name': 'test2', 'reason': None, 'source': {'arguments': None, 'name': 'MultiEvaluator'}, 'value': False},
        ]
    )

    # Test with async evaluator
    evaluator = AsyncEvaluator()
    results = await run_evaluator(evaluator, ctx)
    assert adapter.dump_python(results) == snapshot(
        [
            {
                'name': 'AsyncEvaluator',
                'reason': None,
                'source': {'arguments': None, 'name': 'AsyncEvaluator'},
                'value': True,
            }
        ]
    )

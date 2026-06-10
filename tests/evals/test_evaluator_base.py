from __future__ import annotations as _annotations

import asyncio
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import TypeAdapter

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals import PydanticEvalsDeprecationWarning
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


def test_evaluation_result_positional_construction_warns():
    """Positional construction of `EvaluationResult` is deprecated ahead of v2 making it kw-only."""

    @dataclass
    class DummyEvaluator(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    source = DummyEvaluator().as_spec()
    with pytest.warns(PydanticEvalsDeprecationWarning, match='positional arguments is deprecated'):
        result = EvaluationResult('test', True, 'Success', source)
    assert result.name == 'test'
    assert result.value is True
    assert result.reason == 'Success'
    assert result.source == source


def test_evaluator_failure_positional_construction_warns():
    """Positional construction of `EvaluatorFailure` is deprecated ahead of v2 making it kw-only."""

    @dataclass
    class DummyEvaluator(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    source = DummyEvaluator().as_spec()
    with pytest.warns(PydanticEvalsDeprecationWarning, match='positional arguments is deprecated'):
        failure = EvaluatorFailure('test', 'boom', 'traceback...', source)
    assert failure.name == 'test'
    assert failure.error_message == 'boom'
    assert failure.error_stacktrace == 'traceback...'
    assert failure.source == source


def test_evaluation_result_kwargs_does_not_warn():
    """Keyword construction should not emit a deprecation warning."""

    @dataclass
    class DummyEvaluator(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    source = DummyEvaluator().as_spec()
    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticEvalsDeprecationWarning)
        EvaluationResult(name='test', value=True, reason='Success', source=source)
        EvaluatorFailure(name='test', error_message='boom', error_stacktrace='tb', source=source)


def test_strict_abc_meta():
    """Test _StrictABCMeta metaclass."""
    from abc import abstractmethod

    from pydantic_evals.evaluators.report_evaluator import ReportEvaluator

    # Subclasses that don't implement inherited abstract methods are rejected at definition time
    with pytest.raises(TypeError, match="must implement all abstract methods.*'evaluate'"):

        @dataclass
        class InvalidEvaluator(Evaluator[Any, Any, Any]):  # pyright: ignore[reportUnusedClass]
            pass

    with pytest.raises(TypeError, match="must implement all abstract methods.*'evaluate'"):

        @dataclass
        class InvalidReportEvaluator(ReportEvaluator[Any, Any, Any]):  # pyright: ignore[reportUnusedClass]
            pass

    # Subclasses that add new abstract methods but don't implement inherited ones are also rejected
    with pytest.raises(TypeError, match="must implement all abstract methods.*'evaluate'"):

        @dataclass
        class PartialAbstract(Evaluator[Any, Any, Any]):  # pyright: ignore[reportUnusedClass]
            @abstractmethod
            def other(self) -> None: ...

    # Classes that define their own abstract methods (new abstract layers) are allowed
    # — this is how Evaluator and ReportEvaluator themselves work
    assert hasattr(Evaluator, '__abstractmethods__')
    assert 'evaluate' in Evaluator.__abstractmethods__
    assert hasattr(ReportEvaluator, '__abstractmethods__')
    assert 'evaluate' in ReportEvaluator.__abstractmethods__


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


def test_evaluation_name_attribute_emits_deprecation_warning():
    """Relying on the `evaluation_name` attribute to customize the default name is deprecated."""

    @dataclass
    class CustomNameViaAttr(Evaluator[Any, Any, Any]):
        evaluation_name: str | None = 'custom'

        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    evaluator = CustomNameViaAttr()
    with pytest.warns(PydanticEvalsDeprecationWarning, match='evaluation_name'):
        assert evaluator.get_default_evaluation_name() == 'custom'


def test_evaluation_name_method_override_does_not_warn():
    """Overriding `get_default_evaluation_name` is the supported, warning-free path."""

    @dataclass
    class CustomNameViaMethod(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

        def get_default_evaluation_name(self) -> str:
            return 'overridden'

    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticEvalsDeprecationWarning)
        assert CustomNameViaMethod().get_default_evaluation_name() == 'overridden'


def test_evaluator_version_default_is_none():
    """The base `get_evaluator_version` returns None when no version is declared."""

    @dataclass
    class Unversioned(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    assert Unversioned().get_evaluator_version() is None


def test_evaluator_version_attribute_emits_deprecation_warning():
    """Relying on the `evaluator_version` attribute is deprecated."""

    @dataclass
    class VersionedViaAttr(Evaluator[Any, Any, Any]):
        evaluator_version = 'v2'

        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    with pytest.warns(PydanticEvalsDeprecationWarning, match='evaluator_version'):
        assert VersionedViaAttr().get_evaluator_version() == 'v2'


def test_evaluator_version_method_override_does_not_warn():
    """Overriding `get_evaluator_version` is the supported, warning-free path."""

    @dataclass
    class VersionedViaMethod(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

        def get_evaluator_version(self) -> str | None:
            return 'v3'

    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticEvalsDeprecationWarning)
        assert VersionedViaMethod().get_evaluator_version() == 'v3'


def test_builtin_evaluators_with_evaluation_name_do_not_warn():
    """Built-in evaluators that expose `evaluation_name` as a dataclass field shouldn't self-warn."""
    from pydantic_evals.evaluators.common import Equals

    with warnings.catch_warnings():
        warnings.simplefilter('error', PydanticEvalsDeprecationWarning)
        assert Equals(value=42, evaluation_name='int_match').get_default_evaluation_name() == 'int_match'
        assert Equals(value=42).get_default_evaluation_name() == 'Equals'


async def test_evaluator_serialization():
    """Test evaluator serialization."""

    @dataclass
    class ExampleEvaluator(Evaluator[Any, Any, Any]):
        value: int = 42
        optional: str | None = None
        default_value: bool = False
        default_factory_value: list[int] = field(default_factory=list[int])

        def evaluate(self, ctx: EvaluatorContext) -> bool:
            raise NotImplementedError

    # Test with default values
    evaluator = ExampleEvaluator()
    adapter = TypeAdapter(Evaluator)
    assert adapter.dump_python(evaluator, context=None) == snapshot({'arguments': None, 'name': 'ExampleEvaluator'})
    assert adapter.dump_python(evaluator, context={'use_short_form': True}) == snapshot('ExampleEvaluator')

    # Test with a single non-default value (first field) — uses tuple form
    evaluator = ExampleEvaluator(value=100)
    assert adapter.dump_python(evaluator, context=None) == snapshot({'arguments': [100], 'name': 'ExampleEvaluator'})
    assert adapter.dump_python(evaluator, context={'use_short_form': True}) == snapshot({'ExampleEvaluator': 100})

    # Test with a single non-default value (non-first field) — uses dict form
    evaluator = ExampleEvaluator(optional='test')
    spec = evaluator.as_spec()
    assert spec.arguments == {'optional': 'test'}
    assert adapter.dump_python(evaluator, context=None) == snapshot(
        {'arguments': {'optional': 'test'}, 'name': 'ExampleEvaluator'}
    )
    assert adapter.dump_python(evaluator, context={'use_short_form': True}) == snapshot(
        {'ExampleEvaluator': {'optional': 'test'}}
    )

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
                'evaluator_version': None,
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
                'evaluator_version': None,
            }
        ]
    )

    # Test with dictionary result
    evaluator = MultiEvaluator()
    results = await run_evaluator(evaluator, ctx)
    assert adapter.dump_python(results) == snapshot(
        [
            {
                'name': 'test1',
                'reason': None,
                'source': {'arguments': None, 'name': 'MultiEvaluator'},
                'value': True,
                'evaluator_version': None,
            },
            {
                'name': 'test2',
                'reason': None,
                'source': {'arguments': None, 'name': 'MultiEvaluator'},
                'value': False,
                'evaluator_version': None,
            },
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
                'evaluator_version': None,
            }
        ]
    )


@pytest.mark.parametrize(
    'output',
    [
        pytest.param(math.nan, id='nan-scalar'),
        pytest.param(math.inf, id='inf-scalar'),
        pytest.param(-math.inf, id='negative-inf-scalar'),
        pytest.param({'score': math.nan}, id='nan-mapping'),
        pytest.param(EvaluationReason(value=math.nan, reason='not finite'), id='nan-reason'),
        pytest.param({'score': EvaluationReason(value=math.inf, reason='not finite')}, id='inf-mapped-reason'),
    ],
)
async def test_run_evaluator_rejects_non_finite_numbers(output: Any):
    """Adapter validation should reject non-finite numbers before report conversion."""
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

    @dataclass
    class NonFiniteEvaluator(Evaluator[Any, Any, Any]):
        def evaluate(self, ctx: EvaluatorContext) -> Any:
            return output

    results = await run_evaluator(NonFiniteEvaluator(), ctx)

    assert isinstance(results, EvaluatorFailure)
    assert results.name == 'NonFiniteEvaluator'
    assert results.error_type == 'ValueError'
    assert 'returned a value of an invalid type' in results.error_message

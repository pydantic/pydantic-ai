"""Tests for using plain functions as evaluators.

These run local tasks and evaluators, so there's no provider boundary to record.
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Generator, Sequence
from dataclasses import dataclass
from functools import partial
from operator import attrgetter
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml
from pydantic_core import PydanticSerializationError

from .._inline_snapshot import snapshot
from ..conftest import IsStr, try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import (
        Equals,
        EvaluationReason,
        EvaluationResult,
        Evaluator,
        EvaluatorContext,
        EvaluatorFailure,
        EvaluatorFunc,
        EvaluatorOutput,
        EvaluatorSpec,
        FunctionEvaluator,
    )
    from pydantic_evals.online import OnlineEvalConfig, OnlineEvaluator, run_evaluators, wait_for_evaluations
    from pydantic_evals.online_capability import OnlineEvaluation
    from pydantic_evals.otel.span_tree import SpanTree

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

pytestmark = pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed')

needs_logfire = pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed')


def exact_match(ctx: EvaluatorContext[str, str, None]) -> bool:
    return ctx.output == ctx.expected_output


async def output_length(ctx: EvaluatorContext[str, str, None]) -> EvaluationReason:
    return EvaluationReason(value=len(ctx.output), reason='number of characters')


def checks(ctx: EvaluatorContext[str, str, None]) -> EvaluatorOutput:
    return {'is_upper': ctx.output.isupper(), 'label': 'shout'}


def has_prefix(ctx: EvaluatorContext[str, str, None], prefix: str) -> bool:
    return ctx.output.startswith(prefix)


def broken(ctx: EvaluatorContext[str, str, None]) -> bool:
    raise ValueError('boom')


class AsyncCallable:
    def __call__(self, ctx: EvaluatorContext[str, str, None]) -> Awaitable[EvaluatorOutput]:
        return self._evaluate(ctx)

    async def _evaluate(self, ctx: EvaluatorContext[str, str, None]) -> EvaluatorOutput:
        return ctx.output == 'HELLO'


@dataclass
class PrefixCheck:
    prefix: str

    def check(self, ctx: EvaluatorContext[str, str, None]) -> bool:
        return ctx.output.startswith(self.prefix)  # pragma: no cover


class Deferred:
    """An awaitable that isn't a coroutine, like an `asyncio.Task`."""

    def __init__(self, value: EvaluatorOutput) -> None:
        self.value = value

    def __await__(self) -> Generator[Any, Any, EvaluatorOutput]:
        return self._resolve().__await__()

    async def _resolve(self) -> EvaluatorOutput:
        return self.value


def deferred(ctx: EvaluatorContext[str, str, None]) -> Awaitable[EvaluatorOutput]:
    return Deferred(ctx.output == 'HELLO')


def upper(inputs: str) -> str:
    return inputs.upper()


def _make_context(output: str) -> EvaluatorContext[str, str, None]:
    return EvaluatorContext(
        name='test',
        inputs='input',
        output=output,
        expected_output=None,
        metadata=None,
        duration=0.0,
        _span_tree=SpanTree(),
        attributes={},
        metrics={},
    )


async def test_dataset_and_case_function_evaluators():
    """Sync, async, mapping-returning, and lambda functions run at dataset and case level, named after the function."""
    dataset = Dataset(
        name='functions',
        cases=[
            Case(name='hello', inputs='hello', expected_output='HELLO', evaluators=[lambda ctx: ctx.output == 'HELLO']),
            Case(name='world', inputs='world', expected_output='nope'),
        ],
        evaluators=[exact_match, output_length, checks],
    )
    dataset.add_evaluator(partial(has_prefix, prefix='H'), specific_case='hello')
    dataset.add_evaluator(AsyncCallable())

    report = await dataset.evaluate(upper, progress=False)

    assert [(case.name, case.assertions, case.scores, case.labels) for case in report.cases] == snapshot(
        [
            (
                'hello',
                {
                    'exact_match': EvaluationResult(
                        name='exact_match',
                        value=True,
                        reason=None,
                        source=EvaluatorSpec(name='exact_match', arguments=None),
                    ),
                    'is_upper': EvaluationResult(
                        name='is_upper', value=True, reason=None, source=EvaluatorSpec(name='checks', arguments=None)
                    ),
                    'AsyncCallable.__call__': EvaluationResult(
                        name='AsyncCallable.__call__',
                        value=True,
                        reason=None,
                        source=EvaluatorSpec(name='AsyncCallable.__call__', arguments=None),
                    ),
                    '<lambda>': EvaluationResult(
                        name='<lambda>', value=True, reason=None, source=EvaluatorSpec(name='<lambda>', arguments=None)
                    ),
                    'has_prefix': EvaluationResult(
                        name='has_prefix',
                        value=True,
                        reason=None,
                        source=EvaluatorSpec(name='has_prefix', arguments=None),
                    ),
                },
                {
                    'output_length': EvaluationResult(
                        name='output_length',
                        value=5,
                        reason='number of characters',
                        source=EvaluatorSpec(name='output_length', arguments=None),
                    )
                },
                {
                    'label': EvaluationResult(
                        name='label', value='shout', reason=None, source=EvaluatorSpec(name='checks', arguments=None)
                    )
                },
            ),
            (
                'world',
                {
                    'exact_match': EvaluationResult(
                        name='exact_match',
                        value=False,
                        reason=None,
                        source=EvaluatorSpec(name='exact_match', arguments=None),
                    ),
                    'is_upper': EvaluationResult(
                        name='is_upper', value=True, reason=None, source=EvaluatorSpec(name='checks', arguments=None)
                    ),
                    'AsyncCallable.__call__': EvaluationResult(
                        name='AsyncCallable.__call__',
                        value=False,
                        reason=None,
                        source=EvaluatorSpec(name='AsyncCallable.__call__', arguments=None),
                    ),
                },
                {
                    'output_length': EvaluationResult(
                        name='output_length',
                        value=5,
                        reason='number of characters',
                        source=EvaluatorSpec(name='output_length', arguments=None),
                    )
                },
                {
                    'label': EvaluationResult(
                        name='label', value='shout', reason=None, source=EvaluatorSpec(name='checks', arguments=None)
                    )
                },
            ),
        ]
    )


async def test_add_case_and_failure():
    """A function passed to `add_case` runs on that case, and one that raises is reported as an evaluator failure."""
    dataset = Dataset[str, str, None](name='failures', cases=[])
    dataset.add_case(name='case', inputs='hello', evaluators=[broken])

    report = await dataset.evaluate(upper, progress=False)

    assert report.cases[0].evaluator_failures == snapshot(
        [
            EvaluatorFailure(
                name='broken',
                error_message='ValueError: boom',
                error_stacktrace=IsStr(),
                source=EvaluatorSpec(name='broken', arguments=None),
                error_type='ValueError',
            )
        ]
    )


async def test_function_evaluator_name_and_version():
    """An explicit `FunctionEvaluator` can set the result name and version; the source still names the function."""
    evaluator = FunctionEvaluator(exact_match, evaluation_name='matches', version='v2')
    assert repr(evaluator) == 'FunctionEvaluator(matches)'
    assert evaluator.as_spec() == EvaluatorSpec(name='exact_match', arguments={'evaluation_name': 'matches'})

    results, failures = await run_evaluators([evaluator, output_length], _make_context('HELLO'))

    assert failures == []
    assert results == snapshot(
        [
            EvaluationResult(
                name='matches',
                value=False,
                reason=None,
                source=EvaluatorSpec(name='exact_match', arguments={'evaluation_name': 'matches'}),
                evaluator_version='v2',
            ),
            EvaluationResult(
                name='output_length',
                value=5,
                reason='number of characters',
                source=EvaluatorSpec(name='output_length', arguments=None),
            ),
        ]
    )


def test_class_evaluators_are_not_wrapped():
    """`Evaluator` instances are stored as given, so existing class-based evaluators behave exactly as before."""

    class IsHello(Evaluator[str, str, None]):
        def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> bool:
            return ctx.output == 'hello'  # pragma: no cover

    evaluator = IsHello()
    function_evaluator = FunctionEvaluator(exact_match)
    dataset = Dataset(
        name='mixed',
        cases=[Case(inputs='x', evaluators=[evaluator, function_evaluator])],
        evaluators=[evaluator, exact_match],
    )

    assert dataset.cases[0].evaluators[0] is evaluator
    assert dataset.cases[0].evaluators[1] is function_evaluator
    assert dataset.evaluators[0] is evaluator
    assert dataset.evaluators[1] == FunctionEvaluator(exact_match)


@pytest.mark.parametrize('fmt', ['yaml', 'json'])
async def test_serialization_round_trip(tmp_path: Path, fmt: Literal['yaml', 'json']):
    """Function evaluators are saved under the function's name, and loaded again by passing the functions back."""
    dataset = Dataset[str, str, Any](
        name='round_trip',
        cases=[
            Case(
                name='hello',
                inputs='hello',
                expected_output='HELLO',
                evaluators=[FunctionEvaluator(exact_match, version='2'), Equals('HELLO')],
            )
        ],
        evaluators=[checks, FunctionEvaluator(output_length, evaluation_name='length', version='1')],
    )
    path = tmp_path / f'dataset.{fmt}'
    dataset.to_file(path, custom_evaluator_types=[exact_match, output_length, checks])
    content = path.read_text()

    # YAML is a superset of JSON, so this reads both formats.
    data = yaml.safe_load(content)
    assert (data['evaluators'], data['cases'][0]['evaluators']) == snapshot(
        (
            ['checks', {'output_length': {'evaluation_name': 'length', 'version': '1'}}],
            [{'exact_match': {'version': '2'}}, {'Equals': 'HELLO'}],
        )
    )

    loaded = Dataset[str, str, Any].from_file(path, custom_evaluator_types=[exact_match, output_length, checks])
    assert loaded.evaluators == dataset.evaluators
    assert loaded.cases[0].evaluators == dataset.cases[0].evaluators

    loaded.to_file(path, custom_evaluator_types=[exact_match, output_length, checks])
    assert path.read_text() == content

    report = await loaded.evaluate(upper, progress=False)
    case = report.cases[0]
    assert {
        result.name: (result.value, result.evaluator_version)
        for result in [*case.assertions.values(), *case.scores.values(), *case.labels.values()]
    } == snapshot(
        {
            'exact_match': (True, '2'),
            'Equals': (True, None),
            'is_upper': (True, None),
            'length': (5, '1'),
            'label': ('shout', None),
        }
    )


def test_json_schema():
    """The JSON schema lists each function by name, with its optional `evaluation_name` and `version`."""
    schema = Dataset[str, str, Any].model_json_schema_with_evaluators([exact_match])

    assert schema['$defs']['Case']['properties']['evaluators']['items']['anyOf'][:2] == snapshot(
        [{'const': 'exact_match', 'type': 'string'}, {'$ref': '#/$defs/spec_exact_match'}]
    )
    assert (schema['$defs']['spec_exact_match'], schema['$defs']['spec_params_exact_match']) == snapshot(
        (
            {
                'additionalProperties': False,
                'properties': {'exact_match': {'$ref': '#/$defs/spec_params_exact_match'}},
                'required': ['exact_match'],
                'title': 'spec_exact_match',
                'type': 'object',
            },
            {
                'additionalProperties': False,
                'properties': {
                    'evaluation_name': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Evaluation Name'},
                    'version': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'title': 'Version'},
                },
                'title': 'spec_params_exact_match',
                'type': 'object',
            },
        )
    )


def test_from_dict():
    """Functions are loaded from a dictionary, as done for hosted datasets, alongside evaluator classes."""

    @dataclass
    class IsHello(Evaluator[str, str, Any]):
        def evaluate(self, ctx: EvaluatorContext[str, str, Any]) -> bool:
            return ctx.output == 'hello'  # pragma: no cover

    dataset = Dataset[str, str, Any].from_dict(
        {
            'name': 'hosted',
            'cases': [{'inputs': 'x', 'evaluators': [{'exact_match': {'evaluation_name': 'matches'}}, 'IsHello']}],
            'evaluators': ['output_length'],
        },
        custom_evaluator_types=[IsHello, exact_match, output_length],
    )

    assert dataset.cases[0].evaluators == [FunctionEvaluator(exact_match, evaluation_name='matches'), IsHello()]
    assert dataset.evaluators == [FunctionEvaluator(output_length)]


@pytest.mark.parametrize(
    'function',
    [
        pytest.param(lambda ctx: True, id='lambda'),  # pyright: ignore[reportUnknownLambdaType, reportUnknownArgumentType]
        pytest.param(partial(has_prefix, prefix='H'), id='partial'),
        pytest.param(AsyncCallable(), id='callable-object'),
        pytest.param(PrefixCheck('H').check, id='bound-method'),
        pytest.param(['HELLO'].count, id='built-in-bound-method'),
    ],
)
def test_unnamed_function_not_serializable(tmp_path: Path, function: EvaluatorFunc[str, str, Any]):
    """Only a function with a name can be saved and loaded, and a failed save leaves no files behind."""
    dataset = Dataset[str, str, Any](name='unsaveable', cases=[Case(inputs='x', evaluators=[function])])

    for path in (tmp_path / 'dataset.yaml', tmp_path / 'dataset.json'):
        with pytest.raises(
            PydanticSerializationError, match=r"Function evaluator '[^']+' cannot be serialized, as it is identified"
        ):
            dataset.to_file(path)
    assert list(tmp_path.iterdir()) == []

    with pytest.raises(TypeError, match='cannot be used in `custom_evaluator_types`, as an evaluator function is'):
        Dataset[str, str, Any].from_dict({'name': 'ds', 'cases': []}, custom_evaluator_types=[function])


def test_two_functions_with_one_name_not_serializable(tmp_path: Path):
    """Two different functions saved under one name could not be told apart when loading, so saving fails."""

    def make_exact_match() -> EvaluatorFunc[str, str, Any]:
        def exact_match(ctx: EvaluatorContext[str, str, Any]) -> bool:
            return True  # pragma: no cover

        return exact_match

    dataset = Dataset[str, str, Any](
        name='clash', cases=[Case(inputs='x', evaluators=[make_exact_match()])], evaluators=[exact_match]
    )
    with pytest.raises(PydanticSerializationError, match="Two different evaluator functions are named 'exact_match'"):
        dataset.to_file(tmp_path / 'dataset.yaml')
    assert list(tmp_path.iterdir()) == []

    # The same function used in several places is saved under its name each time.
    Dataset[str, str, Any](
        name='reused', cases=[Case(inputs='x', evaluators=[exact_match])], evaluators=[exact_match]
    ).to_file(tmp_path / 'dataset.yaml')


def test_duplicate_names():
    """Functions must have names that are unique among the custom evaluators, but may replace a default evaluator."""

    def make_exact_match() -> EvaluatorFunc[str, str, Any]:
        def exact_match(ctx: EvaluatorContext[str, str, Any]) -> bool:
            return True  # pragma: no cover

        return exact_match

    @dataclass
    class ExactMatch(Evaluator[str, str, Any]):
        @classmethod
        def get_serialization_name(cls) -> str:
            return 'exact_match'

        def evaluate(self, ctx: EvaluatorContext[str, str, Any]) -> bool:
            return True  # pragma: no cover

    for custom_evaluator_types in ([exact_match, make_exact_match()], [ExactMatch, exact_match]):
        with pytest.raises(ValueError) as exc_info:
            Dataset[str, str, Any].from_dict({'name': 'ds', 'cases': []}, custom_evaluator_types=custom_evaluator_types)
        assert str(exc_info.value) == snapshot(
            "Duplicate evaluator name: 'exact_match'. An evaluator function is identified by its name, so it must be "
            'different from the names of the other `custom_evaluator_types`. Rename the function.'
        )

    # Passing the same function twice is the same mistake as passing a class twice, not a name clash.
    with pytest.raises(ValueError, match="Duplicate evaluator class name: 'exact_match'"):
        Dataset[str, str, Any].from_dict({'name': 'ds', 'cases': []}, custom_evaluator_types=[exact_match, exact_match])

    def Equals(ctx: EvaluatorContext[str, str, Any]) -> bool:
        return True  # pragma: no cover

    dataset = Dataset[str, str, Any].from_dict(
        {'name': 'ds', 'cases': [], 'evaluators': ['Equals']}, custom_evaluator_types=[Equals]
    )
    assert dataset.evaluators == [FunctionEvaluator(Equals)]


def test_load_errors():
    """A function that wasn't passed, or arguments it doesn't take, fail like they do for evaluator classes."""
    with pytest.raises(ExceptionGroup) as exc_info:
        Dataset[str, str, Any].from_dict(
            {
                'name': 'ds',
                'cases': [{'inputs': 'x', 'evaluators': ['output_length']}],
                'evaluators': [{'exact_match': 'matches'}, {'exact_match': {'name': 'matches'}}],
            },
            custom_evaluator_types=[exact_match],
        )
    assert [str(error) for error in exc_info.value.exceptions] == snapshot(
        [
            "Failed to instantiate evaluator 'exact_match' for dataset: an evaluator function only takes the `evaluation_name` and `version` arguments by name",
            "Failed to instantiate evaluator 'exact_match' for dataset: exact_match() got an unexpected keyword argument 'name'",
            "Evaluator 'output_length' is not in the provided `custom_evaluator_types`. Valid choices: ['exact_match', 'Equals', 'EqualsExpected', 'Contains', 'IsInstance', 'MaxDuration', 'LLMJudge', 'HasMatchingSpan', 'ToolCorrectness', 'TrajectoryMatch', 'ArgumentCorrectness', 'MaxToolCalls', 'MaxModelRequests', 'GEval']. If you are trying to use a custom evaluator, you must include it in the `custom_evaluator_types` argument.",
        ]
    )


def test_custom_evaluator_types_typing():
    """`custom_evaluator_types` takes evaluator classes and functions together, type-checked against the dataset."""

    def any_context(ctx: EvaluatorContext) -> bool:
        return True  # pragma: no cover

    def int_inputs(ctx: EvaluatorContext[int, str, Any]) -> bool:
        return True  # pragma: no cover

    def list_output(ctx: EvaluatorContext[str, str, Any]) -> list[str]:
        return []  # pragma: no cover

    data: dict[str, Any] = {'name': 'ds', 'cases': []}
    Dataset[str, str, Any].from_dict(data, custom_evaluator_types=[Equals, exact_match, output_length])
    Dataset[str, str, Any].from_dict(data, custom_evaluator_types=(any_context, Equals))
    custom_evaluator_types: list[type[Evaluator[str, str, Any]] | EvaluatorFunc[str, str, Any]] = [Equals, exact_match]
    Dataset[str, str, Any].model_json_schema_with_evaluators(custom_evaluator_types)

    Dataset[str, str, Any].from_dict(data, custom_evaluator_types=[int_inputs])  # pyright: ignore[reportArgumentType]
    Dataset[str, str, Any].from_dict(data, custom_evaluator_types=[list_output])  # pyright: ignore[reportArgumentType]
    with pytest.raises(TypeError, match=r"Equals\(value='x'\) cannot be used in `custom_evaluator_types`"):
        Dataset[str, str, Any].from_dict(data, custom_evaluator_types=[Equals('x')])  # pyright: ignore[reportArgumentType]


@needs_logfire
async def test_evaluator_span_name(capfire: CaptureLogfire):
    """The evaluator span is labelled with the function's name."""
    dataset = Dataset(name='spans', cases=[Case(inputs='hello')], evaluators=[exact_match])

    await dataset.evaluate(upper, progress=False)

    spans = capfire.exporter.exported_spans_as_dict()
    assert [
        span['attributes']['logfire.msg'] for span in spans if span['name'] == 'evaluator: {evaluator_name}'
    ] == snapshot(['Calling evaluator: exact_match'])


async def test_online_evaluate_decorator():
    """Functions passed to the online `evaluate` decorator, bare or in an `OnlineEvaluator`, run on each call."""
    results: list[EvaluationResult[Any]] = []

    def sink(
        new_results: Sequence[EvaluationResult[Any]],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext[Any, Any, Any],
    ) -> None:
        results.extend(new_results)

    config = OnlineEvalConfig(default_sink=sink)

    @config.evaluate(output_length, OnlineEvaluator(evaluator=FunctionEvaluator(exact_match, version='v1')))
    async def shout(text: str) -> str:
        return text.upper()

    assert await shout('hi') == 'HI'
    await wait_for_evaluations()

    assert sorted(results, key=lambda result: result.name) == snapshot(
        [
            EvaluationResult(
                name='exact_match',
                value=False,
                reason=None,
                source=EvaluatorSpec(name='exact_match', arguments=None),
                evaluator_version='v1',
            ),
            EvaluationResult(
                name='output_length',
                value=2,
                reason='number of characters',
                source=EvaluatorSpec(name='output_length', arguments=None),
            ),
        ]
    )


async def test_online_evaluation_capability():
    """Functions passed to the `OnlineEvaluation` capability run after each agent run."""
    results: list[EvaluationResult[Any]] = []

    def sink(
        new_results: Sequence[EvaluationResult[Any]],
        failures: Sequence[EvaluatorFailure],
        context: EvaluatorContext[Any, Any, Any],
    ) -> None:
        results.extend(new_results)

    def non_empty(ctx: EvaluatorContext[str, str, None]) -> bool:
        return bool(ctx.output)

    agent = Agent(
        TestModel(),
        capabilities=[OnlineEvaluation(evaluators=[non_empty], config=OnlineEvalConfig(default_sink=sink))],
    )

    await agent.run('hello')
    await wait_for_evaluations()

    assert results == snapshot(
        [
            EvaluationResult(
                name='non_empty', value=True, reason=None, source=EvaluatorSpec(name='non_empty', arguments=None)
            )
        ]
    )


def test_evaluator_function_typing():
    """Function evaluators are type-checked against the dataset's type parameters."""

    def int_inputs(ctx: EvaluatorContext[int, str, None]) -> bool:
        return True  # pragma: no cover

    typed: EvaluatorFunc[str, str, None] = exact_match
    dataset = Dataset[str, str, None](name='typed', cases=[], evaluators=[typed])
    dataset.add_evaluator(int_inputs)  # pyright: ignore[reportArgumentType]
    Case[str, str, None](inputs='x', evaluators=[int_inputs])  # pyright: ignore[reportArgumentType]


def test_non_function_evaluators_rejected():
    """An evaluator class or a non-callable is rejected up front, instead of failing on every case at run time."""

    class IsHello(Evaluator[str, str, None]):
        def evaluate(self, ctx: EvaluatorContext[str, str, None]) -> bool:
            return ctx.output == 'hello'  # pragma: no cover

    with pytest.raises(TypeError, match="Expected an evaluator instance or function, got the class 'IsHello'"):
        Dataset[str, str, None](name='classes', cases=[], evaluators=[IsHello])  # pyright: ignore[reportArgumentType]

    with pytest.raises(TypeError, match="Expected an evaluator instance or function, got 'nope'"):
        Case[str, str, None](inputs='x', evaluators=['nope'])  # pyright: ignore[reportArgumentType]


async def test_non_coroutine_awaitable():
    """A function returning an awaitable that isn't a coroutine has its result awaited."""
    results, failures = await run_evaluators([deferred], _make_context('HELLO'))

    assert failures == []
    assert results == snapshot(
        [
            EvaluationResult(
                name='deferred', value=True, reason=None, source=EvaluatorSpec(name='deferred', arguments=None)
            )
        ]
    )


def test_evaluate_sync():
    """`evaluate_sync` returns a sync function's result and runs an awaitable one to completion."""
    context = _make_context('HELLO')
    assert FunctionEvaluator(exact_match).evaluate_sync(context) is False
    assert FunctionEvaluator(deferred).evaluate_sync(context) is True


async def test_unnamed_callable():
    """A callable without a `__name__` is named after its type instead of aborting the evaluation."""
    results, failures = await run_evaluators([attrgetter('output')], _make_context('HELLO'))

    assert failures == []
    assert results == snapshot(
        [
            EvaluationResult(
                name='attrgetter', value='HELLO', reason=None, source=EvaluatorSpec(name='attrgetter', arguments=None)
            )
        ]
    )


def test_online_evaluator_wraps_function():
    """A function given to `OnlineEvaluator` without a type checker is wrapped instead of failing at dispatch."""
    online_evaluator = OnlineEvaluator(evaluator=exact_match)  # pyright: ignore[reportArgumentType]
    assert online_evaluator.evaluator == FunctionEvaluator(exact_match)

"""Tests for report-level evaluators and experiment-wide analyses."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

from ..conftest import try_import

with try_import() as imports_successful:
    from logfire.testing import CaptureLogfire

    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import (
        ConfusionMatrixEvaluator,
        Evaluator,
        EvaluatorContext,
        KolmogorovSmirnovEvaluator,
        PrecisionRecallEvaluator,
        ReportEvaluator,
        ReportEvaluatorContext,
        ROCAUCEvaluator,
    )
    from pydantic_evals.evaluators.evaluator import EvaluatorOutput
    from pydantic_evals.reporting import EvaluationReport, ReportCase
    from pydantic_evals.reporting.analyses import (
        ConfusionMatrix,
        LinePlot,
        PrecisionRecall,
        ReportAnalysis,
        ScalarResult,
        TableResult,
    )

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
    pytest.mark.anyio,
]


# --- Test models ---


class TaskInput(BaseModel):
    text: str


class TaskOutput(BaseModel):
    label: str
    score: float = 0.5


# --- ReportAnalysis serialization tests ---


def test_confusion_matrix_serialization():
    cm = ConfusionMatrix(
        class_labels=['cat', 'dog'],
        matrix=[[5, 2], [1, 8]],
    )
    data = cm.model_dump()
    assert data['type'] == 'confusion_matrix'
    assert data['class_labels'] == ['cat', 'dog']
    assert data['matrix'] == [[5, 2], [1, 8]]

    # Round-trip through discriminated union
    adapter: TypeAdapter[ReportAnalysis] = TypeAdapter(ReportAnalysis)
    restored = adapter.validate_python(data)
    assert isinstance(restored, ConfusionMatrix)
    assert restored.class_labels == ['cat', 'dog']


def test_precision_recall_serialization():
    pr = PrecisionRecall(
        curves=[],
    )
    data = pr.model_dump()
    assert data['type'] == 'precision_recall'

    adapter: TypeAdapter[ReportAnalysis] = TypeAdapter(ReportAnalysis)
    restored = adapter.validate_python(data)
    assert isinstance(restored, PrecisionRecall)


def test_scalar_result_serialization():
    sr = ScalarResult(title='Accuracy', value=0.95, unit='%')
    data = sr.model_dump()
    assert data['type'] == 'scalar'
    assert data['value'] == 0.95

    adapter: TypeAdapter[ReportAnalysis] = TypeAdapter(ReportAnalysis)
    restored = adapter.validate_python(data)
    assert isinstance(restored, ScalarResult)
    assert restored.title == 'Accuracy'


def test_table_result_serialization():
    tr = TableResult(
        title='Per-class F1',
        columns=['Class', 'F1'],
        rows=[['cat', 0.9], ['dog', 0.85]],
    )
    data = tr.model_dump()
    assert data['type'] == 'table'

    adapter: TypeAdapter[ReportAnalysis] = TypeAdapter(ReportAnalysis)
    restored = adapter.validate_python(data)
    assert isinstance(restored, TableResult)
    assert len(restored.rows) == 2


# --- ConfusionMatrixEvaluator tests ---


def _make_report_case(
    name: str,
    output: Any = None,
    expected_output: Any = None,
    labels: dict[str, Any] | None = None,
    scores: dict[str, Any] | None = None,
    assertions: dict[str, Any] | None = None,
    metrics: dict[str, float | int] | None = None,
    metadata: Any = None,
) -> ReportCase[Any, Any, Any]:
    from pydantic_evals.evaluators.evaluator import EvaluationResult
    from pydantic_evals.evaluators.spec import EvaluatorSpec

    _source = EvaluatorSpec(name='test', arguments=None)

    def _make_eval_result(key: str, val: Any) -> Any:
        return EvaluationResult(name=key, value=val, reason=None, source=_source)

    return ReportCase(
        name=name,
        inputs={},
        metadata=metadata,
        expected_output=expected_output,
        output=output,
        metrics=metrics or {},
        attributes={},
        scores={k: _make_eval_result(k, v) for k, v in (scores or {}).items()},
        labels={k: _make_eval_result(k, v) for k, v in (labels or {}).items()},
        assertions={k: _make_eval_result(k, v) for k, v in (assertions or {}).items()},
        task_duration=0.1,
        total_duration=0.2,
    )


def _make_report(cases: list[ReportCase]) -> EvaluationReport:
    return EvaluationReport(name='test', cases=cases)


def test_confusion_matrix_evaluator_from_expected_output_and_output():
    cases = [
        _make_report_case('c1', output='cat', expected_output='cat'),
        _make_report_case('c2', output='dog', expected_output='cat'),
        _make_report_case('c3', output='dog', expected_output='dog'),
        _make_report_case('c4', output='cat', expected_output='dog'),
    ]
    report = _make_report(cases)

    evaluator = ConfusionMatrixEvaluator(
        predicted_from='output',
        expected_from='expected_output',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    result = evaluator.evaluate(ctx)

    assert isinstance(result, ConfusionMatrix)
    assert result.class_labels == ['cat', 'dog']
    # matrix[expected_idx][predicted_idx]
    # cat->cat=1, cat->dog=1, dog->cat=1, dog->dog=1
    assert result.matrix == [[1, 1], [1, 1]]


def test_confusion_matrix_evaluator_from_labels():
    cases = [
        _make_report_case('c1', expected_output='positive', labels={'predicted': 'positive'}),
        _make_report_case('c2', expected_output='negative', labels={'predicted': 'positive'}),
        _make_report_case('c3', expected_output='negative', labels={'predicted': 'negative'}),
    ]
    report = _make_report(cases)

    evaluator = ConfusionMatrixEvaluator(
        predicted_from='labels',
        predicted_key='predicted',
        expected_from='expected_output',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    result = evaluator.evaluate(ctx)

    assert isinstance(result, ConfusionMatrix)
    assert result.class_labels == ['negative', 'positive']
    # expected=negative, predicted=negative: 1
    # expected=negative, predicted=positive: 1
    # expected=positive, predicted=positive: 1
    assert result.matrix == [[1, 1], [0, 1]]


def test_confusion_matrix_evaluator_from_metadata():
    cases = [
        _make_report_case('c1', expected_output='A', metadata={'pred': 'A'}),
        _make_report_case('c2', expected_output='B', metadata={'pred': 'A'}),
    ]
    report = _make_report(cases)

    evaluator = ConfusionMatrixEvaluator(
        predicted_from='metadata',
        predicted_key='pred',
        expected_from='expected_output',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    result = evaluator.evaluate(ctx)

    assert isinstance(result, ConfusionMatrix)
    assert result.class_labels == ['A', 'B']
    assert result.matrix == [[1, 0], [1, 0]]


def test_confusion_matrix_evaluator_skips_none():
    cases = [
        _make_report_case('c1', output='cat', expected_output='cat'),
        _make_report_case('c2', output='dog', expected_output=None),  # should be skipped
    ]
    report = _make_report(cases)

    evaluator = ConfusionMatrixEvaluator(predicted_from='output', expected_from='expected_output')
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    result = evaluator.evaluate(ctx)

    assert isinstance(result, ConfusionMatrix)
    assert result.class_labels == ['cat']
    assert result.matrix == [[1]]


def test_confusion_matrix_labels_requires_key():
    evaluator = ConfusionMatrixEvaluator(predicted_from='labels', predicted_key=None)
    cases = [_make_report_case('c1', expected_output='a', labels={})]
    report = _make_report(cases)
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)

    with pytest.raises(ValueError, match="'key' is required"):
        evaluator.evaluate(ctx)


# --- PrecisionRecallEvaluator tests ---


def test_precision_recall_evaluator_basic():
    cases = [
        _make_report_case('c1', scores={'confidence': 0.9}, assertions={'is_correct': True}),
        _make_report_case('c2', scores={'confidence': 0.8}, assertions={'is_correct': True}),
        _make_report_case('c3', scores={'confidence': 0.3}, assertions={'is_correct': False}),
        _make_report_case('c4', scores={'confidence': 0.1}, assertions={'is_correct': False}),
    ]
    report = _make_report(cases)

    evaluator = PrecisionRecallEvaluator(
        score_from='scores',
        score_key='confidence',
        positive_from='assertions',
        positive_key='is_correct',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert isinstance(results, list)
    assert len(results) == 2
    pr_result = results[0]
    assert isinstance(pr_result, PrecisionRecall)
    assert len(pr_result.curves) == 1
    curve = pr_result.curves[0]
    assert curve.name == 'test'
    assert len(curve.points) > 0
    assert curve.auc is not None
    # Perfect separation: AUC should be 1.0
    assert curve.auc == 1.0

    # ScalarResult with AUC is also returned for queryability
    scalar = results[1]
    assert isinstance(scalar, ScalarResult)
    assert scalar.title == 'Precision-Recall Curve AUC'
    assert scalar.value == 1.0


def test_precision_recall_evaluator_from_metrics():
    cases = [
        _make_report_case('c1', metrics={'score': 0.9}, assertions={'positive': True}),
        _make_report_case('c2', metrics={'score': 0.1}, assertions={'positive': False}),
    ]
    report = _make_report(cases)

    evaluator = PrecisionRecallEvaluator(
        score_from='metrics',
        score_key='score',
        positive_from='assertions',
        positive_key='positive',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert isinstance(results, list)
    pr_result = results[0]
    assert isinstance(pr_result, PrecisionRecall)
    assert len(pr_result.curves) == 1


def test_precision_recall_evaluator_downsamples():
    """PrecisionRecallEvaluator downsamples when there are more unique thresholds than n_thresholds."""
    cases = [_make_report_case(f'c{i}', scores={'s': i * 0.1}, assertions={'p': i >= 5}) for i in range(10)]
    report = _make_report(cases)

    evaluator = PrecisionRecallEvaluator(
        score_from='scores',
        score_key='s',
        positive_from='assertions',
        positive_key='p',
        n_thresholds=3,
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    pr_result = results[0]
    assert isinstance(pr_result, PrecisionRecall)
    curve = pr_result.curves[0]
    # With n_thresholds=3, the anchor + 10 unique thresholds = 11 points should be downsampled to 3
    assert len(curve.points) == 3


def test_precision_recall_evaluator_empty():
    report = _make_report([])
    evaluator = PrecisionRecallEvaluator(
        score_from='scores',
        score_key='s',
        positive_from='assertions',
        positive_key='p',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert isinstance(results, list)
    assert len(results) == 2
    pr_result = results[0]
    assert isinstance(pr_result, PrecisionRecall)
    assert len(pr_result.curves) == 0
    scalar_result = results[1]
    assert isinstance(scalar_result, ScalarResult)
    assert math.isnan(scalar_result.value)


def test_precision_recall_assertions_requires_key():
    evaluator = PrecisionRecallEvaluator(
        score_from='scores',
        score_key='s',
        positive_from='assertions',
        positive_key=None,
    )
    cases = [_make_report_case('c1', scores={'s': 0.5})]
    report = _make_report(cases)
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)

    with pytest.raises(ValueError, match="'positive_key' is required"):
        evaluator.evaluate(ctx)


def test_precision_recall_labels_requires_key():
    evaluator = PrecisionRecallEvaluator(
        score_from='scores',
        score_key='s',
        positive_from='labels',
        positive_key=None,
    )
    cases = [_make_report_case('c1', scores={'s': 0.5})]
    report = _make_report(cases)
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)

    with pytest.raises(ValueError, match="'positive_key' is required"):
        evaluator.evaluate(ctx)


# --- Custom ReportEvaluator test ---


def test_custom_report_evaluator():
    @dataclass
    class AccuracyEvaluator(ReportEvaluator):
        def evaluate(self, ctx: ReportEvaluatorContext) -> ScalarResult:
            if not ctx.report.cases:  # pragma: no cover
                return ScalarResult(title='Accuracy', value=0.0, unit='%')
            correct = sum(1 for case in ctx.report.cases if case.output == case.expected_output)
            accuracy = correct / len(ctx.report.cases) * 100
            return ScalarResult(title='Accuracy', value=accuracy, unit='%')

    cases = [
        _make_report_case('c1', output='cat', expected_output='cat'),
        _make_report_case('c2', output='dog', expected_output='cat'),
        _make_report_case('c3', output='dog', expected_output='dog'),
    ]
    report = _make_report(cases)
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)

    evaluator = AccuracyEvaluator()
    result = evaluator.evaluate(ctx)

    assert isinstance(result, ScalarResult)
    assert result.title == 'Accuracy'
    assert abs(result.value - 66.66666666666667) < 0.01


# --- Integration test: Dataset with report_evaluators ---


async def test_dataset_with_report_evaluators():
    """Integration test: Dataset with report_evaluators runs them after cases."""

    @dataclass
    class LabelEvaluator(Evaluator[TaskInput, str, None]):
        def evaluate(self, ctx: EvaluatorContext[TaskInput, str, None]) -> EvaluatorOutput:
            if ctx.expected_output is not None:
                return ctx.output == ctx.expected_output
            return True  # pragma: no cover

    dataset = Dataset[TaskInput, str, None](
        cases=[
            Case(name='c1', inputs=TaskInput(text='meow'), expected_output='cat'),
            Case(name='c2', inputs=TaskInput(text='woof'), expected_output='dog'),
            Case(name='c3', inputs=TaskInput(text='purr'), expected_output='cat'),
        ],
        evaluators=[LabelEvaluator()],
        report_evaluators=[
            ConfusionMatrixEvaluator(
                predicted_from='output',
                expected_from='expected_output',
                title='Label Confusion',
            ),
        ],
    )

    async def task(inputs: TaskInput) -> str:
        if 'meow' in inputs.text or 'purr' in inputs.text:
            return 'cat'
        return 'dog'

    report = await dataset.evaluate(task, progress=False)

    assert len(report.cases) == 3
    assert len(report.analyses) == 1

    analysis = report.analyses[0]
    assert isinstance(analysis, ConfusionMatrix)
    assert analysis.title == 'Label Confusion'
    assert 'cat' in analysis.class_labels
    assert 'dog' in analysis.class_labels


async def test_dataset_report_evaluator_returns_list():
    @dataclass
    class MultiAnalysisEvaluator(ReportEvaluator):
        def evaluate(self, ctx: ReportEvaluatorContext) -> list[ReportAnalysis]:
            n = len(ctx.report.cases)
            return [
                ScalarResult(title='Total Cases', value=n),
                ScalarResult(title='Case Count Squared', value=n * n),
            ]

    dataset = Dataset[TaskInput, str, None](
        cases=[
            Case(name='c1', inputs=TaskInput(text='a')),
            Case(name='c2', inputs=TaskInput(text='b')),
        ],
        report_evaluators=[MultiAnalysisEvaluator()],
    )

    async def task(inputs: TaskInput) -> str:
        return 'x'

    report = await dataset.evaluate(task, progress=False)

    assert len(report.analyses) == 2
    first = report.analyses[0]
    second = report.analyses[1]
    assert isinstance(first, ScalarResult)
    assert first.title == 'Total Cases'
    assert first.value == 2
    assert isinstance(second, ScalarResult)
    assert second.title == 'Case Count Squared'
    assert second.value == 4


# --- Rendering test ---


def test_report_rendering_includes_analyses():
    cases = [
        _make_report_case('c1', output='cat', expected_output='cat'),
    ]
    report = _make_report(cases)
    report.analyses = [
        ScalarResult(title='Accuracy', value=100.0, unit='%'),
        ConfusionMatrix(
            title='CM',
            class_labels=['cat'],
            matrix=[[1]],
        ),
    ]

    rendered = report.render(width=120)
    assert 'Accuracy: 100.0 %' in rendered
    assert 'CM' in rendered


def test_report_rendering_include_analyses_false():
    cases = [
        _make_report_case('c1', output='cat', expected_output='cat'),
    ]
    report = _make_report(cases)
    report.analyses = [
        ScalarResult(title='Accuracy', value=100.0, unit='%'),
    ]

    rendered = report.render(width=120, include_analyses=False)
    assert 'Accuracy: 100.0 %' not in rendered


def test_report_rendering_include_evaluator_failures_false():
    from pydantic_evals.evaluators.evaluator import EvaluatorFailure
    from pydantic_evals.evaluators.spec import EvaluatorSpec

    report = _make_report([_make_report_case('c1', output='x', expected_output='x')])
    report.report_evaluator_failures = [
        EvaluatorFailure(
            name='BrokenEvaluator',
            error_message='ValueError: oops',
            error_stacktrace='Traceback ...',
            source=EvaluatorSpec(name='BrokenEvaluator', arguments=None),
        ),
    ]

    rendered = report.render(width=120, include_evaluator_failures=False)
    assert 'Report Evaluator Failures' not in rendered
    assert 'BrokenEvaluator' not in rendered


# --- EvaluationReport.analyses default ---


def test_evaluation_report_analyses_default():
    report = EvaluationReport(name='test', cases=[])
    assert report.analyses == []


# --- ReportEvaluator serialization tests ---


def test_report_evaluator_get_serialization_name():
    """get_serialization_name works as classmethod and on instance."""
    assert ConfusionMatrixEvaluator.get_serialization_name() == 'ConfusionMatrixEvaluator'
    assert PrecisionRecallEvaluator.get_serialization_name() == 'PrecisionRecallEvaluator'
    # Also works on instance
    assert ConfusionMatrixEvaluator().get_serialization_name() == 'ConfusionMatrixEvaluator'


def test_report_evaluator_as_spec_no_args():
    """Report evaluator with all defaults produces spec with no arguments."""
    from pydantic_evals.evaluators.spec import EvaluatorSpec

    evaluator = ConfusionMatrixEvaluator()
    spec = evaluator.as_spec()
    assert isinstance(spec, EvaluatorSpec)
    assert spec.name == 'ConfusionMatrixEvaluator'
    assert spec.arguments is None


def test_report_evaluator_as_spec_with_args():
    """Report evaluator with non-default args produces spec with arguments."""
    evaluator = ConfusionMatrixEvaluator(predicted_from='labels', predicted_key='pred', title='Custom CM')
    spec = evaluator.as_spec()
    assert spec.name == 'ConfusionMatrixEvaluator'
    assert isinstance(spec.arguments, dict)
    assert spec.arguments['predicted_from'] == 'labels'
    assert spec.arguments['predicted_key'] == 'pred'
    assert spec.arguments['title'] == 'Custom CM'


def test_report_evaluator_as_spec_single_arg_non_first_field():
    """Report evaluator with one non-default arg that isn't the first field uses dict form."""
    evaluator = ConfusionMatrixEvaluator(title='My Matrix')
    spec = evaluator.as_spec()
    assert spec.name == 'ConfusionMatrixEvaluator'
    # title is not the first field, so dict form is used to preserve the field name
    assert isinstance(spec.arguments, dict)
    assert spec.arguments == {'title': 'My Matrix'}


def test_report_evaluator_as_spec_single_arg_first_field():
    """Report evaluator with one non-default arg that is the first field uses tuple form."""
    evaluator = ConfusionMatrixEvaluator(predicted_from='labels')
    spec = evaluator.as_spec()
    assert spec.name == 'ConfusionMatrixEvaluator'
    assert isinstance(spec.arguments, tuple)
    assert spec.arguments == ('labels',)


def test_report_evaluator_build_serialization_arguments_excludes_defaults():
    """ConfusionMatrixEvaluator with all defaults returns empty dict."""
    evaluator = ConfusionMatrixEvaluator()
    args = evaluator.build_serialization_arguments()
    assert args == {}


def test_report_evaluator_serializes_in_model_dump():
    """Dataset with report evaluators includes them in model_dump output."""
    dataset = Dataset[str, str, None](
        cases=[Case(inputs='hello', expected_output='world')],
        report_evaluators=[ConfusionMatrixEvaluator()],
    )
    dumped = dataset.model_dump(mode='json', context={'use_short_form': True})
    assert 'report_evaluators' in dumped
    assert dumped['report_evaluators'] == ['ConfusionMatrixEvaluator']


def test_report_evaluator_serializes_with_args_in_model_dump():
    """Dataset with report evaluators with args includes them in model_dump output."""
    dataset = Dataset[str, str, None](
        cases=[Case(inputs='hello', expected_output='world')],
        report_evaluators=[ConfusionMatrixEvaluator(title='Custom')],
    )
    dumped = dataset.model_dump(mode='json', context={'use_short_form': True})
    assert dumped['report_evaluators'] == [{'ConfusionMatrixEvaluator': {'title': 'Custom'}}]


def test_report_evaluator_repr():
    """Custom @dataclass(repr=False) report evaluator inherits no-defaults repr."""

    @dataclass(repr=False)
    class CustomEvaluator(ReportEvaluator):
        threshold: float = 0.5

        def evaluate(self, ctx: ReportEvaluatorContext) -> ReportAnalysis:  # pragma: no cover
            ...

    evaluator = CustomEvaluator()
    assert repr(evaluator).endswith('CustomEvaluator()')

    evaluator_with_args = CustomEvaluator(threshold=0.8)
    assert repr(evaluator_with_args).endswith('CustomEvaluator(threshold=0.8)')


# --- Additional coverage tests ---


def test_confusion_matrix_evaluator_metadata_non_dict():
    """ConfusionMatrixEvaluator with metadata_from but non-dict metadata returns str(metadata)."""
    cases = [
        _make_report_case('c1', expected_output='A', metadata='some_string'),
    ]
    report = _make_report(cases)

    evaluator = ConfusionMatrixEvaluator(
        predicted_from='metadata',
        predicted_key=None,
        expected_from='expected_output',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    result = evaluator.evaluate(ctx)

    assert isinstance(result, ConfusionMatrix)
    assert result.class_labels == ['A', 'some_string']
    assert result.matrix == [[0, 1], [0, 0]]


def test_confusion_matrix_evaluator_metadata_key_with_non_dict():
    """ConfusionMatrixEvaluator with metadata key but non-dict metadata skips the case."""
    cases = [
        _make_report_case('c1', expected_output='A', metadata='some_string'),
        _make_report_case('c2', expected_output='B', metadata={'pred': 'B'}),
    ]
    report = _make_report(cases)

    evaluator = ConfusionMatrixEvaluator(
        predicted_from='metadata',
        predicted_key='pred',
        expected_from='expected_output',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    result = evaluator.evaluate(ctx)

    assert isinstance(result, ConfusionMatrix)
    # c1 should be skipped (non-dict metadata with key), only c2 used
    assert result.class_labels == ['B']
    assert result.matrix == [[1]]


def test_precision_recall_evaluator_skips_missing_scores():
    """PrecisionRecallEvaluator skips cases missing score or positive data."""
    cases = [
        _make_report_case('c1', scores={'confidence': 0.9}, assertions={'is_correct': True}),
        _make_report_case('c2', scores={}, assertions={'is_correct': False}),  # missing score
        _make_report_case('c3', scores={'confidence': 0.3}, assertions={}),  # missing assertion
    ]
    report = _make_report(cases)

    evaluator = PrecisionRecallEvaluator(
        score_key='confidence',
        positive_from='assertions',
        positive_key='is_correct',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    pr_result = results[0]
    assert isinstance(pr_result, PrecisionRecall)
    assert len(pr_result.curves) == 1
    # Only case c1 should have been used (1 scored case)


def test_precision_recall_evaluator_positive_from_expected_output():
    """PrecisionRecallEvaluator with positive_from='expected_output'."""
    cases = [
        _make_report_case('c1', scores={'conf': 0.9}, expected_output='yes'),
        _make_report_case('c2', scores={'conf': 0.1}, expected_output=''),
        _make_report_case('c3', scores={'conf': 0.5}, expected_output=None),  # skipped
    ]
    report = _make_report(cases)

    evaluator = PrecisionRecallEvaluator(
        score_key='conf',
        positive_from='expected_output',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    pr_result = results[0]
    assert isinstance(pr_result, PrecisionRecall)
    assert len(pr_result.curves) == 1


def test_precision_recall_evaluator_positive_from_labels():
    """PrecisionRecallEvaluator with positive_from='labels'."""
    cases = [
        _make_report_case('c1', scores={'conf': 0.9}, labels={'is_pos': 'yes'}),
        _make_report_case('c2', scores={'conf': 0.1}, labels={'is_pos': ''}),
    ]
    report = _make_report(cases)

    evaluator = PrecisionRecallEvaluator(
        score_key='conf',
        positive_from='labels',
        positive_key='is_pos',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    pr_result = results[0]
    assert isinstance(pr_result, PrecisionRecall)
    assert len(pr_result.curves) == 1


async def test_async_report_evaluator():
    """Async report evaluator is awaited through evaluate_async."""

    @dataclass
    class AsyncEvaluator(ReportEvaluator):
        async def evaluate(self, ctx: ReportEvaluatorContext) -> ScalarResult:
            return ScalarResult(title='Async Result', value=42)

    evaluator = AsyncEvaluator()
    report = _make_report([_make_report_case('c1', output='x')])
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    result = await evaluator.evaluate_async(ctx)

    assert isinstance(result, ScalarResult)
    assert result.value == 42


def test_report_evaluator_build_serialization_arguments_with_default_factory():
    """build_serialization_arguments handles fields with default_factory."""
    from dataclasses import field as dc_field

    @dataclass
    class EvalWithFactory(ReportEvaluator):
        tags: list[str] = dc_field(default_factory=list[str])

        def evaluate(self, ctx: ReportEvaluatorContext) -> ReportAnalysis:  # pragma: no cover
            ...

    # Default value (empty list) — should be excluded
    evaluator = EvalWithFactory()
    args = evaluator.build_serialization_arguments()
    assert args == {}

    # Non-default value — should be included
    evaluator_with_tags = EvalWithFactory(tags=['a', 'b'])
    args = evaluator_with_tags.build_serialization_arguments()
    assert args == {'tags': ['a', 'b']}


def test_report_rendering_with_failures():
    """Report rendering includes report_evaluator_failures."""
    from pydantic_evals.evaluators.evaluator import EvaluatorFailure
    from pydantic_evals.evaluators.spec import EvaluatorSpec

    report = _make_report([_make_report_case('c1', output='x', expected_output='x')])
    report.report_evaluator_failures = [
        EvaluatorFailure(
            name='BrokenEvaluator',
            error_message='ValueError: something went wrong',
            error_stacktrace='Traceback ...',
            source=EvaluatorSpec(name='BrokenEvaluator', arguments=None),
        ),
    ]

    rendered = report.render(width=120)
    assert 'Report Evaluator Failures:' in rendered
    assert 'BrokenEvaluator' in rendered
    assert 'something went wrong' in rendered


def test_report_rendering_scalar_without_unit():
    """ScalarResult rendering without a unit."""
    report = _make_report([_make_report_case('c1', output='x', expected_output='x')])
    report.analyses = [
        ScalarResult(title='Count', value=10),
    ]

    rendered = report.render(width=120)
    assert 'Count: 10' in rendered


def test_report_rendering_precision_recall():
    """PrecisionRecall rendering."""
    from pydantic_evals.reporting.analyses import PrecisionRecallCurve, PrecisionRecallPoint

    report = _make_report([_make_report_case('c1', output='x', expected_output='x')])
    report.analyses = [
        PrecisionRecall(
            title='PR Curve',
            curves=[
                PrecisionRecallCurve(
                    name='test_curve',
                    points=[PrecisionRecallPoint(threshold=0.5, precision=0.8, recall=0.7)],
                    auc=0.75,
                ),
            ],
        ),
    ]

    rendered = report.render(width=120)
    assert 'PR Curve' in rendered
    assert 'test_curve' in rendered
    assert 'AUC=0.7500' in rendered


def test_report_rendering_table_result():
    """TableResult rendering."""
    report = _make_report([_make_report_case('c1', output='x', expected_output='x')])
    report.analyses = [
        TableResult(
            title='Summary Table',
            columns=['Name', 'Value'],
            rows=[['accuracy', 0.95], ['f1', 0.9]],
        ),
    ]

    rendered = report.render(width=120)
    assert 'Summary Table' in rendered
    assert 'accuracy' in rendered


async def test_report_evaluator_exception_during_evaluate():
    """Report evaluator that raises an exception records a failure."""

    @dataclass
    class BrokenEvaluator(ReportEvaluator):
        def evaluate(self, ctx: ReportEvaluatorContext) -> ReportAnalysis:
            raise RuntimeError('evaluator broke')

    dataset = Dataset[str, str, None](
        cases=[Case(inputs='hello', expected_output='world')],
        report_evaluators=[BrokenEvaluator()],
    )

    async def task(inputs: str) -> str:
        return inputs

    report = await dataset.evaluate(task, progress=False)
    assert len(report.report_evaluator_failures) == 1
    assert 'evaluator broke' in report.report_evaluator_failures[0].error_message


async def test_report_evaluator_failure_does_not_block_others():
    """When one report evaluator fails, subsequent evaluators still run."""

    @dataclass
    class BrokenEvaluator(ReportEvaluator):
        def evaluate(self, ctx: ReportEvaluatorContext) -> ReportAnalysis:
            raise RuntimeError('first evaluator broke')

    @dataclass
    class WorkingEvaluator(ReportEvaluator):
        def evaluate(self, ctx: ReportEvaluatorContext) -> ScalarResult:
            return ScalarResult(title='Count', value=len(ctx.report.cases))

    dataset = Dataset[str, str, None](
        cases=[Case(inputs='hello', expected_output='world')],
        report_evaluators=[BrokenEvaluator(), WorkingEvaluator()],
    )

    async def task(inputs: str) -> str:
        return inputs

    report = await dataset.evaluate(task, progress=False)
    # The broken evaluator's failure is captured
    assert len(report.report_evaluator_failures) == 1
    assert 'first evaluator broke' in report.report_evaluator_failures[0].error_message
    # The working evaluator still ran and produced its analysis
    assert len(report.analyses) == 1
    assert isinstance(report.analyses[0], ScalarResult)
    assert report.analyses[0].value == 1


async def test_report_evaluator_failures_set_on_span(capfire: CaptureLogfire):
    """Report evaluator failures are set as a span attribute on the experiment span."""

    @dataclass
    class BrokenEvaluator(ReportEvaluator):
        def evaluate(self, ctx: ReportEvaluatorContext) -> ReportAnalysis:
            raise RuntimeError('evaluator broke')

    dataset = Dataset[str, str, None](
        cases=[Case(inputs='hello', expected_output='world')],
        report_evaluators=[BrokenEvaluator()],
    )

    async def task(inputs: str) -> str:
        return inputs

    report = await dataset.evaluate(task, progress=False)
    assert len(report.report_evaluator_failures) == 1

    spans = capfire.exporter.exported_spans_as_dict(parse_json_attributes=True)
    experiment_spans = [s for s in spans if s['name'] == 'evaluate {name}']
    assert len(experiment_spans) == 1
    attrs = experiment_spans[0]['attributes']
    failures = attrs.get('logfire.experiment.report_evaluator_failures')
    assert failures is not None
    assert len(failures) == 1
    assert failures[0]['name'] == 'BrokenEvaluator'
    assert 'evaluator broke' in failures[0]['error_message']
    assert 'error_stacktrace' in failures[0]
    assert failures[0]['source']['name'] == 'BrokenEvaluator'


def test_builtin_report_evaluator_repr():
    """Built-in report evaluators use the no-defaults repr."""
    evaluator = ConfusionMatrixEvaluator()
    assert repr(evaluator) == 'ConfusionMatrixEvaluator()'

    evaluator_with_args = ConfusionMatrixEvaluator(predicted_from='labels', predicted_key='pred')
    assert repr(evaluator_with_args) == "ConfusionMatrixEvaluator(predicted_from='labels', predicted_key='pred')"


# --- LinePlot serialization tests ---


def test_line_plot_serialization():
    from pydantic_evals.reporting.analyses import LinePlotCurve, LinePlotPoint

    lp = LinePlot(
        title='ROC Curve',
        x_label='FPR',
        y_label='TPR',
        x_range=(0, 1),
        y_range=(0, 1),
        curves=[
            LinePlotCurve(
                name='test',
                points=[LinePlotPoint(x=0.0, y=0.0), LinePlotPoint(x=0.5, y=0.8), LinePlotPoint(x=1.0, y=1.0)],
            ),
            LinePlotCurve(name='Random', points=[LinePlotPoint(x=0, y=0), LinePlotPoint(x=1, y=1)], style='dashed'),
        ],
    )
    data = lp.model_dump()
    assert data['type'] == 'line_plot'
    assert data['x_label'] == 'FPR'
    assert data['y_label'] == 'TPR'
    assert data['x_range'] == (0, 1)
    assert len(data['curves']) == 2
    assert data['curves'][1]['style'] == 'dashed'

    # Round-trip through discriminated union
    adapter: TypeAdapter[ReportAnalysis] = TypeAdapter(ReportAnalysis)
    restored = adapter.validate_python(data)
    assert isinstance(restored, LinePlot)
    assert restored.x_label == 'FPR'
    assert len(restored.curves) == 2


# --- ROCAUCEvaluator tests ---


def test_roc_auc_evaluator_basic():
    """ROCAUCEvaluator computes ROC curve and AUC for perfect separation."""
    cases = [
        _make_report_case('c1', scores={'confidence': 0.9}, assertions={'is_correct': True}),
        _make_report_case('c2', scores={'confidence': 0.8}, assertions={'is_correct': True}),
        _make_report_case('c3', scores={'confidence': 0.3}, assertions={'is_correct': False}),
        _make_report_case('c4', scores={'confidence': 0.1}, assertions={'is_correct': False}),
    ]
    report = _make_report(cases)

    evaluator = ROCAUCEvaluator(
        score_key='confidence',
        positive_from='assertions',
        positive_key='is_correct',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert isinstance(results, list)
    assert len(results) == 2

    line_plot = results[0]
    assert isinstance(line_plot, LinePlot)
    assert line_plot.x_label == 'False Positive Rate'
    assert line_plot.y_label == 'True Positive Rate'
    assert line_plot.x_range == (0, 1)
    assert line_plot.y_range == (0, 1)
    assert len(line_plot.curves) == 2  # ROC curve + random baseline
    assert line_plot.curves[1].style == 'dashed'  # baseline is dashed

    scalar = results[1]
    assert isinstance(scalar, ScalarResult)
    assert scalar.title == 'ROC Curve AUC'
    # Perfect separation: AUC should be 1.0
    assert scalar.value == 1.0


def test_roc_auc_evaluator_downsamples():
    """ROCAUCEvaluator downsamples curve points when there are more than n_thresholds."""
    cases = [_make_report_case(f'c{i}', scores={'s': i * 0.1}, assertions={'p': i >= 5}) for i in range(10)]
    report = _make_report(cases)

    evaluator = ROCAUCEvaluator(
        score_key='s',
        positive_from='assertions',
        positive_key='p',
        n_thresholds=3,
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    line_plot = results[0]
    assert isinstance(line_plot, LinePlot)
    roc_curve = line_plot.curves[0]
    # With n_thresholds=3, the ROC points should be downsampled
    assert len(roc_curve.points) <= 3


def test_roc_auc_evaluator_empty():
    """ROCAUCEvaluator returns empty plot and NaN scalar for no data."""
    report = _make_report([])
    evaluator = ROCAUCEvaluator(
        score_key='s',
        positive_from='assertions',
        positive_key='p',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert len(results) == 2
    assert isinstance(results[0], LinePlot)
    assert len(results[0].curves) == 0
    assert isinstance(results[1], ScalarResult)
    assert math.isnan(results[1].value)


def test_roc_auc_evaluator_all_same_class():
    """ROCAUCEvaluator returns empty plot and NaN scalar when all cases are the same class."""
    cases = [
        _make_report_case('c1', scores={'s': 0.9}, assertions={'p': True}),
        _make_report_case('c2', scores={'s': 0.5}, assertions={'p': True}),
    ]
    report = _make_report(cases)

    evaluator = ROCAUCEvaluator(score_key='s', positive_from='assertions', positive_key='p')
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert len(results) == 2
    assert isinstance(results[0], LinePlot)
    assert len(results[0].curves) == 0
    assert isinstance(results[1], ScalarResult)
    assert math.isnan(results[1].value)


def test_roc_auc_evaluator_from_metrics():
    """ROCAUCEvaluator works with score_from='metrics'."""
    cases = [
        _make_report_case('c1', metrics={'score': 0.9}, assertions={'positive': True}),
        _make_report_case('c2', metrics={'score': 0.1}, assertions={'positive': False}),
    ]
    report = _make_report(cases)

    evaluator = ROCAUCEvaluator(
        score_from='metrics',
        score_key='score',
        positive_from='assertions',
        positive_key='positive',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert len(results) == 2
    assert isinstance(results[0], LinePlot)
    assert isinstance(results[1], ScalarResult)


# --- KolmogorovSmirnovEvaluator tests ---


def test_ks_evaluator_basic():
    """KolmogorovSmirnovEvaluator computes KS plot and statistic."""
    cases = [
        _make_report_case('c1', scores={'confidence': 0.9}, assertions={'is_correct': True}),
        _make_report_case('c2', scores={'confidence': 0.8}, assertions={'is_correct': True}),
        _make_report_case('c3', scores={'confidence': 0.3}, assertions={'is_correct': False}),
        _make_report_case('c4', scores={'confidence': 0.1}, assertions={'is_correct': False}),
    ]
    report = _make_report(cases)

    evaluator = KolmogorovSmirnovEvaluator(
        score_key='confidence',
        positive_from='assertions',
        positive_key='is_correct',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert isinstance(results, list)
    assert len(results) == 2

    line_plot = results[0]
    assert isinstance(line_plot, LinePlot)
    assert line_plot.x_label == 'Score'
    assert line_plot.y_label == 'Cumulative Probability'
    assert line_plot.y_range == (0, 1)
    assert len(line_plot.curves) == 2  # Positive + Negative CDFs
    assert line_plot.curves[0].name == 'Positive'
    assert line_plot.curves[1].name == 'Negative'

    scalar = results[1]
    assert isinstance(scalar, ScalarResult)
    assert scalar.title == 'KS Statistic'
    # Perfect separation: KS should be 1.0
    assert scalar.value == 1.0


def test_ks_evaluator_empty():
    """KolmogorovSmirnovEvaluator returns empty plot and NaN scalar for no data."""
    report = _make_report([])
    evaluator = KolmogorovSmirnovEvaluator(
        score_key='s',
        positive_from='assertions',
        positive_key='p',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert len(results) == 2
    assert isinstance(results[0], LinePlot)
    assert len(results[0].curves) == 0
    assert isinstance(results[1], ScalarResult)
    assert math.isnan(results[1].value)


def test_ks_evaluator_all_same_class():
    """KolmogorovSmirnovEvaluator returns empty plot and NaN scalar when all cases are the same class."""
    cases = [
        _make_report_case('c1', scores={'s': 0.9}, assertions={'p': True}),
        _make_report_case('c2', scores={'s': 0.5}, assertions={'p': True}),
    ]
    report = _make_report(cases)

    evaluator = KolmogorovSmirnovEvaluator(score_key='s', positive_from='assertions', positive_key='p')
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    results = evaluator.evaluate(ctx)

    assert len(results) == 2
    assert isinstance(results[0], LinePlot)
    assert len(results[0].curves) == 0
    assert isinstance(results[1], ScalarResult)
    assert math.isnan(results[1].value)


# --- LinePlot rendering test ---


def test_report_rendering_line_plot():
    """LinePlot rendering."""
    from pydantic_evals.reporting.analyses import LinePlotCurve, LinePlotPoint

    report = _make_report([_make_report_case('c1', output='x', expected_output='x')])
    report.analyses = [
        LinePlot(
            title='ROC Curve',
            x_label='FPR',
            y_label='TPR',
            curves=[
                LinePlotCurve(
                    name='test_curve',
                    points=[LinePlotPoint(x=0.0, y=0.0), LinePlotPoint(x=0.5, y=0.8), LinePlotPoint(x=1.0, y=1.0)],
                ),
            ],
        ),
    ]

    rendered = report.render(width=120)
    assert 'ROC Curve' in rendered
    assert 'test_curve' in rendered

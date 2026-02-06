"""Tests for report-level evaluators and experiment-wide analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals import Case, Dataset
    from pydantic_evals.evaluators import (
        ConfusionMatrixEvaluator,
        Evaluator,
        EvaluatorContext,
        PrecisionRecallEvaluator,
        ReportEvaluator,
        ReportEvaluatorContext,
    )
    from pydantic_evals.evaluators.evaluator import EvaluatorOutput
    from pydantic_evals.reporting import EvaluationReport, ReportCase
    from pydantic_evals.reporting.analyses import (
        ConfusionMatrix,
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
    adapter = TypeAdapter(ReportAnalysis)
    restored = adapter.validate_python(data)
    assert isinstance(restored, ConfusionMatrix)
    assert restored.class_labels == ['cat', 'dog']


def test_precision_recall_serialization():
    pr = PrecisionRecall(
        curves=[],
    )
    data = pr.model_dump()
    assert data['type'] == 'precision_recall'

    adapter = TypeAdapter(ReportAnalysis)
    restored = adapter.validate_python(data)
    assert isinstance(restored, PrecisionRecall)


def test_scalar_result_serialization():
    sr = ScalarResult(title='Accuracy', value=0.95, unit='%')
    data = sr.model_dump()
    assert data['type'] == 'scalar'
    assert data['value'] == 0.95

    adapter = TypeAdapter(ReportAnalysis)
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

    adapter = TypeAdapter(ReportAnalysis)
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
) -> ReportCase:
    from pydantic_evals.evaluators.evaluator import EvaluationResult
    from pydantic_evals.evaluators.spec import EvaluatorSpec

    def _make_eval_result(key: str, val: Any) -> EvaluationResult:
        return EvaluationResult(name=key, value=val, reason=None, source=EvaluatorSpec(name='test', arguments=None))

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
    result = evaluator.evaluate(ctx)

    assert isinstance(result, PrecisionRecall)
    assert len(result.curves) == 1
    curve = result.curves[0]
    assert curve.name == 'test'
    assert len(curve.points) > 0
    assert curve.auc is not None
    assert curve.auc > 0


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
    result = evaluator.evaluate(ctx)

    assert isinstance(result, PrecisionRecall)
    assert len(result.curves) == 1


def test_precision_recall_evaluator_empty():
    report = _make_report([])
    evaluator = PrecisionRecallEvaluator(
        score_from='scores',
        score_key='s',
        positive_from='assertions',
        positive_key='p',
    )
    ctx = ReportEvaluatorContext(name='test', report=report, experiment_metadata=None)
    result = evaluator.evaluate(ctx)

    assert isinstance(result, PrecisionRecall)
    assert len(result.curves) == 0


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
            if not ctx.report.cases:
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
            return True

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
        def evaluate(self, ctx: ReportEvaluatorContext) -> list[ScalarResult]:
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
    assert report.analyses[0].title == 'Total Cases'  # type: ignore[union-attr]
    assert report.analyses[0].value == 2  # type: ignore[union-attr]
    assert report.analyses[1].title == 'Case Count Squared'  # type: ignore[union-attr]
    assert report.analyses[1].value == 4  # type: ignore[union-attr]


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


def test_report_evaluator_as_spec_single_arg():
    """Report evaluator with exactly one non-default arg uses tuple form."""
    evaluator = ConfusionMatrixEvaluator(title='My Matrix')
    spec = evaluator.as_spec()
    assert spec.name == 'ConfusionMatrixEvaluator'
    assert isinstance(spec.arguments, tuple)
    assert spec.arguments == ('My Matrix',)


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
    assert dumped['report_evaluators'] == [{'ConfusionMatrixEvaluator': 'Custom'}]


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

"""Tests for pydantic_evals._otel_emit — OTel event emission for online evals."""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_evals._otel_emit import emit_otel_events
    from pydantic_evals.evaluators.evaluator import EvaluationResult, EvaluatorFailure, EvaluatorSpec
    from pydantic_evals.online import SpanReference

with try_import() as logfire_import_successful:
    from logfire.testing import CaptureLogfire

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='pydantic-evals not installed'),
    pytest.mark.skipif(not logfire_import_successful(), reason='logfire not installed'),
]


if TYPE_CHECKING or imports_successful():

    def _spec(name: str = 'Correctness') -> EvaluatorSpec:
        return EvaluatorSpec(name=name, arguments=None)

    def _result(
        name: str,
        value: Any,
        reason: str | None = None,
        evaluator_version: str | None = None,
    ) -> EvaluationResult:
        return EvaluationResult(
            name=name,
            value=value,
            reason=reason,
            source=_spec(name),
            evaluator_version=evaluator_version,
        )

    def _target(name: str = 'my_agent') -> str:
        return name


def test_emits_event_with_parent_span(capfire: CaptureLogfire):
    span_ref = SpanReference(trace_id='0' * 31 + '1', span_id='0' * 15 + '2')

    emit_otel_events(
        results=[_result('Correctness', True, reason='looks right')],
        failures=[],
        span_reference=span_ref,
        target=_target(),
    )

    finished = capfire.log_exporter.get_finished_logs()
    assert len(finished) == 1
    record = finished[0].log_record
    attrs = dict(record.attributes or {})

    assert record.event_name == 'gen_ai.evaluation.result'
    assert record.body == 'Correctness: pass'
    assert attrs['gen_ai.evaluation.name'] == 'Correctness'
    assert attrs['gen_ai.evaluation.score.value'] == 1.0
    assert attrs['gen_ai.evaluation.score.label'] == 'pass'
    assert attrs['gen_ai.evaluation.explanation'] == 'looks right'
    assert attrs['gen_ai.evaluation.target'] == 'my_agent'
    assert 'gen_ai.evaluation.evaluator_source' in attrs

    # Parented to referenced span.
    assert record.trace_id == 1
    assert record.span_id == 2


def test_bool_false_emits_fail_label(capfire: CaptureLogfire):
    emit_otel_events(
        results=[_result('X', False)],
        failures=[],
        span_reference=None,
        target=_target('f'),
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert attrs['gen_ai.evaluation.score.value'] == 0.0
    assert attrs['gen_ai.evaluation.score.label'] == 'fail'


def test_numeric_score_only(capfire: CaptureLogfire):
    emit_otel_events(
        results=[_result('X', 0.73)],
        failures=[],
        span_reference=None,
        target=_target('f'),
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert attrs['gen_ai.evaluation.score.value'] == 0.73
    assert 'gen_ai.evaluation.score.label' not in attrs


def test_string_label_only(capfire: CaptureLogfire):
    emit_otel_events(
        results=[_result('X', 'excellent')],
        failures=[],
        span_reference=None,
        target=_target('f'),
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert attrs['gen_ai.evaluation.score.label'] == 'excellent'
    assert 'gen_ai.evaluation.score.value' not in attrs


def test_failure_emits_error_type_and_no_score(capfire: CaptureLogfire):
    failure = EvaluatorFailure(
        name='X',
        error_message='boom',
        error_stacktrace='tb',
        source=_spec('X'),
    )
    emit_otel_events(
        results=[],
        failures=[failure],
        span_reference=None,
        target=_target('f'),
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert attrs['error.type'] == 'pydantic_evals.EvaluatorFailure'
    assert attrs['gen_ai.evaluation.explanation'] == 'boom'
    assert 'gen_ai.evaluation.score.value' not in attrs
    assert 'gen_ai.evaluation.score.label' not in attrs


def test_evaluator_version_and_extra_attributes(capfire: CaptureLogfire):
    emit_otel_events(
        results=[_result('X', True, evaluator_version='v2')],
        failures=[],
        span_reference=None,
        target=_target('f'),
        extra_attributes={'team': 'platform'},
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert attrs['gen_ai.evaluation.evaluator_version'] == 'v2'
    assert attrs['team'] == 'platform'


def test_no_version_attribute_when_none(capfire: CaptureLogfire):
    emit_otel_events(
        results=[_result('X', True)],
        failures=[],
        span_reference=None,
        target=_target('f'),
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert 'gen_ai.evaluation.evaluator_version' not in attrs


def test_failure_carries_evaluator_version(capfire: CaptureLogfire):
    failure = EvaluatorFailure(
        name='X',
        error_message='boom',
        error_stacktrace='tb',
        source=_spec('X'),
        evaluator_version='v3',
    )
    emit_otel_events(
        results=[],
        failures=[failure],
        span_reference=None,
        target=_target('f'),
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert attrs['gen_ai.evaluation.evaluator_version'] == 'v3'


def test_empty_results_and_failures_emits_nothing(capfire: CaptureLogfire):
    emit_otel_events(
        results=[],
        failures=[],
        span_reference=None,
        target=_target('f'),
    )

    assert list(capfire.log_exporter.get_finished_logs()) == []


def test_failure_without_error_message_omits_explanation(capfire: CaptureLogfire):
    failure = EvaluatorFailure(
        name='X',
        error_message='',
        error_stacktrace='tb',
        source=_spec('X'),
    )
    emit_otel_events(
        results=[],
        failures=[failure],
        span_reference=None,
        target=_target('f'),
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert attrs['error.type'] == 'pydantic_evals.EvaluatorFailure'
    assert 'gen_ai.evaluation.explanation' not in attrs


def test_non_scalar_value_yields_no_score_attrs(capfire: CaptureLogfire):
    """Defensive: a value that isn't bool/int/float/str leaves score attrs unset."""
    # A `None` value bypasses the EvaluationScalar contract and exercises the fall-through.
    emit_otel_events(
        results=[_result('X', value=None)],
        failures=[],
        span_reference=None,
        target=_target('f'),
    )

    attrs = dict(capfire.log_exporter.get_finished_logs()[0].log_record.attributes or {})
    assert 'gen_ai.evaluation.score.value' not in attrs
    assert 'gen_ai.evaluation.score.label' not in attrs


def test_body_formatting_for_score_types(capfire: CaptureLogfire):
    """The log body is a short human-readable summary shown in the live trace view."""
    emit_otel_events(
        results=[
            _result('Passed', True),
            _result('Failed', False),
            _result('Score', 0.73),
            _result('Whole', 5),
            _result('Tone', 'neutral'),
        ],
        failures=[],
        span_reference=None,
        target=_target('f'),
    )

    bodies = [r.log_record.body for r in capfire.log_exporter.get_finished_logs()]
    assert bodies == [
        'Passed: pass',
        'Failed: fail',
        'Score: 0.73',
        'Whole: 5',
        'Tone: neutral',
    ]


def test_failure_body(capfire: CaptureLogfire):
    """Failure body includes the error message when present, name alone otherwise."""
    emit_otel_events(
        results=[],
        failures=[
            EvaluatorFailure(
                name='Judge',
                error_message='ValueError: bad input',
                error_stacktrace='tb',
                source=_spec('Judge'),
            ),
            EvaluatorFailure(
                name='Silent',
                error_message='',
                error_stacktrace='tb',
                source=_spec('Silent'),
            ),
        ],
        span_reference=None,
        target=_target('f'),
    )

    bodies = [r.log_record.body for r in capfire.log_exporter.get_finished_logs()]
    assert bodies == ['Judge failed: ValueError: bad input', 'Silent failed']

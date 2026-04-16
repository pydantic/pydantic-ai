"""Internal OTel event emission for online evaluation dispatch.

Emits one `gen_ai.evaluation.result` log event per `EvaluationResult` or
`EvaluatorFailure`, following the OpenTelemetry semantic conventions:
https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-events/#event-gen_aievaluationresult

This is called unconditionally by `dispatch_evaluators` for every evaluator
run — it is the default observability surface for online evals, matching how
offline evals emit OTel spans via `logfire_span`. If no OTel SDK is configured
in the user's process, `get_logger()` returns a no-op logger and events are
silently dropped.

Events are parented to the span being evaluated when a `span_reference` is
supplied, so they appear nested under the original function call in the trace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry import context as otel_context, trace
from opentelemetry._logs import LogRecord, get_logger
from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from .evaluators.evaluator import EvaluationResult, EvaluatorFailure
from .evaluators.spec import EvaluatorSpec

__all__ = ('emit_otel_events',)


# --- Attribute name constants -------------------------------------------------
# Standard OTel semconv names. Stable.
_ATTR_EVAL_NAME = 'gen_ai.evaluation.name'
_ATTR_SCORE_VALUE = 'gen_ai.evaluation.score.value'
_ATTR_SCORE_LABEL = 'gen_ai.evaluation.score.label'
_ATTR_EXPLANATION = 'gen_ai.evaluation.explanation'
_ATTR_ERROR_TYPE = 'error.type'

# Extensions beyond the current OTel semconv, placed in the `gen_ai.*` namespace
# on the assumption that this is where analogous attributes will land upstream.
# If upstream adopts different names, update these constants together with
# downstream queries/materialized views.
_ATTR_TARGET = 'gen_ai.evaluation.target'
_ATTR_EVALUATOR_SOURCE = 'gen_ai.evaluation.evaluator_source'
_ATTR_EVALUATOR_VERSION = 'gen_ai.evaluation.evaluator_version'

_EVENT_NAME = 'gen_ai.evaluation.result'
_OTEL_SCOPE = 'pydantic-evals'

# Acquired lazily: calling `get_logger()` at import time is safe (OTel returns
# a proxy that picks up the configured provider later), but we delay it to the
# first call so test fixtures that replace the global provider work predictably.
_logger: Any | None = None


def _get_logger() -> Any:
    global _logger
    if _logger is None:
        _logger = get_logger(_OTEL_SCOPE)
    return _logger


def emit_otel_events(
    *,
    results: Sequence[EvaluationResult],
    failures: Sequence[EvaluatorFailure],
    span_reference: Any | None,
    target: str,
    extra_attributes: Mapping[str, Any] | None = None,
) -> None:
    """Emit one `gen_ai.evaluation.result` OTel event per result/failure.

    Each `EvaluationResult` produces one event; each `EvaluatorFailure`
    produces one event with `error.type` set and no score. Events are parented
    to the span referenced by `span_reference` when supplied, so they appear
    nested under the original function call in the trace.

    Args:
        results: Evaluation results from a single evaluator run. Each result's
            `evaluator_version` is written to `gen_ai.evaluation.evaluator_version`.
        failures: Failures from a single evaluator run. Each failure's
            `evaluator_version` is written to `gen_ai.evaluation.evaluator_version`.
        span_reference: Optional reference to the span being evaluated.
            When supplied, emitted events are parented to that span.
        target: Name of the function/agent being evaluated. Written to
            `gen_ai.evaluation.target`.
        extra_attributes: Optional extra attributes to include on every emitted
            event. Escape hatch for custom metadata without subclassing.
    """
    if not results and not failures:
        return

    parent_ctx = _build_parent_context(span_reference)
    if parent_ctx is None:
        for result in results:
            _emit_result(result, target, extra_attributes)
        for failure in failures:
            _emit_failure(failure, target, extra_attributes)
        return

    token = otel_context.attach(parent_ctx)
    try:
        for result in results:
            _emit_result(result, target, extra_attributes)
        for failure in failures:
            _emit_failure(failure, target, extra_attributes)
    finally:
        otel_context.detach(token)


# --- emission helpers ---------------------------------------------------------


def _emit_result(
    result: EvaluationResult,
    target: str,
    extra_attributes: Mapping[str, Any] | None,
) -> None:
    attrs = _base_attrs(target, result.evaluator_version, extra_attributes)
    attrs[_ATTR_EVAL_NAME] = result.name
    _set_score_attrs(attrs, result.value)
    if result.reason is not None:
        attrs[_ATTR_EXPLANATION] = result.reason
    attrs[_ATTR_EVALUATOR_SOURCE] = _serialize_evaluator_source(result.source)
    _get_logger().emit(LogRecord(event_name=_EVENT_NAME, attributes=attrs))


def _emit_failure(
    failure: EvaluatorFailure,
    target: str,
    extra_attributes: Mapping[str, Any] | None,
) -> None:
    attrs = _base_attrs(target, failure.evaluator_version, extra_attributes)
    attrs[_ATTR_EVAL_NAME] = failure.name
    attrs[_ATTR_ERROR_TYPE] = 'pydantic_evals.EvaluatorFailure'
    if failure.error_message:
        attrs[_ATTR_EXPLANATION] = failure.error_message
    attrs[_ATTR_EVALUATOR_SOURCE] = _serialize_evaluator_source(failure.source)
    _get_logger().emit(LogRecord(event_name=_EVENT_NAME, attributes=attrs))


def _base_attrs(
    target: str,
    evaluator_version: str | None,
    extra_attributes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        _ATTR_TARGET: target,
    }
    if evaluator_version is not None:
        attrs[_ATTR_EVALUATOR_VERSION] = evaluator_version
    if extra_attributes:
        attrs.update(extra_attributes)
    return attrs


# --- Module-level helpers -----------------------------------------------------


def _build_parent_context(span_reference: Any | None) -> Context | None:
    """Build an OTel context with a non-recording parent span, or None."""
    if span_reference is None:
        return None
    try:
        trace_id = int(span_reference.trace_id, 16)
        span_id = int(span_reference.span_id, 16)
    except (AttributeError, ValueError, TypeError):  # pragma: no cover
        return None
    span_ctx = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    return trace.set_span_in_context(NonRecordingSpan(span_ctx))


def _set_score_attrs(attrs: dict[str, Any], value: Any) -> None:
    """Populate `score.value` and/or `score.label` from a pydantic-evals scalar.

    - bool → both: `score.value` = 0.0/1.0, `score.label` = `"pass"`/`"fail"`.
      Dual representation so numeric queries and categorical queries both work.
    - int/float → `score.value` only.
    - str → `score.label` only (evaluator returned a categorical tag).
    """
    if isinstance(value, bool):
        attrs[_ATTR_SCORE_VALUE] = 1.0 if value else 0.0
        attrs[_ATTR_SCORE_LABEL] = 'pass' if value else 'fail'
    elif isinstance(value, (int, float)):
        attrs[_ATTR_SCORE_VALUE] = float(value)
    elif isinstance(value, str):
        attrs[_ATTR_SCORE_LABEL] = value


def _serialize_evaluator_source(source: EvaluatorSpec) -> str:
    """JSON-serialize an EvaluatorSpec to a string attribute.

    We store as a string (rather than individual attributes) because OTel log
    attributes are scalar/sequence typed and the spec's `arguments` field can
    be an arbitrary dict of kwargs. The materialized view can JSON-parse this
    column in DataFusion when needed.
    """
    return source.model_dump_json()

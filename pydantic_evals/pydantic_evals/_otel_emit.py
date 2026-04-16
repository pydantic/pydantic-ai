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

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context, trace
from opentelemetry._logs import LogRecord, get_logger
from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from .evaluators.evaluator import EvaluationResult, EvaluatorFailure

if TYPE_CHECKING:
    from .online import EvaluationTarget

__all__ = ('emit_otel_events',)


# --- Attribute name constants -------------------------------------------------
# Standard OTel semconv names. Stable.
_ATTR_EVAL_NAME = 'gen_ai.evaluation.name'
_ATTR_SCORE_VALUE = 'gen_ai.evaluation.score.value'
_ATTR_SCORE_LABEL = 'gen_ai.evaluation.score.label'
_ATTR_EXPLANATION = 'gen_ai.evaluation.explanation'
_ATTR_ERROR_TYPE = 'error.type'

# Logfire extensions — not official semconv yet. Namespace is subject to change:
# if OTel adopts these upstream (or we negotiate a `gen_ai.*` namespace for them),
# update these constants and downstream queries/materialized views together.
_ATTR_TARGET = 'logfire.evaluation.target'
_ATTR_TARGET_TYPE = 'logfire.evaluation.target_type'
_ATTR_EVALUATOR_SOURCE = 'logfire.evaluation.evaluator_source'
_ATTR_EVALUATOR_VERSION = 'logfire.evaluation.evaluator_version'

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
    target: EvaluationTarget,
    evaluator_version: str | None,
    extra_attributes: Mapping[str, Any] | None = None,
) -> None:
    """Emit one `gen_ai.evaluation.result` OTel event per result/failure.

    Each `EvaluationResult` produces one event; each `EvaluatorFailure`
    produces one event with `error.type` set and no score. Events are parented
    to the span referenced by `span_reference` when supplied, so they appear
    nested under the original function call in the trace.

    Args:
        results: Evaluation results from a single evaluator run.
        failures: Failures from a single evaluator run.
        span_reference: Optional reference to the span being evaluated.
            When supplied, emitted events are parented to that span.
        target: Identifies the function/agent being evaluated. Written to
            `logfire.evaluation.target` / `logfire.evaluation.target_type`.
        evaluator_version: Optional evaluator version tag.
        extra_attributes: Optional extra attributes to include on every emitted
            event. Escape hatch for custom metadata without subclassing.
    """
    if not results and not failures:
        return

    parent_ctx = _build_parent_context(span_reference)
    if parent_ctx is None:
        for result in results:
            _emit_result(result, target, evaluator_version, extra_attributes)
        for failure in failures:
            _emit_failure(failure, target, evaluator_version, extra_attributes)
        return

    token = otel_context.attach(parent_ctx)
    try:
        for result in results:
            _emit_result(result, target, evaluator_version, extra_attributes)
        for failure in failures:
            _emit_failure(failure, target, evaluator_version, extra_attributes)
    finally:
        otel_context.detach(token)


# --- emission helpers ---------------------------------------------------------


def _emit_result(
    result: EvaluationResult,
    target: EvaluationTarget,
    evaluator_version: str | None,
    extra_attributes: Mapping[str, Any] | None,
) -> None:
    attrs = _base_attrs(target, evaluator_version, extra_attributes)
    attrs[_ATTR_EVAL_NAME] = result.name
    _set_score_attrs(attrs, result.value)
    if result.reason is not None:
        attrs[_ATTR_EXPLANATION] = result.reason
    attrs[_ATTR_EVALUATOR_SOURCE] = _serialize_evaluator_source(result.source)
    _get_logger().emit(LogRecord(event_name=_EVENT_NAME, attributes=attrs))


def _emit_failure(
    failure: EvaluatorFailure,
    target: EvaluationTarget,
    evaluator_version: str | None,
    extra_attributes: Mapping[str, Any] | None,
) -> None:
    attrs = _base_attrs(target, evaluator_version, extra_attributes)
    attrs[_ATTR_EVAL_NAME] = failure.name
    attrs[_ATTR_ERROR_TYPE] = 'pydantic_evals.EvaluatorFailure'
    if failure.error_message:
        attrs[_ATTR_EXPLANATION] = failure.error_message
    attrs[_ATTR_EVALUATOR_SOURCE] = _serialize_evaluator_source(failure.source)
    _get_logger().emit(LogRecord(event_name=_EVENT_NAME, attributes=attrs))


def _base_attrs(
    target: EvaluationTarget,
    evaluator_version: str | None,
    extra_attributes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {
        _ATTR_TARGET: target.name,
        _ATTR_TARGET_TYPE: target.type,
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


def _serialize_evaluator_source(source: Any) -> str:
    """JSON-serialize an EvaluatorSpec to a string attribute.

    We store as a string (rather than individual attributes) because the
    spec can contain nested kwargs of arbitrary shape. The materialized view
    can JSON-parse this column in DataFusion when needed.
    """
    if hasattr(source, 'model_dump'):
        try:
            return json.dumps(source.model_dump(), default=str)
        except (TypeError, ValueError):  # pragma: no cover
            pass
    if hasattr(source, '__dict__'):
        try:
            return json.dumps(
                {k: v for k, v in source.__dict__.items() if not k.startswith('_')},
                default=str,
            )
        except (TypeError, ValueError):  # pragma: no cover
            pass
    return json.dumps({'repr': repr(source)})

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

from opentelemetry import baggage, trace
from opentelemetry._logs import LogRecord, get_logger
from opentelemetry.context import Context
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from .evaluators.evaluator import EvaluationResult, EvaluatorFailure
from .evaluators.spec import EvaluatorSpec

__all__ = ('build_parent_context', 'emit_otel_events')


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
    target: str,
    include_baggage: bool = True,
    extra_attributes: Mapping[str, Any] | None = None,
) -> None:
    """Emit one `gen_ai.evaluation.result` OTel event per result/failure.

    Each `EvaluationResult` produces one event; each `EvaluatorFailure` produces
    one event with `error.type` set and no score. The caller is responsible for
    attaching the appropriate parent OTel context (e.g. via
    `build_parent_context`) so events appear nested under the function call.

    Args:
        results: Evaluation results from a single evaluator run. Each result's
            `evaluator_version` is written to `gen_ai.evaluation.evaluator_version`.
        failures: Failures from a single evaluator run. Each failure's
            `evaluator_version` is written to `gen_ai.evaluation.evaluator_version`.
        target: Name of the function/agent being evaluated. Written to
            `gen_ai.evaluation.target`.
        include_baggage: When True (the default), each emitted event also carries
            every key from the current OTel baggage as an attribute. Standard
            `gen_ai.*` and `error.type` attributes always win on conflict.
        extra_attributes: Optional extra attributes to include on every emitted
            event. Escape hatch for custom metadata without subclassing.
    """
    if not results and not failures:
        return

    base_extras = _collect_extra_attributes(include_baggage, extra_attributes)
    for result in results:
        _emit_result(result, target, base_extras)
    for failure in failures:
        _emit_failure(failure, target, base_extras)


def _collect_extra_attributes(
    include_baggage: bool,
    extra_attributes: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    """Merge OTel baggage with caller-provided extras (caller wins)."""
    if not include_baggage:
        return extra_attributes
    bag = baggage.get_all()
    if not bag and not extra_attributes:
        return None
    merged: dict[str, Any] = {str(k): v for k, v in bag.items()}
    if extra_attributes:
        merged.update(extra_attributes)
    return merged


def build_parent_context(span_reference: Any | None) -> Context | None:
    """Public: build an OTel context with a non-recording parent span, or None."""
    return _build_parent_context(span_reference)


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
    _get_logger().emit(LogRecord(event_name=_EVENT_NAME, body=_format_result_body(result), attributes=attrs))


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
    _get_logger().emit(LogRecord(event_name=_EVENT_NAME, body=_format_failure_body(failure), attributes=attrs))


def _base_attrs(
    target: str,
    evaluator_version: str | None,
    extra_attributes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    # Apply extras first so standard attributes always win — consistent with the
    # standard attrs written by _emit_result / _emit_failure after this returns.
    attrs: dict[str, Any] = dict(extra_attributes) if extra_attributes else {}
    attrs[_ATTR_TARGET] = target
    if evaluator_version is not None:
        attrs[_ATTR_EVALUATOR_VERSION] = evaluator_version
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


def _format_result_body(result: EvaluationResult) -> str:
    """Build the human-readable log body for a successful evaluation.

    The body is shown inline in the Logfire live trace view, so it should be
    short and dense. Format mirrors a simple `key=value` expression.
    """
    return f'evaluation: {result.name}={_format_score(result.value)}'


def _format_failure_body(failure: EvaluatorFailure) -> str:
    """Build the human-readable log body for an evaluator failure."""
    if failure.error_message:
        return f'evaluation: {failure.name} failed: {failure.error_message}'
    return f'evaluation: {failure.name} failed'


def _format_score(value: Any) -> str:
    """Render a `EvaluationScalar` for display in the log body.

    Bools render as the literal `True`/`False`, strings are quoted so they
    read like a value expression, numerics stay bare.
    """
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, float):
        # `g` drops trailing zeros but falls back to scientific notation for
        # very large/small values; that's fine for a short status string.
        return format(value, 'g')
    if isinstance(value, str):
        return repr(value)
    return str(value)


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

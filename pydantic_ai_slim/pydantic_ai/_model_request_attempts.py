"""Record the attempts at a model request that failed before one produced the response, or before all of them failed.

An attempt is recorded twice: as a [`ModelRequestAttempt`][pydantic_ai.messages.ModelRequestAttempt] on the
response that answered (or on the `FallbackExceptionGroup` when none did), and, when the request is
instrumented, as an ERROR child span of the request's `chat` span. Both are built here, so every place
that makes more than one attempt at a request records them the same way.
"""

from __future__ import annotations as _annotations

from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import perf_counter_ns
from typing import TYPE_CHECKING

from opentelemetry.trace import Span, Status, StatusCode, Tracer, set_span_in_context
from opentelemetry.util.types import AttributeValue

from ._genai_prices import fill_response_cost
from ._instrumentation import (
    model_attributes,
    record_exception,
    response_attributes,
    set_error_status,
    span_include_content,
)
from .exceptions import FallbackExceptionGroup
from .messages import ModelRequestAttempt, ModelResponse

if TYPE_CHECKING:
    from .models import Model

__all__ = ('AttemptStart', 'carried_attempts', 'carry_attempts', 'failed_attempt', 'record_attempt_span')

ATTEMPT_ATTRIBUTE = 'pydantic_ai.model_request.attempt'
"""The zero-based position of an attempt among the attempts at its request."""


@dataclass(frozen=True)
class AttemptStart:
    """When an attempt started: a wall-clock timestamp for the record, and a monotonic one to time it by."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    _monotonic_ns: int = field(default_factory=perf_counter_ns)

    def elapsed(self) -> timedelta:
        """How long it has been since the attempt started, unaffected by changes to the system clock."""
        return timedelta(microseconds=(perf_counter_ns() - self._monotonic_ns) / 1e3)


_CARRIED_ATTEMPTS = '_pydantic_ai_failed_attempts'


def carry_attempts(error: BaseException, attempts: list[ModelRequestAttempt]) -> None:
    """Carry the attempts a model made before `error` ended its request, so the run still counts their usage.

    A model that makes attempts of its own, like `FallbackModel`, lists them on the response that answers, or on the
    `FallbackExceptionGroup` when every one failed. When the request ends with another error instead, like a
    cancellation, a timeout or an error `fallback_on` doesn't cover, the rejected responses among them were still
    billed, so they ride on that error to the run, after any an inner model carried.
    """
    # A group lists the attempts itself.
    if attempts and not isinstance(error, FallbackExceptionGroup):
        carried = carried_attempts(error)
        setattr(error, _CARRIED_ATTEMPTS, [*carried, *(attempt for attempt in attempts if attempt not in carried)])


def carried_attempts(error: BaseException) -> list[ModelRequestAttempt]:
    """The attempts `carry_attempts` put on `error`, if any."""
    attempts: list[ModelRequestAttempt] = getattr(error, _CARRIED_ATTEMPTS, [])
    return attempts


def failed_attempt(
    model: Model, failure: Exception | ModelResponse, *, start: AttemptStart, duration: timedelta
) -> ModelRequestAttempt:
    """Describe an attempt at a request to `model` that raised `failure`, or returned the rejected response `failure`.

    A rejected response's cost is filled in first, so the attempt carries it.
    """
    timestamp = start.timestamp
    if isinstance(failure, Exception):
        return ModelRequestAttempt(
            model_name=model.model_name,
            provider_name=model.system,
            outcome='error',
            error=_describe_error(failure),
            timestamp=timestamp,
            duration=duration,
        )
    fill_response_cost(failure)
    return ModelRequestAttempt(
        model_name=failure.model_name or model.model_name,
        provider_name=failure.provider_name or model.system,
        outcome='rejected',
        timestamp=timestamp,
        duration=duration,
        usage=failure.usage,
    )


def record_attempt_span(
    attempt: ModelRequestAttempt,
    failure: Exception | ModelResponse,
    *,
    model: Model,
    index: int,
    parent: Span,
    tracer: Tracer,
    rejected_status: str = 'Response rejected',
) -> None:
    """Record a failed attempt as an ERROR child span of `parent`, the request's `chat` span.

    `rejected_status` is the span's status description when `failure` is a rejected response, which says what
    rejected it where that's known.

    The `chat` span keeps its own outcome, that of the model that answered, the way a failed tool call
    gets its own ERROR span under an agent run that goes on to succeed. The span is only opened once
    the attempt has failed, back-dated to when it started, so the answering attempt, which `chat`
    already describes, gets none, and spans the tried model opened itself, like a decision model's
    `decide`, sit beside this span rather than inside it. It is deliberately not named `chat`, so
    model-call views don't count it as a model call, but a rejected response's usage and cost are
    recorded on it, since they were billed. An error's message and stack trace follow `parent`'s
    `include_content`, since a provider's error response can echo the request.

    Best-effort: telemetry never changes the outcome of the request.
    """
    with suppress(Exception):
        attributes: dict[str, AttributeValue] = {**model_attributes(model), ATTEMPT_ATTRIBUTE: index}
        if isinstance(failure, ModelResponse):
            attributes.update(response_attributes(failure, failure.model_name))
        start_time = _to_ns(attempt.timestamp)
        span = tracer.start_span(
            f'model request attempt {model.model_name}',
            context=set_span_in_context(parent),
            attributes=attributes,
            start_time=start_time,
        )
        # Ended even if describing the failure raises, or the span would never be exported.
        try:
            if isinstance(failure, Exception):
                include_content = span_include_content(parent)
                record_exception(span, failure, include_content=include_content)
                set_error_status(span, failure, include_content=include_content)
            else:
                span.set_status(Status(StatusCode.ERROR, rejected_status))
        finally:
            span.end(start_time + round(attempt.duration.total_seconds() * 1e9))


def _to_ns(value: datetime) -> int:
    return int(value.timestamp() * 1e9)


def _describe_error(error: Exception) -> str:
    """Describe an error as `'ExceptionType: message'`, the way an OTel span's ERROR status does."""
    name = type(error).__name__
    try:
        message = str(error)
    except Exception:
        # An exception whose `__str__` raises shouldn't cost the attempt its record.
        return name
    return f'{name}: {message}' if message else name

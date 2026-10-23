"""Record the attempts at a model request that failed before one produced the response, or before all of them failed.

An attempt is recorded twice: as a [`ModelRequestAttempt`][pydantic_ai.messages.ModelRequestAttempt] on the
response that answered (or on the `FallbackExceptionGroup` when none did), and, when the request is
instrumented, as an ERROR child span of the request's `chat` span. Both are built here, so every place
that makes more than one attempt at a request records them the same way.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
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
from .messages import ModelRequestAttempt, ModelResponse
from .models.wrapper import WrapperModel

if TYPE_CHECKING:
    from .models import Model

__all__ = (
    'AttemptStart',
    'FailedAttemptCommitter',
    'committing_failed_attempts',
    'failed_attempt',
    'failed_attempt_committer',
    'record_attempt_span',
)

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


FailedAttemptCommitter = Callable[[ModelRequestAttempt], None]
"""Commits the usage of a failed attempt at a request to the agent run making it."""

_failed_attempt_committer: ContextVar[tuple[Model, FailedAttemptCommitter] | None] = ContextVar(
    'pydantic_ai.failed_attempt_committer', default=None
)


@contextmanager
def committing_failed_attempts(model: Model, commit: FailedAttemptCommitter) -> Generator[None]:
    """Have the request about to be made to `model` hand each attempt that fails to `commit`, as soon as it fails.

    The agent graph uses this to count a rejected response's usage towards the run when it's rejected,
    rather than once the request has answered, so a run that's cancelled or times out while a later
    attempt is still running keeps it. Only `model` itself (or the model it wraps) takes the committer:
    see `failed_attempt_committer`.
    """
    token = _failed_attempt_committer.set((model, commit))
    try:
        yield
    finally:
        _failed_attempt_committer.reset(token)


def failed_attempt_committer(model: Model) -> FailedAttemptCommitter | None:
    """The committer for a request to `model`, if the innermost `committing_failed_attempts` is for `model`.

    It's also for `model` when it was for a `WrapperModel` that wraps `model`, at any depth. Any other
    model making a request inside that scope, such as one that a custom model calls itself, gets
    `None`: its attempts aren't listed on the response the scope's request ends with, so committing
    their usage would count usage that request doesn't report.
    """
    if (bound := _failed_attempt_committer.get()) is None:
        return None
    request_model, commit = bound
    while request_model is not model:
        if not isinstance(request_model, WrapperModel):
            return None
        request_model = request_model.wrapped
    return commit


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
) -> None:
    """Record a failed attempt as an ERROR child span of `parent`, the request's `chat` span.

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
                span.set_status(Status(StatusCode.ERROR, 'Response rejected by a `fallback_on` response handler'))
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

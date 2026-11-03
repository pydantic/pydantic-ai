"""The core shared by both ways to fall back between models.

[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel] walks its chain inside a single model
request, so it also works outside an agent (`direct.model_request`, a standalone `Model`); the
[`ModelFallback`][pydantic_ai.capabilities.ModelFallback] capability walks it through the agent's own attempt
loop, one [`RetryModelRequest`][pydantic_ai.exceptions.RetryModelRequest] at a time. Everything they
have in common lives here, so a fix lands once and the two can never disagree:

- what counts as a failure (`FallbackPredicates`), and what an exhausted chain raises (`FallbackChain`);
- the records and spans of the attempts that failed (`AttemptLog`), which the agent's attempt loop also
  keeps for any capability that raises `RetryModelRequest`;
- opening a candidate's stream so that an error before its first event can still fall back
  (`request_stream_until_first_event`), and the warning for response handlers on a streamed request;
- pinning a suspended response to the model that started it.
"""

from __future__ import annotations as _annotations

import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard, assert_never

from opentelemetry.trace import Span, Tracer

from ._model_request_attempts import AttemptStart, failed_attempt, record_attempt_span
from ._utils import PeekableAsyncStream, await_maybe, get_first_param_type, is_str_dict
from .exceptions import FallbackExceptionGroup, UserError
from .messages import (
    _PYDANTIC_AI_METADATA_KEY,  # pyright: ignore[reportPrivateUsage]
    ModelMessage,
    ModelRequestAttempt,
    ModelResponse,
    ModelResponseStreamEvent,
)
from .models._request_timeout import stream_under_deadline

if TYPE_CHECKING:
    from ._run_context import RunContext
    from .models import Model, ModelRequestParameters, StreamedResponse
    from .models._request_timeout import RequestDeadline
    from .settings import ModelSettings

__all__ = (
    'ExceptionHandler',
    'ResponseHandler',
    'FallbackOn',
    'FallbackPredicates',
    'FallbackChain',
    'AttemptLog',
    'request_stream_until_first_event',
    'warn_response_handlers_not_applied',
    'ResponseRejected',
    'raise_fallback_exception_group',
    'stamp_continuation_pin',
    'continuation_pin',
    'FALLBACK_MODEL_PIN_KEY',
)

FALLBACK_MODEL_PIN_KEY = 'fallback_model_id'
"""Where `FallbackModel` pins a suspended response to the inner model that started it."""


def stamp_continuation_pin(response: ModelResponse | StreamedResponse, model_id: str, *, key: str) -> None:
    """Record which model produced a suspended response, so its continuation goes back to it.

    Stored in `metadata['__pydantic_ai__']` to keep framework routing state apart from provider data.
    `response` is a `ModelResponse` or a `StreamedResponse`, whose metadata ends up on the response.
    """
    metadata = response.metadata if response.metadata is not None else {}
    response.metadata = metadata
    pydantic_ai_meta = metadata.get(_PYDANTIC_AI_METADATA_KEY)
    if not is_str_dict(pydantic_ai_meta):
        pydantic_ai_meta = {}
        metadata[_PYDANTIC_AI_METADATA_KEY] = pydantic_ai_meta
    pydantic_ai_meta[key] = model_id


def continuation_pin(response: ModelResponse, *, key: str) -> str | None:
    """The model a suspended response was pinned to by `stamp_continuation_pin` under `key`, if any."""
    pydantic_ai_meta = (response.metadata or {}).get(_PYDANTIC_AI_METADATA_KEY)
    model_id = pydantic_ai_meta.get(key) if is_str_dict(pydantic_ai_meta) else None
    return model_id if isinstance(model_id, str) else None


ExceptionHandler = Callable[[Exception], Awaitable[bool]] | Callable[[Exception], bool]
"""A sync or async callable that decides whether an exception should trigger fallback."""

ResponseHandler = Callable[[ModelResponse], Awaitable[bool]] | Callable[[ModelResponse], bool]
"""A sync or async callable that decides whether a model response should trigger fallback."""

FallbackOn = (
    type[Exception]
    | tuple[type[Exception], ...]
    | ExceptionHandler
    | ResponseHandler
    | Sequence[type[Exception] | ExceptionHandler | ResponseHandler]
)
"""The type of the `fallback_on` parameter to [`ModelFallback`][pydantic_ai.capabilities.ModelFallback]
and [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]."""


class ResponseRejected(Exception):
    """Raised within a `FallbackExceptionGroup` when model responses are rejected by a response handler."""

    def __init__(self, rejected_count: int):
        super().__init__(f'{rejected_count} model response(s) rejected by fallback_on handler')


def _is_exception_type(value: Any) -> TypeGuard[type[Exception]]:
    """Check if value is a single exception type."""
    return isinstance(value, type) and issubclass(value, Exception)


def _exception_types_to_handler(exception_types: tuple[type[Exception], ...]) -> ExceptionHandler:
    """Create an exception handler from a tuple of exception types."""

    def handler(exc: Exception) -> bool:
        return isinstance(exc, exception_types)

    return handler


@dataclass
class FallbackPredicates:
    """The parsed `fallback_on` predicates, split by what they inspect."""

    exception_handlers: list[ExceptionHandler]
    response_handlers: list[ResponseHandler]

    @classmethod
    def parse(cls, fallback_on: FallbackOn, *, owner: str) -> FallbackPredicates:
        """Parse the `fallback_on` argument into exception and response handlers.

        `owner` names the class in the error raised for an empty `fallback_on`, so the message
        points at whichever entry point the user actually called.
        """
        predicates = cls(exception_handlers=[], response_handlers=[])
        if _is_exception_type(fallback_on):
            # Single exception type
            predicates.exception_handlers.append(_exception_types_to_handler((fallback_on,)))
        elif callable(fallback_on):
            # Single callable - auto-detect by type hints
            predicates._add_handler(fallback_on)
        elif isinstance(fallback_on, Sequence) and not isinstance(fallback_on, (str, bytes)):
            # Sequence of mixed handlers/types
            for item in fallback_on:
                if _is_exception_type(item):
                    predicates.exception_handlers.append(_exception_types_to_handler((item,)))
                elif callable(item):
                    predicates._add_handler(item)
                else:
                    # Types guarantee all items are exception types or callables
                    assert_never(item)
        else:
            assert_never(fallback_on)  # type: ignore[arg-type]  # pyright can't narrow str/bytes exclusion

        if not predicates.exception_handlers and not predicates.response_handlers:
            raise UserError(
                f'`{owner}` created with an empty `fallback_on`: all exceptions will propagate and all responses '
                'will be accepted. Use `fallback_on=(ModelAPIError,)` for the default behavior.'
            )
        return predicates

    def _add_handler(self, handler: Callable[..., Any]) -> None:
        """Add a handler, auto-detecting its type by inspecting type hints."""
        # A response handler's first parameter is annotated as exactly `ModelResponse` (no `Optional`, no subclass);
        # anything else, including no annotation, is an exception handler.
        if get_first_param_type(handler) is ModelResponse:
            self.response_handlers.append(handler)
        else:
            self.exception_handlers.append(handler)

    async def should_fallback(self, value: Exception | ModelResponse) -> bool:
        """Check if any handler wants to trigger fallback."""
        handlers = self.exception_handlers if isinstance(value, Exception) else self.response_handlers
        for handler in handlers:
            # pyright can't narrow handler's param type from the isinstance check on value
            result = await await_maybe(handler(value))  # type: ignore[arg-type]
            if result:
                return True
        return False


def raise_fallback_exception_group(
    exceptions: list[Exception],
    rejected_responses: list[ModelResponse],
    attempts: list[ModelRequestAttempt],
    *,
    owner: str,
) -> NoReturn:
    """Raise a `FallbackExceptionGroup` combining exceptions and response rejections.

    Args:
        exceptions: Exceptions raised by models.
        rejected_responses: Responses rejected by `fallback_on` handlers.
        attempts: Every attempt that was made, in order.
        owner: The class whose chain was exhausted, named in the group's message.
    """
    all_errors = list(exceptions)
    if rejected_responses:
        all_errors.append(ResponseRejected(len(rejected_responses)))
    group = FallbackExceptionGroup(f'All models from {owner} failed', all_errors)
    group.attempts = attempts
    raise group


@dataclass
class AttemptLog:
    """The attempts at a request that failed before one produced the response, in order.

    Each is a [`ModelRequestAttempt`][pydantic_ai.messages.ModelRequestAttempt], which goes on the response
    that answers or on the `FallbackExceptionGroup` that ends the request, and an ERROR child span of the
    request's `chat` span when the request is instrumented.
    """

    attempts: list[ModelRequestAttempt] = field(default_factory=list[ModelRequestAttempt])

    def record(
        self,
        model: Model,
        failure: Exception | ModelResponse,
        start: AttemptStart,
        *,
        parent: Span | None,
        tracer: Tracer | None,
        rejected_status: str = 'Response rejected',
    ) -> None:
        """Record that the attempt at `model` started at `start` failed with `failure`, an error or a rejected response.

        `parent` is the request's `chat` span and `tracer` the one it was opened with; without both, only the
        record is kept. `rejected_status` is the span's status for a rejected response (see `record_attempt_span`).
        """
        # A nested `FallbackModel` recorded the attempts it made itself, and their usage was billed too.
        if isinstance(failure, ModelResponse):
            self.attempts.extend(failure.failed_attempts or [])
        elif isinstance(failure, FallbackExceptionGroup):
            self.attempts.extend(failure.attempts)
        attempt = failed_attempt(model, failure, start=start, duration=start.elapsed())
        self.attempts.append(attempt)
        if parent is not None and tracer is not None:
            record_attempt_span(
                attempt,
                failure,
                model=model,
                index=len(self.attempts) - 1,
                parent=parent,
                tracer=tracer,
                rejected_status=rejected_status,
            )

    def attach(self, response: ModelResponse) -> ModelResponse:
        """`response`, carrying the attempts that failed before it."""
        if not self.attempts:
            return response
        return replace(response, failed_attempts=[*self.attempts, *(response.failed_attempts or [])])

    def attach_to_stream(self, stream: StreamedResponse) -> None:
        """Have the response `stream` produces carry the attempts that failed before it was opened."""
        if self.attempts:
            stream.failed_attempts = [*self.attempts, *(stream.failed_attempts or [])]


@dataclass
class FallbackChain:
    """One request's way down a fallback chain: what `fallback_on` moved on from, and the attempts that failed.

    `FallbackModel` keeps one per request, and the `ModelFallback` capability one per request step. The capability
    leaves `log` to the agent's attempt loop, which records every attempt a hook moves on from.
    """

    predicates: FallbackPredicates
    owner: str
    """The class whose chain this is, named in the group raised when it is exhausted."""
    exceptions: list[Exception] = field(default_factory=list[Exception])
    rejected_responses: list[ModelResponse] = field(default_factory=list[ModelResponse])
    log: AttemptLog = field(default_factory=AttemptLog)

    async def falls_back(self, failure: Exception | ModelResponse) -> bool:
        """Whether `fallback_on` moves on from `failure`, an error or a response, noting it for the group if so."""
        if not await self.predicates.should_fallback(failure):
            return False
        if isinstance(failure, Exception):
            self.exceptions.append(failure)
        else:
            self.rejected_responses.append(failure)
        return True

    def exhausted(self) -> NoReturn:
        """Raise the `FallbackExceptionGroup` for a chain with no model left to attempt."""
        raise_fallback_exception_group(self.exceptions, self.rejected_responses, self.log.attempts, owner=self.owner)

    def warn_if_streamed_responses_unjudged(self) -> None:
        """Warn that a streamed request accepts its response without consulting the response handlers."""
        if self.predicates.response_handlers:
            warn_response_handlers_not_applied(self.owner)


def warn_response_handlers_not_applied(owner: str) -> None:
    """Warn that `owner`'s response handlers can't judge a streamed response, which is accepted as is."""
    warnings.warn(
        f'`{owner}` response handlers in `fallback_on` are not applied to streamed requests, '
        'so the streamed response is accepted as is. A request is streamed whenever something consumes '
        "the run's events, such as `run_stream()`, `run_stream_events()`, `iter()` with `node.stream()`, "
        'an `event_stream_handler`, or an event listener or hook.',
        UserWarning,
        stacklevel=2,
    )


@asynccontextmanager
async def request_stream_until_first_event(
    model: Model,
    deadline: RequestDeadline | None,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    run_context: RunContext[Any] | None,
) -> AsyncGenerator[StreamedResponse]:
    """Open `model`'s stream and wait for its first event before handing it over.

    Providers can accept a streamed request and then report a failure before producing any output,
    like an OpenAI Responses `response.failed` after `response.created`, or a Bedrock throttling
    error as the first stream event. Waiting for the first event means such an error is raised here,
    while the request can still fall back to another model. The event is replayed to the consumer,
    so the stream it gets is unchanged. A stream that ends without any event is handed over as is. The stream
    is opened under `deadline`, if any, so waiting for the first event counts towards `request_timeout` like every
    event after it, and is what `first_event_timeout` bounds.

    Shared by `FallbackModel` for each candidate and by the agent's primed stream, which the `ModelFallback`
    capability relies on.
    """
    async with stream_under_deadline(
        deadline, model.request_stream(messages, model_settings, model_request_parameters, run_context)
    ) as response:
        events: PeekableAsyncStream[ModelResponseStreamEvent, AsyncIterator[ModelResponseStreamEvent]] = (
            PeekableAsyncStream(aiter(response))
        )
        await events.peek()
        # `StreamedResponse.__aiter__` hands out the iterator stored here, so the consumer gets the peeked event too.
        response._event_iterator = events  # pyright: ignore[reportPrivateUsage]
        yield response

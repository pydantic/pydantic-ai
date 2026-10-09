"""Shared machinery for falling back between models.

Used by both entry points: the [`Fallback`][pydantic_ai.capabilities.Fallback] capability, which
drives the agent's own attempt loop, and
[`FallbackModel`][pydantic_ai.models.fallback.FallbackModel], which loops internally so it also
works outside an agent (`direct.model_request`, a standalone `Model`). Keeping the predicates and
the error aggregation here means the two can never disagree about what counts as a failure.
"""

from __future__ import annotations as _annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard, assert_never

from ._utils import await_maybe, get_first_param_type, is_str_dict
from .exceptions import FallbackExceptionGroup, UserError
from .messages import ModelRequestAttempt, ModelResponse

if TYPE_CHECKING:
    from .models import StreamedResponse

__all__ = (
    'ExceptionHandler',
    'ResponseHandler',
    'FallbackOn',
    'FallbackPredicates',
    'ResponseRejected',
    'raise_fallback_exception_group',
    'stamp_continuation_pin',
    'continuation_pin',
    'FALLBACK_MODEL_PIN_KEY',
    'FALLBACK_CAPABILITY_PIN_KEY',
)

_PYDANTIC_AI_METADATA_KEY = '__pydantic_ai__'
FALLBACK_MODEL_PIN_KEY = 'fallback_model_id'
"""Where `FallbackModel` pins a suspended response to the inner model that started it."""
FALLBACK_CAPABILITY_PIN_KEY = 'fallback_candidate'
"""Where the `Fallback` capability pins a suspended response to the candidate that served it.

Separate from `FALLBACK_MODEL_PIN_KEY` so a `FallbackModel` candidate keeps its own inner pin.
"""


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
"""The type of the `fallback_on` parameter to [`Fallback`][pydantic_ai.capabilities.Fallback]
and [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]."""


class ResponseRejected(Exception):
    """Raised within a `FallbackExceptionGroup` when model responses are rejected by a response handler."""

    def __init__(self, rejected_count: int):
        super().__init__(f'{rejected_count} model response(s) rejected by fallback_on handler')


def _is_response_handler(handler: Callable[..., Any]) -> bool:
    """Check if a callable is a response handler based on type hints.

    Returns True if the first parameter is type-hinted as ModelResponse.
    Returns False otherwise (including if there are no type hints).
    """
    first_param_type = get_first_param_type(handler)
    if first_param_type is None:
        return False
    # Only support exact ModelResponse type (no Optional, no subclasses)
    return first_param_type is ModelResponse


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
        if _is_response_handler(handler):
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

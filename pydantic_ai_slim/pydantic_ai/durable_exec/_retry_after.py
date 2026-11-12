"""The wait a model error asked for before a durable engine attempts the failed request again."""

from __future__ import annotations

from ..exceptions import ModelAPIError

__all__ = ('model_error_retry_after',)


def model_error_retry_after(error: BaseException, *, maximum: float) -> float | None:
    """The seconds to wait before attempting a durable unit that failed with `error` again, if a model error asked for a wait.

    A [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError]'s
    [`retry_after`][pydantic_ai.exceptions.ModelAPIError.retry_after] is the wait the provider asked for, with a
    `Retry-After` header or the like. An engine that retries a failed model request waits this instead of its
    configured backoff, as a durable wait: a Temporal activity's `next_retry_delay`, say, or the delay of an AWS
    Lambda step's retry decision. The model error is looked for in `error` and in the exceptions it was raised from
    or while handling, since an engine or a wrapper can raise its own error around it.

    Args:
        error: The exception the durable unit failed with.
        maximum: The longest wait in seconds, typically the engine's maximum retry interval, so that a provider
            asking for hours doesn't park the run for hours.

    Returns:
        The model error's `retry_after`, up to `maximum`, or `None` when no model error in the chain asked for a
            wait, so the engine's configured backoff applies.
    """
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ModelAPIError) and current.retry_after is not None:
            return min(current.retry_after, maximum)
        current = current.__cause__ or current.__context__
    return None

"""What a model stream in a durable unit produced before an error or cancellation ended it there."""

from __future__ import annotations

from ..messages import ModelResponse

_STREAMED_BEFORE_FAILING = '_pydantic_ai_streamed_before_failing'


def mark_streamed_before_failing(error: BaseException, response: ModelResponse) -> None:
    """Mark `error` as having ended a model stream in a durable unit after it produced output, with that output.

    The unit consumes the stream, delivering its events to any in-boundary event handler, before the run sees any
    of it. A failure partway through therefore reaches the run before the stream it opened, where it would read as
    a failure to open one: something to retry or fall back from, with no response to count the usage of. The mark
    lets the run treat it as the failure of a stream that had started, as it does outside durable execution.

    A cancellation the run's stream receives while the unit is still running is marked with what the unit returned
    for it, so the run records that output, as it records the output of a stream cancelled outside durable execution.
    """
    setattr(error, _STREAMED_BEFORE_FAILING, response)


def streamed_before_failing(error: BaseException) -> ModelResponse | None:
    """What a durable unit's model stream produced before `error` ended it, if `mark_streamed_before_failing` marked it."""
    response = getattr(error, _STREAMED_BEFORE_FAILING, None)
    return response if isinstance(response, ModelResponse) else None

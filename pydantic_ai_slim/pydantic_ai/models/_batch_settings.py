"""The batch arguments and resubmission policy shared by `BatchModelRequests` and `model_request_batch`."""

from __future__ import annotations

import math
from datetime import timedelta
from typing import TYPE_CHECKING

from ..exceptions import (
    ModelAPIError,
    ModelConnectionError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelServerError,
    ModelUnavailableError,
    UserError,
)

if TYPE_CHECKING:
    from . import Model
    from .batch import BatchResult

DEFAULT_MAX_REQUESTS = 1_000
DEFAULT_MAX_RESUBMISSIONS = 2
DEFAULT_POLL_INTERVAL = timedelta(seconds=30)
DEFAULT_MAX_POLL_INTERVAL = timedelta(minutes=10)
MAX_FAILED_CHECKS = 3
"""How many checks of a batch in a row may fail in a way a live request is retried on, each tried again at the next poll."""

_TRANSIENT_STATUS_CODES = frozenset({408, 409, 429})
"""The 4xx statuses a provider SDK retries a live request on."""


def check_batch_intervals(
    *, max_requests: int | None, max_resubmissions: int, poll_interval: timedelta, max_poll_interval: timedelta
) -> None:
    """Check the arguments that don't depend on the model."""
    if max_requests is not None and max_requests <= 0:
        raise UserError('`max_requests` must be positive.')
    if max_resubmissions < 0:
        raise UserError('`max_resubmissions` must not be negative.')
    if poll_interval <= timedelta(0) or max_poll_interval < poll_interval:
        raise UserError('`poll_interval` must be positive and no longer than `max_poll_interval`.')


def poll_delay(polls: int, poll_interval: timedelta, max_poll_interval: timedelta) -> float:
    """Seconds to wait before checking on a batch after `polls` checks.

    The wait starts at `poll_interval` and grows by half with each check, up to `max_poll_interval`.
    """
    interval, max_interval = poll_interval.total_seconds(), max_poll_interval.total_seconds()
    # The interval reaches the maximum after this many checks; capping the exponent keeps it from overflowing.
    saturated = math.ceil(math.log(max_interval / interval, 1.5))
    return min(interval * 1.5 ** min(polls, saturated), max_interval)


def batch_size(model: Model, max_requests: int | None) -> int:
    """The most requests per batch for `model`: `max_requests` capped by, or else, the provider limit."""
    if (provider_limit := model.batch_max_requests) is None:
        raise UserError(model._no_batch_api_message())  # pyright: ignore[reportPrivateUsage]
    return provider_limit if max_requests is None else min(max_requests, provider_limit)


def check_batch_settings(
    model: Model,
    *,
    max_requests: int | None,
    max_resubmissions: int,
    poll_interval: timedelta,
    max_poll_interval: timedelta,
) -> int:
    """Check the arguments, and return the most requests per batch: `max_requests` capped by, or else, the provider limit."""
    check_batch_intervals(
        max_requests=max_requests,
        max_resubmissions=max_resubmissions,
        poll_interval=poll_interval,
        max_poll_interval=max_poll_interval,
    )
    return batch_size(model, max_requests)


def transient(error: ModelAPIError) -> bool:
    """Whether a batch request that failed with `error` may succeed if it's submitted again.

    These are the errors a provider SDK retries a live request on: a rate limit, an unavailable or overloaded model,
    a server error, a timeout or a conflict. A request rejected for its content, its size, its credentials or the
    account's quota fails the same way again, and a canceled one was meant to stop.
    """
    if isinstance(error, ModelRateLimitError | ModelUnavailableError | ModelServerError):
        return True
    if isinstance(error, ModelQuotaExceededError | ModelContextWindowExceededError) or not isinstance(
        error, ModelHTTPError
    ):
        return False
    # An error without a category, like a custom model's, by its status.
    return error.status_code in _TRANSIENT_STATUS_CODES or error.status_code >= 500


def check_failure_tolerated(error: ModelAPIError, failed_checks: int) -> bool:
    """Whether a check of a batch that failed with `error`, the `failed_checks`th failure in a row, is tried again.

    A check that failed the way a live request is retried on, like a dropped connection or a server error, says
    nothing about the batch, so it's checked again at the next poll, up to `MAX_FAILED_CHECKS` failures in a row.
    """
    return failed_checks <= MAX_FAILED_CHECKS and (transient(error) or isinstance(error, ModelConnectionError))


def resubmittable(result: BatchResult, error: ModelAPIError) -> bool:
    """Whether a request whose `result` raised `error` may succeed in a new batch: one that expired, or a transient error.

    Not one that was canceled: providers don't cancel requests of their own accord, so someone meant to stop it.
    """
    return result.status == 'expired' or transient(error)


def missing_result_error(model_name: str, batch_id: str) -> ModelAPIError:
    """The error for a request a batch ended without a result for."""
    return ModelAPIError(model_name, f'Batch {batch_id!r} ended without a result for the request.')

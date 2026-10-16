"""Checking the batch arguments shared by `BatchModel` and `model_request_batch`."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from ..exceptions import UserError

if TYPE_CHECKING:
    from . import Model

DEFAULT_POLL_INTERVAL = timedelta(seconds=30)
DEFAULT_MAX_POLL_INTERVAL = timedelta(minutes=10)


def check_batch_settings(
    model: Model, *, max_requests: int | None, poll_interval: timedelta, max_poll_interval: timedelta
) -> int:
    """Check the arguments, and return the most requests per batch: `max_requests` capped by, or else, the provider limit."""
    if (provider_limit := model.batch_max_requests) is None:
        raise UserError(f'`{type(model).__name__}` {model.model_name!r} has no batch API.')
    if max_requests is not None and max_requests <= 0:
        raise UserError('`max_requests` must be positive.')
    if poll_interval <= timedelta(0) or max_poll_interval < poll_interval:
        raise UserError('`poll_interval` must be positive and no longer than `max_poll_interval`.')
    return provider_limit if max_requests is None else min(max_requests, provider_limit)

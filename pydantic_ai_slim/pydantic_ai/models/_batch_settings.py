"""Resolving `BatchSettings`, shared by `BatchModel` and `model_request_batch`."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import timedelta
from typing import TYPE_CHECKING

from ..exceptions import UserError

if TYPE_CHECKING:
    from . import Model
    from .batch import BatchSettings


@dataclass(frozen=True)
class ResolvedBatchSettings:
    max_requests: int
    max_wait: timedelta | None
    poll_interval: timedelta
    max_poll_interval: timedelta


def resolve_batch_settings(
    model: Model, settings: BatchSettings | None, *, max_requests: int | None
) -> ResolvedBatchSettings:
    """Fill in the defaults of `settings` and validate them, defaulting to `max_requests` or else the provider limit."""
    if (provider_limit := model.batch_max_requests) is None:
        raise UserError(f'`{type(model).__name__}` {model.model_name!r} has no batch API.')
    settings = settings or {}
    resolved = ResolvedBatchSettings(
        max_requests=settings.get('max_requests', max_requests or provider_limit),
        max_wait=settings.get('max_wait', timedelta(seconds=30)),
        poll_interval=settings.get('poll_interval', timedelta(seconds=30)),
        max_poll_interval=settings.get('max_poll_interval', timedelta(minutes=10)),
    )
    if resolved.max_requests <= 0:
        raise UserError('`BatchSettings.max_requests` must be positive.')
    if resolved.max_wait is not None and resolved.max_wait <= timedelta(0):
        raise UserError('`BatchSettings.max_wait` must be positive.')
    if resolved.poll_interval <= timedelta(0) or resolved.max_poll_interval < resolved.poll_interval:
        raise UserError(
            '`BatchSettings.poll_interval` must be positive and no longer than `BatchSettings.max_poll_interval`.'
        )
    return replace(resolved, max_requests=min(resolved.max_requests, provider_limit))

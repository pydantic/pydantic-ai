from __future__ import annotations

from datetime import timedelta

from hatchet_sdk import ConcurrencyExpression, DefaultFilter, StickyStrategy
from hatchet_sdk.labels import DesiredWorkerLabel
from hatchet_sdk.rate_limit import RateLimit
from hatchet_sdk.runnables.types import Duration
from pydantic import BaseModel


class TaskConfig(BaseModel):
    name: str
    description: str | None = None
    on_events: list[str] | None = None
    on_crons: list[str] | None = None
    version: str | None = None
    sticky: StickyStrategy | None = None
    default_priority: int = 1
    concurrency: ConcurrencyExpression | list[ConcurrencyExpression] | None = None
    schedule_timeout: Duration = timedelta(minutes=5)
    execution_timeout: Duration = timedelta(seconds=60)
    retries: int = 0
    rate_limits: list[RateLimit] | None = None
    desired_worker_labels: dict[str, DesiredWorkerLabel] | None = None
    backoff_factor: float | None = None
    backoff_max_seconds: int | None = None
    default_filters: list[DefaultFilter] | None = None

"""Deprecated import location for `pydantic_ai_harness.spend_limits`.

Renamed to the snake_case of its capability, `SpendLimits`. Importing from here still works but emits a
`HarnessDeprecationWarning`. Import from `pydantic_ai_harness.spend_limits` instead.
"""

from typing import TYPE_CHECKING

from pydantic_ai_harness import spend_limits as _spend_limits
from pydantic_ai_harness._warn import forward_renamed, warn_module_renamed
from pydantic_ai_harness.spend_limits import (
    SPEND_LIMITS_EVENTS,
    BatchSpendStore,
    Budget,
    BudgetSpec,
    BudgetStatus,
    InMemorySpendStore,
    PostgresSpendStore,
    PriceFunc,
    RedisClient,
    RedisSpendStore,
    SpendBudgetStatus,
    SpendCallback,
    SpendEntry,
    SpendLimitExceeded,
    SpendLimits,
    SpendRecordedEvent,
    SpendSnapshot,
    SpendStore,
    Spent,
    UnpricedModelError,
    UnpricedModelWarning,
    Window,
)

if TYPE_CHECKING:
    from pydantic_ai_harness.spend_limits import SpendCompositionWarning

warn_module_renamed('spend', 'spend_limits')

__all__ = [
    'BatchSpendStore',
    'Budget',
    'BudgetSpec',
    'BudgetStatus',
    'InMemorySpendStore',
    'PostgresSpendStore',
    'PriceFunc',
    'RedisClient',
    'RedisSpendStore',
    'SpendCallback',
    'SpendCompositionWarning',
    'SpendBudgetStatus',
    'SpendEntry',
    'SpendRecordedEvent',
    'SpendLimits',
    'SpendLimitExceeded',
    'SpendSnapshot',
    'SpendStore',
    'Spent',
    'SPEND_LIMITS_EVENTS',
    'UnpricedModelError',
    'UnpricedModelWarning',
    'Window',
]


def __getattr__(name: str) -> object:
    return forward_renamed(_spend_limits, name)

"""Spend tracking and budget enforcement for Pydantic AI agents."""

from pydantic_ai_harness.spend._budget import Budget, BudgetSpec, Window
from pydantic_ai_harness.spend._capability import PriceFunc, SpendCallback, SpendLimits
from pydantic_ai_harness.spend._events import SPEND_LIMITS_EVENTS, SpendBudgetStatus, SpendRecordedEvent
from pydantic_ai_harness.spend._exceptions import (
    SpendCompositionWarning,
    SpendLimitExceeded,
    UnpricedModelError,
    UnpricedModelWarning,
)
from pydantic_ai_harness.spend._redis import RedisClient, RedisSpendStore
from pydantic_ai_harness.spend._snapshot import BudgetStatus, SpendSnapshot, Spent
from pydantic_ai_harness.spend._store import BatchSpendStore, InMemorySpendStore, SpendEntry, SpendStore

__all__ = [
    'BatchSpendStore',
    'Budget',
    'BudgetSpec',
    'BudgetStatus',
    'InMemorySpendStore',
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

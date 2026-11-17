"""Spend tracking and budget enforcement for Pydantic AI agents."""

from pydantic_ai_harness.spend_limits._budget import Budget, BudgetSpec, Window
from pydantic_ai_harness.spend_limits._capability import PriceFunc, SpendLimits
from pydantic_ai_harness.spend_limits._events import SPEND_LIMITS_EVENTS, SpendBudgetStatus, SpendRecordedEvent
from pydantic_ai_harness.spend_limits._exceptions import (
    SpendLimitExceeded,
    UnpricedModelError,
    UnpricedModelWarning,
)
from pydantic_ai_harness.spend_limits._postgres import PostgresSpendStore
from pydantic_ai_harness.spend_limits._redis import RedisClient, RedisSpendStore
from pydantic_ai_harness.spend_limits._snapshot import BudgetStatus, Spent
from pydantic_ai_harness.spend_limits._store import BatchSpendStore, InMemorySpendStore, SpendEntry

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
    'SpendBudgetStatus',
    'SpendEntry',
    'SpendRecordedEvent',
    'SpendLimits',
    'SpendLimitExceeded',
    'Spent',
    'SPEND_LIMITS_EVENTS',
    'UnpricedModelError',
    'UnpricedModelWarning',
    'Window',
]

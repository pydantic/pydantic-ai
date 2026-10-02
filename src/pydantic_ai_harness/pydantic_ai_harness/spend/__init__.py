"""Spend tracking and budget enforcement for Pydantic AI agents."""

import warnings
from typing import TYPE_CHECKING

from pydantic_ai_harness._warn import HarnessDeprecationWarning
from pydantic_ai_harness.spend import _exceptions
from pydantic_ai_harness.spend._budget import Budget, BudgetSpec, Window
from pydantic_ai_harness.spend._capability import PriceFunc, SpendCallback, SpendLimits
from pydantic_ai_harness.spend._events import SPEND_LIMITS_EVENTS, SpendBudgetStatus, SpendRecordedEvent
from pydantic_ai_harness.spend._exceptions import (
    SpendLimitExceeded,
    UnpricedModelError,
    UnpricedModelWarning,
)
from pydantic_ai_harness.spend._redis import RedisClient, RedisSpendStore
from pydantic_ai_harness.spend._snapshot import BudgetStatus, SpendSnapshot, Spent
from pydantic_ai_harness.spend._store import BatchSpendStore, InMemorySpendStore, SpendEntry, SpendStore

if TYPE_CHECKING:
    from pydantic_ai_harness.spend._exceptions import SpendCompositionWarning

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


def __getattr__(name: str) -> object:
    if name == 'SpendCompositionWarning':
        warnings.warn(
            '`pydantic_ai_harness.spend.SpendCompositionWarning` is deprecated and no longer emitted: '
            '`SpendLimits` now counts every billed response whatever order capabilities are listed in. '
            'Remove references to it; this deprecated alias will be removed in a future release.',
            category=HarnessDeprecationWarning,
            stacklevel=2,
        )
        return _exceptions.SpendCompositionWarning
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

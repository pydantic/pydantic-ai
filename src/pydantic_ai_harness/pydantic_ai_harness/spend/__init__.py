"""Deprecated import location for `pydantic_ai_harness.spend_limits`.

Renamed to the snake_case of its capability, `SpendLimits`. Importing from here still works but emits a
`HarnessDeprecationWarning`. Import from `pydantic_ai_harness.spend_limits` instead.
"""

import warnings
from typing import TYPE_CHECKING

from pydantic_ai_harness._warn import HarnessDeprecationWarning, warn_module_renamed
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
    from pydantic_ai_harness.spend_limits._exceptions import SpendCompositionWarning

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
    if name == 'SpendCompositionWarning':
        from pydantic_ai_harness.spend_limits import _exceptions

        warnings.warn(
            '`pydantic_ai_harness.spend.SpendCompositionWarning` is deprecated and no longer emitted: '
            '`SpendLimits` now counts every billed response whatever order capabilities are listed in. '
            'Remove references to it; this deprecated alias will be removed in a future release.',
            category=HarnessDeprecationWarning,
            stacklevel=2,
        )
        return _exceptions.SpendCompositionWarning
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

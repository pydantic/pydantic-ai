"""Events emitted by spend tracking."""

from dataclasses import dataclass
from decimal import Decimal

from pydantic_ai import CapabilityEvent
from pydantic_ai.usage import RequestUsage

SPEND_LIMITS_EVENTS = 'spend_limits'


@dataclass(frozen=True, kw_only=True)
class SpendBudgetStatus:
    """Serializable budget status included in a `SpendRecordedEvent`."""

    name: str
    window: str
    usd_limit: Decimal | None
    token_limit: int | None
    key: str
    spent_usd: Decimal
    spent_tokens: int
    requests: int
    unpriced_requests: int
    remaining_usd: Decimal | None
    remaining_tokens: int | None
    warning: bool
    exhausted: bool


@dataclass(kw_only=True)
class SpendRecordedEvent(CapabilityEvent, namespace=SPEND_LIMITS_EVENTS, name='recorded'):
    """A model response was recorded against the configured spend windows."""

    model: str | None
    usage: RequestUsage
    usd: Decimal
    priced: bool
    budgets: tuple[SpendBudgetStatus, ...]

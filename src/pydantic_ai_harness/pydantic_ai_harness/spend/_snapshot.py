"""What a window has accumulated, and what one response added to it."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pydantic_ai.usage import RequestUsage

    from pydantic_ai_harness.spend._budget import Budget

MONEY_PRECISION = 40
"""Significant digits every arithmetic step on `Spent.usd` is pinned to.

`Decimal` reads its precision from the thread's context, so an application that lowered
`getcontext().prec` for its own arithmetic would round money here. A counter Redis holds
exactly as billionths would come back with the application's significant digits, and the
in-process store would write the rounded number back and drift from it on every response.
Wide enough for the signed 64-bit range of billionths a counter can reach.
"""


@contextmanager
def money_precision() -> Generator[None]:
    """Run `Decimal` arithmetic at `MONEY_PRECISION` rather than at whatever the caller set.

    Wraps the operation rather than replacing it, so one guard covers a sum, a difference
    and a product instead of one helper per operator. Copies the caller's context and moves
    only `prec`, so rounding and traps are left as the application chose them.
    """
    with localcontext() as context:
        context.prec = MONEY_PRECISION
        yield


@dataclass(frozen=True, kw_only=True)
class Spent:
    """Everything one window has accumulated so far."""

    usd: Decimal = Decimal(0)
    """Priced cost. Requests with no resolvable price contribute nothing here."""

    tokens: int = 0
    """Total tokens, counted whether or not the request could be priced."""

    requests: int = 0
    """Model requests recorded against this window."""

    unpriced_requests: int = 0
    """How many of `requests` had no resolvable price, so `usd` understates them."""


@dataclass(frozen=True, kw_only=True)
class BudgetStatus:
    """A budget and how much of it is left."""

    budget: Budget[Any]
    """The budget this describes.

    Unparameterised because this is a reading: the dependency type only types
    `Budget.scope`, and nothing calls a scope through a status.
    """

    key: str
    """The store key it accumulates under. Useful for debugging a scope or window."""

    spent: Spent
    """What the budget's current window has accumulated."""

    remaining_usd: Decimal | None
    """`None` when the budget sets no USD limit."""

    remaining_tokens: int | None
    """`None` when the budget sets no token limit."""

    warning: bool
    """Whether spend has crossed `Budget.warn_at`. Always `False` without one."""

    exhausted: bool
    """Whether a further request would be refused."""


@dataclass(frozen=True, kw_only=True)
class SpendSnapshot:
    """What one model response cost, and where every budget stands after it."""

    model: str | None
    """The model that produced the response, or `None` if it reported none."""

    usage: RequestUsage
    """The response's usage verbatim, including cache reads, writes, and audio."""

    usd: Decimal
    """What this response cost. Zero when it could not be priced."""

    priced: bool
    """Whether `usd` is a real price or a stand-in zero."""

    budgets: tuple[BudgetStatus, ...]
    """One entry per configured budget, in the order they were declared."""

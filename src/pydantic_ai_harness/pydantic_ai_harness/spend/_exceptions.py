"""Exceptions raised by the spend capability."""

from __future__ import annotations

from pydantic_ai.exceptions import UsageLimitExceeded, UserError


class SpendLimitExceeded(UsageLimitExceeded):
    """Raised when a [`Budget`][pydantic_ai_harness.spend.Budget] is exhausted.

    Subclasses [`UsageLimitExceeded`][pydantic_ai.exceptions.UsageLimitExceeded]
    so an application that already stops on a usage limit stops on a spend limit
    too, while code that needs to tell "the daily budget is gone" from "this run
    used too many tokens" can catch this type specifically.
    """

    _HINT = (
        'Raise the budget, widen its window, or wait for the window to roll over. '
        'See https://pydantic.dev/docs/ai/harness/spend/'
    )


class UnpricedModelWarning(UserWarning):
    """Warned once per model when an incomplete or unpriced response leaves a USD ceiling understated.

    Only warned under `on_unpriced='zero'`, and only while a `Budget` carries a
    `usd` ceiling. A response with incomplete usage may add a known cost subtotal,
    but the omitted usage leaves its full cost unknown. A response with no registry
    price adds nothing in dollars. Known token totals for incomplete usage are lower
    bounds.

    Deduplicated per model name for the life of the capability instance, so a
    model with an incomplete or unavailable price reports once rather than once per request.
    """


class SpendCompositionWarning(UserWarning):
    """Deprecated: no longer emitted, since `SpendLimits` counts every billed response whatever the capability order."""


class UnpricedModelError(UserError):
    """Raised when `on_unpriced='raise'` and a complete price is unavailable for a response.

    Either the model is absent from the `genai-prices` registry (a local or
    custom deployment), the response carries no model name, or its usage is
    incomplete. Incomplete usage can still contribute a known cost subtotal.
    Supply `SpendLimits.price` to price it yourself, or use `on_unpriced='zero'`
    to record any known subtotal and surface the gap as `Spent.unpriced_requests`.
    """

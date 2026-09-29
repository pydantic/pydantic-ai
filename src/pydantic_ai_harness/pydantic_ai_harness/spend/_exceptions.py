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
    """Warned once per model when an unpriced response counts as free against a USD ceiling.

    Only warned under `on_unpriced='zero'`, and only while a `Budget` carries a
    `usd` ceiling. That is the combination where the gap is silent: the response
    contributes nothing in dollars, so that ceiling cannot be reached however
    many such requests are made. A token ceiling still holds, because tokens are
    counted whether or not a price was found.

    Deduplicated per model name for the life of the capability instance, so a
    model the registry does not know reports once rather than once per request.
    """


class SpendCompositionWarning(UserWarning):
    """Warned when wrapper ordering can hide billed responses from `SpendLimits`."""


class UnpricedModelError(UserError):
    """Raised when `on_unpriced='raise'` and no price could be resolved for a response.

    Either the model is absent from the `genai-prices` registry (a local or
    custom deployment) or the response carries no model name. Supply
    `SpendLimits.price` to price it yourself, or use `on_unpriced='zero'` to
    count the request as free and surface it as `Spent.unpriced_requests`.
    """

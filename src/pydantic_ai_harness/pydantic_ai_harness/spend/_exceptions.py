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
    """Warned when another capability is composed so that it wraps inside the accrual.

    Pydantic AI orders the `innermost` tier against non-innermost capabilities only.
    Among themselves the one listed later nests further in, so a capability listed
    after `SpendLimits` wraps inside it. Such a capability can await a response and
    then raise, which sends the run to a fresh request while the rejected response --
    generated, billed, and kept in history -- is never counted.

    This reports the ordering, not an under-count that has happened. Reaching one
    needs the nested capability to reject a response it has already awaited, and whether
    it does is its own business: an `InputGuardrail` gets there only when `parallel=True`,
    its guard blocks, and the provider answers before the guard does. Sequentially the
    guard raises before the request is made, and a parallel guard that blocks first
    cancels the call, so neither leaves anything billed to count. None of those conditions
    is read here -- `parallel` can be flipped without moving anything in the list, so the
    ordering is the durable property and the one you control. List `SpendLimits` last
    among the innermost capabilities to remove it.

    Silence it with::

        import warnings
        from pydantic_ai_harness.spend import SpendCompositionWarning

        warnings.filterwarnings('ignore', category=SpendCompositionWarning)
    """


class UnpricedModelError(UserError):
    """Raised when `on_unpriced='raise'` and no price could be resolved for a response.

    Either the model is absent from the `genai-prices` registry (a local or
    custom deployment) or the response carries no model name. Supply
    `SpendLimits.price` to price it yourself, or use `on_unpriced='zero'` to
    count the request as free and surface it as `Spent.unpriced_requests`.
    """

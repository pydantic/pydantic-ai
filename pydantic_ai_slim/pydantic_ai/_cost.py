"""Best-effort response cost calculation with [genai-prices](https://github.com/pydantic/genai-prices)."""

from __future__ import annotations

import warnings
from decimal import Decimal
from typing import TYPE_CHECKING

from ._warnings import CostCalculationFailedWarning

if TYPE_CHECKING:
    from .messages import ModelResponse


def best_effort_cost(response: ModelResponse) -> Decimal:
    """Best-effort cost of a response in USD; a pricing failure never fails the run.

    `genai-prices` raises `LookupError` for providers/models it doesn't know about (including `test` and
    `function` models) and `ValueError` for usage it can't price (e.g. inconsistent cache token counts);
    both are expected and contribute nothing to the total. Any other error is unexpected and surfaced as a
    `CostCalculationFailedWarning` (rather than raised, since cost is best-effort and must not fail the run).
    """
    if not response.model_name:
        # Without a model name (e.g. a synthetic response from a capability) there's nothing to price.
        return Decimal(0)
    try:
        return response.cost().total_price
    except (LookupError, ValueError):
        # NOTE(Marcelo): We can allow some kind of hook on the provider level, which we could retrieve via
        # `ctx.deps.model.provider.calculate_cost`, but I'm not sure how would the API look like. Maybe a new parameter
        # on the `Provider` classes, that parameter would be a callable that receives the same parameters as `genai_prices`.
        return Decimal(0)
    except Exception as e:
        warnings.warn(
            f'Failed to calculate the cost of the response: {type(e).__name__}: {e}', CostCalculationFailedWarning
        )
        return Decimal(0)

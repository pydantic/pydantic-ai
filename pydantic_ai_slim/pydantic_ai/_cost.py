"""Best-effort response cost calculation with [genai-prices](https://github.com/pydantic/genai-prices)."""

from __future__ import annotations

import warnings
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from genai_prices import calc_price

from pydantic_ai.usage import RequestUsage, RunUsage

from ._warnings import CostCalculationFailedWarning

if TYPE_CHECKING:
    from genai_prices.types import PriceCalculation

    from .messages import ModelResponse


def best_effort_price_calculation(response: ModelResponse) -> PriceCalculation | None:
    """Best-effort price calculation for a response; a pricing failure never fails the run.

    `genai-prices` raises `LookupError` for providers/models it doesn't know about (including `test` and
    `function` models) and `ValueError` for usage it can't price (e.g. inconsistent cache token counts);
    both are expected and return `None`. Any other error is unexpected and surfaced as a
    `CostCalculationFailedWarning` (rather than raised, since pricing is best-effort and must not fail the run).
    """
    if not response.model_name:
        # Without a model name (e.g. a synthetic response from a capability) there's nothing to price.
        return None
    try:
        return response.cost()
    except (LookupError, ValueError):
        # NOTE(Marcelo): We can allow some kind of hook on the provider level, which we could retrieve via
        # `ctx.deps.model.provider.calculate_cost`, but I'm not sure how would the API look like. Maybe a new parameter
        # on the `Provider` classes, that parameter would be a callable that receives the same parameters as `genai_prices`.
        return None
    except Exception as e:
        warnings.warn(
            f'Failed to get cost from response: {type(e).__name__}: {e}',
            CostCalculationFailedWarning,
            stacklevel=2,
        )
        return None


def calculate_price_for_usage(
    usage: RequestUsage | RunUsage,
    *,
    model_name: str | None = None,
    provider_api_url: str | None = None,
    provider_name: str | None = None,
    genai_request_timestamp: datetime | None = None,
) -> PriceCalculation:
    assert model_name, 'Model name is required to calculate price'

    if provider_api_url:
        try:
            return calc_price(
                usage,
                model_name,
                provider_api_url=provider_api_url,
                genai_request_timestamp=genai_request_timestamp,
            )
        except LookupError:
            pass

    return calc_price(
        usage,
        model_name,
        provider_id=provider_name,
        genai_request_timestamp=genai_request_timestamp,
    )


def best_effort_cost(response: ModelResponse) -> Decimal:
    """Best-effort cost of a response in USD; a pricing failure never fails the run."""
    price_calculation = best_effort_price_calculation(response)
    if price_calculation is None:
        return Decimal(0)
    return price_calculation.total_price

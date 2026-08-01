"""Best-effort response cost calculation with [genai-prices](https://github.com/pydantic/genai-prices)."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from genai_prices import calc_price

from ._warnings import CostCalculationFailedWarning

if TYPE_CHECKING:
    from genai_prices.types import PriceCalculation

    from .messages import ModelResponse
    from .usage import RequestUsage, RunUsage


def calculate_price_for_usage(
    usage: RequestUsage | RunUsage,
    *,
    model_name: str,
    provider_api_url: str | None = None,
    provider_name: str | None = None,
    genai_request_timestamp: datetime | None = None,
) -> PriceCalculation:
    """Calculate the price of a usage object with [genai-prices](https://github.com/pydantic/genai-prices).

    Tries matching on `provider_api_url` first as it's more specific, then falls back to `provider_name`.
    `genai-prices` errors propagate to the caller; the `best_effort_*` helpers degrade them to `None` instead.
    """
    if provider_api_url:
        try:
            return calc_price(
                usage,
                model_name,
                provider_api_url=provider_api_url,
                genai_request_timestamp=genai_request_timestamp,
            )
        except LookupError:
            # genai-prices doesn't know this URL, but the provider name may still resolve.
            pass

    return calc_price(
        usage,
        model_name,
        provider_id=provider_name,
        genai_request_timestamp=genai_request_timestamp,
    )


def _best_effort_price(compute: Callable[[], PriceCalculation], *, source: str) -> PriceCalculation | None:
    """Run `compute`, degrading a pricing failure to `None` so it never fails the run.

    `genai-prices` raises `LookupError` for providers/models it doesn't know about (including `test` and
    `function` models) and `ValueError` for usage it can't price (e.g. cache token counts that imply a
    negative uncached remainder); both are expected and return `None`. Any other error is unexpected and
    surfaced as a `CostCalculationFailedWarning` (rather than raised, since pricing must not fail the run).
    """
    try:
        return compute()
    except (LookupError, ValueError):
        # NOTE(Marcelo): We can allow some kind of hook on the provider level, which we could retrieve via
        # `ctx.deps.model.provider.calculate_cost`, but I'm not sure how would the API look like. Maybe a new parameter
        # on the `Provider` classes, that parameter would be a callable that receives the same parameters as `genai_prices`.
        return None
    except Exception as e:
        warnings.warn(
            f'Failed to get cost from {source}: {type(e).__name__}: {e}',
            CostCalculationFailedWarning,
            stacklevel=3,
        )
        return None


def best_effort_price_calculation(response: ModelResponse) -> PriceCalculation | None:
    """Best-effort price calculation for a response; a pricing failure never fails the run."""
    if not response.model_name:
        # Without a model name (e.g. a synthetic response from a capability) there's nothing to price.
        return None
    return _best_effort_price(response.cost, source='response')


def fill_response_cost(response: ModelResponse) -> None:
    """Fill `response.usage.cost` with a best-effort price if it's still unset.

    An already-set cost is never overwritten, so a provider-reported cost could take precedence in future; no model
    sets one today. If pricing data is unavailable the cost stays `None`, distinguishing "unknown" from a genuine
    zero cost.
    """
    if response.usage.cost is None and (price := best_effort_price_calculation(response)) is not None:
        response.usage.cost = price.total_price


def best_effort_usage_cost(
    usage: RequestUsage | RunUsage,
    *,
    model_name: str,
    provider_api_url: str | None = None,
    provider_name: str | None = None,
) -> Decimal | None:
    """Best-effort cost of a bare usage object (e.g. from `count_tokens`) in USD; a pricing failure never fails the run."""
    price = _best_effort_price(
        lambda: calculate_price_for_usage(
            usage,
            model_name=model_name,
            provider_api_url=provider_api_url,
            provider_name=provider_name,
        ),
        source='usage',
    )
    return price.total_price if price is not None else None

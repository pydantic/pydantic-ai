"""Best-effort response cost calculation with [genai-prices](https://github.com/pydantic/genai-prices)."""

from __future__ import annotations

import warnings
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

from genai_prices import calc_price

from ._warnings import CostCalculationFailedWarning

if TYPE_CHECKING:
    from genai_prices.types import PriceCalculation

    from .messages import ModelResponse
    from .usage import RequestUsage, RunUsage


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


def cost_from_provider_details(response: ModelResponse) -> Decimal | None:
    """Best-effort cost the provider reported for a response, or `None` to fall back to `genai-prices`.

    Two shapes are recognised, both tolerated as untyped `provider_details` so malformed values fall
    through to `None` rather than raising:

    - OpenRouter reports its own cost at the top level as `provider_details['cost']`.
    - The Pydantic AI Gateway reports a `cost_estimate` nested under `provider_details['usage']['pydantic_ai_gateway']`.

    A top-level OpenRouter `cost` takes precedence over the nested gateway estimate. Floats are converted
    via `str` so binary-float noise doesn't leak into the `Decimal`.
    """
    provider_details = response.provider_details
    if not provider_details:
        return None

    # OpenRouter: a top-level `cost` (a reported `0.0` is a value, not "unknown", so only `None` is skipped).
    if (cost := provider_details.get('cost')) is not None:
        return Decimal(str(cost))

    # Pydantic AI Gateway: a `cost_estimate` nested under `usage.pydantic_ai_gateway`.
    usage = provider_details.get('usage')
    if isinstance(usage, dict):
        gateway = cast('dict[str, Any]', usage).get('pydantic_ai_gateway')
        if isinstance(gateway, dict):
            cost_estimate = cast('dict[str, Any]', gateway).get('cost_estimate')
            if cost_estimate is not None:
                return Decimal(str(cost_estimate))

    return None


def calculate_price_for_usage(
    usage: RequestUsage | RunUsage,
    *,
    model_name: str | None = None,
    provider_api_url: str | None = None,
    provider_name: str | None = None,
    genai_request_timestamp: datetime | None = None,
) -> PriceCalculation:
    """Calculate the price of a usage object with [genai-prices](https://github.com/pydantic/genai-prices).

    Tries matching on `provider_api_url` first as it's more specific, then falls back to `provider_name`.
    Unlike `best_effort_price_calculation`, this propagates `genai-prices` lookup errors to the caller.
    """
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


def best_effort_usage_cost(
    usage: RequestUsage | RunUsage,
    *,
    model_name: str,
    provider_api_url: str | None = None,
    provider_name: str | None = None,
) -> Decimal:
    """Best-effort cost of a bare usage object (e.g. from `count_tokens`) in USD; a pricing failure never fails the run.

    Mirrors the error handling of `best_effort_price_calculation`: `LookupError`/`ValueError` from `genai-prices`
    (unknown provider/model, unpriceable usage) return `Decimal(0)`, and any unexpected error is surfaced as a
    `CostCalculationFailedWarning` rather than raised.
    """
    try:
        return calculate_price_for_usage(
            usage,
            model_name=model_name,
            provider_api_url=provider_api_url,
            provider_name=provider_name,
        ).total_price
    except (LookupError, ValueError):
        return Decimal(0)
    except Exception as e:
        warnings.warn(
            f'Failed to get cost from usage: {type(e).__name__}: {e}',
            CostCalculationFailedWarning,
            stacklevel=2,
        )
        return Decimal(0)


def best_effort_cost(response: ModelResponse) -> Decimal:
    """Best-effort cost of a response in USD; a pricing failure never fails the run."""
    if (provider_cost := cost_from_provider_details(response)) is not None:
        return provider_cost
    price_calculation = best_effort_price_calculation(response)
    if price_calculation is None:
        return Decimal(0)
    return price_calculation.total_price

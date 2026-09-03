"""Shared [genai-prices](https://github.com/pydantic/genai-prices) helpers: best-effort cost calculation, the provider lookup order used for usage extraction, and the context window lookup."""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING

from genai_prices import calc_price
from genai_prices.data_snapshot import get_snapshot

from ._warnings import CostCalculationFailedWarning
from .exceptions import UserError

if TYPE_CHECKING:
    from genai_prices.types import PriceCalculation

    from .messages import ModelResponse
    from .models._abstract import AbstractModel
    from .usage import RequestUsage, RunUsage


def preload_pricing_data() -> None:
    """Load genai-prices' deferred data snapshot at `Model` construction, keeping the one-time cost off the event loop.

    See https://github.com/pydantic/pydantic-ai/issues/7405.
    """
    get_snapshot()


def iter_provider_references(
    *,
    provider_api_url: str | None = None,
    provider_id: str | None = None,
    provider_fallback: str | None = None,
) -> Iterator[tuple[str | None, str | None]]:
    """Yield `(provider_id, provider_api_url)` genai-prices lookup references, most specific first.

    The API URL identifies a provider more precisely than a name (e.g. several providers reselling the
    same models), so it's tried first, then the provider ID, then the fallback ID; references with
    nothing to match on are skipped. Shared by `RequestUsage.extract` and `lookup_context_window` so
    the lookup order is defined once.
    """
    for candidate_id, candidate_url in ((None, provider_api_url), (provider_id, None), (provider_fallback, None)):
        if candidate_id or candidate_url:
            yield candidate_id, candidate_url


def lookup_context_window(
    model: AbstractModel | str,
    *,
    provider_api_url: str | None = None,
    provider_name: str | None = None,
) -> int | None:
    """Look up a model's context window in [genai-prices](https://github.com/pydantic/genai-prices) data.

    Takes a model instance, whose `model_name`, `system`, and `base_url` are matched on, or a model name
    together with the provider name and/or API URL to match on. Returns the context window recorded for
    the model under the first provider reference that knows it, or `None` if none does or no context
    window is recorded.
    """
    if isinstance(model, str):
        model_name = model
    else:
        model_name, provider_name = model.model_name, model.system
        try:
            provider_api_url = model.base_url
        except (AttributeError, UserError):
            # HuggingFace may have no base URL, and Bedrock Mantle resolves its profile inside `__init__`
            # before the client that `base_url` reads exists; either just means no URL to match on.
            provider_api_url = None
    for candidate_id, candidate_url in iter_provider_references(
        provider_api_url=provider_api_url, provider_id=provider_name
    ):
        try:
            _, model_info = get_snapshot().find_provider_model(
                model_name, provider=None, provider_id=candidate_id, provider_api_url=candidate_url
            )
        except LookupError:
            continue
        return model_info.context_window
    return None


def calculate_price_for_usage(
    usage: RequestUsage | RunUsage,
    *,
    model_name: str,
    provider_api_url: str | None = None,
    provider_name: str | None = None,
    genai_request_timestamp: datetime | None = None,
) -> PriceCalculation:
    """Price a usage object with [genai-prices](https://github.com/pydantic/genai-prices), propagating its errors.

    Tries matching on `provider_api_url` first as it's more specific, then falls back to `provider_name`.
    Only `ModelResponse.cost()` wants this behaviour; everything internal goes through `best_effort_price`.
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


def best_effort_price(
    usage: RequestUsage | RunUsage,
    *,
    model_name: str | None,
    provider_api_url: str | None = None,
    provider_name: str | None = None,
    genai_request_timestamp: datetime | None = None,
) -> PriceCalculation | None:
    """Price a usage object, degrading any failure to `None`; pricing must never fail a run.

    A missing model name (e.g. a synthetic response from a capability) leaves nothing to look up.
    `genai-prices` raises `LookupError` for providers/models it doesn't know about (including `test` and
    `function` models) and `ValueError` for usage it can't price (e.g. cache token counts that imply a
    negative uncached remainder); both are expected. Anything else is unexpected and surfaces as a
    `CostCalculationFailedWarning` rather than being raised.
    """
    if not model_name:
        return None
    try:
        return calculate_price_for_usage(
            usage,
            model_name=model_name,
            provider_api_url=provider_api_url,
            provider_name=provider_name,
            genai_request_timestamp=genai_request_timestamp,
        )
    except (LookupError, ValueError):
        # NOTE(Marcelo): We can allow some kind of hook on the provider level, which we could retrieve via
        # `ctx.deps.model.provider.calculate_cost`, but I'm not sure how would the API look like. Maybe a new parameter
        # on the `Provider` classes, that parameter would be a callable that receives the same parameters as `genai_prices`.
        return None
    except Exception as e:
        warnings.warn(f'Failed to get cost: {type(e).__name__}: {e}', CostCalculationFailedWarning, stacklevel=2)
        return None


def fill_response_cost(response: ModelResponse) -> None:
    """Fill `response.usage.cost` with a best-effort price if it's still unset.

    An already-set cost is never overwritten, so a provider-reported cost could take precedence in future; no model
    sets one today. If pricing data is unavailable the cost stays `None`, distinguishing "unknown" from a genuine
    zero cost.
    """
    if (
        response.usage.cost is None
        and (
            price := best_effort_price(
                response.usage,
                model_name=response.model_name,
                provider_api_url=response.provider_url,
                provider_name=response.provider_name,
                genai_request_timestamp=response.timestamp,
            )
        )
        is not None
    ):
        response.usage.cost = price.total_price

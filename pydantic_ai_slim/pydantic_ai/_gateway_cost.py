"""Read the cost the Pydantic AI Gateway reported for a request, from the response body or headers."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, cast


def gateway_reported_cost(usage_data: object) -> Decimal | None:
    """Cost the Pydantic AI Gateway injected into a response's `usage` object, if any.

    The gateway adds e.g. `usage.logfire_ai_gateway = {'cost': '0.0038'}` (a decimal string; a JSON number is accepted
    too); older and self-hosted gateways only send the deprecated `usage.pydantic_ai_gateway.cost_estimate`. A reported `0` (e.g. a gateway response-cache hit) is a real cost,
    not a missing one.
    """
    if not isinstance(usage_data, Mapping):
        return None
    for block_name, field_name in (('logfire_ai_gateway', 'cost'), ('pydantic_ai_gateway', 'cost_estimate')):
        block = cast('Mapping[str, Any]', usage_data).get(block_name)
        if (
            isinstance(block, Mapping)
            and (cost := _parse_cost(cast('Mapping[str, Any]', block).get(field_name))) is not None
        ):
            return cost
    return None


def gateway_header_cost(headers: Mapping[str, str] | None) -> Decimal | None:
    """Cost the Pydantic AI Gateway reported in a response header like `logfire-ai-gateway-cost: 0.0038USD`, if any.

    Used for SDKs that drop unknown keys from the response body but keep the (lowercased) headers: google-genai and
    boto3. Older and self-hosted gateways only send the deprecated `pydantic-ai-gateway-price-estimate` header.
    """
    if not headers:
        return None
    for header in ('logfire-ai-gateway-cost', 'pydantic-ai-gateway-price-estimate'):
        value = headers.get(header)
        if value is not None and value.endswith('USD') and (cost := _parse_cost(value[:-3])) is not None:
            return cost
    return None


def _parse_cost(value: object) -> Decimal | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        cost = Decimal(str(value))
    except InvalidOperation:
        return None
    # NaN would break `cost_limit` comparisons and a negative cost would reduce the run total.
    return cost if cost.is_finite() and cost >= 0 else None

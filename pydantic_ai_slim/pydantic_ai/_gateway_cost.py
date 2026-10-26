"""Read the cost the Pydantic AI Gateway reported for a request, from the response body or headers."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

from ._utils import is_str_dict


def gateway_reported_cost(usage_data: object) -> Decimal | None:
    """Cost the Pydantic AI Gateway injected into a response's `usage` object, if any.

    The gateway adds e.g. `usage.logfire_ai_gateway = {'cost': '0.0038'}` (a decimal string; a JSON number is accepted
    too); older and self-hosted gateways only send the deprecated `usage.pydantic_ai_gateway.cost_estimate`. A
    reported `0` (e.g. a gateway response-cache hit) is a real cost, not a missing one.
    """
    if not is_str_dict(usage_data):
        return None
    for block_name, field_name in (('logfire_ai_gateway', 'cost'), ('pydantic_ai_gateway', 'cost_estimate')):
        block = usage_data.get(block_name)
        if is_str_dict(block) and (cost := parse_cost(block.get(field_name))) is not None:
            return cost
    return None


def gateway_header_cost(headers: Mapping[str, str] | None) -> Decimal | None:
    """Cost the Pydantic AI Gateway reported in a response header like `logfire-ai-gateway-cost: 0.0038USD`, if any.

    Used for SDKs that drop unknown keys from the response body but keep the (lowercased) headers: google-genai and
    boto3. The deprecated `pydantic-ai-gateway-price-estimate` header is ignored because it rounds small positive
    charges to zero.
    """
    if not headers:
        return None
    value = headers.get('logfire-ai-gateway-cost')
    if value is not None and value.endswith('USD'):
        return parse_cost(value[:-3])
    return None


def parse_cost(value: object) -> Decimal | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        cost = Decimal(str(value))
    except InvalidOperation:
        return None
    # NaN would break `cost_limit` comparisons and a negative cost would reduce the run total.
    return cost if cost.is_finite() and cost >= 0 else None

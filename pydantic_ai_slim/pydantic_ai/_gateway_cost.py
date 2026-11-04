"""Read the cost the Pydantic AI Gateway reported for a request, from the response body or headers."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from ._utils import is_str_dict

if TYPE_CHECKING:
    from .messages import ModelResponse


def gateway_reported_cost(response_data: object) -> Decimal | None:
    """Cost the Pydantic AI Gateway injected into a response's `usage` object, if any.

    The gateway adds e.g. `usage.logfire_ai_gateway = {'cost': '0.0038'}` (a decimal string; a JSON number is accepted
    too); older and self-hosted gateways only send the deprecated `usage.pydantic_ai_gateway.cost_estimate`. A
    reported `0` (e.g. a gateway response-cache hit) is a real cost, not a missing one.
    """
    usage_data = response_data.get('usage') if is_str_dict(response_data) else None
    if not is_str_dict(usage_data):
        return None
    for block_name, field_name in (('logfire_ai_gateway', 'cost'), ('pydantic_ai_gateway', 'cost_estimate')):
        block = usage_data.get(block_name)
        if is_str_dict(block) and (cost := parse_cost(block.get(field_name))) is not None:
            return cost
    return None


def fill_gateway_header_cost(response: ModelResponse, headers: Mapping[str, str] | None) -> None:
    """Use the cost in the gateway's response header, for SDKs that drop it from the response body.

    google-genai and boto3 keep the (lowercased) headers. Like the body cost, the header holds the full decimal, so
    it wins over the genai-prices estimate, and a `0` (e.g. a gateway response-cache hit) is a real cost.
    """
    if (cost := _gateway_header_cost(headers)) is not None:
        response.usage.cost = cost


def _gateway_header_cost(headers: Mapping[str, str] | None) -> Decimal | None:
    # The deprecated `pydantic-ai-gateway-price-estimate` header isn't read: it's rounded to 4 decimals.
    value = headers.get('logfire-ai-gateway-cost') if headers else None
    if value is None or not value.endswith('USD'):
        return None
    return parse_cost(value[:-3])


def parse_cost(value: object) -> Decimal | None:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        cost = Decimal(str(value))
    except InvalidOperation:
        return None
    # NaN would break `cost_limit` comparisons and a negative cost would reduce the run total.
    return cost if cost.is_finite() and cost >= 0 else None

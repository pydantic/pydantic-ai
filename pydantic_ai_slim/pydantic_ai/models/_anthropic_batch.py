"""The Anthropic Message Batches API used by `AnthropicModel.batch_*` methods.

A batch request carries the same parameters `AnthropicModel` would send on its own, prepared by the
same builder. It gets none of that request's recovery retries (an expired history container, a stale
thinking block), because a batch result can't be retried in place: such an error reaches the run as
the model error it is, and the run's own retry handling decides what happens next.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import cast

from anthropic import AsyncAnthropic, Omit, omit as OMIT
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages import BetaMessageBatchErroredResult, BetaMessageBatchSucceededResult
from pydantic import TypeAdapter
from pydantic_core import to_json
from typing_extensions import TypedDict

from .._genai_prices import best_effort_price
from .._utils import is_str_dict
from ..exceptions import ModelAPIError, ModelHTTPError, UserError
from ..messages import ModelMessage, ModelResponse
from ..settings import ModelSettings
from . import ModelRequestParameters, check_allow_model_requests, get_user_agent
from .anthropic import (
    _BATCH_UNSUPPORTED_CLIENTS,  # pyright: ignore[reportPrivateUsage]
    AnthropicModel,
    AnthropicModelSettings,
    _map_api_errors,  # pyright: ignore[reportPrivateUsage]
)
from .batch import BatchRequest, BatchResult, BatchStatus

_BATCH_PRICE_MULTIPLIER = Decimal('0.5')

_ERROR_STATUS_CODES = {
    'invalid_request_error': 400,
    'authentication_error': 401,
    'billing_error': 402,
    'permission_error': 403,
    'not_found_error': 404,
    'rate_limit_error': 429,
    'api_error': 500,
    'timeout_error': 504,
    'overloaded_error': 529,
}
"""The HTTP status the Messages API answers each error type with, so a batch error raises the same `ModelHTTPError`."""


class _Group(TypedDict):
    endpoint: str
    credential: str
    betas: list[str]


_GROUP_ADAPTER = TypeAdapter(_Group)


async def batch_encode(
    model: AnthropicModel,
    messages: list[ModelMessage],
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
    *,
    custom_id: str,
) -> BatchRequest:
    prepared_settings, model_request_parameters = model.prepare_request(model_settings, model_request_parameters)
    settings = cast(AnthropicModelSettings, prepared_settings or {})
    if 'timeout' in settings:
        raise UserError(
            'Anthropic Message Batches do not support per-request `timeout`; configure the client timeout instead.'
        )
    if settings.get('anthropic_speed') is not None:
        raise UserError('Anthropic Message Batches do not support `anthropic_speed`.')
    prepared = await model._prepare_messages_create(messages, settings, model_request_parameters)  # pyright: ignore[reportPrivateUsage]
    if set(prepared.extra_headers) != {'User-Agent'}:
        # A batch is created with one set of headers, so per-request ones can't be honored.
        raise UserError('Anthropic Message Batches do not support per-request `extra_headers`; set them on the client.')

    params = model._messages_create_params(  # pyright: ignore[reportPrivateUsage]
        messages, settings, prepared, prepared.container, prepared.initial_thinking, prepared.initial_thinking_override
    )
    body = {name: value for name, value in params.items() if name != 'extra_body' and not isinstance(value, Omit)}
    if is_str_dict(extra_body := params['extra_body']):
        body.update(extra_body)

    if 'speed' in body or body.get('stream') is True or body.get('max_tokens') == 0:
        raise UserError('Anthropic Message Batches do not support `speed`, `stream=True`, or `max_tokens=0`.')
    return BatchRequest(
        custom_id=custom_id, group=_batch_group(model, prepared.initial_betas), body=to_json(body).decode()
    )


async def batch_create(model: AnthropicModel, requests: Sequence[BatchRequest]) -> str:
    check_allow_model_requests()
    if not requests:
        raise UserError('A provider batch must contain at least one request.')
    group = _GROUP_ADAPTER.validate_json(requests[0].group)
    if any(request.group != requests[0].group for request in requests):
        raise UserError(
            'Requests in one Anthropic batch must share a `group`; these came from different client configurations or beta flags.'
        )
    _validate_batch_group(model, requests[0].group)
    with _map_api_errors(model.model_name):
        batch = (
            await _batch_client(model)
            .with_options(max_retries=0)
            .beta.messages.batches.create(
                requests=[
                    # The body is this provider's own `encode` output.
                    {
                        'custom_id': request.custom_id,
                        'params': cast(MessageCreateParamsNonStreaming, json.loads(request.body)),
                    }
                    for request in requests
                ],
                betas=group['betas'] or OMIT,
                extra_headers={'User-Agent': get_user_agent()},
            )
        )
    return batch.id


async def batch_status(model: AnthropicModel, batch_id: str, *, group: str | None = None) -> BatchStatus:
    check_allow_model_requests()
    if group is not None:
        _validate_batch_group(model, group)
    with _map_api_errors(model.model_name):
        batch = await _batch_client(model).beta.messages.batches.retrieve(batch_id)
    return batch.processing_status


async def batch_results(model: AnthropicModel, batch_id: str, *, group: str | None = None) -> list[BatchResult] | None:
    if await batch_status(model, batch_id, group=group) != 'ended':
        return None
    with _map_api_errors(model.model_name):
        results: list[BatchResult] = []
        async for item in await _batch_client(model).beta.messages.batches.results(batch_id):
            result = item.result
            body = result.model_dump_json() if result.type in ('succeeded', 'errored') else None
            results.append(BatchResult(custom_id=item.custom_id, status=result.type, body=body))
    return results


async def batch_cancel(model: AnthropicModel, batch_id: str, *, group: str | None = None) -> None:
    check_allow_model_requests()
    if group is not None:
        _validate_batch_group(model, group)
    with _map_api_errors(model.model_name):
        await _batch_client(model).beta.messages.batches.cancel(batch_id)


def batch_decode(
    model: AnthropicModel,
    result: BatchResult,
    model_settings: ModelSettings | None,
    model_request_parameters: ModelRequestParameters,
) -> ModelResponse:
    if result.status == 'succeeded':
        assert result.body is not None
        message = BetaMessageBatchSucceededResult.model_validate_json(result.body).message
        with warnings.catch_warnings():
            # `encode` already warned about any setting the request dropped.
            warnings.simplefilter('ignore')
            prepared_settings, model_request_parameters = model.prepare_request(
                model_settings, model_request_parameters
            )
        response = model._process_response(  # pyright: ignore[reportPrivateUsage]
            message, model_request_parameters, cast(AnthropicModelSettings, prepared_settings or {})
        )
        price = best_effort_price(
            response.usage,
            model_name=response.model_name,
            provider_api_url=response.provider_url,
            provider_name=response.provider_name,
            genai_request_timestamp=response.timestamp,
        )
        # Anthropic bills every batch token, cache reads and writes included, at half the standard price.
        if price is not None:
            response.usage.cost = price.total_price * _BATCH_PRICE_MULTIPLIER
        response.provider_details = {
            **(response.provider_details or {}),
            'batch': {'item_id': result.custom_id},
        }
        return response
    if result.status == 'errored':
        assert result.body is not None
        error = BetaMessageBatchErroredResult.model_validate_json(result.body).error
        raise ModelHTTPError(
            status_code=_ERROR_STATUS_CODES.get(error.error.type, 500),
            model_name=model.model_name,
            body=error.model_dump(mode='json'),
        )
    raise ModelAPIError(model.model_name, f'The batch request was {result.status} before the model ran it.')


def _batch_group(model: AnthropicModel, betas: Iterable[str]) -> str:
    group: _Group = {
        'endpoint': hashlib.sha256(str(_batch_client(model).base_url).encode()).hexdigest(),
        'credential': hashlib.sha256(
            to_json(
                {
                    'headers': {
                        name: value
                        for name, value in _batch_client(model).default_headers.items()
                        if isinstance(value, str)
                        and name.lower() != 'user-agent'
                        and not name.lower().startswith('x-stainless-')
                    },
                    'query': _batch_client(model).default_query,
                }
            )
        ).hexdigest(),
        'betas': sorted(betas),
    }
    return json.dumps(group, sort_keys=True)


def _validate_batch_group(model: AnthropicModel, group: str) -> None:
    expected = _GROUP_ADAPTER.validate_json(group)
    if group != _batch_group(model, expected['betas']):
        raise UserError(
            'The Anthropic client does not match the batch request group. Use the same client configuration for every request in a batch.'
        )


def _batch_client(model: AnthropicModel) -> AsyncAnthropic:
    client = model.client
    if isinstance(client, _BATCH_UNSUPPORTED_CLIENTS):
        raise UserError('This Anthropic endpoint does not support Message Batches.')
    if not (client.api_key or client.auth_token):
        raise UserError('Anthropic Message Batches require static Anthropic API credentials on the client.')
    return client

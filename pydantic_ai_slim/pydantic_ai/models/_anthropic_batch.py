"""The Anthropic Message Batches API behind `AnthropicModel.batch_provider`.

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
from typing import Any, cast

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
    AnthropicModel,
    AnthropicModelSettings,
    _build_extra_body,  # pyright: ignore[reportPrivateUsage]
    _map_api_errors,  # pyright: ignore[reportPrivateUsage]
    _resolve_anthropic_service_tier,  # pyright: ignore[reportPrivateUsage]
)
from .batch import BatchRequest, BatchResult, BatchStatus

_MAX_REQUESTS = 100_000
"""Anthropic's limit on requests in one Message Batch."""

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


class AnthropicBatchProvider:
    """Send `AnthropicModel` requests through the Anthropic Message Batches API at its batch price."""

    def __init__(self, model: AnthropicModel, client: AsyncAnthropic) -> None:
        if not (client.api_key or client.auth_token):
            raise UserError('`BatchModel` requires static Anthropic API credentials on the registered client.')
        self._model = model
        self._client = client

    @property
    def max_requests(self) -> int:
        return _MAX_REQUESTS

    async def encode(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        *,
        custom_id: str,
    ) -> BatchRequest:
        model = self._model
        prepared_settings, model_request_parameters = model.prepare_request(model_settings, model_request_parameters)
        settings = cast(AnthropicModelSettings, prepared_settings or {})
        if 'timeout' in settings:
            raise UserError(
                'Per-request `timeout` is not supported for `BatchModel`; configure the registered Anthropic client timeout instead.'
            )
        if settings.get('anthropic_speed') is not None:
            raise UserError('Anthropic Message Batches do not support `anthropic_speed`.')
        prepared = await model._prepare_messages_create(messages, settings, model_request_parameters)  # pyright: ignore[reportPrivateUsage]
        if set(prepared.extra_headers) != {'User-Agent'}:
            # A batch is created with one set of headers, so per-request ones can't be honored.
            raise UserError('`extra_headers` are not supported for `BatchModel` requests; set them on the client.')

        params: MessageCreateParamsNonStreaming = {
            'max_tokens': prepared.max_tokens,
            'messages': prepared.anthropic_messages,
            'model': model.model_name,
        }
        body: dict[str, Any] = {**params}
        for name, value in {
            'system': prepared.system_prompt,
            'tools': prepared.tools,
            'tool_choice': prepared.tool_choice,
            'mcp_servers': prepared.mcp_servers,
            'output_config': prepared.output_config,
            'cache_control': prepared.auto_cache_control,
            'thinking': prepared.initial_thinking,
            'context_management': prepared.context_management,
            'container': prepared.container,
        }.items():
            if value and not isinstance(value, Omit):
                body[name] = value
        if (stop_sequences := settings.get('stop_sequences')) is not None:
            body['stop_sequences'] = stop_sequences
        if (metadata := settings.get('anthropic_metadata')) is not None:
            body['metadata'] = metadata
        if not isinstance(service_tier := _resolve_anthropic_service_tier(settings), Omit):
            body['service_tier'] = service_tier
        if not isinstance(diagnostics := model._cache_diagnostics(messages, settings), Omit):  # pyright: ignore[reportPrivateUsage]
            body['diagnostics'] = diagnostics
        if is_str_dict(extra_body := _build_extra_body(settings, prepared.initial_thinking_override)):
            body.update(extra_body)

        if 'speed' in body or body.get('stream') is True or body.get('max_tokens') == 0:
            raise UserError('Anthropic Message Batches do not support `speed`, `stream=True`, or `max_tokens=0`.')
        return BatchRequest(custom_id=custom_id, group=self._group(prepared.initial_betas), body=to_json(body).decode())

    async def create(self, requests: Sequence[BatchRequest]) -> str:
        check_allow_model_requests()
        if not requests:
            raise UserError('A provider batch must contain at least one request.')
        group = _GROUP_ADAPTER.validate_json(requests[0].group)
        if any(request.group != requests[0].group for request in requests):
            raise UserError(
                'The registered Anthropic client does not match the batch request group. Deploy compatible client settings on every worker in the task queue.'
            )
        self._validate_group(requests[0].group)
        with _map_api_errors(self._model.model_name):
            batch = await self._client.with_options(max_retries=0).beta.messages.batches.create(
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
        return batch.id

    async def status(self, batch_id: str, *, group: str | None = None) -> BatchStatus:
        check_allow_model_requests()
        if group is not None:
            self._validate_group(group)
        with _map_api_errors(self._model.model_name):
            batch = await self._client.beta.messages.batches.retrieve(batch_id)
        return batch.processing_status

    async def results(self, batch_id: str, *, group: str | None = None) -> list[BatchResult] | None:
        if await self.status(batch_id, group=group) != 'ended':
            return None
        with _map_api_errors(self._model.model_name):
            results: list[BatchResult] = []
            async for item in await self._client.beta.messages.batches.results(batch_id):
                result = item.result
                body = result.model_dump_json() if result.type in ('succeeded', 'errored') else None
                results.append(BatchResult(custom_id=item.custom_id, status=result.type, body=body))
        return results

    async def cancel(self, batch_id: str, *, group: str | None = None) -> None:
        check_allow_model_requests()
        if group is not None:
            self._validate_group(group)
        with _map_api_errors(self._model.model_name):
            await self._client.beta.messages.batches.cancel(batch_id)

    def decode(
        self,
        result: BatchResult,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        model = self._model
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
                response.usage.cost = price.total_price / 2
            response.provider_details = {
                **(response.provider_details or {}),
                'batch': {'price_multiplier': '0.5'},
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

    def _group(self, betas: Iterable[str]) -> str:
        group: _Group = {
            'endpoint': hashlib.sha256(str(self._client.base_url).encode()).hexdigest(),
            'credential': hashlib.sha256(
                to_json(
                    {
                        'headers': {
                            name: value
                            for name, value in self._client.default_headers.items()
                            if isinstance(value, str)
                            and name.lower() != 'user-agent'
                            and not name.lower().startswith('x-stainless-')
                        },
                        'query': self._client.default_query,
                    }
                )
            ).hexdigest(),
            'betas': sorted(betas),
        }
        return json.dumps(group, sort_keys=True)

    def _validate_group(self, group: str) -> None:
        expected = _GROUP_ADAPTER.validate_json(group)
        if group != self._group(expected['betas']):
            raise UserError(
                'The registered Anthropic client does not match the batch request group. Deploy compatible client settings on every worker in the task queue.'
            )

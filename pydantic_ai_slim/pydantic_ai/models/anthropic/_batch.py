"""The Anthropic Message Batches API used by `AnthropicModel.batch_*` methods.

A batch request carries the same parameters `AnthropicModel` would send on its own, prepared by the
same builder. It gets none of that request's recovery retries (an expired history container, a stale
thinking block), because a batch result can't be retried in place: such an error reaches the run as
the model error it is, and the run's own retry handling decides what happens next.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast, get_args

from anthropic import AsyncAnthropic, Omit, omit as OMIT
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages import (
    BetaMessageBatch,
    BetaMessageBatchSucceededResult,
)
from pydantic import TypeAdapter
from pydantic_core import to_json
from typing_extensions import TypedDict

from ..._utils import is_str_dict
from ...exceptions import UserError
from ...messages import ModelResponse
from .. import check_allow_model_requests, get_user_agent
from .._batch_codec import (
    batch_priced,
    check_batch_bytes,
    credential_hash,
    group_matches,
    prepare_decode,
    refuse_http_client_auth,
    require_ended,
    shared_group,
    unrun_error,
)
from ..batch import Batch, BatchRequest, BatchResult, BatchStatus, EncodedBatchRequest
from ._errors import ERROR_TYPE_STATUS_CODES, api_error

if TYPE_CHECKING:
    from . import AnthropicModel, AnthropicModelSettings

_CUSTOM_ID = re.compile(r'[a-zA-Z0-9_-]{1,64}')
"""The `custom_id`s Message Batches accept."""

_ENVELOPE = b'{"requests":[]}'
"""What the create request's body adds around its requests."""

MAX_BATCH_BYTES = 256_000_000 - len(_ENVELOPE)
"""The most bytes of requests Anthropic accepts in a batch, whose create request is limited to 256 MB."""

_STATUSES: tuple[BatchStatus, ...] = get_args(BatchStatus)


class _Group(TypedDict):
    endpoint: str
    credential: str
    betas: list[str]


_GROUP_ADAPTER = TypeAdapter(_Group)


async def batch_encode(model: AnthropicModel, request: BatchRequest) -> EncodedBatchRequest:
    prepared_settings, model_request_parameters = model.prepare_request(
        request.model_settings, request.model_request_parameters
    )
    settings = cast('AnthropicModelSettings', prepared_settings or {})
    if 'timeout' in settings:
        raise UserError(
            'Anthropic Message Batches do not support per-request `timeout`; configure the client timeout instead.'
        )
    if 'max_retries' in settings:
        raise UserError(
            'Anthropic Message Batches do not support per-request `max_retries`: the provider runs batched requests.'
        )
    if settings.get('anthropic_speed') is not None:
        raise UserError('Anthropic Message Batches do not support `anthropic_speed`.')
    prepared = await model._prepare_messages_create(request.messages, settings, model_request_parameters)  # pyright: ignore[reportPrivateUsage]
    if set(prepared.extra_headers) != {'User-Agent'}:
        # A batch is created with one set of headers, so per-request ones can't be honored.
        raise UserError('Anthropic Message Batches do not support per-request `extra_headers`; set them on the client.')

    params = model._messages_create_params(  # pyright: ignore[reportPrivateUsage]
        request.messages,
        settings,
        prepared,
        prepared.container,
        prepared.initial_thinking,
        prepared.initial_thinking_override,
    )
    body = {name: value for name, value in params.items() if name != 'extra_body' and not isinstance(value, Omit)}
    if is_str_dict(extra_body := params['extra_body']):
        body.update(extra_body)

    if 'speed' in body or body.get('stream') is True or body.get('max_tokens') == 0:
        raise UserError('Anthropic Message Batches do not support `speed`, `stream=True`, or `max_tokens=0`.')
    encoded = EncodedBatchRequest(
        custom_id=request.custom_id, group=_batch_group(model, prepared.initial_betas), body=to_json(body).decode()
    )
    check_batch_bytes([encoded], request_size, MAX_BATCH_BYTES, 'Anthropic')
    return encoded


def request_size(request: EncodedBatchRequest) -> int:
    """The bytes `request` adds to the create request's body, as the SDK serializes it: its item and a comma."""
    # `{"custom_id":...,"params":...},`, where the body is compact JSON as the SDK writes it.
    return len(b'{"custom_id":,"params":},') + len(to_json(request.custom_id)) + len(request.body.encode())


async def batch_create(model: AnthropicModel, requests: Sequence[EncodedBatchRequest]) -> Batch:
    check_allow_model_requests()
    group = shared_group(requests, 'Anthropic', 'client configurations or beta flags')
    _validate_batch_group(model, group)
    check_batch_bytes(requests, request_size, MAX_BATCH_BYTES, 'Anthropic')
    # Checked here rather than when encoding, as `BatchModelRequests` gives a request its ID after encoding it.
    if invalid := [request.custom_id for request in requests if not _CUSTOM_ID.fullmatch(request.custom_id)]:
        raise UserError(
            f'Anthropic Message Batches need `custom_id`s of 1 to 64 letters, digits, `_` or `-`, not {invalid[0]!r}.'
        )
    batch = (
        await _batch_client(model)
        .with_options(max_retries=0)
        .beta.messages.batches.create(
            requests=[
                # The body is this provider's own `batch_encode` output.
                {
                    'custom_id': request.custom_id,
                    'params': cast(MessageCreateParamsNonStreaming, json.loads(request.body)),
                }
                for request in requests
            ],
            betas=_GROUP_ADAPTER.validate_json(group)['betas'] or OMIT,
            extra_headers={'User-Agent': get_user_agent()},
        )
    )
    return _to_batch(model, batch, group)


async def batch_status(model: AnthropicModel, batch: Batch) -> Batch:
    check_allow_model_requests()
    _validate_batch_group(model, batch.group, collecting=True)
    message_batch = await _batch_client(model).beta.messages.batches.retrieve(batch.id)
    return _to_batch(model, message_batch, batch.group)


async def batch_results(model: AnthropicModel, batch: Batch) -> list[BatchResult]:
    check_allow_model_requests()
    require_ended(batch)
    _validate_batch_group(model, batch.group, collecting=True)
    results: list[BatchResult] = []
    async for item in await _batch_client(model).beta.messages.batches.results(batch.id):
        result = item.result
        body = result.model_dump_json() if result.type in ('succeeded', 'errored') else None
        results.append(BatchResult(custom_id=item.custom_id, status=result.type, body=body))
    return results


async def batch_cancel(model: AnthropicModel, batch: Batch) -> Batch:
    check_allow_model_requests()
    _validate_batch_group(model, batch.group, collecting=True)
    message_batch = await _batch_client(model).beta.messages.batches.cancel(batch.id)
    return _to_batch(model, message_batch, batch.group)


def batch_decode(model: AnthropicModel, result: BatchResult, request: BatchRequest) -> ModelResponse:
    if result.status == 'succeeded':
        assert result.body is not None
        message = BetaMessageBatchSucceededResult.model_validate_json(result.body).message
        prepared_settings, model_request_parameters = prepare_decode(model, request)
        response = model._process_response(  # pyright: ignore[reportPrivateUsage]
            message, model_request_parameters, cast('AnthropicModelSettings', prepared_settings or {})
        )
        return batch_priced(response)
    if result.status == 'errored':
        assert result.body is not None
        # The error is the body the Messages API would have answered the request with, read without validation so an
        # error type added later still raises an error.
        body = json.loads(result.body).get('error')
        error = body.get('error') if is_str_dict(body) else None
        error_type, message = (error.get('type'), error.get('message')) if is_str_dict(error) else (None, None)
        raise api_error(
            model.model_name,
            model._provider.model_id_namespace,  # pyright: ignore[reportPrivateUsage]
            status_code=ERROR_TYPE_STATUS_CODES.get(error_type, 500) if isinstance(error_type, str) else 500,
            body=body,
            message=message if isinstance(message, str) else 'The batch request failed.',
            headers=None,
            in_stream=False,
        )
    raise unrun_error(model.model_name, result)


def _to_batch(model: AnthropicModel, batch: BetaMessageBatch, group: str) -> Batch:
    counts = batch.request_counts
    return Batch(
        id=batch.id,
        model_name=model.model_name,
        provider_name=model.system,
        group=group,
        # A status added after this was written is one the batch hasn't ended in.
        status=batch.processing_status if batch.processing_status in _STATUSES else 'in_progress',
        created_at=batch.created_at,
        request_count=counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired,
    )


def _batch_group(model: AnthropicModel, betas: Iterable[str]) -> str:
    group: _Group = {
        'endpoint': hashlib.sha256(str(_batch_client(model).base_url).encode()).hexdigest(),
        # The client's `default_headers` already put its custom headers after its API key's, as it sends them.
        'credential': credential_hash([_batch_client(model).default_headers], _batch_client(model).default_query),
        'betas': sorted(betas),
    }
    return json.dumps(group, sort_keys=True)


def batch_credential(model: AnthropicModel) -> str:
    """The endpoint and credential `model` reaches batches with, as hashes."""
    group = _GROUP_ADAPTER.validate_json(_batch_group(model, ()))
    return f'{group["endpoint"]}:{group["credential"]}'


def _validate_batch_group(model: AnthropicModel, group: str, *, collecting: bool = False) -> None:
    _batch_client(model)
    expected = _GROUP_ADAPTER.validate_json(group)
    if not group_matches(group, _batch_group(model, expected['betas']), collecting=collecting):
        raise UserError(
            'The Anthropic client does not match the batch request group. Use the client configuration that encoded the requests.'
        )


def _batch_client(model: AnthropicModel) -> AsyncAnthropic:
    client = model.client
    if model.batch_max_requests is None or not isinstance(client, AsyncAnthropic):
        raise UserError('This Anthropic endpoint does not support Message Batches.')
    if not (client.api_key or client.auth_token):
        raise UserError('Anthropic Message Batches require static Anthropic API credentials on the client.')
    refuse_http_client_auth(client._client, 'Anthropic')  # pyright: ignore[reportPrivateUsage]
    return client

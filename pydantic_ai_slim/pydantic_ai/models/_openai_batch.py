"""The OpenAI Batch API used by `OpenAIResponsesModel.batch_*` methods.

A batch request carries the same Responses API parameters `OpenAIResponsesModel` would send on its own, prepared by
the same builder. The requests are uploaded as a JSONL file, and the results are read back from the output and error
files the batch writes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Required, cast

from openai import AsyncOpenAI, Omit
from openai.types import Batch as OpenAIBatch, responses
from pydantic import BaseModel, JsonValue, TypeAdapter
from pydantic_core import to_json
from typing_extensions import TypedDict

from .._utils import is_str_dict
from ..exceptions import ModelHTTPError, UserError
from ..messages import ModelResponse
from . import check_allow_model_requests, get_user_agent
from ._batch_codec import (
    batch_priced,
    credential_hash,
    prepare_decode,
    refuse_http_client_auth,
    require_ended,
    shared_group,
    unrun_error,
)
from .batch import Batch, BatchRequest, BatchResult, BatchResultStatus, BatchStatus, EncodedBatchRequest

if TYPE_CHECKING:
    from .openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

_ENDPOINT = '/v1/responses'

_STATUSES: dict[str, BatchStatus] = {
    'validating': 'in_progress',
    'in_progress': 'in_progress',
    'finalizing': 'in_progress',
    'cancelling': 'canceling',
    'completed': 'ended',
    'failed': 'ended',
    'expired': 'ended',
    'cancelled': 'ended',
}

_UNRUN_STATUSES: dict[str, BatchResultStatus] = {'batch_expired': 'expired', 'batch_cancelled': 'canceled'}
"""The error codes of a request the batch ended before running."""


class _Group(TypedDict):
    endpoint: str
    credential: str
    model: str


class _BatchDetails(TypedDict, total=False):
    """What a `Batch` handle records under `provider_details`, so its results can be read without looking it up."""

    status: Required[str]
    input_file_id: Required[str]
    output_file_id: str
    error_file_id: str
    errors: list[JsonValue]


_DETAILS_ADAPTER = TypeAdapter(_BatchDetails)


class _LineResponse(BaseModel):
    status_code: int
    body: JsonValue


class _LineError(BaseModel):
    code: str | None = None
    message: str | None = None


class _OutputLine(BaseModel):
    """A line of a batch's output or error file."""

    custom_id: str
    response: _LineResponse | None = None
    error: _LineError | None = None


class _InputLine(BaseModel):
    custom_id: str


async def batch_encode(model: OpenAIResponsesModel, request: BatchRequest) -> EncodedBatchRequest:
    group = _batch_group(model)
    prepared_settings, model_request_parameters = model.prepare_request(
        request.model_settings, request.model_request_parameters
    )
    settings = cast('OpenAIResponsesModelSettings', prepared_settings or {})
    if 'timeout' in settings:
        raise UserError('The OpenAI Batch API does not support per-request `timeout`; configure the client timeout.')
    if 'max_retries' in settings:
        raise UserError(
            'The OpenAI Batch API does not support per-request `max_retries`: OpenAI runs batched requests.'
        )
    if settings.get('openai_background'):
        raise UserError('The OpenAI Batch API does not support `openai_background`.')
    create_params, settings = await model._prepare_responses_create(  # pyright: ignore[reportPrivateUsage]
        request.messages, settings, model_request_parameters
    )
    if settings.get('extra_headers'):
        # A batch is created with one set of headers, so per-request ones can't be honored.
        raise UserError('The OpenAI Batch API does not support per-request `extra_headers`; set them on the client.')
    # Merged as the SDK merges `extra_body` into the body of a create call.
    body: dict[str, object] = dict(create_params)
    if (extra_body := settings.get('extra_body')) is not None:
        if not is_str_dict(extra_body):
            raise UserError('OpenAI Batch API `extra_body` must be a mapping with string keys.')
        body.update(extra_body)
    body = {key: value for key, value in body.items() if not isinstance(value, Omit)}
    if body.get('stream') or body.get('background'):
        raise UserError('The OpenAI Batch API does not support `stream` or `background`.')
    return EncodedBatchRequest(custom_id=request.custom_id, group=group, body=to_json(body).decode())


async def batch_create(model: OpenAIResponsesModel, requests: Sequence[EncodedBatchRequest]) -> Batch:
    check_allow_model_requests()
    group = shared_group(requests, 'OpenAI', 'client configurations or models')
    _validate_batch_group(model, group)
    client = _batch_client(model).with_options(max_retries=0)
    lines = (
        # The body is this provider's own `batch_encode` output.
        to_json({'custom_id': request.custom_id, 'method': 'POST', 'url': _ENDPOINT, 'body': json.loads(request.body)})
        for request in requests
    )
    headers = {'User-Agent': get_user_agent()}
    file = await client.files.create(file=('batch.jsonl', b'\n'.join(lines)), purpose='batch', extra_headers=headers)
    batch = await client.batches.create(
        input_file_id=file.id, endpoint=_ENDPOINT, completion_window='24h', extra_headers=headers
    )
    return _to_batch(model, batch, group, len(requests))


async def batch_status(model: OpenAIResponsesModel, batch: Batch) -> Batch:
    check_allow_model_requests()
    _validate_batch_group(model, batch.group)
    openai_batch = await _batch_client(model).batches.retrieve(batch.id)
    return _to_batch(model, openai_batch, batch.group, batch.request_count)


async def batch_results(model: OpenAIResponsesModel, batch: Batch) -> list[BatchResult]:
    check_allow_model_requests()
    require_ended(batch)
    _validate_batch_group(model, batch.group)
    client = _batch_client(model)
    # The handle of an ended batch already names its final files, so they aren't looked up again.
    details = _DETAILS_ADAPTER.validate_python(batch.provider_details)
    results: dict[str, BatchResult] = {}
    for file_id in (details.get('output_file_id'), details.get('error_file_id')):
        if file_id:
            for line in (await client.files.content(file_id)).text.splitlines():
                result = _to_result(_OutputLine.model_validate_json(line))
                results[result.custom_id] = result
    if details['status'] != 'completed':
        # The output and error files have no line for a request the batch didn't run, so the input file lists them.
        for line in (await client.files.content(details['input_file_id'])).text.splitlines():
            if (custom_id := _InputLine.model_validate_json(line).custom_id) not in results:
                results[custom_id] = _unrun_result(details, custom_id)
    return list(results.values())


async def batch_cancel(model: OpenAIResponsesModel, batch: Batch) -> Batch:
    check_allow_model_requests()
    _validate_batch_group(model, batch.group)
    openai_batch = await _batch_client(model).batches.cancel(batch.id)
    return _to_batch(model, openai_batch, batch.group, batch.request_count)


def batch_decode(model: OpenAIResponsesModel, result: BatchResult, request: BatchRequest) -> ModelResponse:
    if result.status == 'succeeded':
        assert result.body is not None
        # Built without validation, as the SDK builds the responses it receives.
        response = responses.Response.model_construct(**json.loads(result.body))
        prepared_settings, model_request_parameters = prepare_decode(model, request)
        model_response = model._process_response(  # pyright: ignore[reportPrivateUsage]
            response, cast('OpenAIResponsesModelSettings', prepared_settings or {}), model_request_parameters
        )
        return batch_priced(model_response)
    if result.status == 'errored':
        assert result.body is not None
        error = _LineResponse.model_validate_json(result.body)
        raise ModelHTTPError(status_code=error.status_code, model_name=model.model_name, body=error.body)
    raise unrun_error(model.model_name, result)


def _to_result(line: _OutputLine) -> BatchResult:
    if (response := line.response) is not None:
        if 200 <= response.status_code < 300:
            return BatchResult(custom_id=line.custom_id, status='succeeded', body=to_json(response.body).decode())
        return BatchResult(custom_id=line.custom_id, status='errored', body=response.model_dump_json())
    error = line.error or _LineError()
    if status := _UNRUN_STATUSES.get(error.code or ''):
        return BatchResult(custom_id=line.custom_id, status=status)
    body = _LineResponse(status_code=500, body={'error': error.model_dump(mode='json')})
    return BatchResult(custom_id=line.custom_id, status='errored', body=body.model_dump_json())


def _unrun_result(details: _BatchDetails, custom_id: str) -> BatchResult:
    if details['status'] == 'failed':
        # The input file failed validation, so the batch ran none of its requests.
        body = _LineResponse(status_code=400, body={'errors': details.get('errors', [])})
        return BatchResult(custom_id=custom_id, status='errored', body=body.model_dump_json())
    return BatchResult(custom_id=custom_id, status='expired' if details['status'] == 'expired' else 'canceled')


def _to_batch(model: OpenAIResponsesModel, batch: OpenAIBatch, group: str, request_count: int) -> Batch:
    details: _BatchDetails = {'status': batch.status, 'input_file_id': batch.input_file_id}
    if batch.output_file_id:
        details['output_file_id'] = batch.output_file_id
    if batch.error_file_id:
        details['error_file_id'] = batch.error_file_id
    if batch.errors and batch.errors.data:
        details['errors'] = [error.model_dump(mode='json', exclude_none=True) for error in batch.errors.data]
    return Batch(
        id=batch.id,
        model_name=model.model_name,
        provider_name=model.system,
        group=group,
        status=_STATUSES[batch.status],
        created_at=datetime.fromtimestamp(batch.created_at, tz=UTC),
        # A batch counts its requests once it has validated its input file.
        request_count=(batch.request_counts and batch.request_counts.total) or request_count,
        provider_details=dict(details),
    )


def _batch_group(model: OpenAIResponsesModel) -> str:
    client = _batch_client(model)
    group: _Group = {
        'endpoint': hashlib.sha256(str(client.base_url).encode()).hexdigest(),
        # The client sends its API key apart from its default headers.
        'credential': credential_hash({**client.default_headers, **client.auth_headers}, client.default_query),
        # An OpenAI batch runs one model.
        'model': model.model_name,
    }
    return json.dumps(group, sort_keys=True)


def _validate_batch_group(model: OpenAIResponsesModel, group: str) -> None:
    if group != _batch_group(model):
        raise UserError(
            'The OpenAI client or model does not match the batch request group. '
            'Use the model and client configuration that encoded the requests.'
        )


def _batch_client(model: OpenAIResponsesModel) -> AsyncOpenAI:
    if model.batch_max_requests is None:
        raise UserError(f'The {model.system!r} provider does not support the OpenAI Batch API.')
    client = model.client
    # A batch is grouped, submitted and polled with one credential, which an API key callback could vary per request,
    # and which a workload identity keeps out of the client's headers, so clients of different identities would share
    # a group.
    if getattr(client, '_api_key_provider', None) is not None or getattr(client, 'workload_identity', None) is not None:
        raise UserError(
            'The OpenAI Batch API requires a static API key on the client, not an API key callback or workload identity.'
        )
    refuse_http_client_auth(client._client, 'OpenAI')  # pyright: ignore[reportPrivateUsage]
    return client

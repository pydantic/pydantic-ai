"""The OpenAI Batch API used by `OpenAIResponsesModel.batch_*` methods.

A batch request carries the same Responses API parameters `OpenAIResponsesModel` would send on its own, prepared by
the same builder. The requests are uploaded as a JSONL file, and the results are read back from the output and error
files the batch writes.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Required, cast

from openai import APIError, APIStatusError, AsyncOpenAI, Omit
from openai.types import Batch as OpenAIBatch, responses
from pydantic import BaseModel, JsonValue, TypeAdapter
from pydantic_core import to_json
from typing_extensions import TypedDict

from .._utils import is_str_dict
from ..exceptions import UserError
from ..messages import ModelResponse
from . import check_allow_model_requests, get_user_agent
from ._batch_codec import (
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
from .batch import Batch, BatchRequest, BatchResult, BatchResultStatus, BatchStatus, EncodedBatchRequest
from .openai import (
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
    _status_error,  # pyright: ignore[reportPrivateUsage]
)

_ENDPOINT = '/v1/responses'

MAX_BATCH_BYTES = 200_000_000
"""The most bytes OpenAI accepts in a batch's input file."""

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


@dataclass
class _UploadedInputFile:
    file_id: str
    completed: bool = False
    """Whether OpenAI said the batch completed, rather than a handle claiming it."""


_UPLOADED_INPUT_FILES: OrderedDict[tuple[str, str], _UploadedInputFile] = OrderedDict()
"""The input file this process uploaded for each batch it created, to delete once every request ran.

Keyed by the endpoint and credential that created the batch, from `batch_credential`, and its ID, as batch IDs are only
unique within one account: a client of another account never finds, or deletes, this one's file.

A `Batch` handle can come from a stored message history, which a caller controls, so the file IDs it names are only
read, never deleted, and its status doesn't say when the file can be: only OpenAI's own answer that the batch
completed does. The oldest are forgotten first, which leaves their files for OpenAI to delete.
"""
_UPLOADED_INPUT_FILES_SIZE = 1024


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
    settings = cast('OpenAIResponsesModelSettings', dict(prepared_settings or {}))
    if 'timeout' in settings:
        raise UserError('The OpenAI Batch API does not support per-request `timeout`; configure the client timeout.')
    # The provider runs a batched request itself, with no SDK client retrying it, so a retry count has nothing to apply
    # to and is ignored, like settings a model doesn't support.
    settings.pop('max_retries', None)
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
    encoded = EncodedBatchRequest(custom_id=request.custom_id, group=group, body=to_json(body).decode())
    check_batch_bytes([encoded], request_size, MAX_BATCH_BYTES, 'OpenAI')
    return encoded


def request_size(request: EncodedBatchRequest) -> int:
    """The bytes `request` adds to a batch's input file: its line and the newline that separates it from the next."""
    return len(_input_line(request)) + 1


def _input_line(request: EncodedBatchRequest) -> bytes:
    # The body is this provider's own `batch_encode` output, so it's embedded as is.
    return b''.join(
        (
            b'{"custom_id":',
            to_json(request.custom_id),
            b',"method":"POST","url":"',
            _ENDPOINT.encode(),
            b'","body":',
            request.body.encode(),
            b'}',
        )
    )


async def batch_create(model: OpenAIResponsesModel, requests: Sequence[EncodedBatchRequest]) -> Batch:
    check_allow_model_requests()
    group = shared_group(requests, 'OpenAI', 'client configurations or models')
    _validate_batch_group(model, group)
    check_batch_bytes(requests, request_size, MAX_BATCH_BYTES, 'OpenAI')
    client = _batch_client(model).with_options(max_retries=0)
    lines = (_input_line(request) for request in requests)
    headers = {'User-Agent': get_user_agent()}
    file = await client.files.create(file=('batch.jsonl', b'\n'.join(lines)), purpose='batch', extra_headers=headers)
    try:
        batch = await client.batches.create(
            input_file_id=file.id, endpoint=_ENDPOINT, completion_window='24h', extra_headers=headers
        )
    except APIStatusError as error:
        if error.status_code < 500:
            # OpenAI refused to create the batch, so nothing reads the file, which it would otherwise keep for 30 days.
            # After a server error or no response, the batch may have been created, so the file is left for it.
            with suppress(APIError):
                await client.files.delete(file.id)
        raise
    _UPLOADED_INPUT_FILES[batch_credential(model), batch.id] = _UploadedInputFile(file.id)
    while len(_UPLOADED_INPUT_FILES) > _UPLOADED_INPUT_FILES_SIZE:
        _UPLOADED_INPUT_FILES.popitem(last=False)
    return _to_batch(model, batch, group, len(requests))


async def batch_status(model: OpenAIResponsesModel, batch: Batch) -> Batch:
    check_allow_model_requests()
    _validate_batch_group(model, batch.group, collecting=True)
    openai_batch = await _batch_client(model).batches.retrieve(batch.id)
    if (
        openai_batch.status == 'completed'
        and (uploaded := _UPLOADED_INPUT_FILES.get((batch_credential(model), batch.id))) is not None
    ):
        uploaded.completed = True
    return _to_batch(model, openai_batch, batch.group, batch.request_count)


async def batch_results(model: OpenAIResponsesModel, batch: Batch) -> list[BatchResult]:
    check_allow_model_requests()
    require_ended(batch)
    _validate_batch_group(model, batch.group, collecting=True)
    client = _batch_client(model)
    # The handle of an ended batch already names its final files, so they aren't looked up again.
    details = _DETAILS_ADAPTER.validate_python(batch.provider_details)
    results: dict[str, BatchResult] = {}
    for file_id in (details.get('output_file_id'), details.get('error_file_id')):
        if file_id:
            for line in (await client.files.content(file_id)).text.splitlines():
                result = _to_result(_OutputLine.model_validate_json(line))
                results[result.custom_id] = result
    if details['status'] == 'completed':
        uploaded_key = (batch_credential(model), batch.id)
        if (uploaded := _UPLOADED_INPUT_FILES.get(uploaded_key)) is not None and uploaded.completed:
            del _UPLOADED_INPUT_FILES[uploaded_key]
            # Every request ran, so the input file isn't needed again; OpenAI would otherwise keep it for 30 days.
            with suppress(APIError):
                await client.files.delete(uploaded.file_id)
    else:
        # The output and error files have no line for a request the batch didn't run, so the input file lists them.
        lines = (await client.files.content(details['input_file_id'])).text.splitlines()
        for line_number, line in enumerate(lines, start=1):
            if (custom_id := _InputLine.model_validate_json(line).custom_id) not in results:
                results[custom_id] = _unrun_result(details, custom_id, line_number)
    return list(results.values())


async def batch_cancel(model: OpenAIResponsesModel, batch: Batch) -> Batch:
    check_allow_model_requests()
    _validate_batch_group(model, batch.group, collecting=True)
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
        line = _LineResponse.model_validate_json(result.body)
        # The body of an HTTP error wraps the error object, which the SDK unwraps into the error it raises.
        body: JsonValue = line.body.get('error', line.body) if is_str_dict(line.body) else line.body
        error: JsonValue = body
        if is_str_dict(body) and isinstance(errors := body.get('errors'), list) and errors:
            # The batch failed validation, and the first error stands for the request, as the SDK's would.
            error = cast(list[JsonValue], errors)[0]
        error = error if is_str_dict(error) else {}
        code, error_type, message = error.get('code'), error.get('type'), error.get('message')
        raise _status_error(
            model.model_name,
            model._provider.model_id_namespace,  # pyright: ignore[reportPrivateUsage]
            status_code=line.status_code,
            body=body,
            code=str(code) if code is not None else None,
            error_type=error_type if isinstance(error_type, str) else None,
            message=message if isinstance(message, str) else None,
        )
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


def _unrun_result(details: _BatchDetails, custom_id: str, line_number: int) -> BatchResult:
    if details['status'] == 'failed':
        # The input file failed validation, so the batch ran none of its requests. A request gets the errors of its
        # own line and those of no line; when they're all about other lines, it gets them all, which name them.
        errors = details.get('errors', [])
        own: list[JsonValue] = [
            error for error in errors if not is_str_dict(error) or error.get('line') in (line_number, None)
        ]
        body = _LineResponse(status_code=400, body={'errors': own or errors})
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
        # A status added after this was written is one the batch hasn't ended in, as the known ending ones are listed.
        status=_STATUSES.get(batch.status, 'in_progress'),
        created_at=datetime.fromtimestamp(batch.created_at, tz=UTC),
        # A batch counts its requests once it has validated its input file.
        request_count=(batch.request_counts and batch.request_counts.total) or request_count,
        provider_details=dict(details),
    )


def _batch_group(model: OpenAIResponsesModel) -> str:
    client = _batch_client(model)
    group: _Group = {
        'endpoint': hashlib.sha256(str(client.base_url).encode()).hexdigest(),
        # The client sends its API key's header under its default headers, which can replace it.
        'credential': credential_hash([client.auth_headers, client.default_headers], client.default_query),
        # An OpenAI batch runs one model.
        'model': model.model_name,
    }
    return json.dumps(group, sort_keys=True)


def batch_credential(model: OpenAIResponsesModel) -> str:
    """The endpoint and credential `model` reaches batches with, as hashes."""
    group: _Group = json.loads(_batch_group(model))
    return f'{group["endpoint"]}:{group["credential"]}'


def _validate_batch_group(model: OpenAIResponsesModel, group: str, *, collecting: bool = False) -> None:
    if not group_matches(group, _batch_group(model), collecting=collecting):
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

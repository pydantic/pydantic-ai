"""The Gemini API Batch Mode used by `GoogleModel.batch_*` methods.

A batch request carries the same `generateContent` request `GoogleModel` would send on its own, prepared by the same
builder and converted to the API's JSON by the converters `google-genai` uses for inlined batch requests. Requests
are uploaded as a JSONL file rather than inlined, so a batch can hold as much as the API's 2 GB file limit instead of
its 20 MB inline limit, and results are read from the JSONL file the job writes.

Google Cloud (Vertex AI) runs batch prediction jobs from Cloud Storage or BigQuery instead, so it isn't supported.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NotRequired, cast

from google.genai import Client
from google.genai._common import (
    convert_to_dict,
    encode_unserializable_types,
)
from google.genai.batches import (
    _GenerateContentResponse_from_mldev,  # pyright: ignore[reportPrivateUsage]
    _InlinedRequest_to_mldev,  # pyright: ignore[reportPrivateUsage]
)
from google.genai.types import (
    BatchJob,
    CreateBatchJobConfigDict,
    GenerateContentResponse,
    HttpOptionsDict,
    InlinedRequest,
    JobState,
    UploadFileConfigDict,
)
from pydantic import TypeAdapter
from pydantic_core import to_json
from typing_extensions import TypedDict

from .._utils import is_str_dict, now_utc
from ..exceptions import ModelHTTPError, UserError
from ..messages import ModelResponse
from . import check_allow_model_requests
from ._batch_codec import (
    BatchJobError,
    batch_priced,
    group_matches,
    prepare_decode,
    refuse_http_client_auth,
    require_ended,
    shared_group,
    unrun_error,
)
from .batch import Batch, BatchRequest, BatchResult, BatchStatus, EncodedBatchRequest

if TYPE_CHECKING:
    from .google import GoogleModel, GoogleModelSettings

_STATUSES: dict[JobState, BatchStatus] = {
    JobState.JOB_STATE_UNSPECIFIED: 'in_progress',
    JobState.JOB_STATE_QUEUED: 'in_progress',
    JobState.JOB_STATE_PENDING: 'in_progress',
    JobState.JOB_STATE_RUNNING: 'in_progress',
    JobState.JOB_STATE_PAUSED: 'in_progress',
    JobState.JOB_STATE_UPDATING: 'in_progress',
    JobState.JOB_STATE_CANCELLING: 'canceling',
    JobState.JOB_STATE_SUCCEEDED: 'ended',
    JobState.JOB_STATE_PARTIALLY_SUCCEEDED: 'ended',
    JobState.JOB_STATE_FAILED: 'ended',
    JobState.JOB_STATE_CANCELLED: 'ended',
    JobState.JOB_STATE_EXPIRED: 'ended',
}

# https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
_RPC_STATUS_CODES = {
    1: 499,  # CANCELLED
    2: 500,  # UNKNOWN
    3: 400,  # INVALID_ARGUMENT
    4: 504,  # DEADLINE_EXCEEDED
    5: 404,  # NOT_FOUND
    6: 409,  # ALREADY_EXISTS
    7: 403,  # PERMISSION_DENIED
    8: 429,  # RESOURCE_EXHAUSTED
    9: 400,  # FAILED_PRECONDITION
    10: 409,  # ABORTED
    11: 400,  # OUT_OF_RANGE
    12: 501,  # UNIMPLEMENTED
    13: 500,  # INTERNAL
    14: 503,  # UNAVAILABLE
    15: 500,  # DATA_LOSS
    16: 401,  # UNAUTHENTICATED
}
"""The HTTP status `generateContent` answers each `google.rpc.Code` with, so a batch error raises the same `ModelHTTPError`."""


class _Group(TypedDict):
    endpoint: str
    credential: str
    model: str


class _ResultLine(TypedDict):
    """A line of a job's results file: the response to the request with `key`, or the `google.rpc.Status` it failed with."""

    key: str
    response: NotRequired[dict[str, Any]]
    error: NotRequired[dict[str, Any]]


_RESULT_LINE_ADAPTER = TypeAdapter(_ResultLine)


async def batch_encode(model: GoogleModel, request: BatchRequest) -> EncodedBatchRequest:
    prepared_settings, model_request_parameters = model.prepare_request(
        request.model_settings, request.model_request_parameters
    )
    settings = cast('GoogleModelSettings', prepared_settings or {})
    # A batch is created with one set of HTTP options, and its requests run long after, so neither can be honored.
    if 'timeout' in settings:
        raise UserError('Gemini batch requests do not support per-request `timeout`; configure the client instead.')
    if 'max_retries' in settings:
        raise UserError('Gemini batch requests do not support per-request `max_retries`: Google runs batched requests.')
    if 'extra_headers' in settings:
        raise UserError('Gemini batch requests do not support per-request `extra_headers`; set them on the client.')
    client = _batch_client(model)
    contents, config = await model._build_content_and_config(  # pyright: ignore[reportPrivateUsage]
        request.messages, settings, model_request_parameters
    )
    # The converter leaves out `http_options`, as `batches.create` does for inlined requests.
    converted = _InlinedRequest_to_mldev(
        client._api_client,  # pyright: ignore[reportPrivateUsage]
        # Validated from the dicts `_build_content_and_config` builds for `generate_content`, as the constructor's types
        # only name the SDK's models.
        InlinedRequest.model_validate({'contents': contents, 'config': config}),
    )
    body = encode_unserializable_types(convert_to_dict(converted['request']))
    return EncodedBatchRequest(custom_id=request.custom_id, group=_batch_group(model), body=to_json(body).decode())


async def batch_create(model: GoogleModel, requests: Sequence[EncodedBatchRequest]) -> Batch:
    check_allow_model_requests()
    group = shared_group(requests, 'Gemini', 'client configurations or models')
    _validate_batch_group(model, group)
    client = _batch_client(model)
    lines = (
        # The body is this provider's own `batch_encode` output.
        to_json({'key': request.custom_id, 'request': json.loads(request.body)})
        for request in requests
    )
    file = await client.aio.files.upload(
        file=io.BytesIO(b'\n'.join(lines)),
        config=UploadFileConfigDict(mime_type='application/jsonl', display_name='pydantic-ai-batch'),
    )
    assert file.name is not None
    # Unlike the upload, the job can't be created twice safely: the first attempt may have created it.
    no_retries = HttpOptionsDict(retry_options={'attempts': 1})
    job = await client.aio.batches.create(
        model=model.model_name,
        src=file.name,
        config=CreateBatchJobConfigDict(display_name='pydantic-ai-batch', http_options=no_retries),
    )
    return _to_batch(model, job, group, len(requests), {'input_file': file.name})


async def batch_status(model: GoogleModel, batch: Batch) -> Batch:
    check_allow_model_requests()
    _validate_batch_group(model, batch.group, collecting=True)
    job = await _batch_client(model).aio.batches.get(name=batch.id)
    return _to_batch(model, job, batch.group, batch.request_count, batch.provider_details)


async def batch_results(model: GoogleModel, batch: Batch) -> list[BatchResult]:
    check_allow_model_requests()
    require_ended(batch)
    _validate_batch_group(model, batch.group, collecting=True)
    client = _batch_client(model)
    job = await client.aio.batches.get(name=batch.id)
    if job.dest is None or job.dest.file_name is None:
        # A job that expired, or was canceled or failed before it ran a request, writes no results. The job doesn't
        # say which requests it held, so a failed or canceled one fails all of them with its own reason.
        if job.state == JobState.JOB_STATE_FAILED:
            error = job.error
            code = error.code if error is not None else None
            raise BatchJobError(
                status_code=_RPC_STATUS_CODES.get(code, 500) if code is not None else 500,
                model_name=model.model_name,
                body={'error': error.model_dump(exclude_none=True) if error is not None else None},
            )
        if job.state == JobState.JOB_STATE_CANCELLED:
            raise BatchJobError(
                status_code=_RPC_STATUS_CODES[1],
                model_name=model.model_name,
                body={'error': {'code': 1, 'message': 'The batch job was canceled before the model ran the request.'}},
            )
        return []
    content = await client.aio.files.download(file=job.dest.file_name)
    results: list[BatchResult] = []
    for line in content.splitlines():
        item = _RESULT_LINE_ADAPTER.validate_json(line)
        if 'response' in item:
            results.append(
                BatchResult(custom_id=item['key'], status='succeeded', body=to_json(item['response']).decode())
            )
        else:
            results.append(
                BatchResult(custom_id=item['key'], status='errored', body=to_json(item.get('error')).decode())
            )
    return results


async def batch_cancel(model: GoogleModel, batch: Batch) -> Batch:
    check_allow_model_requests()
    _validate_batch_group(model, batch.group, collecting=True)
    client = _batch_client(model)
    await client.aio.batches.cancel(name=batch.id)
    job = await client.aio.batches.get(name=batch.id)
    return _to_batch(model, job, batch.group, batch.request_count, batch.provider_details)


def batch_decode(model: GoogleModel, result: BatchResult, request: BatchRequest) -> ModelResponse:
    if result.status == 'succeeded':
        assert result.body is not None
        response = GenerateContentResponse._from_response(  # pyright: ignore[reportPrivateUsage]
            response=_GenerateContentResponse_from_mldev(json.loads(result.body)), kwargs={}
        )
        _, model_request_parameters = prepare_decode(model, request)
        model_response = model._process_response(response, model_request_parameters)  # pyright: ignore[reportPrivateUsage]
        return batch_priced(model_response)
    if result.status == 'errored':
        assert result.body is not None
        error = json.loads(result.body)
        code = error.get('code') if is_str_dict(error) else None
        raise ModelHTTPError(
            status_code=_RPC_STATUS_CODES.get(code, 500) if isinstance(code, int) else 500,
            model_name=model.model_name,
            body={'error': error},
        )
    raise unrun_error(model.model_name, result)


def _to_batch(
    model: GoogleModel, job: BatchJob, group: str, request_count: int, provider_details: dict[str, Any] | None
) -> Batch:
    assert job.name is not None
    state = job.state or JobState.JOB_STATE_UNSPECIFIED
    return Batch(
        id=job.name,
        model_name=model.model_name,
        provider_name=model.system,
        group=group,
        status=_STATUSES.get(state, 'in_progress'),
        created_at=job.create_time or now_utc(),
        request_count=request_count,
        provider_details={**(provider_details or {}), 'state': state.value},
    )


_CREDENTIAL_HEADERS = ('x-goog-api-key', 'authorization')
"""The headers a Gemini API request authenticates with; the SDK adds and changes its other headers as it goes."""


def _batch_group(model: GoogleModel) -> str:
    api_client = _batch_client(model)._api_client  # pyright: ignore[reportPrivateUsage]
    # The SDK sends the API key as a header, which `http_options.headers` can override, and which joins the HTTP
    # client's own headers, so the credential is what those headers end up sending rather than the client's `api_key`.
    client_headers = api_client._async_httpx_client.headers  # pyright: ignore[reportPrivateUsage]
    request_headers = (api_client._http_options.headers or {}).items()  # pyright: ignore[reportPrivateUsage]
    credential: list[list[str]] = []
    for name in _CREDENTIAL_HEADERS:
        # A request's headers replace the client's of the same name, and differently cased ones are each sent.
        sent = sorted(value for key, value in request_headers if key.lower() == name)
        credential.append(sent or client_headers.get_list(name))
    group: _Group = {
        'endpoint': hashlib.sha256(model.base_url.encode()).hexdigest(),
        'credential': hashlib.sha256(json.dumps(credential).encode()).hexdigest(),
        'model': model.model_name,
    }
    return json.dumps(group, sort_keys=True)


def _validate_batch_group(model: GoogleModel, group: str, *, collecting: bool = False) -> None:
    if not group_matches(group, _batch_group(model), collecting=collecting):
        raise UserError(
            'The Gemini client or model does not match the batch request group. Use the configuration that encoded the requests.'
        )


def _batch_client(model: GoogleModel) -> Client:
    if model.batch_max_requests is None:
        raise UserError('Gemini batch requests are only supported by the Gemini API, not Google Cloud (Vertex AI).')
    refuse_http_client_auth(model.client._api_client._async_httpx_client, 'Gemini')  # pyright: ignore[reportPrivateUsage]
    return model.client

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
import sys
from collections import OrderedDict
from collections.abc import Sequence
from contextlib import suppress
from typing import Any, NotRequired, cast

import httpx2
from google.genai import Client, errors
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
from ..exceptions import UserError
from ..messages import ModelResponse
from . import check_allow_model_requests
from ._batch_codec import (
    BatchJobError,
    batch_priced,
    check_batch_bytes,
    group_matches,
    prepare_decode,
    refuse_http_client_auth,
    require_ended,
    shared_group,
    unrun_error,
)
from .batch import Batch, BatchRequest, BatchResult, BatchResultStatus, BatchStatus, EncodedBatchRequest
from .google import (
    GoogleModel,
    GoogleModelSettings,
    _http_error,  # pyright: ignore[reportPrivateUsage]
)

MAX_BATCH_BYTES = 2_000_000_000
"""The most bytes the Gemini API accepts in a batch's input file."""

MAX_BATCH_REQUESTS = sys.maxsize
"""The Gemini API limits a batch by the size of its input file, `MAX_BATCH_BYTES`, rather than its number of requests."""

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
_RPC_STATUSES: dict[int, tuple[str, int]] = {
    1: ('CANCELLED', 499),
    2: ('UNKNOWN', 500),
    3: ('INVALID_ARGUMENT', 400),
    4: ('DEADLINE_EXCEEDED', 504),
    5: ('NOT_FOUND', 404),
    6: ('ALREADY_EXISTS', 409),
    7: ('PERMISSION_DENIED', 403),
    8: ('RESOURCE_EXHAUSTED', 429),
    9: ('FAILED_PRECONDITION', 400),
    10: ('ABORTED', 409),
    11: ('OUT_OF_RANGE', 400),
    12: ('UNIMPLEMENTED', 501),
    13: ('INTERNAL', 500),
    14: ('UNAVAILABLE', 503),
    15: ('DATA_LOSS', 500),
    16: ('UNAUTHENTICATED', 401),
}
"""The status name and HTTP status `generateContent` answers each `google.rpc.Code` with."""


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

_UPLOADED_INPUT_FILES: OrderedDict[tuple[str, str], str] = OrderedDict()
"""The input file this process uploaded for each job it created, to delete once the job has ended.

Keyed by the endpoint and credential that created the job, from `batch_credential`, and its name, as job names are only
unique within one project: a client of another project never finds, or deletes, this one's file.

A `Batch` handle can come from a stored message history, which a caller controls, and the Gemini API doesn't say which
file a job reads, so only a file this process uploaded for the job is ever deleted. The oldest are forgotten first,
which leaves their files to expire on their own.
"""
_UPLOADED_INPUT_FILES_SIZE = 1024


async def batch_encode(model: GoogleModel, request: BatchRequest) -> EncodedBatchRequest:
    prepared_settings, model_request_parameters = model.prepare_request(
        request.model_settings, request.model_request_parameters
    )
    settings = cast('GoogleModelSettings', dict(prepared_settings or {}))
    # A batch is created with one set of HTTP options, and its requests run long after, so neither can be honored.
    if 'timeout' in settings:
        raise UserError('Gemini batch requests do not support per-request `timeout`; configure the client instead.')
    # The provider runs a batched request itself, with no SDK client retrying it, so a retry count has nothing to apply
    # to and is ignored, like settings a model doesn't support.
    settings.pop('max_retries', None)
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
    encoded = EncodedBatchRequest(custom_id=request.custom_id, group=_batch_group(model), body=to_json(body).decode())
    check_batch_bytes([encoded], request_size, MAX_BATCH_BYTES, 'Gemini')
    return encoded


def request_size(request: EncodedBatchRequest) -> int:
    """The bytes `request` adds to a batch's input file: its line and the newline that separates it from the next."""
    return len(_input_line(request)) + 1


def _input_line(request: EncodedBatchRequest) -> bytes:
    # The body is this provider's own `batch_encode` output, so it's embedded as is.
    return b''.join((b'{"key":', to_json(request.custom_id), b',"request":', request.body.encode(), b'}'))


async def batch_create(model: GoogleModel, requests: Sequence[EncodedBatchRequest]) -> Batch:
    check_allow_model_requests()
    group = shared_group(requests, 'Gemini', 'client configurations or models')
    _validate_batch_group(model, group)
    check_batch_bytes(requests, request_size, MAX_BATCH_BYTES, 'Gemini')
    client = _batch_client(model)
    file = await client.aio.files.upload(
        file=io.BytesIO(b'\n'.join(_input_line(request) for request in requests)),
        config=UploadFileConfigDict(mime_type='application/jsonl', display_name='pydantic-ai-batch'),
    )
    assert file.name is not None
    # Unlike the upload, the job can't be created twice safely: the first attempt may have created it.
    no_retries = HttpOptionsDict(retry_options={'attempts': 1})
    try:
        job = await client.aio.batches.create(
            model=model.model_name,
            src=file.name,
            config=CreateBatchJobConfigDict(display_name='pydantic-ai-batch', http_options=no_retries),
        )
    except errors.ClientError:
        # Google refused to create the job, so nothing reads the file, which would otherwise count against the
        # project's file storage until it expires. After a server error or no response, the job may have been created,
        # so the file is left for it.
        with suppress(errors.APIError, httpx2.TransportError):
            await client.aio.files.delete(name=file.name)
        raise
    assert job.name is not None
    _UPLOADED_INPUT_FILES[batch_credential(model), job.name] = file.name
    while len(_UPLOADED_INPUT_FILES) > _UPLOADED_INPUT_FILES_SIZE:
        _UPLOADED_INPUT_FILES.popitem(last=False)
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
    if _STATUSES.get(job.state or JobState.JOB_STATE_UNSPECIFIED) == 'ended' and (
        input_file := _UPLOADED_INPUT_FILES.pop((batch_credential(model), batch.id), None)
    ):
        # The job has ended, so its input file isn't needed again; it would otherwise count against the project's
        # file storage until it expires. The handle's `input_file` isn't trusted: see `_UPLOADED_INPUT_FILES`.
        with suppress(errors.APIError, httpx2.TransportError):
            await client.aio.files.delete(name=input_file)
    if job.dest is None or job.dest.file_name is None:
        # A job that expired, or was canceled or failed before it ran a request, writes no results. The job doesn't
        # say which requests it held, so its outcome fails all of them with its own reason.
        if job.state == JobState.JOB_STATE_FAILED:
            error = job.error
            code = error.code if error is not None else None
            raise BatchJobError(
                status_code=_RPC_STATUSES[code][1] if code in _RPC_STATUSES else 500,
                model_name=model.model_name,
                body={'error': error.model_dump(exclude_none=True) if error is not None else None},
            )
        if job.state == JobState.JOB_STATE_CANCELLED:
            raise BatchJobError(
                status_code=_RPC_STATUSES[1][1],
                model_name=model.model_name,
                body={'error': {'code': 1, 'message': 'The batch job was canceled before the model ran the request.'}},
            )
        if job.state == JobState.JOB_STATE_EXPIRED:
            # Ended by the provider rather than anyone's intent, like a live request's deadline.
            raise BatchJobError(
                status_code=_RPC_STATUSES[4][1],
                model_name=model.model_name,
                body={'error': {'code': 4, 'message': 'The batch job expired before the model ran the request.'}},
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
        error = error if is_str_dict(error) else {}
        message = message if isinstance(message := error.get('message'), str) else None
        if isinstance(code := error.get('code'), int) and code in _RPC_STATUSES:
            status, status_code = _RPC_STATUSES[code]
            # As the REST API answers a request with this status: its HTTP status as the `code`, and its name.
            body = {'error': {**error, 'code': status_code, 'status': status}}
        else:
            status, status_code, body = None, 500, {'error': error}
        raise _http_error(
            model.model_name,
            model._provider.model_id_namespace,  # pyright: ignore[reportPrivateUsage]
            status_code=status_code,
            status=status,
            message=message,
            body=body,
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


def unlisted_status(batch: Batch) -> BatchResultStatus | None:
    """Why a job left requests out of its results: a job canceled or expired while it ran writes rows for only some."""
    state = (batch.provider_details or {}).get('state')
    if state == JobState.JOB_STATE_CANCELLED.value:
        return 'canceled'
    if state == JobState.JOB_STATE_EXPIRED.value:
        return 'expired'
    return None


def batch_credential(model: GoogleModel) -> str:
    """The endpoint and credential `model` reaches batches with, as hashes."""
    group: _Group = json.loads(_batch_group(model))
    return f'{group["endpoint"]}:{group["credential"]}'


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

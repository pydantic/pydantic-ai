"""The Gemini batch codec's behavior that involves no network call.

Submission, status, results and cancellation against the live API are recorded in `test_google_batch_vcr.py`.
These pin what a recording can't: the request body the codec builds (cassettes aren't matched on bodies), the errors
raised before anything is sent, and results the API rarely produces on demand.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, ModelResponse, ToolDefinition, UserPromptPart
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.direct import model_request_batch
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelOverloadedError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelServerError,
    ModelUnavailableError,
    UserError,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._batch_model import BatchPool
from pydantic_ai.models.batch import Batch, BatchRequest, BatchResult
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.settings import ModelSettings

from ..batch_utils import FAST_BATCH_SETTINGS, FAST_POLLING, UNRUN_MESSAGES
from ..conftest import try_import

with try_import() as imports_successful:
    from google.genai import Client, errors
    from google.genai.types import BatchJob, BatchJobDestination, File, HttpOptions, JobError, JobState

    from pydantic_ai.models import _google_batch as google_batch
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.profiles.google import GoogleModelProfile
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


def _request(custom_id: str = 'item', settings: ModelSettings | None = None) -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id, messages=[ModelRequest(parts=[UserPromptPart('Hello')])], model_settings=settings
    )


async def test_gemini_batch_request_is_the_generate_content_request_without_secrets() -> None:
    """The codec runs the request builder `GoogleModel.request` uses, and converts it to the REST body."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='secret-key'))
    request = BatchRequest(
        custom_id='item',
        messages=[ModelRequest(parts=[UserPromptPart('Hello')], instructions='Be brief.')],
        model_settings=GoogleModelSettings(max_tokens=100, stop_sequences=['END'], google_labels={'team': 'ai'}),
        model_request_parameters=ModelRequestParameters(
            function_tools=[ToolDefinition(name='lookup', parameters_json_schema={'type': 'object'})]
        ),
    )

    encoded = await model.batch_encode(request)

    assert encoded.custom_id == 'item'
    assert json.loads(encoded.body) == snapshot(
        {
            'contents': [{'parts': [{'text': 'Hello'}], 'role': 'user'}],
            'systemInstruction': {'parts': [{'text': 'Be brief.'}], 'role': 'user'},
            'generationConfig': {
                'maxOutputTokens': 100,
                'stopSequences': ['END'],
                'responseModalities': ['TEXT'],
            },
            'tools': [
                {
                    'functionDeclarations': [
                        {'name': 'lookup', 'description': '', 'parameters_json_schema': {'type': 'object'}}
                    ]
                }
            ],
            'toolConfig': {'functionCallingConfig': {'mode': 'VALIDATED'}},
            'labels': {'team': 'ai'},
        }
    )
    group = json.loads(encoded.group)
    assert group['model'] == 'gemini-2.5-flash-lite'
    assert 'secret-key' not in encoded.group
    assert 'generativelanguage' not in encoded.group


@pytest.mark.parametrize(
    'settings, setting',
    [
        ({'timeout': 10}, 'timeout'),
        ({'max_retries': 0}, 'max_retries'),
        ({'extra_headers': {'x-custom': '1'}}, 'extra_headers'),
    ],
)
async def test_gemini_batch_rejects_per_request_http_options(settings: ModelSettings, setting: str) -> None:
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    with pytest.raises(UserError, match=f'do not support per-request `{setting}`'):
        await model.batch_encode(_request(settings=settings))


async def test_gemini_batch_is_not_on_google_cloud() -> None:
    """Vertex AI batch prediction reads Cloud Storage or BigQuery, which `GoogleModel` doesn't manage."""
    model = GoogleModel(
        'gemini-2.5-flash-lite', provider=GoogleCloudProvider(project='project', location='us-central1')
    )

    assert model.batch_max_requests is None
    with pytest.raises(UserError) as exc_info:
        BatchPool().wrap(model)
    assert str(exc_info.value) == snapshot(
        "`GoogleModel` 'gemini-2.5-flash-lite' has no batch API on Google Cloud, whose batch prediction jobs read "
        'from Cloud Storage or BigQuery: batch it through the Gemini API, with the `google:` provider.'
    )
    with pytest.raises(UserError, match='only supported by the Gemini API, not Google Cloud'):
        await model.batch_encode(_request())


def test_gemini_batch_is_on_the_gemini_api() -> None:
    """The Gemini API limits a batch by its input file's 2 GB, not its number of requests."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    assert model.batch_max_requests == sys.maxsize
    assert model.batch_max_bytes == 2_000_000_000
    cloud = GoogleModel(
        'gemini-2.5-flash-lite', provider=GoogleCloudProvider(project='project', location='us-central1')
    )
    assert cloud.batch_max_bytes is None


def test_gemini_batch_is_off_through_the_gateway() -> None:
    """The Pydantic AI Gateway routes Google models to Google Cloud, which has no Batch Mode."""
    model = GoogleModel(
        'gemini-2.5-flash-lite',
        provider=gateway_provider('google', api_key='paig_test', base_url='https://gateway.example'),
    )
    assert model.batch_max_requests is None


def test_gemini_batch_follows_the_profile_flag() -> None:
    """`profile=` overrides the transport's default of `google_supports_batch_api` in either direction."""
    opted_out = GoogleModel(
        'gemini-2.5-flash-lite',
        provider=GoogleProvider(api_key='test-key'),
        profile=GoogleModelProfile(google_supports_batch_api=False),
    )
    assert opted_out.batch_max_requests is None
    assert opted_out.batch_max_bytes is None
    with pytest.raises(UserError) as exc_info:
        BatchPool().wrap(opted_out)
    assert str(exc_info.value) == snapshot("`GoogleModel` 'gemini-2.5-flash-lite' has no batch API.")

    opted_in = GoogleModel(
        'gemini-2.5-flash-lite',
        provider=GoogleCloudProvider(project='project', location='us-central1'),
        profile=GoogleModelProfile(google_supports_batch_api=True),
    )
    assert opted_in.batch_max_requests == sys.maxsize


async def test_gemini_batch_create_validates_requests(allow_model_requests: None) -> None:
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    other_model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))
    other_key = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='other-key'))
    request = await model.batch_encode(_request('one'))

    with pytest.raises(UserError, match='at least one request'):
        await model.batch_create([])
    with pytest.raises(UserError, match='must share a `group`'):
        await model.batch_create([request, await other_model.batch_encode(_request('two'))])
    with pytest.raises(UserError, match='does not match the batch request group'):
        await other_model.batch_create([request])
    with pytest.raises(UserError, match='does not match the batch request group'):
        await other_key.batch_create([request])


async def test_gemini_batch_group_is_the_credential_the_client_sends(allow_model_requests: None) -> None:
    """A credential set through header overrides, rather than the client's API key, keeps clients' requests apart."""

    def model(api_key: str = 'placeholder', **headers: str) -> GoogleModel:
        client = Client(api_key=api_key, http_options=HttpOptions(headers=headers))
        return GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(client=client))

    async def group(model: GoogleModel) -> str:
        return (await model.batch_encode(_request())).group

    tenant = model(**{'x-goog-api-key': 'tenant-key'})
    groups = [
        await group(model()),
        await group(tenant),
        await group(model(**{'X-Goog-Api-Key': 'other-tenant-key'})),
        await group(model(Authorization='Bearer token')),
    ]
    assert len(set(groups)) == len(groups)
    # The HTTP client's own credential headers count too, unless the request's replace them.
    http_client = httpx2.AsyncClient(headers={'Authorization': 'Bearer token'})
    assert (
        await group(
            GoogleModel(
                'gemini-2.5-flash-lite', provider=GoogleProvider(api_key='placeholder', http_client=http_client)
            )
        )
        == groups[3]
    )
    # Overriding the API key header with the API key is the same credential.
    assert await group(model(api_key='tenant-key')) == groups[1]
    assert 'tenant-key' not in groups[1]
    with pytest.raises(UserError, match='does not match the batch request group'):
        await model().batch_create([await tenant.batch_encode(_request())])


async def test_gemini_batch_rejects_an_http_client_auth() -> None:
    """An HTTP client's `auth` can send any credential, which the client's headers don't show, so its group can't either."""
    http_client = httpx2.AsyncClient(auth=('tenant-a', 'secret'))
    model = GoogleModel(
        'gemini-2.5-flash-lite', provider=GoogleProvider(api_key='placeholder', http_client=http_client)
    )
    with pytest.raises(UserError, match='not an `auth` on its HTTP client'):
        await model.batch_encode(_request())


async def test_gemini_batch_handles_check_their_group_and_status(allow_model_requests: None) -> None:
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    other_model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test-key'))
    group = (await model.batch_encode(_request())).group
    batch = Batch(
        id='batches/123',
        model_name=model.model_name,
        provider_name=model.system,
        group=group,
        status='in_progress',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        request_count=1,
    )

    with pytest.raises(UserError, match='has not ended yet'):
        await model.batch_results(batch)
    for method in (other_model.batch_status, other_model.batch_cancel):
        with pytest.raises(UserError, match='does not match the batch request group'):
            await method(batch)


@pytest.mark.parametrize(
    'job, expected',
    [
        (
            {'state': 'JOB_STATE_FAILED', 'error': {'code': 8, 'message': 'Quota exceeded.'}},
            (
                ModelHTTPError,
                "status_code: 429, model_name: gemini-2.5-flash-lite, body: {'error': {'code': 8, 'message': 'Quota exceeded.'}}",
            ),
        ),
        (
            {'state': 'JOB_STATE_FAILED'},
            (ModelHTTPError, "status_code: 500, model_name: gemini-2.5-flash-lite, body: {'error': None}"),
        ),
        (
            {'state': 'JOB_STATE_CANCELLED'},
            (
                ModelHTTPError,
                "status_code: 499, model_name: gemini-2.5-flash-lite, body: {'error': {'code': 1, 'message': 'The batch job was canceled before the model ran the request.'}}",
            ),
        ),
        (
            {'state': 'JOB_STATE_EXPIRED'},
            (
                ModelHTTPError,
                "status_code: 504, model_name: gemini-2.5-flash-lite, body: {'error': {'code': 4, 'message': 'The batch job expired before the model ran the request.'}}",
            ),
        ),
        # A job that says it succeeded without writing results leaves its requests without one.
        ({'state': 'JOB_STATE_SUCCEEDED'}, None),
    ],
)
async def test_gemini_batch_job_without_results_reports_why(
    allow_model_requests: None,
    monkeypatch: pytest.MonkeyPatch,
    job: dict[str, Any],
    expected: tuple[type[Exception], str] | None,
) -> None:
    """A job that ended without a results file doesn't say which requests it held, so its own outcome stands for all."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    batch = Batch(
        id='batches/123',
        model_name=model.model_name,
        provider_name=model.system,
        group=(await model.batch_encode(_request())).group,
        status='ended',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        request_count=1,
    )

    async def get(*, name: str, config: Any = None) -> BatchJob:
        return BatchJob.model_validate({'name': name, **job})

    monkeypatch.setattr(model.client.aio.batches, 'get', get)

    if expected is None:
        assert await model.batch_results(batch) == []
    else:
        error_type, message = expected
        with pytest.raises(error_type) as exc_info:
            await model.batch_results(batch)
        assert str(exc_info.value) == message


@pytest.mark.parametrize('status', ['expired', 'canceled'])
def test_gemini_batch_request_that_never_ran_is_a_model_error(status: str) -> None:
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    result = BatchResult(custom_id='item', status=status)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ModelAPIError, match=UNRUN_MESSAGES[status]):
        model.batch_decode(result, _request())


@pytest.mark.parametrize(
    'error, error_class, status_code, body',
    [
        (
            {'code': 3, 'message': 'Invalid argument.'},
            ModelRequestRejectedError,
            400,
            {'error': {'code': 400, 'message': 'Invalid argument.', 'status': 'INVALID_ARGUMENT'}},
        ),
        (
            {
                'code': 3,
                'message': 'The input token count (2000000) exceeds the maximum number of tokens allowed (1048576).',
            },
            ModelContextWindowExceededError,
            400,
            None,
        ),
        (
            {'code': 8, 'message': 'Quota exceeded.'},
            ModelRateLimitError,
            429,
            {'error': {'code': 429, 'message': 'Quota exceeded.', 'status': 'RESOURCE_EXHAUSTED'}},
        ),
        ({'code': 13, 'message': 'Internal error.'}, ModelServerError, 500, None),
        ({'code': 14, 'message': 'The model is overloaded. Please try again later.'}, ModelOverloadedError, 503, None),
        ({'code': 14, 'message': 'Service unavailable.'}, ModelUnavailableError, 503, None),
        (
            {'code': 99, 'message': 'Unknown code.'},
            ModelServerError,
            500,
            {'error': {'code': 99, 'message': 'Unknown code.'}},
        ),
        ({'message': 'No code.'}, ModelServerError, 500, {'error': {'message': 'No code.'}}),
    ],
)
def test_gemini_batch_error_raises_the_error_of_its_rpc_code(
    error: dict[str, object], error_class: type[ModelAPIError], status_code: int, body: object
) -> None:
    """A per-request error is a `google.rpc.Status`, which raises the error `generateContent` would have for it.

    That's the HTTP status and category the REST API gives the same status, and its body as the REST API sends it.
    """
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    result = BatchResult(custom_id='item', status='errored', body=json.dumps(error))
    with pytest.raises(error_class) as exc_info:
        model.batch_decode(result, _request())
    assert isinstance(exc_info.value, ModelHTTPError)
    assert exc_info.value.status_code == status_code
    if body is not None:
        assert exc_info.value.body == body


def test_gemini_batch_error_suggests_a_known_model_for_an_unknown_one() -> None:
    model = GoogleModel('gemini-2.5-flash-lit', provider=GoogleProvider(api_key='test-key'))
    message = 'models/gemini-2.5-flash-lit is not found for API version v1beta, or is not supported.'
    result = BatchResult(custom_id='item', status='errored', body=json.dumps({'code': 5, 'message': message}))
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(result, _request())
    assert (exc_info.value.status_code, exc_info.value.provider_error_code) == (404, 'NOT_FOUND')
    assert exc_info.value.suggested_model_id == 'google:gemini-2.5-flash-lite'


_OK_RESPONSE = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Hi!'}]}, 'finishReason': 'STOP'}]}


def _fake_jobs(
    monkeypatch: pytest.MonkeyPatch, model: GoogleModel, job: dict[str, Any], rows: list[dict[str, Any]] | None = None
) -> list[str]:
    """Serve `job` for every job, with `rows` in its results file, and return the names of the files deleted."""
    deleted: list[str] = []

    async def upload(*, file: Any, config: Any = None) -> File:
        return File(name='files/input')

    async def create(*, model: str, src: str, config: Any = None) -> BatchJob:
        return BatchJob(name='batches/1', state=JobState.JOB_STATE_PENDING)

    async def get(*, name: str, config: Any = None) -> BatchJob:
        return BatchJob.model_validate({'name': name, **job})

    async def download(*, file: str, config: Any = None) -> bytes:
        return '\n'.join(json.dumps(row) for row in rows or []).encode()

    async def delete(*, name: str, config: Any = None) -> None:
        deleted.append(name)

    monkeypatch.setattr(model.client.aio.files, 'upload', upload)
    monkeypatch.setattr(model.client.aio.files, 'download', download)
    monkeypatch.setattr(model.client.aio.files, 'delete', delete)
    monkeypatch.setattr(model.client.aio.batches, 'create', create)
    monkeypatch.setattr(model.client.aio.batches, 'get', get)
    return deleted


@pytest.mark.parametrize(
    'state, status',
    [
        (None, 'in_progress'),
        ('JOB_STATE_UNSPECIFIED', 'in_progress'),
        ('JOB_STATE_QUEUED', 'in_progress'),
        ('JOB_STATE_PENDING', 'in_progress'),
        ('JOB_STATE_RUNNING', 'in_progress'),
        ('JOB_STATE_PAUSED', 'in_progress'),
        ('JOB_STATE_UPDATING', 'in_progress'),
        ('JOB_STATE_CANCELLING', 'canceling'),
        ('JOB_STATE_SUCCEEDED', 'ended'),
        ('JOB_STATE_PARTIALLY_SUCCEEDED', 'ended'),
        ('JOB_STATE_FAILED', 'ended'),
        ('JOB_STATE_CANCELLED', 'ended'),
        ('JOB_STATE_EXPIRED', 'ended'),
    ],
)
async def test_gemini_batch_status_maps_every_job_state(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch, state: str | None, status: str
) -> None:
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    _fake_jobs(monkeypatch, model, {'state': state})
    batch = await model.batch_create([await model.batch_encode(_request())])
    batch = await model.batch_status(batch)
    assert batch.status == status
    assert batch.provider_details == {'input_file': 'files/input', 'state': state or 'JOB_STATE_UNSPECIFIED'}


async def test_gemini_partially_succeeded_job_returns_its_rows(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job that partially succeeded writes a row for each request it ran: a response or an error, matched by key.

    A request it has no row for isn't in the results, which the caller reports as unrun. Once the job has ended, its
    input file is deleted.
    """
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    rows = [
        {'key': 'failed', 'error': {'code': 8, 'message': 'Quota exceeded.'}},
        {'key': 'ok', 'response': _OK_RESPONSE},
    ]
    deleted = _fake_jobs(
        monkeypatch,
        model,
        {'state': 'JOB_STATE_PARTIALLY_SUCCEEDED', 'dest': {'file_name': 'files/out'}},
        rows,
    )
    requests = {custom_id: _request(custom_id) for custom_id in ('ok', 'failed', 'missing')}
    batch = await model.batch_status(
        await model.batch_create([await model.batch_encode(request) for request in requests.values()])
    )
    assert batch.status == 'ended'
    results = {result.custom_id: result for result in await model.batch_results(batch)}
    assert {custom_id: result.status for custom_id, result in results.items()} == {
        'failed': 'errored',
        'ok': 'succeeded',
    }
    assert model.batch_decode(results['ok'], requests['ok']).text == 'Hi!'
    with pytest.raises(ModelRateLimitError):
        model.batch_decode(results['failed'], requests['failed'])
    assert deleted == ['files/input']


async def test_gemini_batch_results_only_delete_the_input_file_this_process_uploaded(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handle's `input_file` comes from a stored history a caller controls, so it never names the file to delete."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    deleted = _fake_jobs(monkeypatch, model, {'state': 'JOB_STATE_SUCCEEDED', 'dest': {'file_name': 'files/out'}})
    batch = await model.batch_status(await model.batch_create([await model.batch_encode(_request())]))

    forged = replace(batch, provider_details={**(batch.provider_details or {}), 'input_file': 'files/someone-elses'})
    await model.batch_results(forged)
    assert deleted == ['files/input']
    # Collecting it again, or a job this process didn't create, deletes nothing.
    await model.batch_results(forged)
    await model.batch_results(replace(forged, id='batches/from-history'))
    assert deleted == ['files/input']


async def test_gemini_batch_results_dont_delete_a_file_another_project_uploaded_for_a_job_with_the_same_name(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Job names are only unique within a project, so another project's job of the same name isn't this upload's."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    _fake_jobs(monkeypatch, model, {'state': 'JOB_STATE_PENDING'})
    batch = await model.batch_create([await model.batch_encode(_request())])
    other = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='other-project-key'))
    deleted = _fake_jobs(monkeypatch, other, {'state': 'JOB_STATE_SUCCEEDED', 'dest': {'file_name': 'files/out'}})

    handle = replace(batch, group=(await other.batch_encode(_request())).group)
    await other.batch_results(await other.batch_status(handle))
    assert deleted == []


@pytest.mark.parametrize(
    'make_error, deleted',
    [
        # Built lazily, as `errors` only exists when google-genai is installed.
        (lambda: errors.ClientError(429, {'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED'}}), True),
        (lambda: errors.ServerError(500, {'error': {'code': 500, 'status': 'INTERNAL'}}), False),
    ],
    ids=['refused', 'server-error'],
)
async def test_gemini_refused_job_deletes_its_input_file(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch, make_error: Callable[[], Exception], deleted: bool
) -> None:
    """Nothing reads the file of a job Google refused; after a server error, the job may exist and read it."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    deleted_files = _fake_jobs(monkeypatch, model, {'state': 'JOB_STATE_PENDING'})
    error = make_error()

    async def create(*, model: str, src: str, config: Any = None) -> BatchJob:
        raise error

    monkeypatch.setattr(model.client.aio.batches, 'create', create)
    with pytest.raises(ModelHTTPError):
        await model.batch_create([await model.batch_encode(_request())])
    assert deleted_files == (['files/input'] if deleted else [])


async def test_gemini_batch_results_keep_the_input_file_of_a_job_still_running(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A handle that claims its job ended doesn't make the job's input file deletable while the job still runs."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    deleted = _fake_jobs(monkeypatch, model, {'state': 'JOB_STATE_RUNNING'})
    batch = await model.batch_create([await model.batch_encode(_request())])

    assert await model.batch_results(replace(batch, status='ended')) == []
    assert deleted == []


async def test_gemini_succeeded_job_whose_rows_all_failed(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job can succeed with every request failing on its own, so each raises its own error."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    rows = [
        {'key': 'one', 'error': {'code': 3, 'message': 'Invalid argument.'}},
        {'key': 'two', 'error': {'code': 13, 'message': 'Internal error.'}},
    ]
    _fake_jobs(monkeypatch, model, {'state': 'JOB_STATE_SUCCEEDED', 'dest': {'file_name': 'files/out'}}, rows)
    requests = [_request('one'), _request('two')]
    batch = await model.batch_status(await model.batch_create([await model.batch_encode(r) for r in requests]))
    one, two = await model.batch_results(batch)
    assert (one.status, two.status) == ('errored', 'errored')
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(one, requests[0])
    assert exc_info.value.status_code == 400
    with pytest.raises(ModelServerError):
        model.batch_decode(two, requests[1])


@pytest.mark.parametrize(
    'make_error',
    [
        # Built lazily, as `errors` only exists when google-genai is installed.
        lambda: errors.ClientError(404, {'error': {'code': 404, 'message': 'Not found', 'status': 'NOT_FOUND'}}),
        lambda: httpx2.ConnectError('Connection refused'),
    ],
    ids=['not-found', 'connect-error'],
)
async def test_gemini_batch_results_survive_failing_to_delete_the_input_file(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch, make_error: Callable[[], Exception]
) -> None:
    """Deleting the input file is best effort: the file expires on its own, and the results matter more."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    _fake_jobs(
        monkeypatch,
        model,
        {'state': 'JOB_STATE_SUCCEEDED', 'dest': {'file_name': 'files/out'}},
        [{'key': 'ok', 'response': _OK_RESPONSE}],
    )

    error = make_error()

    async def delete(*, name: str, config: Any = None) -> None:
        raise error

    monkeypatch.setattr(model.client.aio.files, 'delete', delete)
    batch = await model.batch_status(await model.batch_create([await model.batch_encode(_request('ok'))]))
    [result] = await model.batch_results(batch)
    assert result.status == 'succeeded'


async def test_gemini_batch_size_limits(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request counts the bytes of its line in the input file, which the Gemini API limits to 2 GB.

    A request too large for any batch is refused when it's encoded, and requests too large for one batch when
    they're submitted, before anything is uploaded.
    """
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    uploads: list[bytes] = []
    _fake_jobs(monkeypatch, model, {'state': 'JOB_STATE_PENDING'})

    async def upload(*, file: Any, config: Any = None) -> File:
        uploads.append(file.getvalue())
        return File(name='files/input')

    monkeypatch.setattr(model.client.aio.files, 'upload', upload)
    encoded = [await model.batch_encode(_request(custom_id)) for custom_id in ('one', 'twö')]
    sizes = [model.batch_request_size(request) for request in encoded]
    await model.batch_create(encoded)
    # Lines are separated by newlines, which each request counts.
    assert len(uploads[0]) == sum(sizes) - 1
    assert [json.loads(line)['key'] for line in uploads[0].splitlines()] == ['one', 'twö']

    monkeypatch.setattr(google_batch, 'MAX_BATCH_BYTES', sizes[0])
    with pytest.raises(UserError, match=f'These requests take {sum(sizes):,} bytes, more than the {sizes[0]:,} bytes'):
        await model.batch_create(encoded)
    with pytest.raises(UserError, match=f'This request takes {sizes[1]:,} bytes, more than the {sizes[0]:,} bytes'):
        await model.batch_encode(_request('twö'))
    assert len(uploads) == 1


async def test_gemini_batch_request_is_the_request_sent_outside_a_batch(allow_model_requests: None) -> None:
    """A batch request carries exactly the body `generateContent` would get: tools, output schema, thinking, safety."""
    sent: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.read()))
        return httpx2.Response(200, json=_OK_RESPONSE)

    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(handle))
    model = GoogleModel('gemini-3-flash-preview', provider=GoogleProvider(api_key='test-key', http_client=http_client))
    messages: list[Any] = [ModelRequest(parts=[UserPromptPart('Hello')], instructions='Answer briefly')]
    settings = GoogleModelSettings(
        max_tokens=100,
        google_thinking_config={'thinking_budget': 512, 'include_thoughts': True},
        google_safety_settings=[{'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_LOW_AND_ABOVE'}],  # pyright: ignore[reportArgumentType]
    )
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='weather', parameters_json_schema={'type': 'object', 'properties': {}})],
        output_mode='native',
        output_object=OutputObjectDefinition(
            json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
            name='City',
        ),
        allow_text_output=False,
    )
    await model.request(messages, settings, parameters)
    encoded = await model.batch_encode(
        BatchRequest(custom_id='item', messages=messages, model_settings=settings, model_request_parameters=parameters)
    )
    assert json.loads(encoded.body) == sent[0]


async def test_model_request_batch_keeps_other_jobs_results_when_one_fails(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A job that failed without results fails only its own requests; the other jobs' responses are kept."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    batches = model.client.aio.batches
    files = model.client.aio.files
    uploads: list[bytes] = []

    async def upload(*, file: Any, config: Any = None) -> File:
        uploads.append(file.getvalue())
        return File(name=f'files/input-{len(uploads)}')

    async def create(*, model: str, src: str, config: Any = None) -> BatchJob:
        return BatchJob(name=f'batches/{src.rsplit("-", 1)[1]}', state=JobState.JOB_STATE_PENDING)

    async def get(*, name: str, config: Any = None) -> BatchJob:
        if name == 'batches/1':
            return BatchJob(
                name=name, state=JobState.JOB_STATE_SUCCEEDED, dest=BatchJobDestination(file_name='files/out-1')
            )
        return BatchJob(name=name, state=JobState.JOB_STATE_FAILED, error=JobError(code=8, message='Quota exceeded.'))

    async def download(*, file: str, config: Any = None) -> bytes:
        key = json.loads(uploads[0])['key']
        response = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Hi!'}]}, 'finishReason': 'STOP'}]}
        return json.dumps({'key': key, 'response': response}).encode()

    monkeypatch.setattr(files, 'upload', upload)
    monkeypatch.setattr(files, 'download', download)
    monkeypatch.setattr(batches, 'create', create)
    monkeypatch.setattr(batches, 'get', get)

    first, second = await model_request_batch(
        model,
        [_request('first'), _request('second')],
        max_requests=1,
        poll_interval=timedelta(milliseconds=1),
        max_poll_interval=timedelta(milliseconds=1),
    )

    assert isinstance(first, ModelResponse) and first.text == 'Hi!'
    assert isinstance(second, ModelHTTPError)
    assert second.status_code == 429


@pytest.mark.parametrize('state', ['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'])
async def test_gemini_job_that_ended_early_returns_the_rows_it_wrote(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    """A job that failed, was canceled or expired after writing results keeps them; its other requests have no row."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    deleted = _fake_jobs(
        monkeypatch,
        model,
        {'state': state, 'dest': {'file_name': 'files/out'}},
        [{'key': 'ran', 'response': _OK_RESPONSE}],
    )
    requests = [_request('ran'), _request('unrun')]
    batch = await model.batch_status(await model.batch_create([await model.batch_encode(r) for r in requests]))
    [result] = await model.batch_results(batch)
    assert (result.custom_id, result.status) == ('ran', 'succeeded')
    assert deleted == ['files/input']


def test_gemini_batch_credential_tells_clients_apart_without_secrets() -> None:
    """Checks and results of a batch are only shared between models that reach it with one credential."""

    def credential(api_key: str) -> str:
        model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key=api_key))
        return model._batch_credential()  # pyright: ignore[reportPrivateUsage]

    assert credential('first-key') == credential('first-key') != credential('second-key')
    assert 'first-key' not in credential('first-key')


async def test_gemini_batch_forgets_the_oldest_uploaded_input_files(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a bounded number of uploads are remembered; a forgotten job's input file is left to expire."""
    monkeypatch.setattr('pydantic_ai.models._google_batch._UPLOADED_INPUT_FILES_SIZE', 0)
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    deleted = _fake_jobs(monkeypatch, model, {'state': 'JOB_STATE_SUCCEEDED', 'dest': {'file_name': 'files/out'}})
    batch = await model.batch_status(await model.batch_create([await model.batch_encode(_request())]))
    await model.batch_results(batch)
    assert deleted == []


def _counting_creates(monkeypatch: pytest.MonkeyPatch, model: GoogleModel) -> list[str]:
    """Record the source file of every job created, on top of `_fake_jobs`."""
    created: list[str] = []
    create = model.client.aio.batches.create

    async def counted(*, model: str, src: str, config: Any = None) -> BatchJob:
        created.append(src)
        return await create(model=model, src=src, config=config)

    monkeypatch.setattr(model.client.aio.batches, 'create', counted)
    return created


@pytest.mark.parametrize(
    'state, error, submissions',
    [
        pytest.param('JOB_STATE_CANCELLED', UNRUN_MESSAGES['canceled'], 1, id='canceled'),
        pytest.param('JOB_STATE_EXPIRED', UNRUN_MESSAGES['expired'], 3, id='expired'),
    ],
)
async def test_gemini_job_that_ended_partway_says_why_it_left_requests_out(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch, state: str, error: str, submissions: int
) -> None:
    """A job canceled or expired while it ran writes rows for only some requests; the others ended the same way.

    A canceled request isn't submitted again, as someone meant to stop it, while an expired one is.
    """
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    _fake_jobs(
        monkeypatch,
        model,
        {'state': state, 'dest': {'file_name': 'files/out'}},
        [{'key': 'other', 'response': _OK_RESPONSE}],
    )
    created = _counting_creates(monkeypatch, model)
    agent = Agent(model, capabilities=[BatchModelRequests(**FAST_BATCH_SETTINGS)])

    with pytest.raises(ModelAPIError, match=error):
        await agent.run('Hello')
    assert len(created) == submissions

    created.clear()
    (response,) = await model_request_batch(model, [_request()], **FAST_POLLING)
    assert isinstance(response, ModelAPIError) and re.search(error, str(response))
    assert len(created) == submissions

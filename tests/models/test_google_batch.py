"""The Gemini batch codec's behavior that involves no network call.

Submission, status, results and cancellation against the live API are recorded in `test_google_batch_vcr.py`.
These pin what a recording can't: the request body the codec builds (cassettes aren't matched on bodies), the errors
raised before anything is sent, and results the API rarely produces on demand.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx2
import pytest
from inline_snapshot import snapshot

from pydantic_ai import ModelRequest, ModelResponse, ToolDefinition, UserPromptPart
from pydantic_ai.direct import model_request_batch
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._batch_model import BatchPool
from pydantic_ai.models.batch import Batch, BatchRequest, BatchResult
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

with try_import() as imports_successful:
    from google.genai import Client
    from google.genai.types import BatchJob, BatchJobDestination, File, HttpOptions, JobError, JobState

    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
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
    with pytest.raises(UserError, match='has no batch API'):
        BatchPool().wrap(model)
    with pytest.raises(UserError, match='only supported by the Gemini API, not Google Cloud'):
        await model.batch_encode(_request())


def test_gemini_batch_is_on_the_gemini_api() -> None:
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    assert model.batch_max_requests == 100_000


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
        ({'state': 'JOB_STATE_EXPIRED'}, None),
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
    with pytest.raises(ModelAPIError, match=f'The batch request was {status} before the model ran it.'):
        model.batch_decode(result, _request())


@pytest.mark.parametrize(
    'error, status_code',
    [
        ({'code': 3, 'message': 'Invalid argument.'}, 400),
        ({'code': 8, 'message': 'Quota exceeded.'}, 429),
        ({'code': 99, 'message': 'Unknown code.'}, 500),
        ({'message': 'No code.'}, 500),
    ],
)
def test_gemini_batch_error_raises_the_http_error_of_its_rpc_code(error: dict[str, object], status_code: int) -> None:
    """A per-request error carries a `google.rpc.Code`, which maps to the HTTP status `generateContent` would return."""
    model = GoogleModel('gemini-2.5-flash-lite', provider=GoogleProvider(api_key='test-key'))
    result = BatchResult(custom_id='item', status='errored', body=json.dumps(error))
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(result, _request())
    assert exc_info.value.status_code == status_code
    assert exc_info.value.body == {'error': error}


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

"""The Gemini batch codec's behavior that involves no network call.

Submission, status, results and cancellation against the live API are recorded in `test_google_batch_vcr.py`.
These pin what a recording can't: the request body the codec builds (cassettes aren't matched on bodies), the errors
raised before anything is sent, and results the API rarely produces on demand.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from inline_snapshot import snapshot

from pydantic_ai import ModelRequest, ToolDefinition, UserPromptPart
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.batch import Batch, BatchModel, BatchRequest, BatchResult
from pydantic_ai.settings import ModelSettings

from ..conftest import try_import

with try_import() as imports_successful:
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
        BatchModel(model)
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

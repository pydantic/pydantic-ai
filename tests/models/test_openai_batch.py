"""Unit tests for the OpenAI Batch API codec behind `OpenAIResponsesModel.batch_*`.

The round trip against the real API is recorded in `test_openai_batch_vcr.py`. These tests cover what that can't:
encoding (which makes no request), the gating of other providers, and batch outcomes the API can't be made to produce
on demand (an expired or failed batch, error lines), which are served by an in-process fake of the Batch API.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._batch_model import BatchPool
from pydantic_ai.models.batch import Batch, BatchRequest
from pydantic_ai.tools import ToolDefinition
from tests.conftest import try_import

with try_import() as imports_successful:
    import httpx2
    from openai import AsyncAzureOpenAI, AsyncOpenAI
    from openai.auth import WorkloadIdentity

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.models.openai_codex import OpenAICodexModel
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.gateway import gateway_provider
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as bedrock_mantle_imports_successful:
    from pydantic_ai.models.bedrock_mantle import BedrockMantleResponsesModel
    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def _request(custom_id: str = 'item', **settings: Any) -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id,
        messages=[ModelRequest(parts=[UserPromptPart('Hello')])],
        model_settings=OpenAIResponsesModelSettings(**settings),
    )


def _model(api_key: str = 'test-key', model_name: str = 'gpt-4.1-nano', **client_kwargs: Any) -> OpenAIResponsesModel:
    return OpenAIResponsesModel(
        model_name, provider=OpenAIProvider(openai_client=AsyncOpenAI(api_key=api_key, **client_kwargs))
    )


@pytest.mark.parametrize(
    'model_factory,max_requests',
    [
        pytest.param(lambda: _model(), 50_000, id='openai'),
        pytest.param(
            lambda: OpenAIResponsesModel(
                'gpt-4.1-nano',
                provider=gateway_provider('openai', api_key='paig_test', base_url='https://gateway.example'),
            ),
            50_000,
            id='gateway',
        ),
        pytest.param(
            lambda: OpenAIResponsesModel(
                'gpt-4.1-nano',
                provider=AzureProvider(azure_endpoint='https://test.openai.azure.com', api_version='v1', api_key='k'),
            ),
            None,
            id='azure',
        ),
        pytest.param(
            lambda: OpenAIResponsesModel(
                'gpt-4.1-nano',
                provider=OpenAIProvider(
                    openai_client=AsyncAzureOpenAI(
                        azure_endpoint='https://test.openai.azure.com', api_version='v1', api_key='k'
                    )
                ),
            ),
            None,
            id='azure-client',
        ),
        pytest.param(
            lambda: OpenAIResponsesModel('openai/gpt-4.1-nano', provider=OpenRouterProvider(api_key='k')),
            None,
            id='openrouter',
        ),
        pytest.param(
            lambda: OpenAICodexModel('gpt-5.6', provider=OpenAIProvider(api_key='k')), None, id='codex-subclass'
        ),
        pytest.param(lambda: OpenAIChatModel('gpt-4.1-nano', provider=OpenAIProvider(api_key='k')), None, id='chat'),
    ],
)
async def test_batch_api_is_only_on_openai_responses(
    model_factory: Callable[[], OpenAIResponsesModel | OpenAIChatModel], max_requests: int | None
) -> None:
    model = model_factory()
    assert model.batch_max_requests == max_requests
    if max_requests is None:
        with pytest.raises(UserError, match='has no batch API'):
            BatchPool().wrap(model)
    if max_requests is None and isinstance(model, OpenAIResponsesModel):
        with pytest.raises(UserError, match='does not support the OpenAI Batch API'):
            await model.batch_encode(_request())


@pytest.mark.skipif(not bedrock_mantle_imports_successful(), reason='bedrock-mantle not installed')
def test_bedrock_mantle_has_no_openai_batch_api() -> None:
    model = BedrockMantleResponsesModel(
        'openai.gpt-oss-120b',
        provider=BedrockMantleProvider(openai_client=AsyncOpenAI(api_key='k', base_url='https://mantle.example')),
    )
    assert model.batch_max_requests is None


async def test_batch_encode_carries_the_responses_params() -> None:
    """The codec runs the same request builder as `OpenAIResponsesModel.request`, flattening `extra_body` the way the SDK does."""
    model = _model()
    request = BatchRequest(
        custom_id='item',
        messages=[ModelRequest(parts=[UserPromptPart('Hello')], instructions='Answer briefly')],
        model_settings=OpenAIResponsesModelSettings(
            max_tokens=100, temperature=0.5, openai_user='user', extra_body={'metadata': {'source': 'test'}}
        ),
        model_request_parameters=ModelRequestParameters(
            function_tools=[ToolDefinition(name='weather', parameters_json_schema={'type': 'object', 'properties': {}})]
        ),
    )
    encoded = await model.batch_encode(request)
    assert encoded.custom_id == 'item'
    assert json.loads(encoded.body) == snapshot(
        {
            'model': 'gpt-4.1-nano',
            'input': [{'role': 'user', 'content': 'Hello'}],
            'instructions': 'Answer briefly',
            'tools': [
                {
                    'name': 'weather',
                    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False},
                    'type': 'function',
                    'description': None,
                    'strict': False,
                }
            ],
            'tool_choice': 'auto',
            'max_output_tokens': 100,
            'temperature': 0.5,
            'user': 'user',
            'metadata': {'source': 'test'},
        }
    )


async def test_batch_encode_continues_a_stored_response() -> None:
    """A batch request can continue a stored response, so `openai_previous_response_id` is sent as on its own."""
    model = _model()
    request = BatchRequest(
        custom_id='item',
        messages=[
            ModelRequest(parts=[UserPromptPart('Hello')]),
            ModelResponse(parts=[TextPart('Hi')], provider_name='openai', provider_response_id='resp_123'),
            ModelRequest(parts=[UserPromptPart('Again')]),
        ],
        model_settings=OpenAIResponsesModelSettings(openai_previous_response_id='auto'),
    )
    body = json.loads((await model.batch_encode(request)).body)
    assert body['previous_response_id'] == 'resp_123'
    assert body['input'] == [{'role': 'user', 'content': 'Again'}]


async def test_batch_group_holds_the_model_but_no_secrets() -> None:
    model = _model(api_key='secret-key', organization='org-secret', default_headers={'X-Tenant': 'tenant-secret'})
    group = (await model.batch_encode(_request())).group
    assert json.loads(group).keys() == {'endpoint', 'credential', 'model'}
    assert json.loads(group)['model'] == 'gpt-4.1-nano'
    for secret in ('secret-key', 'org-secret', 'tenant-secret', 'api.openai.com'):
        assert secret not in group

    for other in (_model(api_key='other-key'), _model(model_name='gpt-5-mini'), _model(organization='org-other')):
        assert (await other.batch_encode(_request())).group != group


async def test_batch_create_refuses_a_different_client_or_model(allow_model_requests: None) -> None:
    first = _model(api_key='first-key')
    request = await first.batch_encode(_request())
    batch = Batch(
        id='batch_1',
        model_name='gpt-4.1-nano',
        provider_name='openai',
        group=request.group,
        status='ended',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        request_count=1,
    )
    for other in (_model(api_key='second-key'), _model(api_key='first-key', model_name='gpt-5-mini')):
        with pytest.raises(UserError, match='does not match the batch request group'):
            await other.batch_create([request])
        with pytest.raises(UserError, match='does not match the batch request group'):
            await other.batch_status(batch)
        with pytest.raises(UserError, match='does not match the batch request group'):
            await other.batch_results(batch)
        with pytest.raises(UserError, match='does not match the batch request group'):
            await other.batch_cancel(batch)

    mixed = await _model(model_name='gpt-5-mini').batch_encode(_request('other'))
    with pytest.raises(UserError, match='must share a `group`'):
        await first.batch_create([request, mixed])
    with pytest.raises(UserError, match='at least one request'):
        await first.batch_create([])


@pytest.mark.parametrize(
    'settings,match',
    [
        ({'timeout': 10}, 'per-request `timeout`'),
        ({'max_retries': 0}, 'per-request `max_retries`'),
        ({'extra_headers': {'X-Tenant': 'a'}}, 'per-request `extra_headers`'),
        ({'openai_background': True}, '`openai_background`'),
        ({'extra_body': {'stream': True}}, '`stream` or `background`'),
        ({'extra_body': {'background': True}}, '`stream` or `background`'),
        ({'extra_body': ['not', 'a', 'mapping']}, '`extra_body` must be a mapping'),
    ],
)
async def test_batch_encode_rejects_what_a_batch_cannot_honor(settings: dict[str, Any], match: str) -> None:
    with pytest.raises(UserError, match=match):
        await _model().batch_encode(_request(**settings))


async def test_batch_requires_a_static_api_key() -> None:
    """A key callback could choose a different credential per request, while a batch is submitted and polled with one."""

    async def api_key() -> str:
        return 'tenant-key'  # pragma: no cover

    model = OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(openai_client=AsyncOpenAI(api_key=api_key)))
    with pytest.raises(UserError, match='requires a static API key'):
        await model.batch_encode(_request())


async def test_batch_rejects_a_workload_identity() -> None:
    """Clients of different workload identities have the same headers, so their requests would share a batch group."""
    identity: WorkloadIdentity = {
        'identity_provider_id': 'idp',
        'service_account_id': 'tenant-a',
        'provider': {'token_type': 'jwt', 'get_token': lambda: 'token'},
    }
    model = OpenAIResponsesModel(
        'gpt-4.1-nano', provider=OpenAIProvider(openai_client=AsyncOpenAI(workload_identity=identity))
    )
    with pytest.raises(UserError, match='not an API key callback or workload identity'):
        await model.batch_encode(_request())


@dataclass
class _FakeBatchAPI:
    """Serves the Files and Batches endpoints a batch round trip uses."""

    batch: dict[str, Any]
    files: dict[str, str] = field(default_factory=dict[str, str])
    requests: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path.removeprefix('/v1')
        self.requests.append((request.method, path))
        if request.method == 'POST' and path == '/files':
            # The JSONL file is the last part of the multipart upload.
            self.files['file-input'] = request.read().decode().split('\r\n\r\n')[-1].rsplit('\r\n--', 1)[0]
            return httpx2.Response(
                200,
                json={
                    'id': 'file-input',
                    'object': 'file',
                    'bytes': 1,
                    'created_at': 0,
                    'filename': 'batch.jsonl',
                    'purpose': 'batch',
                },
            )
        if path.startswith('/files/') and path.endswith('/content'):
            return httpx2.Response(200, content=self.files[path.split('/')[2]])
        if path == '/batches/batch_missing':
            return httpx2.Response(404, json={'error': {'message': 'No batch found', 'type': 'invalid_request_error'}})
        if path.endswith('/cancel'):
            self.batch['status'] = 'cancelling'
        return httpx2.Response(200, json=self.batch)


def _batch_json(status: str, **fields: Any) -> dict[str, Any]:
    return {
        'id': 'batch_1',
        'object': 'batch',
        'endpoint': '/v1/responses',
        'completion_window': '24h',
        'input_file_id': 'file-input',
        'created_at': 1_767_225_600,
        'status': status,
        'request_counts': {'total': 0, 'completed': 0, 'failed': 0},
        **fields,
    }


def _fake_model(api: _FakeBatchAPI) -> OpenAIResponsesModel:
    client = AsyncOpenAI(api_key='test-key', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle)))
    return OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(openai_client=client))


def _line(custom_id: str, response: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> str:
    return json.dumps({'id': f'batch_req_{custom_id}', 'custom_id': custom_id, 'response': response, 'error': error})


@pytest.mark.parametrize(
    'openai_status,status',
    [
        ('validating', 'in_progress'),
        ('in_progress', 'in_progress'),
        ('finalizing', 'in_progress'),
        ('cancelling', 'canceling'),
        ('completed', 'ended'),
        ('failed', 'ended'),
        ('expired', 'ended'),
        ('cancelled', 'ended'),
    ],
)
async def test_batch_status_maps_every_openai_status(
    allow_model_requests: None, openai_status: str, status: str
) -> None:
    api = _FakeBatchAPI(batch=_batch_json('validating'))
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request())])
    assert batch.request_count == 1
    assert batch.created_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert json.loads(api.files['file-input']) == snapshot(
        {
            'custom_id': 'item',
            'method': 'POST',
            'url': '/v1/responses',
            'body': {'model': 'gpt-4.1-nano', 'input': [{'role': 'user', 'content': 'Hello'}]},
        }
    )

    api.batch = _batch_json(openai_status, request_counts={'total': 3, 'completed': 1, 'failed': 0})
    batch = await model.batch_status(batch)
    assert batch.status == status
    assert batch.request_count == 3
    assert batch.provider_details == {'status': openai_status, 'input_file_id': 'file-input'}


async def test_batch_cancel(allow_model_requests: None) -> None:
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request())])
    batch = await model.batch_cancel(batch)
    assert batch.status == 'canceling'
    assert api.requests[-1] == ('POST', '/batches/batch_1/cancel')


async def test_batch_api_errors_are_model_errors(allow_model_requests: None) -> None:
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request())])
    batch.id = 'batch_missing'
    with pytest.raises(ModelHTTPError) as exc_info:
        await model.batch_status(batch)
    assert exc_info.value.status_code == 404


async def test_batch_results_require_an_ended_batch(allow_model_requests: None) -> None:
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request())])
    with pytest.raises(UserError, match='has not ended yet'):
        await model.batch_results(batch)


async def test_batch_results_read_every_kind_of_line(allow_model_requests: None) -> None:
    """An expired batch reports what ran, what failed, and what it never got to."""
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    requests = [
        _request(custom_id) for custom_id in ('ok', 'http-error', 'line-error', 'expired', 'cancelled', 'unrun')
    ]
    batch = await model.batch_create([await model.batch_encode(request) for request in requests])

    response_body: dict[str, Any] = {
        'id': 'resp_1',
        'object': 'response',
        'created_at': 1_767_225_600,
        'model': 'gpt-4.1-nano-2025-04-14',
        'status': 'completed',
        'output': [
            {
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': 'Hi!', 'annotations': []}],
            }
        ],
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'usage': {
            'input_tokens': 10,
            'input_tokens_details': {'cached_tokens': 0},
            'output_tokens': 2,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': 12,
        },
    }
    error_body = {'error': {'message': 'Invalid input', 'type': 'invalid_request_error'}}
    api.files['file-output'] = _line('ok', {'status_code': 200, 'body': response_body}) + '\n'
    api.files['file-errors'] = '\n'.join(
        [
            _line('http-error', {'status_code': 400, 'body': error_body}),
            _line('line-error', error={'code': 'server_error', 'message': 'Something broke'}),
            _line('expired', error={'code': 'batch_expired', 'message': 'This request expired'}),
            _line('cancelled', error={'code': 'batch_cancelled', 'message': 'This request was cancelled'}),
        ]
    )
    api.batch = _batch_json('expired', output_file_id='file-output', error_file_id='file-errors')
    batch = await model.batch_status(batch)
    assert batch.provider_details == {
        'status': 'expired',
        'input_file_id': 'file-input',
        'output_file_id': 'file-output',
        'error_file_id': 'file-errors',
    }

    results = {result.custom_id: result for result in await model.batch_results(batch)}
    assert {custom_id: result.status for custom_id, result in results.items()} == {
        'ok': 'succeeded',
        'http-error': 'errored',
        'line-error': 'errored',
        'expired': 'expired',
        'cancelled': 'canceled',
        'unrun': 'expired',
    }
    by_id = {request.custom_id: request for request in requests}

    response = model.batch_decode(results['ok'], by_id['ok'])
    assert response.text == 'Hi!'
    assert response.provider_response_id == 'resp_1'
    assert response.provider_details is not None and response.provider_details['processing'] == 'batch'

    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(results['http-error'], by_id['http-error'])
    assert (exc_info.value.status_code, exc_info.value.body) == (400, error_body)

    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(results['line-error'], by_id['line-error'])
    assert (exc_info.value.status_code, exc_info.value.body) == (
        500,
        {'error': {'code': 'server_error', 'message': 'Something broke'}},
    )

    for custom_id, status in (('expired', 'expired'), ('cancelled', 'canceled'), ('unrun', 'expired')):
        with pytest.raises(ModelAPIError, match=f'The batch request was {status} before the model ran it'):
            model.batch_decode(results[custom_id], by_id[custom_id])


async def test_batch_results_of_a_failed_batch(allow_model_requests: None) -> None:
    """A batch whose input file failed validation ran none of its requests, and each one raises the batch's errors."""
    api = _FakeBatchAPI(batch=_batch_json('validating'))
    model = _fake_model(api)
    requests = [_request('one'), _request('two')]
    batch = await model.batch_create([await model.batch_encode(request) for request in requests])
    errors = {'object': 'list', 'data': [{'code': 'invalid_request', 'message': 'Bad line', 'line': 2}]}
    api.batch = _batch_json('failed', errors=errors)
    results = await model.batch_results(await model.batch_status(batch))
    assert [(result.custom_id, result.status) for result in results] == [('one', 'errored'), ('two', 'errored')]
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(results[0], requests[0])
    assert (exc_info.value.status_code, exc_info.value.body) == (
        400,
        {'errors': [{'code': 'invalid_request', 'message': 'Bad line', 'line': 2}]},
    )

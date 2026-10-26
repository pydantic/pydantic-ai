"""Unit tests for the OpenAI Batch API codec behind `OpenAIResponsesModel.batch_*`.

The round trip against the real API is recorded in `test_openai_batch_vcr.py`. These tests cover what that can't:
encoding (which makes no request), the gating of other providers, and batch outcomes the API can't be made to produce
on demand (an expired or failed batch, error lines), which are served by an in-process fake of the Batch API.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelServerError,
    ModelUnavailableError,
    UserError,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._batch_model import BatchPool
from pydantic_ai.models.batch import Batch, BatchRequest
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.tools import ToolDefinition
from tests.batch_utils import UNRUN_MESSAGES
from tests.conftest import try_import

with try_import() as imports_successful:
    import httpx2
    from openai import AsyncAzureOpenAI, AsyncOpenAI
    from openai.auth import WorkloadIdentity

    from pydantic_ai.models import _openai_batch as openai_batch
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
    ('model_factory', 'message'),
    [
        pytest.param(
            lambda: OpenAIChatModel('gpt-4.1-nano', provider=OpenAIProvider(api_key='k')),
            "`OpenAIChatModel` 'gpt-4.1-nano' has no batch API: batch it through the Responses API, with "
            '`OpenAIResponsesModel` or the `openai:` prefix.',
            id='openai',
        ),
        pytest.param(
            lambda: OpenAIChatModel('openai/gpt-4.1-nano', provider=OpenRouterProvider(api_key='k')),
            "`OpenAIChatModel` 'openai/gpt-4.1-nano' has no batch API.",
            id='openrouter',
        ),
    ],
)
def test_batching_a_chat_completions_model_points_to_the_responses_api(
    model_factory: Callable[[], OpenAIChatModel], message: str
) -> None:
    with pytest.raises(UserError) as exc_info:
        BatchPool().wrap(model_factory())
    assert str(exc_info.value) == message


@pytest.mark.parametrize(
    'model_factory,max_requests',
    [
        pytest.param(lambda: _model(), 50_000, id='openai'),
        pytest.param(
            lambda: OpenAIResponsesModel(
                'gpt-4.1-nano',
                provider=gateway_provider('openai', api_key='paig_test', base_url='https://gateway.example'),
            ),
            None,
            id='gateway',
        ),
        pytest.param(
            lambda: OpenAIResponsesModel(
                'gpt-4.1-nano',
                provider=gateway_provider('openai', api_key='paig_test', base_url='https://gateway.example'),
                profile=OpenAIModelProfile(openai_supports_batch_api=True),
            ),
            50_000,
            id='gateway-opt-in',
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
        pytest.param(
            lambda: OpenAIResponsesModel(
                'gpt-4.1-nano',
                provider=OpenAIProvider(api_key='k'),
                profile=OpenAIModelProfile(openai_responses_requires_streaming=True),
            ),
            None,
            id='requires-streaming-profile',
        ),
        pytest.param(lambda: OpenAIChatModel('gpt-4.1-nano', provider=OpenAIProvider(api_key='k')), None, id='chat'),
        pytest.param(
            lambda: OpenAIResponsesModel(
                'openai/gpt-4.1-nano',
                provider=OpenRouterProvider(api_key='k'),
                profile=OpenAIModelProfile(openai_supports_batch_api=True),
            ),
            50_000,
            id='openai-compatible-opt-in',
        ),
        pytest.param(
            lambda: OpenAIResponsesModel(
                'gpt-4.1-nano',
                provider=OpenAIProvider(api_key='k'),
                profile=OpenAIModelProfile(openai_supports_batch_api=False),
            ),
            None,
            id='openai-opt-out',
        ),
        pytest.param(
            lambda: OpenAIResponsesModel(
                'gpt-4.1-nano',
                provider=OpenAIProvider(
                    openai_client=AsyncAzureOpenAI(
                        azure_endpoint='https://test.openai.azure.com', api_version='v1', api_key='k'
                    )
                ),
                profile=OpenAIModelProfile(openai_supports_batch_api=True),
            ),
            50_000,
            id='azure-client-opt-in',
        ),
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


@pytest.mark.parametrize('header', ['Authorization', 'authorization', 'api-key'])
async def test_batch_group_is_the_credential_the_client_sends(header: str) -> None:
    """A credential in the client's default headers replaces its API key, so clients that differ only there don't share."""

    def model(credential: str) -> OpenAIResponsesModel:
        value = f'Bearer {credential}' if header.lower() == 'authorization' else credential
        return _model(api_key='placeholder', default_headers={header: value})

    async def group(model: OpenAIResponsesModel) -> str:
        return (await model.batch_encode(_request())).group

    tenant_a, tenant_b = model('tenant-a'), model('tenant-b')
    assert await group(tenant_a) != await group(tenant_b)
    assert tenant_a._batch_credential() != tenant_b._batch_credential()  # pyright: ignore[reportPrivateUsage]
    assert await group(tenant_a) == await group(model('tenant-a'))
    if header.lower() == 'authorization':
        # Overriding the API key's header with the same key is the same credential.
        assert await group(tenant_a) == await group(_model(api_key='tenant-a'))


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
    # Only submitting needs the same credential: the provider decides who may check on an existing batch.
    other = _model(api_key='first-key', model_name='gpt-5-mini')
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


async def test_batch_rejects_an_http_client_auth() -> None:
    """An HTTP client's `auth` can send any credential, which the client's headers don't show, so its group can't either."""
    client = AsyncOpenAI(api_key='placeholder', http_client=httpx2.AsyncClient(auth=('tenant-a', 'secret')))
    model = OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(openai_client=client))
    with pytest.raises(UserError, match='not an `auth` on its HTTP client'):
        await model.batch_encode(_request())


@dataclass
class _FakeBatchAPI:
    """Serves the Files and Batches endpoints a batch round trip uses."""

    batch: dict[str, Any]
    files: dict[str, str] = field(default_factory=dict[str, str])
    requests: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    delete_status: int = 200
    create_error: tuple[int, dict[str, Any]] | None = None
    response_bodies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path.removeprefix('/v1')
        self.requests.append((request.method, path))
        if request.method == 'DELETE' and path.startswith('/files/'):
            if self.delete_status != 200:
                return httpx2.Response(self.delete_status, json={'error': {'message': 'Nope', 'type': 'server_error'}})
            return httpx2.Response(200, json={'id': path.split('/')[2], 'object': 'file', 'deleted': True})
        if request.method == 'POST' and path == '/batches' and self.create_error is not None:
            status_code, body = self.create_error
            return httpx2.Response(status_code, json=body)
        if request.method == 'POST' and path == '/responses':
            # A request sent on its own, outside a batch.
            self.response_bodies.append(json.loads(request.read()))
            return httpx2.Response(200, json=_RESPONSE_BODY)
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


_RESPONSE_BODY: dict[str, Any] = {
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

    # The body is the error object, as the SDK unwraps it from an error response.
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(results['http-error'], by_id['http-error'])
    assert (exc_info.value.status_code, exc_info.value.body) == (400, error_body['error'])

    with pytest.raises(ModelServerError) as exc_info:
        model.batch_decode(results['line-error'], by_id['line-error'])
    assert isinstance(exc_info.value, ModelHTTPError)
    assert (exc_info.value.status_code, exc_info.value.body) == (
        500,
        {'code': 'server_error', 'message': 'Something broke'},
    )

    for custom_id, status in (('expired', 'expired'), ('cancelled', 'canceled'), ('unrun', 'expired')):
        with pytest.raises(ModelAPIError, match=UNRUN_MESSAGES[status]):
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
    # The input file is still needed to tell which requests the batch held, so it's kept.
    assert ('DELETE', '/files/file-input') not in api.requests
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(results[0], requests[0])
    assert (exc_info.value.status_code, exc_info.value.body, exc_info.value.provider_error_code) == (
        400,
        {'errors': [{'code': 'invalid_request', 'message': 'Bad line', 'line': 2}]},
        'invalid_request',
    )


async def test_failed_batch_gives_each_request_the_errors_of_its_line(allow_model_requests: None) -> None:
    """A batch that failed validation names the lines at fault: their requests get their own errors.

    The other requests didn't run because of those lines, so they get every error, which names them. An error of
    no line, about the batch as a whole, is every request's.
    """
    api = _FakeBatchAPI(batch=_batch_json('validating'))
    model = _fake_model(api)
    requests = [_request('one'), _request('two'), _request('three')]
    batch = await model.batch_create([await model.batch_encode(request) for request in requests])
    line_two = {'code': 'streaming_unsupported', 'message': 'Streaming is not supported.', 'line': 2}
    line_three = {'code': 'unsupported_parameter', 'message': 'Background is not supported.', 'line': 3}
    api.batch = _batch_json('failed', errors={'object': 'list', 'data': [line_two, line_three]})
    results = {result.custom_id: result for result in await model.batch_results(await model.batch_status(batch))}

    def body(custom_id: str) -> object:
        with pytest.raises(ModelHTTPError) as exc_info:
            model.batch_decode(results[custom_id], requests[0])
        return exc_info.value.body

    assert body('one') == {'errors': [line_two, line_three]}
    assert body('two') == {'errors': [line_two]}
    assert body('three') == {'errors': [line_three]}

    whole_batch = {'code': 'token_limit_exceeded', 'message': 'Enqueued token limit reached.'}
    api.batch = _batch_json('failed', errors={'object': 'list', 'data': [line_two, whole_batch]})
    results = {result.custom_id: result for result in await model.batch_results(await model.batch_status(batch))}
    assert body('one') == {'errors': [whole_batch]}
    assert body('two') == {'errors': [line_two, whole_batch]}


@pytest.mark.parametrize(
    'line,error_class,status_code',
    [
        pytest.param(
            {
                'response': {
                    'status_code': 400,
                    'body': {
                        'error': {
                            'message': "This model's maximum context length is 1047576 tokens.",
                            'type': 'invalid_request_error',
                            'param': 'input',
                            'code': 'context_length_exceeded',
                        }
                    },
                }
            },
            ModelContextWindowExceededError,
            400,
            id='context-window',
        ),
        pytest.param(
            {
                'response': {
                    'status_code': 429,
                    'body': {
                        'error': {'message': 'Rate limit reached.', 'type': 'requests', 'code': 'rate_limit_exceeded'}
                    },
                }
            },
            ModelRateLimitError,
            429,
            id='rate-limit',
        ),
        pytest.param(
            {
                'response': {
                    'status_code': 429,
                    'body': {
                        'error': {
                            'message': 'You exceeded your quota.',
                            'type': 'insufficient_quota',
                            'code': 'insufficient_quota',
                        }
                    },
                }
            },
            ModelQuotaExceededError,
            429,
            id='quota',
        ),
        pytest.param(
            {
                'response': {
                    'status_code': 503,
                    'body': {'error': {'message': 'The server is overloaded.', 'type': 'server_error'}},
                }
            },
            ModelOverloadedError,
            503,
            id='overloaded',
        ),
        pytest.param(
            {
                'response': {
                    'status_code': 503,
                    'body': {'error': {'message': 'Try again later.', 'type': 'server_error'}},
                }
            },
            ModelUnavailableError,
            503,
            id='unavailable',
        ),
        pytest.param(
            {'error': {'code': 'server_error', 'message': 'The server had an error.'}},
            ModelServerError,
            500,
            id='line-error',
        ),
    ],
)
async def test_batch_error_lines_raise_the_category_of_the_same_error_outside_a_batch(
    allow_model_requests: None, line: dict[str, Any], error_class: type[ModelAPIError], status_code: int
) -> None:
    """A failed request raises the error category its error would have outside a batch, as code catches it by that."""
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    request = _request('item')
    batch = await model.batch_create([await model.batch_encode(request)])
    api.files['file-errors'] = json.dumps(
        {'id': 'batch_req_1', 'custom_id': 'item', 'response': None, 'error': None, **line}
    )
    api.batch = _batch_json('completed', error_file_id='file-errors')
    [result] = await model.batch_results(await model.batch_status(batch))
    with pytest.raises(error_class) as exc_info:
        model.batch_decode(result, request)
    assert isinstance(exc_info.value, ModelHTTPError)
    assert exc_info.value.status_code == status_code


async def test_batch_error_for_an_unknown_previous_response(allow_model_requests: None) -> None:
    """A request continuing a response OpenAI doesn't have fails on its own, as it would outside a batch.

    The error line is the one the live API wrote for such a request; the batch's other requests still ran.
    """
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    requests = [_request('ok'), _request('e3_prev_id_bogus')]
    batch = await model.batch_create([await model.batch_encode(request) for request in requests])
    api.files['file-output'] = _line('ok', {'status_code': 200, 'body': _RESPONSE_BODY})
    error = {
        'message': "Previous response with id 'resp_doesnotexist123' not found.",
        'type': 'invalid_request_error',
        'param': 'previous_response_id',
        'code': 'previous_response_not_found',
    }
    api.files['file-errors'] = json.dumps(
        {'custom_id': 'e3_prev_id_bogus', 'response': {'status_code': 400, 'body': {'error': error}}, 'error': None}
    )
    api.batch = _batch_json('completed', output_file_id='file-output', error_file_id='file-errors')
    results = {result.custom_id: result for result in await model.batch_results(await model.batch_status(batch))}
    assert model.batch_decode(results['ok'], requests[0]).text == 'Hi!'
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(results['e3_prev_id_bogus'], requests[1])
    error_info = exc_info.value
    assert isinstance(error_info, ModelRequestRejectedError)
    assert (
        error_info.status_code,
        error_info.body,
        error_info.provider_error_code,
        error_info.provider_error_type,
    ) == (
        400,
        error,
        'previous_response_not_found',
        'invalid_request_error',
    )


async def test_batch_error_suggests_a_known_model_for_an_unknown_one(allow_model_requests: None) -> None:
    """A batch whose model OpenAI doesn't know suggests a close known model, as a request on its own does."""
    api = _FakeBatchAPI(batch=_batch_json('validating'))
    model = OpenAIResponsesModel(
        'gpt-4.1-nan',
        provider=OpenAIProvider(
            openai_client=AsyncOpenAI(
                api_key='k', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle))
            )
        ),
    )
    request = _request()
    batch = await model.batch_create([await model.batch_encode(request)])
    error = {'code': 'model_not_found', 'message': "The provided model 'gpt-4.1-nan' is not supported.", 'line': 1}
    api.batch = _batch_json('failed', errors={'object': 'list', 'data': [error]})
    [result] = await model.batch_results(await model.batch_status(batch))
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(result, request)
    assert exc_info.value.suggested_model_id == 'openai:gpt-4.1-nano'


async def test_batch_status_of_an_unknown_status_keeps_waiting(allow_model_requests: None) -> None:
    """OpenAI lists the statuses a batch ends in, so one added later is a batch that hasn't ended."""
    api = _FakeBatchAPI(batch=_batch_json('validating'))
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request())])
    api.batch = _batch_json('paused')
    batch = await model.batch_status(batch)
    assert batch.status == 'in_progress'
    assert batch.provider_details == {'status': 'paused', 'input_file_id': 'file-input'}


async def test_cancelled_batch_returns_what_it_ran(allow_model_requests: None) -> None:
    """A batch cancelled partway keeps the responses it finished; the requests it didn't get to were canceled."""
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    requests = [_request('done'), _request('cancelled'), _request('unrun')]
    batch = await model.batch_create([await model.batch_encode(request) for request in requests])
    api.files['file-output'] = _line('done', {'status_code': 200, 'body': _RESPONSE_BODY})
    api.files['file-errors'] = _line('cancelled', error={'code': 'batch_cancelled', 'message': 'Cancelled.'})
    api.batch = _batch_json('cancelled', output_file_id='file-output', error_file_id='file-errors')
    results = {result.custom_id: result for result in await model.batch_results(await model.batch_status(batch))}
    assert {custom_id: result.status for custom_id, result in results.items()} == {
        'done': 'succeeded',
        'cancelled': 'canceled',
        'unrun': 'canceled',
    }
    assert model.batch_decode(results['done'], requests[0]).text == 'Hi!'
    assert ('DELETE', '/files/file-input') not in api.requests


@pytest.mark.parametrize('delete_status', [200, 404, 500])
async def test_completed_batch_deletes_its_input_file(allow_model_requests: None, delete_status: int) -> None:
    """Once every request has run, the input file isn't needed; deleting it is best effort, so failing to is fine."""
    api = _FakeBatchAPI(batch=_batch_json('in_progress'), delete_status=delete_status)
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request('ok'))])
    api.files['file-output'] = _line('ok', {'status_code': 200, 'body': _RESPONSE_BODY})
    api.batch = _batch_json('completed', output_file_id='file-output')
    [result] = await model.batch_results(await model.batch_status(batch))
    assert result.status == 'succeeded'
    assert api.requests[-1] == ('DELETE', '/files/file-input')


async def test_batch_results_only_delete_the_input_file_this_process_uploaded(allow_model_requests: None) -> None:
    """A handle can come from a stored history a caller controls, so the input file it names is never deleted."""
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request('ok'))])
    api.files['file-output'] = _line('ok', {'status_code': 200, 'body': _RESPONSE_BODY})
    api.batch = _batch_json('completed', output_file_id='file-output')
    ended = await model.batch_status(batch)

    forged = replace(ended, provider_details={**(ended.provider_details or {}), 'input_file_id': 'file-someone-elses'})
    await model.batch_results(forged)
    # A batch this process didn't create, or one whose file it already deleted, deletes nothing.
    await model.batch_results(replace(forged, id='batch_from_history'))
    await model.batch_results(forged)
    assert [path for method, path in api.requests if method == 'DELETE'] == ['/files/file-input']


async def test_batch_results_delete_the_input_file_only_once_openai_says_the_batch_completed(
    allow_model_requests: None,
) -> None:
    """A handle claiming its batch completed doesn't make the input file of a batch still running deletable."""
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request('ok'))])
    api.files['file-output'] = _line('ok', {'status_code': 200, 'body': _RESPONSE_BODY})
    forged = replace(
        batch,
        status='ended',
        provider_details={**(batch.provider_details or {}), 'status': 'completed', 'output_file_id': 'file-output'},
    )

    await model.batch_results(forged)
    assert [path for method, path in api.requests if method == 'DELETE'] == []

    api.batch = _batch_json('completed', output_file_id='file-output')
    await model.batch_results(await model.batch_status(batch))
    assert [path for method, path in api.requests if method == 'DELETE'] == ['/files/file-input']


async def test_batch_results_dont_delete_a_file_another_account_uploaded_for_a_batch_with_the_same_id(
    allow_model_requests: None,
) -> None:
    """Batch IDs are only unique within an account, so another account's batch of the same ID isn't this upload's."""
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    await model.batch_create([await model.batch_encode(_request('ok'))])
    other_client = AsyncOpenAI(
        api_key='other-account-key', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle))
    )
    other = OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(openai_client=other_client))
    api.files['file-output'] = _line('ok', {'status_code': 200, 'body': _RESPONSE_BODY})
    api.batch = _batch_json('completed', output_file_id='file-output')
    handle = Batch(
        id='batch_1',
        model_name='gpt-4.1-nano',
        provider_name='openai',
        group=(await other.batch_encode(_request('ok'))).group,
        status='in_progress',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        request_count=1,
    )

    await other.batch_results(await other.batch_status(handle))
    assert [path for method, path in api.requests if method == 'DELETE'] == []


async def test_batch_forgets_the_oldest_uploaded_input_files(
    allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a bounded number of uploads are remembered; a forgotten batch's input file is left for OpenAI to delete."""
    monkeypatch.setattr('pydantic_ai.models._openai_batch._UPLOADED_INPUT_FILES_SIZE', 0)
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request('ok'))])
    api.files['file-output'] = _line('ok', {'status_code': 200, 'body': _RESPONSE_BODY})
    api.batch = _batch_json('completed', output_file_id='file-output')
    await model.batch_results(await model.batch_status(batch))
    assert [path for method, path in api.requests if method == 'DELETE'] == []


async def test_batch_size_limits(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request counts the bytes of its line in the input file, which OpenAI limits to 200 MB.

    A request too large for any batch is refused when it's encoded, and requests too large for one batch when
    they're submitted, before anything is uploaded.
    """
    api = _FakeBatchAPI(batch=_batch_json('validating'))
    model = _fake_model(api)
    assert model.batch_max_bytes == 200_000_000
    assert _model(model_name='gpt-4.1-nano').batch_max_bytes == 200_000_000
    assert OpenAIChatModel('gpt-4.1-nano', provider=OpenAIProvider(api_key='k')).batch_max_bytes is None
    assert WrapperModel(model).batch_max_bytes == 200_000_000

    encoded = [await model.batch_encode(_request(custom_id)) for custom_id in ('one', 'two-ü')]
    sizes = [model.batch_request_size(request) for request in encoded]
    assert [WrapperModel(model).batch_request_size(request) for request in encoded] == sizes
    await model.batch_create(encoded)
    # Lines are separated by newlines, which each request counts.
    assert len(api.files['file-input'].encode()) == sum(sizes) - 1

    monkeypatch.setattr(openai_batch, 'MAX_BATCH_BYTES', sizes[0])
    with pytest.raises(UserError, match=f'These requests take {sum(sizes):,} bytes, more than the {sizes[0]:,} bytes'):
        await model.batch_create(encoded)
    with pytest.raises(UserError, match=f'This request takes {sizes[1]:,} bytes, more than the {sizes[0]:,} bytes'):
        await model.batch_encode(_request('two-ü'))
    assert api.requests.count(('POST', '/files')) == 1


async def test_batch_request_is_the_request_sent_outside_a_batch(allow_model_requests: None) -> None:
    """A batch request carries exactly the body the request would send on its own: tools, output schema, reasoning."""
    api = _FakeBatchAPI(batch=_batch_json('validating'))
    model = _fake_model(api)
    messages: list[Any] = [ModelRequest(parts=[UserPromptPart('Hello')], instructions='Answer briefly')]
    settings = OpenAIResponsesModelSettings(
        max_tokens=100, openai_reasoning_effort='low', openai_reasoning_summary='auto', openai_store=False
    )
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='weather', parameters_json_schema={'type': 'object', 'properties': {}})],
        output_mode='native',
        output_object=OutputObjectDefinition(
            json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
            name='City',
        ),
    )
    await model.request(messages, settings, parameters)
    encoded = await model.batch_encode(
        BatchRequest(custom_id='item', messages=messages, model_settings=settings, model_request_parameters=parameters)
    )
    [sent] = api.response_bodies
    # A request on its own says it isn't streamed, which a batch, which never streams, leaves out.
    assert sent.pop('stream') is False
    assert json.loads(encoded.body) == sent
    assert 'background' not in sent


async def test_batch_usage_has_cached_and_reasoning_tokens(allow_model_requests: None) -> None:
    """A batched response reports cached and reasoning tokens as the request would on its own, priced at batch rates.

    Served by the fake: OpenAI doesn't reliably serve a batched request from the cache, so it can't be recorded.
    """
    api = _FakeBatchAPI(batch=_batch_json('in_progress'))
    model = _fake_model(api)
    request = _request('usage')
    batch = await model.batch_create([await model.batch_encode(request)])
    usage = {
        'input_tokens': 2000,
        'input_tokens_details': {'cached_tokens': 1536},
        'output_tokens': 50,
        'output_tokens_details': {'reasoning_tokens': 30},
        'total_tokens': 2050,
    }
    api.files['file-output'] = _line('usage', {'status_code': 200, 'body': {**_RESPONSE_BODY, 'usage': usage}})
    api.batch = _batch_json('completed', output_file_id='file-output')
    [result] = await model.batch_results(await model.batch_status(batch))
    response = model.batch_decode(result, request)
    assert (response.usage.input_tokens, response.usage.cache_read_tokens, response.usage.output_tokens) == (
        2000,
        1536,
        50,
    )
    assert response.usage.output_reasoning_tokens == 30
    standard = ModelResponse(
        parts=response.parts, usage=response.usage, model_name=response.model_name, provider_name=response.provider_name
    )
    # `genai-prices` has batch prices for uncached input and output; cached input costs the same either way.
    batch_price, standard_price = response.cost().model_price, standard.cost().model_price
    assert isinstance(batch_price.input_mtok, Decimal) and isinstance(standard_price.input_mtok, Decimal)
    assert isinstance(batch_price.output_mtok, Decimal) and isinstance(standard_price.output_mtok, Decimal)
    assert (batch_price.input_mtok * 2, batch_price.output_mtok * 2) == (
        standard_price.input_mtok,
        standard_price.output_mtok,
    )
    assert response.cost().total_price < standard.cost().total_price


@pytest.mark.parametrize('status_code, deleted', [(429, True), (400, True), (500, False)])
async def test_a_refused_batch_deletes_its_input_file(
    allow_model_requests: None, status_code: int, deleted: bool
) -> None:
    """Nothing reads the file of a batch OpenAI refused; after a server error, the batch may exist and read it."""
    error = {'error': {'message': 'Refused', 'type': 'invalid_request_error'}}
    api = _FakeBatchAPI(batch=_batch_json('validating'), create_error=(status_code, error))
    model = _fake_model(api)
    with pytest.raises(ModelHTTPError):
        await model.batch_create([await model.batch_encode(_request())])
    assert (('DELETE', '/files/file-input') in api.requests) is deleted


@pytest.mark.parametrize(
    'status_code,body,error_class',
    [
        (
            429,
            {'error': {'message': 'Too many batches.', 'type': 'requests', 'code': 'rate_limit_exceeded'}},
            ModelRateLimitError,
        ),
        (500, {'error': {'message': 'The server had an error.', 'type': 'server_error'}}, ModelServerError),
    ],
)
async def test_batch_create_errors_are_categorized_and_not_retried(
    allow_model_requests: None, status_code: int, body: dict[str, Any], error_class: type[ModelAPIError]
) -> None:
    """A failed submission raises its category, and isn't retried: OpenAI may have created the batch anyway."""
    api = _FakeBatchAPI(batch=_batch_json('validating'), create_error=(status_code, body))
    model = _fake_model(api)
    with pytest.raises(error_class):
        await model.batch_create([await model.batch_encode(_request())])
    assert api.requests.count(('POST', '/batches')) == 1

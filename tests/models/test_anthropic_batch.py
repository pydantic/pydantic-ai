"""The Anthropic Message Batches codec, for the parts that make no network call.

Submission and results are recorded in `test_anthropic_batch_vcr.py`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.capabilities import BatchModelRequests, NativeTool
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelAuthenticationError,
    ModelContextWindowExceededError,
    ModelHTTPError,
    ModelNotFoundError,
    ModelOverloadedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelServerError,
    UserError,
)
from pydantic_ai.messages import NativeToolCallPart, NativeToolReturnPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._batch_model import BatchPool
from pydantic_ai.models.batch import Batch, BatchRequest, BatchResult, EncodedBatchRequest
from pydantic_ai.models.test import TestModel
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.output import OutputObjectDefinition
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..batch_utils import UNRUN_MESSAGES
from ..conftest import try_import

with try_import() as imports_successful:
    import httpx2
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings, _batch as anthropic_batch
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.providers.gateway import gateway_provider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


def _request(custom_id: str = 'item', settings: ModelSettings | None = None, prompt: str = 'hello') -> BatchRequest:
    return BatchRequest(
        custom_id=custom_id, messages=[ModelRequest(parts=[UserPromptPart(prompt)])], model_settings=settings
    )


def _batch(group: str) -> Batch:
    return Batch(
        id='batch',
        model_name='claude-sonnet-4-5',
        provider_name='anthropic',
        group=group,
        status='ended',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        request_count=1,
    )


def test_anthropic_batch_api_is_only_on_the_claude_api_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('AWS_BEARER_TOKEN_BEDROCK', raising=False)
    direct = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))
    bedrock = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(
            anthropic_client=AsyncAnthropicBedrock(aws_region='us-east-1', aws_access_key='a', aws_secret_key='b')
        ),
    )

    assert direct.batch_max_requests == 100_000
    assert bedrock.batch_max_requests is None
    with pytest.raises(UserError, match=r'has no batch API on this platform: .*`anthropic:` provider'):
        BatchPool().wrap(bedrock)


def test_anthropic_batch_api_is_off_through_the_gateway() -> None:
    """The Pydantic AI Gateway doesn't serve Message Batches yet, though it reaches the Claude API's own client."""
    provider = gateway_provider('anthropic', api_key='paig_test', base_url='https://gateway.example')
    model = AnthropicModel('claude-sonnet-4-5', provider=provider)
    assert isinstance(provider.client, AsyncAnthropic)
    assert model.batch_max_requests is None
    with pytest.raises(UserError, match='has no batch API through the Pydantic AI Gateway yet'):
        BatchPool().wrap(model)

    opted_in = AnthropicModel(
        'claude-sonnet-4-5', provider=provider, profile=AnthropicModelProfile(anthropic_supports_message_batches=True)
    )
    assert opted_in.batch_max_requests == 100_000


async def test_anthropic_batch_api_follows_the_profile_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """`profile=` overrides the client-class default of `anthropic_supports_message_batches` in either direction."""
    monkeypatch.delenv('AWS_BEARER_TOKEN_BEDROCK', raising=False)
    opted_out = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(api_key='test-key'),
        profile=AnthropicModelProfile(anthropic_supports_message_batches=False),
    )
    assert opted_out.profile.get('anthropic_supports_message_batches') is False
    assert opted_out.batch_max_requests is None
    with pytest.raises(UserError, match=r'has no batch API\.$'):
        BatchPool().wrap(opted_out)

    opted_in = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(
            anthropic_client=AsyncAnthropicBedrock(aws_region='us-east-1', aws_access_key='a', aws_secret_key='b')
        ),
        profile=AnthropicModelProfile(anthropic_supports_message_batches=True),
    )
    assert opted_in.batch_max_requests == 100_000
    # Message Batches go through an `AsyncAnthropic` client, which a Bedrock client isn't.
    with pytest.raises(UserError, match='does not support Message Batches'):
        await opted_in.batch_encode(_request())


async def test_anthropic_batch_request_carries_the_messages_params_without_secrets() -> None:
    """The codec runs the same request builder as `AnthropicModel.request`, and its group never holds the key."""
    model = AnthropicModel(
        'claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=AsyncAnthropic(api_key='secret-key'))
    )

    request = await model.batch_encode(_request('item1', {'max_tokens': 100, 'stop_sequences': ['END']}, 'Hello'))

    body = json.loads(request.body)
    assert body['model'] == 'claude-sonnet-4-5'
    assert body['max_tokens'] == 100
    assert body['stop_sequences'] == ['END']
    assert body['messages'][0]['role'] == 'user'
    assert 'secret-key' not in request.group
    group = json.loads(request.group)
    assert group['betas'] == []
    assert 'api.anthropic.com' not in request.group


async def test_anthropic_batch_rejects_per_request_headers() -> None:
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with pytest.raises(UserError, match='do not support per-request `extra_headers`'):
        await model.batch_encode(_request('item1', {'extra_headers': {'X-Tenant': 'a'}}))


@pytest.mark.parametrize('status', ['expired', 'canceled'])
def test_anthropic_batch_request_that_never_ran_is_a_model_error(status: str) -> None:
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))

    with pytest.raises(ModelAPIError, match=UNRUN_MESSAGES[status]):
        model.batch_decode(
            BatchResult(custom_id='item1', status=status),  # pyright: ignore[reportArgumentType]
            _request('item1'),
        )


async def test_anthropic_batch_group_is_the_credential_the_client_sends() -> None:
    """A credential in the client's default headers replaces its API key, whatever the header's casing."""

    async def group(**client_kwargs: Any) -> str:
        client = AsyncAnthropic(api_key='placeholder', **client_kwargs)
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))
        return (await model.batch_encode(_request())).group

    tenant_a = await group(default_headers={'x-api-key': 'tenant-a'})
    assert tenant_a != await group(default_headers={'x-api-key': 'tenant-b'})
    assert tenant_a == await group(default_headers={'X-Api-Key': 'tenant-a'})
    assert await group(default_headers={'Authorization': 'Bearer a'}) != await group(
        default_headers={'authorization': 'Bearer b'}
    )


async def test_anthropic_batch_refuses_a_different_registered_client(allow_model_requests: None) -> None:
    first = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='first-key'))
    second = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='second-key'))
    request = await first.batch_encode(_request())
    with pytest.raises(UserError, match='does not match the batch request group'):
        await second.batch_create([request])
    # Only submitting needs the same credential: the provider decides who may check on an existing batch.
    elsewhere = AnthropicModel(
        'claude-sonnet-4-5', provider=AnthropicProvider(api_key='first-key', base_url='https://proxy.example')
    )
    with pytest.raises(UserError, match='does not match the batch request group'):
        await elsewhere.batch_status(_batch(request.group))
    with pytest.raises(UserError, match='does not match the batch request group'):
        await elsewhere.batch_results(_batch(request.group))
    with pytest.raises(UserError, match='does not match the batch request group'):
        await elsewhere.batch_cancel(_batch(request.group))
    other = await second.batch_encode(_request('other'))
    with pytest.raises(UserError, match='must share a `group`'):
        await first.batch_create([request, other])
    with pytest.raises(UserError, match='at least one request'):
        await first.batch_create([])


@pytest.mark.parametrize(
    'settings',
    [
        {'anthropic_speed': 'fast'},
        {'max_tokens': 0},
        {'timeout': 10},
        {'max_retries': 0},
        {'extra_body': {'speed': 'standard'}},
        {'extra_body': {'stream': True}},
    ],
)
async def test_anthropic_batch_rejects_synchronous_only_parameters(settings: ModelSettings) -> None:
    model = AnthropicModel('claude-opus-4-6', provider=AnthropicProvider(api_key='test'))
    with pytest.raises(UserError, match='not support'):
        await model.batch_encode(_request(settings=settings))


async def test_anthropic_batch_preserves_supported_settings() -> None:

    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test'))
    request = await model.batch_encode(
        BatchRequest(
            custom_id='item',
            messages=[ModelRequest(parts=[UserPromptPart('hello')], instructions='Answer briefly')],
            model_settings=AnthropicModelSettings(
                max_tokens=100,
                anthropic_metadata={'user_id': 'user'},
                anthropic_service_tier='standard_only',
                anthropic_cache=True,
                anthropic_cache_diagnostics=True,
                extra_body={'temperature': 0.5},
            ),
            model_request_parameters=ModelRequestParameters(
                function_tools=[
                    ToolDefinition(name='weather', parameters_json_schema={'type': 'object', 'properties': {}})
                ]
            ),
        )
    )
    body = json.loads(request.body)
    assert body['metadata'] == {'user_id': 'user'}
    assert body['service_tier'] == 'standard_only'
    assert body['diagnostics'] == {'previous_message_id': None}
    assert body['temperature'] == 0.5
    assert body['system'][0]['text'] == 'Answer briefly'
    assert body['tools'][0]['name'] == 'weather'


async def test_anthropic_batch_requires_static_credentials() -> None:
    client = AsyncAnthropic(api_key='')
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))
    assert model.batch_max_requests == 100_000
    BatchPool().wrap(model)
    with pytest.raises(UserError, match='require static Anthropic API credentials'):
        await model.batch_encode(BatchRequest(custom_id='item', messages=[]))


async def test_anthropic_batch_rejects_an_http_client_auth() -> None:
    """An HTTP client's `auth` can send any credential, which the client's headers don't show, so its group can't either."""
    client = AsyncAnthropic(api_key='placeholder', http_client=httpx2.AsyncClient(auth=('tenant-a', 'secret')))
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))
    with pytest.raises(UserError, match='not an `auth` on its HTTP client'):
        await model.batch_encode(_request())


async def test_direct_batch_methods_reject_a_bedrock_client(
    monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
) -> None:
    monkeypatch.delenv('AWS_BEARER_TOKEN_BEDROCK', raising=False)
    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(
            anthropic_client=AsyncAnthropicBedrock(aws_region='us-east-1', aws_access_key='a', aws_secret_key='b')
        ),
    )
    with pytest.raises(UserError, match='does not support Message Batches'):
        await model.batch_status(_batch('{}'))


def _message(content: list[dict[str, Any]], stop_reason: str = 'end_turn', **fields: Any) -> dict[str, Any]:
    return {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5-20250929',
        'content': content,
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {'input_tokens': 10, 'output_tokens': 5},
        **fields,
    }


def _batch_json(batch_id: str, status: str = 'ended') -> dict[str, Any]:
    return {
        'id': batch_id,
        'type': 'message_batch',
        'processing_status': status,
        'request_counts': {'processing': 0, 'succeeded': 1, 'errored': 0, 'canceled': 0, 'expired': 0},
        'created_at': '2026-01-01T00:00:00Z',
        'expires_at': '2026-01-02T00:00:00Z',
        'ended_at': None,
        'cancel_initiated_at': None,
        'archived_at': None,
        'results_url': f'https://api.anthropic.com/v1/messages/batches/{batch_id}/results',
    }


@dataclass
class _FakeBatchAPI:
    """Serves Message Batches, answering each batch's requests with the next of `results` for their `custom_id`.

    With `replies`, each batch's requests are answered with the next reply instead. Requests sent on their own,
    outside a batch, are answered with `message`.
    """

    results: list[dict[str, dict[str, Any]]] = field(default_factory=list[dict[str, dict[str, Any]]])
    replies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    status: str = 'ended'
    message: dict[str, Any] = field(default_factory=lambda: _message([{'type': 'text', 'text': 'Hi!'}]))
    created: list[tuple[list[dict[str, Any]], str | None]] = field(
        default_factory=list[tuple[list[dict[str, Any]], str | None]]
    )
    raw_creates: list[bytes] = field(default_factory=list[bytes])
    sent: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    create_status: int = 200

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path
        if path == '/v1/messages/batches' and self.create_status != 200:
            self.raw_creates.append(request.read())
            error = {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}}
            return httpx2.Response(self.create_status, json=error)
        if path == '/v1/messages':
            self.sent.append(json.loads(request.read()))
            return httpx2.Response(200, json=self.message)
        if path == '/v1/messages/batches':
            self.raw_creates.append(request.read())
            items = json.loads(request.read())['requests']
            self.created.append((items, request.headers.get('anthropic-beta')))
            if self.replies:
                reply = {'type': 'succeeded', 'message': self.replies[len(self.created) - 1]}
                self.results.append({item['custom_id']: reply for item in items})
            return httpx2.Response(200, json=_batch_json(f'batch_{len(self.created)}', 'in_progress'))
        batch_id = path.split('/')[4]
        if path.endswith('/results'):
            results = self.results[int(batch_id.removeprefix('batch_')) - 1]
            lines = [json.dumps({'custom_id': custom_id, 'result': result}) for custom_id, result in results.items()]
            return httpx2.Response(200, content='\n'.join(lines))
        return httpx2.Response(200, json=_batch_json(batch_id, self.status))


def _fake_model(api: _FakeBatchAPI, model_name: str = 'claude-sonnet-4-5') -> AnthropicModel:
    client = AsyncAnthropic(
        api_key='test-key', http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle))
    )
    return AnthropicModel(model_name, provider=AnthropicProvider(anthropic_client=client))


def _errored(error_type: str, message: str) -> BatchResult:
    error = {'type': 'error', 'error': {'type': error_type, 'message': message}, 'request_id': 'req_1'}
    return BatchResult(custom_id='item', status='errored', body=json.dumps({'type': 'errored', 'error': error}))


@pytest.mark.parametrize(
    'error_type,message,error_class,status_code',
    [
        ('invalid_request_error', 'max_tokens: 10000000 > 64000', ModelRequestRejectedError, 400),
        (
            'invalid_request_error',
            'prompt is too long: 210000 tokens > 200000 maximum',
            ModelContextWindowExceededError,
            400,
        ),
        (
            'invalid_request_error',
            'Your credit balance is too low to access the Anthropic API.',
            ModelQuotaExceededError,
            400,
        ),
        ('authentication_error', 'invalid x-api-key', ModelAuthenticationError, 401),
        ('billing_error', 'Billing issue.', ModelQuotaExceededError, 402),
        ('permission_error', 'Not allowed.', ModelAuthenticationError, 403),
        ('not_found_error', 'Not found.', ModelNotFoundError, 404),
        ('request_too_large', 'Request too large.', ModelHTTPError, 413),
        ('rate_limit_error', 'Rate limited.', ModelRateLimitError, 429),
        ('api_error', 'Internal server error.', ModelServerError, 500),
        ('timeout_error', 'Timed out.', ModelServerError, 504),
        ('overloaded_error', 'Overloaded.', ModelOverloadedError, 529),
        ('some_future_error', 'Something new.', ModelServerError, 500),
    ],
)
def test_anthropic_batch_errors_raise_the_category_of_the_same_error_outside_a_batch(
    error_type: str, message: str, error_class: type[ModelAPIError], status_code: int
) -> None:
    """An errored result carries the error body the Messages API would have answered with, and raises its category."""
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test-key'))
    with pytest.raises(error_class) as exc_info:
        model.batch_decode(_errored(error_type, message), _request())
    error = exc_info.value
    assert isinstance(error, ModelHTTPError)
    if error_class is ModelHTTPError:
        assert type(error) is ModelHTTPError
    assert (error.status_code, error.provider_error_type) == (status_code, error_type)
    assert error.body == {'type': 'error', 'error': {'type': error_type, 'message': message}, 'request_id': 'req_1'}


def test_anthropic_batch_error_suggests_a_known_model_for_an_unknown_one() -> None:
    model = AnthropicModel('claude-sonet-4-5', provider=AnthropicProvider(api_key='test-key'))
    with pytest.raises(ModelHTTPError) as exc_info:
        model.batch_decode(_errored('not_found_error', 'model: claude-sonet-4-5'), _request())
    assert exc_info.value.suggested_model_id == 'anthropic:claude-sonnet-4-5'


async def test_anthropic_batch_results_read_every_result_type(allow_model_requests: None) -> None:
    """A batch ends with each request succeeded, errored, canceled or expired, matched to it by `custom_id`.

    A refusal is a succeeded result, read as the same request's refusal would be outside a batch. A result for a
    request the batch didn't hold is passed on, for the caller to ignore.
    """
    refusal = _message(
        [], stop_reason='refusal', stop_details={'type': 'refusal', 'category': 'cyber', 'explanation': 'No.'}
    )
    api = _FakeBatchAPI(
        results=[
            {
                'expired': {'type': 'expired'},
                'refused': {'type': 'succeeded', 'message': refusal},
                'overloaded': {
                    'type': 'errored',
                    'error': {'type': 'error', 'error': {'type': 'overloaded_error', 'message': 'Overloaded'}},
                },
                'canceled': {'type': 'canceled'},
                'ok': {'type': 'succeeded', 'message': _message([{'type': 'text', 'text': 'Hi!'}])},
                'stray': {'type': 'canceled'},
            }
        ]
    )
    model = _fake_model(api)
    requests = {custom_id: _request(custom_id) for custom_id in ('ok', 'refused', 'overloaded', 'canceled', 'expired')}
    batch = await model.batch_create([await model.batch_encode(request) for request in requests.values()])
    results = {result.custom_id: result for result in await model.batch_results(await model.batch_status(batch))}
    assert {custom_id: result.status for custom_id, result in results.items()} == {
        'expired': 'expired',
        'refused': 'succeeded',
        'overloaded': 'errored',
        'canceled': 'canceled',
        'ok': 'succeeded',
        'stray': 'canceled',
    }

    assert model.batch_decode(results['ok'], requests['ok']).text == 'Hi!'
    refused = model.batch_decode(results['refused'], requests['refused'])
    assert (refused.parts, refused.finish_reason) == ([], 'content_filter')
    assert refused.provider_details is not None
    assert refused.provider_details['processing'] == 'batch'
    assert refused.provider_details['refusal_category'] == 'cyber'
    with pytest.raises(ModelOverloadedError):
        model.batch_decode(results['overloaded'], requests['overloaded'])
    for status in ('canceled', 'expired'):
        with pytest.raises(ModelAPIError, match=UNRUN_MESSAGES[status]):
            model.batch_decode(results[status], requests[status])


async def test_anthropic_batch_status_of_an_unknown_status_keeps_waiting(allow_model_requests: None) -> None:
    api = _FakeBatchAPI(status='paused')
    model = _fake_model(api)
    batch = await model.batch_create([await model.batch_encode(_request())])
    assert (await model.batch_status(batch)).status == 'in_progress'


@pytest.mark.parametrize('custom_id', ['', 'a' * 65, 'has space', 'dotted.id', 'slash/id', 'ünïcode'])
async def test_anthropic_batch_refuses_a_custom_id_it_cannot_take(allow_model_requests: None, custom_id: str) -> None:
    """Anthropic would reject the whole batch for one bad `custom_id`, so it's refused before anything is sent."""
    api = _FakeBatchAPI(results=[{}])
    model = _fake_model(api)
    encoded = [await model.batch_encode(_request('valid')), await model.batch_encode(_request(custom_id))]
    with pytest.raises(UserError, match='need `custom_id`s of 1 to 64 letters, digits, `_` or `-`'):
        await model.batch_create(encoded)
    assert api.created == []


@pytest.mark.parametrize('custom_id', ['a', 'A' * 64, 'run_1-retry', '0123456789abcdef0123456789abcdef-12'])
async def test_anthropic_batch_takes_a_valid_custom_id(allow_model_requests: None, custom_id: str) -> None:
    """Including the `<hash>-<n>` form `BatchModelRequests` generates."""
    api = _FakeBatchAPI(results=[{}])
    model = _fake_model(api)
    await model.batch_create([await model.batch_encode(_request(custom_id))])
    assert [item['custom_id'] for item in api.created[0][0]] == [custom_id]


@pytest.mark.parametrize(
    'client',
    [
        pytest.param(lambda: AsyncAnthropicVertex(project_id='p', region='us-east5', access_token='x'), id='vertex'),
        pytest.param(lambda: AsyncAnthropicFoundry(api_key='x', base_url='https://example.com'), id='foundry'),
        pytest.param(
            lambda: AsyncAnthropicBedrockMantle(aws_region='us-east-1', aws_access_key='x', aws_secret_key='y'),
            id='bedrock-mantle',
        ),
    ],
)
async def test_anthropic_batch_api_is_not_on_other_platforms(client: Any) -> None:
    """Vertex AI, Microsoft Foundry and Bedrock batch through their platforms' own jobs, not Message Batches."""
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client()))
    assert model.batch_max_requests is None
    assert model.batch_max_bytes is None
    with pytest.raises(UserError, match='has no batch API'):
        BatchPool().wrap(model)
    with pytest.raises(UserError, match='does not support Message Batches'):
        await model.batch_encode(_request())


async def test_anthropic_batch_size_limits(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A request counts the bytes it adds to the create request, which Anthropic limits to 256 MB.

    A request too large for any batch is refused when it's encoded, and requests too large for one batch when
    they're submitted.
    """
    api = _FakeBatchAPI(results=[{}])
    model = _fake_model(api)
    assert model.batch_max_bytes == 256_000_000 - len(b'{"requests":[]}')
    encoded = [await model.batch_encode(_request(custom_id, prompt='Héllo')) for custom_id in ('one', 'two')]
    sizes = [model.batch_request_size(request) for request in encoded]
    await model.batch_create(encoded)
    # Requests are separated by commas, which each counts, inside the envelope.
    assert len(api.raw_creates[0]) == len(b'{"requests":[]}') + sum(sizes) - 1

    monkeypatch.setattr(anthropic_batch, 'MAX_BATCH_BYTES', sizes[0])
    with pytest.raises(UserError, match=f'These requests take {sum(sizes):,} bytes, more than the {sizes[0]:,} bytes'):
        await model.batch_create(encoded)
    with pytest.raises(UserError, match=r'This request takes .* bytes, more than the'):
        await model.batch_encode(_request('three', prompt='A longer prompt'))
    assert len(api.created) == 1


async def test_anthropic_batches_are_grouped_by_beta_flags(allow_model_requests: None) -> None:
    """Beta flags are sent for a whole batch, so they're part of its group, and sent when it's created."""
    api = _FakeBatchAPI(results=[{}, {}])
    model = _fake_model(api)
    plain = await model.batch_encode(_request('plain'))
    beta = await model.batch_encode(
        _request('beta', AnthropicModelSettings(anthropic_betas=['interleaved-thinking-2025-05-14']))
    )
    assert json.loads(plain.group)['betas'] == []
    assert json.loads(beta.group)['betas'] == ['interleaved-thinking-2025-05-14']
    with pytest.raises(UserError, match='must share a `group`'):
        await model.batch_create([plain, beta])

    await model.batch_create([plain])
    await model.batch_create([beta])
    assert [
        (custom_ids, betas)
        for custom_ids, betas in ((tuple(r['custom_id'] for r in items), b) for items, b in api.created)
    ] == [
        (('plain',), 'message-batches-2024-09-24'),
        (('beta',), 'interleaved-thinking-2025-05-14,message-batches-2024-09-24'),
    ]


async def test_anthropic_batch_request_is_the_request_sent_outside_a_batch(allow_model_requests: None) -> None:
    """A batch request carries exactly the body the request would send on its own: tools, output schema, thinking."""
    api = _FakeBatchAPI()
    model = _fake_model(api, 'claude-sonnet-4-5')
    messages: list[Any] = [ModelRequest(parts=[UserPromptPart('Hello')], instructions='Answer briefly')]
    settings = AnthropicModelSettings(
        max_tokens=2048,
        anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024},
        anthropic_cache_instructions=True,
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
    [sent] = api.sent
    # A request on its own says it isn't streamed, which a batch, which never streams, leaves out.
    assert sent.pop('stream') is False
    assert json.loads(encoded.body) == sent


async def test_anthropic_paused_turn_continues_in_another_batch(allow_model_requests: None) -> None:
    """A server tool loop that pauses its turn comes back as a `pause_turn` result; the run continues the turn in the
    next batch, sending the paused content back, and the turn keeps both segments."""
    paused = _message(
        [
            {'type': 'server_tool_use', 'id': 'srvtoolu_1', 'name': 'web_search', 'input': {'query': 'pydantic'}},
            {
                'type': 'web_search_tool_result',
                'tool_use_id': 'srvtoolu_1',
                'content': [
                    {
                        'type': 'web_search_result',
                        'url': 'https://pydantic.dev',
                        'title': 'Pydantic',
                        'encrypted_content': 'abc',
                    }
                ],
            },
        ],
        stop_reason='pause_turn',
    )
    final = _message([{'type': 'text', 'text': 'Pydantic validates data.'}], id='msg_2')
    api = _FakeBatchAPI(replies=[paused, final])
    model = _fake_model(api)
    batching = BatchModelRequests(
        max_requests=1, poll_interval=timedelta(milliseconds=1), max_poll_interval=timedelta(milliseconds=1)
    )
    agent = Agent(model, capabilities=[batching, NativeTool(WebSearchTool())])
    result = await agent.run('What does Pydantic do?')

    assert result.output == 'Pydantic validates data.'
    [first], [second] = (items for items, _ in api.created)
    assert second['params']['messages'][-1] == snapshot(
        {
            'role': 'assistant',
            'content': [
                {'id': 'srvtoolu_1', 'type': 'server_tool_use', 'name': 'web_search', 'input': {'query': 'pydantic'}},
                {
                    'tool_use_id': 'srvtoolu_1',
                    'type': 'web_search_tool_result',
                    'content': [
                        {
                            'encrypted_content': 'abc',
                            'page_age': None,
                            'title': 'Pydantic',
                            'type': 'web_search_result',
                            'url': 'https://pydantic.dev',
                        }
                    ],
                },
            ],
        }
    )
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [type(part) for part in response.parts] == [NativeToolCallPart, NativeToolReturnPart, TextPart]
    assert first['params']['messages'] == [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'What does Pydantic do?'}]}
    ]


async def test_a_model_without_a_byte_limit_counts_a_request_as_its_id_and_body() -> None:
    """A batch API that limits only the number of requests has no byte limit; a request still has a size."""
    model = TestModel()
    assert model.batch_max_bytes is None
    encoded = EncodedBatchRequest(custom_id='ïd', group='group', body='{"a":"ü"}')
    assert model.batch_request_size(encoded) == len('ïd'.encode()) + len('{"a":"ü"}'.encode())


async def test_anthropic_batch_create_errors_are_categorized_and_not_retried(allow_model_requests: None) -> None:
    """A failed submission raises its category, and isn't retried: Anthropic may have created the batch anyway."""
    api = _FakeBatchAPI(create_status=529)
    model = _fake_model(api)
    with pytest.raises(ModelOverloadedError):
        await model.batch_create([await model.batch_encode(_request())])
    assert len(api.raw_creates) == 1


def test_anthropic_batch_credential_tells_clients_apart_without_secrets() -> None:
    """Checks and results of a batch are only shared between models that reach it with one credential."""

    def credential(api_key: str) -> str:
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=api_key))
        return model._batch_credential()  # pyright: ignore[reportPrivateUsage]

    assert credential('first-key') == credential('first-key') != credential('second-key')
    assert 'first-key' not in credential('first-key')

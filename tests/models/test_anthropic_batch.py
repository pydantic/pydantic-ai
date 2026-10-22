"""The Anthropic Message Batches codec, for the parts that make no network call.

Submission and results are recorded in `test_anthropic_batch_vcr.py`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from pydantic_ai import ModelRequest, UserPromptPart
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._batch_model import BatchPool
from pydantic_ai.models.batch import Batch, BatchRequest, BatchResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    import httpx2
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

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
    with pytest.raises(UserError, match='has no batch API'):
        BatchPool().wrap(bedrock)


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

    with pytest.raises(ModelAPIError, match=f'The batch request was {status} before the model ran it.'):
        model.batch_decode(
            BatchResult(custom_id='item1', status=status),  # pyright: ignore[reportArgumentType]
            _request('item1'),
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

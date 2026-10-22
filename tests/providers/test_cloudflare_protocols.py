"""Synthetic transport failures and shared client ownership are not reproducible with VCR."""

import httpx2
import pytest

from pydantic_ai.embeddings import infer_embedding_model
from pydantic_ai.exceptions import (
    ModelConnectionError,
    ModelHTTPError,
    ModelTimeoutError,
    UnexpectedModelBehavior,
    UserError,
)
from pydantic_ai.images import infer_image_generation_model
from pydantic_ai.models.cloudflare_clef import CloudflareClefModel
from pydantic_ai.providers.cloudflare import CloudflareProvider
from pydantic_ai.providers.system_one import SystemOneProvider

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.embeddings.cloudflare import CloudflareEmbeddingModel
    from pydantic_ai.models.cloudflare import CloudflareChatModel


@pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK not installed')
async def test_inference_rejects_other_provider():
    async with SystemOneProvider(api_key='token', base_url='https://example.com') as provider:
        with pytest.raises(UserError, match='Cloudflare embeddings require a `CloudflareProvider`'):
            infer_embedding_model('cloudflare:@cf/baai/bge-m3', provider_factory=lambda _: provider)
        with pytest.raises(UserError, match='Cloudflare images require a `CloudflareProvider`'):
            infer_image_generation_model(
                'cloudflare:@cf/black-forest-labs/flux-2-klein-4b', provider_factory=lambda _: provider
            )


@pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK not installed')
async def test_shared_client_lifecycle():
    provider = CloudflareProvider(api_key='test-token', account_id='test-account')
    chat = CloudflareChatModel('@cf/meta/llama-4-scout-17b-16e-instruct', provider=provider)
    assert chat.system == 'cloudflare'
    assert chat.base_url == 'https://api.cloudflare.com/client/v4/accounts/test-account/ai/v1/'
    clef = CloudflareClefModel('@cf/cloudflare/clef', provider=provider)
    async with clef:
        original_http = provider.client
        original_sdk = chat.client
        async with chat:
            assert clef.client is original_http
            assert not original_sdk.is_closed()
        assert not original_http.is_closed
    assert original_http.is_closed
    assert original_sdk.is_closed()
    async with chat:
        assert provider.client is not original_http
        assert chat.client is not original_sdk
        assert not chat.client.is_closed()
    assert provider.client.is_closed


@pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK not installed')
def test_default_protocol_providers(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('CLOUDFLARE_API_TOKEN', 'token')
    monkeypatch.setenv('CLOUDFLARE_ACCOUNT_ID', 'account')
    chat = CloudflareChatModel('@cf/meta/llama-4-scout-17b-16e-instruct')
    embedding = CloudflareEmbeddingModel('@cf/baai/bge-m3')
    assert chat.system == 'cloudflare'
    assert embedding.system == 'cloudflare'
    assert chat.model_name == '@cf/meta/llama-4-scout-17b-16e-instruct'
    assert embedding.model_name == '@cf/baai/bge-m3'
    assert chat.base_url == embedding.base_url


@pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK not installed')
@pytest.mark.usefixtures('allow_model_requests')
async def test_embeddings_refresh_closed_owned_client(monkeypatch: pytest.MonkeyPatch):
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                'data': [{'index': 0, 'embedding': [0.1, 0.2], 'object': 'embedding'}],
                'model': '@cf/baai/bge-m3',
                'object': 'list',
                'usage': {'prompt_tokens': 1, 'total_tokens': 1},
            },
        )

    monkeypatch.setattr(
        'pydantic_ai.providers.cloudflare.create_async_httpx2_client',
        lambda: httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    provider = CloudflareProvider(api_key='token', account_id='account')
    model = CloudflareEmbeddingModel('@cf/baai/bge-m3', provider=provider)
    original = provider.client
    await original.aclose()
    result = await model.embed('hello', input_type='query')
    assert result.embeddings == [[0.1, 0.2]]
    assert provider.client is not original
    await provider.client.aclose()


async def test_discovery_pagination_cache_and_task_filter():
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        assert request.headers['authorization'] == 'Bearer test-token'
        assert request.url.params['include_deprecated'] == 'false'
        task = request.url.params['task']
        names = ['@cf/cloudflare/clef', '@cf/cloudflare/clef-omni', '@cf/example/chat']
        return httpx2.Response(
            200,
            json={
                'result': [
                    *({'name': name, 'task': {'name': task}} for name in names),
                    {'name': '@cf/example/embedding', 'task': {'name': 'Text Embeddings'}},
                ]
                if request.url.params['page'] == '1'
                else [],
                'result_info': {'total_count': 101},
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        provider = CloudflareProvider(api_key='test-token', account_id='test-account', http_client=client)
        result = await provider.discover()
        assert result == ['@cf/cloudflare/clef', '@cf/example/chat']
        result.clear()
        assert await provider.discover() == ['@cf/cloudflare/clef', '@cf/example/chat']
        assert len(requests) == 2
        await provider.discover(refresh=True)
        assert len(requests) == 4
        await provider.discover(task='Text Embeddings')
        assert len(requests) == 6


@pytest.mark.parametrize(
    ('body', 'expected_count'),
    [
        ({'result': [{'name': '@cf/example/chat', 'task': {'name': 'Text Generation'}}]}, 1),
        ({'result': []}, 0),
        ({'result': [], 'result_info': {'total_count': 100}}, 0),
    ],
)
async def test_discovery_single_page(body: dict[str, object], expected_count: int):
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json=body))) as client:
        provider = CloudflareProvider(api_key='token', account_id='account', http_client=client)
        assert len(await provider.discover()) == expected_count


async def test_discovery_invalid_response():
    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda _: httpx2.Response(200, json={'result': 'bad'}))
    ) as client:
        provider = CloudflareProvider(api_key='token', account_id='account', http_client=client)
        with pytest.raises(UnexpectedModelBehavior, match='Invalid Cloudflare model catalog'):
            await provider.discover()


@pytest.mark.parametrize('json_body', [True, False])
async def test_native_http_errors(json_body: bool):
    response = (
        httpx2.Response(429, json={'errors': [{'message': 'rate limited'}]})
        if json_body
        else httpx2.Response(429, text='rate limited')
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _: response)) as client:
        provider = CloudflareProvider(api_key='token', account_id='account', http_client=client)
        with pytest.raises(ModelHTTPError) as error:
            await provider.discover()
        assert error.value.status_code == 429


@pytest.mark.parametrize('timeout', [True, False])
async def test_native_transport_errors(timeout: bool):
    def handler(_: httpx2.Request) -> httpx2.Response:
        if timeout:
            raise httpx2.ReadTimeout('timed out')
        raise httpx2.ConnectError('connection failed')

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        provider = CloudflareProvider(api_key='token', account_id='account', http_client=client)
        with pytest.raises(ModelTimeoutError if timeout else ModelConnectionError):
            await provider.discover()

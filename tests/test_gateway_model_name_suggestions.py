from __future__ import annotations

from typing import Any, Literal

import httpx
import httpx2
import pytest
from typing_extensions import assert_never

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.models import infer_model
from pydantic_ai.providers.gateway import gateway_provider

from .conftest import TestEnv, try_import

with try_import() as openai_imports:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

with try_import() as anthropic_imports:
    from pydantic_ai.models.anthropic import AnthropicModel

with try_import() as google_imports:
    from google.genai import errors

    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as bedrock_imports:
    from botocore.exceptions import ClientError

    from pydantic_ai.models.bedrock import BedrockConverseModel

with try_import() as groq_imports:
    from pydantic_ai.models.groq import GroqModel

pytestmark = pytest.mark.anyio


GatewayProvider = Literal['openai-responses', 'openai-chat', 'anthropic', 'google', 'groq']


@pytest.mark.parametrize(
    ('provider', 'model_name', 'expected_suggestion'),
    [
        pytest.param(
            'openai-responses',
            'gpt-5.2-proo',
            'gateway/openai:gpt-5.2-pro',
            id='openai-responses',
            marks=pytest.mark.skipif(not openai_imports(), reason='openai not installed'),
        ),
        pytest.param(
            'openai-chat',
            'gpt-5.2-proo',
            'gateway/openai:gpt-5.2-pro',
            id='openai-chat',
            marks=pytest.mark.skipif(not openai_imports(), reason='openai not installed'),
        ),
        pytest.param(
            'anthropic',
            'claude-sonet-4-5',
            'gateway/anthropic:claude-sonnet-4-5',
            id='anthropic',
            marks=pytest.mark.skipif(not anthropic_imports(), reason='anthropic not installed'),
        ),
        pytest.param(
            'google',
            'gemini-3.6-flahs',
            'gateway/google-cloud:gemini-3.6-flash',
            id='google',
            marks=pytest.mark.skipif(not google_imports(), reason='google not installed'),
        ),
        pytest.param(
            'groq',
            'llama-3.3-70b-versatlie',
            'gateway/groq:llama-3.3-70b-versatile',
            id='groq',
            marks=pytest.mark.skipif(not groq_imports(), reason='groq not installed'),
        ),
    ],
)
async def test_gateway_direct_model_suggests_gateway_model_id(
    provider: GatewayProvider,
    model_name: str,
    expected_suggestion: str,
    allow_model_requests: None,
    env: TestEnv,
    monkeypatch: pytest.MonkeyPatch,
):
    def not_found_body() -> dict[str, Any]:
        match provider:
            case 'openai-responses' | 'openai-chat' | 'groq':
                return {'error': {'code': 'model_not_found'}}
            case 'anthropic':
                return {'error': {'type': 'not_found_error', 'message': f'model: {model_name}'}}
            case 'google':
                return {
                    'error': {
                        'code': 404,
                        'message': f'models/{model_name} is not found for API version v1beta.',
                        'status': 'NOT_FOUND',
                    }
                }
            case _:
                assert_never(provider)

    def create_gateway_http_client() -> httpx.AsyncClient:
        async def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json=not_found_body())

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    def create_gateway_httpx2_client() -> httpx2.AsyncClient:
        async def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(404, json=not_found_body())

        return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    env.set('PYDANTIC_AI_GATEWAY_API_KEY', 'test-key')
    env.set('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.example.com/proxy')
    monkeypatch.setattr('pydantic_ai.providers.gateway.create_async_http_client', create_gateway_http_client)
    monkeypatch.setattr('pydantic_ai.providers.gateway.create_async_httpx2_client', create_gateway_httpx2_client)

    match provider:
        case 'openai-responses':
            model = OpenAIResponsesModel(model_name, provider='gateway')
        case 'openai-chat':
            model = OpenAIChatModel(model_name, provider='gateway')
        case 'anthropic':
            model = AnthropicModel(model_name, provider='gateway')
        case 'google':
            model = GoogleModel(model_name, provider='gateway')
        case 'groq':
            model = GroqModel(model_name, provider='gateway')
        case _:
            assert_never(provider)

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('hello')

    model_id_namespace = expected_suggestion.split(':', maxsplit=1)[0]
    expected_system = model_id_namespace.removeprefix('gateway/')
    assert model.system == expected_system
    assert model.model_id == f'{expected_system}:{model_name}'
    assert exc_info.value.suggested_model_id == expected_suggestion


@pytest.mark.skipif(not bedrock_imports(), reason='bedrock not installed')
async def test_gateway_bedrock_direct_model_suggests_gateway_model_id(
    allow_model_requests: None, env: TestEnv, monkeypatch: pytest.MonkeyPatch
):
    model_name = 'us.amazon.nova-micro-v1:O'
    env.set('PYDANTIC_AI_GATEWAY_API_KEY', 'test-key')
    env.set('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.example.com/proxy')
    model = BedrockConverseModel(model_name, provider='gateway')

    def raise_client_error(**_kwargs: object) -> None:
        raise ClientError(
            {
                'Error': {'Code': 'ValidationException', 'Message': 'The provided model identifier is invalid.'},
                'ResponseMetadata': {
                    'RequestId': '',
                    'HostId': '',
                    'HTTPStatusCode': 400,
                    'HTTPHeaders': {},
                    'RetryAttempts': 0,
                },
            },
            'Converse',
        )

    monkeypatch.setattr(model.client, 'converse', raise_client_error)

    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('hello')

    assert exc_info.value.suggested_model_id == 'gateway/bedrock:us.amazon.nova-premier-v1:0'


def _openai_model_not_found_client() -> httpx2.AsyncClient:
    async def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={'error': {'code': 'model_not_found'}})

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_inferred_gateway_model_suggests_gateway_model_id(
    allow_model_requests: None, env: TestEnv, monkeypatch: pytest.MonkeyPatch
):
    env.set('PYDANTIC_AI_GATEWAY_API_KEY', 'test-key')
    env.set('PYDANTIC_AI_GATEWAY_BASE_URL', 'https://gateway.example.com/proxy')
    monkeypatch.setattr('pydantic_ai.providers.gateway.create_async_httpx2_client', _openai_model_not_found_client)
    model = infer_model('gateway/openai:gpt-5.2-proo')

    assert model.system == 'openai'
    assert model.model_id == 'openai:gpt-5.2-proo'
    with pytest.raises(ModelHTTPError) as exc_info:
        await Agent(model).run('hello')

    assert exc_info.value.suggested_model_id == 'gateway/openai:gpt-5.2-pro'


@pytest.mark.skipif(not openai_imports(), reason='openai not installed')
async def test_explicit_gateway_provider_suggests_gateway_model_id(allow_model_requests: None):
    async with _openai_model_not_found_client() as http_client:
        provider = gateway_provider(
            'openai',
            api_key='test-key',
            base_url='https://gateway.example.com/proxy',
            http_client=http_client,
        )
        model = OpenAIResponsesModel('gpt-5.2-proo', provider=provider)

        assert provider.name == 'openai'
        assert model.system == 'openai'
        assert model.model_id == 'openai:gpt-5.2-proo'
        with pytest.raises(ModelHTTPError) as exc_info:
            await Agent(model).run('hello')

    assert exc_info.value.suggested_model_id == 'gateway/openai:gpt-5.2-pro'


@pytest.mark.skipif(not google_imports(), reason='google not installed')
@pytest.mark.parametrize(
    'error',
    [
        pytest.param('Resource not found.', id='non-object-error'),
        pytest.param(
            {
                'code': 404,
                'message': 'models/gemini-3.6-flahs is not supported for generateContent.',
                'status': 'NOT_FOUND',
            },
            id='unsupported-method',
        ),
    ],
)
async def test_google_not_found_without_model_resolution_does_not_suggest(
    allow_model_requests: None, error: str | dict[str, int | str]
):
    async def handler(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={'error': error})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        model = GoogleModel(
            'gemini-3.6-flahs',
            provider=GoogleProvider(api_key='test-key', http_client=http_client, base_url='https://google.example.com'),
        )
        with pytest.raises(ModelHTTPError) as exc_info:
            await Agent(model).run('hello')

    assert exc_info.value.suggested_model_id is None
    assert isinstance(exc_info.value.__cause__, errors.ClientError)

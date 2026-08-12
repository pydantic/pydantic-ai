from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import httpx2
import pytest
from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers import Provider
from pydantic_ai.providers.alibaba import AlibabaProvider
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider
from pydantic_ai.providers.cerebras import CerebrasProvider
from pydantic_ai.providers.crusoe import CrusoeProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.providers.fireworks import FireworksProvider
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.providers.litellm import LiteLLMProvider
from pydantic_ai.providers.moonshotai import MoonshotAIProvider
from pydantic_ai.providers.nebius import NebiusProvider
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.ovhcloud import OVHcloudProvider
from pydantic_ai.providers.sambanova import SambaNovaProvider
from pydantic_ai.providers.snowflake import SnowflakeProvider
from pydantic_ai.providers.together import TogetherProvider
from pydantic_ai.providers.vercel import VercelProvider
from pydantic_ai.providers.zai import ZaiProvider

ProviderFactory = Callable[[httpx.AsyncClient | httpx2.AsyncClient | None], Provider[AsyncOpenAI]]


@dataclass(frozen=True)
class Case:
    id: str
    create: ProviderFactory


CASES = [
    Case('openai', lambda http_client: OpenAIProvider(api_key='test', http_client=http_client)),
    Case(
        'azure',
        lambda http_client: AzureProvider(
            azure_endpoint='https://example-resource.openai.azure.com',
            api_version='2025-04-01-preview',
            api_key='test',
            http_client=http_client,
        ),
    ),
    Case(
        'bedrock-mantle',
        lambda http_client: BedrockMantleProvider(region_name='us-east-1', api_key='test', http_client=http_client),
    ),
    Case(
        'alibaba',
        lambda http_client: AlibabaProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'cerebras',
        lambda http_client: CerebrasProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'crusoe',
        lambda http_client: CrusoeProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case('deepseek', lambda http_client: DeepSeekProvider(api_key='test', http_client=http_client)),
    Case(
        'fireworks',
        lambda http_client: FireworksProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'heroku',
        lambda http_client: HerokuProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'litellm',
        lambda http_client: LiteLLMProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'moonshotai',
        lambda http_client: MoonshotAIProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'nebius',
        lambda http_client: NebiusProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'ollama',
        lambda http_client: OllamaProvider(
            base_url='http://localhost:11434/v1', api_key='test', http_client=http_client
        ),
    ),
    Case('openrouter', lambda http_client: OpenRouterProvider(api_key='test', http_client=http_client)),
    Case(
        'ovhcloud',
        lambda http_client: OVHcloudProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case('sambanova', lambda http_client: SambaNovaProvider(api_key='test', http_client=http_client)),
    Case('snowflake', lambda http_client: SnowflakeProvider(account='test', token='test', http_client=http_client)),
    Case(
        'together',
        lambda http_client: TogetherProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'vercel',
        lambda http_client: VercelProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
    Case(
        'zai',
        lambda http_client: ZaiProvider(
            api_key='test',
            http_client=http_client,  # pyright: ignore[reportArgumentType]
        ),
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_openai_compatible_providers_default_to_httpx2(case: Case) -> None:
    provider = case.create(None)

    assert isinstance(provider.client._client, httpx2.AsyncClient)  # pyright: ignore[reportPrivateUsage]
    async with provider:
        pass

    assert provider.client._client.is_closed  # pyright: ignore[reportPrivateUsage]


@pytest.mark.anyio
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_openai_compatible_providers_accept_httpx2_clients(case: Case) -> None:
    async with httpx2.AsyncClient() as http_client:
        provider = case.create(http_client)

        assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]
        async with provider:
            pass
        assert not http_client.is_closed


@pytest.mark.anyio
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_openai_compatible_providers_deprecate_legacy_httpx_clients(case: Case) -> None:
    async with httpx.AsyncClient() as http_client:
        with pytest.warns(
            PydanticAIDeprecationWarning,
            match=r'`httpx\.AsyncClient`.*removed in v3.*`httpx2\.AsyncClient`',
        ):
            provider = case.create(http_client)

        assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]
        async with provider:
            pass
        assert not http_client.is_closed


@pytest.mark.anyio
async def test_openai_compatible_provider_preserves_caller_owned_sdk_client() -> None:
    async with httpx2.AsyncClient() as http_client:
        openai_client = AsyncOpenAI(
            api_key='test',
            http_client=http_client,
        )
        provider = OpenAIProvider(openai_client=openai_client)

        assert provider.client is openai_client
        async with provider:
            pass
        assert not http_client.is_closed


def _chat_completion() -> dict[str, object]:
    return {
        'id': 'chatcmpl-test',
        'created': 1,
        'model': 'gpt-4o',
        'object': 'chat.completion',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'hello'}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


async def test_openai_provider_uses_caller_owned_httpx2_client(allow_model_requests: None) -> None:
    async def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == '/v1/chat/completions'
        assert json.loads(request.content)['messages'][0]['content'] == 'hello'
        return httpx2.Response(200, json=_chat_completion())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        provider = OpenAIProvider(api_key='test', http_client=http_client)
        settings: OpenAIChatModelSettings = {'timeout': httpx.Timeout(1)}
        result = await Agent(OpenAIChatModel('gpt-4o', provider=provider)).run('hello', model_settings=settings)

        assert result.output == 'hello'
        assert not http_client.is_closed


async def test_openai_provider_uses_deprecated_caller_owned_httpx_client(allow_model_requests: None) -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == '/v1/chat/completions'
        assert json.loads(request.content)['messages'][0]['content'] == 'hello'
        return httpx.Response(200, json=_chat_completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http_client:
        with pytest.warns(PydanticAIDeprecationWarning, match='httpx2.AsyncClient') as warnings:
            provider = OpenAIProvider(api_key='test', http_client=http_client)
        assert warnings[0].filename == __file__
        result = await Agent(OpenAIChatModel('gpt-4o', provider=provider)).run('hello')

        assert result.output == 'hello'
        assert not http_client.is_closed

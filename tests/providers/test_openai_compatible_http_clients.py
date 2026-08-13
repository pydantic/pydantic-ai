from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest

from pydantic_ai.providers import Provider

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

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


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

ProviderFactory = Callable[[], Provider['AsyncOpenAI']]
ProviderWithHTTPClientFactory = Callable[[httpx.AsyncClient], Provider['AsyncOpenAI']]


@dataclass(frozen=True)
class Case:
    id: str
    create: ProviderFactory
    create_with_http_client: ProviderWithHTTPClientFactory


CASES = [
    Case(
        'alibaba',
        lambda: AlibabaProvider(api_key='test'),
        lambda http_client: AlibabaProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'azure',
        lambda: AzureProvider(
            azure_endpoint='https://example-resource.openai.azure.com',
            api_version='2025-04-01-preview',
            api_key='test',
        ),
        lambda http_client: AzureProvider(
            azure_endpoint='https://example-resource.openai.azure.com',
            api_version='2025-04-01-preview',
            api_key='test',
            http_client=http_client,
        ),
    ),
    Case(
        'bedrock-mantle',
        lambda: BedrockMantleProvider(region_name='us-east-1', api_key='test'),
        lambda http_client: BedrockMantleProvider(region_name='us-east-1', api_key='test', http_client=http_client),
    ),
    Case(
        'cerebras',
        lambda: CerebrasProvider(api_key='test'),
        lambda http_client: CerebrasProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'crusoe',
        lambda: CrusoeProvider(api_key='test'),
        lambda http_client: CrusoeProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'deepseek',
        lambda: DeepSeekProvider(api_key='test'),
        lambda http_client: DeepSeekProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'fireworks',
        lambda: FireworksProvider(api_key='test'),
        lambda http_client: FireworksProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'heroku',
        lambda: HerokuProvider(api_key='test'),
        lambda http_client: HerokuProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'litellm',
        lambda: LiteLLMProvider(api_key='test'),
        lambda http_client: LiteLLMProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'moonshotai',
        lambda: MoonshotAIProvider(api_key='test'),
        lambda http_client: MoonshotAIProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'nebius',
        lambda: NebiusProvider(api_key='test'),
        lambda http_client: NebiusProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'ollama',
        lambda: OllamaProvider(base_url='http://localhost:11434/v1', api_key='test'),
        lambda http_client: OllamaProvider(
            base_url='http://localhost:11434/v1', api_key='test', http_client=http_client
        ),
    ),
    Case(
        'openai',
        lambda: OpenAIProvider(api_key='test'),
        lambda http_client: OpenAIProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'openrouter',
        lambda: OpenRouterProvider(api_key='test'),
        lambda http_client: OpenRouterProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'ovhcloud',
        lambda: OVHcloudProvider(api_key='test'),
        lambda http_client: OVHcloudProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'sambanova',
        lambda: SambaNovaProvider(api_key='test'),
        lambda http_client: SambaNovaProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'snowflake',
        lambda: SnowflakeProvider(account='test', token='test'),
        lambda http_client: SnowflakeProvider(account='test', token='test', http_client=http_client),
    ),
    Case(
        'together',
        lambda: TogetherProvider(api_key='test'),
        lambda http_client: TogetherProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'vercel',
        lambda: VercelProvider(api_key='test'),
        lambda http_client: VercelProvider(api_key='test', http_client=http_client),
    ),
    Case(
        'zai',
        lambda: ZaiProvider(api_key='test'),
        lambda http_client: ZaiProvider(api_key='test', http_client=http_client),
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_openai_compatible_provider_http_client_lifecycle(case: Case) -> None:
    provider = case.create()

    first_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
    assert isinstance(first_client, httpx.AsyncClient)
    async with provider:
        assert not first_client.is_closed
    assert first_client.is_closed

    async with provider:
        second_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert isinstance(second_client, httpx.AsyncClient)
        assert second_client is not first_client
        assert not second_client.is_closed
    assert second_client.is_closed


@pytest.mark.anyio
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_openai_compatible_provider_preserves_caller_owned_http_client(case: Case) -> None:
    async with httpx.AsyncClient() as http_client:
        provider = case.create_with_http_client(http_client)

        assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]
        async with provider:
            pass
        assert not http_client.is_closed


@pytest.mark.anyio
async def test_openai_compatible_provider_preserves_caller_owned_sdk_client() -> None:
    async with httpx.AsyncClient() as http_client:
        openai_client = AsyncOpenAI(api_key='test', http_client=http_client)
        provider = OpenAIProvider(openai_client=openai_client)

        assert provider.client is openai_client
        async with provider:
            pass
        assert not http_client.is_closed

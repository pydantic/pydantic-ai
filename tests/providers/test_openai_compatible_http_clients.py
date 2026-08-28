from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.providers import Provider

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
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
ProviderWithHTTPClientFactory = Callable[[httpx.AsyncClient | httpx2.AsyncClient], Provider['AsyncOpenAI']]


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

IMPORT_GUARD_CASES = [
    ('alibaba', 'use the Alibaba provider'),
    ('azure', 'use the Azure provider'),
    ('bedrock_mantle', 'use the Bedrock Mantle provider'),
    ('cerebras', 'use the Cerebras provider'),
    ('crusoe', 'use the Crusoe provider'),
    ('deepseek', 'use the DeepSeek provider'),
    ('fireworks', 'use the Fireworks AI provider'),
    ('heroku', 'use the Heroku provider'),
    ('litellm', 'use the LiteLLM provider'),
    ('moonshotai', 'use the MoonshotAI provider'),
    ('nebius', 'use the Nebius provider'),
    ('ollama', 'use the Ollama provider'),
    ('openai', 'use the OpenAI provider'),
    ('openrouter', 'use the OpenRouter provider'),
    ('ovhcloud', 'use OVHcloud AI Endpoints provider'),
    ('sambanova', 'use the SambaNova provider'),
    ('snowflake', 'use the Snowflake provider'),
    ('together', 'use the Together AI provider'),
    ('vercel', 'use the Vercel provider'),
    ('zai', 'use the Z.AI provider'),
]


@pytest.fixture(scope='module')
def import_guard_errors() -> dict[str, str | None]:
    # One subprocess for all providers: each interpreter spawn cold-imports pydantic_ai under
    # coverage (~7s in CI), so 20 per-provider subprocesses cost minutes while one costs seconds.
    # Each module still gets a fresh import: the guard fires at module import time, and no
    # provider module is imported twice.
    code = """
import builtins
import importlib
import json
import sys

original_import = builtins.__import__

def import_without_openai(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and (name == "openai" or name.startswith("openai.")):
        raise ImportError("simulated missing OpenAI SDK")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_openai

errors = {}
for module in sys.argv[1:]:
    try:
        importlib.import_module(f"pydantic_ai.providers.{module}")
    except ImportError as exc:
        errors[module] = str(exc)
    else:
        errors[module] = None
print(json.dumps(errors))
"""
    modules = [module for module, _ in IMPORT_GUARD_CASES]
    result = subprocess.run([sys.executable, '-c', code, *modules], text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.xdist_group(name='provider_import_guard')
@pytest.mark.parametrize(('module', 'error_hint'), IMPORT_GUARD_CASES)
def test_openai_compatible_provider_import_guard(
    module: str, error_hint: str, import_guard_errors: dict[str, str | None]
) -> None:
    error = import_guard_errors[module]
    assert error is not None, f'importing pydantic_ai.providers.{module} without openai did not raise ImportError'
    assert error_hint in error


@pytest.mark.anyio
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_openai_compatible_provider_http_client_lifecycle(case: Case) -> None:
    provider = case.create()

    first_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
    assert isinstance(first_client, httpx2.AsyncClient)
    async with provider:
        assert not first_client.is_closed
    assert first_client.is_closed

    async with provider:
        second_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert isinstance(second_client, httpx2.AsyncClient)
        assert second_client is not first_client
        assert not second_client.is_closed
    assert second_client.is_closed


@pytest.mark.anyio
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_openai_compatible_provider_preserves_caller_owned_httpx2_client(case: Case) -> None:
    async with httpx2.AsyncClient() as http_client:
        provider = case.create_with_http_client(http_client)

        assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]
        async with provider:
            pass
        assert not http_client.is_closed


@pytest.mark.anyio
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_openai_compatible_provider_deprecates_caller_owned_httpx_client(case: Case) -> None:
    async with httpx.AsyncClient() as http_client:
        with pytest.warns(
            PydanticAIDeprecationWarning,
            match=r'`httpx\.AsyncClient`.*removed in v3.*`httpx2\.AsyncClient`',
        ):
            provider = case.create_with_http_client(http_client)

        assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]
        async with provider:
            pass
        assert not http_client.is_closed


@pytest.mark.anyio
async def test_openai_compatible_provider_preserves_caller_owned_sdk_client() -> None:
    async with httpx2.AsyncClient() as http_client:
        openai_client = AsyncOpenAI(api_key='test', http_client=http_client)
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

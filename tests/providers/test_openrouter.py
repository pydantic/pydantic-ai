import re

import httpx
import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_openrouter_provider():
    provider = OpenRouterProvider(api_key='api-key')
    assert provider.name == 'openrouter'
    assert provider.base_url == 'https://openrouter.ai/api/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_openrouter_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OPENROUTER_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OPENROUTER_API_KEY` environment variable or pass it via `OpenRouterProvider(api_key=...)`'
            'to use the OpenRouter provider.'
        ),
    ):
        OpenRouterProvider()


def test_openrouter_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = OpenRouterProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_openrouter_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = OpenRouterProvider(openai_client=openai_client)
    assert provider.client == openai_client


async def test_openrouter_with_google_model(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenAIChatModel('google/gemini-2.0-flash-exp:free', provider=provider)
    agent = Agent(model, instructions='Be helpful.')
    response = await agent.run('Tell me a joke.')
    assert response.output == snapshot("""\
Why don't scientists trust atoms? \n\

Because they make up everything!
""")


def test_openrouter_provider_model_profile(mocker: MockerFixture):
    provider = OpenRouterProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.openrouter'
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    openai_model_profile_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)
    anthropic_model_profile_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    grok_model_profile_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    amazon_model_profile_mock = mocker.patch(f'{ns}.amazon_model_profile', wraps=amazon_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    moonshotai_model_profile_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)

    google_profile = provider.model_profile('google/gemini-2.5-pro-preview')
    google_model_profile_mock.assert_called_with('gemini-2.5-pro-preview')
    assert google_profile is not None
    assert google_profile.json_schema_transformer == GoogleJsonSchemaTransformer

    google_profile = provider.model_profile('google/gemma-3n-e4b-it:free')
    google_model_profile_mock.assert_called_with('gemma-3n-e4b-it')
    assert google_profile is not None
    assert google_profile.json_schema_transformer == GoogleJsonSchemaTransformer

    openai_profile = provider.model_profile('openai/o1-mini')
    openai_model_profile_mock.assert_called_with('o1-mini')
    assert openai_profile is not None
    assert openai_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    anthropic_profile = provider.model_profile('anthropic/claude-3.5-sonnet')
    anthropic_model_profile_mock.assert_called_with('claude-3.5-sonnet')
    assert anthropic_profile is not None
    assert anthropic_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistralai/mistral-large-2407')
    mistral_model_profile_mock.assert_called_with('mistral-large-2407')
    assert mistral_profile is not None
    assert mistral_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    qwen_profile = provider.model_profile('qwen/qwen-2.5-coder-32b')
    qwen_model_profile_mock.assert_called_with('qwen-2.5-coder-32b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    grok_profile = provider.model_profile('x-ai/grok-3')
    grok_model_profile_mock.assert_called_with('grok-3')
    assert grok_profile is not None
    assert grok_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    cohere_profile = provider.model_profile('cohere/command-a')
    cohere_model_profile_mock.assert_called_with('command-a')
    assert cohere_profile is not None
    assert cohere_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    amazon_profile = provider.model_profile('amazon/titan-text-express-v1')
    amazon_model_profile_mock.assert_called_with('titan-text-express-v1')
    assert amazon_profile is not None
    assert amazon_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    deepseek_profile = provider.model_profile('deepseek/deepseek-r1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    meta_profile = provider.model_profile('meta-llama/llama-4-maverick')
    meta_model_profile_mock.assert_called_with('llama-4-maverick')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    moonshotai_profile = provider.model_profile('moonshotai/kimi-k2')
    moonshotai_model_profile_mock.assert_called_with('kimi-k2')
    assert moonshotai_profile is not None
    assert moonshotai_profile.ignore_streamed_leading_whitespace is True
    assert moonshotai_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown/model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

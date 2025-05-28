import re

import httpx
import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import DEFAULT_PROFILE

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIModel
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
    model = OpenAIModel('google/gemini-2.0-flash-exp:free', provider=provider)
    agent = Agent(model, instructions='Be helpful.')
    response = await agent.run('Tell me a joke.')
    assert response.output == snapshot("""\
Why don't scientists trust atoms? \n\

Because they make up everything!
""")


def test_openrouter_provider_model_profile(mocker: MockerFixture):
    provider = OpenRouterProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.openrouter'
    google_model_profile = mocker.patch(f'{ns}.google_model_profile', return_value=DEFAULT_PROFILE)
    openai_model_profile = mocker.patch(f'{ns}.openai_model_profile', return_value=DEFAULT_PROFILE)
    anthropic_model_profile = mocker.patch(f'{ns}.anthropic_model_profile', return_value=DEFAULT_PROFILE)
    mistral_model_profile = mocker.patch(f'{ns}.mistral_model_profile', return_value=DEFAULT_PROFILE)
    qwen_model_profile = mocker.patch(f'{ns}.qwen_model_profile', return_value=DEFAULT_PROFILE)
    grok_model_profile = mocker.patch(f'{ns}.grok_model_profile', return_value=DEFAULT_PROFILE)
    cohere_model_profile = mocker.patch(f'{ns}.cohere_model_profile', return_value=DEFAULT_PROFILE)
    amazon_model_profile = mocker.patch(f'{ns}.amazon_model_profile', return_value=DEFAULT_PROFILE)
    deepseek_model_profile = mocker.patch(f'{ns}.deepseek_model_profile', return_value=DEFAULT_PROFILE)
    meta_model_profile = mocker.patch(f'{ns}.meta_model_profile', return_value=DEFAULT_PROFILE)

    google_profile = provider.model_profile('google/gemini-2.5-pro-preview')
    google_model_profile.assert_called_with('gemini-2.5-pro-preview')
    assert google_profile == google_model_profile.return_value

    google_profile = provider.model_profile('google/gemma-3n-e4b-it:free')
    google_model_profile.assert_called_with('gemma-3n-e4b-it')
    assert google_profile == google_model_profile.return_value

    openai_profile = provider.model_profile('openai/o1-mini')
    openai_model_profile.assert_called_with('o1-mini')
    assert openai_profile == openai_model_profile.return_value

    anthropic_profile = provider.model_profile('anthropic/claude-3.5-sonnet')
    anthropic_model_profile.assert_called_with('claude-3.5-sonnet')
    assert anthropic_profile == anthropic_model_profile.return_value

    mistral_profile = provider.model_profile('mistralai/mistral-large-2407')
    mistral_model_profile.assert_called_with('mistral-large-2407')
    assert mistral_profile == mistral_model_profile.return_value

    qwen_profile = provider.model_profile('qwen/qwen-2.5-coder-32b')
    qwen_model_profile.assert_called_with('qwen-2.5-coder-32b')
    assert qwen_profile == qwen_model_profile.return_value

    grok_profile = provider.model_profile('x-ai/grok-3')
    grok_model_profile.assert_called_with('grok-3')
    assert grok_profile == grok_model_profile.return_value

    cohere_profile = provider.model_profile('cohere/command-a')
    cohere_model_profile.assert_called_with('command-a')
    assert cohere_profile == cohere_model_profile.return_value

    amazon_profile = provider.model_profile('amazon/titan-text-express-v1')
    amazon_model_profile.assert_called_with('titan-text-express-v1')
    assert amazon_profile == amazon_model_profile.return_value

    deepseek_profile = provider.model_profile('deepseek/deepseek-r1')
    deepseek_model_profile.assert_called_with('deepseek-r1')
    assert deepseek_profile == deepseek_model_profile.return_value

    meta_profile = provider.model_profile('meta-llama/llama-4-maverick')
    meta_model_profile.assert_called_with('llama-4-maverick')
    assert meta_profile == meta_model_profile.return_value

    unknown_profile = provider.model_profile('unknown/model')
    assert unknown_profile is None

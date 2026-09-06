import re

import httpx2
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.aitokenking import AITokenKingProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_aitokenking_provider():
    provider = AITokenKingProvider(api_key='your-api-key')
    assert provider.name == 'aitokenking'
    assert provider.base_url == 'https://api.aitokenking.com.tw/api/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'


def test_aitokenking_provider_need_api_key(env: TestEnv) -> None:
    env.remove('AITOKENKING_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `AITOKENKING_API_KEY` environment variable or pass it via '
            '`AITokenKingProvider(api_key=...)` to use the AI Token King provider.'
        ),
    ):
        AITokenKingProvider()


def test_aitokenking_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='your-api-key')
    provider = AITokenKingProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_aitokenking_pass_http_client():
    http_client = httpx2.AsyncClient()
    provider = AITokenKingProvider(api_key='your-api-key', http_client=http_client)
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'


def test_aitokenking_model_profile(mocker: MockerFixture):
    provider = AITokenKingProvider(api_key='your-api-key')

    ns = 'pydantic_ai.providers.aitokenking'

    anthropic_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)
    openai_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    zai_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)

    profile = provider.model_profile('claude-opus-5')
    anthropic_mock.assert_called_with('claude-opus-5')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('deepseek-v3.2')
    deepseek_mock.assert_called_with('deepseek-v3.2')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('gemini-3.1-pro-preview')
    google_mock.assert_called_with('gemini-3.1-pro-preview')
    assert profile is not None

    profile = provider.model_profile('kimi-k2.7-code')
    moonshotai_mock.assert_called_with('kimi-k2.7-code')
    assert profile is not None

    profile = provider.model_profile('gpt-5.6-terra')
    openai_mock.assert_called_with('gpt-5.6-terra')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    qwen_profile = provider.model_profile('qwen3.8-max')
    qwen_mock.assert_called_with('qwen3.8-max')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    profile = provider.model_profile('glm-5.3')
    zai_mock.assert_called_with('glm-5.3')
    assert profile is not None

    # The model name is lower-cased before the prefix match, so a mixed-case id still resolves.
    provider.model_profile('Claude-Opus-5')
    anthropic_mock.assert_called_with('claude-opus-5')

    # Families the gateway serves that Pydantic AI has no profile for (MiniMax, Seed) must still return
    # a usable OpenAI-compatible profile rather than `None`.
    minimax_profile = provider.model_profile('minimax-m3')
    assert minimax_profile is not None
    assert minimax_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

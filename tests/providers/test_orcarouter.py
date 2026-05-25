import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.orcarouter import OrcaRouterProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_orcarouter_provider():
    provider = OrcaRouterProvider(api_key='api-key')
    assert provider.name == 'orcarouter'
    assert provider.base_url == 'https://api.orcarouter.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_orcarouter_provider_need_api_key(env: TestEnv) -> None:
    env.remove('ORCAROUTER_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `ORCAROUTER_API_KEY` environment variable or pass it via `OrcaRouterProvider(api_key=...)`'
            ' to use the OrcaRouter provider.'
        ),
    ):
        OrcaRouterProvider()


def test_orcarouter_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = OrcaRouterProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_orcarouter_with_http_client():
    http_client = httpx.AsyncClient()
    provider = OrcaRouterProvider(api_key='test-key', http_client=http_client)
    assert provider.client.api_key == 'test-key'
    assert str(provider.client.base_url) == 'https://api.orcarouter.ai/v1/'


def test_orcarouter_provider_model_profile(mocker: MockerFixture):
    provider = OrcaRouterProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.orcarouter'

    anthropic_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    grok_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    meta_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    mistral_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)
    openai_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)

    profile = provider.model_profile('openai/gpt-4o')
    openai_mock.assert_called_with('gpt-4o')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('anthropic/claude-sonnet-4-5')
    anthropic_mock.assert_called_with('claude-sonnet-4-5')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('deepseek/deepseek-chat')
    deepseek_mock.assert_called_with('deepseek-chat')
    assert profile is not None

    profile = provider.model_profile('google/gemini-1.5-pro')
    google_mock.assert_called_with('gemini-1.5-pro')
    assert profile is not None

    profile = provider.model_profile('meta-llama/llama-3-70b')
    meta_mock.assert_called_with('llama-3-70b')
    assert profile is not None

    profile = provider.model_profile('mistralai/mistral-large')
    mistral_mock.assert_called_with('mistral-large')
    assert profile is not None

    profile = provider.model_profile('moonshotai/kimi-k2-0711-preview')
    moonshotai_mock.assert_called_with('kimi-k2-0711-preview')
    assert profile is not None

    profile = provider.model_profile('qwen/qwen-2.5-72b-instruct')
    qwen_mock.assert_called_with('qwen-2.5-72b-instruct')
    assert profile is not None

    profile = provider.model_profile('x-ai/grok-beta')
    grok_mock.assert_called_with('grok-beta')
    assert profile is not None


def test_orcarouter_provider_model_name_without_slash():
    profile = OrcaRouterProvider.model_profile('some-direct-model')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer


def test_orcarouter_provider_unknown_vendor():
    profile = OrcaRouterProvider.model_profile('unknown/some-model')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

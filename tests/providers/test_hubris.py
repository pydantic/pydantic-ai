import re

import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.hubris import HubrisProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_hubris_provider():
    provider = HubrisProvider(api_key='api-key')
    assert provider.name == 'hubris'
    assert provider.base_url == 'https://api.hubris.pw/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_hubris_provider_need_api_key(env: TestEnv) -> None:
    env.remove('HUBRIS_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `HUBRIS_API_KEY` environment variable or pass it via '
            '`HubrisProvider(api_key=...)` to use the Hubris provider.'
        ),
    ):
        HubrisProvider()


def test_hubris_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = HubrisProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_hubris_provider_model_profile(mocker: MockerFixture):
    provider = HubrisProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.hubris'

    anthropic_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    openai_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    grok_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    zai_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)

    # Hubris model ids keep the vendor prefix of the gateway catalog
    profile = provider.model_profile('anthropic/claude-sonnet-5')
    anthropic_mock.assert_called_with('claude-sonnet-5')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('openai/gpt-5.6-luna')
    openai_mock.assert_called_with('gpt-5.6-luna')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    google_profile = provider.model_profile('google/gemini-3.7-flash')
    google_mock.assert_called_with('gemini-3.7-flash')
    assert google_profile is not None
    assert google_profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    profile = provider.model_profile('x-ai/grok-4.6')
    grok_mock.assert_called_with('grok-4.6')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('z-ai/glm-5.3-flash')
    zai_mock.assert_called_with('glm-5.3-flash')
    assert profile is not None

    qwen_profile = provider.model_profile('qwen/qwen3.8-max-0902')
    qwen_mock.assert_called_with('qwen3.8-max-0902')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    profile = provider.model_profile('deepseek/deepseek-v4-flash-0731')
    deepseek_mock.assert_called_with('deepseek-v4-flash-0731')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('moonshotai/kimi-k3')
    moonshotai_mock.assert_called_with('kimi-k3')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    # Unknown vendor prefix falls back to the plain OpenAI profile
    unknown_profile = provider.model_profile('unknown-provider/unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


def test_hubris_provider_model_name_without_slash():
    profile = HubrisProvider.model_profile('invalid-model-name')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

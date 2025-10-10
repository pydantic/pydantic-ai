import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.nebius import NebiusProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_nebius_provider():
    provider = NebiusProvider(api_key='api-key')
    assert provider.name == 'nebius'
    assert provider.base_url == 'https://api.studio.nebius.com/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_nebius_provider_need_api_key(env: TestEnv) -> None:
    env.remove('NEBIUS_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `NEBIUS_API_KEY` environment variable or pass it via '
            '`NebiusProvider(api_key=...)` to use the Nebius AI Studio provider.'
        ),
    ):
        NebiusProvider()


def test_nebius_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = NebiusProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_nebius_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = NebiusProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_nebius_provider_model_profile(mocker: MockerFixture):
    provider = NebiusProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.nebius'

    # Mock all profile functions
    meta_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    harmony_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)
    mistral_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)

    # Test meta provider
    meta_profile = provider.model_profile('meta-llama/Llama-3.3-70B-Instruct')
    meta_mock.assert_called_with('llama-3.3-70b-instruct')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test deepseek provider
    profile = provider.model_profile('deepseek-ai/DeepSeek-R1-0528')
    deepseek_mock.assert_called_with('deepseek-r1-0528')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test qwen provider
    qwen_profile = provider.model_profile('Qwen/Qwen3-30B-A3B')
    qwen_mock.assert_called_with('qwen3-30b-a3b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test google provider
    google_profile = provider.model_profile('google/gemma-2-2b-it')
    google_mock.assert_called_with('gemma-2-2b-it')
    assert google_profile is not None
    assert google_profile.json_schema_transformer == GoogleJsonSchemaTransformer

    # Test harmony (for openai gpt-oss) provider
    profile = provider.model_profile('openai/gpt-oss-120b')
    harmony_mock.assert_called_with('gpt-oss-120b')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test mistral provider
    profile = provider.model_profile('mistralai/Devstral-Small-2505')
    mistral_mock.assert_called_with('devstral-small-2505')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test moonshotai provider
    moonshotai_profile = provider.model_profile('moonshotai/Kimi-K2-Instruct')
    moonshotai_mock.assert_called_with('kimi-k2-instruct')
    assert moonshotai_profile is not None
    assert moonshotai_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test unknown provider
    unknown_profile = provider.model_profile('unknown-provider/unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer


def test_nebius_provider_invalid_model_name():
    provider = NebiusProvider(api_key='api-key')

    with pytest.raises(UserError, match="Model name must be in 'provider/model' format"):
        provider.model_profile('invalid-model-name')

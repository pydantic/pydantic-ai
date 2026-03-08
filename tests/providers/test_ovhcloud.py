import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.ovhcloud import OVHcloudProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_ovhcloud_provider():
    provider = OVHcloudProvider(api_key='your-api-key')
    assert provider.name == 'ovhcloud'
    assert provider.base_url == 'https://oai.endpoints.kepler.ai.cloud.ovh.net/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'


def test_ovhcloud_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OVHCLOUD_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OVHCLOUD_API_KEY` environment variable or pass it via '
            '`OVHcloudProvider(api_key=...)` to use OVHcloud AI Endpoints provider.'
        ),
    ):
        OVHcloudProvider()


def test_ovhcloud_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='your-api-key')
    provider = OVHcloudProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_ovhcloud_pass_http_client():
    http_client = httpx.AsyncClient()
    provider = OVHcloudProvider(api_key='your-api-key', http_client=http_client)
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'


def test_ovhcloud_model_profile(mocker: MockerFixture):
    provider = OVHcloudProvider(api_key='your-api-key')

    ns = 'pydantic_ai.providers.ovhcloud'

    # Mock all profile functions
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    harmony_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)
    meta_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    mistral_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)

    # Test deepseek provider
    profile = provider.model_profile('DeepSeek-R1-Distill-Llama-70B')
    deepseek_mock.assert_called_with('deepseek-r1-distill-llama-70b')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test harmony (for openai gpt-oss) provider
    profile = provider.model_profile('gpt-oss-120b')
    harmony_mock.assert_called_with('gpt-oss-120b')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test meta provider
    meta_profile = provider.model_profile('Llama-3.3-70B-Instruct')
    meta_mock.assert_called_with('llama-3.3-70b-instruct')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test mistral provider
    profile = provider.model_profile('Mistral-Small-3.2-24B-Instruct-2506')
    mistral_mock.assert_called_with('mistral-small-3.2-24b-instruct-2506')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test qwen provider
    qwen_profile = provider.model_profile('Qwen3-32B')
    qwen_mock.assert_called_with('qwen3-32b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test unknown provider
    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

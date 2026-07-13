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
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.crusoe import CrusoeProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_crusoe_provider():
    provider = CrusoeProvider(api_key='api-key')
    assert provider.name == 'crusoe'
    assert provider.base_url == 'https://managed-inference-api-proxy.crusoecloud.com/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_crusoe_provider_need_api_key(env: TestEnv) -> None:
    env.remove('CRUSOE_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `CRUSOE_API_KEY` environment variable or pass it via '
            '`CrusoeProvider(api_key=...)` to use the Crusoe provider.'
        ),
    ):
        CrusoeProvider()


def test_crusoe_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = CrusoeProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_crusoe_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = CrusoeProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_crusoe_provider_model_profile(mocker: MockerFixture):
    provider = CrusoeProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.crusoe'

    # Mock all profile functions
    meta_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    harmony_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)

    # Test meta provider
    meta_profile = provider.model_profile('meta-llama/Llama-3.3-70B-Instruct')
    meta_mock.assert_called_with('llama-3.3-70b-instruct')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    # Test deepseek provider
    profile = provider.model_profile('deepseek-ai/DeepSeek-R1-0528')
    deepseek_mock.assert_called_with('deepseek-r1-0528')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    # Test qwen provider
    qwen_profile = provider.model_profile('Qwen/Qwen3-235B-A22B-Instruct-2507')
    qwen_mock.assert_called_with('qwen3-235b-a22b-instruct-2507')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    # Test google provider
    google_profile = provider.model_profile('google/gemma-3-12b-it')
    google_mock.assert_called_with('gemma-3-12b-it')
    assert google_profile is not None
    assert google_profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    # Test harmony (for openai gpt-oss) provider
    profile = provider.model_profile('openai/gpt-oss-120b')
    harmony_mock.assert_called_with('gpt-oss-120b')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    # Test moonshotai provider
    moonshotai_profile = provider.model_profile('moonshotai/Kimi-K2-Thinking')
    moonshotai_mock.assert_called_with('kimi-k2-thinking')
    assert moonshotai_profile is not None
    assert moonshotai_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    # Test unknown provider
    unknown_profile = provider.model_profile('unknown-provider/unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


def test_crusoe_provider_model_name_without_slash():
    profile = CrusoeProvider.model_profile('invalid-model-name')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.together import TogetherProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_together_provider():
    provider = TogetherProvider(api_key='api-key')
    assert provider.name == 'together'
    assert provider.base_url == 'https://api.together.xyz/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_together_provider_need_api_key(env: TestEnv) -> None:
    env.remove('TOGETHER_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `TOGETHER_API_KEY` environment variable or pass it via `TogetherProvider(api_key=...)`'
            'to use the Together AI provider.'
        ),
    ):
        TogetherProvider()


def test_together_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = TogetherProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_together_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = TogetherProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_together_provider_model_profile(mocker: MockerFixture):
    provider = TogetherProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.together'
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)

    deepseek_profile = provider.model_profile('deepseek-ai/DeepSeek-R1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    meta_profile = provider.model_profile('meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8')
    meta_model_profile_mock.assert_called_with('llama-4-maverick-17b-128e-instruct-fp8')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    qwen_profile = provider.model_profile('Qwen/QwQ-32B')
    qwen_model_profile_mock.assert_called_with('qwq-32b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistralai/Devstral-Small-2505')
    mistral_model_profile_mock.assert_called_with('devstral-small-2505')
    assert mistral_profile is not None
    assert mistral_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    google_profile = provider.model_profile('google/gemma-3-27b-it')
    google_model_profile_mock.assert_called_with('gemma-3-27b-it')
    assert google_profile is not None
    assert google_profile.json_schema_transformer == GoogleJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown/model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

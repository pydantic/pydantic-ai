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

    from pydantic_ai.providers.fireworks import FireworksProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_fireworks_provider():
    provider = FireworksProvider(api_key='api-key')
    assert provider.name == 'fireworks'
    assert provider.base_url == 'https://api.fireworks.ai/inference/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_fireworks_provider_need_api_key(env: TestEnv) -> None:
    env.remove('FIREWORKS_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `FIREWORKS_API_KEY` environment variable or pass it via `FireworksProvider(api_key=...)`'
            'to use the Fireworks AI provider.'
        ),
    ):
        FireworksProvider()


def test_fireworks_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = FireworksProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_fireworks_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = FireworksProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_fireworks_provider_model_profile(mocker: MockerFixture):
    provider = FireworksProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.fireworks'
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)

    deepseek_profile = provider.model_profile('accounts/fireworks/models/deepseek-v3')
    deepseek_model_profile_mock.assert_called_with('deepseek-v3')
    assert deepseek_profile is not None
    assert deepseek_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    meta_profile = provider.model_profile('accounts/fireworks/models/llama4-maverick-instruct-basic')
    meta_model_profile_mock.assert_called_with('llama4-maverick-instruct-basic')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    qwen_profile = provider.model_profile('accounts/fireworks/models/qwen3-235b-a22b')
    qwen_model_profile_mock.assert_called_with('qwen3-235b-a22b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    mistral_profile = provider.model_profile('accounts/fireworks/models/mistral-small-24b-instruct-2501')
    mistral_model_profile_mock.assert_called_with('mistral-small-24b-instruct-2501')
    assert mistral_profile is not None
    assert mistral_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    google_profile = provider.model_profile('accounts/fireworks/models/gemma-7b-it')
    google_model_profile_mock.assert_called_with('gemma-7b-it')
    assert google_profile is not None
    assert google_profile.json_schema_transformer == GoogleJsonSchemaTransformer

    unknown_profile = provider.model_profile('accounts/fireworks/models/unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

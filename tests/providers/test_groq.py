from __future__ import annotations as _annotations

import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from groq import AsyncGroq

    from pydantic_ai.providers.groq import GroqProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='groq not installed')


def test_groq_provider():
    provider = GroqProvider(api_key='api-key')
    assert provider.name == 'groq'
    assert provider.base_url == 'https://api.groq.com'
    assert isinstance(provider.client, AsyncGroq)
    assert provider.client.api_key == 'api-key'


def test_groq_provider_need_api_key(env: TestEnv) -> None:
    env.remove('GROQ_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
            'to use the Groq provider.'
        ),
    ):
        GroqProvider()


def test_groq_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = GroqProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_groq_provider_pass_groq_client() -> None:
    groq_client = AsyncGroq(api_key='api-key')
    provider = GroqProvider(groq_client=groq_client)
    assert provider.client == groq_client


def test_groq_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test with environment variable for base_url
    monkeypatch.setenv('GROQ_BASE_URL', 'https://custom.groq.com/v1')
    provider = GroqProvider(api_key='api-key')
    assert provider.base_url == 'https://custom.groq.com/v1'


def test_groq_provider_model_profile(mocker: MockerFixture):
    groq_client = AsyncGroq(api_key='api-key')
    provider = GroqProvider(groq_client=groq_client)

    ns = 'pydantic_ai.providers.groq'
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    moonshotai_model_profile_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)

    meta_profile = provider.model_profile('meta-llama/Llama-Guard-4-12B')
    meta_model_profile_mock.assert_called_with('llama-guard-4-12b')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('llama-3.3-70b-versatile')
    meta_model_profile_mock.assert_called_with('llama-3.3-70b-versatile')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    google_profile = provider.model_profile('gemma2-9b-it')
    google_model_profile_mock.assert_called_with('gemma2-9b-it')
    assert google_profile is not None
    assert google_profile.json_schema_transformer == GoogleJsonSchemaTransformer

    deepseek_profile = provider.model_profile('deepseek-r1-distill-llama-70b')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1-distill-llama-70b')
    assert deepseek_profile is None

    mistral_profile = provider.model_profile('mistral-saba-24b')
    mistral_model_profile_mock.assert_called_with('mistral-saba-24b')
    assert mistral_profile is None

    qwen_profile = provider.model_profile('qwen-qwq-32b')
    qwen_model_profile_mock.assert_called_with('qwen-qwq-32b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # MoonshotAI model should remove the "moonshotai/" prefix before passing to profile
    moonshotai_profile = provider.model_profile('moonshotai/kimi-k2-instruct')
    moonshotai_model_profile_mock.assert_called_with('kimi-k2-instruct')
    assert moonshotai_profile is None

    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is None

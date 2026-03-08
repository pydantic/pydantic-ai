import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.ollama import OllamaProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_ollama_provider():
    provider = OllamaProvider(base_url='http://localhost:11434/v1/')
    assert provider.name == 'ollama'
    assert provider.base_url == 'http://localhost:11434/v1/'
    assert isinstance(provider.client, openai.AsyncOpenAI)


def test_ollama_provider_need_base_url(env: TestEnv) -> None:
    env.remove('OLLAMA_BASE_URL')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OLLAMA_BASE_URL` environment variable or pass it via `OllamaProvider(base_url=...)`'
            'to use the Ollama provider.'
        ),
    ):
        OllamaProvider()


def test_ollama_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = OllamaProvider(http_client=http_client, base_url='http://localhost:11434/v1/')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_ollama_provider_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(base_url='http://localhost:11434/v1/', api_key='test')
    provider = OllamaProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_ollama_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test with environment variable for base_url
    monkeypatch.setenv('OLLAMA_BASE_URL', 'https://custom.ollama.com/v1/')
    provider = OllamaProvider()
    assert provider.base_url == 'https://custom.ollama.com/v1/'


def test_ollama_provider_model_profile(mocker: MockerFixture):
    provider = OllamaProvider(base_url='http://localhost:11434/v1/')

    ns = 'pydantic_ai.providers.ollama'
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    harmony_model_profile_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)

    meta_profile = provider.model_profile('llama3.2')
    meta_model_profile_mock.assert_called_with('llama3.2')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    google_profile = provider.model_profile('gemma3')
    google_model_profile_mock.assert_called_with('gemma3')
    assert google_profile is not None
    assert google_profile.json_schema_transformer == GoogleJsonSchemaTransformer

    deepseek_profile = provider.model_profile('deepseek-r1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistral-small')
    mistral_model_profile_mock.assert_called_with('mistral-small')
    assert mistral_profile is not None
    assert mistral_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    qwen_profile = provider.model_profile('qwen3')
    qwen_model_profile_mock.assert_called_with('qwen3')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert qwen_profile.ignore_streamed_leading_whitespace is True

    qwen_profile = provider.model_profile('qwq')
    qwen_model_profile_mock.assert_called_with('qwq')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer
    assert qwen_profile.ignore_streamed_leading_whitespace is True

    cohere_profile = provider.model_profile('command-r')
    cohere_model_profile_mock.assert_called_with('command-r')
    assert cohere_profile is not None
    assert cohere_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    harmony_profile = provider.model_profile('gpt-oss')
    harmony_model_profile_mock.assert_called_with('gpt-oss')
    assert harmony_profile is not None
    assert harmony_profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert harmony_profile.ignore_streamed_leading_whitespace is True

    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

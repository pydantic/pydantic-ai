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

    from pydantic_ai.providers.vllm import VLLMProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_vllm_provider() -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    assert provider.name == 'vllm'
    assert provider.base_url == 'http://localhost:8000/v1/'
    assert isinstance(provider.client, openai.AsyncOpenAI)


def test_vllm_provider_need_base_url(env: TestEnv) -> None:
    env.remove('VLLM_BASE_URL')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `VLLM_BASE_URL` environment variable or pass it via `VLLMProvider(base_url=...)`'
            ' to use the vLLM provider.'
        ),
    ):
        VLLMProvider()


def test_vllm_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VLLM_BASE_URL', 'https://custom.vllm.com/v1/')
    provider = VLLMProvider()
    assert provider.base_url == 'https://custom.vllm.com/v1/'


def test_vllm_provider_api_key_placeholder(env: TestEnv) -> None:
    # vLLM servers do not always require an API key, so a non-empty placeholder is used.
    env.remove('VLLM_API_KEY')
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    assert provider.client.api_key == 'api-key-not-set'


def test_vllm_provider_with_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('VLLM_BASE_URL', 'http://localhost:8000/v1/')
    monkeypatch.setenv('VLLM_API_KEY', 'env-key')
    provider = VLLMProvider()
    assert provider.client.api_key == 'env-key'


def test_vllm_provider_explicit_api_key() -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/', api_key='explicit-key')
    assert provider.client.api_key == 'explicit-key'


def test_vllm_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = VLLMProvider(http_client=http_client, base_url='http://localhost:8000/v1/')
    assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]


def test_vllm_provider_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(base_url='http://localhost:8000/v1/', api_key='test')
    provider = VLLMProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_vllm_provider_openai_client_is_exclusive() -> None:
    openai_client = openai.AsyncOpenAI(base_url='http://localhost:8000/v1/', api_key='test')
    with pytest.raises(AssertionError, match='Cannot provide both `openai_client` and `base_url`'):
        VLLMProvider(openai_client=openai_client, base_url='http://localhost:8000/v1/')


async def test_vllm_provider_recreates_closed_owned_client() -> None:
    # Re-entering the provider after its owned HTTP client was closed recreates the client
    # via the factory, exercising `_set_http_client`.
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    owned = provider._own_http_client  # pyright: ignore[reportPrivateUsage]
    assert owned is not None
    await owned.aclose()
    async with provider:
        new_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
        assert new_client is not owned
        assert not new_client.is_closed


def test_vllm_provider_model_profile(mocker: MockerFixture) -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')

    ns = 'pydantic_ai.providers.vllm'
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    harmony_model_profile_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)

    meta_profile = provider.model_profile('llama-3-8b')
    meta_model_profile_mock.assert_called_with('llama-3-8b')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    google_profile = provider.model_profile('gemma-3')
    google_model_profile_mock.assert_called_with('gemma-3')
    assert google_profile is not None
    assert google_profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    qwen_profile = provider.model_profile('qwen3')
    qwen_model_profile_mock.assert_called_with('qwen3')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    # `qwq` maps to the qwen profile too.
    qwq_profile = provider.model_profile('qwq-32b')
    qwen_model_profile_mock.assert_called_with('qwq-32b')
    assert qwq_profile is not None
    assert qwq_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    deepseek_profile = provider.model_profile('deepseek-r1')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistral-small')
    mistral_model_profile_mock.assert_called_with('mistral-small')
    assert mistral_profile is not None
    assert mistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    cohere_profile = provider.model_profile('command-r')
    cohere_model_profile_mock.assert_called_with('command-r')
    assert cohere_profile is not None
    assert cohere_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    harmony_profile = provider.model_profile('gpt-oss-20b')
    harmony_model_profile_mock.assert_called_with('gpt-oss-20b')
    assert harmony_profile is not None
    assert harmony_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


def test_vllm_provider_model_profile_is_case_insensitive(mocker: MockerFixture) -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    qwen_model_profile_mock = mocker.patch('pydantic_ai.providers.vllm.qwen_model_profile', wraps=qwen_model_profile)
    provider.model_profile('Qwen3-32B')
    qwen_model_profile_mock.assert_called_once_with('qwen3-32b')


def test_vllm_provider_profile_disables_multiple_system_messages() -> None:
    # Regression guard for #5812: strict chat templates served by vLLM (recent Qwen, Mistral, Gemma)
    # reject more than one leading system message, so every vLLM profile disables them and the
    # provider does not support strict tool definitions (issue #4116).
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    for model in ('llama-3-8b', 'qwen3', 'mistral-small', 'gemma-3', 'command-r', 'gpt-oss-20b', 'unknown-model'):
        profile = provider.model_profile(model)
        assert profile is not None
        assert profile.get('openai_chat_supports_multiple_system_messages', True) is False
        assert profile.get('openai_supports_strict_tool_definition', True) is False


def test_vllm_provider_base_profile_flags() -> None:
    provider = VLLMProvider(base_url='http://localhost:8000/v1/')
    profile = provider.model_profile('unknown-model')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert profile.get('openai_chat_thinking_field', None) == 'reasoning'
    assert profile.get('supports_json_schema_output', False) is True
    assert profile.get('supports_json_object_output', False) is True

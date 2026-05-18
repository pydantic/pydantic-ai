import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.sambanova import SambaNovaProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_sambanova_provider_init():
    provider = SambaNovaProvider(api_key='test-key')
    assert provider.name == 'sambanova'
    assert provider.base_url == 'https://api.sambanova.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'test-key'


def test_sambanova_provider_env_key(env: TestEnv):
    env.set('SAMBANOVA_API_KEY', 'env-key')
    provider = SambaNovaProvider()
    assert provider.client.api_key == 'env-key'


def test_sambanova_provider_missing_key(env: TestEnv):
    env.remove('SAMBANOVA_API_KEY')
    with pytest.raises(UserError, match='Set the `SAMBANOVA_API_KEY`'):
        SambaNovaProvider()


def test_infer_provider(env: TestEnv):
    # infer_provider instantiates the class, so we need an env var or it raises UserError
    env.set('SAMBANOVA_API_KEY', 'key')
    provider = infer_provider('sambanova')
    assert isinstance(provider, SambaNovaProvider)


def test_meta_llama_profile():
    provider = SambaNovaProvider(api_key='key')
    # Meta Llama model -> expect meta profile wrapped in OpenAI compatibility
    profile = provider.model_profile('Meta-Llama-3.1-8B-Instruct')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_deepseek_profile():
    provider = SambaNovaProvider(api_key='key')
    # DeepSeek model -> expect deepseek profile wrapped in OpenAI compatibility
    profile = provider.model_profile('DeepSeek-R1-0528')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_qwen_profile():
    provider = SambaNovaProvider(api_key='key')
    # Qwen model -> expect qwen profile wrapped in OpenAI compatibility
    profile = provider.model_profile('Qwen3-32B')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_llama4_profile():
    provider = SambaNovaProvider(api_key='key')
    # Llama 4 model -> expect meta profile wrapped in OpenAI compatibility
    profile = provider.model_profile('Llama-4-Maverick-17B-128E-Instruct')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_mistral_profile():
    provider = SambaNovaProvider(api_key='key')
    # Mistral-based model -> expect mistral profile wrapped in OpenAI compatibility
    profile = provider.model_profile('E5-Mistral-7B-Instruct')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_unknown_model_profile():
    provider = SambaNovaProvider(api_key='key')
    # Unknown model -> should return OpenAI compatibility wrapper with None base profile
    profile = provider.model_profile('unknown-model')
    assert isinstance(profile, OpenAIModelProfile)


def test_sambanova_provider_with_openai_client():
    client = openai.AsyncOpenAI(api_key='foo', base_url='https://api.sambanova.ai/v1')
    provider = SambaNovaProvider(openai_client=client)
    assert provider.client is client


def test_sambanova_provider_with_http_client():
    http_client = httpx.AsyncClient()
    provider = SambaNovaProvider(api_key='foo', http_client=http_client)
    assert provider.client.api_key == 'foo'
    # The line `self._client = AsyncOpenAI(..., http_client=http_client)` is executed,
    # which is enough for coverage.


def test_sambanova_provider_custom_base_url():
    provider = SambaNovaProvider(api_key='test-key', base_url='https://custom.endpoint.com/v1')
    assert provider.base_url == 'https://custom.endpoint.com/v1'
    assert str(provider.client.base_url).rstrip('/') == 'https://custom.endpoint.com/v1'


def test_sambanova_provider_env_base_url(env: TestEnv):
    env.set('SAMBANOVA_API_KEY', 'key')
    env.set('SAMBANOVA_BASE_URL', 'https://env.endpoint.com/v1')
    provider = SambaNovaProvider()
    assert provider.base_url == 'https://env.endpoint.com/v1'

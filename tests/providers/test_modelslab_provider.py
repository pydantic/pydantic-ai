import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.providers.modelslab import ModelsLabProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')

DEFAULT_BASE_URL = 'https://modelslab.com/api/uncensored-chat/v1'


def test_modelslab_provider_init(env: TestEnv) -> None:
    env.set('MODELSLAB_API_KEY', 'test-key')
    provider = ModelsLabProvider()
    assert provider.name == 'modelslab'
    assert DEFAULT_BASE_URL in provider.base_url
    assert isinstance(provider.client, AsyncOpenAI)


def test_modelslab_provider_custom_base_url(env: TestEnv) -> None:
    custom_url = 'https://custom.modelslab.example.com/v1'
    provider = ModelsLabProvider(api_key='test-key', base_url=custom_url)
    assert custom_url in provider.base_url


def test_modelslab_provider_with_openai_client(env: TestEnv) -> None:
    client = AsyncOpenAI(api_key='test-key', base_url=DEFAULT_BASE_URL)
    provider = ModelsLabProvider(openai_client=client)
    assert provider.client is client


def test_modelslab_provider_with_http_client(env: TestEnv) -> None:
    env.set('MODELSLAB_API_KEY', 'test-key')
    http_client = httpx.AsyncClient()
    provider = ModelsLabProvider(http_client=http_client)
    assert provider.client.http_client is http_client


def test_modelslab_provider_no_api_key(env: TestEnv) -> None:
    env.remove('MODELSLAB_API_KEY')
    with pytest.raises(UserError, match='MODELSLAB_API_KEY'):
        ModelsLabProvider()


def test_modelslab_model_profile_llama(env: TestEnv) -> None:
    env.set('MODELSLAB_API_KEY', 'test-key')
    provider = ModelsLabProvider()
    profile = provider.model_profile('meta-llama/Meta-Llama-3-8B-Instruct')
    assert isinstance(profile, OpenAIModelProfile)


def test_modelslab_model_profile_deepseek(env: TestEnv) -> None:
    env.set('MODELSLAB_API_KEY', 'test-key')
    provider = ModelsLabProvider()
    profile = provider.model_profile('deepseek-chat')
    assert isinstance(profile, OpenAIModelProfile)


def test_modelslab_model_profile_mistral(env: TestEnv) -> None:
    env.set('MODELSLAB_API_KEY', 'test-key')
    provider = ModelsLabProvider()
    profile = provider.model_profile('mistral-7b-instruct')
    assert isinstance(profile, OpenAIModelProfile)


def test_modelslab_model_profile_unknown(env: TestEnv) -> None:
    env.set('MODELSLAB_API_KEY', 'test-key')
    provider = ModelsLabProvider()
    profile = provider.model_profile('some-custom-model')
    assert isinstance(profile, OpenAIModelProfile)

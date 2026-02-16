import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.forge import ForgeProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_forge_provider_init():
    provider = ForgeProvider(api_key='test-key')
    assert provider.name == 'forge'
    assert provider.base_url == 'https://api.forge.tensorblock.co/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'test-key'


def test_forge_provider_env_key(env: TestEnv):
    env.set('FORGE_API_KEY', 'env-key')
    provider = ForgeProvider()
    assert provider.client.api_key == 'env-key'


def test_forge_provider_missing_key(env: TestEnv):
    env.remove('FORGE_API_KEY')
    with pytest.raises(UserError, match='Set the `FORGE_API_KEY`'):
        ForgeProvider()


def test_infer_provider(env: TestEnv):
    env.set('FORGE_API_KEY', 'key')
    provider = infer_provider('forge')
    assert isinstance(provider, ForgeProvider)


def test_openai_model_profile():
    provider = ForgeProvider(api_key='key')
    profile = provider.model_profile('OpenAI/gpt-4o')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_deepseek_model_profile():
    provider = ForgeProvider(api_key='key')
    profile = provider.model_profile('DeepSeek/deepseek-reasoner')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_anthropic_model_profile():
    provider = ForgeProvider(api_key='key')
    profile = provider.model_profile('Anthropic/claude-sonnet-4-5-20250929')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_unknown_model_profile():
    provider = ForgeProvider(api_key='key')
    profile = provider.model_profile('unknown-model')
    assert isinstance(profile, OpenAIModelProfile)


def test_forge_provider_with_openai_client():
    client = openai.AsyncOpenAI(api_key='foo', base_url='https://api.forge.tensorblock.co/v1')
    provider = ForgeProvider(openai_client=client)
    assert provider.client is client


def test_forge_provider_with_http_client():
    http_client = httpx.AsyncClient()
    provider = ForgeProvider(api_key='foo', http_client=http_client)
    assert provider.client.api_key == 'foo'


def test_forge_provider_custom_base_url():
    provider = ForgeProvider(api_key='test-key', base_url='https://custom.endpoint.com/v1')
    assert provider.base_url == 'https://custom.endpoint.com/v1'
    assert str(provider.client.base_url).rstrip('/') == 'https://custom.endpoint.com/v1'


def test_forge_provider_env_base_url(env: TestEnv):
    env.set('FORGE_API_KEY', 'key')
    env.set('FORGE_API_BASE', 'https://env.endpoint.com/v1')
    provider = ForgeProvider()
    assert provider.base_url == 'https://env.endpoint.com/v1'

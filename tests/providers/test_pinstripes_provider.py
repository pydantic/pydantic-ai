import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.pinstripes import PinstripeProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_pinstripe_provider_init():
    provider = PinstripeProvider(api_key='test-key')
    assert provider.name == 'pinstripes'
    assert provider.base_url == 'https://pinstripes.io/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'test-key'


def test_pinstripe_provider_env_key(env: TestEnv):
    env.set('PINSTRIPES_API_KEY', 'env-key')
    provider = PinstripeProvider()
    assert provider.client.api_key == 'env-key'


def test_pinstripe_provider_missing_key(env: TestEnv):
    env.remove('PINSTRIPES_API_KEY')
    with pytest.raises(UserError, match='Set the `PINSTRIPES_API_KEY`'):
        PinstripeProvider()


def test_infer_provider(env: TestEnv):
    env.set('PINSTRIPES_API_KEY', 'key')
    provider = infer_provider('pinstripes')
    assert isinstance(provider, PinstripeProvider)


def test_qwen_profile():
    provider = PinstripeProvider(api_key='key')
    profile = provider.model_profile('ps/qwen3-35b')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_deepseek_profile():
    provider = PinstripeProvider(api_key='key')
    profile = provider.model_profile('ps/deepseek-v4-flash')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile is not None


def test_unknown_model_profile():
    provider = PinstripeProvider(api_key='key')
    profile = provider.model_profile('ps/glm-4.5-air')
    assert isinstance(profile, OpenAIModelProfile)


def test_pinstripe_provider_with_openai_client():
    client = openai.AsyncOpenAI(api_key='foo', base_url='https://pinstripes.io/v1')
    provider = PinstripeProvider(openai_client=client)
    assert provider.client is client


def test_pinstripe_provider_with_http_client():
    http_client = httpx.AsyncClient()
    provider = PinstripeProvider(api_key='foo', http_client=http_client)
    assert provider.client.api_key == 'foo'


def test_pinstripe_provider_custom_base_url():
    provider = PinstripeProvider(api_key='test-key', base_url='https://custom.endpoint.com/v1')
    assert provider.base_url == 'https://custom.endpoint.com/v1'
    assert str(provider.client.base_url).rstrip('/') == 'https://custom.endpoint.com/v1'


def test_pinstripe_provider_env_base_url(env: TestEnv):
    env.set('PINSTRIPES_API_KEY', 'key')
    env.set('PINSTRIPES_BASE_URL', 'https://env.endpoint.com/v1')
    provider = PinstripeProvider()
    assert provider.base_url == 'https://env.endpoint.com/v1'

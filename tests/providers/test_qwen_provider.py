import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.qwen import QwenProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_qwen_provider_init():
    provider = QwenProvider(api_key='test-key')
    assert provider.name == 'qwen'
    assert provider.base_url == 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'test-key'


def test_qwen_provider_env_key(env: TestEnv):
    env.set('QWEN_API_KEY', 'env-key')
    provider = QwenProvider()
    assert provider.client.api_key == 'env-key'


def test_qwen_provider_dashscope_env_key_fallback(env: TestEnv):
    env.remove('QWEN_API_KEY')
    env.set('DASHSCOPE_API_KEY', 'dash-key')
    provider = QwenProvider()
    assert provider.client.api_key == 'dash-key'


def test_qwen_provider_missing_key(env: TestEnv):
    env.remove('QWEN_API_KEY')
    env.remove('DASHSCOPE_API_KEY')
    with pytest.raises(UserError, match='Set the `QWEN_API_KEY`'):
        QwenProvider()


def test_infer_provider(env: TestEnv):
    # infer_provider instantiates the class, so we need an env var or it raises UserError
    env.set('QWEN_API_KEY', 'key')
    provider = infer_provider('qwen')
    assert isinstance(provider, QwenProvider)


def test_qwen_omni_profile_audio_uri():
    provider = QwenProvider(api_key='key')
    # Omni model -> expect 'uri' encoding
    profile = provider.model_profile('qwen-omni-turbo')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.openai_chat_audio_input_encoding == 'uri'


def test_qwen_non_omni_profile_default():
    provider = QwenProvider(api_key='key')
    # Non-omni model -> expect default (base64)
    profile = provider.model_profile('qwen-max')
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.openai_chat_audio_input_encoding == 'base64'

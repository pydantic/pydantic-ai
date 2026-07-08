import httpx
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import BinaryContent, DocumentUrl

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers import infer_provider
    from pydantic_ai.providers.alibaba import AlibabaProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_alibaba_provider_init():
    provider = AlibabaProvider(api_key='test-key')
    assert provider.name == 'alibaba'
    assert provider.base_url == 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'test-key'


def test_alibaba_provider_env_key(env: TestEnv):
    env.set('ALIBABA_API_KEY', 'env-key')
    provider = AlibabaProvider()
    assert provider.client.api_key == 'env-key'


def test_alibaba_provider_dashscope_env_key(env: TestEnv):
    env.remove('ALIBABA_API_KEY')
    env.set('DASHSCOPE_API_KEY', 'dashscope-key')
    provider = AlibabaProvider()
    assert provider.client.api_key == 'dashscope-key'


def test_alibaba_provider_env_key_precedence(env: TestEnv):
    # ALIBABA_API_KEY takes precedence over DASHSCOPE_API_KEY
    env.set('ALIBABA_API_KEY', 'alibaba-key')
    env.set('DASHSCOPE_API_KEY', 'dashscope-key')
    provider = AlibabaProvider()
    assert provider.client.api_key == 'alibaba-key'


def test_alibaba_provider_missing_key(env: TestEnv):
    env.remove('ALIBABA_API_KEY')
    env.remove('DASHSCOPE_API_KEY')
    with pytest.raises(UserError, match='Set the `ALIBABA_API_KEY`'):
        AlibabaProvider()


def test_infer_provider(env: TestEnv):
    # infer_provider instantiates the class, so we need an env var or it raises UserError
    env.set('ALIBABA_API_KEY', 'key')
    provider = infer_provider('alibaba')
    assert isinstance(provider, AlibabaProvider)


def test_qwen_omni_profile_audio_uri():
    provider = AlibabaProvider(api_key='key')
    # Omni model -> expect 'uri' encoding
    profile = provider.model_profile('qwen-omni-turbo')
    assert isinstance(profile, dict)
    assert profile.get('openai_chat_audio_input_encoding', 'base64') == 'uri'
    # The document-input flag must survive the omni branch's merge_profile call.
    assert profile.get('openai_chat_supports_document_input') is False


def test_qwen_non_omni_profile_default():
    provider = AlibabaProvider(api_key='key')
    # Non-omni model -> expect default (base64)
    profile = provider.model_profile('qwen-max')
    assert isinstance(profile, dict)
    assert profile.get('openai_chat_audio_input_encoding', 'base64') == 'base64'


def test_alibaba_provider_with_openai_client():
    client = openai.AsyncOpenAI(api_key='foo')
    provider = AlibabaProvider(openai_client=client)
    assert provider.client is client


def test_alibaba_provider_with_http_client():
    http_client = httpx.AsyncClient()
    provider = AlibabaProvider(api_key='foo', http_client=http_client)
    assert provider.client.api_key == 'foo'
    # The line `self._client = AsyncOpenAI(..., http_client=http_client)` is executed,
    # which is enough for coverage.


def test_alibaba_provider_custom_base_url():
    provider = AlibabaProvider(api_key='test-key', base_url='https://custom.endpoint.com/v1')
    assert provider.base_url == 'https://custom.endpoint.com/v1'
    assert str(provider.client.base_url).rstrip('/') == 'https://custom.endpoint.com/v1'


async def test_alibaba_document_input_not_supported(allow_model_requests: None):
    provider = AlibabaProvider(api_key='test-key')
    model = OpenAIChatModel(model_name='qwen-max', provider=provider)
    agent = Agent(model)

    with pytest.raises(UserError, match='alibaba.*does not support document input'):
        await agent.run(
            [
                'Summarize this document',
                BinaryContent(data=b'%PDF-1.4 test', media_type='application/pdf'),
            ]
        )


async def test_alibaba_document_url_input_not_supported(allow_model_requests: None):
    provider = AlibabaProvider(api_key='test-key')
    model = OpenAIChatModel(model_name='qwen-max', provider=provider)
    agent = Agent(model)

    with pytest.raises(UserError, match='alibaba.*does not support document input'):
        await agent.run(['Summarize this document', DocumentUrl(url='https://example.com/test.pdf')])

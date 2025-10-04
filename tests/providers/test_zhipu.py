from __future__ import annotations as _annotations

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.providers.zhipu import ZhipuProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


def test_init_with_api_key():
    """Test ZhipuProvider initialization with API key."""
    provider = ZhipuProvider(api_key='test-api-key')
    assert provider.name == 'zhipu'
    assert provider.base_url == 'https://open.bigmodel.cn/api/paas/v4/'
    assert provider.client.api_key == 'test-api-key'


def test_init_with_custom_base_url():
    """Test ZhipuProvider initialization with custom base URL."""
    provider = ZhipuProvider(api_key='test-api-key', base_url='https://custom.url/')
    assert provider.base_url == 'https://custom.url/'


def test_init_with_env_var(monkeypatch: pytest.MonkeyPatch):
    """Test ZhipuProvider initialization with environment variable."""
    monkeypatch.setenv('ZHIPU_API_KEY', 'env-api-key')
    provider = ZhipuProvider()
    assert provider.client.api_key == 'env-api-key'


def test_init_without_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test ZhipuProvider initialization without API key raises error."""
    monkeypatch.delenv('ZHIPU_API_KEY', raising=False)
    with pytest.raises(UserError, match='Set the `ZHIPU_API_KEY` environment variable'):
        ZhipuProvider()


def test_init_with_openai_client():
    """Test ZhipuProvider initialization with existing OpenAI client."""
    client = AsyncOpenAI(api_key='test-key', base_url='https://open.bigmodel.cn/api/paas/v4/')
    provider = ZhipuProvider(openai_client=client)
    assert provider.client is client
    assert provider.name == 'zhipu'


def test_init_with_openai_client_and_api_key_raises():
    """Test that providing both openai_client and api_key raises an error."""
    client = AsyncOpenAI(api_key='test-key', base_url='https://open.bigmodel.cn/api/paas/v4/')
    with pytest.raises(AssertionError, match='Cannot provide both'):
        ZhipuProvider(openai_client=client, api_key='another-key')


def test_model_profile_glm_4_5():
    """Test model profile for GLM-4.5."""
    provider = ZhipuProvider(api_key='test-key')
    profile = provider.model_profile('glm-4.5')
    assert profile is not None
    assert profile.supports_json_schema_output is True
    assert profile.supports_json_object_output is True
    assert profile.supports_image_output is False
    assert profile.openai_supports_strict_tool_definition is False


def test_model_profile_glm_4v():
    """Test model profile for GLM-4V (vision model)."""
    provider = ZhipuProvider(api_key='test-key')
    profile = provider.model_profile('glm-4v-plus')
    assert profile is not None
    assert profile.supports_image_output is True


def test_model_profile_codegeex():
    """Test model profile for CodeGeeX."""
    provider = ZhipuProvider(api_key='test-key')
    profile = provider.model_profile('codegeex-4')
    assert profile is not None
    assert profile.supports_json_schema_output is True

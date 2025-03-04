import os
from unittest.mock import patch

import httpx
import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.deepseek import DeepSeekProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_deep_seek_provider():
    provider = DeepSeekProvider()
    assert provider.name == 'deepseek'
    assert provider.base_url == 'https://api.deepseek.com'
    assert isinstance(provider.client, openai.AsyncOpenAI)


def test_deep_seek_provider_need_api_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        provider = DeepSeekProvider()
        assert provider.client.api_key == 'api-key-not-set'


def test_deep_seek_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = DeepSeekProvider(http_client=http_client)
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_deep_seek_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = DeepSeekProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_deep_seek_provider_pass_api_key() -> None:
    provider = DeepSeekProvider(api_key='api-key')
    assert provider.client.api_key == 'api-key'

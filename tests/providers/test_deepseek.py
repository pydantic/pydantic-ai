import os
import re
from unittest.mock import patch

import httpx
import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.deepseek import DeepSeekProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_deep_seek_provider():
    provider = DeepSeekProvider(api_key='api-key')
    assert provider.name == 'deepseek'
    assert provider.base_url == 'https://api.deepseek.com'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_deep_seek_provider_need_api_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            ValueError,
            match=re.escape(
                'Set the `DEEPSEEK_API_KEY` environment variable or pass it via `DeepSeekProvider(api_key=...)`'
                'to use the DeepSeek provider.'
            ),
        ):
            DeepSeekProvider()


def test_deep_seek_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = DeepSeekProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_deep_seek_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = DeepSeekProvider(openai_client=openai_client)
    assert provider.client == openai_client

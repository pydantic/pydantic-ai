from __future__ import annotations as _annotations

import os
import re
from unittest.mock import patch

import httpx
import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from groq import AsyncGroq

    from pydantic_ai.providers.groq import GroqProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='groq not installed')


def test_groq_provider():
    provider = GroqProvider(api_key='api-key')
    assert provider.name == 'groq'
    assert provider.base_url == 'https://api.groq.com'
    assert isinstance(provider.client, AsyncGroq)
    assert provider.client.api_key == 'api-key'


def test_groq_provider_need_api_key() -> None:
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(
            ValueError,
            match=re.escape(
                'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
                'to use the Groq provider.'
            ),
        ):
            GroqProvider()


def test_groq_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = GroqProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_groq_provider_pass_groq_client() -> None:
    groq_client = AsyncGroq(api_key='api-key')
    provider = GroqProvider(groq_client=groq_client)
    assert provider.client == groq_client


def test_groq_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test with environment variable for base_url
    monkeypatch.setenv('GROQ_BASE_URL', 'https://custom.groq.com/v1')
    provider = GroqProvider(api_key='api-key')
    assert provider.base_url == 'https://custom.groq.com/v1'


def test_infer_groq_provider():
    with patch.dict(os.environ, {'GROQ_API_KEY': 'test-api-key'}, clear=False):
        from pydantic_ai.providers import infer_provider

        provider = infer_provider('groq')
        assert provider.name == 'groq'
        assert isinstance(provider, GroqProvider)

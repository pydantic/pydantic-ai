from __future__ import annotations as _annotations

import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

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


def test_groq_provider_need_api_key(env: TestEnv) -> None:
    env.remove('GROQ_API_KEY')
    with pytest.raises(
        UserError,
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

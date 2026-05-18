from __future__ import annotations as _annotations

import httpx
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from cohere import AsyncClientV2
    from cohere.core.http_client import AsyncHttpClient

    from pydantic_ai.providers.cohere import CohereProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='cohere not installed')


def test_cohere_provider() -> None:
    provider = CohereProvider(api_key='api-key')
    assert provider.name == 'cohere'
    assert provider.base_url == 'https://api.cohere.com'
    assert isinstance(provider.client, AsyncClientV2)
    assert provider.client._client_wrapper._token == 'api-key'  # type: ignore[reportPrivateUsage]


def test_cohere_provider_need_api_key(env: TestEnv) -> None:
    env.remove('CO_API_KEY')
    with pytest.raises(UserError, match='CO_API_KEY'):
        CohereProvider()


def test_cohere_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = CohereProvider(http_client=http_client, api_key='api-key')
    # The AsyncClientV2 wraps our httpx client in an AsyncHttpClient
    # So we just check that the httpx_client is an instance of AsyncHttpClient
    assert isinstance(provider.client._client_wrapper.httpx_client, AsyncHttpClient)  # type: ignore[reportPrivateUsage]


def test_cohere_provider_pass_cohere_client() -> None:
    cohere_client = AsyncClientV2(api_key='test-api-key')
    provider = CohereProvider(cohere_client=cohere_client)
    assert provider.client == cohere_client


def test_cohere_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    custom_base_url = 'https://custom.cohere.com/'
    # Test with environment variable for base_url
    monkeypatch.setenv('CO_BASE_URL', custom_base_url)
    provider = CohereProvider(api_key='api-key')
    assert provider.base_url == custom_base_url

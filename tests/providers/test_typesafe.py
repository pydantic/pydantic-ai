from __future__ import annotations as _annotations

import httpx2
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from typesafe_sdk import AsyncTypeSafeClient

    from pydantic_ai.providers.typesafe import TypeSafeProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='typesafe-sdk not installed')


def test_typesafe_provider() -> None:
    provider = TypeSafeProvider(api_key='api-key')
    assert provider.name == 'typesafe'
    assert provider.base_url == 'https://api.typesafe.ai'
    assert isinstance(provider.client, AsyncTypeSafeClient)
    assert provider.client._config.api_key == 'api-key'  # type: ignore[reportPrivateUsage]


def test_typesafe_provider_need_api_key(env: TestEnv) -> None:
    env.remove('TYPESAFE_API_KEY')
    with pytest.raises(UserError, match='TYPESAFE_API_KEY'):
        TypeSafeProvider()


def test_typesafe_provider_env(env: TestEnv) -> None:
    env.set('TYPESAFE_API_KEY', 'env-key')
    env.set('TYPESAFE_BASE_URL', 'https://typesafe.example.com')
    provider = TypeSafeProvider()
    assert provider.base_url == 'https://typesafe.example.com'
    assert provider.client._config.api_key == 'env-key'  # type: ignore[reportPrivateUsage]


def test_typesafe_provider_base_url_argument() -> None:
    provider = TypeSafeProvider(api_key='api-key', base_url='https://typesafe.example.com/v2')
    assert provider.base_url == 'https://typesafe.example.com/v2'
    assert provider.client._config.base_url == 'https://typesafe.example.com/v2'  # type: ignore[reportPrivateUsage]


def test_typesafe_provider_pass_http_client() -> None:
    http_client = httpx2.AsyncClient()
    provider = TypeSafeProvider(http_client=http_client, api_key='api-key')
    assert provider.client._http_client is http_client  # type: ignore[reportPrivateUsage]


def test_typesafe_provider_pass_typesafe_client() -> None:
    typesafe_client = AsyncTypeSafeClient(api_key='api-key', base_url='https://typesafe.example.com')
    provider = TypeSafeProvider(typesafe_client=typesafe_client)
    assert provider.client is typesafe_client
    assert provider.base_url == 'https://typesafe.example.com'


async def test_typesafe_provider_recreates_closed_http_client() -> None:
    """A provider that owns its HTTP client replaces it on re-entry once it has been closed."""
    provider = TypeSafeProvider(api_key='api-key')
    async with provider:
        first = provider.client._http_client  # type: ignore[reportPrivateUsage]
    assert first.is_closed
    async with provider:
        assert provider.client._http_client is not first  # type: ignore[reportPrivateUsage]
        assert not provider.client._http_client.is_closed  # type: ignore[reportPrivateUsage]

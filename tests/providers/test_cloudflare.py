from __future__ import annotations as _annotations

import httpx
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from cloudflare import AsyncCloudflare

    from pydantic_ai.providers.cloudflare import CloudflareProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='cloudflare not installed')


def test_cloudflare_provider() -> None:
    provider = CloudflareProvider(api_key='api-key')
    assert provider.name == 'cloudflare'
    assert isinstance(provider.client, AsyncCloudflare)
    assert provider.base_url.startswith('https://api.cloudflare.com')


def test_cloudflare_provider_need_api_key(env: TestEnv) -> None:
    env.remove('CLOUDFLARE_API_KEY')
    with pytest.raises(UserError, match='CLOUDFLARE_API_KEY'):
        CloudflareProvider()


def test_cloudflare_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = CloudflareProvider(api_key='api-key', http_client=http_client)
    assert isinstance(provider.client, AsyncCloudflare)


def test_cloudflare_provider_pass_client() -> None:
    cloudflare_client = AsyncCloudflare(api_key='test-api-key')
    provider = CloudflareProvider(cloudflare_client=cloudflare_client)
    assert provider.client == cloudflare_client

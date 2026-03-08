import asyncio
import re

import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from xai_sdk import AsyncClient

    from pydantic_ai.providers.xai import XaiProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed')


def test_xai_provider():
    provider = XaiProvider(api_key='api-key')
    assert provider.name == 'xai'
    assert provider.base_url == 'https://api.x.ai/v1'
    assert isinstance(provider.client, AsyncClient)


def test_xai_provider_need_api_key(env: TestEnv) -> None:
    env.remove('XAI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `XAI_API_KEY` environment variable or pass it via `XaiProvider(api_key=...)`'
            'to use the xAI provider.'
        ),
    ):
        XaiProvider()


def test_xai_pass_xai_client() -> None:
    xai_client = AsyncClient(api_key='api-key')
    provider = XaiProvider(xai_client=xai_client)
    assert provider.client == xai_client


def test_xai_model_profile():
    from pydantic_ai.profiles.grok import GrokModelProfile

    provider = XaiProvider(api_key='api-key')
    profile = provider.model_profile('grok-4-1-fast-non-reasoning')
    assert isinstance(profile, GrokModelProfile)
    assert profile.grok_supports_builtin_tools is True


def test_xai_provider_recreates_client_on_new_loop():
    """Test that XaiProvider returns a fresh client when the event loop changes."""
    provider = XaiProvider(api_key='api-key')

    clients: list[AsyncClient] = []

    async def get_client():
        return provider.client

    # Run in two separate event loops — the client should be recreated.
    clients.append(asyncio.run(get_client()))
    clients.append(asyncio.run(get_client()))

    assert isinstance(clients[0], AsyncClient)
    assert isinstance(clients[1], AsyncClient)
    assert clients[0] is not clients[1]


def test_xai_provider_reuses_client_on_same_loop():
    """Test that XaiProvider reuses the client within the same event loop."""
    provider = XaiProvider(api_key='api-key')

    async def get_clients_same_loop():
        c1 = provider.client
        c2 = provider.client
        return c1, c2

    c1, c2 = asyncio.run(get_clients_same_loop())
    assert c1 is c2

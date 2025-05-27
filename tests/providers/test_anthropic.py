from __future__ import annotations as _annotations

import httpx
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic

    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='need to install anthropic')


def test_anthropic_provider():
    provider = AnthropicProvider(api_key='api-key')
    assert provider.name == 'anthropic'
    assert provider.base_url == 'https://api.anthropic.com'
    assert isinstance(provider.client, AsyncAnthropic)
    assert provider.client.api_key == 'api-key'


def test_anthropic_provider_need_api_key(env: TestEnv) -> None:
    env.remove('ANTHROPIC_API_KEY')
    with pytest.raises(UserError, match=r'.*ANTHROPIC_API_KEY.*'):
        AnthropicProvider()


def test_anthropic_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = AnthropicProvider(http_client=http_client, api_key='api-key')
    assert isinstance(provider.client, AsyncAnthropic)
    # Verify the http_client is being used by the AsyncAnthropic client
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_anthropic_provider_pass_anthropic_client() -> None:
    anthropic_client = AsyncAnthropic(api_key='api-key')
    provider = AnthropicProvider(anthropic_client=anthropic_client)
    assert provider.client == anthropic_client


def test_anthropic_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test with environment variable for base_url
    custom_base_url = 'https://custom.anthropic.com/v1'
    monkeypatch.setenv('ANTHROPIC_BASE_URL', custom_base_url)
    provider = AnthropicProvider(api_key='api-key')
    assert provider.base_url.rstrip('/') == custom_base_url.rstrip('/')

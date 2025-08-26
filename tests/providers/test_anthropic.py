from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='need to install anthropic')


def test_anthropic_provider():
    provider = AnthropicProvider(api_key='api-key')
    assert provider.name == 'anthropic'
    assert provider.base_url == 'https://api.anthropic.com'
    assert isinstance(provider.client, AsyncAnthropic)
    assert provider.client.api_key == 'api-key'


def test_anthropic_provider_pass_anthropic_client() -> None:
    anthropic_client = AsyncAnthropic(api_key='api-key')
    provider = AnthropicProvider(anthropic_client=anthropic_client)
    assert provider.client == anthropic_client
    bedrock_client = AsyncAnthropicBedrock(
        aws_secret_key='aws-secret-key',
        aws_access_key='aws-access-key',
        aws_region='us-west-2',
        aws_profile='default',
        aws_session_token='aws-session-token',
    )
    provider = AnthropicProvider(anthropic_client=bedrock_client)
    assert provider.client == bedrock_client


def test_anthropic_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test with environment variable for base_url
    custom_base_url = 'https://custom.anthropic.com/v1'
    monkeypatch.setenv('ANTHROPIC_BASE_URL', custom_base_url)
    provider = AnthropicProvider(api_key='api-key')
    assert provider.base_url.rstrip('/') == custom_base_url.rstrip('/')

from __future__ import annotations as _annotations

import httpx
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


def test_anthropic_provider_with_aws_credentials() -> None:
    provider = AnthropicProvider(
        aws_secret_key='aws-secret-key',
        aws_access_key='aws-access-key',
        aws_region='us-west-2',
        aws_profile='default',
        aws_session_token='aws-session-token',
    )
    assert provider.name == 'anthropic'
    assert provider.base_url == 'https://bedrock-runtime.us-west-2.amazonaws.com'
    assert isinstance(provider.client, AsyncAnthropicBedrock)


def test_anthropic_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = AnthropicProvider(http_client=http_client, api_key='api-key')
    assert isinstance(provider.client, AsyncAnthropic)
    # Verify the http_client is being used by the AsyncAnthropic client
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]
    bedrock_provider = AnthropicProvider(
        aws_secret_key='aws-secret-key',
        aws_access_key='aws-access_key',
        aws_region='us-west-2',
        aws_profile='default',
        aws_session_token='aws-session-token',
        http_client=http_client,
    )
    assert isinstance(bedrock_provider.client, AsyncAnthropicBedrock)
    assert bedrock_provider.client._client == http_client  # type: ignore[reportPrivateUsage]


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


def test_bedrock_anthropic_provider_with_envs(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test with environment variables for AWS credentials
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'aws-secret-access-key')
    monkeypatch.setenv('AWS_ACCESS_KEY_ID', 'aws-access-key-id')
    monkeypatch.setenv('AWS_SESSION_TOKEN', 'aws-session-token')
    monkeypatch.setenv('AWS_PROFILE', 'default')
    monkeypatch.setenv('AWS_REGION', 'us-west-2')
    bedrock_provider = AnthropicProvider()
    assert bedrock_provider.name == 'anthropic'
    assert bedrock_provider.base_url == 'https://bedrock-runtime.us-west-2.amazonaws.com'
    assert isinstance(bedrock_provider.client, AsyncAnthropicBedrock)

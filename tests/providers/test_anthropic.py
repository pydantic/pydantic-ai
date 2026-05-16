from __future__ import annotations as _annotations

import pytest

from ..conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

    from pydantic_ai.native_tools._tool_search import ToolSearchTool
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile
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


@pytest.mark.parametrize(
    'model_name',
    [
        # Direct Anthropic API ids (with and without date suffix)
        'claude-haiku-4-5',
        'claude-haiku-4-5-20251001',
        # Amazon Bedrock ids: `anthropic.` provider segment, optional geo prefix and `-vN(:M)?` version suffix
        'anthropic.claude-haiku-4-5',
        'anthropic.claude-haiku-4-5-20251001-v1:0',
        'us.anthropic.claude-haiku-4-5-20251001-v1:0',
        'global.anthropic.claude-haiku-4-5',
        # Anthropic on Vertex AI: `@`-delimited version
        'claude-haiku-4-5@20251001',
    ],
)
def test_anthropic_provider_model_profile_normalizes_transport_specific_ids(model_name: str):
    """`AnthropicProvider.model_profile` resolves capability flags from the bare `claude-...` name,
    even when the underlying client (Bedrock/Vertex) carries a transport-specific model id."""
    profile = AnthropicProvider.model_profile(model_name)
    assert isinstance(profile, AnthropicModelProfile)
    assert profile.supports_json_schema_output is True
    assert ToolSearchTool in profile.supported_native_tools


def test_anthropic_provider_model_profile_older_model_still_resolves():
    """Normalization must not over-strip: an older model without structured-output support
    still resolves to the right (negative) flags."""
    profile = AnthropicProvider.model_profile('anthropic.claude-3-5-sonnet-20240620-v1:0')
    assert isinstance(profile, AnthropicModelProfile)
    assert profile.supports_json_schema_output is False
    assert ToolSearchTool not in profile.supported_native_tools

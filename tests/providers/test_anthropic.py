from __future__ import annotations as _annotations

import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

    from pydantic_ai.exceptions import UserError
    from pydantic_ai.native_tools import SUPPORTED_NATIVE_TOOLS
    from pydantic_ai.native_tools._tool_search import ToolSearchTool
    from pydantic_ai.providers.anthropic import AnthropicProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='need to install anthropic')


def test_anthropic_provider():
    provider = AnthropicProvider(api_key='api-key')
    assert provider.name == 'anthropic'
    assert provider.base_url == 'https://api.anthropic.com'
    assert isinstance(provider.client, AsyncAnthropic)
    assert provider.client.api_key == 'api-key'


def test_anthropic_provider_without_api_key_raises_error(env: TestEnv):
    env.remove('ANTHROPIC_API_KEY')
    with pytest.raises(
        UserError,
        match=(
            r'Set the `ANTHROPIC_API_KEY` environment variable or pass it via `AnthropicProvider\(api_key=\.\.\.\)`'
            r" to use the Anthropic provider\. To try Pydantic AI without an API key, use the built-in test model: `Agent\('test'\)`\."
        ),
    ):
        AnthropicProvider()


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
    assert isinstance(profile, dict)
    assert profile.get('supports_json_schema_output', False) is True
    assert ToolSearchTool in profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)


def test_anthropic_provider_model_profile_older_model_still_resolves():
    """Normalization must not over-strip: an older model without structured-output support
    still resolves to the right (negative) flags."""
    profile = AnthropicProvider.model_profile('anthropic.claude-3-5-sonnet-20240620-v1:0')
    assert isinstance(profile, dict)
    assert profile.get('supports_json_schema_output', False) is False
    assert ToolSearchTool not in profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)


@pytest.mark.parametrize(
    ('model_name', 'supported'),
    [
        ('claude-fable-5', True),
        ('claude-mythos-5', True),
        ('claude-opus-4-8', True),
        ('claude-opus-5', True),
        # Sonnet 5 accepts the entry with a 200 and then ignores it, so it is deliberately out.
        ('claude-sonnet-5', False),
        ('claude-opus-4-7', False),
        ('claude-sonnet-4-6', False),
        ('claude-haiku-4-5', False),
        # A Bedrock id is normalized to the bare name before the prefix check. A Vertex-style
        # `@<date>` suffix isn't — it reaches the check verbatim, which is why an unsupported model
        # carrying one still resolves to False for the ordinary reason.
        ('us.anthropic.claude-opus-4-8-v1:0', True),
        ('claude-sonnet-4-6@20260101', False),
    ],
)
def test_anthropic_provider_model_profile_inline_system_prompts(model_name: str, supported: bool):
    """Only the models Anthropic publishes for the feature honor a `{'role': 'system'}` entry.

    The flag is set by `AnthropicProvider.model_profile()` rather than `anthropic_model_profile()`
    because it describes the Messages API: Bedrock Converse and the OpenAI-compatible gateways route
    the same models but hoist a non-leading system prompt into their own top-level system field.

    `claude-sonnet-5` is the case worth pinning. It doesn't reject the entry — the request succeeds —
    it just doesn't act on it, which makes a green request a trap rather than evidence. Anthropic
    documents the feature as unavailable there, and measurement agrees, so the flag stays off and a
    future contributor who tries the request and sees a 200 has this to read.
    """
    profile = AnthropicProvider.model_profile(model_name)
    assert isinstance(profile, dict)
    assert profile.get('supports_inline_system_prompts', False) is supported

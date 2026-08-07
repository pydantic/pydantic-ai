import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_moonshotai_provider():
    """Test basic MoonshotAI provider initialization."""
    provider = MoonshotAIProvider(api_key='api-key')
    assert provider.name == 'moonshotai'
    assert provider.base_url == 'https://api.moonshot.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_moonshotai_provider_need_api_key(env: TestEnv) -> None:
    """Test that MoonshotAI provider requires an API key."""
    env.remove('MOONSHOTAI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `MOONSHOTAI_API_KEY` environment variable or pass it via `MoonshotAIProvider(api_key=...)`'
            ' to use the MoonshotAI provider.'
        ),
    ):
        MoonshotAIProvider()


def test_moonshotai_provider_pass_http_client() -> None:
    """Test passing a custom HTTP client to MoonshotAI provider."""
    http_client = httpx.AsyncClient()
    provider = MoonshotAIProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_moonshotai_pass_openai_client() -> None:
    """Test passing a custom OpenAI client to MoonshotAI provider."""
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = MoonshotAIProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_moonshotai_provider_creates_http_client() -> None:
    """Test MoonshotAI provider creates its own HTTP client."""
    provider = MoonshotAIProvider(api_key='api-key')
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_moonshotai_model_profile():
    provider = MoonshotAIProvider(api_key='api-key')
    model = OpenAIChatModel('kimi-k2-0711-preview', provider=provider)
    assert isinstance(model.profile, dict)
    assert model.profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert model.profile.get('openai_supports_tool_choice_required', True) is False
    assert model.profile.get('supports_json_object_output', False) is True


def test_moonshotai_model_profile_thinking():
    # Unit (not VCR): these pin the profile flags resolved from the model id at model-build time, which
    # the cassette doesn't exercise — it records the wire round-trip, not the resolved profile, and our
    # matchers aren't body-sensitive, so a regression flipping a model's reasoning flags could still
    # replay an existing recording green. A direct profile assertion is what catches it.
    provider = MoonshotAIProvider(api_key='api-key')

    # Reasoning models advertise thinking; it's always-on since Moonshot rejects reasoning_effort='none'.
    for reasoning_model in ('kimi-k2.5', 'kimi-k2.6', 'kimi-k2.7-code', 'kimi-k2.7-code-highspeed', 'kimi-k3'):
        profile = provider.model_profile(reasoning_model)
        assert profile is not None
        assert profile.get('supports_thinking') is True
        assert profile.get('thinking_always_enabled') is True
        assert profile.get('openai_chat_thinking_field') == 'reasoning_content'
        assert profile.get('openai_chat_send_back_thinking_parts') == 'field'

    # Instruct/base models don't reason, so thinking stays off.
    for non_reasoning_model in ('moonshot-v1-8k', 'moonshot-v1-auto', 'kimi-k2-0711-preview', 'kimi-latest'):
        profile = provider.model_profile(non_reasoning_model)
        assert profile is not None
        assert profile.get('supports_thinking', False) is False
        assert profile.get('thinking_always_enabled', False) is False
        assert profile.get('openai_chat_thinking_field') == 'reasoning_content'

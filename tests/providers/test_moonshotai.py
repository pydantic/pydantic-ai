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
    """`supports_thinking` must be set for Kimi reasoning models so unified `thinking` reaches them.

    Unit test (not VCR): recording against the MoonshotAI API needs live credentials, and the
    cassette matchers aren't always sensitive to the request body, so a VCR test could match an
    existing recording and stay green even if the request stopped carrying thinking parameters.
    Asserting the profile flag directly pins the model-name -> capability mapping against drift,
    mirroring `test_zai_provider_model_profile`.
    """
    provider = MoonshotAIProvider(api_key='api-key')

    # Kimi reasoning models: `thinking` must be forwarded, so `supports_thinking` is True.
    for reasoning_model in ('kimi-thinking-preview', 'kimi-k2-thinking'):
        profile = provider.model_profile(reasoning_model)
        assert profile is not None
        assert profile.get('supports_thinking') is True
        # The provider wires up the response-shape fields regardless of the model.
        assert profile.get('openai_chat_thinking_field') == 'reasoning_content'
        assert profile.get('openai_chat_send_back_thinking_parts') == 'field'

    # Instruct/base models are not reasoning models: `supports_thinking` stays False so thinking
    # parameters are not sent (guards against enabling thinking on non-reasoning Kimi models).
    for non_reasoning_model in ('moonshot-v1-8k', 'kimi-k2-0711-preview', 'kimi-latest'):
        profile = provider.model_profile(non_reasoning_model)
        assert profile is not None
        assert profile.get('supports_thinking', False) is False
        # The provider still advertises the response-shape fields even on non-thinking models.
        assert profile.get('openai_chat_thinking_field') == 'reasoning_content'

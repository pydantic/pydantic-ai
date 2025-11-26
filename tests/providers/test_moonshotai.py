import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

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


def test_moonshotai_provider_with_cached_http_client() -> None:
    """Test MoonshotAI provider using cached HTTP client (covers line 76)."""
    # This should use the else branch with cached_async_http_client
    provider = MoonshotAIProvider(api_key='api-key')
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_moonshotai_model_profile():
    provider = MoonshotAIProvider(api_key='api-key')
    model = OpenAIChatModel('kimi-k2-0711-preview', provider=provider)
    assert isinstance(model.profile, OpenAIModelProfile)
    assert model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert model.profile.openai_supports_tool_choice_required is False
    assert model.profile.supports_json_object_output is True

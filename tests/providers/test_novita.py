import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.novita import NovitaProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_novita_provider():
    """Test basic Novita provider initialization."""
    provider = NovitaProvider(api_key='api-key')
    assert provider.name == 'novita'
    assert provider.base_url == 'https://api.novita.ai/openai'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_novita_provider_need_api_key(env: TestEnv) -> None:
    """Test that Novita provider requires an API key."""
    env.remove('NOVITA_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `NOVITA_API_KEY` environment variable or pass it via `NovitaProvider(api_key=...)` '
            'to use the Novita provider.'
        ),
    ):
        NovitaProvider()


def test_novita_provider_pass_http_client() -> None:
    """Test passing a custom HTTP client to Novita provider."""
    http_client = httpx.AsyncClient()
    provider = NovitaProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_novita_pass_openai_client() -> None:
    """Test passing a custom OpenAI client to Novita provider."""
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = NovitaProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_novita_provider_with_cached_http_client() -> None:
    """Test Novita provider using cached HTTP client."""
    provider = NovitaProvider(api_key='api-key')
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_novita_model_profile():
    """Test Novita model profile configuration."""
    provider = NovitaProvider(api_key='api-key')
    model = OpenAIChatModel('moonshotai/kimi-k2.5', provider=provider)
    assert isinstance(model.profile, OpenAIModelProfile)
    assert model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert model.profile.supports_json_object_output is True

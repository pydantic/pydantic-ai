import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.grok import GrokProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_grok_provider():
    provider = GrokProvider(api_key='api-key')
    assert provider.name == 'grok'
    assert provider.base_url == 'https://api.x.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_grok_provider_need_api_key(env: TestEnv) -> None:
    env.remove('GROK_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GROK_API_KEY` environment variable or pass it via `GrokProvider(api_key=...)`'
            'to use the Grok provider.'
        ),
    ):
        GrokProvider()


def test_grok_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = GrokProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_grok_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = GrokProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_grok_model_profile():
    provider = GrokProvider(api_key='api-key')
    model = OpenAIChatModel('grok-3', provider=provider)
    assert isinstance(model.profile, OpenAIModelProfile)
    assert model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert model.profile.openai_supports_strict_tool_definition is False

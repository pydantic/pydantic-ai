from __future__ import annotations as _annotations

import re

import httpx
import pytest

from pydantic_ai import WebSearchTool
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.perplexity import PerplexityProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_perplexity_provider():
    provider = PerplexityProvider(api_key='api-key')
    assert provider.name == 'perplexity'
    assert provider.base_url == 'https://api.perplexity.ai'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_perplexity_provider_need_api_key(env: TestEnv) -> None:
    env.remove('PERPLEXITY_API_KEY')
    env.remove('PPLX_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `PERPLEXITY_API_KEY` environment variable or pass it via `PerplexityProvider(api_key=...)`'
            ' to use the Perplexity provider.'
        ),
    ):
        PerplexityProvider()


def test_perplexity_provider_pplx_alias(env: TestEnv) -> None:
    env.remove('PERPLEXITY_API_KEY')
    env.set('PPLX_API_KEY', 'aliased-key')
    provider = PerplexityProvider()
    assert provider.client.api_key == 'aliased-key'


def test_perplexity_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = PerplexityProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_perplexity_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = PerplexityProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_perplexity_model_profile_enables_web_search() -> None:
    provider = PerplexityProvider(api_key='api-key')
    model = OpenAIChatModel('sonar-pro', provider=provider)
    profile = OpenAIModelProfile.from_profile(model.profile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert profile.openai_chat_supports_web_search is True
    assert WebSearchTool in model.profile.supported_builtin_tools


def test_perplexity_reasoning_model_profile() -> None:
    provider = PerplexityProvider(api_key='api-key')
    model = OpenAIChatModel('sonar-reasoning-pro', provider=provider)
    assert model.profile.supports_thinking is True
    assert model.profile.ignore_streamed_leading_whitespace is True

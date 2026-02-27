import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.avian import AvianProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_avian_provider():
    provider = AvianProvider(api_key='api-key')
    assert provider.name == 'avian'
    assert provider.base_url == 'https://api.avian.io/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_avian_provider_need_api_key(env: TestEnv) -> None:
    env.remove('AVIAN_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `AVIAN_API_KEY` environment variable or pass it via `AvianProvider(api_key=...)` '
            'to use the Avian provider.'
        ),
    ):
        AvianProvider()


def test_avian_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = AvianProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_avian_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = AvianProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_avian_provider_model_profile():
    provider = AvianProvider(api_key='api-key')

    profile = provider.model_profile('deepseek/deepseek-v3.2')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('moonshotai/kimi-k2.5')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('z-ai/glm-5')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    profile = provider.model_profile('minimax/minimax-m2.5')
    assert profile is not None
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

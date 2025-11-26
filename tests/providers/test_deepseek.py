import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_deep_seek_provider():
    provider = DeepSeekProvider(api_key='api-key')
    assert provider.name == 'deepseek'
    assert provider.base_url == 'https://api.deepseek.com'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_deep_seek_provider_need_api_key(env: TestEnv) -> None:
    env.remove('DEEPSEEK_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `DEEPSEEK_API_KEY` environment variable or pass it via `DeepSeekProvider(api_key=...)`'
            'to use the DeepSeek provider.'
        ),
    ):
        DeepSeekProvider()


def test_deep_seek_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = DeepSeekProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_deep_seek_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = DeepSeekProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_deep_seek_model_profile():
    provider = DeepSeekProvider(api_key='api-key')
    model = OpenAIChatModel('deepseek-r1', provider=provider)
    assert model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer

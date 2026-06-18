import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.atlascloud import AtlasCloudProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_atlascloud_provider():
    provider = AtlasCloudProvider(api_key='api-key')
    assert provider.name == 'atlascloud'
    assert provider.base_url == 'https://api.atlascloud.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_atlascloud_provider_need_api_key(env: TestEnv) -> None:
    env.remove('ATLASCLOUD_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `ATLASCLOUD_API_KEY` environment variable or pass it via `AtlasCloudProvider(api_key=...)`'
            ' to use the Atlas Cloud provider.'
        ),
    ):
        AtlasCloudProvider()


def test_atlascloud_provider_api_key_from_env(env: TestEnv) -> None:
    env.set('ATLASCLOUD_API_KEY', 'env-api-key')
    provider = AtlasCloudProvider()
    assert provider.client.api_key == 'env-api-key'


def test_atlascloud_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = AtlasCloudProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_atlascloud_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = AtlasCloudProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_atlascloud_model_profile():
    provider = AtlasCloudProvider(api_key='api-key')
    model = OpenAIChatModel('deepseek-chat', provider=provider)
    assert isinstance(model.profile, OpenAIModelProfile)
    assert model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer


def test_atlascloud_non_deepseek_model_profile():
    provider = AtlasCloudProvider(api_key='api-key')
    profile = provider.model_profile('gpt-4o')
    assert profile is not None
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer

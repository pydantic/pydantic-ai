import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.cerebras import CerebrasProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_cerebras_provider():
    provider = CerebrasProvider(api_key='api-key')
    assert provider.name == 'cerebras'
    assert provider.base_url == 'https://api.cerebras.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_cerebras_provider_need_api_key(env: TestEnv) -> None:
    env.remove('CEREBRAS_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `CEREBRAS_API_KEY` environment variable or pass it via `CerebrasProvider(api_key=...)`'
            'to use the Cerebras provider.'
        ),
    ):
        CerebrasProvider()


def test_cerebras_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = CerebrasProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_cerebras_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = CerebrasProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_cerebras_model_profile():
    provider = CerebrasProvider(api_key='api-key')
    model = OpenAIModel('qwen-3-32b', provider=provider)
    assert isinstance(model.profile, OpenAIModelProfile)
    assert model.profile.json_schema_transformer == GoogleJsonSchemaTransformer
    assert model.profile.openai_supports_strict_tool_definition is False

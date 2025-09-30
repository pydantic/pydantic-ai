import re

import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.heroku import HerokuProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_heroku_provider():
    provider = HerokuProvider(api_key='api-key')
    assert provider.name == 'heroku'
    assert provider.base_url == 'https://us.inference.heroku.com/v1/'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_heroku_provider_need_api_key(env: TestEnv) -> None:
    env.remove('HEROKU_INFERENCE_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `HEROKU_INFERENCE_KEY` environment variable or pass it via `HerokuProvider(api_key=...)`'
            'to use the Heroku provider.'
        ),
    ):
        HerokuProvider()


def test_heroku_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = HerokuProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_heroku_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = HerokuProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_heroku_model_profile():
    provider = HerokuProvider(api_key='api-key')
    model = OpenAIChatModel('claude-3-7-sonnet', provider=provider)
    assert isinstance(model.profile, OpenAIModelProfile)
    assert model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer


async def test_heroku_model_provider_claude_3_7_sonnet(allow_model_requests: None, heroku_inference_key: str):
    provider = HerokuProvider(api_key=heroku_inference_key)
    m = OpenAIChatModel('claude-3-7-sonnet', provider=provider)
    agent = Agent(m)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        "The capital of France is Paris. It's not only the political capital but also a major cultural and economic hub in Europe, known for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
    )

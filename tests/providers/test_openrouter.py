import re

import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_openrouter_provider():
    provider = OpenRouterProvider(api_key='api-key')
    assert provider.name == 'openrouter'
    assert provider.base_url == 'https://openrouter.ai/api/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_openrouter_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OPENROUTER_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OPENROUTER_API_KEY` environment variable or pass it via `OpenRouterProvider(api_key=...)`'
            'to use the OpenRouter provider.'
        ),
    ):
        OpenRouterProvider()


def test_openrouter_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = OpenRouterProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_openrouter_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = OpenRouterProvider(openai_client=openai_client)
    assert provider.client == openai_client


async def test_openrouter_with_google_model(allow_model_requests: None, openrouter_api_key: str) -> None:
    provider = OpenRouterProvider(api_key=openrouter_api_key)
    model = OpenAIModel('google/gemini-2.0-flash-exp:free', provider=provider)
    agent = Agent(model, instructions='Be helpful.')
    response = await agent.run('Tell me a joke.')
    assert response.output == snapshot("""\
Why don't scientists trust atoms? \n\

Because they make up everything!
""")

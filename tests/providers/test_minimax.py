import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.minimax import MiniMaxProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_minimax_provider():
    provider = MiniMaxProvider(api_key='api-key')
    assert provider.name == 'minimax'
    assert provider.base_url == 'https://api.minimax.io/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_minimax_provider_need_api_key(env: TestEnv) -> None:
    env.remove('MINIMAX_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `MINIMAX_API_KEY` environment variable or pass it via `MiniMaxProvider(api_key=...)`'
            'to use the MiniMax provider.'
        ),
    ):
        MiniMaxProvider()


def test_minimax_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = MiniMaxProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_minimax_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = MiniMaxProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_minimax_model_profile():
    provider = MiniMaxProvider(api_key='api-key')
    model = OpenAIChatModel('MiniMax-M2.5', provider=provider)
    assert model.profile.supports_json_schema_output is False
    assert model.profile.supports_json_object_output is False

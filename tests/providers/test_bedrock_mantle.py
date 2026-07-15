from __future__ import annotations

import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.profiles.openai import openai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider


pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='openai not installed')]


def test_bedrock_mantle_provider(env: TestEnv) -> None:
    env.set('AWS_BEARER_TOKEN_BEDROCK', 'env-api-key')
    env.set('AWS_DEFAULT_REGION', 'us-west-2')
    provider = BedrockMantleProvider()
    assert provider.name == 'bedrock-mantle'
    assert provider.base_url == 'https://bedrock-mantle.us-west-2.api.aws/openai/v1/'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'env-api-key'


def test_bedrock_mantle_aws_region_fallback(env: TestEnv) -> None:
    env.remove('AWS_DEFAULT_REGION')
    env.set('AWS_REGION', 'eu-west-1')
    provider = BedrockMantleProvider(api_key='api-key')
    assert provider.base_url == 'https://bedrock-mantle.eu-west-1.api.aws/openai/v1/'


def test_bedrock_mantle_provider_need_api_key(env: TestEnv) -> None:
    env.remove('AWS_BEARER_TOKEN_BEDROCK')
    with pytest.raises(UserError, match='AWS_BEARER_TOKEN_BEDROCK'):
        BedrockMantleProvider(region_name='us-east-1')


def test_bedrock_mantle_provider_need_region(env: TestEnv) -> None:
    env.remove('AWS_DEFAULT_REGION')
    env.remove('AWS_REGION')
    with pytest.raises(UserError, match=re.escape('`AWS_DEFAULT_REGION` or `AWS_REGION`')):
        BedrockMantleProvider(api_key='api-key')


def test_bedrock_mantle_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = BedrockMantleProvider(openai_client=openai_client)
    assert provider.client is openai_client


def test_bedrock_mantle_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = BedrockMantleProvider(region_name='us-east-1', api_key='api-key', http_client=http_client)
    assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]


async def test_bedrock_mantle_provider_reopens_http_client() -> None:
    provider = BedrockMantleProvider(region_name='us-east-1', api_key='api-key')
    first_http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
    async with provider:
        pass
    assert first_http_client.is_closed

    async with provider:
        assert provider.client._client is not first_http_client  # pyright: ignore[reportPrivateUsage]
        assert not provider.client._client.is_closed  # pyright: ignore[reportPrivateUsage]


def test_bedrock_mantle_model_profile(mocker: MockerFixture) -> None:
    profile_mock = mocker.patch('pydantic_ai.providers.bedrock_mantle.openai_model_profile', wraps=openai_model_profile)
    profile = BedrockMantleProvider.model_profile('openai.gpt-5.6-luna')
    profile_mock.assert_called_once_with('gpt-5.6-luna')
    assert profile is not None
    assert profile.get('tool_call_id_scope') == 'response'


def test_bedrock_mantle_infer_model(env: TestEnv) -> None:
    env.set('AWS_BEARER_TOKEN_BEDROCK', 'api-key')
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    model = infer_model('bedrock-mantle:openai.gpt-5.6-luna')
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == 'openai.gpt-5.6-luna'
    assert model.system == 'bedrock-mantle'

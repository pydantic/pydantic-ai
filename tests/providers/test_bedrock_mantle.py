from __future__ import annotations

import re

import httpx
import pytest

from pydantic_ai import ModelProfile
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.profiles import merge_profile
from pydantic_ai.profiles.openai import openai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.bedrock_mantle import BedrockMantleProvider


pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='openai not installed')]


def test_bedrock_mantle_provider(env: TestEnv) -> None:
    """Provider configuration is local behavior that a VCR recording cannot observe."""
    env.set('AWS_BEARER_TOKEN_BEDROCK', 'env-api-key')
    env.set('AWS_DEFAULT_REGION', 'us-west-2')
    provider = BedrockMantleProvider()
    assert provider.name == 'bedrock-mantle'
    assert provider.base_url == 'https://bedrock-mantle.us-west-2.api.aws/openai/v1/'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'env-api-key'


def test_bedrock_mantle_aws_region_fallback(env: TestEnv) -> None:
    """Environment fallback is resolved before a request, so a recording cannot distinguish it."""
    env.remove('AWS_DEFAULT_REGION')
    env.set('AWS_REGION', 'eu-west-1')
    provider = BedrockMantleProvider(api_key='api-key')
    assert provider.base_url == 'https://bedrock-mantle.eu-west-1.api.aws/openai/v1/'


def test_bedrock_mantle_custom_base_url(env: TestEnv) -> None:
    """Base URL selection is local configuration that a recording cannot distinguish."""
    env.remove('AWS_DEFAULT_REGION')
    env.remove('AWS_REGION')
    provider = BedrockMantleProvider(base_url='https://bedrock-mantle.us-east-1.api.aws/v1', api_key='api-key')
    assert provider.base_url == 'https://bedrock-mantle.us-east-1.api.aws/v1/'


def test_bedrock_mantle_provider_need_api_key(env: TestEnv) -> None:
    """The missing-key guard runs before any request can be recorded."""
    env.remove('AWS_BEARER_TOKEN_BEDROCK')
    with pytest.raises(UserError, match='AWS_BEARER_TOKEN_BEDROCK'):
        BedrockMantleProvider(region_name='us-east-1')


def test_bedrock_mantle_provider_need_region(env: TestEnv) -> None:
    """The missing-region guard runs before any request can be recorded."""
    env.remove('AWS_DEFAULT_REGION')
    env.remove('AWS_REGION')
    with pytest.raises(UserError, match=re.escape('`AWS_DEFAULT_REGION` or `AWS_REGION`')):
        BedrockMantleProvider(api_key='api-key')


def test_bedrock_mantle_provider_invalid_region() -> None:
    """Region validation runs before any request can be recorded."""
    with pytest.raises(UserError, match=re.escape("Invalid AWS region name: 'us-east-1@attacker.example/'")):
        BedrockMantleProvider(region_name='us-east-1@attacker.example/', api_key='api-key')


def test_bedrock_mantle_pass_openai_client() -> None:
    """Client identity is local injection behavior that a recording cannot observe."""
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = BedrockMantleProvider(openai_client=openai_client)
    assert provider.client is openai_client


def test_bedrock_mantle_pass_http_client() -> None:
    """HTTP client identity is local injection behavior that a recording cannot observe."""
    http_client = httpx.AsyncClient()
    provider = BedrockMantleProvider(region_name='us-east-1', api_key='api-key', http_client=http_client)
    assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]


async def test_bedrock_mantle_provider_reopens_http_client() -> None:
    """Client lifecycle happens outside the request and cannot be verified from a recording."""
    provider = BedrockMantleProvider(region_name='us-east-1', api_key='api-key')
    first_http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
    async with provider:
        pass
    assert first_http_client.is_closed

    async with provider:
        assert provider.client._client is not first_http_client  # pyright: ignore[reportPrivateUsage]
        assert not provider.client._client.is_closed  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('model_name', ['gpt-5.6-luna', 'gpt-oss-120b'])
def test_bedrock_mantle_model_profile(model_name: str) -> None:
    """Profile resolution is local behavior that is not represented in a recording."""
    assert BedrockMantleProvider.model_profile(f'openai.{model_name}') == merge_profile(
        openai_model_profile(model_name), ModelProfile(tool_call_id_scope='response')
    )


def test_bedrock_mantle_infer_model(env: TestEnv) -> None:
    """Model routing is resolved locally before a request can be recorded."""
    env.set('AWS_BEARER_TOKEN_BEDROCK', 'api-key')
    env.set('AWS_DEFAULT_REGION', 'us-east-1')
    model = infer_model('bedrock-mantle:openai.gpt-5.6-luna')
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == 'openai.gpt-5.6-luna'
    assert model.system == 'bedrock-mantle'

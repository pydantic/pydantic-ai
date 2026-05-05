from __future__ import annotations as _annotations

import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.zai import zai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models import infer_model
    from pydantic_ai.models.zai import ZaiModel
    from pydantic_ai.providers.zai import ZaiProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@pytest.mark.anyio
async def test_zai_provider():
    provider = ZaiProvider(api_key='api-key')
    assert provider.name == 'zai'
    assert provider.base_url == 'https://api.z.ai/api/paas/v4'
    assert isinstance(provider.client, AsyncOpenAI)
    assert provider.client.api_key == 'api-key'

    first_http_client = provider.client._client  # type: ignore[reportPrivateUsage]
    async with provider:
        assert provider.client._client == first_http_client  # type: ignore[reportPrivateUsage]
        assert not first_http_client.is_closed
    assert first_http_client.is_closed

    async with provider:
        second_http_client = provider.client._client  # type: ignore[reportPrivateUsage]
        assert second_http_client is not first_http_client
        assert not second_http_client.is_closed
    assert second_http_client.is_closed


def test_zai_provider_need_api_key(env: TestEnv) -> None:
    env.remove('ZAI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `ZAI_API_KEY` environment variable or pass it via `ZaiProvider(api_key=...)` '
            'to use the Z.AI provider.'
        ),
    ):
        ZaiProvider()


def test_zai_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = ZaiProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_zai_provider_pass_openai_client() -> None:
    openai_client = AsyncOpenAI(api_key='api-key')
    provider = ZaiProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_zai_provider_model_profile(mocker: MockerFixture):
    openai_client = AsyncOpenAI(api_key='api-key')
    provider = ZaiProvider(openai_client=openai_client)

    ns = 'pydantic_ai.providers.zai'
    zai_model_profile_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)

    profile = provider.model_profile('glm-4.7')
    zai_model_profile_mock.assert_called_with('glm-4.7')
    assert profile is not None
    assert isinstance(profile, OpenAIModelProfile)
    assert profile.json_schema_transformer == OpenAIJsonSchemaTransformer
    assert profile.supports_thinking is True
    assert profile.thinking_always_enabled is False
    assert profile.openai_chat_thinking_field == 'reasoning_content'
    assert profile.openai_chat_send_back_thinking_parts == 'field'

    profile_air = provider.model_profile('glm-4.5-air')
    zai_model_profile_mock.assert_called_with('glm-4.5-air')
    assert profile_air is not None
    assert isinstance(profile_air, OpenAIModelProfile)
    assert profile_air.supports_thinking is True
    assert profile_air.openai_chat_thinking_field == 'reasoning_content'
    assert profile_air.openai_chat_send_back_thinking_parts == 'field'

    # The provider always sets the Z.AI response-shape fields, even on non-thinking
    # models — those fields describe Z.AI's API regardless of whether a given model
    # produces reasoning content. `supports_thinking` is what gates client behavior.
    profile_vision = provider.model_profile('glm-4.6v')
    zai_model_profile_mock.assert_called_with('glm-4.6v')
    assert profile_vision is not None
    assert isinstance(profile_vision, OpenAIModelProfile)
    assert profile_vision.supports_thinking is False
    assert profile_vision.openai_chat_thinking_field == 'reasoning_content'

    profile_non_thinking = provider.model_profile('glm-4-32b-0414-128k')
    zai_model_profile_mock.assert_called_with('glm-4-32b-0414-128k')
    assert profile_non_thinking is not None
    assert isinstance(profile_non_thinking, OpenAIModelProfile)
    assert profile_non_thinking.supports_thinking is False
    assert profile_non_thinking.openai_chat_thinking_field == 'reasoning_content'


def test_infer_zai_model(env: TestEnv):
    env.set('ZAI_API_KEY', 'test-api-key')
    model = infer_model('zai:glm-4.7')
    assert isinstance(model, ZaiModel)
    assert model.model_name == 'glm-4.7'

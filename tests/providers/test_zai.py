from __future__ import annotations as _annotations

import re

import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
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

    first_http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
    async with provider:
        assert provider.client._client == first_http_client  # pyright: ignore[reportPrivateUsage]
        assert not first_http_client.is_closed
    assert first_http_client.is_closed

    async with provider:
        second_http_client = provider.client._client  # pyright: ignore[reportPrivateUsage]
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
    assert profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer
    assert profile.get('supports_thinking') is True
    assert profile.get('thinking_always_enabled', False) is False
    assert profile.get('openai_chat_thinking_field') == 'reasoning_content'
    assert profile.get('openai_chat_send_back_thinking_parts') == 'field'
    # glm-4.7 only supports thinking on/off, so no per-request reasoning effort.
    assert profile.get('zai_supports_reasoning_effort', False) is False

    # GLM-5.2 additionally accepts a per-request reasoning effort level.
    profile_5_2 = provider.model_profile('glm-5.2')
    zai_model_profile_mock.assert_called_with('glm-5.2')
    assert profile_5_2 is not None
    assert profile_5_2.get('supports_thinking') is True
    assert profile_5_2.get('zai_supports_reasoning_effort') is True

    # GLM-5.3 always reasons, no longer supports disabling thinking, and only accepts a subset of the
    # unified effort levels, so it carries an effort mapping.
    profile_5_3 = provider.model_profile('glm-5.3')
    zai_model_profile_mock.assert_called_with('glm-5.3')
    assert profile_5_3 is not None
    assert profile_5_3.get('supports_thinking') is True
    assert profile_5_3.get('thinking_always_enabled') is True
    assert profile_5_3.get('zai_supports_reasoning_effort') is True
    assert profile_5_3.get('zai_reasoning_effort_mapping') == {'minimal': 'low', 'medium': 'high', 'xhigh': 'max'}
    assert profile_5_2.get('zai_reasoning_effort_mapping') is None

    # `glm-5.3-flash` inherits the whole GLM-5.3 profile from the shared prefix.
    profile_5_3_flash = provider.model_profile('glm-5.3-flash')
    zai_model_profile_mock.assert_called_with('glm-5.3-flash')
    assert profile_5_3_flash == profile_5_3

    # `glm-5.1` reasons and can still be told not to, and predates per-request effort.
    profile_5_1 = provider.model_profile('glm-5.1')
    zai_model_profile_mock.assert_called_with('glm-5.1')
    assert profile_5_1 is not None
    assert profile_5_1.get('supports_thinking') is True
    assert profile_5_1.get('thinking_always_enabled', False) is False
    assert profile_5_1.get('zai_supports_reasoning_effort', False) is False

    profile_air = provider.model_profile('glm-4.5-air')
    zai_model_profile_mock.assert_called_with('glm-4.5-air')
    assert profile_air is not None
    assert profile_air.get('supports_thinking') is True
    assert profile_air.get('openai_chat_thinking_field') == 'reasoning_content'
    assert profile_air.get('openai_chat_send_back_thinking_parts') == 'field'

    # Vision models support thinking too, per the Z.AI docs.
    profile_vision = provider.model_profile('glm-4.6v')
    zai_model_profile_mock.assert_called_with('glm-4.6v')
    assert profile_vision is not None
    assert profile_vision.get('supports_thinking') is True
    assert profile_vision.get('openai_chat_thinking_field') == 'reasoning_content'

    # The provider always sets the Z.AI response-shape fields, even on non-thinking
    # models — those fields describe Z.AI's API regardless of whether a given model
    # produces reasoning content. `supports_thinking` is what gates client behavior.
    profile_non_thinking = provider.model_profile('glm-4-32b-0414-128k')
    zai_model_profile_mock.assert_called_with('glm-4-32b-0414-128k')
    assert profile_non_thinking is not None
    assert profile_non_thinking.get('supports_thinking', False) is False
    assert profile_non_thinking.get('openai_chat_thinking_field') == 'reasoning_content'


def test_infer_zai_model(env: TestEnv):
    env.set('ZAI_API_KEY', 'test-api-key')
    model = infer_model('zai:glm-4.7')
    assert isinstance(model, ZaiModel)
    assert model.model_name == 'glm-4.7'

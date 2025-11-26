from __future__ import annotations as _annotations

import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.models import infer_model
    from pydantic_ai.models.cerebras import CerebrasModel
    from pydantic_ai.providers.cerebras import CerebrasProvider, zai_model_profile


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_cerebras_provider():
    provider = CerebrasProvider(api_key='api-key')
    assert provider.name == 'cerebras'
    assert provider.base_url == 'https://api.cerebras.ai/v1'
    assert isinstance(provider.client, AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_cerebras_provider_need_api_key(env: TestEnv) -> None:
    env.remove('CEREBRAS_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `CEREBRAS_API_KEY` environment variable or pass it via `CerebrasProvider(api_key=...)` '
            'to use the Cerebras provider.'
        ),
    ):
        CerebrasProvider()


def test_cerebras_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = CerebrasProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_cerebras_provider_pass_openai_client() -> None:
    openai_client = AsyncOpenAI(api_key='api-key')
    provider = CerebrasProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_cerebras_provider_model_profile(mocker: MockerFixture):
    openai_client = AsyncOpenAI(api_key='api-key')
    provider = CerebrasProvider(openai_client=openai_client)

    ns = 'pydantic_ai.providers.cerebras'
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    harmony_model_profile_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)
    zai_model_profile_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)

    # Test llama model - uses meta profile which has InlineDefsJsonSchemaTransformer
    meta_profile = provider.model_profile('llama-3.3-70b')
    meta_model_profile_mock.assert_called_with('llama-3.3-70b')
    assert meta_profile is not None
    assert isinstance(meta_profile, OpenAIModelProfile)
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test qwen model - uses qwen profile which has InlineDefsJsonSchemaTransformer
    qwen_profile = provider.model_profile('qwen-3-32b')
    qwen_model_profile_mock.assert_called_with('qwen-3-32b')
    assert qwen_profile is not None
    assert isinstance(qwen_profile, OpenAIModelProfile)
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test gpt-oss model (harmony) - uses OpenAIJsonSchemaTransformer
    harmony_profile = provider.model_profile('gpt-oss-120b')
    harmony_model_profile_mock.assert_called_with('gpt-oss-120b')
    assert harmony_profile is not None
    assert isinstance(harmony_profile, OpenAIModelProfile)
    assert harmony_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test zai model
    zai_profile = provider.model_profile('zai-glm-4.6')
    zai_model_profile_mock.assert_called_with('zai-glm-4.6')
    assert zai_profile is not None
    assert isinstance(zai_profile, OpenAIModelProfile)
    assert zai_profile.supports_json_object_output is True
    assert zai_profile.supports_json_schema_output is True
    assert zai_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test unknown model - should still return a profile with OpenAIJsonSchemaTransformer
    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert isinstance(unknown_profile, OpenAIModelProfile)
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Verify unsupported model settings are set for all profiles
    for profile in [meta_profile, qwen_profile, harmony_profile, zai_profile, unknown_profile]:
        assert isinstance(profile, OpenAIModelProfile)
        assert 'frequency_penalty' in profile.openai_unsupported_model_settings
        assert 'logit_bias' in profile.openai_unsupported_model_settings
        assert 'presence_penalty' in profile.openai_unsupported_model_settings
        assert 'parallel_tool_calls' in profile.openai_unsupported_model_settings
        assert 'service_tier' in profile.openai_unsupported_model_settings


def test_infer_cerebras_model(env: TestEnv):
    """Test that infer_model correctly creates a CerebrasModel from a model name string."""
    env.set('CEREBRAS_API_KEY', 'test-api-key')
    model = infer_model('cerebras:llama-3.3-70b')
    assert isinstance(model, CerebrasModel)
    assert model.model_name == 'llama-3.3-70b'

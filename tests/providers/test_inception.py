import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.inception import inception_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.inception import InceptionProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_inception_provider():
    provider = InceptionProvider(api_key='your-api-key')
    assert provider.name == 'inception'
    assert provider.base_url == 'https://api.inceptionlabs.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'


def test_inception_provider_need_api_key(env: TestEnv) -> None:
    env.remove('INCEPTION_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `INCEPTION_API_KEY` environment variable or pass it via `InceptionProvider(api_key=...)` '
            'to use the Inception provider.'
        ),
    ):
        InceptionProvider()


def test_inception_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='your-api-key')
    provider = InceptionProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_inception_pass_http_client():
    http_client = httpx.AsyncClient()
    provider = InceptionProvider(api_key='your-api-key', http_client=http_client)
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'


def test_inception_model_profile(mocker: MockerFixture):
    provider = InceptionProvider(api_key='your-api-key')

    inception_mock = mocker.patch(
        'pydantic_ai.providers.inception.inception_model_profile', wraps=inception_model_profile
    )

    # Mercury 2 supports schema-aligned JSON output and thinking
    profile = provider.model_profile('Mercury-2')
    inception_mock.assert_called_with('mercury-2')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert profile.get('openai_chat_supports_max_completion_tokens') is False
    assert profile.get('supports_json_schema_output') is True
    assert profile.get('supports_thinking') is True

    # Legacy mercury models don't
    legacy_profile = provider.model_profile('mercury-coder')
    inception_mock.assert_called_with('mercury-coder')
    assert legacy_profile is not None
    assert legacy_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert legacy_profile.get('supports_json_schema_output') is False
    assert legacy_profile.get('supports_thinking') is False

    # Test unknown model
    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert 'supports_json_schema_output' not in unknown_profile
    assert 'supports_thinking' not in unknown_profile

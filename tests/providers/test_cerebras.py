import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.cerebras import CerebrasProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_cerebras_provider():
    provider = CerebrasProvider(api_key='ghp_test_token')
    assert provider.name == 'cerebras'
    assert provider.base_url == 'https://api.cerebras.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'ghp_test_token'


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


def test_github_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = CerebrasProvider(http_client=http_client, api_key='ghp_test_token')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_github_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='ghp_test_token')
    provider = CerebrasProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_cerebras_provider_model_profile(mocker: MockerFixture):
    provider = CerebrasProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.cerebras'
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    harmony_model_profile_mock = mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile)

    meta_profile = provider.model_profile('llama4-maverick-instruct-basic')
    meta_model_profile_mock.assert_called_with('llama4-maverick-instruct-basic')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    qwen_profile = provider.model_profile('qwen3-235b-a22b')
    qwen_model_profile_mock.assert_called_with('qwen3-235b-a22b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    qwen_profile = provider.model_profile('qwen-3-coder-480b')
    qwen_model_profile_mock.assert_called_with('qwen-3-coder-480b')
    assert qwen_profile is not None
    assert qwen_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    openai_profile = provider.model_profile('gpt-oss-120b')
    harmony_model_profile_mock.assert_called_with('gpt-oss-120b')
    assert openai_profile is not None
    assert openai_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

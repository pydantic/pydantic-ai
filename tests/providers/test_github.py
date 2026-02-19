import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.cohere import cohere_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.github import GitHubProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_github_provider():
    provider = GitHubProvider(api_key='ghp_test_token')
    assert provider.name == 'github'
    assert provider.base_url == 'https://models.github.ai/inference'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'ghp_test_token'


def test_github_provider_need_api_key(env: TestEnv) -> None:
    env.remove('GITHUB_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GITHUB_API_KEY` environment variable or pass it via `GitHubProvider(api_key=...)`'
            ' to use the GitHub Models provider.'
        ),
    ):
        GitHubProvider()


def test_github_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = GitHubProvider(http_client=http_client, api_key='ghp_test_token')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_github_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='ghp_test_token')
    provider = GitHubProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_github_provider_model_profile(mocker: MockerFixture):
    provider = GitHubProvider(api_key='ghp_test_token')

    ns = 'pydantic_ai.providers.github'
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    cohere_model_profile_mock = mocker.patch(f'{ns}.cohere_model_profile', wraps=cohere_model_profile)
    grok_model_profile_mock = mocker.patch(f'{ns}.grok_model_profile', wraps=grok_model_profile)
    openai_model_profile_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)

    meta_profile = provider.model_profile('meta/Llama-3.2-11B-Vision-Instruct')
    meta_model_profile_mock.assert_called_with('llama-3.2-11b-vision-instruct')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('meta/Llama-3.1-405B-Instruct')
    meta_model_profile_mock.assert_called_with('llama-3.1-405b-instruct')
    assert meta_profile is not None
    assert meta_profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    deepseek_profile = provider.model_profile('deepseek/deepseek-coder')
    deepseek_model_profile_mock.assert_called_with('deepseek-coder')
    assert deepseek_profile is not None
    assert deepseek_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistral-ai/mixtral-8x7b-instruct')
    mistral_model_profile_mock.assert_called_with('mixtral-8x7b-instruct')
    assert mistral_profile is not None
    assert mistral_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    cohere_profile = provider.model_profile('cohere/command-r-plus')
    cohere_model_profile_mock.assert_called_with('command-r-plus')
    assert cohere_profile is not None
    assert cohere_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    grok_profile = provider.model_profile('xai/grok-3-mini')
    grok_model_profile_mock.assert_called_with('grok-3-mini')
    assert grok_profile is not None
    assert grok_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    microsoft_profile = provider.model_profile('microsoft/Phi-3.5-mini-instruct')
    openai_model_profile_mock.assert_called_with('phi-3.5-mini-instruct')
    assert microsoft_profile is not None
    assert microsoft_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('some-unknown-model')
    openai_model_profile_mock.assert_called_with('some-unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    unknown_profile_with_prefix = provider.model_profile('unknown-publisher/some-unknown-model')
    openai_model_profile_mock.assert_called_with('some-unknown-model')
    assert unknown_profile_with_prefix is not None
    assert unknown_profile_with_prefix.json_schema_transformer == OpenAIJsonSchemaTransformer

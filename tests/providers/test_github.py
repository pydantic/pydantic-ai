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
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('meta/Llama-3.1-405B-Instruct')
    meta_model_profile_mock.assert_called_with('llama-3.1-405b-instruct')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    deepseek_profile = provider.model_profile('deepseek/deepseek-coder')
    deepseek_model_profile_mock.assert_called_with('deepseek-coder')
    assert deepseek_profile is not None
    assert deepseek_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    mistral_profile = provider.model_profile('mistral-ai/mixtral-8x7b-instruct')
    mistral_model_profile_mock.assert_called_with('mixtral-8x7b-instruct')
    assert mistral_profile is not None
    assert mistral_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    cohere_profile = provider.model_profile('cohere/command-r-plus')
    cohere_model_profile_mock.assert_called_with('command-r-plus')
    assert cohere_profile is not None
    assert cohere_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    grok_profile = provider.model_profile('xai/grok-3-mini')
    grok_model_profile_mock.assert_called_with('grok-3-mini')
    assert grok_profile is not None
    assert grok_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    openai_profile = provider.model_profile('openai/o3')
    openai_model_profile_mock.assert_called_with('o3')
    assert openai_profile is not None
    assert openai_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    microsoft_profile = provider.model_profile('microsoft/Phi-3.5-mini-instruct')
    openai_model_profile_mock.assert_called_with('phi-3.5-mini-instruct')
    assert microsoft_profile is not None
    assert microsoft_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    unknown_profile = provider.model_profile('some-unknown-model')
    openai_model_profile_mock.assert_called_with('some-unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    openai_model_profile_mock.reset_mock()
    unknown_profile_with_prefix = provider.model_profile('unknown-publisher/some-unknown-model')
    # An unrecognised publisher gets no family profile at all, only the OpenAI-compatible base.
    # Without the reset, `assert_called_with` would pass against the previous unprefixed lookup.
    openai_model_profile_mock.assert_not_called()
    assert unknown_profile_with_prefix is not None
    assert unknown_profile_with_prefix.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


# The `openai/`-prefixed names GitHub actually publishes, one per capability shape that
# `openai_model_profile` distinguishes: a reasoning model that can't disable reasoning, one whose
# system role is remapped, and a non-reasoning model.
# https://models.github.ai/catalog/models
@pytest.mark.parametrize('model_name', ['openai/o3', 'openai/o1-mini', 'openai/gpt-4.1'])
def test_github_openai_prefixed_models_resolve_the_openai_profile(model_name: str):
    """GitHub publishes OpenAI's own models under `openai/`, which must reach `openai_model_profile`.

    The publisher map had no `openai` entry, so these fell through to the OpenAI-compatible base
    alone and every capability flag resolved to unset. `supports_thinking` unset means the unified
    `thinking` setting is silently dropped, so `Agent('github:openai/o3')` with `thinking='high'`
    sent no `reasoning_effort` at all — a no-op with no error, on the largest publisher in the
    catalog. `openai_system_prompt_role` unset means `o1-mini` keeps the `system` role its API
    rejects.

    A unit test rather than a VCR one: profile resolution happens before any request, and the
    defect is a *missing* request field, which the cassette matchers aren't sensitive to — a
    recording made against the broken code plays back identically.
    """
    profile = GitHubProvider.model_profile(model_name)
    assert profile is not None
    expected = openai_model_profile(model_name.removeprefix('openai/'))
    # Compared whole rather than flag by flag so a capability added to `openai_model_profile`
    # later is covered here without this test being updated.
    assert {key: profile.get(key) for key in expected} == expected

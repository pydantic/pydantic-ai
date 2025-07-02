import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.github_models import GitHubModelsProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_github_models_provider():
    provider = GitHubModelsProvider(api_key='ghp_test_token')
    assert provider.name == 'github_models'
    assert provider.base_url == 'https://models.github.ai/inference'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'ghp_test_token'


def test_github_models_provider_need_api_key(env: TestEnv) -> None:
    env.remove('GITHUB_TOKEN')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GITHUB_TOKEN` environment variable or pass it via `GitHubModelsProvider(api_key=...)`'
            ' to use the GitHub Models provider.'
        ),
    ):
        GitHubModelsProvider()


def test_github_models_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = GitHubModelsProvider(http_client=http_client, api_key='ghp_test_token')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_github_models_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='ghp_test_token')
    provider = GitHubModelsProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_github_models_model_profiles():
    from pydantic_ai.profiles._json_schema import InlineDefsJsonSchemaTransformer

    provider = GitHubModelsProvider(api_key='ghp_test_token')

    # Test Grok models (returns None profile, so gets OpenAI defaults)
    grok_model = OpenAIModel('xai/grok-3-mini', provider=provider)
    assert isinstance(grok_model.profile, OpenAIModelProfile)
    assert grok_model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test Meta models (returns profile with InlineDefsJsonSchemaTransformer)
    meta_model = OpenAIModel('meta/Llama-3.2-11B-Vision-Instruct', provider=provider)
    assert isinstance(meta_model.profile, OpenAIModelProfile)
    assert meta_model.profile.json_schema_transformer == InlineDefsJsonSchemaTransformer

    # Test Microsoft models (uses OpenAI profile)
    microsoft_model = OpenAIModel('microsoft/Phi-3.5-mini-instruct', provider=provider)
    assert isinstance(microsoft_model.profile, OpenAIModelProfile)
    assert microsoft_model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer

    # Test default fallback
    unknown_model = OpenAIModel('some-unknown-model', provider=provider)
    assert isinstance(unknown_model.profile, OpenAIModelProfile)
    assert unknown_model.profile.json_schema_transformer == OpenAIJsonSchemaTransformer

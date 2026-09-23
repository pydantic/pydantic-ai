import re

import httpx2
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.grok import grok_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.opper import OpperProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


def test_opper_provider():
    provider = OpperProvider(api_key='your-api-key')
    assert provider.name == 'opper'
    assert provider.base_url == 'https://api.opper.ai/v3/compat'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'your-api-key'


def test_opper_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OPPER_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OPPER_API_KEY` environment variable or pass it via '
            '`OpperProvider(api_key=...)` to use the Opper provider.'
        ),
    ):
        OpperProvider()


def test_opper_provider_from_env(env: TestEnv) -> None:
    env.set('OPPER_API_KEY', 'env-api-key')
    provider = OpperProvider()
    assert provider.client.api_key == 'env-api-key'


def test_opper_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='your-api-key')
    provider = OpperProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_opper_pass_http_client():
    http_client = httpx2.AsyncClient()
    provider = OpperProvider(api_key='your-api-key', http_client=http_client)
    assert isinstance(provider.client, openai.AsyncOpenAI)


@pytest.mark.parametrize(
    ('model_name', 'profile_func'),
    (
        ('claude-sonnet-4-6', anthropic_model_profile),
        ('gemini-3.8-flash', google_model_profile),
        ('gemma-3-27b-it', google_model_profile),
        ('deepseek-v4-pro', deepseek_model_profile),
        ('kimi-k3', moonshotai_model_profile),
        ('qwen3-max', qwen_model_profile),
        ('grok-4.5', grok_model_profile),
        ('glm-5.2', zai_model_profile),
        ('llama-3.3-70b-instruct', meta_model_profile),
        ('mistral-large-2512', mistral_model_profile),
        ('ministral-8b', mistral_model_profile),
        ('gpt-5.5', openai_model_profile),
        ('o3-mini', openai_model_profile),
    ),
)
def test_opper_model_profile_matches_the_model_family(model_name: str, profile_func):
    """Opper ids are bare pool names, so the family is read off the name itself."""
    profile = OpperProvider.model_profile(model_name)
    assert profile is not None
    expected = profile_func(model_name)
    assert expected is not None
    assert profile.json_schema_transformer is OpenAIJsonSchemaTransformer


def test_opper_gpt_oss_resolves_to_harmony_not_openai():
    """`gpt-oss` must win over the shorter `gpt` prefix."""
    profile = OpperProvider.model_profile('gpt-oss-120b')
    expected = harmony_model_profile('gpt-oss-120b')
    assert expected is not None
    assert profile is not None
    assert profile.json_schema_transformer is OpenAIJsonSchemaTransformer


def test_opper_route_pinned_id_resolves_the_same_family():
    """A `provider/model` id pins a route; the family still comes from the model name."""
    pooled = OpperProvider.model_profile('gpt-5.5')
    pinned = OpperProvider.model_profile('azure/gpt-5.5')
    assert pooled is not None
    assert pinned is not None
    assert pinned.json_schema_transformer is pooled.json_schema_transformer

    pinned_claude = OpperProvider.model_profile('aws/claude-sonnet-4-6-eu')
    assert pinned_claude is not None


def test_opper_unknown_model_still_gets_the_openai_transformer():
    profile = OpperProvider.model_profile('some-unlisted-model')
    assert profile is not None
    assert profile.json_schema_transformer is OpenAIJsonSchemaTransformer

import re

import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.otari import OtariProvider, OtariRegion


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_otari_provider(env: TestEnv) -> None:
    env.remove('OTARI_BASE_URL')
    provider = OtariProvider(api_key='otari-key')
    assert provider.name == 'otari'
    assert provider.base_url == 'https://api.otari.ai/api/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'otari-key'


def test_otari_provider_reads_the_environment(env: TestEnv) -> None:
    env.set('OTARI_API_KEY', 'env-key')
    env.set('OTARI_BASE_URL', 'http://localhost:8000/api/v1')
    provider = OtariProvider()
    assert provider.client.api_key == 'env-key'
    assert provider.base_url == 'http://localhost:8000/api/v1'


def test_otari_provider_need_api_key(env: TestEnv) -> None:
    env.remove('OTARI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `OTARI_API_KEY` environment variable or pass it via '
            '`OtariProvider(api_key=...)` to use the Otari provider.'
        ),
    ):
        OtariProvider()


@pytest.mark.parametrize(
    ('region', 'base_url'),
    [('us', 'https://api.otari.ai/api/v1'), ('eu', 'https://eu.api.otari.ai/api/v1')],
)
def test_otari_region(env: TestEnv, region: OtariRegion, base_url: str) -> None:
    # An explicit region wins over `OTARI_BASE_URL`.
    env.set('OTARI_BASE_URL', 'http://localhost:8000/api/v1')
    provider = OtariProvider(api_key='otari-key', region=region)
    assert provider.base_url == base_url
    assert str(provider.client.base_url) == f'{base_url}/'


def test_otari_region_and_base_url_conflict() -> None:
    with pytest.raises(UserError, match=re.escape('Pass either `region` or `base_url`/`openai_client`')):
        OtariProvider(api_key='otari-key', base_url='http://localhost:8000/api/v1', region='eu')


def test_otari_self_hosted_base_url() -> None:
    provider = OtariProvider(api_key='otari-key', base_url='https://otari.internal.example/api/v1')
    assert provider.base_url == 'https://otari.internal.example/api/v1'


def test_otari_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='otari-key', base_url='https://otari.internal.example/api/v1')
    provider = OtariProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert provider.base_url == 'https://otari.internal.example/api/v1/'


@pytest.mark.parametrize(
    ('model_name', 'json_schema_transformer'),
    [
        # Otari's own `provider:model` form and the `provider/model` form pick the same family profile.
        ('anthropic:claude-sonnet-5-5', OpenAIJsonSchemaTransformer),
        ('anthropic/claude-sonnet-5-5', OpenAIJsonSchemaTransformer),
        # Otari's own provider ids, which the shared vendor map spells differently.
        ('gemini:gemini-2.5-flash', GoogleJsonSchemaTransformer),
        ('vertexai:gemini-2.5-flash', GoogleJsonSchemaTransformer),
        # A routing policy or alias name, and an unknown provider, get the OpenAI-compatible default.
        ('my-routing-policy', OpenAIJsonSchemaTransformer),
        ('unknown:model', OpenAIJsonSchemaTransformer),
    ],
)
def test_otari_model_profile(model_name: str, json_schema_transformer: type) -> None:
    profile = OtariProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('json_schema_transformer') is json_schema_transformer


def test_otari_model_profile_reads_both_name_forms_alike() -> None:
    assert OtariProvider.model_profile('anthropic:claude-sonnet-5-5') == OtariProvider.model_profile(
        'anthropic/claude-sonnet-5-5'
    )

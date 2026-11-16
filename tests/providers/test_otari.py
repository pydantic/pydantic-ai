import re

import httpx2
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles._family import _VENDOR_FAMILY_PROFILES  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.otari import (
        _OTARI_VENDOR_PROFILES,  # pyright: ignore[reportPrivateUsage]
        OtariProvider,
    )


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
def test_otari_region(env: TestEnv, region: str, base_url: str) -> None:
    # An explicit region wins over `OTARI_BASE_URL`.
    env.set('OTARI_BASE_URL', 'http://localhost:8000/api/v1')
    provider = OtariProvider(api_key='otari-key', region=region)  # pyright: ignore[reportArgumentType]
    assert provider.base_url == base_url
    assert str(provider.client.base_url) == f'{base_url}/'


def test_otari_unknown_region() -> None:
    with pytest.raises(UserError, match=re.escape("Unknown Otari region 'ap'; expected one of us, eu.")):
        OtariProvider(api_key='otari-key', region='ap')  # pyright: ignore[reportArgumentType]


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


def test_otari_pass_http_client() -> None:
    http_client = httpx2.AsyncClient()
    provider = OtariProvider(api_key='otari-key', http_client=http_client)
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'otari-key'


@pytest.mark.parametrize(
    ('model_name', 'family_function', 'family_model_name'),
    [
        ('anthropic:claude-sonnet-5-5', 'anthropic_model_profile', 'claude-sonnet-5-5'),
        ('anthropic/claude-sonnet-5-5', 'anthropic_model_profile', 'claude-sonnet-5-5'),
        ('openai:gpt-5.2', 'openai_model_profile', 'gpt-5.2'),
        ('mistral:mistral-large-latest', 'mistral_model_profile', 'mistral-large-latest'),
    ],
)
def test_otari_model_profile_uses_the_shared_vendor_map(
    mocker: MockerFixture, model_name: str, family_function: str, family_model_name: str
) -> None:
    vendor = model_name.replace(':', '/').split('/', 1)[0]
    mock = mocker.Mock(wraps=_VENDOR_FAMILY_PROFILES[vendor])
    mocker.patch.dict(_VENDOR_FAMILY_PROFILES, {vendor: mock})
    profile = OtariProvider.model_profile(model_name)
    mock.assert_called_once_with(family_model_name)
    assert profile is not None
    assert profile.get('json_schema_transformer') is not None


@pytest.mark.parametrize(
    ('model_name', 'family_model_name'),
    [
        ('gemini:gemini-2.5-flash', 'gemini-2.5-flash'),
        ('vertexai:gemini-2.5-flash', 'gemini-2.5-flash'),
        ('bedrock:anthropic.claude-sonnet-5-5', 'anthropic.claude-sonnet-5-5'),
        ('azure:gpt-5.2', 'gpt-5.2'),
        ('groq:llama-3.3-70b-versatile', 'llama-3.3-70b-versatile'),
        ('moonshot:kimi-k2.6', 'kimi-k2.6'),
    ],
)
def test_otari_model_profile_maps_otari_provider_ids(
    mocker: MockerFixture, model_name: str, family_model_name: str
) -> None:
    vendor = model_name.split(':', 1)[0]
    mock = mocker.Mock(wraps=_OTARI_VENDOR_PROFILES[vendor])
    mocker.patch.dict(_OTARI_VENDOR_PROFILES, {vendor: mock})
    OtariProvider.model_profile(model_name)
    mock.assert_called_once_with(family_model_name)


def test_otari_model_profile_google_family() -> None:
    profile = OtariProvider.model_profile('gemini:gemini-2.5-flash')
    assert profile is not None
    assert profile.get('json_schema_transformer') is GoogleJsonSchemaTransformer


@pytest.mark.parametrize('model_name', ['my-routing-policy', 'unknown:model', 'unknown/model'])
def test_otari_model_profile_without_a_known_vendor(model_name: str) -> None:
    profile = OtariProvider.model_profile(model_name)
    assert profile is not None
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer


def test_otari_model_name_is_sent_unchanged() -> None:
    model = OpenAIChatModel('anthropic:claude-sonnet-5-5', provider=OtariProvider(api_key='otari-key'))
    assert model.model_name == 'anthropic:claude-sonnet-5-5'
    assert model.system == 'otari'

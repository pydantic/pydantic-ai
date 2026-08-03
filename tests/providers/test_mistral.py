from __future__ import annotations as _annotations

import re

import httpx
import pytest

from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from mistralai.client import Mistral

    from pydantic_ai.providers.mistral import MistralProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='mistral not installed')


def test_mistral_provider():
    provider = MistralProvider(api_key='api-key')
    assert provider.name == 'mistral'
    assert provider.base_url == 'https://api.mistral.ai'
    assert isinstance(provider.client, Mistral)
    assert provider.client.sdk_configuration.security.api_key == 'api-key'  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]


def test_mistral_provider_need_api_key(env: TestEnv) -> None:
    env.remove('MISTRAL_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `MISTRAL_API_KEY` environment variable or pass it via `MistralProvider(api_key=...)`'
            ' to use the Mistral provider.'
        ),
    ):
        MistralProvider()


def test_mistral_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = MistralProvider(http_client=http_client, api_key='api-key')
    assert provider.client.sdk_configuration.async_client == http_client


def test_mistral_provider_pass_groq_client() -> None:
    mistral_client = Mistral(api_key='api-key')
    provider = MistralProvider(mistral_client=mistral_client)
    assert provider.client == mistral_client


def test_mistral_provider_with_base_url() -> None:
    # Test with environment variable for base_url
    provider = MistralProvider(
        mistral_client=Mistral(api_key='test-api-key', server_url='https://custom.mistral.com/v1'),
    )
    assert provider.base_url == 'https://custom.mistral.com/v1'


def test_mistral_provider_model_profile_sets_inline_flag():
    profile = MistralProvider.model_profile('mistral-large-latest')
    assert profile.get('supports_inline_system_prompts', False) is True
    assert profile.get('supports_thinking', False) is False

    magistral_profile = MistralProvider.model_profile('magistral-medium-2509')
    assert magistral_profile.get('supports_inline_system_prompts', False) is True
    assert magistral_profile.get('supports_thinking', False) is True

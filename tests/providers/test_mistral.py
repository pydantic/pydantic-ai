from __future__ import annotations as _annotations

import re

import httpx
import httpx2
import pytest

from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.exceptions import UserError

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from mistralai.client import Mistral

    from pydantic_ai.providers.mistral import MistralProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='mistral not installed')


async def test_mistral_provider():
    provider = MistralProvider(api_key='api-key')
    assert provider.name == 'mistral'
    assert provider.base_url == 'https://api.mistral.ai'
    assert isinstance(provider.client, Mistral)
    assert provider.client.sdk_configuration.security.api_key == 'api-key'  # pyright: ignore[reportFunctionMemberAccess, reportOptionalMemberAccess]
    assert isinstance(provider.client.sdk_configuration.async_client, httpx2.AsyncClient)

    async with provider:
        pass

    assert provider.client.sdk_configuration.async_client.is_closed


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


async def test_mistral_provider_pass_httpx2_client() -> None:
    async with httpx2.AsyncClient() as http_client:
        provider = MistralProvider(http_client=http_client, api_key='api-key')
        assert provider.client.sdk_configuration.async_client is http_client

        async with provider:
            pass

        assert not http_client.is_closed


async def test_mistral_provider_deprecates_legacy_httpx_client() -> None:
    async with httpx.AsyncClient() as http_client:
        with pytest.warns(
            PydanticAIDeprecationWarning,
            match=r'`httpx\.AsyncClient`.*removed in v3.*`httpx2\.AsyncClient`',
        ) as warnings:
            provider = MistralProvider(http_client=http_client, api_key='api-key')

        assert warnings[0].filename == __file__
        assert provider.client.sdk_configuration.async_client is http_client
        async with provider:
            pass
        assert not http_client.is_closed


async def test_mistral_provider_pass_mistral_client() -> None:
    async with httpx.AsyncClient() as http_client:
        mistral_client = Mistral(api_key='api-key', async_client=http_client)
        provider = MistralProvider(mistral_client=mistral_client)
        assert provider.client is mistral_client

        async with provider:
            pass

        assert provider.client.sdk_configuration.async_client is http_client
        assert not http_client.is_closed


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

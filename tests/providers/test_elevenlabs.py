from __future__ import annotations as _annotations

import httpx2
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.elevenlabs import ElevenLabsProvider

from ..conftest import TestEnv

pytestmark = pytest.mark.anyio


def test_elevenlabs_provider(env: TestEnv):
    env.set('ELEVENLABS_API_KEY', 'env-api-key')
    provider = ElevenLabsProvider()
    assert provider.name == 'elevenlabs'
    assert provider.base_url == 'https://api.elevenlabs.io'
    assert provider.api_key == 'env-api-key'
    assert isinstance(provider.client, httpx2.AsyncClient)


def test_elevenlabs_provider_need_api_key(env: TestEnv) -> None:
    env.remove('ELEVENLABS_API_KEY')
    with pytest.raises(UserError, match='ELEVENLABS_API_KEY'):
        ElevenLabsProvider()


def test_elevenlabs_provider_explicit_key_and_regional_base_url() -> None:
    provider = ElevenLabsProvider(api_key='api-key', base_url='https://api.eu.residency.elevenlabs.io/')
    assert provider.api_key == 'api-key'
    # A trailing slash is normalized away so REST paths and the WebSocket URL join cleanly.
    assert provider.base_url == 'https://api.eu.residency.elevenlabs.io'


async def test_elevenlabs_provider_passes_http_client_through() -> None:
    async with httpx2.AsyncClient() as http_client:
        provider = ElevenLabsProvider(api_key='api-key', http_client=http_client)
        assert provider.client is http_client
        # A caller-owned client is not adopted for lifecycle management.
        assert provider._own_http_client is None  # pyright: ignore[reportPrivateUsage]


async def test_elevenlabs_provider_replaces_recreated_http_client() -> None:
    # The HTTP client *is* the provider's client, so the base class's client re-creation on re-entry
    # replaces it directly.
    provider = ElevenLabsProvider(api_key='api-key')
    async with httpx2.AsyncClient() as replacement:
        provider._set_http_client(replacement)  # pyright: ignore[reportPrivateUsage]
        assert provider.client is replacement


def test_elevenlabs_provider_realtime_profile() -> None:
    profile = ElevenLabsProvider.realtime_model_profile('agent_123')
    assert profile.get('supports_text_output') is True
    assert profile.get('supports_manual_turn_control') is False
    assert profile.get('supports_interruption') is False
    assert profile.get('supports_output_truncation') is False
    assert profile.get('supports_session_seeding') is False
    assert profile.get('audio_input_sample_rate') == 16000
    assert profile.get('audio_output_sample_rate') == 16000

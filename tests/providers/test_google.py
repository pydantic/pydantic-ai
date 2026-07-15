from __future__ import annotations as _annotations

from unittest.mock import patch

import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from google.genai.types import HttpRetryOptions

    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

    class FakeCredentials:
        """Minimal stand-in for `google.auth.credentials.Credentials`."""

        def __init__(self, *, scopes: list[str] | None = None) -> None:
            self.scopes = scopes
            self.scoped_with: list[str] | None = None

        def with_scopes(self, scopes: list[str]) -> FakeCredentials:
            return FakeCredentials(scopes=scopes)


pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

# `retry_options` only changes behavior on transient 429/5xx responses, which a recorded cassette
# can't reliably reproduce, so these unit tests assert the resolved HTTP config directly via the
# SDK's `get_read_only_http_options()` accessor rather than running an agent against a cassette.


def test_google_provider_retry_options(env: TestEnv):
    env.set('GOOGLE_API_KEY', 'test-key')
    retry = HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=30.0)
    provider = GoogleProvider(api_key='test-key', retry_options=retry)
    assert provider.name == 'google'
    opts = provider.client._api_client.get_read_only_http_options()  # pyright: ignore[reportPrivateUsage]
    assert opts['retry_options']['attempts'] == 4
    assert opts['retry_options']['initial_delay'] == 2.0
    assert opts['retry_options']['max_delay'] == 30.0


def test_google_provider_no_retry_options(env: TestEnv):
    env.set('GOOGLE_API_KEY', 'test-key')
    provider = GoogleProvider(api_key='test-key')
    opts = provider.client._api_client.get_read_only_http_options()  # pyright: ignore[reportPrivateUsage]
    assert opts['retry_options'] is None


def test_google_cloud_provider_retry_options():
    retry = HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=30.0)
    provider = GoogleCloudProvider(project='pydantic-ai', location='us-central1', retry_options=retry)
    assert provider.name == 'google-cloud'
    opts = provider.client._api_client.get_read_only_http_options()  # pyright: ignore[reportPrivateUsage]
    assert opts['retry_options']['attempts'] == 4
    assert opts['retry_options']['initial_delay'] == 2.0
    assert opts['retry_options']['max_delay'] == 30.0


def test_google_cloud_provider_no_retry_options():
    provider = GoogleCloudProvider(project='pydantic-ai', location='us-central1')
    opts = provider.client._api_client.get_read_only_http_options()  # pyright: ignore[reportPrivateUsage]
    assert opts['retry_options'] is None


def test_google_cloud_provider_scopes_credentials(env: TestEnv):
    """Unscoped credentials are scoped with the cloud-platform scope before reaching the Client."""
    env.set('GOOGLE_CLOUD_PROJECT', 'pydantic-ai')
    env.set('GOOGLE_CLOUD_LOCATION', 'us-central1')
    credentials = FakeCredentials()

    captured: dict[str, object] = {}

    class _CapturingClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    with patch('pydantic_ai.providers.google_cloud.Client', _CapturingClient):
        GoogleCloudProvider(credentials=credentials, project='pydantic-ai')

    forwarded_credentials = captured['credentials']
    assert forwarded_credentials is not credentials
    assert forwarded_credentials.scopes == ['https://www.googleapis.com/auth/cloud-platform']


def test_google_cloud_provider_preserves_existing_scopes(env: TestEnv):
    """Credentials that already have scopes are forwarded untouched (no re-scoping)."""
    env.set('GOOGLE_CLOUD_PROJECT', 'pydantic-ai')
    env.set('GOOGLE_CLOUD_LOCATION', 'us-central1')
    credentials = FakeCredentials(scopes=['https://www.googleapis.com/auth/devstorage.read_only'])

    captured: dict[str, object] = {}

    class _CapturingClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    with patch('pydantic_ai.providers.google_cloud.Client', _CapturingClient):
        GoogleCloudProvider(credentials=credentials, project='pydantic-ai')

    forwarded_credentials = captured['credentials']
    assert forwarded_credentials is credentials
    assert forwarded_credentials.scopes == ['https://www.googleapis.com/auth/devstorage.read_only']


def test_google_cloud_provider_string_shortcut_uses_adc(env: TestEnv):
    """The bare `google-cloud:` path does not pass an api_key, letting the SDK use ADC."""
    env.set('GOOGLE_API_KEY', 'should-be-ignored')
    env.set('GEMINI_API_KEY', 'also-ignored')
    env.set('GOOGLE_CLOUD_PROJECT', 'pydantic-ai')
    env.set('GOOGLE_CLOUD_LOCATION', 'us-central1')

    captured: dict[str, object] = {}

    class _CapturingClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    with patch('pydantic_ai.providers.google_cloud.Client', _CapturingClient):
        GoogleCloudProvider()

    assert captured['api_key'] is None
    assert captured['vertexai'] is True


def test_google_cloud_provider_explicit_api_key_still_passed(env: TestEnv):
    """An explicitly-passed `api_key` (Express Mode) still reaches the Client unchanged."""
    captured: dict[str, object] = {}

    class _CapturingClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    with patch('pydantic_ai.providers.google_cloud.Client', _CapturingClient):
        GoogleCloudProvider(api_key='your-api-key')

    assert captured['api_key'] == 'your-api-key'
    assert captured['vertexai'] is True

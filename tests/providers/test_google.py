from __future__ import annotations as _annotations

import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from google.genai.types import HttpRetryOptions

    from pydantic_ai.exceptions import UserError
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')

# `retry_options` only changes behavior on transient 429/5xx responses, which a recorded cassette
# can't reliably reproduce, so these unit tests assert the resolved HTTP config directly via the
# SDK's `get_read_only_http_options()` accessor rather than running an agent against a cassette.


def test_google_provider_without_api_key_raises_error(env: TestEnv):
    env.remove('GOOGLE_API_KEY')
    env.remove('GEMINI_API_KEY')
    with pytest.raises(
        UserError,
        match=(
            r'Set the `GOOGLE_API_KEY` environment variable or pass it via `GoogleProvider\(api_key=\.\.\.\)`'
            r" to use the Gemini API\. To try Pydantic AI without an API key, use the built-in test model: `Agent\('test'\)`\."
        ),
    ):
        GoogleProvider()  # pyright: ignore[reportCallIssue]  # deliberately no api_key, to test the missing-key error


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

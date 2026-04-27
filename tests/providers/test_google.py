from __future__ import annotations as _annotations

import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from google.genai.types import HttpRetryOptions

    from pydantic_ai.providers.google import GoogleProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='google-genai not installed')


def test_google_provider_retry_options(env: TestEnv):
    env.set('GOOGLE_API_KEY', 'test-key')
    retry = HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=30.0)
    provider = GoogleProvider(api_key='test-key', retry_options=retry)
    assert provider.name == 'google-gla'
    # get_read_only_http_options() is the SDK's public accessor for the
    # resolved HTTP configuration, avoiding deep private attribute access.
    opts = provider.client._api_client.get_read_only_http_options()  # type: ignore[reportPrivateUsage]
    assert opts['retry_options']['attempts'] == 4
    assert opts['retry_options']['initial_delay'] == 2.0
    assert opts['retry_options']['max_delay'] == 30.0


def test_google_provider_no_retry_options(env: TestEnv):
    env.set('GOOGLE_API_KEY', 'test-key')
    provider = GoogleProvider(api_key='test-key')
    opts = provider.client._api_client.get_read_only_http_options()  # type: ignore[reportPrivateUsage]
    assert opts['retry_options'] is None

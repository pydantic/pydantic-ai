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
    # Verify the retry options were passed through to the underlying HttpOptions.
    http_options = provider.client._api_client._http_options  # type: ignore[reportPrivateUsage]
    assert http_options.retry_options is not None
    assert http_options.retry_options.attempts == 4
    assert http_options.retry_options.initial_delay == 2.0
    assert http_options.retry_options.max_delay == 30.0


def test_google_provider_no_retry_options(env: TestEnv):
    env.set('GOOGLE_API_KEY', 'test-key')
    provider = GoogleProvider(api_key='test-key')
    http_options = provider.client._api_client._http_options  # type: ignore[reportPrivateUsage]
    assert http_options.retry_options is None

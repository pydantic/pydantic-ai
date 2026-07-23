from __future__ import annotations as _annotations

from collections.abc import Sequence
from typing import cast
from unittest.mock import patch

import pytest

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from google.auth import crypt
    from google.auth.credentials import AnonymousCredentials
    from google.genai.types import HttpRetryOptions
    from google.oauth2 import service_account

    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models import infer_model
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider
    from pydantic_ai.providers.google_cloud import GoogleCloudProvider

    class FakeSigner(crypt.Signer):
        @property
        def key_id(self) -> str | None:
            raise NotImplementedError

        def sign(self, message: str | bytes) -> bytes:
            raise NotImplementedError

    def service_account_credentials(*, scopes: Sequence[str] | None = None) -> service_account.Credentials:
        return service_account.Credentials(
            signer=FakeSigner(),
            service_account_email='service-account@example.com',
            token_uri='https://oauth2.googleapis.com/token',
            scopes=scopes,
        )


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


def test_google_cloud_provider_scopes_credentials():
    """Unscoped credentials gain the cloud-platform scope before reaching the client.

    This is a unit test because credential transformation happens before any request a cassette could record.
    """
    credentials = service_account_credentials()
    provider = GoogleCloudProvider(credentials=credentials, project='pydantic-ai', location='us-central1')

    forwarded_credentials = cast(
        'service_account.Credentials',
        provider.client._api_client._credentials,  # pyright: ignore[reportPrivateUsage]
    )
    assert forwarded_credentials is not credentials
    assert forwarded_credentials.scopes == ['https://www.googleapis.com/auth/cloud-platform']


def test_google_cloud_provider_preserves_existing_scopes():
    """Credentials that already have scopes are forwarded untouched.

    This is a unit test because credential transformation happens before any request a cassette could record.
    """
    credentials = service_account_credentials(scopes=['https://www.googleapis.com/auth/devstorage.read_only'])
    provider = GoogleCloudProvider(credentials=credentials, project='pydantic-ai', location='us-central1')

    assert provider.client._api_client._credentials is credentials  # pyright: ignore[reportPrivateUsage]


def test_google_cloud_provider_preserves_non_scoped_credentials():
    """Credentials that cannot be scoped are forwarded untouched.

    This is a unit test because credential transformation happens before any request a cassette could record.
    """
    credentials = AnonymousCredentials()
    provider = GoogleCloudProvider(credentials=credentials, project='pydantic-ai', location='us-central1')

    assert provider.client._api_client._credentials is credentials  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('api_key_env_var', ['GOOGLE_API_KEY', 'GEMINI_API_KEY'])
def test_google_cloud_provider_api_key_from_env(env: TestEnv, api_key_env_var: str):
    """An environment API key still selects Vertex AI Express Mode.

    This is a unit test because authentication routing happens before any request a cassette could record.
    """
    for name in {
        'GOOGLE_APPLICATION_CREDENTIALS',
        'GOOGLE_CLOUD_PROJECT',
        'GOOGLE_CLOUD_LOCATION',
        'GOOGLE_API_KEY',
        'GEMINI_API_KEY',
    } - {api_key_env_var}:
        env.remove(name)
    env.set(api_key_env_var, 'your-api-key')

    provider = GoogleCloudProvider()
    assert provider.client._api_client.api_key == 'your-api-key'  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('api_key', [None, ''])
def test_google_cloud_provider_adc_env_takes_precedence_over_api_key(env: TestEnv, api_key: str | None):
    """Application credentials take precedence over an API key from the environment.

    An empty `api_key` counts as unset, so it must not resurrect an environment API key either.
    This is a unit test because it verifies SDK authentication routing before a request is sent.
    """
    env.set('GOOGLE_APPLICATION_CREDENTIALS', '/path/to/service-account.json')
    env.set('GOOGLE_API_KEY', 'should-be-ignored')
    env.set('GEMINI_API_KEY', 'also-ignored')
    env.remove('GOOGLE_CLOUD_PROJECT')
    env.remove('GOOGLE_CLOUD_LOCATION')
    credentials = AnonymousCredentials()

    with patch('google.auth.default', return_value=(credentials, 'pydantic-ai')):
        provider = GoogleCloudProvider(api_key=api_key)

    api_client = provider.client._api_client  # pyright: ignore[reportPrivateUsage]
    assert api_client.api_key is None
    assert api_client._credentials is credentials  # pyright: ignore[reportPrivateUsage]


def test_google_cloud_provider_project_env_takes_precedence_over_api_key(env: TestEnv):
    """A cloud project takes precedence over an API key from the environment.

    This is a unit test because authentication routing happens before any request a cassette could record.
    """
    env.remove('GOOGLE_APPLICATION_CREDENTIALS')
    env.remove('GOOGLE_CLOUD_LOCATION')
    env.set('GOOGLE_API_KEY', 'should-be-ignored')
    env.set('GEMINI_API_KEY', 'also-ignored')
    env.set('GOOGLE_CLOUD_PROJECT', 'pydantic-ai')

    provider = GoogleCloudProvider()
    api_client = provider.client._api_client  # pyright: ignore[reportPrivateUsage]
    assert api_client.api_key is None
    assert api_client.project == 'pydantic-ai'


def test_google_cloud_provider_location_env_takes_precedence_over_api_key(env: TestEnv):
    """A cloud location takes precedence over an API key from the environment.

    This is a unit test because authentication routing happens before any request a cassette could record.
    """
    env.remove('GOOGLE_APPLICATION_CREDENTIALS')
    env.remove('GOOGLE_CLOUD_PROJECT')
    env.set('GOOGLE_API_KEY', 'should-be-ignored')
    env.set('GEMINI_API_KEY', 'also-ignored')
    env.set('GOOGLE_CLOUD_LOCATION', 'global')
    credentials = AnonymousCredentials()

    with patch('google.auth.default', return_value=(credentials, 'pydantic-ai')):
        provider = GoogleCloudProvider()

    api_client = provider.client._api_client  # pyright: ignore[reportPrivateUsage]
    assert api_client.api_key is None
    assert api_client.location == 'global'


def test_google_cloud_provider_explicit_api_key_still_passed(env: TestEnv):
    """An explicit API key selects Express Mode even when ADC environment variables are set.

    This is a unit test because authentication routing happens before any request a cassette could record.
    """
    env.set('GOOGLE_APPLICATION_CREDENTIALS', '/path/to/service-account.json')
    env.set('GOOGLE_CLOUD_PROJECT', 'adc-project')
    env.set('GOOGLE_CLOUD_LOCATION', 'global')
    provider = GoogleCloudProvider(api_key='your-api-key')
    assert provider.client._api_client.api_key == 'your-api-key'  # pyright: ignore[reportPrivateUsage]


def test_google_cloud_provider_adc_kwargs_take_precedence_over_explicit_api_key():
    """Explicit `credentials` select credential-based authentication even when `api_key` is also passed.

    This is a unit test because authentication routing happens before any request a cassette could record.
    """
    credentials = service_account_credentials()
    provider = GoogleCloudProvider(
        api_key='your-api-key', credentials=credentials, project='pydantic-ai', location='us-central1'
    )
    api_client = provider.client._api_client  # pyright: ignore[reportPrivateUsage]
    assert api_client.api_key is None
    forwarded_credentials = api_client._credentials  # pyright: ignore[reportPrivateUsage]
    assert isinstance(forwarded_credentials, service_account.Credentials)
    assert forwarded_credentials.service_account_email == 'service-account@example.com'
    # The credentials fed in are unscoped, so this proves the forwarded object went through scoping.
    assert forwarded_credentials.scopes == ['https://www.googleapis.com/auth/cloud-platform']


def test_google_cloud_provider_google_api_key_takes_precedence_over_gemini_api_key(env: TestEnv):
    """`GOOGLE_API_KEY` wins when both environment API keys are set, matching the SDK.

    This is a unit test because authentication routing happens before any request a cassette could record.
    """
    for name in ('GOOGLE_APPLICATION_CREDENTIALS', 'GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION'):
        env.remove(name)
    env.set('GOOGLE_API_KEY', 'google-api-key')
    env.set('GEMINI_API_KEY', 'gemini-api-key')

    provider = GoogleCloudProvider()
    assert provider.client._api_client.api_key == 'google-api-key'  # pyright: ignore[reportPrivateUsage]


def test_google_cloud_model_string_uses_adc_from_env(env: TestEnv):
    """The `google-cloud:` model string picks up `GOOGLE_APPLICATION_CREDENTIALS` despite environment API keys.

    Regression test for issue #6499, which reported the failure through this shortcut.
    This is a unit test because authentication routing happens before any request a cassette could record.
    """
    env.set('GOOGLE_APPLICATION_CREDENTIALS', '/path/to/service-account.json')
    env.set('GOOGLE_API_KEY', 'should-be-ignored')
    env.set('GEMINI_API_KEY', 'also-ignored')
    env.remove('GOOGLE_CLOUD_PROJECT')
    env.remove('GOOGLE_CLOUD_LOCATION')
    credentials = AnonymousCredentials()

    with patch('google.auth.default', return_value=(credentials, 'pydantic-ai')):
        model = infer_model('google-cloud:gemini-3-flash-preview')

    assert isinstance(model, GoogleModel)
    api_client = model.client._api_client  # pyright: ignore[reportPrivateUsage]
    assert api_client.api_key is None
    assert api_client._credentials is credentials  # pyright: ignore[reportPrivateUsage]

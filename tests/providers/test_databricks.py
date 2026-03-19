# pyright: reportPrivateUsage=false
from __future__ import annotations as _annotations

import httpx
import pytest

import pydantic_ai.providers.databricks as db_mod
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.databricks import DatabricksProvider

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

_has_databricks_sdk = db_mod._has_databricks_sdk

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai or databricks-sdk not installed')


def test_databricks_provider():
    """Base sanity check of databricks provider."""
    provider = DatabricksProvider(api_key='api-key', base_url='https://mock.databricks.com/serving-endpoints/')
    assert provider.name == 'databricks'
    assert provider.base_url == 'https://mock.databricks.com/serving-endpoints/'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_databricks_provider_add_serving_endpoint():
    """Test auto host formatting."""
    provider = DatabricksProvider(api_key='api-key', base_url='https://mock.databricks.com/')
    assert provider.name == 'databricks'
    assert provider.base_url == 'https://mock.databricks.com/serving-endpoints/'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_databricks_provider_pass_http_client() -> None:
    """Test http client pass."""
    http_client = httpx.AsyncClient()
    provider = DatabricksProvider(http_client=http_client, api_key='api-key', base_url='https://mock.databricks.com/')
    assert provider.client._client == http_client


def test_databricks_provider_env_vars(env: TestEnv):
    """Test initializing using environment variables."""
    env.set('DATABRICKS_HOST', 'https://env-url.com')
    env.set('DATABRICKS_TOKEN', 'env-key')

    provider = DatabricksProvider()
    assert provider.base_url == 'https://env-url.com/serving-endpoints/'
    assert provider.client.base_url == 'https://env-url.com/serving-endpoints/'
    assert provider.client.api_key == 'env-key'


def test_databricks_provider_env_vars_base_url_and_key(env: TestEnv):
    """Test initializing using environment variables, different env var names."""
    env.set('DATABRICKS_BASE_URL', 'https://env-url.com')
    env.set('DATABRICKS_API_KEY', 'env-key')

    provider = DatabricksProvider()
    assert provider.base_url == 'https://env-url.com/serving-endpoints/'
    assert provider.client.base_url == 'https://env-url.com/serving-endpoints/'
    assert provider.client.api_key == 'env-key'


def test_databricks_provider_pass_openai_client() -> None:
    """Test passing an existing AsyncOpenAI client directly."""
    client = openai.AsyncOpenAI(api_key='test', base_url='https://test.com')
    provider = DatabricksProvider(openai_client=client)
    assert provider.client is client


def test_databricks_provider_model_profile() -> None:
    """Test that model_profile returns a valid profile."""
    provider = DatabricksProvider(api_key='test', base_url='https://test.com')
    profile = provider.model_profile('databricks-gpt-5-2')
    assert profile is not None


def test_databricks_provider_no_sdk_no_credentials(env: TestEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test error message when SDK is not installed and no explicit credentials."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_BASE_URL')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')
    monkeypatch.setattr(db_mod, '_has_databricks_sdk', False)
    with pytest.raises(ImportError, match='databricks-sdk'):
        DatabricksProvider()


@pytest.mark.skipif(not _has_databricks_sdk, reason='databricks-sdk not installed')
def test_databricks_provider_sdk_no_host_failure(env: TestEnv) -> None:
    """Test error when SDK can't find a host (no base_url, no env config)."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_BASE_URL')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')
    env.remove('DATABRICKS_CONFIG_PROFILE')
    env.remove('DATABRICKS_CONFIG_FILE')
    env.remove('HOME')
    with pytest.raises(UserError, match="Couldn't find host url"):
        DatabricksProvider()


@pytest.mark.skipif(not _has_databricks_sdk, reason='databricks-sdk not installed')
def test_databricks_provider_sdk_no_credentials_with_base_url(env: TestEnv) -> None:
    """Test error when SDK can't find credentials (base_url given, no api_key)."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')
    env.remove('DATABRICKS_CONFIG_PROFILE')
    env.remove('DATABRICKS_CONFIG_FILE')
    env.remove('HOME')
    with pytest.raises(UserError, match="Couldn't retrieve credentials"):
        DatabricksProvider(base_url='https://test.databricks.com')

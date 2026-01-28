from __future__ import annotations as _annotations

import httpx
import pytest

from pydantic_ai.providers.databricks import DatabricksProvider

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai


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
    provider = DatabricksProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


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

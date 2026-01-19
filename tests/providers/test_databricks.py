from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.databricks import databricks_model_profile

from ..conftest import try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI

    from pydantic_ai.providers.databricks import DatabricksProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


def test_databricks_provider_init_args():
    provider = DatabricksProvider(
        base_url='https://my-workspace.cloud.databricks.com',
        api_key='dapi123456789',
    )
    assert provider.name == 'databricks'
    # Test normalization of the URL (appending /serving-endpoints)
    assert provider.base_url == snapshot('https://my-workspace.cloud.databricks.com/serving-endpoints')
    assert isinstance(provider.client, AsyncOpenAI)
    assert provider.client.api_key == 'dapi123456789'


def test_databricks_provider_init_args_existing_suffix():
    """Test that /serving-endpoints is not duplicated if already present."""
    provider = DatabricksProvider(
        base_url='https://my-workspace.cloud.databricks.com/serving-endpoints',
        api_key='dapi123456789',
    )
    assert provider.base_url == snapshot('https://my-workspace.cloud.databricks.com/serving-endpoints')


def test_databricks_provider_env_host_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('DATABRICKS_HOST', 'https://env-host.com')
    monkeypatch.setenv('DATABRICKS_TOKEN', 'env-token')

    provider = DatabricksProvider()
    assert provider.base_url == snapshot('https://env-host.com/serving-endpoints')
    assert provider.client.api_key == 'env-token'


def test_databricks_provider_env_base_url_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('DATABRICKS_BASE_URL', 'https://alt-host.com')
    monkeypatch.setenv('DATABRICKS_API_KEY', 'alt-key')

    provider = DatabricksProvider()
    assert provider.base_url == snapshot('https://alt-host.com/serving-endpoints')
    assert provider.client.api_key == 'alt-key'


def test_databricks_provider_missing_config(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('DATABRICKS_HOST', raising=False)
    monkeypatch.delenv('DATABRICKS_BASE_URL', raising=False)
    monkeypatch.delenv('DATABRICKS_TOKEN', raising=False)
    monkeypatch.delenv('DATABRICKS_API_KEY', raising=False)

    with pytest.raises(UserError) as exc_info:
        DatabricksProvider()

    assert str(exc_info.value) == snapshot(
        'Set `DATABRICKS_HOST` or `DATABRICKS_BASE_URL` environment variable, '
        'or pass `base_url` to use the Databricks provider.'
    )


def test_databricks_provider_pass_client():
    client = AsyncOpenAI(
        base_url='https://custom.databricks.com/serving-endpoints',
        api_key='custom-key',
    )
    provider = DatabricksProvider(openai_client=client)
    assert provider.client is client
    assert provider.base_url == 'https://custom.databricks.com/serving-endpoints/'


def test_databricks_provider_default_api_key(monkeypatch: pytest.MonkeyPatch):
    """Test api_key defaults to 'nop' if host is set but key is missing."""
    monkeypatch.setenv('DATABRICKS_HOST', 'https://host.com')
    monkeypatch.delenv('DATABRICKS_TOKEN', raising=False)
    monkeypatch.delenv('DATABRICKS_API_KEY', raising=False)

    provider = DatabricksProvider()
    assert provider.client.api_key == 'nop'


def test_databricks_provider_model_profile(mocker: MockerFixture):
    provider = DatabricksProvider(base_url='https://workspace.databricks.com', api_key='key')

    ns = 'pydantic_ai.providers.databricks'
    databricks_profile_mock = mocker.patch(f'{ns}.databricks_model_profile', wraps=databricks_model_profile)

    profile = provider.model_profile('databricks-dbrx-instruct')
    databricks_profile_mock.assert_called_with('databricks-dbrx-instruct')
    assert profile is not None

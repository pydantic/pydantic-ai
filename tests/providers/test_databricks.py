# pyright: reportPrivateUsage=false
from __future__ import annotations as _annotations

from dataclasses import dataclass
from typing import cast

import httpx
import pytest

import pydantic_ai.providers.databricks as db_mod
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.databricks import DatabricksProvider

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai import ModelRequest, TextPart
    from pydantic_ai.direct import model_request
    from pydantic_ai.models.databricks import DatabricksModel

_has_databricks_sdk = db_mod._has_databricks_sdk

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai or databricks-sdk not installed')


@dataclass
class _SdkConfig:
    host: str = 'https://mock.databricks.com'

    def authenticate(self) -> dict[str, str]:
        return {'Authorization': 'Bearer mock-sdk-token'}


class _SdkWorkspaceClient:
    def __init__(self, host: str | None = None, token: str | None = None):
        if host is None:
            host = 'https://mock.databricks.com'
        self.config = _SdkConfig(host=host)


@pytest.fixture()
def databricks_sdk_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(db_mod, '_has_databricks_sdk', True)
    monkeypatch.setattr(db_mod, 'WorkspaceClient', _SdkWorkspaceClient, raising=False)


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


@pytest.mark.vcr
@pytest.mark.anyio
async def test_databricks_sdk_auth_request(
    allow_model_requests: None,
    databricks_sdk_auth: None,
    env: TestEnv,
) -> None:
    """SDK auth path creates a working provider that can make requests."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_BASE_URL')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')

    provider = DatabricksProvider()
    model = DatabricksModel('databricks-gpt-5-2', provider=provider)

    response = await model_request(
        model,
        [ModelRequest.user_text_prompt('What is the capital of France? lowercase, one word reply, no punctuation.')],
    )

    text_part = cast(TextPart, response.parts[0])
    assert 'paris' in text_part.content


def test_databricks_auth_flow() -> None:
    """DatabricksAuth adds SDK auth headers to requests."""
    ws = _SdkWorkspaceClient()
    auth = db_mod.DatabricksAuth(ws)  # type: ignore
    request = httpx.Request('GET', 'https://mock.databricks.com/test')
    flow = auth.auth_flow(request)
    modified_request = next(flow)
    assert modified_request.headers['Authorization'] == 'Bearer mock-sdk-token'


def test_databricks_sdk_auth_base_url_only(databricks_sdk_auth: None, env: TestEnv) -> None:
    """SDK auth with base_url but no api_key uses WorkspaceClient for auth."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_BASE_URL')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')

    provider = DatabricksProvider(base_url='https://custom.databricks.com')
    assert provider.base_url == 'https://custom.databricks.com/serving-endpoints/'
    assert provider.client.api_key == 'nop'


def test_databricks_sdk_auth_api_key_only(databricks_sdk_auth: None, env: TestEnv) -> None:
    """SDK auth with api_key but no base_url discovers host from SDK."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_BASE_URL')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')

    provider = DatabricksProvider(api_key='user-token')
    assert provider.base_url == 'https://mock.databricks.com/serving-endpoints/'
    assert provider.client.api_key == 'user-token'


def test_databricks_sdk_auth_host_with_serving_endpoints(env: TestEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK auth path does not double-append /serving-endpoints."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_BASE_URL')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')

    class _Client:
        def __init__(self, **kwargs: object):
            self.config = _SdkConfig(host='https://mock.databricks.com/serving-endpoints')

    monkeypatch.setattr(db_mod, '_has_databricks_sdk', True)
    monkeypatch.setattr(db_mod, 'WorkspaceClient', _Client, raising=False)

    provider = DatabricksProvider()
    assert 'serving-endpoints/serving-endpoints' not in str(provider.base_url)


def test_databricks_provider_sdk_host_not_configured(env: TestEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """SDK auth raises UserError when host is empty."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_BASE_URL')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')

    class _Client:
        def __init__(self, **kwargs: object):
            self.config = _SdkConfig(host='')

    monkeypatch.setattr(db_mod, '_has_databricks_sdk', True)
    monkeypatch.setattr(db_mod, 'WorkspaceClient', _Client, raising=False)

    with pytest.raises(UserError, match='Databricks host not configured'):
        DatabricksProvider()


def test_databricks_sdk_auth_with_custom_http_client(databricks_sdk_auth: None, env: TestEnv) -> None:
    """SDK auth path uses provided http_client and adds DatabricksAuth to it."""
    env.remove('DATABRICKS_API_KEY')
    env.remove('DATABRICKS_BASE_URL')
    env.remove('DATABRICKS_TOKEN')
    env.remove('DATABRICKS_HOST')

    http_client = httpx.AsyncClient()
    provider = DatabricksProvider(http_client=http_client)
    assert provider.client._client is http_client
    assert provider.base_url == 'https://mock.databricks.com/serving-endpoints/'

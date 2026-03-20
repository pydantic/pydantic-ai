from __future__ import annotations as _annotations

import os
import warnings
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Literal, overload

import anyio.to_thread
import httpx

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import DEFAULT_HTTP_TIMEOUT, cached_async_http_client, get_user_agent
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.databricks import databricks_model_profile
from pydantic_ai.providers import Provider

DatabricksModelName = Literal[
    'databricks-gpt-5-2',
    'databricks-claude-opus-4-5',
    'databricks-gpt-oss-120b',
    'databricks-qwen3-next-80b-a3b-instruct',
]

if TYPE_CHECKING:
    from databricks.sdk import WorkspaceClient
    from openai import AsyncOpenAI


try:
    from openai import AsyncOpenAI
except ImportError as _import_error:
    raise ImportError(
        'Please install the `openai` package to use the Databricks provider, '
        'you can use the `databricks` optional group — `pip install "pydantic-ai-slim[databricks]"`'
    ) from _import_error

_has_databricks_sdk = False
try:
    from databricks.sdk import WorkspaceClient

    _has_databricks_sdk = True
except ImportError:
    pass


def _ensure_serving_endpoints(host: str) -> str:
    """Ensure the host URL ends with /serving-endpoints."""
    if not host.rstrip('/').endswith('serving-endpoints'):
        return f'{host.rstrip("/")}/serving-endpoints'
    return host


def _sdk_auth_client(ws: WorkspaceClient) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient wired with DatabricksAuth for SDK-based authentication."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=DEFAULT_HTTP_TIMEOUT, connect=5),
        headers={'User-Agent': get_user_agent()},
        auth=DatabricksAuth(ws),
    )


class DatabricksAuth(httpx.Auth):
    """Refresh OAuth tokens using Databricks SDK."""

    def __init__(self, client: WorkspaceClient):
        self.db_client = client

    def auth_flow(self, request: httpx.Request):
        headers = self.db_client.config.authenticate()
        for k, v in headers.items():
            request.headers[k] = v
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        headers = await anyio.to_thread.run_sync(self.db_client.config.authenticate)
        for k, v in headers.items():
            request.headers[k] = v
        yield request


class DatabricksProvider(Provider['AsyncOpenAI']):
    """Provider for Databricks Model Serving using the official SDK."""

    @property
    def name(self) -> str:
        return 'databricks'

    @property
    def base_url(self) -> str:
        return str(self._client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return databricks_model_profile(model_name)

    @overload
    def __init__(self, *, openai_client: AsyncOpenAI) -> None: ...

    @overload
    def __init__(self, *, workspace_client: WorkspaceClient) -> None: ...

    @overload
    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None: ...

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        workspace_client: WorkspaceClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if openai_client is not None:
            self._client = openai_client
            return

        if workspace_client is not None:
            host = workspace_client.config.host
            if not host:
                raise UserError('Databricks host not configured on the provided workspace_client.')
            self._client = AsyncOpenAI(
                base_url=_ensure_serving_endpoints(host),
                api_key='nop',
                http_client=_sdk_auth_client(workspace_client),
            )
            return

        api_key = api_key or os.getenv('DATABRICKS_API_KEY') or os.getenv('DATABRICKS_TOKEN')
        base_url = base_url or os.getenv('DATABRICKS_BASE_URL') or os.getenv('DATABRICKS_HOST')

        if api_key and base_url:
            self._client = AsyncOpenAI(
                base_url=_ensure_serving_endpoints(base_url),
                api_key=api_key,
                http_client=http_client or cached_async_http_client(provider='databricks'),
            )
            return

        if not _has_databricks_sdk:
            raise ImportError(
                'Please install the `databricks-sdk` package to use this provider without explicit credentials: '
                '`pip install "pydantic-ai[databricks]"`'
            )

        use_sdk_auth = not api_key

        if not api_key:
            # SDK handles both host discovery and authentication
            try:
                ws = WorkspaceClient(host=base_url or None)
            except Exception as e:
                if base_url:
                    raise UserError(
                        f"Failed to authenticate to databricks workspace. Couldn't retrieve credentials or profile: {e}"
                    )
                raise UserError(f"Failed to authenticate to databricks workspace. Couldn't find host url: {e}")
            api_key = 'nop'
        else:
            # User provided api_key but not base_url; SDK used only for host discovery
            try:
                ws = WorkspaceClient()
            except Exception as e:
                raise UserError(f"Failed to authenticate to databricks workspace. Couldn't find host url: {e}")

        host = ws.config.host
        if not host:
            raise UserError('Databricks host not configured.')

        if use_sdk_auth:
            if http_client is not None:
                warnings.warn(
                    'http_client is ignored when using Databricks SDK authentication; '
                    'a new client with DatabricksAuth will be created instead',
                    UserWarning,
                    stacklevel=2,
                )
            http_client = _sdk_auth_client(ws)
        elif http_client is None:
            http_client = cached_async_http_client(provider='databricks')

        self._client = AsyncOpenAI(
            base_url=_ensure_serving_endpoints(host),
            api_key=api_key,
            http_client=http_client,
        )

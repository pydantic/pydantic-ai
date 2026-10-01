from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from datetime import datetime, timedelta, timezone
from functools import cached_property
from typing import Protocol

import anyio
import httpx2

from .._http import AsyncHTTPClient, create_async_httpx2_client
from ..exceptions import ModelAPIError, UserError
from ..native_tools import WebSearchTool
from ..profiles import ModelProfile, merge_profile
from ..profiles.openai import OpenAIModelProfile
from ._openai_chatgpt_oauth import (
    OpenAIChatGPTClient as OpenAIChatGPTClient,
    OpenAIChatGPTCredentials as OpenAIChatGPTCredentials,
    OpenAIChatGPTOAuthFlow as OpenAIChatGPTOAuthFlow,
    refresh_credentials,
)
from ._openai_compatible import OpenAICompatibleProvider as _OpenAICompatibleProvider
from .openai import OpenAIProvider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai-chatgpt` optional group: `pip install "pydantic-ai-slim[openai-chatgpt]"`'
    ) from _import_error

__all__ = (
    'OpenAIChatGPTClient',
    'OpenAIChatGPTCredentials',
    'OpenAIChatGPTCredentialSource',
    'OpenAIChatGPTOAuthFlow',
    'OpenAIChatGPTProvider',
)


class OpenAIChatGPTCredentialSource(Protocol):
    """Application-owned storage and serialization of rotating ChatGPT credentials.

    `rotate` must exclude competing refreshes of the same registration across all callers
    (including other provider instances/processes). Reload inside that exclusion: if the stored
    token set differs from `expected`, return it; otherwise call `refresh`, atomically persist the
    whole result, and return it only after persistence succeeds. Do not retry an ambiguous exchange
    with a rotating refresh token. The application owns recovery and reporting persistence errors.
    """

    async def load(self) -> OpenAIChatGPTCredentials:
        """Load the selected registration without switching accounts."""
        ...

    async def rotate(
        self,
        expected: OpenAIChatGPTCredentials,
        refresh: Callable[[OpenAIChatGPTCredentials], Awaitable[OpenAIChatGPTCredentials]],
    ) -> OpenAIChatGPTCredentials:
        """Publish one complete refresh under application-owned session exclusion."""
        ...


class _ChatGPTAuth(httpx2.Auth):
    def __init__(self, provider: OpenAIChatGPTProvider) -> None:
        self._provider = provider

    def sync_auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        raise UserError('`OpenAIChatGPTProvider` requires an async HTTP client.')

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        if request.url.scheme != 'https' or request.url.host != 'api.openai.com':
            yield request
            return
        await request.aread()
        credentials = await self._provider._get_credentials()  # pyright: ignore[reportPrivateUsage]
        request.headers['Authorization'] = f'Bearer {credentials.access_token}'
        response = yield request
        if response.status_code == 401:
            await response.aread()
            credentials = await self._provider._get_credentials(expected=credentials)  # pyright: ignore[reportPrivateUsage]
            request.headers['Authorization'] = f'Bearer {credentials.access_token}'
            yield request


class OpenAIChatGPTProvider(_OpenAICompatibleProvider):
    """Public Responses API access using Sign in with ChatGPT plan authorization.

    One instance owns one registration, with single-flight refresh on expiry or HTTP 401. No
    credentials are read from the Codex CLI or API-key environment. With `credentials`, rotation
    lives in memory only; use a `credential_source` to publish it durably. Bind an instance to one
    async event loop, as with other providers carrying async clients.
    """

    def __init__(
        self,
        credentials: OpenAIChatGPTCredentials | None = None,
        *,
        credential_source: OpenAIChatGPTCredentialSource | None = None,
        client: OpenAIChatGPTClient | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        """Create a provider without logging in or loading application storage.

        Args:
            credentials: Complete credentials from `OpenAIChatGPTOAuthFlow`, held in memory.
            credential_source: Application-owned session storage, mutually exclusive with `credentials`.
            client: Provisioned-client configuration used for refresh, when applicable. Its client ID
                must match the credentials. OSS registrations do not use a client secret.
            http_client: A dedicated `httpx2.AsyncClient` with no existing auth. The provider attaches
                its auth, injecting tokens only for HTTPS requests to `api.openai.com`.
        """
        if (credentials is None) == (credential_source is None):
            raise UserError('Supply exactly one of `credentials` or `credential_source` for Sign in with ChatGPT.')
        if http_client is not None and (
            not isinstance(http_client, httpx2.AsyncClient) or http_client.auth is not None
        ):
            raise UserError('`http_client` must be a dedicated `httpx2.AsyncClient` without existing auth.')
        self._credentials = credentials
        self._credential_source = credential_source
        self._oauth_client = client
        self._refresh_error: Exception | None = None
        self._auth = _ChatGPTAuth(self)
        if http_client is None:
            http_client = self._create_http_client()
            self._own_http_client = http_client
            self._http_client_factory = self._create_http_client
        else:
            http_client.auth = self._auth
        self._http_client = http_client
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            api_key='chatgpt-plan-auth',
            http_client=http_client,
            max_retries=0,
        )

    @property
    def name(self) -> str:
        return 'openai-chatgpt'

    @property
    def base_url(self) -> str:
        return 'https://api.openai.com/v1'

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @property
    def credentials(self) -> OpenAIChatGPTCredentials:
        """The token set currently held in memory; unavailable before a source's first load."""
        if self._credentials is None:
            raise UserError('The ChatGPT credential source has not been loaded yet.')
        return self._credentials

    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        return merge_profile(
            OpenAIProvider.model_profile(model_name),
            OpenAIModelProfile(
                openai_unsupported_model_settings=('max_tokens', 'temperature', 'top_p'),
                openai_responses_requires_streaming=True,
                openai_responses_requires_store_false=True,
                openai_supports_input_token_counting=False,
                openai_system_prompt_role='developer',
                supported_native_tools=frozenset({WebSearchTool}),
                # Local tool discovery works, but the Responses tool_search route does not.
                tool_deferral_mode=None,
            ),
        )

    def _create_http_client(self) -> httpx2.AsyncClient:
        client = create_async_httpx2_client()
        client.auth = self._auth
        self._http_client = client
        return client

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        assert isinstance(http_client, httpx2.AsyncClient)
        self._http_client = http_client
        super()._set_http_client(http_client)

    @cached_property
    def _refresh_lock(self) -> anyio.Lock:
        return anyio.Lock()

    async def _get_credentials(self, *, expected: OpenAIChatGPTCredentials | None = None) -> OpenAIChatGPTCredentials:
        async with self._refresh_lock:
            if self._credentials is None:
                assert self._credential_source is not None
                self._credentials = await self._credential_source.load()
            credentials = self.credentials
            if not {'resource.invoke', 'chatgpt.tokens.use.direct'}.issubset(credentials.scopes):
                raise UserError('The ChatGPT credentials do not grant plan usage.')
            if self._oauth_client is not None and credentials.client_id != self._oauth_client.client_id:
                raise UserError('The credentials belong to another ChatGPT client.')
            now = datetime.now(timezone.utc)
            if expected is not None:
                if credentials != expected:
                    return credentials  # A concurrent request already rotated this token set.
                refresh = True
            else:
                refresh = credentials.expires_at <= now + timedelta(seconds=30)
                if credentials.earliest_refresh_at and now < credentials.earliest_refresh_at:
                    refresh = credentials.expires_at <= now
            if not refresh:
                return credentials
            # A dispatched refresh may rotate even when its response or publication is lost.
            # Never automatically spend that token again; recover through app-owned storage or login.
            if self._refresh_error is not None:
                raise self._refresh_error

            if credentials.earliest_refresh_at and now < credentials.earliest_refresh_at:
                raise ModelAPIError(
                    model_name=self.name,
                    message='ChatGPT token refresh is not yet allowed. Retry after `earliest_refresh_at`.',
                )

            async def exchange(value: OpenAIChatGPTCredentials) -> OpenAIChatGPTCredentials:
                # Set before awaiting so cancellation cannot cause a later retry of a spent token.
                self._refresh_error = ModelAPIError(
                    model_name=self.name,
                    message='ChatGPT token refresh outcome is unknown. Reload application storage or sign in again.',
                )
                return await refresh_credentials(value, client=self._oauth_client, http_client=self._http_client)

            try:
                if self._credential_source is None:
                    updated = await exchange(credentials)
                else:
                    updated = await self._credential_source.rotate(credentials, exchange)
                if (updated.client_id, updated.subject, updated.ext_agent_host_id) != (
                    credentials.client_id,
                    credentials.subject,
                    credentials.ext_agent_host_id,
                ):
                    raise UserError('The ChatGPT credential source changed the selected registration.')
                self._credentials = updated
                self._refresh_error = None
            except Exception as exc:
                self._refresh_error = exc
                raise
            return self.credentials

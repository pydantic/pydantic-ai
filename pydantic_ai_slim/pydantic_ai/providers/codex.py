from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator

import httpx

from pydantic_ai.auth.codex import CodexAuth, CodexCredentials, CodexCredentialSource
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Codex provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error

CODEX_BASE_URL = 'https://chatgpt.com/backend-api/codex'
"""Base URL observed in the pinned official Codex client for ChatGPT-authenticated requests."""


class _CodexHTTPAuth(httpx.Auth):
    def __init__(self, credential_source: CodexCredentialSource) -> None:
        self._credential_source = credential_source

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if not self._is_trusted_codex_url(request.url):
            yield request
            return

        replayable = self._is_responses_request(request)
        if replayable:
            try:
                request.content
            except httpx.RequestNotRead:
                replayable = False

        credentials = await self._credential_source.get_credentials()
        self._apply(request, credentials)
        response = yield request

        if response.status_code == 401 and replayable:
            await response.aread()
            credentials = await self._credential_source.get_credentials(
                force_refresh=True, rejected_revision=credentials.revision
            )
            self._apply(request, credentials)
            yield request

    def _apply(self, request: httpx.Request, credentials: CodexCredentials) -> None:
        request.headers['Authorization'] = f'Bearer {credentials.access_token.get_secret_value()}'
        request.headers['ChatGPT-Account-ID'] = credentials.account_id.get_secret_value()
        request.headers['originator'] = 'pydantic-ai'
        if credentials.account_is_fedramp:
            request.headers['X-OpenAI-Fedramp'] = 'true'
        else:
            request.headers.pop('X-OpenAI-Fedramp', None)

    def _is_trusted_codex_url(self, url: httpx.URL) -> bool:
        raw_path = url.raw_path.partition(b'?')[0]
        path_segments = url.path.split('/')
        return (
            url.scheme == 'https'
            and url.host == 'chatgpt.com'
            and url.port is None
            and raw_path.startswith(b'/backend-api/codex/')
            and b'%' not in raw_path
            and b'\\' not in raw_path
            and '.' not in path_segments
            and '..' not in path_segments
        )

    def _is_responses_request(self, request: httpx.Request) -> bool:
        return request.method == 'POST' and request.url.path.rstrip('/') == '/backend-api/codex/responses'


class CodexProvider(Provider[AsyncOpenAI]):
    """Provider for Codex models accessed with ChatGPT subscription credentials.

    Authentication is resolved lazily for each request. Constructing the provider
    never reads credential storage, opens a browser, or starts background work.

    Args:
        credential_source: Application-owned credentials. Defaults to [`CodexAuth`]
            [pydantic_ai.auth.codex.CodexAuth] and its managed local credential store.
        http_client: A dedicated caller-owned HTTP client with no existing auth and
            `follow_redirects=False`. Codex authentication is installed on this client,
            and the provider never closes it.
    """

    @property
    def name(self) -> str:
        return 'codex'

    @property
    def base_url(self) -> str:
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @staticmethod
    def model_profile(model_name: str) -> OpenAIModelProfile:
        return OpenAIModelProfile(
            **openai_model_profile(model_name),
            openai_responses_requires_store_false=True,
            openai_responses_requires_stream=True,
        )

    def __init__(
        self,
        *,
        credential_source: CodexCredentialSource | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if credential_source is None:
            credential_source = CodexAuth()
        self._credential_source = credential_source

        if http_client is None:

            def http_client_factory() -> httpx.AsyncClient:
                client = create_async_http_client()
                client.auth = _CodexHTTPAuth(credential_source)
                return client

            http_client = http_client_factory()
            self._own_http_client = http_client
            self._http_client_factory = http_client_factory
        else:
            if http_client.auth is not None:
                raise ValueError('`http_client` must not already have authentication configured.')
            if http_client.follow_redirects:
                raise ValueError('`http_client` must have `follow_redirects=False`.')
            http_client.auth = _CodexHTTPAuth(credential_source)

        # AsyncOpenAI requires a non-empty API key even though the HTTP auth layer
        # replaces the generated Authorization header before every request.
        self._client = AsyncOpenAI(
            base_url=CODEX_BASE_URL,
            api_key='codex-subscription-auth',
            http_client=http_client,
        )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        http_client.auth = _CodexHTTPAuth(self._credential_source)
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]

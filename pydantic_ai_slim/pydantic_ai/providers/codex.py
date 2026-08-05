from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator

import httpx

from pydantic_ai.auth.codex import CodexAuth, CodexCredentials, CodexCredentialSource
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import create_async_http_client
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Codex provider, '
        'you can use the `codex` optional group — `pip install "pydantic-ai-slim[codex]"`. '
        'The `openai` group alone omits the `filelock` the default credential store needs.'
    ) from _import_error

# Base URL observed in the pinned official Codex client for ChatGPT-authenticated requests.
# `CodexProvider.base_url` exposes it, so it stays private rather than becoming the one public
# base-URL constant in any provider module.
_BASE_URL = httpx.URL('https://chatgpt.com/backend-api/codex')
_RESPONSES_PATH = f'{_BASE_URL.path}/responses'

# The official Codex CLI sends `User-Agent: codex_cli_rs/<version>`, which was once needed to route
# newer model slugs. The backend no longer distinguishes: the SDK's own User-Agent and the Codex CLI's
# return identical results on every slug, including the ones an account is not entitled to, which are
# refused for the account rather than the client. So we send the ordinary Pydantic AI/OpenAI SDK
# User-Agent and identify ourselves through the `originator` header instead.


class _CodexHTTPAuth(httpx.Auth):
    def __init__(self, credential_source: CodexCredentialSource) -> None:
        self._credential_source = credential_source

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if not self._is_trusted_codex_url(request.url):
            yield request
            return

        replayable = self._is_replayable_request(request)
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
            url.scheme == _BASE_URL.scheme
            and url.host == _BASE_URL.host
            and url.port is None
            and raw_path.startswith(_BASE_URL.raw_path.rstrip(b'/') + b'/')
            and b'%' not in raw_path
            and b'\\' not in raw_path
            and '.' not in path_segments
            and '..' not in path_segments
        )

    def _is_replayable_request(self, request: httpx.Request) -> bool:
        """Whether re-sending the request after a refresh is safe.

        The Responses endpoint and its token-counting sibling (reached through
        `UsageLimits(count_tokens_before_request=True)`) both create nothing that a replay would
        duplicate, so a rotated credential can retry them.
        """
        return request.method == 'POST' and request.url.path.rstrip('/') in (
            _RESPONSES_PATH,
            f'{_RESPONSES_PATH}/input_tokens',
        )


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
    def model_profile(model_name: str) -> ModelProfile:
        return merge_profile(
            openai_model_profile(model_name),
            OpenAIModelProfile(
                openai_responses_requires_store_false=True,
                openai_responses_requires_stream=True,
                # The Codex backend answers `400 Unsupported parameter` for each of these, so a portable
                # `ModelSettings` that merely sets one would fail every request. Only generic settings are
                # dropped: `openai_`-prefixed ones are an explicit opt-in into OpenAI semantics, so the
                # backend error is the more useful outcome there. `openai_store` is the exception —
                # `store=False` is a hard backend requirement rather than a capability, so it is forced.
                openai_unsupported_model_settings=('max_tokens', 'temperature', 'top_p'),
            ),
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
            http_client = create_async_http_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_http_client
        else:
            if http_client.auth is not None:
                raise UserError('`http_client` must not already have authentication configured.')
            if http_client.follow_redirects:
                raise UserError('`http_client` must have `follow_redirects=False`.')
        # A client the base class recreates after close is re-authenticated through `_set_http_client`.
        http_client.auth = _CodexHTTPAuth(credential_source)

        # AsyncOpenAI requires a non-empty API key even though the HTTP auth layer
        # replaces the generated Authorization header before every request.
        self._client = AsyncOpenAI(
            base_url=str(_BASE_URL),
            api_key='codex-subscription-auth',
            http_client=http_client,
        )

    def _set_http_client(self, http_client: httpx.AsyncClient) -> None:
        http_client.auth = _CodexHTTPAuth(self._credential_source)
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]

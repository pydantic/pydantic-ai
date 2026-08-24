from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator

import httpx2

from pydantic_ai._http import AsyncHTTPClient, create_async_httpx2_client
from pydantic_ai.auth.openai_codex import OpenAICodexAuth, OpenAICodexCredentials, OpenAICodexCredentialSource
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles import ModelProfile, merge_profile
from pydantic_ai.profiles.openai import OpenAIModelProfile, openai_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the OpenAI Codex provider, '
        'you can use the `openai-codex` optional group — `pip install "pydantic-ai-slim[openai-codex]"`. '
        'The `openai` group alone omits the `filelock` the default credential store needs.'
    ) from _import_error

# Base URL observed in the pinned official Codex client for ChatGPT-authenticated requests.
# `OpenAICodexProvider.base_url` exposes it, so it stays private rather than becoming the one public
# base-URL constant in any provider module.
_BASE_URL = httpx2.URL('https://chatgpt.com/backend-api/codex')
_RESPONSES_PATH = f'{_BASE_URL.path}/responses'

# The official Codex CLI sends `User-Agent: codex_cli_rs/<version>`, which was once needed to route
# newer model slugs. The backend no longer distinguishes: the SDK's own User-Agent and the Codex CLI's
# return identical results on every slug, including the ones an account is not entitled to, which are
# refused for the account rather than the client. So we send the ordinary Pydantic AI/OpenAI SDK
# User-Agent and identify ourselves through the `originator` header instead.


class _OpenAICodexHTTPAuth(httpx2.Auth):
    def __init__(self, credential_source: OpenAICodexCredentialSource) -> None:
        self._credential_source = credential_source

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        if not self._is_trusted_openai_codex_url(request.url):
            yield request
            return

        replayable = self._is_replayable_request(request)
        if replayable:
            try:
                request.content
            except httpx2.RequestNotRead:
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

    def _apply(self, request: httpx2.Request, credentials: OpenAICodexCredentials) -> None:
        request.headers['Authorization'] = f'Bearer {credentials.access_token.get_secret_value()}'
        request.headers['ChatGPT-Account-ID'] = credentials.account_id.get_secret_value()
        request.headers['originator'] = 'pydantic-ai'
        if credentials.account_is_fedramp:
            request.headers['X-OpenAI-Fedramp'] = 'true'
        else:
            request.headers.pop('X-OpenAI-Fedramp', None)

    def _is_trusted_openai_codex_url(self, url: httpx2.URL) -> bool:
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

    def _is_replayable_request(self, request: httpx2.Request) -> bool:
        """Whether re-sending the request after a refresh is safe.

        Both Responses flavors the backend serves create nothing that a replay would duplicate —
        `store=False` is forced, so neither leaves server-side state behind — so a rotated credential
        can retry them. `/responses/input_tokens` is deliberately absent: the backend does not route
        it, and `openai_responses_supports_input_tokens_count=False` stops `count_tokens` before a
        request is ever built.
        """
        return request.method == 'POST' and request.url.path.rstrip('/') in (
            _RESPONSES_PATH,
            f'{_RESPONSES_PATH}/compact',
        )


class OpenAICodexProvider(Provider[AsyncOpenAI]):
    """Provider for OpenAI Codex models accessed with ChatGPT subscription credentials.

    Authentication is resolved lazily for each request. Constructing the provider
    never reads credential storage, opens a browser, or starts background work.

    Args:
        credential_source: Application-owned credentials. Defaults to [`OpenAICodexAuth`]
            [pydantic_ai.auth.openai_codex.OpenAICodexAuth] and its managed local credential store.
        http_client: A dedicated caller-owned `httpx2.AsyncClient` with no existing auth and
            `follow_redirects=False`. OpenAI Codex authentication is installed on this client,
            and the provider never closes it.
    """

    @property
    def name(self) -> str:
        return 'openai-codex'

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
                # The backend does not route `/responses/input_tokens`; the edge answers it with the
                # same challenge page any unknown path gets, so counting would fail as an HTML-bodied
                # `ModelHTTPError` rather than tell the caller the endpoint isn't there.
                openai_responses_supports_input_tokens_count=False,
                # `/responses/compact` answers with a `compaction_summary` item, after a `message`
                # item, where the OpenAI Platform API returns a lone `compaction`.
                openai_responses_compaction_item_type='compaction_summary',
                # The OpenAI Codex backend answers `400 Unsupported parameter` for each of these, so a portable
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
        credential_source: OpenAICodexCredentialSource | None = None,
        http_client: httpx2.AsyncClient | None = None,
    ) -> None:
        if credential_source is None:
            credential_source = OpenAICodexAuth()
        self._credential_source = credential_source

        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = create_async_httpx2_client
        http_client = self._prepare_http_client(http_client)

        # AsyncOpenAI requires a non-empty API key even though the HTTP auth layer
        # replaces the generated Authorization header before every request.
        self._client = AsyncOpenAI(
            base_url=str(_BASE_URL),
            api_key='codex-subscription-auth',
            http_client=http_client,
        )

    def _prepare_http_client(self, http_client: AsyncHTTPClient) -> httpx2.AsyncClient:
        if not isinstance(http_client, httpx2.AsyncClient):
            raise UserError('`http_client` must be an `httpx2.AsyncClient`.')
        if http_client.auth is not None:
            raise UserError('`http_client` must not already have authentication configured.')
        if http_client.follow_redirects:
            raise UserError('`http_client` must have `follow_redirects=False`.')
        http_client.auth = _OpenAICodexHTTPAuth(self._credential_source)
        return http_client

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        http_client = self._prepare_http_client(http_client)
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage]

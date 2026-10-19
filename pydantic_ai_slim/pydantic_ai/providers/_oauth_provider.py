"""Shared credential lifecycle for OpenAI SDK providers that authenticate with OAuth tokens instead of API keys.

[`OpenAICodexProvider`][pydantic_ai.providers.openai_codex.OpenAICodexProvider] and
[`OpenAIChatGPTProvider`][pydantic_ai.providers.openai_chatgpt.OpenAIChatGPTProvider] both inject a
subscription's OAuth access token into OpenAI SDK requests. The provider owns the credential lifecycle:
loading from an application-owned source, refreshing proactively from the unverified JWT `exp` hint,
refreshing single-flight after a 401, and handing rotated tokens back to the source. Subclasses supply
only their endpoint, request headers, and refresh grant.
"""

from __future__ import annotations as _annotations

import base64
import json
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import cached_property
from typing import Any, Generic, Protocol, TypeVar

import anyio
import httpx2
from openai import AsyncOpenAI, OpenAIError
from pydantic import StrictFloat, StrictInt, TypeAdapter, ValidationError

from pydantic_ai._http import create_async_httpx2_client
from pydantic_ai.exceptions import ModelAPIError, UserError

from ._openai_compatible import AsyncHTTPClient, OpenAICompatibleProvider

__all__ = (
    'CredentialsPersistenceError',
    'CredentialsRefreshError',
    'OAuthCredentialSource',
    'OAuthCredentials',
    'OAuthCredentialsAuth',
    'OAuthCredentialsProvider',
    'jwt_claims',
    'jwt_expires_at',
    'post_token_request',
)

# 30s pre-expiry refresh hint (unverified JWT `exp` is a hint, not an authority).
_TOKEN_EXPIRY_BUFFER = timedelta(seconds=30)


class _CredentialsError(ModelAPIError):
    """Base for OAuth credential failures.

    Subclasses [`ModelAPIError`][pydantic_ai.exceptions.ModelAPIError] so the standard handling of
    provider failures (e.g. [`FallbackModel`][pydantic_ai.models.fallback.FallbackModel]) applies;
    the auth layer runs below any specific model, so `model_name` is the provider name.
    """

    # `provider_name` defaults to Codex, the only provider raising these errors before they were shared.
    def __init__(self, message: str, provider_name: str = 'openai-codex'):
        super().__init__(model_name=provider_name, message=message)

    def __reduce__(self) -> tuple[type, tuple[Any, ...]]:
        return self.__class__, (self.message, self.model_name)


class CredentialsRefreshError(_CredentialsError):
    """Exchanging an authorization code or refresh token at the token endpoint failed.

    When the underlying error is `invalid_grant`, the stored grant is no longer usable and a fresh
    authorization is required (for Codex locally: rerun `codex login`; in an app: rerun your sign-in flow).
    """


# The OpenAI SDK retries arbitrary transport exceptions, but propagates `OpenAIError` unchanged.
class CredentialsPersistenceError(_CredentialsError, OpenAIError):
    """Rotated credentials were updated in memory but the persistence callback raised.

    The in-memory credentials are current and were handed to the callback before it failed; the
    error surfaces so callers do not mistake a failed save for durability.
    """


class OAuthCredentials(Protocol):
    """What the shared lifecycle reads from a provider's credentials."""

    @property
    def access_token(self) -> str: ...


CredentialsT = TypeVar('CredentialsT', bound=OAuthCredentials)


class OAuthCredentialSource(Protocol[CredentialsT]):
    """Application-owned storage for a provider's credentials, so refreshed tokens outlive the process.

    The provider owns the credential lifecycle (expiry checks, refresh, single-flight) and calls
    this only to read and write the stored set: `load()` on first use, and `save()` after tokens
    are refreshed.

    Refresh tokens are single-use, so before refreshing, the provider re-reads storage with
    `load()` and adopts newer credentials if another process using the same store already
    refreshed them. That is best-effort, not mutual exclusion: two processes can still race
    between `load()` and `save()`.

    Conformance is structural, but implementations are encouraged to subclass the protocol
    explicitly so type checkers verify the method signatures.
    """

    async def load(self) -> CredentialsT:
        """Return the currently stored credentials."""
        ...

    async def save(self, credentials: CredentialsT) -> None:
        """Durably replace the stored credentials with a freshly rotated set."""
        ...


_ClaimsT = TypeVar('_ClaimsT')
_ResponseT = TypeVar('_ResponseT')


def jwt_claims(token: str, claims: TypeAdapter[_ClaimsT]) -> _ClaimsT | None:
    """Decode a JWT payload without verifying the signature. Returns `None` for anything malformed."""
    try:
        segment = token.split('.')[1]
    except IndexError:
        return None
    padded = segment + '=' * (-len(segment) % 4)
    try:
        return claims.validate_python(json.loads(base64.urlsafe_b64decode(padded)))
    except ValueError:
        return None


@dataclass
class _JwtExpiry:
    # Strict, so a numeric string is not coerced into an expiry hint.
    exp: StrictInt | StrictFloat | None = None


_jwt_expiry_ta = TypeAdapter(_JwtExpiry)


def jwt_expires_at(token: str) -> datetime | None:
    """Best-effort unverified `exp` claim, a refresh hint, never an authority."""
    payload = jwt_claims(token, _jwt_expiry_ta)
    if payload is None or payload.exp is None:
        return None
    try:
        return datetime.fromtimestamp(payload.exp, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


@dataclass
class _TokenErrorResponse:
    """An OAuth token-endpoint error body."""

    error: str | None = None
    error_description: str | None = None


_token_error_response_ta = TypeAdapter(_TokenErrorResponse)


async def post_token_request(
    url: str,
    form: Mapping[str, str],
    response: TypeAdapter[_ResponseT],
    *,
    provider_name: str,
    http_client: httpx2.AsyncClient | None = None,
    rejected_grant_errors: frozenset[str] = frozenset({'invalid_grant'}),
) -> _ResponseT:
    """POST a form-urlencoded OAuth token request and validate the JSON response with `response`.

    When `http_client` is given the request goes through it (so custom transports and proxies apply
    to refreshes too); otherwise an ephemeral client is used. Failures raise
    [`CredentialsRefreshError`][pydantic_ai.providers.openai_codex.CredentialsRefreshError], whose
    message says the grant was rejected when the error code is one of `rejected_grant_errors`.
    """
    if http_client is None:
        async with httpx2.AsyncClient(timeout=httpx2.Timeout(timeout=30, connect=5)) as client:
            result = await client.post(url, data=dict(form), headers={'Accept': 'application/json'})
    else:
        result = await http_client.post(url, data=dict(form), headers={'Accept': 'application/json'})
    if result.status_code != 200:
        try:
            body = _token_error_response_ta.validate_python(result.json())
        except ValueError:
            body = _TokenErrorResponse()
        detail = body.error_description or body.error or result.text[:200]
        hint = '; the grant was rejected, rerun the authorization flow' if body.error in rejected_grant_errors else ''
        raise CredentialsRefreshError(
            f'Token request to {url} failed with status {result.status_code}: {detail}{hint}', provider_name
        )
    try:
        return response.validate_python(result.json())
    except ValueError as e:
        detail = e.json(include_input=False) if isinstance(e, ValidationError) else str(e)
        raise CredentialsRefreshError(
            f'Token endpoint {url} returned an unexpected response.\n\n{detail}', provider_name
        ) from None


class OAuthCredentialsAuth(httpx2.Auth):
    """httpx auth injecting a provider's OAuth headers, with single-flight refresh-and-replay.

    Injects the provider's headers only on HTTPS requests to the provider's host, so a
    caller-supplied client reused for other destinations (or downgraded to plaintext) never leaks
    credentials. On a 401 it performs at most one refresh-and-replay; non-expiry 401s therefore
    cannot loop. The proactive expiry check treats the unverified JWT `exp` as a hint only.
    """

    def __init__(self, provider: OAuthCredentialsProvider[Any]) -> None:
        self._provider = provider

    def sync_auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        # `httpx.Auth`'s default would send the request unauthenticated; refresh-and-replay is async.
        raise UserError(f'`{type(self._provider).__name__}` requires an async HTTP client to inject credentials.')

    async def async_auth_flow(self, request: httpx2.Request) -> AsyncGenerator[httpx2.Request, httpx2.Response]:
        provider = self._provider
        if request.url.scheme != 'https' or request.url.host != provider._host:  # pyright: ignore[reportPrivateUsage]
            # Never send subscription credentials to a foreign destination or over plaintext:
            # a caller-supplied client may be reused for arbitrary requests.
            yield request
            return
        # Buffer the outgoing body so the 401 replay can resend it: a one-shot stream (an async
        # generator upload) would otherwise raise `StreamConsumed` on the second send. Inference
        # payloads are JSON already held in memory, and foreign-host requests bypass this above.
        await request.aread()
        # The provider owns the state and the auth is its wire-side skin. The replay closure carries
        # whatever context the provider's mode (in-memory single-flight, or application credential
        # source) needs.
        credentials, replay = await provider._prepare_request_credentials()  # pyright: ignore[reportPrivateUsage]
        request.headers.update(provider._auth_headers(credentials))  # pyright: ignore[reportPrivateUsage]
        response = yield request
        if response.status_code != 401:
            return
        # Release the connection before replaying.
        await response.aread()
        request.headers.update(provider._auth_headers(await replay()))  # pyright: ignore[reportPrivateUsage]
        yield request  # replay exactly once; its response goes back to the caller


class OAuthCredentialsProvider(OpenAICompatibleProvider, ABC, Generic[CredentialsT]):
    """An OpenAI SDK provider that injects refreshable OAuth credentials instead of an API key.

    One provider instance carries one set of credentials (there is no process-global cache). The
    instance binds its refresh lock to the first event loop that awaits a request, so do not reuse
    it across loops.
    """

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    @property
    def credentials(self) -> CredentialsT:
        """The credentials currently held in memory, rotated in place by refreshes."""
        if self._credentials is None:
            raise UserError(
                '`credentials` is unavailable: the provider either wraps an existing `openai_client`, '
                'which opts out of credential injection entirely, or has a `credential_source` it has '
                'not loaded from yet.'
            )
        return self._credentials

    def __init__(
        self,
        credentials: CredentialsT | None,
        *,
        credential_source: OAuthCredentialSource[CredentialsT] | None,
        openai_client: AsyncOpenAI | None,
        http_client: AsyncHTTPClient | None,
    ) -> None:
        """Set up credential injection; subclasses resolve their default credentials first.

        Exactly one of `credentials`, `credential_source`, and `openai_client` must be given.
        """
        self._credential_source = credential_source
        self._credentials: CredentialsT | None = None
        if openai_client is not None:
            assert credentials is None, 'Cannot provide both `openai_client` and `credentials`'
            assert credential_source is None, 'Cannot provide both `openai_client` and `credential_source`'
            assert http_client is None, 'Cannot provide both `openai_client` and `http_client`'
            self._client = openai_client
            return

        if credential_source is not None:
            # Loaded lazily on first use: construction stays synchronous and does no I/O.
            assert credentials is None, 'Cannot provide both `credentials` and `credential_source`'
        else:
            assert credentials is not None, 'Subclasses resolve default credentials before calling `__init__`'
            self._validate_credentials(credentials)
            self._credentials = credentials
        self._host = httpx2.URL(self.base_url).host
        self._revision = 0
        self._refresh_failures = 0
        self._last_refresh_error: tuple[int, Exception] | None = None
        # What the credential source still held when saving a refresh failed: the set that refresh
        # spent, which must not be adopted back over the unsaved replacement in memory.
        self._unsaved_predecessor: CredentialsT | None = None
        self._auth = OAuthCredentialsAuth(self)
        if http_client is None:
            http_client = create_async_httpx2_client()
            self._own_http_client = http_client
            self._http_client_factory = self._create_http_client
        else:
            if not isinstance(http_client, httpx2.AsyncClient):
                raise UserError(
                    f'`{type(self).__name__}` requires an `httpx2` client for `http_client`: the legacy '
                    '`httpx.AsyncClient` cannot carry its credential-injecting auth.'
                )
            if http_client.auth is not None:
                raise UserError(
                    'The `http_client` already has auth configured (it may belong to another provider); '
                    'pass a dedicated client so credentials cannot mix between providers.'
                )
        http_client.auth = self._auth
        self._http_client = http_client
        self._client = AsyncOpenAI(
            base_url=self.base_url,
            # The SDK merges its own bearer header into requests; `OAuthCredentialsAuth` replaces it.
            api_key='oauth-credentials',
            http_client=http_client,
        )

    @abstractmethod
    def _auth_headers(self, credentials: CredentialsT) -> dict[str, str]:
        """The headers carrying `credentials` on a request to the provider's host."""
        raise NotImplementedError

    @abstractmethod
    async def _refresh_grant(self, credentials: CredentialsT, http_client: httpx2.AsyncClient) -> CredentialsT:
        """Exchange the refresh token in `credentials` for a new credential set."""
        raise NotImplementedError

    def _validate_credentials(self, credentials: CredentialsT) -> None:
        """Reject credentials this provider cannot use, at construction and before each request."""

    def _create_http_client(self) -> httpx2.AsyncClient:
        """Factory used when a closed provider-owned client is reopened."""
        client = create_async_httpx2_client()
        client.auth = self._auth
        self._http_client = client
        return client

    def _set_http_client(self, http_client: AsyncHTTPClient) -> None:
        http_client.auth = self._auth  # pyright: ignore[reportAttributeAccessIssue]
        self._client._client = http_client  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue]

    async def _prepare_request_credentials(self) -> tuple[CredentialsT, Callable[[], Awaitable[CredentialsT]]]:
        """The credentials for an outgoing request, plus a replay resolver for a 401 on it."""
        await self._load_if_needed()
        # Checked before and after every refresh, so refused credentials never reach the API or
        # spend another refresh token, however they were given, loaded, adopted, or refreshed.
        self._validate_credentials(self.credentials)
        await self._refresh_if_stale()
        self._validate_credentials(self.credentials)
        revision_used = self._revision
        # Only share failures that happen after this request starts, not failures from earlier requests.
        refresh_failures = self._refresh_failures

        async def replay() -> CredentialsT:
            await self._refresh_for_401(revision_used, refresh_failures=refresh_failures)
            self._validate_credentials(self.credentials)
            return self.credentials

        return self.credentials, replay

    async def _load_if_needed(self) -> None:
        """First use with a `credential_source`: load the stored credentials."""
        if self._credentials is not None:
            return
        async with self._refresh_lock:
            if self._credentials is None:  # recheck after acquiring: load once, not once per task
                assert self._credential_source is not None
                self._credentials = await self._credential_source.load()

    @cached_property
    def _refresh_lock(self) -> anyio.Lock:
        # Like the base provider's enter lock: bind lazily so we attach to whatever loop first
        # awaits a request rather than construction time.
        return anyio.Lock()

    async def _refresh_if_stale(self) -> None:
        """Proactive refresh from the unverified-JWT `exp` hint.

        Failures are swallowed here: the hint is best-effort, and the 401 path retries with real
        errors surfaced.
        """
        if not self._is_stale():
            return
        try:
            async with self._refresh_lock:
                if self._is_stale():  # recheck after acquiring: single-flight, not just serialized
                    await self._refresh_locked()
        except CredentialsPersistenceError:
            raise  # the refresh itself succeeded; a failed save must never be silent
        except Exception:
            # Transport failures and rejected grants alike fall through to the 401 path, which
            # retries with the still-current token and surfaces real errors.
            pass

    async def _refresh_for_401(self, revision_used: int, *, refresh_failures: int) -> None:
        """Single-flight refresh after a 401 carrying `revision_used`.

        If another task already replaced the credentials since the failed request was sent, no
        network refresh happens: the caller replays with the fresh set directly.
        """
        if self._revision != revision_used:
            return
        async with self._refresh_lock:
            if self._revision != revision_used:  # recheck after acquiring
                return
            if (
                (last := self._last_refresh_error) is not None
                and last[0] == revision_used
                and self._refresh_failures != refresh_failures
            ):
                raise last[1]  # share the single-flight failure instead of re-running it per waiter
            try:
                await self._refresh_locked()
            except Exception as e:
                self._refresh_failures += 1
                self._last_refresh_error = (revision_used, e)
                raise

    async def _refresh_locked(self) -> None:
        # The caller must hold `_refresh_lock`.
        assert self._refresh_lock.locked()
        rejected = self.credentials
        stored: CredentialsT | None = None
        if (source := self._credential_source) is not None:
            # Refresh tokens are single-use, so spending ours when another process already refreshed
            # would invalidate theirs. Re-read storage first and adopt a newer set if one is there.
            stored = await source.load()
            if stored != rejected and stored != self._unsaved_predecessor:
                self._replace(stored)
                self._unsaved_predecessor = None
                return
        # The provider's own client, so custom transports and proxies apply to refreshes too;
        # the auth flow ignores other hosts, so this cannot recurse or leak the bearer.
        self._replace(await self._refresh_grant(rejected, self._http_client))
        if source is not None:
            try:
                await source.save(self.credentials)
            except Exception as e:
                self._unsaved_predecessor = stored
                raise CredentialsPersistenceError(
                    'Credentials were refreshed in memory but saving them to the credential source raised.',
                    self.name,
                ) from e
            self._unsaved_predecessor = None

    def _replace(self, credentials: CredentialsT) -> None:
        # Atomic replace of the complete set, then bump the revision so concurrent 401s observe it.
        self._credentials = credentials
        self._revision += 1

    def _is_stale(self) -> bool:
        # The unverified JWT `exp` claim is a refresh hint, not an authority; refresh proactively
        # once the token is within the pre-expiry buffer.
        expires_at = jwt_expires_at(self.credentials.access_token)
        return expires_at is not None and datetime.now(UTC) >= expires_at - _TOKEN_EXPIRY_BUFFER

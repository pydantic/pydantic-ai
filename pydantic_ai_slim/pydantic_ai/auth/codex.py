"""Core authentication lifecycle for Codex subscriptions.

The CLI is deliberately not imported here. Applications can use the same login,
refresh, status, and logout operations with their own interaction and storage.
"""

from __future__ import annotations as _annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import anyio
import httpx
from anyio.lowlevel import checkpoint
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from ..exceptions import UserError

_REFRESH_SKEW = timedelta(minutes=5)
_REFRESH_AND_SAVE_TIMEOUT = 60


class CodexAuthError(UserError):
    """Base class for Codex authentication errors."""


class CodexLoginRequiredError(CodexAuthError):
    """Raised when no usable Codex subscription credentials are available."""


class CodexCredentialsError(CodexAuthError):
    """Raised when Codex credentials cannot be loaded or stored safely."""


class CodexOAuthError(CodexAuthError):
    """Raised when a Codex login or logout protocol operation fails."""


class CodexRefreshError(CodexAuthError):
    """Raised when Codex credentials cannot be refreshed."""


class CodexAccountMismatchError(CodexRefreshError):
    """Raised when refreshing credentials would switch ChatGPT accounts."""


class CodexCredentials(BaseModel):
    """A complete, immutable Codex credential snapshot.

    Token and account values use [`SecretStr`][pydantic.types.SecretStr], so normal
    representations and serialization do not reveal them. Credential stores must
    explicitly unwrap these values in their narrowly scoped persistence code.
    """

    model_config = ConfigDict(frozen=True, extra='forbid', hide_input_in_errors=True)

    access_token: SecretStr
    refresh_token: SecretStr
    id_token: SecretStr
    expires_at: datetime
    account_id: SecretStr
    revision: str
    """Opaque credential version used for conditional refresh and persistence."""
    account_is_fedramp: bool = False

    @field_validator('expires_at')
    @classmethod
    def _expires_at_is_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('expires_at must be timezone-aware')
        return value

    def is_valid(self, *, skew: timedelta = _REFRESH_SKEW) -> bool:
        """Return whether the access token remains valid beyond `skew`."""
        return self.expires_at > datetime.now(timezone.utc) + skew


class CodexDeviceCode(BaseModel):
    """UI-safe information needed to complete device authorization."""

    model_config = ConfigDict(frozen=True, extra='forbid', hide_input_in_errors=True)

    verification_url: str
    user_code: SecretStr
    expires_at: datetime


class CodexAuthStatus(BaseModel):
    """Secret-free status for managed Codex credentials."""

    model_config = ConfigDict(frozen=True, extra='forbid', hide_input_in_errors=True)

    authenticated: bool
    expires_at: datetime | None = None
    needs_refresh: bool = False
    account_is_fedramp: bool = False


class CodexLogoutResult(BaseModel):
    """Result of removing managed Codex credentials."""

    model_config = ConfigDict(frozen=True, extra='forbid', hide_input_in_errors=True)

    local_credentials_removed: bool
    upstream_revoked: bool = False
    revocation_error: str | None = None


class CodexCredentialSource(Protocol):
    """Resolve one coherent credential snapshot for a Codex request."""

    async def get_credentials(
        self, *, force_refresh: bool = False, rejected_revision: str | None = None
    ) -> CodexCredentials:
        """Return credentials, optionally replacing the version rejected by the service."""
        ...


class CodexCredentialStore(Protocol):
    """Persistence boundary for concurrency-safe Codex credential rotation.

    `exclusive()` must prevent another actor sharing the same logical record from
    rotating it until the context exits. File stores can use a process lock;
    databases can use a lease or row lock.
    """

    def exclusive(self) -> AbstractAsyncContextManager[None]:
        """Acquire exclusive ownership of the active Codex credential record."""
        ...

    async def load(self) -> CodexCredentials | None:
        """Load the active credential snapshot, if present."""
        ...

    async def save(self, credentials: CodexCredentials, *, expected_revision: str | None) -> bool:
        """Conditionally replace the active snapshot.

        Return `False` when the active revision differs from `expected_revision`.
        An expected revision of `None` means the record must not already exist.
        """
        ...

    async def delete(self, *, expected_revision: str | None) -> bool:
        """Conditionally delete the active snapshot, returning whether it was removed."""
        ...


class CodexAuth(CodexCredentialSource):
    """Manage Codex subscription login, persistence, refresh, and logout.

    Constructing this object performs no file, network, browser, or background-task
    work. The default file store is opened only by an explicit operation or request.

    Args:
        store: Application-owned persistence. Defaults to the permission-hardened
            `~/.pydantic-ai/auth.json` file store.
        path: Override the default file path. Mutually exclusive with `store`.
        http_client: Caller-owned client for OAuth protocol requests. It is never closed.
    """

    def __init__(
        self,
        *,
        store: CodexCredentialStore | None = None,
        path: Path | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if store is not None and path is not None:
            raise ValueError('`store` and `path` are mutually exclusive')
        if store is None:
            from ._codex_store import FileCodexCredentialStore

            store = FileCodexCredentialStore(path)
        self._store = store
        self._http_client = http_client

    async def get_credentials(
        self, *, force_refresh: bool = False, rejected_revision: str | None = None
    ) -> CodexCredentials:
        """Return current credentials and safely rotate them when required.

        `rejected_revision` identifies the credential version that received an
        unauthorized response. If another actor has already replaced that version,
        the replacement is reused instead of rotating its refresh token again.
        """
        if rejected_revision is not None and not force_refresh:
            raise ValueError('`rejected_revision` requires `force_refresh=True`')

        observed = await self._load_required()
        if rejected_revision is not None and observed.revision != rejected_revision and observed.is_valid():
            return observed
        if not force_refresh and observed.is_valid():
            return observed

        async with self._exclusive_transaction():
            current = await self._load_required()
            if rejected_revision is not None and current.revision != rejected_revision and current.is_valid():
                resolved = current
            elif current.revision != observed.revision and current.is_valid():
                resolved = current
            elif not force_refresh and current.is_valid():
                resolved = current
            else:
                try:
                    with anyio.fail_after(_REFRESH_AND_SAVE_TIMEOUT):
                        resolved = await self._refresh_and_save(current)
                except TimeoutError:
                    raise CodexRefreshError('Codex credential refresh timed out.') from None

        # Deliver cancellation only after a rotated token is durably stored and the
        # exclusive store context has released its lock.
        await checkpoint()
        return resolved

    async def login_browser(
        self,
        open_url: Callable[[str], object | Awaitable[object]],
        *,
        timeout: float = 300,
    ) -> CodexCredentials:
        """Complete browser PKCE login and persist the resulting credentials.

        The loopback listener is active before `open_url` is called. The callback is
        responsible only for presenting or opening the supplied URL.
        """
        from ._codex_oauth import CodexOAuthClient

        credentials = await CodexOAuthClient(self._http_client).login_browser(open_url, timeout=timeout)
        await self._replace_after_login(credentials)
        return credentials

    async def login_device(
        self,
        show_code: Callable[[CodexDeviceCode], object | Awaitable[object]],
        *,
        timeout: float = 900,
    ) -> CodexCredentials:
        """Complete device authorization and persist the resulting credentials."""
        from ._codex_oauth import CodexOAuthClient

        credentials = await CodexOAuthClient(self._http_client).login_device(show_code, timeout=timeout)
        await self._replace_after_login(credentials)
        return credentials

    async def status(self) -> CodexAuthStatus:
        """Return secret-free status without refreshing credentials."""
        credentials = await self._store.load()
        if credentials is None:
            return CodexAuthStatus(authenticated=False)
        return CodexAuthStatus(
            authenticated=True,
            expires_at=credentials.expires_at,
            needs_refresh=not credentials.is_valid(),
            account_is_fedramp=credentials.account_is_fedramp,
        )

    async def refresh(self) -> CodexCredentials:
        """Force the same safe refresh used for request-time unauthorized recovery."""
        return await self.get_credentials(force_refresh=True)

    async def logout(self, *, local_only: bool = False) -> CodexLogoutResult:
        """Remove local credentials, attempting verified upstream revocation first.

        Revocation is best effort, matching the official Codex client. Local removal
        still proceeds when the upstream operation fails.
        """
        async with self._exclusive_transaction():
            credentials = await self._store.load()
            if credentials is None:
                result = CodexLogoutResult(local_credentials_removed=False)
            else:
                revoked = False
                revocation_error: str | None = None
                try:
                    if not local_only:
                        from ._codex_oauth import CodexOAuthClient

                        await CodexOAuthClient(self._http_client).revoke(credentials)
                        revoked = True
                except CodexAuthError:
                    revocation_error = 'Upstream Codex token revocation failed.'
                finally:
                    removed = await self._store.delete(expected_revision=credentials.revision)

                if not removed:  # pragma: no cover - an exclusive store must make this unreachable
                    raise CodexCredentialsError('Codex credentials changed while they were being removed.')
                result = CodexLogoutResult(
                    local_credentials_removed=True,
                    upstream_revoked=revoked,
                    revocation_error=revocation_error,
                )

        await checkpoint()
        return result

    @asynccontextmanager
    async def _exclusive_transaction(self) -> AsyncGenerator[None]:
        """Acquire cancellably, then shield the transaction and lock release."""
        exclusive = self._store.exclusive()
        await exclusive.__aenter__()
        with anyio.CancelScope(shield=True):
            try:
                yield
            except BaseException as error:
                await exclusive.__aexit__(type(error), error, error.__traceback__)
                raise
            else:
                await exclusive.__aexit__(None, None, None)

    async def _refresh_and_save(self, current: CodexCredentials) -> CodexCredentials:
        from ._codex_oauth import CodexOAuthClient

        refreshed = await CodexOAuthClient(self._http_client).refresh(current)
        saved = await self._store.save(refreshed, expected_revision=current.revision)
        if not saved:  # pragma: no cover - an exclusive store must make this unreachable
            raise CodexCredentialsError('Codex credentials changed while they were being refreshed.')
        return refreshed

    async def _load_required(self) -> CodexCredentials:
        credentials = await self._store.load()
        if credentials is None:
            raise CodexLoginRequiredError(
                'Codex subscription login is required. Run `clai auth login codex` or inject a credential source.'
            )
        return credentials

    async def _replace_after_login(self, credentials: CodexCredentials) -> None:
        async with self._exclusive_transaction():
            current = await self._store.load()
            expected_revision = current.revision if current is not None else None
            saved = await self._store.save(credentials, expected_revision=expected_revision)
            if not saved:  # pragma: no cover - an exclusive store must make this unreachable
                raise CodexCredentialsError('Codex credentials changed while login was being saved.')
        await checkpoint()

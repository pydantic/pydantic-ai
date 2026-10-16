"""GitHub device authorization, with browser interaction and storage left to the caller."""

from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import Annotated, Literal

import anyio
import httpx2
from pydantic import Field, TypeAdapter, ValidationError

from pydantic_ai.exceptions import UserError


@dataclass(frozen=True, kw_only=True)
class GitHubCopilotDeviceAuthorization:
    """A GitHub device authorization challenge. Display `user_code` at `verification_uri`.

    `device_code` is used only by the token exchange and is excluded from `repr`.
    `expires_in` and `interval` are seconds, as returned by GitHub.
    """

    device_code: Annotated[str, Field(min_length=1)] = field(repr=False)
    user_code: Annotated[str, Field(min_length=1)]
    verification_uri: Literal['https://github.com/login/device']
    expires_in: Annotated[int, Field(strict=True, gt=0)]
    interval: Annotated[int, Field(strict=True, gt=0)] = 5


@dataclass(frozen=True, kw_only=True)
class GitHubCopilotCredentials:
    """GitHub OAuth credentials, with tokens excluded from `repr`.

    Pass `access_token` to `GitHubCopilotProvider(api_key=...)`. GitHub authorization does not
    establish Copilot entitlement: the inference API checks the account's subscription and policy.
    Expiry durations, when present, are seconds from issuance. Applications own persistence and
    renewal; the provider does not refresh a token passed through `api_key`.
    """

    access_token: Annotated[str, Field(min_length=1)] = field(repr=False)
    token_type: Literal['bearer']
    scope: str
    refresh_token: Annotated[str | None, Field(min_length=1)] = field(default=None, repr=False)
    expires_in: Annotated[int | None, Field(strict=True, gt=0)] = None
    refresh_token_expires_in: Annotated[int | None, Field(strict=True, gt=0)] = None


class GitHubCopilotOAuthFlow:
    """One GitHub.com device-login flow for a caller-supplied OAuth application.

    Construction does no I/O. Call `start`, display the returned challenge, then call
    `wait_for_authorization`. No browser is opened and no credentials are persisted.
    Use one instance per login attempt; do not call its methods concurrently.
    """

    def __init__(self, *, client_id: str, scope: str = '', http_client: httpx2.AsyncClient | None = None) -> None:
        """Configure GitHub's device flow without borrowing another application's client identity.

        Args:
            client_id: Your registered OAuth application's client ID. Device flow must be enabled
                in its settings. GitHub authorization alone does not guarantee Copilot API access.
            scope: Space-separated GitHub OAuth scopes. No scopes are requested by default.
            http_client: Optional caller-owned client. Otherwise each request uses a temporary
                client that is closed before returning. Redirects are not followed.
        """
        if not client_id.strip():
            raise UserError('`client_id` must be a non-empty GitHub OAuth application client ID.')
        self._client_id = client_id
        self._scope = scope
        self._http_client = http_client
        self._authorization: tuple[GitHubCopilotDeviceAuthorization, float] | None = None

    async def start(self) -> GitHubCopilotDeviceAuthorization:
        """Request a new device code. Replaces any previous challenge on this instance.

        The local expiry deadline uses `expires_in` from receipt of the response.
        Raises `UserError` for an HTTP failure, rejected client, or malformed response.
        Transport errors propagate unchanged.
        """
        self._authorization = None
        response = await self._post('/login/device/code', {'client_id': self._client_id, 'scope': self._scope})
        received_at = monotonic()
        try:
            result = _DEVICE_RESPONSE.validate_json(response.content)
        except ValidationError:
            raise UserError('GitHub returned an invalid device authorization response.') from None
        if isinstance(result, _OAuthError):
            raise UserError(f'GitHub device authorization failed: {result.error}.')
        self._authorization = result, received_at + result.expires_in
        return result

    async def wait_for_authorization(self) -> GitHubCopilotCredentials:
        """Poll until approval, rejection, or expiry; cancellation propagates to the caller.

        Waits at least GitHub's interval before each request, including the first, and increases
        it on `slow_down`. The challenge's original deadline also bounds in-flight requests.
        A challenge can be consumed once: call `start` again after success, failure, or cancellation.
        Raises `UserError` for expiry, denial, an HTTP failure, or a malformed response.
        """
        if self._authorization is None:
            raise UserError('Call `start()` before `wait_for_authorization()`.')
        authorization, deadline = self._authorization
        self._authorization = None
        interval = authorization.interval
        try:
            with anyio.fail_after(max(0, deadline - monotonic())):
                while True:
                    await anyio.sleep(interval)
                    response = await self._post(
                        '/login/oauth/access_token',
                        {
                            'client_id': self._client_id,
                            'device_code': authorization.device_code,
                            'grant_type': 'urn:ietf:params:oauth:grant-type:device_code',
                        },
                    )
                    try:
                        result = _TOKEN_RESPONSE.validate_json(response.content)
                    except ValidationError:
                        raise UserError('GitHub returned an invalid device token response.') from None
                    if isinstance(result, GitHubCopilotCredentials):
                        return result
                    if result.error == 'authorization_pending':
                        continue
                    if result.error == 'slow_down':
                        interval = max(interval + 5, result.interval or 0)
                        continue
                    raise UserError(f'GitHub device authorization failed: {result.error}.')
        except TimeoutError:
            raise UserError('GitHub device authorization expired. Call `start()` to request a new code.') from None

    async def _post(self, path: str, form: dict[str, str]) -> httpx2.Response:
        url = f'https://github.com{path}'
        if self._http_client is None:
            async with httpx2.AsyncClient(timeout=httpx2.Timeout(30, connect=5)) as client:
                response = await client.post(
                    url, data=form, headers={'Accept': 'application/json'}, follow_redirects=False
                )
        else:
            response = await self._http_client.post(
                url, data=form, headers={'Accept': 'application/json'}, follow_redirects=False
            )
        if response.status_code != 200:
            raise UserError(f'GitHub device authorization request failed (HTTP {response.status_code}).')
        return response


@dataclass
class _OAuthError:
    error: Annotated[str, Field(min_length=1, pattern=r'^[a-z_]+$')]
    interval: Annotated[int | None, Field(strict=True, gt=0)] = None


_DEVICE_RESPONSE = TypeAdapter[_OAuthError | GitHubCopilotDeviceAuthorization](
    Annotated[_OAuthError | GitHubCopilotDeviceAuthorization, Field(union_mode='left_to_right')]
)
_TOKEN_RESPONSE = TypeAdapter[_OAuthError | GitHubCopilotCredentials](
    Annotated[_OAuthError | GitHubCopilotCredentials, Field(union_mode='left_to_right')]
)

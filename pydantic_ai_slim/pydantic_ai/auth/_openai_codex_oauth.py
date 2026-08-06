from __future__ import annotations as _annotations

import base64
import hashlib
import inspect
import math
import secrets
import socket
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from typing import TypeVar
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx
from anyio.abc import SocketStream
from anyio.streams.stapled import MultiListener
from anyio.to_thread import run_sync as run_sync_in_worker
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from .._utils import BaseExceptionGroup
from ..exceptions import UserError
from ..models import create_async_http_client
from .openai_codex import (
    OpenAICodexAccountMismatchError,
    OpenAICodexAuthError,
    OpenAICodexCredentials,
    OpenAICodexDeviceCode,
    OpenAICodexLoginRequiredError,
    OpenAICodexOAuthError,
    OpenAICodexRefreshError,
)

_ISSUER = 'https://auth.openai.com'
_CLIENT_ID = 'app_EMoamEEZ73f0CkXaXp7hrann'
_AUTHORIZE_URL = f'{_ISSUER}/oauth/authorize'
_TOKEN_URL = f'{_ISSUER}/oauth/token'
_REVOKE_URL = f'{_ISSUER}/oauth/revoke'
_DEVICE_USER_CODE_URL = f'{_ISSUER}/api/accounts/deviceauth/usercode'
_DEVICE_TOKEN_URL = f'{_ISSUER}/api/accounts/deviceauth/token'
_DEVICE_VERIFICATION_URL = f'{_ISSUER}/codex/device'
_DEVICE_REDIRECT_URI = f'{_ISSUER}/deviceauth/callback'
_CALLBACK_PORTS = (1455, 1457)
_CALLBACK_PATH = '/auth/callback'
_SCOPE = 'openid profile email offline_access api.connectors.read api.connectors.invoke'
_DEVICE_CODE_LIFETIME = 900
_REQUEST_TIMEOUT = 30
_CONNECT_TIMEOUT = 10
# Revocation is best effort and runs while the user waits for `logout` to finish, so it gives up
# sooner than an ordinary protocol request.
_REVOCATION_TIMEOUT = 10

_ModelT = TypeVar('_ModelT', bound=BaseModel)
_CallbackT = TypeVar('_CallbackT')


class _TokenAccountMismatchError(OpenAICodexOAuthError):
    pass


class _TokenResponse(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    id_token: SecretStr
    access_token: SecretStr
    refresh_token: SecretStr


class _RefreshResponse(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    id_token: SecretStr | None = None
    access_token: SecretStr | None = None
    refresh_token: SecretStr | None = None


class _DeviceStartResponse(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    device_auth_id: SecretStr
    # The official Codex client accepts either spelling
    # (`#[serde(alias = "user_code", alias = "usercode")]`). Accepting only one would not degrade
    # device login, it would break it outright, so this matches the reference contract.
    user_code: SecretStr = Field(validation_alias=AliasChoices('user_code', 'usercode'))
    # RFC 8628 §3.2 makes `interval` OPTIONAL and requires clients to poll every 5 seconds without it.
    interval: int = 5
    # The issuer states the deadline as an absolute timestamp rather than RFC 8628's `expires_in`
    # duration. It is what the user is shown, so prefer it and fall back to `_DEVICE_CODE_LIFETIME`.
    expires_at: datetime | None = None

    @field_validator('interval', mode='before')
    @classmethod
    def _parse_interval(cls, value: object) -> object:
        if isinstance(value, str):
            return int(value.strip())
        return value

    @field_validator('interval')
    @classmethod
    def _positive_interval(cls, value: int) -> int:
        if value <= 0:
            raise ValueError('interval must be positive')
        return value


class _DevicePollResponse(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    authorization_code: SecretStr
    code_challenge: SecretStr
    code_verifier: SecretStr


class _ErrorDetail(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    code: str | None = None


class _ErrorResponse(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    error: str | _ErrorDetail | None = None
    code: str | None = None

    def error_code(self) -> str | None:
        if isinstance(self.error, str):
            return self.error
        if isinstance(self.error, _ErrorDetail):
            return self.error.code
        return self.code


class _AuthClaims(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    chatgpt_account_id: str | None = None
    chatgpt_account_is_fedramp: bool | None = None


class _JwtClaims(BaseModel):
    model_config = ConfigDict(extra='ignore', hide_input_in_errors=True)

    exp: int | None = None
    auth: _AuthClaims | None = Field(default=None, validation_alias='https://api.openai.com/auth')


class OpenAICodexOAuthClient:
    def __init__(self, http_client: httpx.AsyncClient | None) -> None:
        self._http_client = http_client

    async def login_browser(
        self,
        open_url: Callable[[str], object | Awaitable[object]],
        *,
        timeout: float,
    ) -> OpenAICodexCredentials:
        _validate_timeout(timeout)
        listener, port = await self._bind_callback_listener()
        verifier = _base64url(secrets.token_bytes(64))
        challenge = _base64url(hashlib.sha256(verifier.encode()).digest())
        state = _base64url(secrets.token_bytes(32))
        # `localhost` rather than the `127.0.0.1` that RFC 8252 §8.3 recommends: the authorization
        # server only accepts redirect URIs it has registered, and the registered ones spell it this
        # way. Changing it to a literal loopback address breaks browser login outright.
        redirect_uri = f'http://localhost:{port}{_CALLBACK_PATH}'
        authorization_url = str(
            httpx.URL(
                _AUTHORIZE_URL,
                params={
                    'response_type': 'code',
                    'client_id': _CLIENT_ID,
                    'redirect_uri': redirect_uri,
                    'scope': _SCOPE,
                    'code_challenge': challenge,
                    'code_challenge_method': 'S256',
                    'id_token_add_organizations': 'true',
                    'codex_cli_simplified_flow': 'true',
                    'state': state,
                    'originator': 'pydantic-ai',
                },
            )
        )

        result_send, result_receive = anyio.create_memory_object_stream[OpenAICodexCredentials | OpenAICodexAuthError](
            1
        )
        result: OpenAICodexCredentials | OpenAICodexAuthError | None = None
        async with self._client() as client, listener, result_send, result_receive:

            async def handle(connection: SocketStream) -> None:
                async with connection:
                    callback_result = await self._handle_callback(
                        connection,
                        client=client,
                        redirect_uri=redirect_uri,
                        verifier=verifier,
                        expected_state=state,
                    )
                    if callback_result is not None:
                        await result_send.send(callback_result)

            with _collapse_single_exception_group():
                async with anyio.create_task_group() as task_group:
                    task_group.start_soon(listener.serve, handle)
                    try:
                        with anyio.fail_after(timeout):
                            try:
                                await _invoke_callback(open_url, authorization_url)
                            except Exception:
                                # The callback is user-supplied, so its exception is discarded rather
                                # than chained: an arbitrary browser-launcher failure can carry the
                                # authorization URL, whose `state` and PKCE challenge would then land
                                # in logs and tracebacks. `from None` also clears `__context__`.
                                raise OpenAICodexOAuthError(
                                    'Unable to open or present the OpenAI Codex authorization URL.'
                                ) from None
                            result = await result_receive.receive()
                    except TimeoutError:
                        raise OpenAICodexOAuthError('OpenAI Codex browser authorization timed out.') from None
                    finally:
                        task_group.cancel_scope.cancel()

        if isinstance(result, OpenAICodexAuthError):
            raise result
        # The receive above always assigns a result
        if result is None:  # pragma: no cover
            raise OpenAICodexOAuthError('OpenAI Codex browser authorization did not complete.')
        return result

    async def login_device(
        self,
        show_code: Callable[[OpenAICodexDeviceCode], object | Awaitable[object]],
        *,
        timeout: float,
    ) -> OpenAICodexCredentials:
        _validate_timeout(timeout)
        effective_timeout = min(timeout, _DEVICE_CODE_LIFETIME)
        async with self._client() as client:
            response = await self._send(
                client,
                'POST',
                _DEVICE_USER_CODE_URL,
                json={'client_id': _CLIENT_ID},
            )
            if response.status_code == 404:
                raise OpenAICodexOAuthError(
                    'OpenAI Codex device authorization is not enabled for this account or workspace.'
                )
            if not response.is_success:
                raise OpenAICodexOAuthError('Unable to start OpenAI Codex device authorization.')
            start = self._validate_response(
                response, _DeviceStartResponse, 'OpenAI Codex returned an invalid device code.'
            )
            device_code = OpenAICodexDeviceCode(
                verification_url=_DEVICE_VERIFICATION_URL,
                user_code=start.user_code,
                expires_at=start.expires_at or datetime.now(timezone.utc) + timedelta(seconds=_DEVICE_CODE_LIFETIME),
            )

            interval = float(start.interval)
            try:
                with anyio.fail_after(effective_timeout):
                    try:
                        await _invoke_callback(show_code, device_code)
                    except Exception:
                        # Discarded rather than chained for the same reason as in `login_browser`: the
                        # callback is user-supplied and its exception can carry the one-time user code.
                        raise OpenAICodexOAuthError(
                            'Unable to present the OpenAI Codex device authorization code.'
                        ) from None
                    while True:
                        response = await self._send(
                            client,
                            'POST',
                            _DEVICE_TOKEN_URL,
                            json={
                                'device_auth_id': start.device_auth_id.get_secret_value(),
                                'user_code': start.user_code.get_secret_value(),
                            },
                        )
                        if response.is_success:
                            poll = self._validate_response(
                                response,
                                _DevicePollResponse,
                                'OpenAI Codex returned an invalid device authorization result.',
                            )
                            verifier = poll.code_verifier.get_secret_value()
                            expected_challenge = _base64url(hashlib.sha256(verifier.encode()).digest())
                            if not secrets.compare_digest(expected_challenge, poll.code_challenge.get_secret_value()):
                                raise OpenAICodexOAuthError(
                                    'OpenAI Codex device authorization returned inconsistent PKCE values.'
                                )
                            return await self._exchange_code(
                                client,
                                code=poll.authorization_code,
                                redirect_uri=_DEVICE_REDIRECT_URI,
                                verifier=verifier,
                            )

                        error_code = self._error_code(response)
                        if error_code in ('access_denied', 'authorization_declined'):
                            raise OpenAICodexOAuthError('OpenAI Codex device authorization was declined.')
                        if error_code in ('expired_token', 'device_code_expired'):
                            raise OpenAICodexOAuthError('The OpenAI Codex device authorization code expired.')
                        if error_code == 'slow_down':
                            interval += 5
                            await _sleep(interval)
                            continue
                        if response.status_code in (403, 404) or error_code == 'authorization_pending':
                            await _sleep(interval)
                            continue
                        raise OpenAICodexOAuthError('OpenAI Codex device authorization failed.')
            except TimeoutError:
                raise OpenAICodexOAuthError('OpenAI Codex device authorization timed out.') from None

    async def refresh(self, current: OpenAICodexCredentials) -> OpenAICodexCredentials:
        # A JSON body, not the `application/x-www-form-urlencoded` one RFC 6749 §6 mandates: this
        # endpoint is the ChatGPT service's, and the official Codex client posts JSON to it. The
        # deviation is deliberate and pinned by the tests; sending a form body here is not a fix.
        try:
            async with self._client() as client:
                response = await self._send(
                    client,
                    'POST',
                    _TOKEN_URL,
                    json={
                        'client_id': _CLIENT_ID,
                        'grant_type': 'refresh_token',
                        'refresh_token': current.refresh_token.get_secret_value(),
                    },
                    timeout=_REQUEST_TIMEOUT,
                )
        except OpenAICodexOAuthError:
            raise OpenAICodexRefreshError('Unable to reach the OpenAI Codex authentication service.') from None
        if not response.is_success:
            error_code = self._error_code(response)
            if response.status_code == 401 or error_code in {
                'refresh_token_expired',
                'refresh_token_reused',
                'refresh_token_invalidated',
            }:
                raise OpenAICodexLoginRequiredError(
                    'OpenAI Codex credentials can no longer be refreshed. Run `clai auth login openai-codex` again.'
                )
            raise OpenAICodexRefreshError('OpenAI Codex credential refresh failed.')

        try:
            refreshed = self._validate_response(
                response, _RefreshResponse, 'OpenAI Codex returned an invalid refresh response.'
            )
            access_token = refreshed.access_token or current.access_token
            refresh_token = refreshed.refresh_token or current.refresh_token
            id_token = refreshed.id_token or current.id_token
            credentials = self._credentials_from_tokens(
                access_token=access_token,
                refresh_token=refresh_token,
                id_token=id_token,
                fallback_account_id=current.account_id,
                fallback_fedramp=current.account_is_fedramp,
            )
        except _TokenAccountMismatchError:
            raise OpenAICodexAccountMismatchError(
                'OpenAI Codex credentials changed to a different ChatGPT account. '
                'Sign in again to select the account explicitly.'
            ) from None
        except OpenAICodexOAuthError as error:
            raise OpenAICodexRefreshError(str(error)) from None
        if credentials.expires_at <= datetime.now(timezone.utc):
            raise OpenAICodexRefreshError('OpenAI Codex returned a refresh response without a usable access token.')
        return credentials

    async def revoke(self, credentials: OpenAICodexCredentials) -> None:
        # JSON rather than the form encoding RFC 7009 §2.1 mandates, for the same reason as `refresh`.
        async with self._client() as client:
            response = await self._send(
                client,
                'POST',
                _REVOKE_URL,
                json={
                    'token': credentials.refresh_token.get_secret_value(),
                    'token_type_hint': 'refresh_token',
                    'client_id': _CLIENT_ID,
                },
                timeout=_REVOCATION_TIMEOUT,
            )
        if not response.is_success:
            raise OpenAICodexOAuthError('OpenAI Codex token revocation failed.')

    async def _bind_callback_listener(self) -> tuple[MultiListener[SocketStream], int]:
        last_error: OSError | None = None
        for port in _CALLBACK_PORTS:
            try:
                listener = await anyio.create_tcp_listener(
                    local_host='127.0.0.1',
                    local_port=port,
                    family=socket.AF_INET,
                )
            except OSError as error:
                last_error = error
            else:
                return listener, port
        raise OpenAICodexOAuthError(
            'Unable to bind the OpenAI Codex login callback on localhost ports 1455 or 1457.'
        ) from last_error

    async def _handle_callback(
        self,
        connection: SocketStream,
        *,
        client: httpx.AsyncClient,
        redirect_uri: str,
        verifier: str,
        expected_state: str,
    ) -> OpenAICodexCredentials | OpenAICodexAuthError | None:
        try:
            target = await _read_request_target(connection)
        except OpenAICodexOAuthError:
            await _send_callback_response(connection, 400, 'Invalid authorization callback.')
            return None

        parsed = urlsplit(target)
        if parsed.path != _CALLBACK_PATH:
            await _send_callback_response(connection, 404, 'Not found.')
            return None

        query = parse_qs(parsed.query, keep_blank_values=True)
        state = _single_query_value(query, 'state')
        if state is None or not secrets.compare_digest(state, expected_state):
            await _send_callback_response(connection, 400, 'Authorization state mismatch.')
            return None
        if _single_query_value(query, 'error') is not None:
            error = OpenAICodexOAuthError('OpenAI Codex authorization was not completed.')
            await _send_callback_response(connection, 400, 'Authorization was not completed.')
            return error
        code = _single_query_value(query, 'code')
        if not code:
            await _send_callback_response(connection, 400, 'Missing authorization code.')
            return None

        try:
            credentials = await self._exchange_code(
                client,
                code=SecretStr(code),
                redirect_uri=redirect_uri,
                verifier=verifier,
            )
        except OpenAICodexAuthError as error:
            await _send_callback_response(connection, 500, 'Authorization could not be completed.')
            return error
        await _send_callback_response(connection, 200, 'Authorization complete. You can return to the terminal.')
        return credentials

    async def _exchange_code(
        self,
        client: httpx.AsyncClient,
        *,
        code: SecretStr,
        redirect_uri: str,
        verifier: str,
    ) -> OpenAICodexCredentials:
        response = await self._send(
            client,
            'POST',
            _TOKEN_URL,
            data={
                'grant_type': 'authorization_code',
                'code': code.get_secret_value(),
                'redirect_uri': redirect_uri,
                'client_id': _CLIENT_ID,
                'code_verifier': verifier,
            },
        )
        if not response.is_success:
            raise OpenAICodexOAuthError('OpenAI Codex authorization-code exchange failed.')
        tokens = self._validate_response(response, _TokenResponse, 'OpenAI Codex returned an invalid token response.')
        return self._credentials_from_tokens(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            id_token=tokens.id_token,
        )

    def _credentials_from_tokens(
        self,
        *,
        access_token: SecretStr,
        refresh_token: SecretStr,
        id_token: SecretStr,
        fallback_account_id: SecretStr | None = None,
        fallback_fedramp: bool = False,
    ) -> OpenAICodexCredentials:
        access_claims = _decode_jwt(access_token, _JwtClaims)
        id_claims = _decode_jwt(id_token, _JwtClaims)
        account_claims = [claims for claims in (access_claims.auth, id_claims.auth) if claims is not None]
        account_ids = {claims.chatgpt_account_id for claims in account_claims if claims.chatgpt_account_id}
        fallback_id = fallback_account_id.get_secret_value() if fallback_account_id is not None else None
        if len(account_ids) > 1 or (fallback_id is not None and account_ids and fallback_id not in account_ids):
            raise _TokenAccountMismatchError('OpenAI Codex tokens identify different ChatGPT accounts or workspaces.')
        account_id = next(iter(account_ids), fallback_id)
        if not account_id:
            raise OpenAICodexOAuthError('OpenAI Codex credentials do not identify a ChatGPT account or workspace.')

        fedramp_values = {
            claims.chatgpt_account_is_fedramp
            for claims in account_claims
            if claims.chatgpt_account_id is not None and claims.chatgpt_account_is_fedramp is not None
        }
        if len(fedramp_values) > 1 or (
            fallback_id is not None and fedramp_values and fallback_fedramp not in fedramp_values
        ):
            raise _TokenAccountMismatchError('OpenAI Codex tokens disagree about the ChatGPT account environment.')
        account_is_fedramp = next(iter(fedramp_values), fallback_fedramp)
        if access_claims.exp is None:
            raise OpenAICodexOAuthError('OpenAI Codex access token does not include an expiration time.')
        try:
            expires_at = datetime.fromtimestamp(access_claims.exp, timezone.utc)
        except (OverflowError, OSError, ValueError) as error:
            raise OpenAICodexOAuthError('OpenAI Codex access token has an invalid expiration time.') from error

        return OpenAICodexCredentials(
            access_token=access_token,
            refresh_token=refresh_token,
            id_token=id_token,
            expires_at=expires_at,
            account_id=SecretStr(account_id),
            account_is_fedramp=account_is_fedramp,
            revision=secrets.token_urlsafe(18),
        )

    def _validate_response(self, response: httpx.Response, model: type[_ModelT], message: str) -> _ModelT:
        try:
            validated = model.model_validate_json(response.content)
        except ValidationError:
            pass
        else:
            return validated
        # Raised outside the `except` block on purpose: `from None` only clears `__cause__`,
        # while `__context__` would still hold the `ValidationError` whose payload carries the
        # raw token. Raising here leaves `__context__` empty, which the secret-leak tests assert.
        raise OpenAICodexOAuthError(message) from None

    def _error_code(self, response: httpx.Response) -> str | None:
        """Return the lower-cased error code, as the official client's own classifier does.

        Every caller compares the result against literals, so normalizing once here is what keeps a
        cased variant from downgrading a terminal `OpenAICodexLoginRequiredError` into a retry that can
        never succeed, or a `slow_down` into an unrecognized failure.
        """
        try:
            code = _ErrorResponse.model_validate_json(response.content).error_code()
        except ValidationError:
            return None
        return code.lower() if code is not None else None

    async def _send(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        *,
        json: dict[str, str] | None = None,
        data: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        try:
            if timeout is None:
                return await client.request(method, url, json=json, data=data, follow_redirects=False)
            return await client.request(method, url, json=json, data=data, timeout=timeout, follow_redirects=False)
        except httpx.HTTPError:
            pass
        raise OpenAICodexOAuthError('Unable to reach the OpenAI Codex authentication service.') from None

    @asynccontextmanager
    async def _client(self) -> AsyncGenerator[httpx.AsyncClient]:
        if self._http_client is not None:
            yield self._http_client
        else:
            async with create_async_http_client(timeout=_REQUEST_TIMEOUT, connect=_CONNECT_TIMEOUT) as client:
                yield client


def _decode_jwt(token: SecretStr, model: type[_ModelT]) -> _ModelT:
    parts = token.get_secret_value().split('.')
    if len(parts) != 3 or any(not part for part in parts):
        raise OpenAICodexOAuthError('OpenAI Codex returned a token with an invalid JWT format.')
    payload = parts[1]
    try:
        decoded = base64.urlsafe_b64decode(payload + '=' * (-len(payload) % 4))
        claims = model.model_validate_json(decoded)
    except (ValueError, ValidationError):
        pass
    else:
        return claims
    raise OpenAICodexOAuthError('OpenAI Codex returned a token with invalid JWT claims.') from None


def _base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode()


async def _read_request_target(connection: SocketStream) -> str:
    request = bytearray()
    while b'\r\n\r\n' not in request:
        if len(request) >= 16_384:
            raise OpenAICodexOAuthError('The OpenAI Codex authorization callback request was too large.')
        try:
            request.extend(await connection.receive(4096))
        except anyio.EndOfStream as error:
            raise OpenAICodexOAuthError('The OpenAI Codex authorization callback ended unexpectedly.') from error
    try:
        first_line = bytes(request).split(b'\r\n', 1)[0].decode('ascii')
        method, target, protocol = first_line.split(' ', 2)
    except (UnicodeDecodeError, ValueError) as error:
        raise OpenAICodexOAuthError('The OpenAI Codex authorization callback was malformed.') from error
    if method != 'GET' or not protocol.startswith('HTTP/1.'):
        raise OpenAICodexOAuthError('The OpenAI Codex authorization callback used an unsupported request method.')
    return target


async def _send_callback_response(connection: SocketStream, status: int, message: str) -> None:
    reason = {200: 'OK', 400: 'Bad Request', 404: 'Not Found', 500: 'Internal Server Error'}[status]
    body = message.encode()
    response = (
        f'HTTP/1.1 {status} {reason}\r\n'
        'Content-Type: text/plain; charset=utf-8\r\n'
        f'Content-Length: {len(body)}\r\n'
        'Connection: close\r\n'
        '\r\n'
    ).encode() + body
    try:
        await connection.send(response)
    except (anyio.BrokenResourceError, anyio.ClosedResourceError):
        pass


def _single_query_value(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values and len(values) == 1 else None


@contextmanager
def _collapse_single_exception_group() -> Generator[None]:
    try:
        yield
    except BaseExceptionGroup as group:
        if len(group.exceptions) == 1:
            error = group.exceptions[0]
            error.__suppress_context__ = True
            raise error
        raise


async def _invoke_callback(callback: Callable[[_CallbackT], object | Awaitable[object]], value: _CallbackT) -> None:
    # `anyio.to_thread.run_sync` rather than `_utils.run_in_executor`, which is the usual way to call
    # blocking code here: only this one abandons the worker on cancellation, and the callback is
    # user-supplied, so a synchronous one that blocks would otherwise hold the login timeout open for
    # as long as it wants. `run_in_executor`'s `disable_threads()` support buys nothing in exchange —
    # interactive login runs in neither of the environments that set it, since a Temporal workflow
    # does not perform OAuth sign-in and emscripten cannot bind the loopback callback listener or
    # open a browser.
    result = await run_sync_in_worker(callback, value, abandon_on_cancel=True)
    if inspect.isawaitable(result):
        await result


def _validate_timeout(timeout: float) -> None:
    if not math.isfinite(timeout) or timeout <= 0:
        raise UserError('`timeout` must be finite and positive')


async def _sleep(delay: float) -> None:
    await anyio.sleep(delay)

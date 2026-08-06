from __future__ import annotations as _annotations

import base64
import hashlib
import json
import socket
import threading
import traceback
import urllib.request
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlsplit

import anyio
import httpx
import pytest
from anyio.lowlevel import checkpoint
from pydantic import SecretStr

from pydantic_ai._utils import BaseExceptionGroup
from pydantic_ai.auth._openai_codex_oauth import _collapse_single_exception_group  # pyright: ignore[reportPrivateUsage]
from pydantic_ai.auth.openai_codex import (
    OpenAICodexAccountMismatchError,
    OpenAICodexAuth,
    OpenAICodexCredentials,
    OpenAICodexCredentialsError,
    OpenAICodexDeviceCode,
    OpenAICodexLoginRequiredError,
    OpenAICodexOAuthError,
    OpenAICodexRefreshError,
)
from pydantic_ai.exceptions import UserError

from ._inline_snapshot import snapshot
from .conftest import try_import

# `_openai_codex_store` imports without the `openai-codex` extra; only its locking needs `filelock`.
with try_import() as file_store_imports_successful:
    import filelock  # pyright: ignore[reportUnusedImport]  # noqa: F401

pytestmark = [pytest.mark.anyio, pytest.mark.xdist_group(name='openai_codex_auth')]

_ACCOUNT_ID = 'account-sensitive-value'
_ACCESS_TOKEN = 'access-sensitive-value'
_REFRESH_TOKEN = 'refresh-sensitive-value'
_ID_TOKEN = 'id-sensitive-value'


def _jwt(payload: dict[str, object]) -> str:
    def encode(value: dict[str, object]) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b'=').decode()

    return f'{encode({"alg": "none"})}.{encode(payload)}.signature'


def _tokens(*, account_id: str = _ACCOUNT_ID, expires_in: int = 3600) -> tuple[str, str]:
    access_token = _jwt({'exp': int((datetime.now(timezone.utc) + timedelta(seconds=expires_in)).timestamp())})
    id_token = _jwt({'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})
    return access_token, id_token


def _credentials(*, expires_in: int = 3600, revision: str = 'revision-1') -> OpenAICodexCredentials:
    access_token, id_token = _tokens(expires_in=expires_in)
    return OpenAICodexCredentials(
        access_token=SecretStr(access_token),
        refresh_token=SecretStr(_REFRESH_TOKEN),
        id_token=SecretStr(id_token),
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=expires_in),
        account_id=SecretStr(_ACCOUNT_ID),
        revision=revision,
    )


class MemoryStore:
    def __init__(self, credentials: OpenAICodexCredentials | None = None) -> None:
        self.credentials = credentials
        self._lock = anyio.Lock()

    @asynccontextmanager
    async def exclusive(self) -> AsyncGenerator[None]:
        async with self._lock:
            yield

    async def load(self) -> OpenAICodexCredentials | None:
        return self.credentials

    async def save(self, credentials: OpenAICodexCredentials, *, expected_revision: str | None) -> bool:
        current_revision = self.credentials.revision if self.credentials is not None else None
        if current_revision != expected_revision:
            return False
        self.credentials = credentials
        return True

    async def delete(self, *, expected_revision: str | None) -> bool:
        if self.credentials is None or self.credentials.revision != expected_revision:
            return False
        self.credentials = None
        return True


class CheckpointExitStore(MemoryStore):
    def __init__(self, credentials: OpenAICodexCredentials | None = None) -> None:
        super().__init__(credentials)
        self.exit_started = anyio.Event()
        self.exit_finished = anyio.Event()

    @asynccontextmanager
    async def exclusive(self) -> AsyncGenerator[None]:
        await self._lock.acquire()
        try:
            yield
        finally:
            self.exit_started.set()
            await checkpoint()
            self._lock.release()
            self.exit_finished.set()


def test_multiple_callback_task_errors_remain_grouped() -> None:
    group = BaseExceptionGroup('multiple errors', [ValueError('first'), RuntimeError('second')])
    with pytest.raises(BaseExceptionGroup) as exc_info:
        with _collapse_single_exception_group():
            raise group
    assert exc_info.value is group


def test_credentials_repr_and_serialization_are_secret_safe() -> None:
    credentials = _credentials()
    rendered = f'{credentials!r}\n{credentials}\n{credentials.model_dump_json()}'
    assert credentials.access_token.get_secret_value() not in rendered
    assert credentials.refresh_token.get_secret_value() not in rendered
    assert credentials.id_token.get_secret_value() not in rendered
    assert credentials.account_id.get_secret_value() not in rendered
    assert '**********' in rendered


def test_credentials_require_timezone_aware_expiry() -> None:
    with pytest.raises(ValueError, match='timezone-aware'):
        OpenAICodexCredentials(
            access_token=SecretStr('access'),
            refresh_token=SecretStr('refresh'),
            id_token=SecretStr('id'),
            expires_at=datetime.now(),
            account_id=SecretStr('account'),
            revision='revision',
        )


def test_auth_store_and_path_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(UserError, match='mutually exclusive'):
        OpenAICodexAuth(store=MemoryStore(), path=tmp_path / 'auth.json')


async def test_memory_store_compare_and_swap_mismatches() -> None:
    current = _credentials(revision='current')
    store = MemoryStore(current)

    assert not await store.save(_credentials(revision='replacement'), expected_revision='stale')
    assert not await store.delete(expected_revision='stale')
    assert store.credentials is current


async def test_store_context_cannot_suppress_transaction_error() -> None:
    class SuppressingStore(MemoryStore):
        @asynccontextmanager
        async def exclusive(self) -> AsyncGenerator[None]:
            try:
                yield
            except RuntimeError:
                pass

        async def load(self) -> OpenAICodexCredentials | None:
            raise RuntimeError('store failure')

    with pytest.raises(RuntimeError, match='store failure'):
        await OpenAICodexAuth(store=SuppressingStore()).logout()


async def test_login_rejects_failed_conditional_save() -> None:
    class RejectingSaveStore(MemoryStore):
        async def save(self, credentials: OpenAICodexCredentials, *, expected_revision: str | None) -> bool:
            return False

    auth = OpenAICodexAuth(store=RejectingSaveStore())
    with pytest.raises(OpenAICodexCredentialsError, match='changed while login'):
        await auth._replace_after_login(_credentials())  # pyright: ignore[reportPrivateUsage]


@pytest.mark.skipif(
    not file_store_imports_successful(), reason='install the `openai-codex` extras to run default file store tests'
)
@pytest.mark.parametrize('method', ['browser', 'device'])
async def test_login_checks_the_store_before_spending_the_sign_in(method: str, tmp_path: Path) -> None:
    """An unusable store fails before the OAuth round trip rather than after it mints a token.

    Persistence is the only step that can fail for store reasons — a missing `openai-codex` extra, an
    unwritable path — so without this check a completed sign-in ends with the user's interaction
    spent and the freshly minted token discarded.
    """
    not_a_directory = tmp_path / 'occupied'
    not_a_directory.write_text('')

    def handle(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError('no upstream request may be sent before the store is known usable')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = OpenAICodexAuth(path=not_a_directory / 'auth.json', http_client=client)
        with pytest.raises(OpenAICodexCredentialsError, match='Unable to lock'):
            if method == 'browser':
                await auth.login_browser(lambda url: None, timeout=10)
            else:
                await auth.login_device(lambda code: None, timeout=10)


@pytest.mark.skipif(
    not file_store_imports_successful(), reason='install the `openai-codex` extras to run default file store tests'
)
async def test_login_rejects_a_malformed_store_before_spending_the_sign_in(tmp_path: Path) -> None:
    """A malformed record only shows up on read, so the preflight has to read and not merely lock.

    Locking alone leaves such a store looking usable, which makes the save after a completed
    sign-in the first thing to fail — discarding exactly the token the preflight exists to protect.
    """
    path = tmp_path / 'auth.json'
    path.write_text(json.dumps({'version': 999, 'providers': {}}), encoding='utf-8')

    def handle(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError('no upstream request may be sent before the store is known usable')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = OpenAICodexAuth(path=path, http_client=client)
        with pytest.raises(OpenAICodexCredentialsError, match=r'malformed|unsupported'):
            await auth.login_browser(lambda url: None, timeout=10)


@pytest.mark.skipif(
    not file_store_imports_successful(), reason='install the `openai-codex` extras to run default file store tests'
)
async def test_default_file_store_is_lazy_and_status_reads_selected_path(tmp_path: Path) -> None:
    path = tmp_path / 'credentials' / 'auth.json'
    auth = OpenAICodexAuth(path=path)
    assert not path.parent.exists()
    assert (await auth.status()).authenticated is False
    assert not path.parent.exists()


async def test_valid_and_already_replaced_credentials_do_not_refresh() -> None:
    current = _credentials(revision='current')
    auth = OpenAICodexAuth(store=MemoryStore(current))

    assert await auth.get_credentials() is current
    assert await auth.get_credentials(force_refresh=True, rejected_revision='older') is current
    with pytest.raises(UserError, match='requires'):
        await auth.get_credentials(rejected_revision='current')

    status = await auth.status()
    assert status.authenticated is True
    assert status.needs_refresh is False
    assert status.expires_at == current.expires_at


async def test_missing_credentials_has_login_guidance() -> None:
    auth = OpenAICodexAuth(store=MemoryStore())
    with pytest.raises(OpenAICodexLoginRequiredError, match=r'clai auth login openai-codex'):
        await auth.get_credentials()
    assert (await auth.status()).model_dump() == snapshot(
        {
            'authenticated': False,
            'expires_at': None,
            'needs_refresh': False,
            'account_is_fedramp': False,
        }
    )


async def test_credentials_that_become_valid_while_locking_are_reused() -> None:
    expired = _credentials(expires_in=-60, revision='same-revision')
    valid = _credentials(expires_in=3600, revision='same-revision')

    class BecomesValidStore(MemoryStore):
        def __init__(self) -> None:
            super().__init__(expired)
            self.loads = 0

        async def load(self) -> OpenAICodexCredentials | None:
            self.loads += 1
            if self.loads == 2:
                self.credentials = valid
            return self.credentials

    assert await OpenAICodexAuth(store=BecomesValidStore()).get_credentials() is valid


async def test_concurrent_refresh_is_single_flight_and_preserves_omitted_tokens() -> None:
    current = _credentials(expires_in=-60)
    refreshed_access_token, _ = _tokens(expires_in=3600)
    refresh_requests = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_requests
        assert request.url.path == '/oauth/token'
        assert request.headers['content-type'] == 'application/json'
        assert request.extensions['timeout'] == {'connect': 30, 'read': 30, 'write': 30, 'pool': 30}
        assert json.loads(request.content) == {
            'client_id': 'app_EMoamEEZ73f0CkXaXp7hrann',
            'grant_type': 'refresh_token',
            'refresh_token': _REFRESH_TOKEN,
        }
        refresh_requests += 1
        return httpx.Response(200, json={'access_token': refreshed_access_token})

    store = MemoryStore(current)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = OpenAICodexAuth(store=store, http_client=client)
        results: list[OpenAICodexCredentials] = []

        async def resolve() -> None:
            results.append(await auth.get_credentials())

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(resolve)
            task_group.start_soon(resolve)

    assert refresh_requests == 1
    assert len(results) == 2
    assert results[0].revision == results[1].revision
    assert results[0].refresh_token.get_secret_value() == _REFRESH_TOKEN
    assert results[0].id_token == current.id_token
    assert results[0].is_valid()


async def test_rejected_revision_is_refreshed_only_once_concurrently() -> None:
    current = _credentials(revision='rejected-revision')
    refreshed_access_token, _ = _tokens(expires_in=3600)
    refresh_requests = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal refresh_requests
        refresh_requests += 1
        await anyio.sleep(0)
        return httpx.Response(200, json={'access_token': refreshed_access_token})

    store = MemoryStore(current)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = OpenAICodexAuth(store=store, http_client=client)
        results: list[OpenAICodexCredentials] = []

        async def recover() -> None:
            results.append(await auth.get_credentials(force_refresh=True, rejected_revision='rejected-revision'))

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(recover)
            task_group.start_soon(recover)

    assert refresh_requests == 1
    assert len(results) == 2
    assert results[0].revision == results[1].revision
    assert results[0].revision != 'rejected-revision'


async def test_cancellation_during_refresh_still_persists_rotated_token() -> None:
    current = _credentials(expires_in=-60)
    refreshed_access_token, _ = _tokens(expires_in=3600)
    request_started = anyio.Event()
    allow_response = anyio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await allow_response.wait()
        return httpx.Response(200, json={'access_token': refreshed_access_token})

    store = MemoryStore(current)
    scopes: list[anyio.CancelScope] = []
    completed = False
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = OpenAICodexAuth(store=store, http_client=client)

        async def rotate() -> None:
            nonlocal completed
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                await auth.refresh()
                # Cancellation must prevent the call from returning
                completed = True  # pragma: no cover

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(rotate)
            await request_started.wait()
            scopes[0].cancel()
            allow_response.set()

    assert not completed
    assert store.credentials is not None
    assert store.credentials.revision != current.revision
    assert store.credentials.access_token.get_secret_value() == refreshed_access_token


class BlockingSaveStore(CheckpointExitStore):
    def __init__(self, credentials: OpenAICodexCredentials) -> None:
        super().__init__(credentials)
        self.save_started = anyio.Event()
        self.allow_save = anyio.Event()

    async def save(self, credentials: OpenAICodexCredentials, *, expected_revision: str | None) -> bool:
        self.save_started.set()
        await self.allow_save.wait()
        return await super().save(credentials, expected_revision=expected_revision)


async def test_cancellation_after_refresh_response_still_completes_save() -> None:
    current = _credentials(expires_in=-60)
    refreshed_access_token, _ = _tokens(expires_in=3600)
    store = BlockingSaveStore(current)
    scopes: list[anyio.CancelScope] = []
    completed = False

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'access_token': refreshed_access_token})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = OpenAICodexAuth(store=store, http_client=client)

        async def rotate() -> None:
            nonlocal completed
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                await auth.refresh()
                # Cancellation must prevent the call from returning
                completed = True  # pragma: no cover

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(rotate)
            await store.save_started.wait()
            scopes[0].cancel()
            store.allow_save.set()

    assert not completed
    assert store.exit_finished.is_set()
    assert store.credentials is not None
    assert store.credentials.revision != current.revision
    assert store.credentials.access_token.get_secret_value() == refreshed_access_token
    with anyio.fail_after(1):
        async with store.exclusive():
            pass


async def test_refresh_and_save_has_bounded_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    current = _credentials(expires_in=-60)
    monkeypatch.setattr('pydantic_ai.auth.openai_codex._REFRESH_AND_SAVE_TIMEOUT', 0.01)

    async def handle(request: httpx.Request) -> httpx.Response:
        await anyio.sleep_forever()
        raise AssertionError('unreachable')  # pragma: no cover

    store = MemoryStore(current)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexRefreshError, match='timed out'):
            await OpenAICodexAuth(store=store, http_client=client).refresh()

    assert store.credentials is current


async def test_refresh_rejects_new_access_token_conflicting_with_retained_id_token() -> None:
    current = _credentials(expires_in=-60)
    conflicting_access = _jwt(
        {
            'exp': int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp()),
            'https://api.openai.com/auth': {'chatgpt_account_id': 'different-account'},
        }
    )

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'access_token': conflicting_access})

    store = MemoryStore(current)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexAccountMismatchError, match='different ChatGPT account'):
            await OpenAICodexAuth(store=store, http_client=client).refresh()

    assert store.credentials is current


async def test_refresh_rejects_account_switch_without_replacing_credentials() -> None:
    current = _credentials(expires_in=-60)
    access_token, id_token = _tokens(account_id='different-account')

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'access_token': access_token, 'id_token': id_token})

    store = MemoryStore(current)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = OpenAICodexAuth(store=store, http_client=client)
        with pytest.raises(OpenAICodexAccountMismatchError, match='different ChatGPT account'):
            await auth.refresh()

    assert store.credentials is current


@pytest.mark.parametrize(
    ('status_code', 'body', 'error_type', 'message'),
    [
        (401, {}, OpenAICodexLoginRequiredError, 'login openai-codex'),
        (400, {'error': 'refresh_token_reused'}, OpenAICodexLoginRequiredError, 'login openai-codex'),
        (500, {}, OpenAICodexRefreshError, 'refresh failed'),
        (200, [], OpenAICodexRefreshError, 'invalid refresh response'),
        (200, {'access_token': 'not-a-jwt'}, OpenAICodexRefreshError, 'invalid JWT'),
    ],
)
async def test_refresh_maps_protocol_failures_to_typed_errors(
    status_code: int,
    body: object,
    error_type: type[Exception],
    message: str,
) -> None:
    current = _credentials(expires_in=-60)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(error_type, match=message):
            await OpenAICodexAuth(store=MemoryStore(current), http_client=client).refresh()


async def test_refresh_uses_existing_account_when_new_tokens_omit_account_claims() -> None:
    current = OpenAICodexCredentials(
        access_token=SecretStr(_jwt({'exp': int((datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp())})),
        refresh_token=SecretStr(_REFRESH_TOKEN),
        id_token=SecretStr(_jwt({})),
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
        account_id=SecretStr(_ACCOUNT_ID),
        account_is_fedramp=True,
        revision='revision-1',
    )
    refreshed_access = _jwt({'exp': int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())})

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={'access_token': refreshed_access})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        refreshed = await OpenAICodexAuth(store=MemoryStore(current), http_client=client).refresh()

    assert refreshed.account_id.get_secret_value() == _ACCOUNT_ID
    assert refreshed.account_is_fedramp is True


async def test_refresh_rejects_expired_or_incomplete_success_response() -> None:
    current = _credentials(expires_in=-60)

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexRefreshError, match='without a usable access token'):
            await OpenAICodexAuth(store=MemoryStore(current), http_client=client).refresh()


@pytest.mark.parametrize('operation', ['refresh', 'revoke'])
async def test_oauth_refresh_and_revoke_never_follow_redirects(operation: str) -> None:
    current = _credentials(expires_in=-60)
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={'location': 'https://example.com/collect'})

    store = MemoryStore(current)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), follow_redirects=True) as client:
        auth = OpenAICodexAuth(store=store, http_client=client)
        if operation == 'refresh':
            with pytest.raises(OpenAICodexRefreshError, match='refresh failed'):
                await auth.refresh()
        else:
            result = await auth.logout()
            assert result.revocation_error == 'Upstream OpenAI Codex token revocation failed.'

    assert len(requests) == 1


async def test_refresh_network_error_does_not_retain_refresh_token() -> None:
    current = _credentials(expires_in=-60)
    sentinel = current.refresh_token.get_secret_value()

    def handle(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f'failed with {sentinel}', request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexRefreshError) as exc_info:
            await OpenAICodexAuth(store=MemoryStore(current), http_client=client).refresh()

    assert sentinel not in ''.join(traceback.format_exception(exc_info.value))


_ISSUER_EXPIRES_AT = datetime(2099, 3, 4, 5, 6, 7, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ('start_body', 'expected_interval', 'expected_expires_at'),
    [
        ({'user_code': 'USER-CODE', 'interval': '1'}, 1.0, None),
        # RFC 8628 §3.2 makes `interval` OPTIONAL and requires a 5-second default; the reference
        # Codex client marks its own field `#[serde(default)]`, so login must not hard-fail on it.
        ({'user_code': 'USER-CODE'}, 5.0, None),
        # The reference client accepts either spelling of the user code
        # (`#[serde(alias = "user_code", alias = "usercode")]`), so rejecting one breaks device
        # login outright rather than degrading it.
        ({'usercode': 'USER-CODE', 'interval': '1'}, 1.0, None),
        # The live issuer states the deadline as an absolute `expires_at` rather than RFC 8628's
        # `expires_in` duration, and it is the value the user is shown, so it wins over the fallback.
        (
            {'user_code': 'USER-CODE', 'interval': '1', 'expires_at': _ISSUER_EXPIRES_AT.isoformat()},
            1.0,
            _ISSUER_EXPIRES_AT,
        ),
    ],
    ids=['interval-sent', 'interval-omitted', 'usercode-alias', 'expires-at-sent'],
)
async def test_device_login_pending_then_success(
    monkeypatch: pytest.MonkeyPatch,
    start_body: dict[str, str],
    expected_interval: float,
    expected_expires_at: datetime | None,
) -> None:
    access_token, id_token = _tokens()
    verifier = 'device-verifier'
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    poll_count = 0
    shown: list[OpenAICodexDeviceCode] = []
    sleep = AsyncMock()
    monkeypatch.setattr('pydantic_ai.auth._openai_codex_oauth._sleep', sleep)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.url.path.endswith('/deviceauth/usercode'):
            return httpx.Response(
                200,
                json={'device_auth_id': 'device-id', **start_body},
            )
        if request.url.path.endswith('/deviceauth/token'):
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(403)
            return httpx.Response(
                200,
                json={
                    'authorization_code': 'authorization-code',
                    'code_challenge': challenge,
                    'code_verifier': verifier,
                },
            )
        assert request.url.path == '/oauth/token'
        form = parse_qs(request.content.decode())
        assert form['redirect_uri'] == ['https://auth.openai.com/deviceauth/callback']
        assert form['code'] == ['authorization-code']
        assert form['code_verifier'] == [verifier]
        return httpx.Response(
            200,
            json={'access_token': access_token, 'refresh_token': _REFRESH_TOKEN, 'id_token': id_token},
        )

    store = MemoryStore()
    started_at = datetime.now(timezone.utc)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        credentials = await OpenAICodexAuth(store=store, http_client=client).login_device(shown.append, timeout=1800)

    assert credentials is store.credentials
    assert shown[0].verification_url == 'https://auth.openai.com/codex/device'
    assert shown[0].user_code.get_secret_value() == 'USER-CODE'
    if expected_expires_at is not None:
        assert shown[0].expires_at == expected_expires_at
    else:
        assert timedelta(seconds=895) < shown[0].expires_at - started_at < timedelta(seconds=901)
    assert 'USER-CODE' not in repr(shown[0])
    sleep.assert_awaited_once_with(expected_interval)


@pytest.mark.parametrize(
    ('status_code', 'body', 'message'),
    [
        (404, {}, 'not enabled'),
        (500, {}, 'Unable to start'),
        (200, {'device_auth_id': 'id'}, 'invalid device code'),
        (200, {'device_auth_id': 'id', 'user_code': 'code', 'interval': 0}, 'invalid device code'),
    ],
)
async def test_device_login_rejects_invalid_start_responses(status_code: int, body: object, message: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexOAuthError, match=message):
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(lambda code: None)


@pytest.mark.parametrize(
    ('status_code', 'body', 'message'),
    [
        (400, {'error': 'access_denied'}, 'declined'),
        (403, {'error': 'authorization_declined'}, 'declined'),
        (400, {'error': {'code': 'device_code_expired'}}, 'expired'),
        (404, {'error': 'expired_token'}, 'expired'),
        (400, {'error': 'unexpected_error'}, 'failed'),
        (400, b'not-json', 'failed'),
        (200, {}, 'invalid device authorization result'),
        (
            200,
            {'authorization_code': 'code', 'code_challenge': 'wrong', 'code_verifier': 'verifier'},
            'inconsistent PKCE',
        ),
    ],
)
async def test_device_login_rejects_terminal_poll_responses(status_code: int, body: object, message: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/deviceauth/usercode'):
            return httpx.Response(
                200,
                json={'device_auth_id': 'device-id', 'user_code': 'USER-CODE', 'interval': 1},
            )
        if isinstance(body, bytes):
            return httpx.Response(status_code, content=body)
        return httpx.Response(status_code, json=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexOAuthError, match=message):
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(lambda code: None)


async def test_device_login_handles_slow_down_and_authorization_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_token, id_token = _tokens()
    verifier = 'device-verifier'
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    poll_count = 0
    sleep = AsyncMock()
    monkeypatch.setattr('pydantic_ai.auth._openai_codex_oauth._sleep', sleep)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.url.path.endswith('/deviceauth/usercode'):
            return httpx.Response(
                200,
                json={'device_auth_id': 'device-id', 'user_code': 'USER-CODE', 'interval': 1},
            )
        if request.url.path.endswith('/deviceauth/token'):
            poll_count += 1
            if poll_count == 1:
                return httpx.Response(400, json={'error': 'slow_down'})
            if poll_count == 2:
                return httpx.Response(400, json={'error': {'code': 'authorization_pending'}})
            return httpx.Response(
                200,
                json={
                    'authorization_code': 'authorization-code',
                    'code_challenge': challenge,
                    'code_verifier': verifier,
                },
            )
        return httpx.Response(
            200,
            json={'access_token': access_token, 'refresh_token': _REFRESH_TOKEN, 'id_token': id_token},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(lambda code: None)

    assert [call.args for call in sleep.await_args_list] == [(6.0,), (6.0,)]


async def test_device_login_backs_off_and_retries_after_a_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RFC 8628 §3.5: a connection timeout must reduce the polling frequency, not end the flow.

    Polling runs for minutes, so ending the sign-in on one blip would strand a login the user can
    still complete. The enclosing timeout still bounds a backend that never comes back.
    """
    access_token, id_token = _tokens()
    verifier = 'device-verifier'
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b'=').decode()
    poll_count = 0
    sleep = AsyncMock()
    monkeypatch.setattr('pydantic_ai.auth._openai_codex_oauth._sleep', sleep)

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal poll_count
        if request.url.path.endswith('/deviceauth/usercode'):
            return httpx.Response(
                200,
                json={'device_auth_id': 'device-id', 'user_code': 'USER-CODE', 'interval': 1},
            )
        if request.url.path.endswith('/deviceauth/token'):
            poll_count += 1
            if poll_count == 1:
                raise httpx.ReadTimeout('simulated poll timeout')
            return httpx.Response(
                200,
                json={
                    'authorization_code': 'authorization-code',
                    'code_challenge': challenge,
                    'code_verifier': verifier,
                },
            )
        return httpx.Response(
            200,
            json={'access_token': access_token, 'refresh_token': _REFRESH_TOKEN, 'id_token': id_token},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        credentials = await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(lambda code: None)

    assert credentials.access_token.get_secret_value() == access_token
    assert poll_count == 2
    # Doubled from the issuer's one-second interval rather than retried at the same rate.
    assert [call.args for call in sleep.await_args_list] == [(2.0,)]


async def test_browser_login_starts_listener_before_presenting_url() -> None:
    access_token, id_token = _tokens()
    authorization_urls: list[str] = []

    def handle_auth(request: httpx.Request) -> httpx.Response:
        assert request.url.path == '/oauth/token'
        return httpx.Response(
            200,
            json={'access_token': access_token, 'refresh_token': _REFRESH_TOKEN, 'id_token': id_token},
        )

    store = MemoryStore()
    credentials: OpenAICodexCredentials | None = None
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as auth_client:
        auth = OpenAICodexAuth(store=store, http_client=auth_client)
        async with anyio.create_task_group() as task_group:

            async def send_callback(callback_url: str) -> None:
                async with httpx.AsyncClient(trust_env=False) as callback_client:
                    parsed_callback = urlsplit(callback_url)
                    not_found = await callback_client.get(
                        f'{parsed_callback.scheme}://{parsed_callback.netloc}/unexpected'
                    )
                    assert not_found.status_code == 404
                    missing_code = await callback_client.get(
                        f'{parsed_callback.scheme}://{parsed_callback.netloc}{parsed_callback.path}'
                        f'?state={parse_qs(parsed_callback.query)["state"][0]}'
                    )
                    assert missing_code.status_code == 400
                    wrong_state = await callback_client.get(callback_url.replace('state=', 'state=wrong-'))
                    assert wrong_state.status_code == 400

                    async with await anyio.connect_tcp('127.0.0.1', parsed_callback.port or 80) as stream:
                        await stream.send(b'not an HTTP request\r\n\r\n')
                        unsupported_response = await stream.receive()
                    assert unsupported_response.startswith(b'HTTP/1.1 400')

                    async with await anyio.connect_tcp('127.0.0.1', parsed_callback.port or 80) as stream:
                        await stream.send(b'\xff\r\n\r\n')
                        malformed_response = await stream.receive()
                    assert malformed_response.startswith(b'HTTP/1.1 400')

                    partial = await anyio.connect_tcp('127.0.0.1', parsed_callback.port or 80)
                    await partial.send(b'GET /auth/callback HTTP/1.1\r\n')
                    await partial.aclose()

                    async with await anyio.connect_tcp('127.0.0.1', parsed_callback.port or 80) as stream:
                        await stream.send(b'x' * 16_384)
                        oversized_response = await stream.receive()
                    assert oversized_response.startswith(b'HTTP/1.1 400')

                    response = await callback_client.get(callback_url)
                    assert response.status_code == 200

            async def open_url(authorization_url: str) -> None:
                authorization_urls.append(authorization_url)
                query = parse_qs(urlsplit(authorization_url).query)
                assert query['scope'] == [
                    'openid profile email offline_access api.connectors.read api.connectors.invoke'
                ]
                callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
                task_group.start_soon(send_callback, callback_url)

            credentials = await auth.login_browser(open_url, timeout=5)

    assert credentials is not None
    assert credentials is store.credentials
    assert len(authorization_urls) == 1


async def test_browser_oauth_error_callback_is_terminal() -> None:
    async def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        callback_url = f'{query["redirect_uri"][0]}?error=access_denied&state={query["state"][0]}'
        async with httpx.AsyncClient(trust_env=False) as callback_client:
            response = await callback_client.get(callback_url)
            assert response.status_code == 400

    with pytest.raises(OpenAICodexOAuthError, match='was not completed'):
        await OpenAICodexAuth(store=MemoryStore()).login_browser(open_url, timeout=5)


@pytest.mark.parametrize(
    ('token_status', 'token_body', 'message'),
    [
        (500, {}, 'exchange failed'),
        (200, {'access_token': 'incomplete'}, 'invalid token response'),
    ],
)
async def test_browser_login_reports_token_exchange_failures(
    token_status: int, token_body: object, message: str
) -> None:
    def handle_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(token_status, json=token_body)

    async def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
        async with httpx.AsyncClient(trust_env=False) as callback_client:
            response = await callback_client.get(callback_url)
            assert response.status_code == 500

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as auth_client:
        with pytest.raises(OpenAICodexOAuthError, match=message):
            await OpenAICodexAuth(store=MemoryStore(), http_client=auth_client).login_browser(open_url, timeout=5)


async def test_oauth_code_exchange_never_follows_redirects() -> None:
    requests: list[httpx.Request] = []

    def handle_auth(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={'location': 'https://example.com/collect'})

    async def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
        async with httpx.AsyncClient(trust_env=False) as callback_client:
            assert (await callback_client.get(callback_url)).status_code == 500

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth), follow_redirects=True) as client:
        with pytest.raises(OpenAICodexOAuthError, match='exchange failed'):
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_browser(open_url, timeout=5)

    assert len(requests) == 1


@pytest.mark.parametrize(
    ('access_payload', 'id_payload', 'message'),
    [
        ({'exp': int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())}, {}, 'do not identify'),
        ({}, {'https://api.openai.com/auth': {'chatgpt_account_id': _ACCOUNT_ID}}, 'expiration time'),
        (
            {'exp': 10**100},
            {'https://api.openai.com/auth': {'chatgpt_account_id': _ACCOUNT_ID}},
            'invalid expiration time',
        ),
    ],
)
async def test_browser_login_rejects_incomplete_token_claims(
    access_payload: dict[str, object], id_payload: dict[str, object], message: str
) -> None:
    def handle_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'access_token': _jwt(access_payload),
                'refresh_token': _REFRESH_TOKEN,
                'id_token': _jwt(id_payload),
            },
        )

    async def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
        async with httpx.AsyncClient(trust_env=False) as callback_client:
            assert (await callback_client.get(callback_url)).status_code == 500

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as client:
        with pytest.raises(OpenAICodexOAuthError, match=message):
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_browser(open_url, timeout=5)


@pytest.mark.parametrize('conflict', ['account', 'fedramp'])
async def test_browser_login_rejects_conflicting_token_account_claims(conflict: str) -> None:
    expiry = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
    access_account = 'different-account' if conflict == 'account' else _ACCOUNT_ID
    access_fedramp = conflict == 'fedramp'
    access_token = _jwt(
        {
            'exp': expiry,
            'https://api.openai.com/auth': {
                'chatgpt_account_id': access_account,
                'chatgpt_account_is_fedramp': access_fedramp,
            },
        }
    )
    id_token = _jwt(
        {
            'https://api.openai.com/auth': {
                'chatgpt_account_id': _ACCOUNT_ID,
                'chatgpt_account_is_fedramp': False,
            }
        }
    )

    def handle_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'access_token': access_token,
                'refresh_token': _REFRESH_TOKEN,
                'id_token': id_token,
            },
        )

    async def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
        async with httpx.AsyncClient(trust_env=False) as callback_client:
            assert (await callback_client.get(callback_url)).status_code == 500

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as client:
        with pytest.raises(OpenAICodexOAuthError, match=r'different ChatGPT accounts|disagree'):
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_browser(open_url, timeout=5)


async def test_browser_login_rejects_invalid_jwt_claims() -> None:
    invalid_claims_token = 'header.bm90LWpzb24.signature'

    def handle_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                'access_token': invalid_claims_token,
                'refresh_token': _REFRESH_TOKEN,
                'id_token': invalid_claims_token,
            },
        )

    async def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
        async with httpx.AsyncClient(trust_env=False) as callback_client:
            assert (await callback_client.get(callback_url)).status_code == 500

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as client:
        with pytest.raises(OpenAICodexOAuthError, match='invalid JWT claims'):
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_browser(open_url, timeout=5)


async def test_browser_login_uses_allowlisted_fallback_port() -> None:
    access_token, id_token = _tokens()

    def handle_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={'access_token': access_token, 'refresh_token': _REFRESH_TOKEN, 'id_token': id_token},
        )

    async def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        assert urlsplit(query['redirect_uri'][0]).port == 1457
        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
        async with httpx.AsyncClient(trust_env=False) as callback_client:
            response = await callback_client.get(callback_url)
            assert response.status_code == 200

    occupied = await anyio.create_tcp_listener(local_host='127.0.0.1', local_port=1455)
    async with occupied, httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as auth_client:
        await OpenAICodexAuth(store=MemoryStore(), http_client=auth_client).login_browser(open_url, timeout=5)


async def test_browser_login_fails_when_both_callback_ports_are_unavailable() -> None:
    first = await anyio.create_tcp_listener(local_host='127.0.0.1', local_port=1455)
    second = await anyio.create_tcp_listener(local_host='127.0.0.1', local_port=1457)
    async with first, second:
        with pytest.raises(OpenAICodexOAuthError, match='Unable to bind'):
            await OpenAICodexAuth(store=MemoryStore()).login_browser(lambda url: None, timeout=5)


async def test_browser_login_reports_presentation_failure() -> None:
    def fail(authorization_url: str) -> None:
        raise RuntimeError('presentation failed')

    with pytest.raises(OpenAICodexOAuthError, match='Unable to open or present'):
        await OpenAICodexAuth(store=MemoryStore()).login_browser(fail, timeout=5)


async def test_browser_login_runs_sync_presentation_callback_in_worker_thread() -> None:
    access_token, id_token = _tokens()
    main_thread = threading.get_ident()
    callback_threads: list[int] = []

    def handle_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={'access_token': access_token, 'refresh_token': _REFRESH_TOKEN, 'id_token': id_token},
        )

    def open_url(authorization_url: str) -> None:
        callback_threads.append(threading.get_ident())
        query = parse_qs(urlsplit(authorization_url).query)
        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
        with urllib.request.urlopen(callback_url, timeout=5) as response:
            assert response.status == 200

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as auth_client:
        credentials = await OpenAICodexAuth(store=MemoryStore(), http_client=auth_client).login_browser(
            open_url, timeout=5
        )

    assert credentials.account_id.get_secret_value() == _ACCOUNT_ID
    assert callback_threads and callback_threads[0] != main_thread


async def test_browser_login_ignores_a_replayed_callback() -> None:
    """A reloaded success page replays a one-use code; only the first callback may exchange it.

    Each connection is served by its own task, so without the claim both would exchange the same
    code and the duplicate's failure could reach the result stream ahead of the real credentials —
    turning a completed sign-in into an error. The replay is also answered 200 rather than 500,
    since from the user's side the authorization did succeed.
    """
    access_token, id_token = _tokens()
    exchanges = 0

    def handle_auth(request: httpx.Request) -> httpx.Response:
        nonlocal exchanges
        exchanges += 1
        return httpx.Response(
            200,
            json={'access_token': access_token, 'refresh_token': _REFRESH_TOKEN, 'id_token': id_token},
        )

    statuses: list[int] = []

    def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'

        def fetch() -> None:
            with urllib.request.urlopen(callback_url, timeout=5) as response:
                statuses.append(response.status)

        threads = [threading.Thread(target=fetch) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as auth_client:
        credentials = await OpenAICodexAuth(store=MemoryStore(), http_client=auth_client).login_browser(
            open_url, timeout=10
        )

    assert credentials.account_id.get_secret_value() == _ACCOUNT_ID
    assert statuses == [200, 200]
    assert exchanges == 1


async def test_browser_login_survives_a_malformed_callback_request() -> None:
    """A stray request is answered 400 rather than ending a login the user can still complete.

    `urlsplit` rejects an unmatched `[` in the netloc with `ValueError`, which is not the
    `OpenAICodexOAuthError` the handler expects; escaping it fails the listener's task group and
    takes the whole sign-in down with it. The real callback afterwards proves the flow survived.
    """
    access_token, id_token = _tokens()

    def handle_auth(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={'access_token': access_token, 'refresh_token': _REFRESH_TOKEN, 'id_token': id_token},
        )

    def open_url(authorization_url: str) -> None:
        query = parse_qs(urlsplit(authorization_url).query)
        redirect = urlsplit(query['redirect_uri'][0])
        assert redirect.hostname is not None and redirect.port is not None
        with socket.create_connection((redirect.hostname, redirect.port), timeout=5) as stray:
            stray.sendall(b'GET http://[/auth/callback HTTP/1.1\r\nHost: localhost\r\n\r\n')
            assert b'400' in stray.recv(1024)

        callback_url = f'{query["redirect_uri"][0]}?code=authorization-code&state={query["state"][0]}'
        with urllib.request.urlopen(callback_url, timeout=5) as response:
            assert response.status == 200

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle_auth)) as auth_client:
        credentials = await OpenAICodexAuth(store=MemoryStore(), http_client=auth_client).login_browser(
            open_url, timeout=10
        )

    assert credentials.account_id.get_secret_value() == _ACCOUNT_ID


async def test_device_login_reports_presentation_failure() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={'device_auth_id': 'device-id', 'user_code': 'USER-CODE', 'interval': 1},
        )

    def fail(code: OpenAICodexDeviceCode) -> None:
        raise RuntimeError('presentation failed')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexOAuthError, match='Unable to present'):
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(fail)


@pytest.mark.parametrize('timeout', [0.01, 1800])
async def test_device_polling_respects_effective_timeout(timeout: float, monkeypatch: pytest.MonkeyPatch) -> None:
    if timeout > 900:
        monkeypatch.setattr('pydantic_ai.auth._openai_codex_oauth._DEVICE_CODE_LIFETIME', 0.01)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/deviceauth/usercode'):
            return httpx.Response(
                200,
                json={'device_auth_id': 'device-id', 'user_code': 'USER-CODE', 'interval': 1},
            )
        return httpx.Response(403)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexOAuthError, match='timed out'):
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(
                lambda code: None, timeout=timeout
            )


def test_device_code_requires_timezone_aware_expiry() -> None:
    with pytest.raises(ValueError, match='timezone-aware'):
        OpenAICodexDeviceCode(
            verification_url='https://auth.example/device',
            user_code=SecretStr('USER-CODE'),
            expires_at=datetime(2030, 1, 2),
        )


async def test_device_login_falls_back_when_the_issuer_expiry_is_naive() -> None:
    """A naive issuer timestamp degrades to the fallback lifetime instead of failing the login.

    The deadline is subtracted from an aware `datetime.now`, so a naive value would otherwise raise
    an uncaught `TypeError`. Dropping it keeps sign-in working; only the shown deadline gets less
    precise, which is what the field's documented fallback is for.
    """
    shown: list[OpenAICodexDeviceCode] = []
    started = datetime.now(timezone.utc)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/deviceauth/usercode'):
            return httpx.Response(
                200,
                json={
                    'device_auth_id': 'device-id',
                    'user_code': 'USER-CODE',
                    'interval': 1,
                    'expires_at': '2026-08-06T12:00:00',
                },
            )
        return httpx.Response(403)

    with anyio.fail_after(30):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(OpenAICodexOAuthError, match='timed out'):
                await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(shown.append, timeout=0.01)

    # The fallback deadline is derived from `_DEVICE_CODE_LIFETIME`, so it sits well in the future.
    assert [code.expires_at > started for code in shown] == [True]


async def test_device_polling_stops_at_the_issuer_deadline() -> None:
    """The expiry shown to the user is the one enforced, not the longer fallback lifetime.

    An issuer deadline shorter than `_DEVICE_CODE_LIFETIME` used to bound only the displayed
    `expires_at`, leaving the loop polling a code that had already expired.
    """
    shown: list[OpenAICodexDeviceCode] = []
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=0.01)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith('/deviceauth/usercode'):
            return httpx.Response(
                200,
                json={
                    'device_auth_id': 'device-id',
                    'user_code': 'USER-CODE',
                    'interval': 1,
                    'expires_at': expires_at.isoformat(),
                },
            )
        return httpx.Response(403)

    # Bounded so a regression fails the run instead of polling for the full fallback lifetime.
    with anyio.fail_after(30):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(OpenAICodexOAuthError, match='timed out'):
                await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(shown.append, timeout=1800)

    assert [code.expires_at for code in shown] == [expires_at]


@pytest.mark.parametrize('method', ['browser', 'device'])
async def test_sync_presentation_callback_cannot_defeat_login_timeout(method: str) -> None:
    callback_started = threading.Event()
    release_callback = threading.Event()

    def block_callback(value: object) -> None:
        callback_started.set()
        release_callback.wait(5)

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith('/deviceauth/usercode')
        return httpx.Response(
            200,
            json={'device_auth_id': 'device-id', 'user_code': 'USER-CODE', 'interval': 1},
        )

    started_at = anyio.current_time()
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            auth = OpenAICodexAuth(store=MemoryStore(), http_client=client)
            with pytest.raises(OpenAICodexOAuthError, match='timed out'):
                if method == 'browser':
                    await auth.login_browser(block_callback, timeout=0.05)
                else:
                    await auth.login_device(block_callback, timeout=0.05)
    finally:
        release_callback.set()

    assert callback_started.is_set()
    assert anyio.current_time() - started_at < 1


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf')])
async def test_login_timeout_must_be_finite_and_positive(timeout: float) -> None:
    auth = OpenAICodexAuth(store=MemoryStore())
    with pytest.raises(UserError, match='finite and positive'):
        await auth.login_browser(lambda url: None, timeout=timeout)
    with pytest.raises(UserError, match='finite and positive'):
        await auth.login_device(lambda code: None, timeout=timeout)


async def test_oauth_validation_error_does_not_retain_secret_in_exception_chain() -> None:
    sentinel = 'plaintext-validation-secret'

    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps({'access_token': sentinel, 'unexpected': sentinel}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(OpenAICodexOAuthError) as exc_info:
            await OpenAICodexAuth(store=MemoryStore(), http_client=client).login_device(lambda code: None)

    error = exc_info.value
    rendered = ''.join(traceback.format_exception(error))
    assert sentinel not in rendered
    assert error.__cause__ is None
    assert error.__context__ is None


async def test_logout_cancellation_waits_for_delete_and_custom_lock_release() -> None:
    store = CheckpointExitStore(_credentials())
    request_started = anyio.Event()
    allow_response = anyio.Event()
    scopes: list[anyio.CancelScope] = []
    completed = False

    async def handle(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await allow_response.wait()
        return httpx.Response(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        auth = OpenAICodexAuth(store=store, http_client=client)

        async def logout() -> None:
            nonlocal completed
            with anyio.CancelScope() as scope:
                scopes.append(scope)
                await auth.logout()
                # Cancellation must prevent the call from returning
                completed = True  # pragma: no cover

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(logout)
            await request_started.wait()
            scopes[0].cancel()
            allow_response.set()

    assert not completed
    assert store.credentials is None
    assert store.exit_finished.is_set()
    with anyio.fail_after(1):
        async with store.exclusive():
            pass


async def test_logout_without_credentials_is_idempotent() -> None:
    result = await OpenAICodexAuth(store=MemoryStore()).logout()
    assert result.local_credentials_removed is False
    assert result.upstream_revoked is False


async def test_logout_reports_successful_upstream_revocation() -> None:
    requests: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        assert request.extensions['timeout'] == {'connect': 10, 'read': 10, 'write': 10, 'pool': 10}
        return httpx.Response(200)

    store = MemoryStore(_credentials())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await OpenAICodexAuth(store=store, http_client=client).logout()

    assert store.credentials is None
    assert result.upstream_revoked is True
    assert requests == [
        {'token': _REFRESH_TOKEN, 'token_type_hint': 'refresh_token', 'client_id': 'app_EMoamEEZ73f0CkXaXp7hrann'}
    ]


@pytest.mark.parametrize('local_only', [False, True])
async def test_logout_always_removes_local_credentials(local_only: bool) -> None:
    revoke_requests = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal revoke_requests
        revoke_requests += 1
        return httpx.Response(500)

    store = MemoryStore(_credentials())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        result = await OpenAICodexAuth(store=store, http_client=client).logout(local_only=local_only)

    assert store.credentials is None
    assert result.local_credentials_removed is True
    assert revoke_requests == (0 if local_only else 1)
    assert result.revocation_error == (None if local_only else 'Upstream OpenAI Codex token revocation failed.')

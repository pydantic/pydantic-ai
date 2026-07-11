from __future__ import annotations as _annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import sniffio
from pydantic import SecretStr

from pydantic_ai._cli import cli
from pydantic_ai.auth.codex import (
    CodexAuthStatus,
    CodexCredentials,
    CodexDeviceCode,
    CodexLogoutResult,
    CodexOAuthError,
)


@pytest.fixture(autouse=True)
def reset_sniffio_cvar() -> Iterator[None]:
    token = sniffio.current_async_library_cvar.set(None)
    try:
        yield
    finally:
        sniffio.current_async_library_cvar.reset(token)


def _credentials() -> CodexCredentials:
    return CodexCredentials(
        access_token=SecretStr('access-secret'),
        refresh_token=SecretStr('refresh-secret'),
        id_token=SecretStr('id-secret'),
        expires_at=datetime(2030, 1, 2, tzinfo=timezone.utc),
        account_id=SecretStr('account-secret'),
        revision='revision-1',
    )


class FakeAuth:
    def __init__(self) -> None:
        self.browser_logins = 0
        self.device_logins = 0
        self.local_only: bool | None = None

    async def login_browser(self, open_url: object) -> CodexCredentials:
        self.browser_logins += 1
        assert callable(open_url)
        open_url('https://auth.example/authorize?state=ephemeral-state')
        return _credentials()

    async def login_device(self, show_code: object) -> CodexCredentials:
        self.device_logins += 1
        assert callable(show_code)
        show_code(
            CodexDeviceCode(
                verification_url='https://auth.example/device',
                user_code=SecretStr('ONE-TIME-CODE'),
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
            )
        )
        return _credentials()

    async def status(self) -> CodexAuthStatus:
        return CodexAuthStatus(
            authenticated=True,
            expires_at=datetime(2030, 1, 2, tzinfo=timezone.utc),
            needs_refresh=False,
        )

    async def refresh(self) -> CodexCredentials:
        return _credentials()

    async def logout(self, *, local_only: bool = False) -> CodexLogoutResult:
        self.local_only = local_only
        return CodexLogoutResult(local_credentials_removed=True)


def _install_fake_auth(monkeypatch: pytest.MonkeyPatch, auth: FakeAuth) -> None:
    monkeypatch.setattr('pydantic_ai._cli.auth.CodexAuth', lambda: auth)


def test_cli_auth_browser_login(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    auth = FakeAuth()
    _install_fake_auth(monkeypatch, auth)
    open_browser = Mock(return_value=True)
    monkeypatch.setattr('pydantic_ai._cli.auth.webbrowser.open', open_browser)

    assert cli(['auth', 'login', 'codex'], prog_name='clai') == 0

    output = capsys.readouterr().out
    assert output == 'Opening a browser to sign in to Codex...\nCodex login complete.\n'
    assert 'ephemeral-state' not in output
    open_browser.assert_called_once()
    assert auth.browser_logins == 1


def test_cli_auth_browser_login_prints_url_when_browser_cannot_open(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    auth = FakeAuth()
    _install_fake_auth(monkeypatch, auth)
    monkeypatch.setattr('pydantic_ai._cli.auth.webbrowser.open', Mock(return_value=False))

    assert cli(['auth', 'login', 'codex'], prog_name='clai') == 0

    output = capsys.readouterr().out
    assert 'Open this URL to continue Codex login:' in output
    assert 'https://auth.example/authorize?state=ephemeral-state' in output


def test_cli_auth_device_login(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    auth = FakeAuth()
    _install_fake_auth(monkeypatch, auth)

    assert cli(['auth', 'login', 'codex', '--method', 'device']) == 0

    output = capsys.readouterr().out
    assert 'https://auth.example/device' in output
    assert 'ONE-TIME-CODE' in output
    assert 'Codex login complete.' in output
    assert auth.device_logins == 1


def test_cli_auth_status_json_is_secret_free(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    auth = FakeAuth()
    _install_fake_auth(monkeypatch, auth)

    assert cli(['auth', 'status', 'codex', '--json']) == 0

    output = capsys.readouterr().out
    assert json.loads(output) == {
        'provider': 'codex',
        'authenticated': True,
        'expires_at': '2030-01-02T00:00:00+00:00',
        'needs_refresh': False,
        'account_is_fedramp': False,
    }
    assert 'secret' not in output


@pytest.mark.parametrize(
    ('status', 'expected'),
    [
        (CodexAuthStatus(authenticated=False), 'Codex authentication: not signed in'),
        (CodexAuthStatus(authenticated=True), 'Codex authentication: ready (expires unknown)'),
        (
            CodexAuthStatus(
                authenticated=True,
                expires_at=datetime(2030, 1, 2, tzinfo=timezone.utc),
                needs_refresh=True,
            ),
            'Codex authentication: refresh required (expires 2030-01-02T00:00:00+00:00)',
        ),
    ],
)
def test_cli_auth_human_status_variants(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: CodexAuthStatus,
    expected: str,
) -> None:
    class StatusAuth(FakeAuth):
        async def status(self) -> CodexAuthStatus:
            return status

    _install_fake_auth(monkeypatch, StatusAuth())
    assert cli(['auth', 'status', 'codex']) == 0
    assert capsys.readouterr().out.strip() == expected


def test_cli_auth_refresh_and_local_logout(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    auth = FakeAuth()
    _install_fake_auth(monkeypatch, auth)

    assert cli(['auth', 'refresh', 'codex']) == 0
    assert cli(['auth', 'logout', 'codex', '--local-only']) == 0

    output = capsys.readouterr().out
    assert 'Codex credentials refreshed (expires 2030-01-02T00:00:00+00:00).' in output
    assert 'Codex logout complete.' in output
    assert auth.local_only is True
    assert 'secret' not in output


def test_cli_auth_logout_warning_without_local_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class LogoutAuth(FakeAuth):
        async def logout(self, *, local_only: bool = False) -> CodexLogoutResult:
            return CodexLogoutResult(
                local_credentials_removed=False,
                revocation_error='Upstream Codex token revocation failed.',
            )

    _install_fake_auth(monkeypatch, LogoutAuth())
    assert cli(['auth', 'logout', 'codex']) == 0
    output = capsys.readouterr().out
    assert 'Warning: Upstream Codex token revocation failed.' in output
    assert 'Codex authentication: not signed in' in output


def test_cli_auth_error_is_rendered_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class FailingAuth(FakeAuth):
        async def refresh(self) -> CodexCredentials:
            raise CodexOAuthError('Codex refresh failed safely.')

    _install_fake_auth(monkeypatch, FailingAuth())

    assert cli(['auth', 'refresh', 'codex']) == 1
    assert capsys.readouterr().out == 'Error: Codex refresh failed safely.\n'

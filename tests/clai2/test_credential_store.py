"""Exercise keyring storage with Windows' UTF-16 credential size limit, and the no-keyring file fallback."""

import io
import os
import stat
import sys
from pathlib import Path
from uuid import UUID

import keyring
import pytest
from keyring.errors import InitError, KeyringLocked, NoKeyringError, PasswordDeleteError
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials
from rich.console import Console

from pydantic_clai2.auth import CodexAuth, CodexCredentials
from pydantic_clai2.credential_store import credentials_path, load_codex_credentials, save_codex_credentials


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


def fake_browser(url: str) -> bool:
    return True


@pytest.fixture
def fallback(tmp_path: Path) -> Path:
    return tmp_path / 'config' / 'credentials.json'


async def test_large_codex_credentials(vault: dict[str, str], fallback: Path) -> None:
    source = CodexCredentials()
    credentials = OpenAICodexCredentials(
        access_token='fake-access' * 500, refresh_token='fake-refresh' * 300, account_id='fake-account'
    )
    await source.save(credentials)
    assert await source.load() == credentials
    assert len(vault) > 1
    refreshed = OpenAICodexCredentials(
        access_token='refreshed' * 500, refresh_token='new-refresh' * 300, account_id='fake-account'
    )
    await source.save(refreshed)
    assert await source.load() == refreshed


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    entries: dict[str, str] = {}

    def get(service: str, account: str) -> str | None:
        assert account == 'openai-codex'
        return entries.get(service)

    def set_value(service: str, account: str, value: str) -> None:
        assert account == 'openai-codex'
        if len(value.encode('utf-16-le')) > 2560:
            raise OSError(1783, 'CredWrite', 'The stub received bad data')
        entries[service] = value

    def delete(service: str, account: str) -> None:
        assert account == 'openai-codex'
        if service not in entries:
            raise PasswordDeleteError('Not found')
        del entries[service]

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)
    return entries


@pytest.mark.parametrize('value', ['x' * 1280, 'x' * 1281, 'x' * 12000, '\U0001f511' * 2000])
def test_windows_round_trip_and_refresh(vault: dict[str, str], fallback: Path, value: str) -> None:
    assert load_codex_credentials(fallback=fallback) is None
    save_codex_credentials(fallback=fallback, value=value)
    assert load_codex_credentials(fallback=fallback) == value
    original_services = set(vault) - {'pydantic-clai2'}
    save_codex_credentials(fallback=fallback, value=value + 'refreshed' * 1000)
    assert load_codex_credentials(fallback=fallback) == value + 'refreshed' * 1000
    assert original_services.isdisjoint(vault)
    save_codex_credentials(fallback=fallback, value='small')
    assert load_codex_credentials(fallback=fallback) == 'small'
    assert vault == {'pydantic-clai2': 'small'}


def test_oversized_single_entry_reproduces_windows_error(vault: dict[str, str], fallback: Path) -> None:
    value = 'x' * 1281
    with pytest.raises(OSError, match='CredWrite'):
        keyring.set_password('pydantic-clai2', 'openai-codex', value)
    assert not vault
    save_codex_credentials(fallback=fallback, value=value)
    assert load_codex_credentials(fallback=fallback) == value


def test_legacy_login(vault: dict[str, str], fallback: Path) -> None:
    vault['pydantic-clai2'] = '{"access_token":"legacy"}'
    assert load_codex_credentials(fallback=fallback) == '{"access_token":"legacy"}'
    save_codex_credentials(fallback=fallback, value='new' * 2000)
    assert load_codex_credentials(fallback=fallback) == 'new' * 2000


@pytest.mark.parametrize('manifest', ['clai-chunks-v1:bad', 'clai-chunks-v1:' + 'a' * 32 + ':0'])
def test_corrupt_manifest_can_be_replaced(vault: dict[str, str], fallback: Path, manifest: str) -> None:
    vault['pydantic-clai2'] = manifest
    with pytest.raises(UserError, match='invalid'):
        load_codex_credentials(fallback=fallback)
    save_codex_credentials(fallback=fallback, value='replacement')
    assert load_codex_credentials(fallback=fallback) == 'replacement'


def test_missing_chunk(vault: dict[str, str], fallback: Path) -> None:
    save_codex_credentials(fallback=fallback, value='x' * 5000)
    del vault[next(service for service in vault if service != 'pydantic-clai2')]
    with pytest.raises(UserError, match='incomplete'):
        load_codex_credentials(fallback=fallback)


@pytest.mark.parametrize('discard', [False, True])
def test_failed_chunk_preserves_login(
    vault: dict[str, str], fallback: Path, monkeypatch: pytest.MonkeyPatch, *, discard: bool
) -> None:
    save_codex_credentials(fallback=fallback, value='previous' * 1000)
    previous = dict(vault)
    original_set = keyring.set_password
    writes = 0

    def fail(service: str, account: str, value: str) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            if discard:
                return
            raise OSError('backend unavailable')
        original_set(service, account, value)

    monkeypatch.setattr(keyring, 'set_password', fail)
    with pytest.raises((OSError, UserError)):
        save_codex_credentials(fallback=fallback, value='replacement' * 1000)
    assert vault == previous
    assert load_codex_credentials(fallback=fallback) == 'previous' * 1000


def test_uncertain_manifest_write_retains_chunks(
    vault: dict[str, str], fallback: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_set = keyring.set_password

    def fail_after_write(service: str, account: str, value: str) -> None:
        original_set(service, account, value)
        if service == 'pydantic-clai2':
            raise OSError('backend unavailable after write')

    monkeypatch.setattr(keyring, 'set_password', fail_after_write)
    with pytest.raises(OSError):
        save_codex_credentials(fallback=fallback, value='replacement' * 1000)
    assert load_codex_credentials(fallback=fallback) == 'replacement' * 1000


@pytest.fixture
def no_keyring(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    """Behave like keyring's `fail` backend: every call raises before touching a store."""
    error: type[Exception] = getattr(request, 'param', NoKeyringError)

    def raise_error(*args: str) -> None:
        raise error('No recommended backend was available')

    monkeypatch.setattr(keyring, 'get_password', raise_error)
    monkeypatch.setattr(keyring, 'set_password', raise_error)
    monkeypatch.setattr(keyring, 'delete_password', raise_error)


@pytest.mark.parametrize('no_keyring', [NoKeyringError, InitError], indirect=True)
def test_file_fallback_when_no_keyring(fallback: Path, no_keyring: None) -> None:
    assert load_codex_credentials(fallback=fallback) is None
    save_codex_credentials(fallback=fallback, value='{"access_token":"first"}')
    assert load_codex_credentials(fallback=fallback) == '{"access_token":"first"}'
    save_codex_credentials(fallback=fallback, value='{"access_token":"refreshed"}')
    assert load_codex_credentials(fallback=fallback) == '{"access_token":"refreshed"}'
    assert list(fallback.parent.iterdir()) == [fallback]
    if sys.platform != 'win32':
        assert stat.S_IMODE(fallback.stat().st_mode) == 0o600


def test_planted_staging_symlink_is_not_followed(
    fallback: Path, no_keyring: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A symlink at the exact staging path the write would use never receives tokens."""
    target = fallback.parent / 'stolen.txt'
    fallback.parent.mkdir()
    target.write_text('untouched')
    staging = fallback.with_name(f'credentials.json.{"0" * 32}.tmp')
    staging.symlink_to(target)
    monkeypatch.setattr('pydantic_clai2.credential_store.uuid4', lambda: UUID(int=0))
    save_codex_credentials(fallback=fallback, value='{"access_token":"secret"}')
    assert target.read_text() == 'untouched'
    assert not staging.is_symlink()
    assert fallback.read_text() == '{"access_token":"secret"}'


def test_staging_race_is_refused(fallback: Path, no_keyring: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file appearing between the stale sweep and the open is refused, not written through."""
    real_open = os.open

    def planting_open(path: str, flags: int, mode: int = 0o777) -> int:
        if flags & os.O_EXCL:
            Path(path).write_text('{"access_token":"attacker"}')
        return real_open(path, flags, mode)

    monkeypatch.setattr('pydantic_clai2.credential_store.os.open', planting_open)
    with pytest.raises(FileExistsError):
        save_codex_credentials(fallback=fallback, value='{"access_token":"secret"}')
    assert not fallback.exists()


def test_stale_staging_files_are_cleared(fallback: Path, no_keyring: None) -> None:
    fallback.parent.mkdir()
    stale = fallback.with_name('credentials.json.deadbeef.tmp')
    stale.write_text('{"access_token":"leaked-after-a-crash"}')
    unrelated = fallback.parent / 'other.tmp'
    unrelated.write_text('mine')
    save_codex_credentials(fallback=fallback, value='{"access_token":"fresh"}')
    assert not stale.exists()
    assert unrelated.read_text() == 'mine'


def test_locked_keyring_is_not_a_fallback(fallback: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def locked(*args: str) -> None:
        raise KeyringLocked('Unlock the keyring first')

    monkeypatch.setattr(keyring, 'get_password', locked)
    with pytest.raises(KeyringLocked):
        save_codex_credentials(fallback=fallback, value='secret')
    assert not fallback.exists()
    with pytest.raises(KeyringLocked):
        load_codex_credentials(fallback=fallback)


def test_keyring_save_removes_plaintext_copy(vault: dict[str, str], fallback: Path) -> None:
    fallback.parent.mkdir()
    fallback.write_text('{"access_token":"from-file"}')
    assert load_codex_credentials(fallback=fallback) == '{"access_token":"from-file"}'
    save_codex_credentials(fallback=fallback, value='{"access_token":"in-keyring"}')
    assert not fallback.exists()
    assert load_codex_credentials(fallback=fallback) == '{"access_token":"in-keyring"}'


async def test_login_reports_plaintext_location(no_keyring: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fallback path is derived from the account, so login can name the file it wrote."""
    credentials = OpenAICodexCredentials(access_token='a', refresh_token='r', account_id='x')

    async def exchange(self: object) -> OpenAICodexCredentials:
        return credentials

    monkeypatch.setattr('pydantic_ai.providers.openai_codex.OpenAICodexOAuthFlow.exchange_code_from_callback', exchange)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    auth = CodexAuth(Console(file=io.StringIO()))
    message = await auth.login([])
    path = credentials_path()
    assert 'plaintext' in message
    assert str(path) in message
    assert await auth.source.load() == credentials


def test_fallback_paths_are_per_account(tmp_path: Path, no_keyring: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """One account's fallback file cannot overwrite another's, and XDG decides the directory."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    codex = credentials_path()
    vllm = credentials_path(account='vllm')
    assert codex.parent == vllm.parent == tmp_path / 'pydantic-clai2'
    assert codex.name != vllm.name
    save_codex_credentials(value='codex-tokens', account='openai-codex')
    save_codex_credentials(value='vllm-token', account='vllm')
    assert load_codex_credentials(fallback=codex) == 'codex-tokens'
    assert load_codex_credentials(account='vllm', fallback=vllm) == 'vllm-token'

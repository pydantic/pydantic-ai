"""One keyring read per session for every credential, older per-entry logins, and the no-keyring file fallback."""

import io
import os
import stat
import sys
from pathlib import Path
from uuid import UUID

import keyring
import pytest
from keyring.errors import InitError, KeyringLocked, NoKeyringError
from rich.console import Console

from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials
from pydantic_clai2.auth import CodexAuth, CodexCredentials
from pydantic_clai2.config import credential_store
from pydantic_clai2.config.credential_store import (
    credentials_path,
    delete_credentials,
    has_credentials,
    load_codex_credentials,
    save_codex_credentials,
)
from tests.clai2.conftest import stored_accounts

Vault = dict[tuple[str, str], str]
KEY = ('pydantic-clai2', 'encryption-key')
LEGACY = ('pydantic-clai2', 'openai-codex')


def fake_browser(url: str) -> bool:
    return True


def new_session() -> None:
    """Forget the key a previous process read, as a restart would."""
    credential_store._stored_key.cache_clear()  # pyright: ignore[reportPrivateUsage]


def legacy_chunks(vault: Vault, value: str, *, count: int = 3) -> None:
    """Store `value` the way an older CLAI split large bundles for Windows' per-entry size limit."""
    generation = 'a' * 32
    vault[LEGACY] = f'clai-chunks-v1:{generation}:{count}'
    size = -(-len(value) // count)
    for index in range(count):
        vault[f'pydantic-clai2.openai-codex.{generation}.{index}', 'openai-codex'] = value[
            index * size : (index + 1) * size
        ]


@pytest.fixture
def fallback(tmp_path: Path) -> Path:
    return tmp_path / 'config' / 'credentials.json'


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch, vault: Vault) -> list[tuple[str, str]]:
    """Reads of entries that exist: on macOS, each one can ask the user for their password."""
    seen: list[tuple[str, str]] = []

    def get(service: str, account: str) -> str | None:
        if (value := vault.get((service, account))) is not None:
            seen.append((service, account))
        return value

    monkeypatch.setattr(keyring, 'get_password', get)
    return seen


def test_a_session_reads_one_keyring_entry(vault: Vault, reads: list[tuple[str, str]]) -> None:
    """Marcelo's startup read `api-keys` once per plugin plus each plugin's own entry: 4 prompts, not 1."""
    accounts = ('api-keys', 'linear', 'posthog', 'mcp-day_ai')
    for account in accounts:
        save_codex_credentials(account=account, value=f'{account}-secret')
    assert list(vault) == [KEY]
    assert stored_accounts() == set(accounts)
    new_session()
    reads.clear()
    for _ in range(3):
        assert [load_codex_credentials(account=account) for account in accounts] == [
            f'{account}-secret' for account in accounts
        ]
    save_codex_credentials(account='linear', value='refreshed')
    assert load_codex_credentials(account='linear') == 'refreshed'
    assert reads == [KEY]


@pytest.mark.parametrize('value', ['small', 'x' * 12000, '\U0001f511' * 2000])
def test_round_trip_is_encrypted_at_rest(vault: Vault, fallback: Path, value: str) -> None:
    assert load_codex_credentials(fallback=fallback) is None
    save_codex_credentials(fallback=fallback, value=value)
    encrypted = fallback.with_suffix('.enc')
    assert value.encode() not in encrypted.read_bytes()
    if sys.platform != 'win32':  # pragma: no branch
        assert stat.S_IMODE(encrypted.stat().st_mode) == 0o600
    assert not fallback.exists()
    assert list(vault) == [KEY]
    new_session()
    assert load_codex_credentials(fallback=fallback) == value


async def test_large_codex_credentials(vault: Vault) -> None:
    source = CodexCredentials()
    credentials = OpenAICodexCredentials(
        access_token='fake-access' * 500, refresh_token='fake-refresh' * 300, account_id='fake-account'
    )
    await source.save_login(credentials)
    assert await source.load() == credentials
    refreshed = OpenAICodexCredentials(
        access_token='refreshed' * 500, refresh_token='new-refresh' * 300, account_id='fake-account'
    )
    await source.save(refreshed)
    assert await source.load() == refreshed
    assert list(vault) == [KEY]


def test_legacy_login_moves_into_an_encrypted_file(vault: Vault, reads: list[tuple[str, str]], fallback: Path) -> None:
    vault[LEGACY] = '{"access_token":"legacy"}'
    assert load_codex_credentials(fallback=fallback) == '{"access_token":"legacy"}'
    assert reads.count(LEGACY) == 1, 'moving it reads the older entry once'
    assert list(vault) == [KEY]
    new_session()
    reads.clear()
    assert load_codex_credentials(fallback=fallback) == '{"access_token":"legacy"}'
    assert reads == [KEY]


def test_legacy_chunked_login_moves_into_an_encrypted_file(vault: Vault, fallback: Path) -> None:
    legacy_chunks(vault, 'x' * 5000)
    assert load_codex_credentials(fallback=fallback) == 'x' * 5000
    assert list(vault) == [KEY]
    assert load_codex_credentials(fallback=fallback) == 'x' * 5000


def test_newer_login_wins_a_migration_race(vault: Vault, monkeypatch: pytest.MonkeyPatch, fallback: Path) -> None:
    """Another process saves a new login between this one reading the older entry and moving it."""
    vault[LEGACY] = 'older'
    original_get = keyring.get_password

    def get(service: str, account: str) -> str | None:
        """The first read is the older entry; the other process saves right after it."""
        value = original_get(service, account)
        monkeypatch.setattr(keyring, 'get_password', original_get)
        save_codex_credentials(fallback=fallback, value='newer')
        return value

    monkeypatch.setattr(keyring, 'get_password', get)
    assert load_codex_credentials(fallback=fallback) == 'newer'
    assert load_codex_credentials(fallback=fallback) == 'newer'


def test_swept_staging_file_leaves_the_move_to_the_other_process(
    vault: Vault, monkeypatch: pytest.MonkeyPatch, fallback: Path
) -> None:
    """A concurrent migration's stale-file sweep removed this one's staging file before it was linked."""
    vault[LEGACY] = 'older'

    def swept(source: str, destination: Path) -> None:
        raise FileNotFoundError(source)

    monkeypatch.setattr('pydantic_clai2.config.credential_store.os.link', swept)
    assert load_codex_credentials(fallback=fallback) == 'older'
    assert vault[LEGACY] == 'older', 'the entry stays until a move succeeds'


def test_key_created_by_another_process_is_used(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two first-ever saves race; the one that waited on the lock reuses the key instead of replacing it."""
    other = 'c2VjcmV0LWtleS1mcm9tLWFub3RoZXItcHJvY2VzcyE='
    original_get = keyring.get_password

    def get(service: str, account: str) -> str | None:
        value = original_get(service, account)
        if (service, account) == KEY and value is None:
            vault[KEY] = other
        return value

    monkeypatch.setattr(keyring, 'get_password', get)
    save_codex_credentials(value='mine')
    assert vault == {KEY: other}
    new_session()
    assert load_codex_credentials() == 'mine'


@pytest.mark.parametrize('damage', ['lost key', 'malformed key', 'corrupt file'])
def test_undecryptable_login_asks_to_reconnect(vault: Vault, fallback: Path, damage: str) -> None:
    save_codex_credentials(fallback=fallback, value='secret')
    new_session()
    if damage == 'lost key':
        del vault[KEY]
    elif damage == 'malformed key':
        vault[KEY] = 'not a Fernet key'
    else:
        fallback.with_suffix('.enc').write_text('not a token')
    with pytest.raises(UserError, match='cannot be decrypted'):
        load_codex_credentials(fallback=fallback)
    save_codex_credentials(fallback=fallback, value='reconnected')
    assert load_codex_credentials(fallback=fallback) == 'reconnected'


@pytest.mark.parametrize('manifest', ['clai-chunks-v1:bad', 'clai-chunks-v1:' + 'a' * 32 + ':0'])
def test_corrupt_manifest_can_be_replaced(vault: Vault, fallback: Path, manifest: str) -> None:
    vault[LEGACY] = manifest
    with pytest.raises(UserError, match='invalid'):
        load_codex_credentials(fallback=fallback)
    save_codex_credentials(fallback=fallback, value='replacement')
    assert load_codex_credentials(fallback=fallback) == 'replacement'
    assert list(vault) == [KEY]


def test_missing_chunk(vault: Vault, fallback: Path) -> None:
    legacy_chunks(vault, 'x' * 5000)
    del vault[f'pydantic-clai2.openai-codex.{"a" * 32}.1', 'openai-codex']
    with pytest.raises(UserError, match='incomplete'):
        load_codex_credentials(fallback=fallback)


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
    assert list(fallback.parent.glob('credentials.*')) == [fallback]
    if sys.platform != 'win32':  # pragma: no branch
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
    monkeypatch.setattr('pydantic_clai2.config.credential_store.uuid4', lambda: UUID(int=0))
    save_codex_credentials(fallback=fallback, value='{"access_token":"secret"}')
    assert target.read_text() == 'untouched'
    assert not staging.is_symlink()
    assert fallback.read_text(encoding='utf-8') == '{"access_token":"secret"}'


def test_staging_race_is_refused(fallback: Path, no_keyring: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file appearing between the stale sweep and the open is refused, not written through."""
    real_open = os.open

    def planting_open(path: str, flags: int, mode: int = 0o777) -> int:
        if flags & os.O_EXCL:  # pragma: no branch
            Path(path).write_text('{"access_token":"attacker"}', encoding='utf-8')
        return real_open(path, flags, mode)

    monkeypatch.setattr('pydantic_clai2.config.credential_store.os.open', planting_open)
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


def test_keyring_save_removes_plaintext_copy(vault: Vault, fallback: Path) -> None:
    fallback.parent.mkdir()
    fallback.write_text('{"access_token":"from-file"}', encoding='utf-8')
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


@pytest.mark.parametrize('legacy', [None, 'small', 'x' * 5000])
def test_delete_removes_every_copy(vault: Vault, fallback: Path, legacy: str | None) -> None:
    save_codex_credentials(fallback=fallback, value='current')
    if legacy == 'small':
        vault[LEGACY] = legacy
    elif legacy is not None:
        legacy_chunks(vault, legacy)
    fallback.write_text('stale', encoding='utf-8')
    delete_credentials(fallback=fallback)
    assert list(vault) == [KEY]
    assert list(fallback.parent.glob('credentials.*')) == []
    delete_credentials(fallback=fallback)
    assert load_codex_credentials(fallback=fallback) is None


def test_delete_clears_a_corrupt_manifest(vault: Vault, fallback: Path) -> None:
    vault[LEGACY] = 'clai-chunks-v1:broken'
    delete_credentials(fallback=fallback)
    assert vault == {}


def test_delete_skips_chunks_that_are_already_gone(fallback: Path) -> None:
    keyring.set_password(*LEGACY, f'clai-chunks-v1:{"0" * 32}:2')
    delete_credentials(fallback=fallback)
    assert load_codex_credentials(fallback=fallback) is None


def test_delete_without_keyring_removes_the_file(no_keyring: None, fallback: Path) -> None:
    save_codex_credentials(fallback=fallback, value='plain')
    delete_credentials(fallback=fallback)
    assert not fallback.exists()


def test_an_older_per_entry_login_counts_as_signed_in_and_moves_into_its_file(
    vault: Vault, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups: list[str] = []
    stored = keyring.get_password

    def get(service: str, account: str) -> str | None:
        lookups.append(account)
        return stored(service, account)

    monkeypatch.setattr(keyring, 'get_password', get)
    vault[LEGACY] = '{"access_token":"legacy"}'
    assert has_credentials(account='openai-codex')
    assert list(vault) == [KEY] and credentials_path().with_suffix('.enc').is_file()
    lookups.clear()
    assert has_credentials(account='openai-codex'), 'the file answers now'
    assert not has_credentials(account='openai-codex@work'), 'profiles came after per-entry logins'
    assert not has_credentials(account='github-copilot')
    assert not has_credentials(account='github-copilot')
    assert lookups == ['github-copilot'], 'an account without an older entry is looked up once'


def test_a_locked_keyring_is_signed_out_until_it_unlocks(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    stored = keyring.get_password

    def locked(service: str, account: str) -> str | None:
        raise KeyringLocked('Unlock the keyring first')

    monkeypatch.setattr(keyring, 'get_password', locked)
    vault[LEGACY] = '{"access_token":"legacy"}'
    assert not has_credentials(account='openai-codex')
    monkeypatch.setattr(keyring, 'get_password', stored)
    assert has_credentials(account='openai-codex'), 'a failed lookup is tried again'

"""Exercise keyring storage with Windows' UTF-16 credential size limit."""

import keyring
import pytest
from keyring.errors import PasswordDeleteError
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials

from pydantic_clai2.auth import CodexCredentials
from pydantic_clai2.credential_store import load_codex_credentials, save_codex_credentials


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


async def test_large_codex_credentials(vault: dict[str, str]) -> None:
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
def test_windows_round_trip_and_refresh(vault: dict[str, str], value: str) -> None:
    assert load_codex_credentials() is None
    save_codex_credentials(value=value)
    assert load_codex_credentials() == value
    original_services = set(vault) - {'pydantic-clai2'}
    save_codex_credentials(value=value + 'refreshed' * 1000)
    assert load_codex_credentials() == value + 'refreshed' * 1000
    assert original_services.isdisjoint(vault)
    save_codex_credentials(value='small')
    assert load_codex_credentials() == 'small'
    assert vault == {'pydantic-clai2': 'small'}


def test_oversized_single_entry_reproduces_windows_error(vault: dict[str, str]) -> None:
    value = 'x' * 1281
    with pytest.raises(OSError, match='CredWrite'):
        keyring.set_password('pydantic-clai2', 'openai-codex', value)
    assert not vault
    save_codex_credentials(value=value)
    assert load_codex_credentials() == value


def test_legacy_login(vault: dict[str, str]) -> None:
    vault['pydantic-clai2'] = '{"access_token":"legacy"}'
    assert load_codex_credentials() == '{"access_token":"legacy"}'
    save_codex_credentials(value='new' * 2000)
    assert load_codex_credentials() == 'new' * 2000


@pytest.mark.parametrize('manifest', ['clai-chunks-v1:bad', 'clai-chunks-v1:' + 'a' * 32 + ':0'])
def test_corrupt_manifest_can_be_replaced(vault: dict[str, str], manifest: str) -> None:
    vault['pydantic-clai2'] = manifest
    with pytest.raises(UserError, match='invalid'):
        load_codex_credentials()
    save_codex_credentials(value='replacement')
    assert load_codex_credentials() == 'replacement'


def test_missing_chunk(vault: dict[str, str]) -> None:
    save_codex_credentials(value='x' * 5000)
    del vault[next(service for service in vault if service != 'pydantic-clai2')]
    with pytest.raises(UserError, match='incomplete'):
        load_codex_credentials()


@pytest.mark.parametrize('discard', [False, True])
def test_failed_chunk_preserves_login(vault: dict[str, str], monkeypatch: pytest.MonkeyPatch, *, discard: bool) -> None:
    save_codex_credentials(value='previous' * 1000)
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
        save_codex_credentials(value='replacement' * 1000)
    assert vault == previous
    assert load_codex_credentials() == 'previous' * 1000


def test_uncertain_manifest_write_retains_chunks(vault: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    original_set = keyring.set_password

    def fail_after_write(service: str, account: str, value: str) -> None:
        original_set(service, account, value)
        if service == 'pydantic-clai2':
            raise OSError('backend unavailable after write')

    monkeypatch.setattr(keyring, 'set_password', fail_after_write)
    with pytest.raises(OSError):
        save_codex_credentials(value='replacement' * 1000)
    assert load_codex_credentials() == 'replacement' * 1000

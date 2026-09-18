"""Exercise the cross-process SQLite lock without timing-dependent sleeps."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from pydantic_ai.exceptions import UserError

from pydantic_clai2 import api_keys, openrouter, vllm
from pydantic_clai2.credential_store import load_codex_credentials


@pytest.mark.parametrize('action', ['save', 'rename', 'delete'])
def test_serialized_updates(monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    api_keys.save_key(name='EXISTING', value='existing-secret')
    original = sqlite3.connect
    first_started, second_started = Event(), Event()

    def connect(path: Path, *, timeout: float) -> sqlite3.Connection:
        connection = original(path, timeout=timeout)
        if not first_started.is_set():
            first_started.set()
        else:
            second_started.set()
        return connection

    def mutate() -> str:
        if action == 'rename':
            return api_keys.rename_key(name='EXISTING', new_name='RENAMED')
        if action == 'delete':
            return api_keys.delete_key(name='EXISTING')
        return api_keys.save_key(name='FIRST', value='first-secret')

    # SQLite connections are independent lock participants, just as in separate processes.
    with ThreadPoolExecutor(max_workers=2) as pool:
        with api_keys.key_transaction():
            monkeypatch.setattr(sqlite3, 'connect', connect)
            first = pool.submit(mutate)
            assert first_started.wait(timeout=5)
            second = pool.submit(api_keys.save_key, name='SECOND', value='second-secret')
            assert second_started.wait(timeout=5)
            assert not first.done() and not second.done()
        first.result(timeout=5)
        second.result(timeout=5)
    keys = api_keys.load_keys()
    assert keys['SECOND'].get_secret_value() == 'second-secret'
    if action == 'save':
        assert set(keys) == {'EXISTING', 'FIRST', 'SECOND'}
    elif action == 'rename':
        assert set(keys) == {'RENAMED', 'SECOND'}
        assert keys['RENAMED'].get_secret_value() == 'existing-secret'
    else:
        assert set(keys) == {'SECOND'}


def test_lock_error_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(path: Path, *, timeout: float) -> sqlite3.Connection:
        raise sqlite3.OperationalError('backend details')

    monkeypatch.setattr(sqlite3, 'connect', fail)
    with pytest.raises(UserError, match='Cannot lock API keys') as error:
        api_keys.save_key(name='KEY', value='secret')
    assert 'backend details' not in str(error.value)


@pytest.mark.parametrize('provider', ['vllm', 'openrouter'])
def test_connection_save_and_rename_share_lock(monkeypatch: pytest.MonkeyPatch, provider: str) -> None:
    api_keys.save_key(name='KEY', value='secret')
    reference = api_keys.KeyReference(name='KEY')
    started = Event()
    original = sqlite3.connect

    def connect(path: Path, *, timeout: float) -> sqlite3.Connection:
        connection = original(path, timeout=timeout)
        started.set()
        return connection

    def save() -> None:
        if provider == 'vllm':
            vllm.save_connection(vllm.Connection(url='http://localhost:8000', token=reference))
        else:
            openrouter.save_connection(openrouter.Connection(token=reference))

    with ThreadPoolExecutor(max_workers=1) as pool:
        with api_keys.key_transaction():
            monkeypatch.setattr(sqlite3, 'connect', connect)
            saving = pool.submit(save)
            assert started.wait(timeout=5)
            assert not saving.done()
        saving.result(timeout=5)
    with pytest.raises(ValueError, match='used by'):
        api_keys.rename_key(name='KEY', new_name='NEW')
    previous = load_codex_credentials(account=provider)
    api_keys.delete_key(name='KEY')
    with pytest.raises(UserError, match='no longer exists'):
        save()
    assert load_codex_credentials(account=provider) == previous

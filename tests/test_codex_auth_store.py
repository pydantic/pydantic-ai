from __future__ import annotations as _annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import anyio
import pytest
from anyio.to_thread import run_sync
from pydantic import SecretStr

from pydantic_ai.auth._codex_store import FileCodexCredentialStore
from pydantic_ai.auth.codex import CodexCredentials, CodexCredentialsError

pytestmark = pytest.mark.anyio


def _credentials(revision: str) -> CodexCredentials:
    return CodexCredentials(
        access_token=SecretStr(f'access-{revision}'),
        refresh_token=SecretStr(f'refresh-{revision}'),
        id_token=SecretStr(f'id-{revision}'),
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        account_id=SecretStr('account-id'),
        revision=revision,
    )


async def test_file_store_round_trip_permissions_and_unrelated_records(tmp_path: Path) -> None:
    path = tmp_path / 'credentials' / 'auth.json'
    path.parent.mkdir(mode=0o777)
    if os.name != 'nt':  # pragma: no branch - platform-specific permission setup
        os.chmod(path.parent, 0o750)
    path.write_text(
        json.dumps({'version': 1, 'providers': {'another-provider': {'value': 'preserve-me'}}}),
        encoding='utf-8',
    )
    store = FileCodexCredentialStore(path)
    credentials = _credentials('revision-1')

    async with store.exclusive():
        assert await store.save(credentials, expected_revision=None)

    assert await store.load() == credentials
    document = json.loads(path.read_text(encoding='utf-8'))
    assert document['providers']['another-provider'] == {'value': 'preserve-me'}
    assert document['providers']['codex']['refresh_token'] == 'refresh-revision-1'
    if os.name != 'nt':  # pragma: no branch - platform-specific permission assertions
        assert path.parent.stat().st_mode & 0o777 == 0o750
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.with_name('auth.json.lock').stat().st_mode & 0o777 == 0o600


async def test_default_file_store_hardens_existing_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    parent = tmp_path / '.pydantic-ai'
    parent.mkdir()
    if os.name != 'nt':  # pragma: no branch - platform-specific permission setup
        os.chmod(parent, 0o750)
    monkeypatch.setattr('pydantic_ai.auth._codex_store.Path.home', lambda: tmp_path)

    store = FileCodexCredentialStore()
    async with store.exclusive():
        assert await store.save(_credentials('revision'), expected_revision=None)

    if os.name != 'nt':  # pragma: no branch - platform-specific permission assertion
        assert parent.stat().st_mode & 0o777 == 0o700


async def test_file_store_creates_missing_parent_with_private_permissions(tmp_path: Path) -> None:
    path = tmp_path / 'missing' / 'auth.json'
    store = FileCodexCredentialStore(path)

    async with store.exclusive():
        assert await store.save(_credentials('revision'), expected_revision=None)

    if os.name != 'nt':  # pragma: no branch - platform-specific permission assertion
        assert path.parent.stat().st_mode & 0o777 == 0o700


async def test_file_store_compare_and_swap(tmp_path: Path) -> None:
    store = FileCodexCredentialStore(tmp_path / 'auth.json')
    async with store.exclusive():
        assert await store.save(_credentials('revision-1'), expected_revision=None)
        assert not await store.save(_credentials('revision-2'), expected_revision='wrong-revision')
        assert await store.save(_credentials('revision-2'), expected_revision='revision-1')
        assert not await store.delete(expected_revision='revision-1')
        assert await store.delete(expected_revision='revision-2')
        assert not await store.delete(expected_revision='revision-2')

    assert await store.load() is None


@pytest.mark.parametrize(
    'content',
    [
        '{not-json',
        json.dumps({'version': 999, 'providers': {}}),
        json.dumps({'version': 1, 'providers': {'codex': {'access_token': 'partial'}}}),
    ],
)
async def test_file_store_rejects_malformed_data(tmp_path: Path, content: str) -> None:
    path = tmp_path / 'auth.json'
    path.write_text(content, encoding='utf-8')
    store = FileCodexCredentialStore(path)

    with pytest.raises(CodexCredentialsError, match=r'malformed|unsupported'):
        await store.load()


async def test_malformed_store_error_does_not_retain_plaintext_document(tmp_path: Path) -> None:
    sentinel = 'plaintext-store-secret'
    path = tmp_path / 'auth.json'
    path.write_text(f'{{"refresh_token":"{sentinel}"', encoding='utf-8')

    with pytest.raises(CodexCredentialsError) as exc_info:
        await FileCodexCredentialStore(path).load()

    error = exc_info.value
    assert sentinel not in ''.join(traceback.format_exception(error))
    assert error.__cause__ is None
    assert error.__context__ is None


async def test_file_store_lock_excludes_a_spawned_process(tmp_path: Path) -> None:
    path = tmp_path / 'auth.json'
    lock_path = path.with_name('auth.json.lock')
    script = """
import sys
from filelock import FileLock

with FileLock(sys.argv[1], timeout=5):
    print('locked', flush=True)
    sys.stdin.readline()
"""
    environment = os.environ.copy()
    environment.pop('COVERAGE_FILE', None)
    environment.pop('COVERAGE_PROCESS_CONFIG', None)
    environment.pop('COVERAGE_PROCESS_START', None)
    process = subprocess.Popen(
        [sys.executable, '-c', script, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        env=environment,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    try:
        assert await run_sync(process.stdout.readline) == 'locked\n'

        acquired = anyio.Event()

        async def acquire() -> None:
            async with FileCodexCredentialStore(path).exclusive():
                acquired.set()

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(acquire)
            await anyio.sleep(0.1)
            assert not acquired.is_set()
            process.stdin.write('\n')
            process.stdin.flush()
            with anyio.fail_after(5):
                await acquired.wait()
    finally:
        process.stdin.close()
        try:
            await run_sync(process.wait, 5)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive subprocess cleanup
            process.terminate()
            await run_sync(process.wait)
        process.stdout.close()


async def test_file_store_lock_timeout_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'auth.json'
    monkeypatch.setattr('pydantic_ai.auth._codex_store._LOCK_TIMEOUT', 0.01)
    monkeypatch.setattr('pydantic_ai.auth._codex_store._LOCK_POLL_INTERVAL', 0.001)

    async with FileCodexCredentialStore(path).exclusive():
        with pytest.raises(CodexCredentialsError, match='Timed out waiting'):
            async with FileCodexCredentialStore(path).exclusive():
                pass


async def test_file_store_wraps_lock_permission_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / 'auth.json'
    lock_path = path.with_name('auth.json.lock')
    real_chmod = os.chmod

    def fail_lock_chmod(target: os.PathLike[str] | str, mode: int) -> None:
        assert Path(target) == lock_path
        assert mode == 0o600
        raise OSError('simulated permission failure')

    monkeypatch.setattr('pydantic_ai.auth._codex_store.os.chmod', fail_lock_chmod)
    with pytest.raises(CodexCredentialsError, match='Unable to lock'):
        async with FileCodexCredentialStore(path).exclusive():
            pass

    # The failed permission hardening must not leave the file lock held.
    monkeypatch.setattr('pydantic_ai.auth._codex_store.os.chmod', real_chmod)
    async with FileCodexCredentialStore(path).exclusive():
        pass


async def test_file_store_wraps_lock_os_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = FileCodexCredentialStore(tmp_path / 'auth.json')

    def fail() -> None:
        raise OSError('simulated lock failure')

    monkeypatch.setattr(store, '_prepare_directory', fail)
    with pytest.raises(CodexCredentialsError, match='Unable to lock'):
        async with store.exclusive():
            pass


@pytest.mark.parametrize(
    ('method_name', 'message'),
    [('_load_sync', 'Unable to read'), ('_save_sync', 'Unable to write'), ('_delete_sync', 'Unable to update')],
)
async def test_file_store_wraps_operation_os_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method_name: str, message: str
) -> None:
    store = FileCodexCredentialStore(tmp_path / 'auth.json')

    def fail(*args: object) -> None:
        raise OSError('simulated operation failure')

    monkeypatch.setattr(store, method_name, fail)
    with pytest.raises(CodexCredentialsError, match=message):
        if method_name == '_load_sync':
            await store.load()
        elif method_name == '_save_sync':
            await store.save(_credentials('revision'), expected_revision=None)
        else:
            await store.delete(expected_revision='revision')


async def test_malformed_store_errors_propagate_from_all_operations(tmp_path: Path) -> None:
    path = tmp_path / 'auth.json'
    path.write_text('{malformed', encoding='utf-8')
    store = FileCodexCredentialStore(path)

    with pytest.raises(CodexCredentialsError):
        await store.save(_credentials('revision'), expected_revision=None)
    with pytest.raises(CodexCredentialsError):
        await store.delete(expected_revision='revision')


async def test_cancelled_file_lock_waiter_does_not_leave_lock_held(tmp_path: Path) -> None:
    path = tmp_path / 'auth.json'
    first = FileCodexCredentialStore(path)
    waiter_finished = anyio.Event()
    scopes: list[anyio.CancelScope] = []

    async def wait_for_lock() -> None:
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            try:
                async with FileCodexCredentialStore(path).exclusive():
                    pytest.fail('cancelled waiter unexpectedly acquired the lock')  # pragma: no cover
            finally:
                waiter_finished.set()

    async with first.exclusive():
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(wait_for_lock)
            while not scopes:
                await anyio.sleep(0)
            await anyio.sleep(0.1)
            scopes[0].cancel()
            with anyio.fail_after(5):
                await waiter_finished.wait()

    with anyio.fail_after(5):
        async with FileCodexCredentialStore(path).exclusive():
            pass


async def test_atomic_replace_failure_preserves_previous_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / 'auth.json'
    store = FileCodexCredentialStore(path)
    async with store.exclusive():
        assert await store.save(_credentials('revision-1'), expected_revision=None)
    original = path.read_bytes()

    def fail_replace(source: str | bytes | os.PathLike[str] | os.PathLike[bytes], destination: object) -> None:
        raise OSError('simulated replacement failure')

    monkeypatch.setattr('pydantic_ai.auth._codex_store.os.replace', fail_replace)
    async with store.exclusive():
        with pytest.raises(CodexCredentialsError, match='Unable to write'):
            await store.save(_credentials('revision-2'), expected_revision='revision-1')

    assert path.read_bytes() == original
    assert list(tmp_path.glob('.auth.json.*.tmp')) == []

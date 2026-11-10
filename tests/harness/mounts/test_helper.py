"""Tests for the mount helper that runs as root in a sandbox, with rclone, the kernel's mount table and processes faked.

Unit tests rather than integration tests: the helper needs root, FUSE and rclone, which only a sandbox has. The Docker
live test (`tests/harness/docker_sandbox/test_mounts_live.py`) runs it for real.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai_harness._mounts import _helper as helper

SECRET = 'the-secret-key'
START_DAEMON = helper.start_daemon
"""The real one, which the `sandbox` fixture replaces."""


def returns(value: object) -> Callable[..., Any]:
    """A stand-in for any function, returning `value`."""

    def fake(*args: Any, **kwargs: Any) -> Any:
        return value

    return fake


def exits(code: int, stdout: str = '', stderr: str = '') -> Callable[..., subprocess.CompletedProcess[str]]:
    """A stand-in for `subprocess.run` whose processes exit with `code`."""

    def run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(argv), code, stdout, stderr)

    return run


class Sandbox:
    """The rclone daemon, the kernel's mount table and the processes the helper starts, in memory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.kernel: dict[str, str] = {}
        """Mount point to mount source."""
        self.daemon: dict[str, str] = {}
        """Mount point to the rclone file system the daemon serves there."""
        self.dead: set[str] = set()
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.processes: list[list[str]] = []
        self.fail: dict[str, str] = {}
        """rclone command to the error it fails with."""
        self.queued = 0
        """Uploads `vfs/stats` reports as queued, each call taking one off."""
        self.mount_answers = True

    def rc(self, command: str, body: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        body = body or {}
        self.calls.append((command, body))
        if command in self.fail:
            raise helper.MountFailed('rclone', self.fail[command])
        if command == 'mount/listmounts':
            return {'mountPoints': [{'MountPoint': path, 'Fs': fs} for path, fs in self.daemon.items()]}
        if command == 'mount/mount':
            path = body['mountPoint']
            self.daemon[path] = body['fs']
            self.dead.discard(path)
            if self.mount_answers:
                self.kernel[path] = body['mountOpt']['DeviceName']
        elif command == 'mount/unmount':
            self.daemon.pop(body['mountPoint'])
            self.kernel.pop(body['mountPoint'], None)
        elif command == 'vfs/stats':
            queued, self.queued = self.queued, max(self.queued - 1, 0)
            return {'diskCache': {'uploadsQueued': queued, 'uploadsInProgress': 0}}
        return {}

    def run(self, argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.processes.append(list(argv))
        if argv[0] == 'fusermount3':
            self.kernel.pop(argv[-1], None)
        return subprocess.CompletedProcess(argv, 0, '', '')

    def alive(self, path: str) -> bool:
        return path in self.kernel and path not in self.dead

    def mounted(self, path: str, fingerprint: str, *, served: bool = True) -> None:
        self.kernel[path] = f'pai-{fingerprint}'
        if served:
            self.daemon[path] = f'pai-{fingerprint}:bucket'

    def commands(self) -> list[str]:
        return [command for command, _ in self.calls if command != 'mount/listmounts']


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Sandbox:
    sandbox = Sandbox(tmp_path)
    state = tmp_path / 'state'
    state.mkdir()
    (tmp_path / 'workspace').mkdir()
    monkeypatch.setattr(helper, 'STATE', str(state))
    monkeypatch.setattr(helper, 'SOCKET', str(state / 'rc.sock'))
    monkeypatch.setattr(helper, 'CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(helper, 'rc', sandbox.rc)
    monkeypatch.setattr(helper, 'kernel_mounts', lambda: dict(sandbox.kernel))
    monkeypatch.setattr(helper, 'alive', sandbox.alive)
    monkeypatch.setattr(helper, 'install_fuse', lambda: None)
    monkeypatch.setattr(helper, 'install_rclone', lambda: '/usr/bin/rclone')
    monkeypatch.setattr(helper, 'start_daemon', returns(None))
    monkeypatch.setattr(helper.subprocess, 'run', sandbox.run)
    monkeypatch.setattr(helper.time, 'sleep', returns(None))
    monkeypatch.setattr(helper.time, 'time', lambda: 1_000.0)
    return sandbox


def wanted(sandbox: Sandbox, name: str = 'data', fingerprint: str = 'aaaa', **changes: Any) -> dict[str, Any]:
    return {
        'path': str(sandbox.root / 'workspace' / name),
        'fingerprint': fingerprint,
        'remote': {'type': 's3', 'provider': 'Other', 'endpoint': 'http://minio:9000'},
        'root': 'bucket/prefix',
        'read_only': False,
        'keys': {'access_key_id': 'the-key-id', 'secret_access_key': SECRET, 'session_token': ''},
        'expires_at': 2_000.0,
    } | changes


def ensure(sandbox: Sandbox, *mounts: dict[str, Any]) -> dict[str, Any]:
    return helper.ensure({'working_dir': str(sandbox.root / 'workspace'), 'mounts': list(mounts)})


def expiry(sandbox: Sandbox, fingerprint: str) -> str:
    return (sandbox.root / 'state' / f'{fingerprint}.expires').read_text()


def test_a_new_folder_is_made_and_mounted(sandbox: Sandbox) -> None:
    mount = wanted(sandbox, 'deep/data')
    path = mount['path']

    assert ensure(sandbox, mount) == {'recovered': []}

    owner = os.stat(sandbox.root / 'workspace')
    assert sandbox.calls[1:] == snapshot(
        [
            ('config/delete', {'name': 'pai-aaaa'}),
            (
                'config/create',
                {
                    'name': 'pai-aaaa',
                    'type': 's3',
                    'parameters': {
                        'provider': 'Other',
                        'endpoint': 'http://minio:9000',
                        'access_key_id': 'the-key-id',
                        'secret_access_key': 'the-secret-key',
                        'session_token': '',
                    },
                },
            ),
            (
                'mount/mount',
                {
                    'fs': 'pai-aaaa:bucket/prefix',
                    'mountPoint': path,
                    'mountOpt': {'AllowOther': True, 'AllowNonEmpty': True, 'DeviceName': 'pai-aaaa'},
                    'vfsOpt': {
                        'CacheMode': 'writes',
                        'DirCacheTime': 10000000000,
                        'Links': True,
                        'ReadOnly': False,
                        'UID': owner.st_uid,
                        'GID': owner.st_gid,
                        'FilePerms': 493,
                        'DirPerms': 493,
                    },
                },
            ),
        ]
    )
    assert sandbox.kernel == {path: 'pai-aaaa'}
    assert os.listdir(path) == [helper.MARKER]
    assert expiry(sandbox, 'aaaa') == '2000.0'


def test_the_same_bucket_only_gets_fresh_keys(sandbox: Sandbox) -> None:
    mount = wanted(sandbox, expires_at=3_000.0)
    sandbox.mounted(mount['path'], 'aaaa')

    ensure(sandbox, mount)

    assert sandbox.commands() == ['backend/command']
    assert sandbox.calls[-1] == (
        'backend/command',
        {'command': 'set', 'fs': 'pai-aaaa:bucket', 'opt': mount['keys']},
    )
    assert expiry(sandbox, 'aaaa') == '3000.0'


def test_another_bucket_is_refused_while_the_keys_there_are_valid(sandbox: Sandbox) -> None:
    mount = wanted(sandbox, fingerprint='bbbb')
    sandbox.mounted(mount['path'], 'aaaa')
    helper.record_expiry('aaaa', 1_500.0)

    with pytest.raises(helper.MountFailed, match='already shows another bucket in this sandbox') as raised:
        ensure(sandbox, mount)

    assert raised.value.code == 'conflict'
    assert sandbox.commands() == []


@pytest.mark.parametrize('recorded', ['500.0', 'garbage', None])
def test_another_bucket_replaces_one_whose_keys_expired(sandbox: Sandbox, recorded: str | None) -> None:
    mount = wanted(sandbox, fingerprint='bbbb')
    sandbox.mounted(mount['path'], 'aaaa')
    if recorded is not None:
        (sandbox.root / 'state' / 'aaaa.expires').write_text(recorded)

    ensure(sandbox, mount)

    assert sandbox.commands() == ['vfs/stats', 'mount/unmount', 'config/delete', 'config/create', 'mount/mount']
    assert sandbox.kernel == {mount['path']: 'pai-bbbb'}


def test_a_dead_mount_is_replaced(sandbox: Sandbox) -> None:
    mount = wanted(sandbox)
    sandbox.mounted(mount['path'], 'aaaa')
    sandbox.dead.add(mount['path'])

    ensure(sandbox, mount)

    assert sandbox.commands() == [
        'backend/command',
        'vfs/stats',
        'mount/unmount',
        'config/delete',
        'config/create',
        'mount/mount',
    ]


def test_a_mount_whose_keys_expired_gets_fresh_ones_before_it_is_probed(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    mount = wanted(sandbox)
    sandbox.mounted(mount['path'], 'aaaa')
    # With expired keys, a lookup in the mount fails until fresh ones arrive.
    sandbox.dead.add(mount['path'])
    rc = sandbox.rc

    def fresh_keys_revive(command: str, body: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        if command == 'backend/command':
            sandbox.dead.discard(mount['path'])
        return rc(command, body, timeout)

    monkeypatch.setattr(helper, 'rc', fresh_keys_revive)

    ensure(sandbox, mount)

    assert sandbox.commands() == ['backend/command']


def test_a_mount_whose_daemon_is_gone_is_detached_and_replaced(sandbox: Sandbox) -> None:
    mount = wanted(sandbox)
    sandbox.mounted(mount['path'], 'aaaa', served=False)
    Path(mount['path']).mkdir()
    (Path(mount['path']) / helper.MARKER).write_bytes(b'')

    ensure(sandbox, mount)

    assert sandbox.processes == [['fusermount3', '-uz', mount['path']], ['umount', '-l', mount['path']]]
    assert sandbox.kernel == {mount['path']: 'pai-aaaa'}


def test_something_else_mounted_there_is_refused(sandbox: Sandbox) -> None:
    mount = wanted(sandbox)
    sandbox.kernel[mount['path']] = 'overlay'

    with pytest.raises(helper.MountFailed, match='already has something else mounted on it'):
        ensure(sandbox, mount)


def test_mounts_at_other_paths_are_left_alone(sandbox: Sandbox) -> None:
    other = str(sandbox.root / 'workspace' / 'other')
    sandbox.mounted(other, 'cccc')

    ensure(sandbox, wanted(sandbox))

    assert sandbox.kernel[other] == 'pai-cccc'
    assert 'mount/unmount' not in sandbox.commands()


def test_a_mount_the_kernel_lost_is_dropped_from_the_daemon_first(sandbox: Sandbox) -> None:
    mount = wanted(sandbox)
    sandbox.daemon[mount['path']] = 'pai-aaaa:bucket'
    Path(mount['path']).mkdir()
    (Path(mount['path']) / helper.MARKER).write_bytes(b'')

    ensure(sandbox, mount)

    assert sandbox.commands() == ['mount/unmount', 'config/delete', 'config/create', 'mount/mount']
    assert sandbox.kernel == {mount['path']: 'pai-aaaa'}


def test_files_written_while_unmounted_are_moved_aside(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.uuid, 'uuid4', lambda: type('UUID', (), {'hex': 'abcdef0123'})())
    monkeypatch.setattr(helper.time, 'gmtime', lambda: (2026, 11, 10, 12, 0, 0, 1, 314, 0))
    mount = wanted(sandbox)
    folder = Path(mount['path'])
    folder.mkdir()
    (folder / helper.MARKER).write_bytes(b'')
    (folder / 'stray.txt').write_text('written while unmounted')

    reply = ensure(sandbox, mount)

    recovered = sandbox.root / 'workspace' / helper.RECOVERED / '20261110T120000Z-abcdef-data'
    assert reply == {'recovered': [str(recovered)]}
    assert (recovered / 'stray.txt').read_text() == 'written while unmounted'
    assert os.listdir(folder) == [helper.MARKER]


def test_a_symlink_where_recovered_files_go_is_refused(sandbox: Sandbox) -> None:
    mount = wanted(sandbox)
    folder = Path(mount['path'])
    folder.mkdir()
    (folder / helper.MARKER).write_bytes(b'')
    (folder / 'stray.txt').write_text('')
    (sandbox.root / 'workspace' / helper.RECOVERED).symlink_to(sandbox.root)

    with pytest.raises(helper.MountFailed, match=re.escape('is a symlink. Remove it')):
        ensure(sandbox, mount)


def test_a_link_put_where_the_marker_goes_is_not_followed(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    mount = wanted(sandbox)
    Path(mount['path']).mkdir()
    target = sandbox.root / 'precious'
    target.write_text('keep')
    (Path(mount['path']) / helper.MARKER).symlink_to(target)
    # As if the link appeared just after the folder was listed empty.
    monkeypatch.setattr(helper.os, 'listdir', returns([]))

    with pytest.raises(helper.MountFailed, match='already holds files'):
        ensure(sandbox, mount)

    assert target.read_text() == 'keep'


def test_a_folder_with_other_files_is_refused(sandbox: Sandbox) -> None:
    mount = wanted(sandbox)
    Path(mount['path']).mkdir()
    (Path(mount['path']) / 'mine.txt').write_text('')

    with pytest.raises(helper.MountFailed, match=re.escape('already holds files. Mount at a new or empty folder.')):
        ensure(sandbox, mount)


def test_a_file_at_the_path_is_refused(sandbox: Sandbox) -> None:
    mount = wanted(sandbox)
    Path(mount['path']).write_text('', encoding='utf-8')

    with pytest.raises(helper.MountFailed, match=re.escape('is a file. Mount at a folder.')):
        ensure(sandbox, mount)


def test_a_path_through_a_symlink_is_refused(sandbox: Sandbox) -> None:
    (sandbox.root / 'elsewhere').mkdir()
    (sandbox.root / 'workspace' / 'link').symlink_to(sandbox.root / 'elsewhere')

    with pytest.raises(helper.MountFailed, match='leads through a symlink'):
        ensure(sandbox, wanted(sandbox, 'link/data'))


@pytest.mark.parametrize('fresh', [True, False])
def test_a_failed_mount_leaves_no_marker_it_wrote(sandbox: Sandbox, fresh: bool) -> None:
    mount = wanted(sandbox)
    if not fresh:
        Path(mount['path']).mkdir()
        (Path(mount['path']) / helper.MARKER).write_bytes(b'')
    sandbox.fail['mount/mount'] = 'rclone mount/mount failed: no bucket'

    with pytest.raises(helper.MountFailed, match='no bucket'):
        ensure(sandbox, mount)

    assert os.listdir(mount['path']) == ([] if fresh else [helper.MARKER])


def test_a_mount_that_does_not_answer_is_reported(sandbox: Sandbox) -> None:
    sandbox.mount_answers = False

    with pytest.raises(helper.MountFailed, match='did not answer right after it was mounted'):
        ensure(sandbox, wanted(sandbox))


def test_an_unmount_waits_for_queued_uploads(sandbox: Sandbox) -> None:
    mount = wanted(sandbox)
    sandbox.mounted(mount['path'], 'aaaa')
    sandbox.dead.add(mount['path'])
    sandbox.queued = 3

    ensure(sandbox, mount)

    assert sandbox.commands().count('vfs/stats') == 4


def test_an_unmount_with_uploads_that_never_finish_is_refused(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helper, 'UPLOAD_WAIT_SECONDS', 1)
    mount = wanted(sandbox)
    sandbox.mounted(mount['path'], 'aaaa')
    sandbox.dead.add(mount['path'])
    sandbox.queued = 100

    with pytest.raises(helper.MountFailed, match='still has files uploading') as raised:
        ensure(sandbox, mount)
    assert raised.value.code == 'busy'


def test_a_mount_that_will_not_unmount_is_reported(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    mount = wanted(sandbox)
    sandbox.mounted(mount['path'], 'aaaa', served=False)
    monkeypatch.setattr(helper.subprocess, 'run', exits(1))

    with pytest.raises(helper.MountFailed, match=re.escape('could not be unmounted. Restart the sandbox')):
        ensure(sandbox, mount)


def test_no_key_reaches_a_command_line(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[list[str]] = []
    monkeypatch.setattr(helper.subprocess, 'Popen', lambda argv, **kwargs: started.append(list(argv)))  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
    monkeypatch.setattr(helper, 'start_daemon', START_DAEMON)
    answers = iter([OSError('no daemon yet')])

    def rc(command: str, body: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        if command == 'rc/noop' and (error := next(answers, None)):
            raise error
        return sandbox.rc(command, body, timeout)

    monkeypatch.setattr(helper, 'rc', rc)
    mount = wanted(sandbox)
    sandbox.mounted(mount['path'], 'bbbb', served=False)
    sandbox.dead.add(mount['path'])

    ensure(sandbox, mount)

    assert started and sandbox.processes
    assert not any(SECRET in arg for argv in [*started, *sandbox.processes] for arg in argv)


def test_the_daemon_ignores_rclone_settings_in_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[dict[str, str]] = []
    answers = iter([OSError('no daemon yet')])

    def rc(command: str, body: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        if (error := next(answers, None)) is not None:
            raise error
        return {}

    def popen(argv: Sequence[str], *, env: dict[str, str], **kwargs: Any) -> None:
        started.append(env)

    monkeypatch.setattr(helper, 'STATE', str(tmp_path))
    monkeypatch.setattr(helper, 'SOCKET', str(tmp_path / 'rc.sock'))
    monkeypatch.setattr(helper, 'CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(helper, 'rc', rc)
    monkeypatch.setattr(helper.subprocess, 'Popen', popen)
    monkeypatch.setenv('RCLONE_LOG_LEVEL', 'DEBUG')
    monkeypatch.setenv('KEEP_ME', 'yes')

    START_DAEMON('/usr/bin/rclone')

    assert [('RCLONE_LOG_LEVEL' in env, env.get('KEEP_ME')) for env in started] == [(False, 'yes')]


def test_the_daemon_already_running_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper, 'rc', returns({}))
    monkeypatch.setattr(helper.subprocess, 'Popen', None)

    START_DAEMON('/usr/bin/rclone')


def test_a_daemon_that_never_answers_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def rc(command: str, body: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        raise helper.MountFailed('rclone', 'no')

    monkeypatch.setattr(helper, 'STATE', str(tmp_path))
    monkeypatch.setattr(helper, 'SOCKET', str(tmp_path / 'rc.sock'))
    monkeypatch.setattr(helper, 'CACHE', str(tmp_path / 'cache'))
    # One attempt fits before the deadline.
    ticks_end = helper.DAEMON_START_SECONDS + 1.0
    ticks = iter([0.0, 0.0])
    monkeypatch.setattr(helper.time, 'monotonic', lambda: next(ticks, ticks_end))
    monkeypatch.setattr(helper.time, 'sleep', returns(None))
    monkeypatch.setattr(helper, 'rc', rc)
    monkeypatch.setattr(helper.subprocess, 'Popen', returns(None))
    (tmp_path / 'rc.sock').write_text('')

    with pytest.raises(helper.MountFailed, match='The rclone daemon did not start'):
        START_DAEMON('/usr/bin/rclone')
    assert not (tmp_path / 'rc.sock').exists()


def test_served_is_empty_without_a_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    def rc(command: str, body: dict[str, Any] | None = None, timeout: float = 60) -> dict[str, Any]:
        raise OSError('no socket')

    monkeypatch.setattr(helper, 'rc', rc)

    assert helper.served() == {}


class FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body

    def read(self) -> bytes:
        return self.body


@pytest.mark.parametrize(
    ('status', 'body', 'expected'),
    [
        (200, b'{"ok": 1}', {'ok': 1}),
        (200, b'', {}),
        (500, b'{"error": "bad", "input": {"secret_access_key": "the-secret-key"}}', 'rclone mount/mount failed: bad.'),
        (404, b'{}', 'rclone mount/mount failed: 404.'),
    ],
)
def test_rc_reports_only_rclones_own_message(
    monkeypatch: pytest.MonkeyPatch, status: int, body: bytes, expected: dict[str, Any] | str
) -> None:
    requests: list[tuple[Any, ...]] = []

    class Connection:
        def __init__(self, host: str, timeout: float) -> None:
            pass

        def request(self, *args: Any) -> None:
            requests.append(args)

        def getresponse(self) -> FakeResponse:
            return FakeResponse(status, body)

    monkeypatch.setattr(helper, '_UnixConnection', Connection)

    if isinstance(expected, dict):
        assert helper.rc('mount/mount', {'a': 1}) == expected
    else:
        with pytest.raises(helper.MountFailed) as raised:
            helper.rc('mount/mount')
        assert str(raised.value).startswith(expected)
        assert SECRET not in str(raised.value)
    assert requests[0][:2] == ('POST', '/mount/mount')


@pytest.mark.parametrize(
    ('output', 'expected'),
    [
        ('rclone v1.75.2\n- os/version: debian', True),
        ('rclone v1.76.0', True),
        ('rclone v2.0.0', True),
        ('rclone v1.75.1', False),
        ('rclone v1.60.1-DEV', False),
        ('something else', False),
    ],
)
def test_an_rclone_on_path_is_used_from_the_version_floor_up(
    monkeypatch: pytest.MonkeyPatch, output: str, expected: bool
) -> None:
    monkeypatch.setattr(helper.subprocess, 'run', exits(0, output))

    assert helper.new_enough('/usr/bin/rclone') is expected


@pytest.mark.parametrize('error', [OSError('missing'), subprocess.TimeoutExpired('rclone', 10)])
def test_an_rclone_that_does_not_run_is_too_old(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    def run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise error

    monkeypatch.setattr(helper.subprocess, 'run', run)

    assert helper.new_enough('/usr/bin/rclone') is False


@pytest.fixture
def pinned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'lib' / 'rclone'
    monkeypatch.setattr(helper, 'RCLONE', str(path))
    monkeypatch.setattr(helper.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(helper.shutil, 'which', returns(None))
    return path


def archive(binary: bytes) -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as zipped:
        zipped.writestr(f'rclone-{helper.RCLONE_VERSION}-linux-amd64/rclone', binary)
    return data.getvalue()


def test_install_uses_a_new_enough_rclone_on_path(pinned: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.shutil, 'which', returns('/usr/bin/rclone'))
    monkeypatch.setattr(helper, 'new_enough', returns(True))

    assert helper.install_rclone() == '/usr/bin/rclone'


def test_install_uses_the_pinned_build_once_downloaded(pinned: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.shutil, 'which', returns('/usr/bin/rclone'))
    monkeypatch.setattr(helper, 'new_enough', returns(False))
    pinned.parent.mkdir()
    pinned.write_bytes(b'')
    pinned.chmod(0o755)

    assert helper.install_rclone() == str(pinned)


def test_install_downloads_and_checks_the_pinned_build(pinned: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    zipped = archive(b'the binary')
    monkeypatch.setitem(helper.RCLONE_SHA256, 'x86_64', hashlib.sha256(zipped).hexdigest())
    requests: list[Any] = []

    def urlopen(request: Any, timeout: float) -> io.BytesIO:
        requests.append(request)
        return io.BytesIO(zipped)

    monkeypatch.setattr(helper.urllib.request, 'urlopen', urlopen)

    assert helper.install_rclone() == str(pinned)

    assert pinned.read_bytes() == b'the binary'
    assert os.access(pinned, os.X_OK)
    assert requests[0].full_url == snapshot('https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-amd64.zip')
    assert requests[0].get_header('User-agent') == 'pydantic-ai-harness'


def test_a_download_that_does_not_match_its_checksum_is_refused(pinned: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.urllib.request, 'urlopen', returns(io.BytesIO(archive(b'tampered'))))

    with pytest.raises(helper.MountFailed, match='does not match its published checksum'):
        helper.install_rclone()
    assert not pinned.exists()


def test_a_failed_download_is_reported(pinned: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def urlopen(request: Any, timeout: float) -> io.BytesIO:
        raise OSError('network unreachable')

    monkeypatch.setattr(helper.urllib.request, 'urlopen', urlopen)

    with pytest.raises(helper.MountFailed, match=r'Could not download rclone \(network unreachable\)'):
        helper.install_rclone()


def test_an_architecture_without_a_build_is_refused(pinned: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(helper.platform, 'machine', lambda: 'riscv64')

    with pytest.raises(helper.MountFailed, match='There is no rclone build for riscv64'):
        helper.install_rclone()


@pytest.mark.parametrize(
    ('found', 'apt', 'installed', 'expected'),
    [
        (True, False, False, []),
        (False, True, True, [['apt-get', 'update', '-q'], ['apt-get', 'install', '-y', '-q', 'fuse3']]),
        (False, False, False, None),
    ],
)
def test_fuse_is_installed_with_apt_when_missing(
    monkeypatch: pytest.MonkeyPatch, found: bool, apt: bool, installed: bool, expected: list[list[str]] | None
) -> None:
    ran: list[list[str]] = []

    def which(name: str) -> str | None:
        present = {'fusermount3': found or (installed and bool(ran)), 'apt-get': apt}
        return name if present.get(name) else None

    def run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        ran.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0)

    monkeypatch.setattr(helper.shutil, 'which', which)
    monkeypatch.setattr(helper.subprocess, 'run', run)

    if expected is None:
        with pytest.raises(helper.MountFailed, match='The sandbox has no `fusermount3`'):
            helper.install_fuse()
    else:
        helper.install_fuse()
        assert ran == expected


def test_the_kernel_table_is_read_with_its_escapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    table = tmp_path / 'mountinfo'
    table.write_bytes(
        b'22 1 0:21 / / rw,relatime - overlay overlay rw\n'
        b'40 22 0:40 / /workspace/my\\040data rw,nosuid - fuse.rclone pai-aaaa rw,user_id=0\n'
        b'41 40 0:41 / /workspace/my\\040data rw,nosuid - fuse.rclone pai-bbbb rw,user_id=0\n'
    )
    monkeypatch.setattr(helper, 'MOUNTINFO', str(table))

    assert helper.kernel_mounts() == {'/': 'overlay', '/workspace/my data': 'pai-bbbb'}


@pytest.mark.parametrize(
    ('outcome', 'expected'),
    [
        (subprocess.CompletedProcess(['stat'], 0, '', ''), True),
        (subprocess.CompletedProcess(['stat'], 1, '', "stat: cannot statx 'x': No such file or directory"), True),
        (
            subprocess.CompletedProcess(['stat'], 1, '', "stat: cannot statx 'x': Transport endpoint is not connected"),
            False,
        ),
        (subprocess.TimeoutExpired('stat', 10), False),
    ],
)
def test_a_mount_is_alive_when_a_lookup_in_it_answers(
    monkeypatch: pytest.MonkeyPatch, outcome: subprocess.CompletedProcess[str] | Exception, expected: bool
) -> None:
    def run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(helper.subprocess, 'run', run)

    assert helper.alive('/workspace/data') is expected


def test_main_answers_with_one_json_line(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    alarms: list[int] = []
    monkeypatch.setattr(helper.signal, 'alarm', alarms.append)
    monkeypatch.setattr(helper.signal, 'signal', returns(None))
    request = {'working_dir': str(sandbox.root / 'workspace'), 'mounts': [wanted(sandbox)]}
    monkeypatch.setattr(sys, 'stdin', io.StringIO(f'{json.dumps(request)}\n'))

    helper.main()

    assert json.loads(capsys.readouterr().out) == {'recovered': []}
    assert alarms == [helper.DEADLINE_SECONDS]
    assert (sandbox.root / 'state' / 'lock').exists()


def test_main_reports_a_refusal_as_json(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(helper.signal, 'alarm', returns(None))
    monkeypatch.setattr(helper.signal, 'signal', returns(None))
    mount = wanted(sandbox)
    sandbox.kernel[mount['path']] = 'overlay'
    request = {'working_dir': str(sandbox.root / 'workspace'), 'mounts': [mount]}
    monkeypatch.setattr(sys, 'stdin', io.StringIO(f'{json.dumps(request)}\n'))

    helper.main()

    assert json.loads(capsys.readouterr().out) == {
        'error': 'conflict',
        'message': f'{mount["path"]} already has something else mounted on it. Mount at another folder.',
    }


def test_a_state_folder_made_by_another_user_is_refused(
    sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(helper.signal, 'alarm', returns(None))
    monkeypatch.setattr(helper.signal, 'signal', returns(None))
    monkeypatch.setattr(helper.os, 'geteuid', returns(os.geteuid() + 1))
    request = {'working_dir': str(sandbox.root / 'workspace'), 'mounts': [wanted(sandbox)]}
    monkeypatch.setattr(sys, 'stdin', io.StringIO(f'{json.dumps(request)}\n'))

    helper.main()

    assert json.loads(capsys.readouterr().out) == {
        'error': 'conflict',
        'message': f'{helper.STATE} was not made by root. Remove it, or restart the sandbox.',
    }
    assert not (sandbox.root / 'state' / 'lock').exists()


def test_a_state_folder_that_is_a_link_is_refused(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    elsewhere = sandbox.root / 'elsewhere'
    elsewhere.mkdir()
    link = sandbox.root / 'linked-state'
    link.symlink_to(elsewhere)
    monkeypatch.setattr(helper, 'STATE', str(link))

    with pytest.raises(helper.MountFailed, match='was not made by root'):
        helper.private_state()


def test_the_state_folder_is_made_private(sandbox: Sandbox) -> None:
    state = sandbox.root / 'state'
    state.chmod(0o755)

    helper.private_state()

    assert state.stat().st_mode & 0o777 == 0o700


def test_running_out_of_time_is_a_refusal() -> None:
    with pytest.raises(helper.MountFailed, match='Mounting took longer than 1800 seconds') as raised:
        helper._out_of_time(14, None)  # pyright: ignore[reportPrivateUsage]
    assert raised.value.code == 'timeout'

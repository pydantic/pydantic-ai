"""The mount helper's rules, run here with the kernel and rclone replaced by a table of what is mounted.

`integration_tests/cloud_storage` runs the helper for real, as root in a container; this covers the cases that are slow
or awkward to set up there: several runs sharing a sandbox, a mount failing halfway, a token about to end.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai_harness._mounts import _helper


class Sandbox:
    """What is mounted where, standing in for the kernel's mount table and the rclone daemon."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.mounted: dict[str, str] = {}
        self.mounts: list[str] = []
        self.unmounts: list[str] = []
        self.fail_mount = False
        state = tmp_path / 'state'
        state.mkdir()
        monkeypatch.setattr(_helper, 'STATE', str(state))
        monkeypatch.setattr(_helper, 'LEASE', str(state / 'lease.json'))
        monkeypatch.setattr(_helper, 'kernel_mounts', lambda: dict(self.mounted))
        monkeypatch.setattr(_helper, 'served', lambda: {path: f'{source}:' for path, source in self.mounted.items()})
        monkeypatch.setattr(_helper, 'alive', _alive)
        for name in ('install_fuse', 'install_rclone', 'start_daemon', 'rc'):
            monkeypatch.setattr(_helper, name, _answer)
        monkeypatch.setattr(_helper, 'unmount', self._unmount)
        monkeypatch.setattr(_helper, 'mount', self._mount)
        self.working_dir = tmp_path / 'w'
        self.working_dir.mkdir()

    def _unmount(self, path: str) -> None:
        self.unmounts.append(path)
        del self.mounted[path]
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(path, _helper.MARKER))

    def _mount(self, request: dict[str, Any], keys: dict[str, str], owner: os.stat_result) -> None:
        if self.fail_mount:
            raise _helper.MountFailed('mount', 'It did not answer.')
        self.mounts.append(request['path'])
        self.mounted[request['path']] = f'pai-{request["fingerprint"]}'
        with open(f'{_helper.STATE}/{request["fingerprint"]}.expires', 'w', encoding='utf-8') as file:
            file.write(str(request.get('keys_expire') or ''))

    def ensure(self, owner: str, *folders: str, sas_expires: float | None = None) -> dict[str, Any]:
        mounts = [
            {
                'path': str(self.working_dir / folder),
                'fingerprint': folder,
                'remote': {'type': 'azureblob' if sas_expires else 's3'},
                'root': folder,
                'read_only': False,
                'keys_expire': sas_expires,
            }
            for folder in folders
        ]
        keys = {folder: {'sas_url': 'https://a/c?sig=x'} if sas_expires else {} for folder in folders}
        request = {'mounts': mounts, 'keys': keys, 'working_dir': str(self.working_dir), 'owner': owner}
        return _helper.ensure({**request, 'lease_seconds': 900})

    def lease(self) -> dict[str, Any]:
        with open(_helper.LEASE, encoding='utf-8') as file:
            return json.load(file)


def _answer(*args: Any, **kwargs: Any) -> Any:
    """What installing and asking the daemon return: nothing worth checking."""
    return {}


def _alive(path: str) -> bool:
    return True


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Sandbox:
    return Sandbox(monkeypatch, tmp_path)


def test_a_run_with_nothing_to_mount_removes_what_no_run_holds(sandbox: Sandbox) -> None:
    sandbox.ensure('earlier', 'old')
    _helper.release('earlier')

    sandbox.ensure('next')

    assert sandbox.unmounts == [str(sandbox.working_dir / 'old')]
    # Nor does it hold the sandbox, so a run after it mounts what it likes.
    assert sandbox.lease() == {'set': [], 'holders': {}}
    sandbox.ensure('after', 'data')
    assert sandbox.mounted == {str(sandbox.working_dir / 'data'): 'pai-data'}


def test_a_concurrent_run_with_other_mounts_is_refused_until_the_holders_release(sandbox: Sandbox) -> None:
    sandbox.ensure('one', 'a')

    with pytest.raises(_helper.MountFailed, match='Another run is using this sandbox with different mounts'):
        sandbox.ensure('two', 'b')
    # The same mounts are shared, and a run with nothing to mount, or another workspace of a holder, leaves them.
    sandbox.ensure('two', 'a')
    sandbox.ensure('three')
    sandbox.ensure('one')
    assert sandbox.unmounts == []

    _helper.release('one')
    with pytest.raises(_helper.MountFailed, match='Another run'):
        sandbox.ensure('three', 'b')
    _helper.release('two')
    sandbox.ensure('three', 'b')
    assert list(sandbox.mounted) == [str(sandbox.working_dir / 'b')]


def test_an_expired_lease_frees_the_sandbox(sandbox: Sandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox.ensure('crashed', 'a')
    later = time.time() + 901
    monkeypatch.setattr(_helper.time, 'time', lambda: later)

    sandbox.ensure('next', 'b')

    assert list(sandbox.lease()['holders']) == ['next']


def test_a_failed_mount_holds_nothing_and_leaves_no_marker(sandbox: Sandbox) -> None:
    sandbox.fail_mount = True
    with pytest.raises(_helper.MountFailed, match='did not answer'):
        sandbox.ensure('one', 'data')

    assert os.listdir(sandbox.working_dir / 'data') == []
    sandbox.fail_mount = False
    sandbox.ensure('two', 'other')


def test_a_folder_with_files_is_refused_and_a_lost_mounts_files_are_moved_aside(sandbox: Sandbox) -> None:
    (sandbox.working_dir / 'mine').mkdir()
    (sandbox.working_dir / 'mine' / 'notes.txt').write_text('kept')
    with pytest.raises(_helper.MountFailed, match=r'already holds files\. Mount at a new or empty folder\.$'):
        sandbox.ensure('one', 'mine')
    assert (sandbox.working_dir / 'mine' / 'notes.txt').read_text() == 'kept'

    sandbox.ensure('one', 'data')
    # As after a restart: the mount is gone, its marker shows, and something wrote into the bare folder.
    sandbox.mounted.clear()
    (sandbox.working_dir / 'data' / 'lost.txt').write_text('lost')
    (moved,) = sandbox.ensure('one', 'data')['recovered']

    assert Path(moved, 'lost.txt').read_text(encoding='utf-8') == 'lost'
    assert Path(moved).parent.name == '.pydantic-ai-recovered'
    assert os.listdir(sandbox.working_dir / 'data') == [_helper.MARKER]


def test_an_azure_mount_is_mounted_again_only_when_its_token_ends_before_the_next_renewal(sandbox: Sandbox) -> None:
    # Mounted with a token that ends within the lease, so the next renewal mounts it with the new one.
    sandbox.ensure('one', 'az', sas_expires=time.time() + 600)
    sandbox.ensure('one', 'az', sas_expires=time.time() + 3600)
    assert sandbox.unmounts == [str(sandbox.working_dir / 'az')]

    # That token outlasts the lease, so renewing leaves the mount, and the commands using it, alone.
    sandbox.ensure('one', 'az', sas_expires=time.time() + 3600)
    assert len(sandbox.unmounts) == 1

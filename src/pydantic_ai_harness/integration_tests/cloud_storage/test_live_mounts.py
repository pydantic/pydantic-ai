"""Buckets mounted with rclone in a Docker sandbox, so commands see them, on S3 and on Azure Blob Storage.

What only a real mount answers:

- that commands and file tools see the same files, and what commands write reaches the bucket;
- that the editing commands agents run (append, `sed -i`, folder renames, symlinks) work on a mount;
- that a read-only mount refuses writes from commands too;
- that files written into the bare folder while nothing was mounted are moved aside, not hidden, and a folder that
  already held files is refused;
- that a mount whose rclone died is mounted again, and a folder something else is mounted on is left alone;
- that a concurrent run with other mounts is refused until the first run releases the sandbox;
- that Azure mounts with a SAS token, and with a new one mounts again.

Needs Docker, and the servers of `conftest.py` in a container named `pai-s3` (or `CLOUD_STORAGE_S3_CONTAINER`), whose
network the sandbox shares. Run with `make integration-cloud-storage`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from pydantic_ai.workspaces import Mount, MountError, Workspace
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend

from .conftest import (
    AZURE_KEY,
    BUCKET,
    Filesystem,
    azure_bucket,
    azure_sas,
    create_azure_container,
    create_s3_bucket,
    s3_bucket,
    s3_keys,
    unavailable,
)

pytestmark = pytest.mark.anyio

SERVERS = os.environ.get('CLOUD_STORAGE_S3_CONTAINER', 'pai-s3')
IMAGE = 'pydantic-ai-mount-test'
FUSE_ARGS = ['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN', '--security-opt', 'apparmor:unconfined']


@pytest.fixture(scope='session')
def image() -> str:
    """The sandbox image, built once."""
    if shutil.which('docker') is None:
        unavailable('no `docker` executable')
    running = subprocess.run(['docker', 'inspect', '--format', '{{.State.Running}}', SERVERS], capture_output=True)
    if running.stdout.strip() != b'true':
        unavailable(f'no running container named {SERVERS!r} with the storage servers')
    subprocess.run(['docker', 'build', '-q', '-t', IMAGE, str(Path(__file__).parent / 'mount_image')], check=True)
    return IMAGE


@pytest.fixture
async def backend(image: str) -> AsyncIterator[DockerSandboxBackend]:
    """A sandbox with FUSE that reaches the storage servers on `127.0.0.1`, removed afterwards."""
    sandbox = DockerSandboxBackend(image=image, docker_args=[*FUSE_ARGS, '--network', f'container:{SERVERS}'])
    try:
        yield sandbox
    finally:
        if sandbox.ref is not None:
            await DockerSandbox().destroy(sandbox.ref)


async def _unmount(backend: DockerSandboxBackend) -> None:
    """As the next run in the sandbox would, with no mounts: the run that mounted releases them, and they go."""
    await backend._mounter.release()  # pyright: ignore[reportPrivateUsage]
    await backend.ensure_mounts([])


async def _as_root(backend: DockerSandboxBackend, command: str) -> None:
    assert backend.ref is not None
    subprocess.run(['docker', 'exec', '--user', '0', backend.ref.id, 'sh', '-c', command], check=True)


async def test_commands_and_file_tools_share_the_mount(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """Both reach the bucket through the one mount, and unmounting sends what is still queued."""
    bucket = new_bucket()
    await bucket.write_bytes('/seed.txt', b'from the bucket')
    workspace = Workspace(backend, mounts=[Mount('data', bucket, commands=True)])

    assert (await workspace.run(['cat', 'data/seed.txt'])).stdout == 'from the bucket'
    await workspace.write_text('data/notes/a.md', 'from file tools')
    shell = await workspace.run(
        'cd data && cat notes/a.md && echo one > log && echo two >> log && sed -i s/one/first/ log'
        ' && mkdir d1 && echo moved > d1/f && mv d1 d2 && ln -s log link && cat link d2/f',
        shell=True,
    )
    assert shell.stdout == 'from file toolsfirst\ntwo\nmoved\n'

    # Unmounting sends what is still queued, so the bucket has every file.
    await _unmount(backend)
    assert await bucket.read_bytes('/notes/a.md') == b'from file tools'
    assert await bucket.read_bytes('/log') == b'first\ntwo\n'
    assert await bucket.read_bytes('/d2/f') == b'moved\n'
    assert not await bucket.exists('/d1/f')


async def test_a_read_only_mount_refuses_commands_writes(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """The mount itself is read-only, so a write fails rather than vanishing."""
    bucket = new_bucket()
    await bucket.write_bytes('/a.txt', b'a')
    workspace = Workspace(backend, mounts=[Mount('data', bucket, read_only=True, commands=True)])

    result = await workspace.run('cat data/a.txt && echo b > data/a.txt', shell=True)

    assert (result.stdout, result.exit_code != 0) == ('a', True)
    assert await bucket.read_bytes('/a.txt') == b'a'


async def test_files_written_while_unmounted_are_moved_aside(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """A mount over them would hide them; moved, they stay where the agent can find them."""
    mount = Mount('data', new_bucket(), commands=True)
    working_dir = await backend.working_dir()
    await backend.ensure_mounts([mount])
    # As after a sandbox restart: the mount is gone, the folder is bare, and something writes into it.
    await _as_root(backend, f'pkill -9 -x rclone; umount -l {working_dir}/data')
    await backend.run('echo lost > data/lost.txt', shell=True)

    with pytest.warns(UserWarning, match='moved to /workspace/.pydantic-ai-recovered/'):
        await backend.ensure_mounts([mount])

    found = await backend.run('cat .pydantic-ai-recovered/*-data/lost.txt; ls data', shell=True)
    assert found.stdout == 'lost\n'


async def test_a_mount_whose_rclone_died_is_mounted_again(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """A dead mount answers every call with an error until it is replaced."""
    bucket = new_bucket()
    await bucket.write_bytes('/a.txt', b'a')
    mount = Mount('data', bucket, commands=True)
    await backend.ensure_mounts([mount])
    await _as_root(backend, 'pkill -9 -x rclone; while pgrep -x rclone > /dev/null; do sleep 0.1; done')
    assert (await backend.run(['cat', 'data/a.txt'])).exit_code != 0

    await backend.ensure_mounts([mount])

    assert (await backend.run(['cat', 'data/a.txt'])).stdout == 'a'


async def test_a_folder_with_something_else_mounted_is_left_alone(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """Never replace a mount this code did not make."""
    working_dir = await backend.working_dir()
    await _as_root(backend, f'mkdir -p {working_dir}/data && mount -t tmpfs tmpfs {working_dir}/data')

    with pytest.raises(MountError, match='already has something else mounted'):
        await backend.ensure_mounts([Mount('data', new_bucket(), commands=True)])


async def test_keys_that_never_expire_stay_out_of_the_sandbox(backend: DockerSandboxBackend) -> None:
    """A credentials function can only be checked once it answers, so the check is at mount time."""
    forever = {'access_key_id': 'any', 'secret_access_key': 'any', 'token': None, 'expires_at': None}
    create_s3_bucket()
    bucket = s3_bucket(f'{BUCKET}/{uuid.uuid4().hex}', credentials=lambda: forever)

    with pytest.raises(MountError, match='do not expire'):
        await backend.ensure_mounts([Mount('data', bucket, commands=True)])


async def test_mounts_only_file_tools_reach_need_nothing_in_the_sandbox(new_bucket: Callable[..., Filesystem]) -> None:
    """The image has neither Python nor rclone, and commands still run: nothing needs mounting."""
    sandbox = DockerSandboxBackend(image='busybox:1.37', docker_args=FUSE_ARGS)
    try:
        bucket = new_bucket()
        await bucket.write_bytes('/a.txt', b'from the bucket')
        workspace = Workspace(sandbox, mounts=[Mount('data', bucket)])

        assert (await workspace.run(['echo', 'ok'])).stdout == 'ok\n'
        assert await workspace.read_text('data/a.txt') == 'from the bucket'
    finally:
        if sandbox.ref is not None:
            await DockerSandbox().destroy(sandbox.ref)


async def test_a_mount_keeps_working_after_it_is_handed_fresh_keys(backend: DockerSandboxBackend) -> None:
    """Ensuring a healthy mount again hands it fresh keys, and it keeps working.

    SeaweedFS accepts any key, so this shows the hand-over works, not that the old keys stop.
    """
    create_s3_bucket()
    issued: list[str] = []

    def fresh_keys() -> Any:
        issued.append(f'key-{len(issued)}')
        return {**s3_keys(), 'access_key_id': issued[-1]}

    bucket = s3_bucket(f'{BUCKET}/{uuid.uuid4().hex}', credentials=fresh_keys)
    await backend.ensure_mounts([Mount('data', bucket, commands=True)])
    await backend.ensure_mounts([Mount('data', bucket, commands=True)])
    await backend.run('echo after > data/after.txt', shell=True)
    await _unmount(backend)

    # Each ensure fetched keys; a refresh the daemon refused would have raised `MountError` instead.
    assert issued[:2] == ['key-0', 'key-1']
    assert await bucket.read_bytes('/after.txt') == b'after\n'


async def test_a_mount_path_with_spaces_and_accents(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """The kernel escapes such paths in its mount table, which the helper must read back exactly."""
    bucket = new_bucket()
    await bucket.write_bytes('/a.txt', b'a')
    mount = Mount('données du jour', bucket, commands=True)

    await backend.ensure_mounts([mount])
    await backend.ensure_mounts([mount])

    assert (await backend.run(['cat', 'données du jour/a.txt'])).stdout == 'a'
    await _unmount(backend)


async def test_a_file_where_the_mount_folder_should_be_is_left_alone(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """Mounting there would hide the file."""
    await backend.run('echo mine > data', shell=True)

    with pytest.raises(MountError, match='is a file'):
        await backend.ensure_mounts([Mount('data', new_bucket(), commands=True)])


async def test_a_folder_that_already_holds_files_is_refused(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """Mounting there would hide them, and moving them aside would take them from where the agent left them."""
    await backend.run('mkdir -p data && echo mine > data/own.txt', shell=True)

    with pytest.raises(MountError, match='already holds files'):
        await backend.ensure_mounts([Mount('data', new_bucket(), commands=True)])

    assert (await backend.run(['cat', 'data/own.txt'])).stdout == 'mine\n'


async def test_a_concurrent_run_with_other_mounts_is_refused_until_the_first_releases(
    backend: DockerSandboxBackend, new_bucket: Callable[..., Filesystem]
) -> None:
    """Swapping the mounts under a running run would show its commands the other run's files."""
    first = Mount('first', new_bucket(), commands=True)
    await backend.ensure_mounts([first])
    assert backend.ref is not None
    other = DockerSandboxBackend(ref=backend.ref)

    with pytest.raises(MountError, match='Another run is using this sandbox with different mounts'):
        await other.ensure_mounts([Mount('second', new_bucket(), commands=True)])
    # The same mounts are shared, and a run with nothing to mount leaves them in place.
    await other.ensure_mounts([first])
    await DockerSandboxBackend(ref=backend.ref).ensure_mounts([])
    assert (await backend.run(['mountpoint', '-q', 'first'])).exit_code == 0

    await backend._mounter.release()  # pyright: ignore[reportPrivateUsage]
    await other._mounter.release()  # pyright: ignore[reportPrivateUsage]
    await other.ensure_mounts([Mount('second', new_bucket(), commands=True)])
    mounted = await backend.run('mountpoint -q second && ! mountpoint -q first', shell=True)
    assert mounted.exit_code == 0


async def test_azure_mounts_with_a_sas_token_and_mounts_again_with_a_new_one(backend: DockerSandboxBackend) -> None:
    """Rclone can't change an Azure mount's keys, so a new token means mounting again, and files stay put."""
    create_azure_container()
    issued: list[Any] = []

    def sas() -> Any:
        issued.append(azure_sas())
        return issued[-1]

    bucket = azure_bucket(f'{BUCKET}/{uuid.uuid4().hex}', credentials=sas)
    await backend.ensure_mounts([Mount('data', bucket, commands=True)])
    await backend.run('echo before > data/before.txt', shell=True)
    await backend.ensure_mounts([Mount('data', bucket, commands=True)])

    assert (await backend.run(['cat', 'data/before.txt'])).stdout == 'before\n'
    await _unmount(backend)
    assert len(issued) == 2
    assert await bucket.read_bytes('/before.txt') == b'before\n'


async def test_an_azure_account_key_mounts_only_when_allowed(backend: DockerSandboxBackend) -> None:
    """The account key reaches every container and never expires, so it is handed over only on request."""
    create_azure_container()
    folder = f'{BUCKET}/{uuid.uuid4().hex}'
    key = {'access_key': AZURE_KEY, 'expires_at': None}
    bucket = azure_bucket(folder, credentials=key, share_long_lived_credentials=True)
    await bucket.write_bytes('/a.txt', b'a')

    workspace = Workspace(backend, mounts=[Mount('data', bucket, commands=True)])

    assert (await workspace.run(['cat', 'data/a.txt'])).stdout == 'a'

"""`DockerSandbox(mounts=...)` against a real container and a real S3 server, MinIO.

Opt in with `PYDANTIC_AI_HARNESS_DOCKER_LIVE=1` on a machine with a Docker daemon and internet access, which the
helper uses to install fuse3 and download rclone in the container:
`PYDANTIC_AI_HARNESS_DOCKER_LIVE=1 uv run pytest -m docker_live tests/harness/docker_sandbox/test_mounts_live.py`.

MinIO no longer publishes community images, so this uses `pgsty/minio`, a build of the same server. The sandbox shares
its network, so the bucket is at `http://127.0.0.1:9000` in the sandbox. One sandbox serves every test, each at a
folder of its own, as the helper's install and download happen once per sandbox.
"""

from __future__ import annotations

import secrets
import subprocess
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from typing import TypeVar

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import Workspace, WorkspaceRef
from pydantic_ai_harness import MountError, S3Mount
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend

pytestmark = pytest.mark.docker_live

MINIO_IMAGE = 'pgsty/minio:RELEASE.2026-08-04T00-00-00Z'
SANDBOX_IMAGE = 'python:3.13-slim'
FUSE = ['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN', '--security-opt', 'apparmor:unconfined']
ACCESS_KEY_ID = 'pai-live'
SECRET = f'pai-live-{secrets.token_hex(8)}'
"""MinIO's root password, used as the mounts' keys: unique, so finding it anywhere is a leak."""
BUCKETS = ('one', 'two')

_T = TypeVar('_T')


def docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['docker', *args], capture_output=True, text=True, timeout=300, check=False)


@pytest.fixture(scope='module')
def minio() -> Iterator[str]:
    name = f'pai-mounts-minio-{secrets.token_hex(4)}'
    started = docker(
        'run', '--detach', '--name', name, '--env', f'MINIO_ROOT_USER={ACCESS_KEY_ID}',
        '--env', f'MINIO_ROOT_PASSWORD={SECRET}', MINIO_IMAGE, 'server', '/data',
    )  # fmt: skip
    assert started.returncode == 0, started.stderr
    try:
        for _ in range(60):
            if (
                docker(
                    'exec', name, 'mc', 'alias', 'set', 'local', 'http://127.0.0.1:9000', ACCESS_KEY_ID, SECRET
                ).returncode
                == 0
            ):
                break
            time.sleep(0.5)
        made = docker('exec', name, 'mc', 'mb', *(f'local/{bucket}' for bucket in BUCKETS))
        assert made.returncode == 0, made.stderr
        yield name
    finally:
        docker('rm', '--force', name)


@pytest.fixture(scope='module')
def sandbox(minio: str) -> Iterator[WorkspaceRef]:
    backend = DockerSandboxBackend(image=SANDBOX_IMAGE, docker_args=[*FUSE, '--network', f'container:{minio}'])
    anyio.run(backend.working_dir)
    assert backend.ref is not None
    try:
        yield backend.ref
    finally:
        docker('rm', '--force', backend.ref.id)


def s3(path: str, bucket: str = 'one', *, expires_in: timedelta = timedelta(hours=1)) -> S3Mount:
    return S3Mount(
        path=path,
        url=f's3://{bucket}/{path}/',
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET,
        expires_at=datetime.now(UTC) + expires_in,
        endpoint='http://127.0.0.1:9000',
        region='us-east-1',
    )


async def in_run(ref: WorkspaceRef, mounts: Sequence[S3Mount], action: Callable[[Workspace], Awaitable[_T]]) -> _T:
    """Run an agent on the sandbox with `mounts`, whose one tool call does `action`."""
    agent = Agent(TestModel(call_tools=['act']), capabilities=[DockerSandbox(mounts=mounts)])
    results: list[_T] = []

    @agent.tool
    async def act(ctx: RunContext[object]) -> str:
        results.append(await action(ctx.workspace))
        return 'done'

    await agent.run('go', workspace=ref)
    return results[0]


def as_root(ref: WorkspaceRef, script: str) -> str:
    result = docker('exec', '--user', '0', ref.id, 'sh', '-c', script)
    assert result.returncode == 0, result.stderr
    return result.stdout


def mount_of(ref: WorkspaceRef, path: str) -> tuple[str, str] | None:
    """The kernel's mount ID and mount source at `path`, or `None`."""
    for line in as_root(ref, 'cat /proc/self/mountinfo').splitlines():
        fields, _, rest = line.partition(' - ')
        if fields.split()[4] == path:
            return fields.split()[0], rest.split()[1]
    return None


async def eventually(check: Callable[[], bool]) -> None:
    """Wait for `check`: rclone uploads a closed file a few seconds later."""
    for _ in range(60):
        if check():
            return
        await anyio.sleep(0.5)
    pytest.fail('the condition never held')


async def test_commands_and_file_tools_see_the_bucket(minio: str, sandbox: WorkspaceRef) -> None:
    async def action(workspace: Workspace) -> str:
        await workspace.run('echo from-the-shell > shared/written.txt', shell=True)
        return await workspace.read_text('shared/written.txt')

    assert await in_run(sandbox, [s3('shared')], action) == 'from-the-shell\n'

    await eventually(
        lambda: docker('exec', minio, 'mc', 'cat', 'local/one/shared/written.txt').stdout == 'from-the-shell\n'
    )


async def test_a_later_run_with_the_same_mount_only_refreshes_its_keys(sandbox: WorkspaceRef) -> None:
    await in_run(sandbox, [s3('kept')], lambda workspace: workspace.write_text('kept/a.txt', 'a'))
    before = mount_of(sandbox, '/workspace/kept')

    assert await in_run(sandbox, [s3('kept')], lambda workspace: workspace.read_text('kept/a.txt')) == 'a'

    assert before is not None and mount_of(sandbox, '/workspace/kept') == before


async def test_another_bucket_at_a_folder_waits_for_the_keys_there_to_expire(sandbox: WorkspaceRef) -> None:
    await in_run(
        sandbox, [s3('swapped', expires_in=timedelta(seconds=20))], lambda workspace: workspace.exists('swapped')
    )
    first = mount_of(sandbox, '/workspace/swapped')

    with pytest.raises(MountError, match='already shows another bucket in this sandbox'):
        await in_run(sandbox, [s3('swapped', 'two')], lambda workspace: workspace.exists('swapped'))

    await anyio.sleep(25)
    await in_run(sandbox, [s3('swapped', 'two')], lambda workspace: workspace.write_text('swapped/b.txt', 'b'))
    second = mount_of(sandbox, '/workspace/swapped')
    assert first is not None and second is not None and second[1] != first[1]


async def test_a_folder_unmounted_by_hand_is_mounted_again(sandbox: WorkspaceRef) -> None:
    await in_run(sandbox, [s3('manual')], lambda workspace: workspace.write_text('manual/c.txt', 'c'))
    as_root(sandbox, 'fusermount3 -u /workspace/manual')
    assert mount_of(sandbox, '/workspace/manual') is None

    assert await in_run(sandbox, [s3('manual')], lambda workspace: workspace.read_text('manual/c.txt')) == 'c'
    assert mount_of(sandbox, '/workspace/manual') is not None


async def test_no_key_is_on_a_command_line_or_in_the_log(sandbox: WorkspaceRef) -> None:
    await in_run(sandbox, [s3('private')], lambda workspace: workspace.exists('private'))

    command_lines = as_root(sandbox, "for f in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < \"$f\"; echo; done")
    assert 'rclone rcd' in command_lines
    assert SECRET not in command_lines
    assert SECRET not in as_root(sandbox, 'cat /dev/shm/pydantic-ai-mounts/rcd.log')


async def test_fresh_keys_take_over_from_the_old_ones(minio: str, sandbox: WorkspaceRef) -> None:
    users = {name: f'{name}-{secrets.token_hex(8)}' for name in ('pai-old', 'pai-new')}
    for user, password in users.items():
        assert docker('exec', minio, 'mc', 'admin', 'user', 'add', 'local', user, password).returncode == 0
        attached = docker('exec', minio, 'mc', 'admin', 'policy', 'attach', 'local', 'readwrite', '--user', user)
        assert attached.returncode == 0, attached.stderr

    def keyed(user: str) -> S3Mount:
        mount = s3('rotated')
        return S3Mount(
            path=mount.path,
            url=mount.url,
            access_key_id=user,
            secret_access_key=users[user],
            expires_at=mount.expires_at,
            endpoint=mount.endpoint,
            region=mount.region,
        )

    await in_run(sandbox, [keyed('pai-old')], lambda workspace: workspace.write_text('rotated/old.txt', 'old'))
    before = mount_of(sandbox, '/workspace/rotated')
    await in_run(sandbox, [keyed('pai-new')], lambda workspace: workspace.exists('rotated'))
    assert docker('exec', minio, 'mc', 'admin', 'user', 'disable', 'local', 'pai-old').returncode == 0

    # Straight through the mount, which no run refreshes, once the listing it cached is out of date.
    piped = subprocess.run(
        ['docker', 'exec', '--interactive', minio, 'mc', 'pipe', 'local/one/rotated/new.txt'],
        input='new',
        capture_output=True,
        text=True,
        check=False,
    )
    assert piped.returncode == 0, piped.stderr
    await anyio.sleep(11)
    assert as_root(sandbox, 'cat /workspace/rotated/new.txt') == 'new'
    assert mount_of(sandbox, '/workspace/rotated') == before


async def test_a_read_only_mount_refuses_writes(sandbox: WorkspaceRef) -> None:
    mount = s3('readonly')
    read_only = S3Mount(
        path=mount.path,
        url=mount.url,
        access_key_id=mount.access_key_id,
        secret_access_key=mount.secret_access_key,
        expires_at=mount.expires_at,
        endpoint=mount.endpoint,
        region=mount.region,
        read_only=True,
    )

    async def action(workspace: Workspace) -> tuple[int, str]:
        result = await workspace.run('echo no > readonly/refused.txt', shell=True)
        return result.exit_code, result.stderr

    exit_code, stderr = await in_run(sandbox, [read_only], action)

    assert exit_code != 0 and 'Read-only file system' in stderr

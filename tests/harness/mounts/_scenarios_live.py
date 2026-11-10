"""The `mounts=` scenarios every sandbox's live mount tests run, against MinIO started inside that sandbox.

Each sandbox's `test_mounts_live.py` builds a `LiveSandbox`, calls `start_store` once, and runs every function in
`SCENARIOS` against it. The store runs in the sandbox itself, at `http://127.0.0.1:9000`, so no sandbox needs a way
back to the test machine. MinIO no longer publishes binaries, so this uses `pgsty/silo`, a build of the same
server; the test reads and writes the bucket from outside the mount with its own pinned rclone, never the helper's.
Every key reaches the sandbox on stdin or in a root-only file, so finding `SECRET` on a command line is a leak.
"""

from __future__ import annotations

import secrets
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import CommandResult, Workspace, WorkspaceRef
from pydantic_ai_harness import MountError, S3Mount

ACCESS_KEY_ID = 'pai-live'
SECRET = f'pai-live-{secrets.token_hex(8)}'
"""MinIO's root password, used as the mounts' keys: unique, so finding it anywhere is a leak."""
BUCKETS = ('one', 'two')
STORE = '/opt/pai-live'
"""Root-only: the store's data, its keys, and the test's own rclone."""

_SILO = 'https://github.com/pgsty/silo/releases/download/RELEASE.2026-09-16T00-00-00Z/silo_20260916000000.0.0_linux_{}.tar.gz'
_SILO_SHA256 = {
    'amd64': '381e745510a8fb64323d7bb3207f95984b7f4ed826f4fcad318f97683c420c73',
    'arm64': '6e697d3e1d70f2343fe829cd6820a0b840d4619f1dde546424de80e529636ed2',
}
"""From the release's asset digests."""
_RCLONE = 'https://downloads.rclone.org/v1.75.2/rclone-v1.75.2-linux-{}.zip'
_RCLONE_SHA256 = {
    'amd64': '349ac8fba6ff65d6247043f1750cdcb518ec5d500ef91463a10d37c0ccdf3702',
    'arm64': '7e1e8d69654941b7b7df84ee74c5f7fb09cce7d5496947fb7bf16b55ff427d10',
}
"""From https://downloads.rclone.org/v1.75.2/SHA256SUMS."""

_T = TypeVar('_T')


@dataclass
class LiveSandbox:
    """One live sandbox, and the provider-specific ways to reach it."""

    ref: WorkspaceRef
    working_dir: str
    """Where relative mount paths land, as the sandbox's commands see it."""
    capability: Callable[[Sequence[S3Mount]], AbstractCapability[Any]]
    """The provider's capability with these mounts, attaching to `ref`."""
    run_as_root: Callable[[str, bytes], Awaitable[CommandResult]]
    """Run a shell script as root with this stdin, outside any capability."""


async def as_root(live: LiveSandbox, script: str, stdin: bytes = b'') -> str:
    result = await live.run_as_root(script, stdin)
    assert result.exit_code == 0, f'{script!r} exited {result.exit_code}: {result.stderr or result.stdout}'
    return result.stdout


def _store_env(password: str) -> bytes:
    return '\n'.join(
        [
            f'MINIO_ROOT_USER={ACCESS_KEY_ID}',
            f'MINIO_ROOT_PASSWORD={password}',
            'RCLONE_CONFIG_ST_TYPE=s3',
            'RCLONE_CONFIG_ST_PROVIDER=Minio',
            'RCLONE_CONFIG_ST_ENDPOINT=http://127.0.0.1:9000',
            f'RCLONE_CONFIG_ST_ACCESS_KEY_ID={ACCESS_KEY_ID}',
            f'RCLONE_CONFIG_ST_SECRET_ACCESS_KEY={password}',
            '',
        ]
    ).encode()


_WITH_KEYS = f'set -a; . {STORE}/env; set +a; '
_FETCH = (
    'python3 -c "import shutil, sys, urllib.request; shutil.copyfileobj(urllib.request.urlopen(urllib.request.Request('
    "sys.argv[1], headers={'User-Agent': 'pai-live'})), sys.stdout.buffer)\""
)
"""Print a URL's body, failing on an error status. Python, as slim images such as Modal's lack `curl`, with a
User-Agent of its own, as rclone's download site refuses Python's."""


async def _serve(live: LiveSandbox, password: str) -> None:
    """(Re)start the store with `password` as its only valid key."""
    await as_root(live, f'umask 077; cat > {STORE}/env', _store_env(password))
    # By the PID it saved: `pkill -f` would match this script's own command line too. A stopped store counts as gone
    # once it is a zombie: Modal's PID 1 does not reap the orphans it adopts, so `kill -0` would succeed forever.
    await as_root(
        live,
        f'if [ -f {STORE}/pid ]; then p=$(cat {STORE}/pid); kill "$p" 2> /dev/null || true;'
        ' while [ -e "/proc/$p" ] && ! grep -q "^State:.*Z" "/proc/$p/status" 2> /dev/null; do sleep 0.2; done; fi;'
        f' {_WITH_KEYS}setsid nohup {STORE}/silo server {STORE}/data --address 127.0.0.1:9000'
        f' --console-address 127.0.0.1:9001 > {STORE}/silo.log 2>&1 < /dev/null & echo $! > {STORE}/pid;'
        f' for i in $(seq 120); do {_FETCH} http://127.0.0.1:9000/minio/health/live > /dev/null && exit 0; sleep 0.5;'
        f' done; tail -5 {STORE}/silo.log; exit 1',
    )


async def start_store(live: LiveSandbox) -> None:
    """Download and start MinIO in the sandbox, with the buckets in `BUCKETS`."""
    script = f"""set -e
case $(uname -m) in x86_64) arch=amd64 ;; aarch64) arch=arm64 ;; *) echo "no build for $(uname -m)"; exit 1 ;; esac
case $arch in amd64) silo={_SILO_SHA256['amd64']} rclone={_RCLONE_SHA256['amd64']} ;;
  arm64) silo={_SILO_SHA256['arm64']} rclone={_RCLONE_SHA256['arm64']} ;; esac
mkdir -p -m 0700 {STORE}/data
cd {STORE}
{_FETCH} "{_SILO.format('$arch')}" > silo.tgz
echo "$silo  silo.tgz" | sha256sum -c - > /dev/null
tar -xzf silo.tgz silo
{_FETCH} "{_RCLONE.format('$arch')}" > rclone.zip
echo "$rclone  rclone.zip" | sha256sum -c - > /dev/null
python3 -c "import sys, zipfile; z = zipfile.ZipFile('rclone.zip'); open('rclone', 'wb').write(z.read(next(n for n in z.namelist() if n.endswith('/rclone'))))"
chmod 0755 silo rclone
rm silo.tgz rclone.zip
"""
    await as_root(live, script)
    await _serve(live, SECRET)
    await as_root(live, f'{_WITH_KEYS}' + ' && '.join(f'{STORE}/rclone mkdir st:{bucket}' for bucket in BUCKETS))


async def store_read(live: LiveSandbox, key: str) -> str | None:
    """The object at `key` (`bucket/path`), read from outside the mount, or `None` if there is none."""
    result = await live.run_as_root(f'{_WITH_KEYS}{STORE}/rclone cat st:{key}', b'')
    return result.stdout if result.exit_code == 0 else None


async def store_write(live: LiveSandbox, key: str, data: str) -> None:
    await as_root(live, f'{_WITH_KEYS}{STORE}/rclone rcat st:{key}', data.encode())


def s3(
    path: str,
    bucket: str = 'one',
    *,
    expires_in: timedelta = timedelta(hours=1),
    secret: str = SECRET,
    read_only: bool = False,
) -> S3Mount:
    return S3Mount(
        path=path,
        url=f's3://{bucket}/{path}/',
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=secret,
        expires_at=datetime.now(UTC) + expires_in,
        endpoint='http://127.0.0.1:9000',
        region='us-east-1',
        read_only=read_only,
    )


async def in_run(live: LiveSandbox, mounts: Sequence[S3Mount], action: Callable[[Workspace], Awaitable[_T]]) -> _T:
    """Run an agent on the sandbox with `mounts`, whose one tool call does `action`."""
    agent = Agent(TestModel(call_tools=['act']), capabilities=[live.capability(mounts)])
    results: list[_T] = []

    @agent.tool
    async def act(ctx: RunContext[object]) -> str:
        results.append(await action(ctx.workspace))
        return 'done'

    await agent.run('go', workspace=live.ref)
    return results[0]


async def mount_of(live: LiveSandbox, path: str) -> tuple[str, str] | None:
    """The kernel's mount ID and mount source at `path` in the working directory, or `None`."""
    for line in (await as_root(live, 'cat /proc/self/mountinfo')).splitlines():
        fields, _, rest = line.partition(' - ')
        if fields.split()[4] == f'{live.working_dir}/{path}':
            return fields.split()[0], rest.split()[1]
    return None


async def eventually(check: Callable[[], Awaitable[bool]]) -> None:
    """Wait for `check`: rclone uploads a closed file a few seconds later."""
    for _ in range(60):
        if await check():
            return
        await anyio.sleep(0.5)
    pytest.fail('the condition never held')


async def commands_and_file_tools_see_the_bucket(live: LiveSandbox) -> None:
    async def action(workspace: Workspace) -> str:
        await workspace.run('echo from-the-shell > shared/written.txt', shell=True)
        return await workspace.read_text('shared/written.txt')

    assert await in_run(live, [s3('shared')], action) == 'from-the-shell\n'

    async def uploaded() -> bool:
        return await store_read(live, 'one/shared/written.txt') == 'from-the-shell\n'

    await eventually(uploaded)


async def a_later_run_with_the_same_mount_only_refreshes_its_keys(live: LiveSandbox) -> None:
    await in_run(live, [s3('kept')], lambda workspace: workspace.write_text('kept/a.txt', 'a'))
    before = await mount_of(live, 'kept')

    assert await in_run(live, [s3('kept')], lambda workspace: workspace.read_text('kept/a.txt')) == 'a'

    assert before is not None and await mount_of(live, 'kept') == before


async def another_bucket_at_a_folder_waits_for_the_keys_there_to_expire(live: LiveSandbox) -> None:
    await in_run(live, [s3('swapped', expires_in=timedelta(seconds=20))], lambda workspace: workspace.exists('swapped'))
    first = await mount_of(live, 'swapped')

    with pytest.raises(MountError, match='already shows another bucket in this sandbox'):
        await in_run(live, [s3('swapped', 'two')], lambda workspace: workspace.exists('swapped'))

    await anyio.sleep(25)
    await in_run(live, [s3('swapped', 'two')], lambda workspace: workspace.write_text('swapped/b.txt', 'b'))
    second = await mount_of(live, 'swapped')
    assert first is not None and second is not None and second[1] != first[1]


async def a_folder_unmounted_by_hand_is_mounted_again(live: LiveSandbox) -> None:
    await in_run(live, [s3('manual')], lambda workspace: workspace.write_text('manual/c.txt', 'c'))
    await as_root(live, f'fusermount3 -u {live.working_dir}/manual')
    assert await mount_of(live, 'manual') is None

    assert await in_run(live, [s3('manual')], lambda workspace: workspace.read_text('manual/c.txt')) == 'c'
    assert await mount_of(live, 'manual') is not None


async def no_key_is_on_a_command_line_or_in_the_log(live: LiveSandbox) -> None:
    await in_run(live, [s3('private')], lambda workspace: workspace.exists('private'))

    command_lines = await as_root(live, "for f in /proc/[0-9]*/cmdline; do tr '\\0' ' ' < \"$f\"; echo; done")
    assert 'rclone rcd' in command_lines
    assert SECRET not in command_lines
    assert SECRET not in await as_root(live, 'cat /dev/shm/pydantic-ai-mounts/rcd.log')


async def fresh_keys_take_over_from_the_old_ones(live: LiveSandbox) -> None:
    await in_run(live, [s3('rotated')], lambda workspace: workspace.write_text('rotated/old.txt', 'old'))
    before = await mount_of(live, 'rotated')
    fresh = f'pai-fresh-{secrets.token_hex(8)}'
    # The store restarts with only the fresh key valid, so the old one, still in the mount, stops working.
    await _serve(live, fresh)
    try:
        await in_run(live, [s3('rotated', secret=fresh)], lambda workspace: workspace.exists('rotated'))
        await store_write(live, 'one/rotated/new.txt', 'new')
        # Straight through the mount, which no run refreshes, once the listing it cached is out of date.
        await anyio.sleep(11)
        assert await as_root(live, f'cat {live.working_dir}/rotated/new.txt') == 'new'
        assert await mount_of(live, 'rotated') == before
    finally:
        await _serve(live, SECRET)


async def a_read_only_mount_refuses_writes(live: LiveSandbox) -> None:
    async def action(workspace: Workspace) -> tuple[int, str]:
        result = await workspace.run('echo no > readonly/refused.txt', shell=True)
        return result.exit_code, result.stderr

    exit_code, stderr = await in_run(live, [s3('readonly', read_only=True)], action)

    assert exit_code != 0 and 'Read-only file system' in stderr


async def a_folder_renamed_in_the_mount_keeps_its_files(live: LiveSandbox) -> None:
    async def action(workspace: Workspace) -> CommandResult:
        return await workspace.run(
            'mkdir -p moved/d1/sub && echo x > moved/d1/sub/x.txt && mv moved/d1 moved/d2 && cat moved/d2/sub/x.txt',
            shell=True,
        )

    result = await in_run(live, [s3('moved')], action)

    assert (result.exit_code, result.stdout) == (0, 'x\n'), result.stderr


async def pip_installs_into_the_mount(live: LiveSandbox) -> None:
    async def action(workspace: Workspace) -> CommandResult:
        return await workspace.run(
            'python3 -m pip install -q --no-cache-dir --target packages/target six==1.16.0'
            ' && PYTHONPATH=packages/target python3 -c "import six; print(six.__version__)"',
            shell=True,
            timeout=300,
        )

    result = await in_run(live, [s3('packages')], action)

    assert (result.exit_code, result.stdout.splitlines()[-1:]) == (0, ['1.16.0']), result.stderr[-2000:]


SCENARIOS: tuple[Callable[[LiveSandbox], Awaitable[None]], ...] = (
    commands_and_file_tools_see_the_bucket,
    a_later_run_with_the_same_mount_only_refreshes_its_keys,
    another_bucket_at_a_folder_waits_for_the_keys_there_to_expire,
    a_folder_unmounted_by_hand_is_mounted_again,
    no_key_is_on_a_command_line_or_in_the_log,
    fresh_keys_take_over_from_the_old_ones,
    a_read_only_mount_refuses_writes,
    a_folder_renamed_in_the_mount_keeps_its_files,
    pip_installs_into_the_mount,
)
"""Every scenario, by name, for `pytest.mark.parametrize`."""

"""The live check every sandbox that mounts buckets runs: a real FUSE mount through its own way of running as root.

The bucket is served from inside the sandbox by `rclone serve s3`, so the check needs no cloud bucket or keys: it
proves what differs per sandbox (FUSE, running as root with stdin, the agent's user) and leaves the storage side to
`integration_tests/cloud_storage`.
"""

from __future__ import annotations

import posixpath
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import anyio
import pytest

from pydantic_ai.workspaces import Mount, SupportsCommands, SupportsMounts, Workspace, WorkspaceBackend
from pydantic_ai_harness.cloud_storage import S3Filesystem

_SERVER = '/var/tmp/pydantic-ai-s3'
"""What `rclone serve s3` serves: each folder in it is a bucket."""


_LISTENING = "import socket; socket.create_connection(('127.0.0.1', 8333), timeout=1)"

_FETCH_RCLONE = """
import io, os, platform, sys, urllib.request, zipfile
arch = 'amd64' if platform.machine() == 'x86_64' else 'arm64'
name = f'rclone-v1.75.2-linux-{arch}'
request = urllib.request.Request(f'https://downloads.rclone.org/v1.75.2/{name}.zip', headers={'User-Agent': 'test'})
with urllib.request.urlopen(request, timeout=120) as reply:
    binary = zipfile.ZipFile(io.BytesIO(reply.read())).read(f'{name}/rclone')
with open(sys.argv[1], 'wb') as file:
    file.write(binary)
os.chmod(sys.argv[1], 0o755)
"""
"""Puts rclone, which serves the bucket, at `argv[1]`; the mount brings its own."""


class MountingBackend(WorkspaceBackend, SupportsCommands, SupportsMounts, Protocol):
    pass


def _keys() -> Any:
    expires_at = datetime.now(UTC) + timedelta(minutes=15)
    return {'access_key_id': 'live', 'secret_access_key': 'live-secret', 'token': None, 'expires_at': expires_at}


async def check_bucket_mounts(backend: MountingBackend) -> None:
    """Mount a bucket, use it from commands and file tools, and find every write in the bucket after unmounting."""
    working_dir = await backend.working_dir()
    fetched = await backend.run(['python3', '-c', _FETCH_RCLONE, f'{_SERVER}-rclone'])
    assert fetched.exit_code == 0, fetched.stderr
    server = await backend.run(
        f'mkdir -p {_SERVER}/live && (setsid nohup {_SERVER}-rclone serve s3 {_SERVER} --addr 127.0.0.1:8333 '
        f'--auth-key live,live-secret > {_SERVER}.log 2>&1 &)',
        shell=True,
    )
    assert server.exit_code == 0, server.stderr
    with anyio.fail_after(60):  # Hang guard, not a speed check.
        while (await backend.run(['python3', '-c', _LISTENING])).exit_code != 0:
            await anyio.sleep(0.5)

    data = S3Filesystem('s3://live/data', endpoint='http://127.0.0.1:8333', region='us-east-1', credentials=_keys)
    mount = Mount('data', data, shell=True)
    workspace = Workspace(backend, mounts=[mount])

    await workspace.write_text('data/notes/a.md', 'from file tools')
    shell = await workspace.run(
        'cd data && cat notes/a.md && echo one > log && echo two >> log && sed -i s/one/first/ log'
        ' && mkdir d1 && echo moved > d1/f && mv d1 d2 && cat log d2/f && touch mine && stat -c %U mine',
        shell=True,
    )
    owner = (await backend.run(['stat', '-c', '%U', working_dir])).stdout.strip()
    assert (shell.exit_code, shell.stdout) == (0, f'from file toolsfirst\ntwo\nmoved\n{owner}\n'), shell.stderr

    # Unmounting sends what is still queued, so the served folder has every file.
    await backend.ensure_mounts([], working_dir)
    served = posixpath.join(_SERVER, 'live', 'data')
    stored = await backend.run(f'cd {served} && cat notes/a.md log d2/f && test ! -e d1', shell=True)
    assert (stored.exit_code, stored.stdout) == (0, 'from file toolsfirst\ntwo\nmoved\n'), stored.stderr

    # As after a sandbox restart: nothing is mounted, and something writes into the bare folder.
    await backend.run('echo lost > data/lost.txt', shell=True)
    with pytest.warns(UserWarning, match='.pydantic-ai-recovered'):
        await backend.ensure_mounts([mount], working_dir)
    again = await backend.run('cat data/log .pydantic-ai-recovered/*-data/lost.txt', shell=True)
    assert again.stdout == 'first\ntwo\nlost\n'
    await backend.ensure_mounts([], working_dir)

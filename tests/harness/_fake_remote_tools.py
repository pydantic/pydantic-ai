"""Stand-ins for the `ssh` and `bwrap` executables, so SSH and bubblewrap workspaces run without a host or Linux."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

BWRAP_WORKS = (
    shutil.which('bwrap') is not None
    and subprocess.run(['bwrap', '--ro-bind', '/', '/', '--unshare-all', 'true'], capture_output=True).returncode == 0
)
"""Whether this machine has a working `bwrap` (Linux with user namespaces), for the tests that need the real one."""

# Runs the remote command on this machine, starting in `$HOME` like a login. `unreachable` and `dropped` fail on
# purpose, `chatty` prints a login banner first, `slow` takes 30 seconds to log in, and `lossy` loses the
# connection (`ssh` exits 255) after the command finishes.
_FAKE_SSH = """#!/bin/sh
bin=$(dirname "$0")
while [ "$1" != -- ]; do printf '%s\\n' "$1" >> "$bin/ssh-options"; shift; done
destination=$2
shift 2
case $destination in
unreachable) echo 'ssh: connect to host unreachable port 22: Connection refused' >&2; exit 255 ;;
dropped) printf '__pydantic_ai_ssh_ready__\\n' >&2; exit 255 ;;
chatty) echo 'Welcome to box!'; echo 'Last login: yesterday' >&2 ;;
slow) sleep 30 ;;
lossy) cd "$HOME" && sh -c "$*"; exit 255 ;;
esac
cd "$HOME" && exec sh -c "$*"
"""

# Records its arguments, and the seccomp filter it's given, and runs the command unsandboxed; `--fake-fail`
# fails like a denied user namespace.
_FAKE_BWRAP = """#!/bin/sh
bin=$(dirname "$0")
printf '%s\\n' "$*" >> "$bin/bwrap-calls"
for arg in "$@"; do
    if [ "$arg" = --fake-fail ]; then echo 'bwrap: setting up uid map: Permission denied' >&2; exit 1; fi
done
while [ "$1" != -- ]; do
    if [ "$1" = --setenv ]; then export "$2=$3"; shift 2; fi
    if [ "$1" = --seccomp ]; then cat <&"$2" > "$bin/seccomp-filter"; shift; fi
    shift
done
shift
exec "$@"
"""


class FakeRemoteTools:
    def __init__(self, bin_dir: Path, home: Path) -> None:
        self.bin_dir = bin_dir
        self.home = home

    @property
    def ssh_options(self) -> list[str]:
        return (self.bin_dir / 'ssh-options').read_text().splitlines()

    @property
    def seccomp_filter(self) -> bytes:
        """The filter the last sandbox read from its `--seccomp` descriptor."""
        return (self.bin_dir / 'seccomp-filter').read_bytes()

    @property
    def bwrap_calls(self) -> list[str]:
        path = self.bin_dir / 'bwrap-calls'
        return path.read_text().splitlines() if path.exists() else []


def install_fake_remote_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeRemoteTools:
    """Put fake `ssh` and `bwrap` first on `PATH`, and point `HOME` (the fake remote login directory) at a new directory.

    Both live beside `tmp_path`, not inside it. A sandbox whose working directory is `tmp_path` must not
    see its own launcher or `~/.ssh` as writable paths it is responsible for protecting.
    """
    root = Path(tempfile.mkdtemp(prefix=f'fake-remote-{tmp_path.name}-', dir=str(tmp_path.parent)))
    bin_dir = root / 'fake-bin'
    home = root / 'fake-home'
    bin_dir.mkdir()
    home.mkdir()
    for name, script in (('ssh', _FAKE_SSH), ('bwrap', _FAKE_BWRAP)):
        path = bin_dir / name
        path.write_text(script)
        path.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    monkeypatch.setenv('HOME', str(home))
    return FakeRemoteTools(bin_dir, home)

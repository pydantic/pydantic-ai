"""A stand-in for the `docker` CLI, so `DockerSandboxBackend` runs without a daemon.

A "container" is a state file naming its working directory on this machine; `exec` runs the command
there, like the real one runs it in the container. Removing a container kills the commands running in it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

# `flaky-image` fails only its first pull, `unstartable-image` creates a container
# that fails to start, `--fake-hang` makes the first `run` hang before it creates anything so a create can be
# cancelled, and `--fake-hang-late` makes it hang after creating the container. A container created with
# `--device /dev/fuse` has FUSE, and a root `exec` runs `python3` as `fake-helper`: it records its stdin line and
# prints the reply the test put in `root-reply`.
_FAKE_DOCKER = """#!/bin/sh
bin=$(dirname "$0")
state="$bin/containers"
mkdir -p "$state"
printf '%s\\n' "$*" >> "$bin/docker-calls"
env > "$bin/docker-env"
command=$1
shift
case $command in
run)
    while [ "$1" != -- ]; do
        case $1 in
        --name) name=$2; shift ;;
        --workdir) workdir=$2; shift ;;
        --label) label=$2; shift ;;
        --device) device=$2; shift ;;
        --entrypoint | --network | --memory | --pids-limit | --tmpfs | --cap-add | --security-opt) shift ;;
        --fake-hang) [ -e "$bin/hung" ] || { touch "$bin/hung"; sleep 30; } ;;
        --fake-hang-late) hang_late=1 ;;
        esac
        shift
    done
    if [ "$2" = flaky-image ] && [ ! -e "$bin/flaky-pulled" ]; then
        touch "$bin/flaky-pulled"
        echo 'docker: Error response from daemon: Get "https://registry-1.docker.io/v2/": net/http: TLS handshake timeout.' >&2
        exit 125
    fi
    # Without `--workdir`, the image's own `WORKDIR`.
    [ -n "$workdir" ] || workdir="$bin/image-workdir"
    if [ -f "$state/$name" ]; then
        echo "docker: Error response from daemon: Conflict. The container name \"/$name\" is already in use." >&2
        exit 125
    fi
    mkdir -p "$workdir" && printf '%s' "$workdir" > "$state/$name" || exit 1
    [ "$label" != ai.pydantic.workspace=true ] || touch "$state/$name.labeled"
    [ "$device" != /dev/fuse ] || touch "$state/$name.fuse"
    if [ "$2" = unstartable-image ]; then
        echo 'docker: Error response from daemon: port is already allocated.' >&2
        exit 125
    fi
    touch "$state/$name.running"
    [ -z "$hang_late" ] || sleep 30
    echo "$name"
    ;;
inspect)
    # `inspect --type container [--format TEMPLATE] -- NAME`: the name is the last argument.
    for name; do :; done
    [ -f "$state/$name" ] || { echo "Error: No such container: $name" >&2; exit 1; }
    [ "$name" != uninspectable ] || { echo 'permission denied while trying to connect to the daemon' >&2; exit 1; }
    if [ -f "$state/$name.labeled" ]; then label=true; else label='<no value>'; fi
    if [ -f "$state/$name.running" ]; then running=true; else running=false; fi
    echo "$label $running"
    ;;
start)
    [ -f "$state/$2" ] || { echo "Error response from daemon: No such container: $2" >&2; exit 1; }
    [ "$2" != broken ] || { echo 'Error response from daemon: port is already allocated' >&2; exit 1; }
    touch "$state/$2.running"
    echo "$2"
    ;;
exec)
    while :; do
        case $1 in
        --workdir) workdir=$2; shift 2 ;;
        --env) export "$2"; shift 2 ;;
        --user) user=$2; shift 2 ;;
        --interactive) shift ;;
        *) break ;;
        esac
    done
    name=$1
    shift
    [ -f "$state/$name" ] || { echo "Error response from daemon: No such container: $name" >&2; exit 1; }
    [ "$name" != silent ] || exit 1
    # Each container gets a directory of its own for the backend's PID files; a `readonly` one's isn't writable.
    if [ "$1" = sh ] && [ "$2" = -c ]; then
        tmp="$state/$name.tmp"
        [ "$name" != readonly ] || tmp=/nonexistent-pydantic-ai-dir
        mkdir -p "$state/$name.tmp"
        script=$(printf '%s' "$3" | sed -e "s#/run/pydantic-ai/#$tmp/#g" -e "s#-c /dev/fuse#-f $state/$name.fuse#")
        shift 3
        set -- sh -c "$script" "$@"
    fi
    # The stop script is the only exec with exactly `sh -c SCRIPT sh TAG`.
    if [ -f "$state/$name.hang-stop" ] && [ $# -eq 5 ]; then sleep 30; fi
    [ -n "$workdir" ] || workdir=$(cat "$state/$name")
    [ "$user" != 0 ] || { env > "$bin/root-env"; PATH="$bin/root-bin:$PATH"; }
    # Like Docker, which reports a missing `--workdir` on stdout.
    cd "$workdir" 2> /dev/null || { echo "OCI runtime exec failed: chdir to cwd (\\"$workdir\\")"; exit 127; }
    # One file per exec, removed when it finishes, so `rm` only ever sees live commands.
    mkdir -p "$state/$name.pids"
    echo $$ > "$state/$name.pids/$$"
    "$@"
    status=$?
    rm -f "$state/$name.pids/$$"
    exit "$status"
    ;;
rm)
    name=$4
    [ -f "$state/$name" ] || { echo "Error response from daemon: No such container: $name" >&2; exit 1; }
    [ "$name" != stuck ] || { echo 'Error response from daemon: removal already in progress' >&2; exit 1; }
    for file in "$state/$name.pids"/*; do
        [ -f "$file" ] || continue
        pid=$(cat "$file")
        # Guard against a reused PID: only kill a process that is still this fake `docker`.
        case $(ps -o args= -p "$pid" 2> /dev/null) in
        *"$bin/docker"*) kill -s KILL -- "-$pid" "$pid" 2> /dev/null ;;
        esac
    done
    rm -rf "$state/$name" "$state/$name".*
    ;;
esac
"""


_FAKE_HELPER = """#!/bin/sh
bin=$(dirname "$(dirname "$0")")
head -n 1 >> "$bin/root-stdin"
cat "$bin/root-reply" 2> /dev/null || echo '{"recovered": []}'
"""


class FakeDocker:
    def __init__(self, bin_dir: Path) -> None:
        self.bin_dir = bin_dir

    @property
    def calls(self) -> list[str]:
        path = self.bin_dir / 'docker-calls'
        return path.read_text().splitlines() if path.exists() else []

    @property
    def image_workdir(self) -> Path:
        return self.bin_dir / 'image-workdir'

    def client_env(self) -> str:
        """The environment the last `docker` call got."""
        return (self.bin_dir / 'docker-env').read_text()

    def containers(self) -> list[str]:
        return sorted(path.name for path in (self.bin_dir / 'containers').glob('*') if not path.suffix)

    def add_container(self, name: str, working_dir: Path, *, labeled: bool = True, fuse: bool = False) -> None:
        """Add a stopped container; `labeled=False` makes it one `DockerSandbox` didn't create."""
        state = self.bin_dir / 'containers'
        state.mkdir(exist_ok=True)
        (state / name).write_text(str(working_dir))
        if labeled:
            (state / f'{name}.labeled').touch()
        if fuse:
            (state / f'{name}.fuse').touch()

    def pid_files(self, name: str) -> set[Path]:
        """The backend's PID files in the container."""
        return set((self.bin_dir / 'containers' / f'{name}.tmp').glob('*.pid'))

    def tmp_files(self, name: str) -> list[Path]:
        return list((self.bin_dir / 'containers' / f'{name}.tmp').glob('*'))

    def root_requests(self) -> list[dict[str, Any]]:
        """The stdin lines of the helper's runs as root, parsed."""
        path = self.bin_dir / 'root-stdin'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def root_env(self) -> str:
        """The environment the last root `exec` got."""
        return (self.bin_dir / 'root-env').read_text()

    def reply_as_root(self, stdout: str) -> None:
        (self.bin_dir / 'root-reply').write_text(stdout)

    def hang_stops(self, name: str) -> None:
        (self.bin_dir / 'containers' / f'{name}.hang-stop').touch()


def install_fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    """Put a fake `docker` first on `PATH`."""
    bin_dir = tmp_path / 'fake-bin'
    bin_dir.mkdir()
    path = bin_dir / 'docker'
    path.write_text(_FAKE_DOCKER)
    path.chmod(0o755)
    (bin_dir / 'podman').write_text(_FAKE_DOCKER)
    (bin_dir / 'podman').chmod(0o755)
    (bin_dir / 'root-bin').mkdir()
    (bin_dir / 'root-bin' / 'python3').write_text(_FAKE_HELPER)
    (bin_dir / 'root-bin' / 'python3').chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return FakeDocker(bin_dir)

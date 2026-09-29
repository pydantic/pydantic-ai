"""A stand-in for the `docker` CLI, so `DockerSandboxBackend` runs without a daemon.

A "container" is a state file naming its working directory on this machine; `exec` runs the command
there, like the real one runs it in the container. Removing a container kills the commands running in it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# `missing-image` fails to pull, and `--fake-hang` makes `run` hang so a create can be cancelled.
_FAKE_DOCKER = """#!/bin/sh
bin=$(dirname "$0")
state="$bin/containers"
mkdir -p "$state"
printf '%s\\n' "$*" >> "$bin/docker-calls"
command=$1
shift
case $command in
run)
    while [ "$1" != -- ]; do
        case $1 in
        --name) name=$2; shift ;;
        --workdir) workdir=$2; shift ;;
        --label | --entrypoint | --network | --memory) shift ;;
        --fake-hang) sleep 30 ;;
        esac
        shift
    done
    if [ "$2" = missing-image ]; then
        echo "Unable to find image 'missing-image:latest' locally" >&2
        echo 'docker: Error response from daemon: pull access denied for missing-image.' >&2
        exit 125
    fi
    mkdir -p "$workdir" && printf '%s' "$workdir" > "$state/$name" && echo "$name"
    ;;
start)
    [ -f "$state/$2" ] || { echo "Error response from daemon: No such container: $2" >&2; exit 1; }
    echo "$2"
    ;;
exec)
    while :; do
        case $1 in
        --workdir) workdir=$2; shift 2 ;;
        --env) export "$2"; shift 2 ;;
        *) break ;;
        esac
    done
    name=$1
    shift
    [ -f "$state/$name" ] || { echo "Error response from daemon: No such container: $name" >&2; exit 1; }
    [ "$name" != silent ] || exit 1
    # The stop script is the only exec with exactly `sh -c SCRIPT sh TAG`.
    if [ -f "$state/$name.hang-stop" ] && [ $# -eq 5 ]; then sleep 30; fi
    [ -n "$workdir" ] || workdir=$(cat "$state/$name")
    cd "$workdir" 2> /dev/null || { echo "OCI runtime exec failed: chdir to cwd (\\"$workdir\\")" >&2; exit 126; }
    echo $$ >> "$state/$name.pids"
    exec "$@"
    ;;
rm)
    name=$4
    [ -f "$state/$name" ] || { echo "Error response from daemon: No such container: $name" >&2; exit 1; }
    [ "$name" != stuck ] || { echo 'Error response from daemon: removal already in progress' >&2; exit 1; }
    if [ -f "$state/$name.pids" ]; then
        for pid in $(cat "$state/$name.pids"); do kill -s KILL -- "-$pid" "$pid" 2> /dev/null; done
    fi
    rm -f "$state/$name" "$state/$name.pids" "$state/$name.hang-stop"
    ;;
esac
"""


class FakeDocker:
    def __init__(self, bin_dir: Path) -> None:
        self.bin_dir = bin_dir

    @property
    def calls(self) -> list[str]:
        path = self.bin_dir / 'docker-calls'
        return path.read_text().splitlines() if path.exists() else []

    def containers(self) -> list[str]:
        return sorted(path.name for path in (self.bin_dir / 'containers').glob('*') if not path.suffix)

    def add_container(self, name: str, working_dir: Path) -> None:
        state = self.bin_dir / 'containers'
        state.mkdir(exist_ok=True)
        (state / name).write_text(str(working_dir))

    def hang_stops(self, name: str) -> None:
        (self.bin_dir / 'containers' / f'{name}.hang-stop').touch()


def install_fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    """Put a fake `docker` first on `PATH`."""
    bin_dir = tmp_path / 'fake-bin'
    bin_dir.mkdir()
    path = bin_dir / 'docker'
    path.write_text(_FAKE_DOCKER)
    path.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return FakeDocker(bin_dir)

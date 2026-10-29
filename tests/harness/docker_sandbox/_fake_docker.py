"""A stand-in for the `docker` CLI, so `DockerSandboxBackend` runs without a daemon.

A "container" is a state file naming its working directory on this machine; `exec` runs the command
there, like the real one runs it in the container. Removing a container kills the commands running in it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ...workspace_fakes import FAKE_SCREENSHOT

# `flaky-image` fails only its first pull, `unstartable-image` creates a container
# that fails to start, `--fake-hang` makes the first `run` hang before it creates anything so a create can be
# cancelled, and `--fake-hang-late` makes it hang after creating the container.
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
        --entrypoint | --network | --memory | --pids-limit | --tmpfs) shift ;;
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
        *) break ;;
        esac
    done
    name=$1
    shift
    [ -f "$state/$name" ] || { echo "Error response from daemon: No such container: $name" >&2; exit 1; }
    [ "$name" != silent ] || exit 1
    # Each container gets a directory of its own for the backend's run files; a `readonly` one's isn't writable.
    if [ "$1" = sh ] && [ "$2" = -c ]; then
        tmp="$state/$name.tmp"
        [ "$name" != readonly ] || tmp=/nonexistent-pydantic-ai-dir
        mkdir -p "$state/$name.tmp"
        export FAKE_TMP="$tmp"
        for arg; do
            shift
            case $arg in */run/pydantic-ai/* | */tmp/.X* | */proc/*)
                arg=$(printf '%s' "$arg" | sed -e "s#/run/pydantic-ai/#$tmp/#g" -e "s#/tmp/\\.X#$tmp/.X#g" \
                    -e "s#/proc/#$tmp/proc/#g") ;;
            esac
            set -- "$@" "$arg"
        done
    fi
    # The stop script is the only exec with exactly `sh -c SCRIPT sh TAG`.
    if [ -f "$state/$name.hang-stop" ] && [ $# -eq 5 ]; then sleep 30; fi
    [ -n "$workdir" ] || workdir=$(cat "$state/$name")
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
image)
    # `image inspect -- TAG`: an image exists once `build` made it, or a test added it.
    [ -f "$bin/images/$3" ] || { echo "Error: No such image: $3" >&2; exit 1; }
    ;;
build)
    # `build --tag TAG -- CONTEXT`: keeps the context's Dockerfile as the image.
    [ ! -e "$bin/fail-builds" ] || { echo 'E: Unable to locate package xvfb' >&2; exit 1; }
    mkdir -p "$bin/images"
    cp "$4/Dockerfile" "$bin/images/$2" || exit 1
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


_FAKE_XDOTOOL = """#!/bin/sh
bin=$(dirname "$0")
printf '%s %s\\n' "$DISPLAY" "$*" >> "$bin/xdotool-calls"
[ ! -e "$bin/no-display" ] || { echo "Can't open display: $DISPLAY" >&2; exit 1; }
[ "$1" != getdisplaygeometry ] || echo '1280 800'
[ "$1" != getmouselocation ] || printf 'X=640\\nY=400\\nSCREEN=0\\nWINDOW=1\\n'
case " $* " in *' NoSuchKey '*) echo "(symbol) No such key name 'NoSuchKey'. Ignoring it." >&2 ;; esac
"""
"""Stands in for `xdotool` in a container: reports a 1280x800 display and the pointer at its middle unless
`no-display` exists, skips the key `NoSuchKey` as xdotool skips a name it doesn't know, and logs each call."""

_FAKE_XWININFO = """#!/bin/sh
[ ! -e "$(dirname "$0")/no-display" ] || { echo "xwininfo: unable to open display" >&2; exit 1; }
printf '  Width: 1280\\n  Height: 800\\n'
"""
"""Stands in for `xwininfo -root`, so a host's real one never answers for the container's display."""

_FAKE_SCROT = """#!/bin/sh
for last; do :; done
cp "$(dirname "$0")/screen.png" "$last"
[ "$1" != -t ] || cp "$(dirname "$0")/screen.png" "${last%.png}-thumb.png"
"""
"""Stands in for `scrot`: writes the fake screenshot, whatever region or thumbnail it is asked for, to the path it is
given, and with `-t` as the thumbnail beside it."""

_FAKE_XVFB = """#!/bin/sh
bin=$(dirname "$0")
echo "$*" >> "$bin/xvfb-starts"
if [ -e "$bin/xvfb-fails" ]; then echo 'Fatal server error: no screens found' >&2; exit 1; fi
if [ -e "$FAKE_TMP/.X1-lock" ]; then echo 'Fatal server error: Server is already active for display 1' >&2; exit 1; fi
rm -f "$bin/no-display"
exec sleep 2
"""
"""Stands in for an X server: brings the fake display up and stays running a while, or fails if `xvfb-fails` or a
lock for its display exists.

Also keeps a host's real Xvfb from starting.
"""

_FAKE_FLUXBOX = """#!/bin/sh
"""
"""Stands in for the window manager and the panel, so a host's real ones never start."""

_FAKE_SETSID = """#!/bin/sh
exec "$@"
"""
"""Stands in for `setsid`, which macOS lacks."""


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

    def add_container(self, name: str, working_dir: Path, *, labeled: bool = True) -> None:
        """Add a stopped container; `labeled=False` makes it one `DockerSandbox` didn't create."""
        state = self.bin_dir / 'containers'
        state.mkdir(exist_ok=True)
        (state / name).write_text(str(working_dir))
        if labeled:
            (state / f'{name}.labeled').touch()

    def pid_files(self, name: str) -> set[Path]:
        """The backend's PID files in the container."""
        return set((self.bin_dir / 'containers' / f'{name}.tmp').glob('*.pid'))

    def tmp_files(self, name: str) -> list[Path]:
        return list((self.bin_dir / 'containers' / f'{name}.tmp').glob('*'))

    def images(self) -> list[str]:
        return sorted(path.name for path in (self.bin_dir / 'images').glob('*'))

    @property
    def xdotool_calls(self) -> list[str]:
        path = self.bin_dir / 'xdotool-calls'
        return path.read_text().splitlines() if path.exists() else []

    def fail_builds(self) -> None:
        (self.bin_dir / 'fail-builds').touch()

    def remove_display(self, *, can_start: bool = True) -> None:
        """Take the display down; `Xvfb` brings it back unless `can_start` is false."""
        (self.bin_dir / 'no-display').touch()
        if not can_start:
            (self.bin_dir / 'xvfb-fails').touch()

    def leave_x_lock(self, name: str, *, owner: str = 'sleep') -> None:
        """Leave a lock naming PID 25 in the container `name`, which runs `owner`: an X server that's gone, as after
        `docker stop`, leaves one whose PID another program now has."""
        tmp = self.bin_dir / 'containers' / f'{name}.tmp'
        (tmp / '.X1-lock').write_text('        25\n')
        (tmp / 'proc' / '25').mkdir(parents=True)
        (tmp / 'proc' / '25' / 'comm').write_text(f'{owner}\n')

    def x_lock_exists(self, name: str) -> bool:
        return (self.bin_dir / 'containers' / f'{name}.tmp' / '.X1-lock').exists()

    @property
    def xvfb_starts(self) -> int:
        path = self.bin_dir / 'xvfb-starts'
        return len(path.read_text().splitlines()) if path.exists() else 0

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
    (bin_dir / 'xdotool').write_text(_FAKE_XDOTOOL)
    (bin_dir / 'xdotool').chmod(0o755)
    (bin_dir / 'screen.png').write_bytes(FAKE_SCREENSHOT)
    for name, script in [
        ('Xvfb', _FAKE_XVFB),
        ('fluxbox', _FAKE_FLUXBOX),
        ('tint2', _FAKE_FLUXBOX),
        ('setsid', _FAKE_SETSID),
        ('scrot', _FAKE_SCROT),
        ('xwininfo', _FAKE_XWININFO),
    ]:
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    return FakeDocker(bin_dir)

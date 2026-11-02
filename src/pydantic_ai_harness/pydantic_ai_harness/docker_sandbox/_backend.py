"""A [workspace backend][pydantic_ai.workspaces.WorkspaceBackend] in a local Docker (or Podman) container.

CLI behavior verified 2026-10-21 against Docker 29.2 (runc, in Colima) and Podman 6.1.4 (crun, in a Podman
machine) on macOS, see https://docs.docker.com/reference/cli/docker/ and
https://docs.podman.io/en/latest/Commands.html. Re-check these before changing how containers are created,
commands run or stopped, or errors read, by running
`PYDANTIC_AI_HARNESS_DOCKER_LIVE=1 DOCKER_REQUIRE_LIVE=1 pytest -m docker_live tests/harness/docker_sandbox` once
as is and once with `PYDANTIC_AI_HARNESS_DOCKER_EXECUTABLE=podman`:

- `docker run --workdir` creates a missing directory; `podman run --workdir` refuses it.
- Killing the local `docker exec` client leaves its command running in the container.
- A `docker exec` process leads a session of its own, so its process group is the command's.
- `docker exec` waits until every copy of the command's output is closed; `podman exec` returns at exit.
- Docker reports a missing `exec --workdir` on stdout.
- A missing container's errors contain `no such container`, in any case.
- `inspect --format` reads labels through `{{index .Config.Labels "<key>"}}`.
- Of repeated `--pids-limit` flags the last wins.
- Both refuse a second `--network` next to `--network none`.
- A `--tmpfs` mount is mode 1777, so a non-root `--user` can write it, also with `--read-only`.
- BusyBox's `kill` rejects `--` before a negative process group, as does dash's after `-0`.
"""

from __future__ import annotations as _annotations

import os
import posixpath
import re
import secrets
import shutil
from collections.abc import Mapping, Sequence
from typing import Literal

import anyio

from pydantic_ai.workspaces import (
    CommandResult,
    LocalWorkspaceBackend,
    SupportsCommands,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness._workspace_provider import check_timeout, command_argv, stop_shielded

__all__ = ('DockerSandboxBackend', 'remove_container')

PROVIDER = 'docker'

_READY = '__pydantic_ai_docker_ready__\n'
"""Printed on stderr before the command starts; anything before it is the `docker` client's own error."""
_DONE = re.compile(r'\n__pydantic_ai_docker_done__([0-9a-f]+):(\d+)\n')
"""Carries the command's exit status, so a container that dies mid-command isn't read as a result."""

_PID_DIR = '/run/pydantic-ai'
"""Where each command records its process ID, so a second `docker exec` can stop it on a timeout.

A tmpfs the backend mounts in every container it creates, rather than `/tmp`, which a command may clear and a
`--read-only` container can't write.
"""

_PIDS_LIMIT = '2048'
"""A new container's process limit, so a runaway fork loop can't exhaust the Docker host's.

`docker_args` can override it, such as with `['--pids-limit', '-1']` to remove it.
"""

_WRAPPER = f"""tag=$1
pidfile={_PID_DIR}/$tag.pid
stopped={_PID_DIR}/$tag.stopped
shift
for old in {_PID_DIR}/*.pid; do
    read -r pid 2> /dev/null < "$old" || continue
    [ -z "$pid" ] || kill -0 "$pid" 2> /dev/null || kill -0 "-$pid" 2> /dev/null || rm -f "$old"
done
if ! echo $$ 2> /dev/null > "$pidfile"; then
    echo "cannot write $pidfile, which stops the command on a timeout: {_PID_DIR} must be writable" >&2
    exit 1
fi
if [ -e "$stopped" ]; then
    rm -f "$pidfile" "$stopped"
    exit 143
fi
printf '%s' '{_READY}' >&2
(exec "$@")
status=$?
printf '\\n__pydantic_ai_docker_done__%s:%d\\n' "$tag" "$status" >&2
exit "$status"
"""
"""Runs as `sh -c WRAPPER sh <tag> <argv...>`: records its PID, marks the start, and reports the exit status.

The PID file outlives the command: background children that hold its output open keep its process group, and
`docker exec`, going, so a timeout still has a group to stop. Each wrapper first removes the files whose process
and group are both gone. An empty file is one another wrapper is still writing.

A subshell `exec` runs the program itself, never a builtin, and exits 127 when it's missing.
"""

_STOP = f"""pidfile={_PID_DIR}/$1.pid
stopped={_PID_DIR}/$1.stopped
: > "$stopped"
read -r pid 2> /dev/null < "$pidfile" || exit 0
[ -n "$pid" ] || exit 0
kill -KILL "-$pid" "$pid" 2> /dev/null
rm -f "$pidfile" "$stopped"
exit 0"""
"""Stop the command recorded under the tag `$1`, and its process group.

A timeout can come before the wrapper has recorded its PID, even before it starts. So the stop first leaves a
`.stopped` marker, which a wrapper that records its PID afterwards finds before it starts the command: whichever
of the two comes second sees the other's file. A marker for a command that already finished stays behind, empty.

Everything before the `rm` is a shell builtin, so the stop still works when the command has used up the
container's process limit and nothing can fork. It kills at once: the command is already past its timeout.

A `docker exec` process leads a session of its own in both runc and crun, so its group is the command's; a
command that started a session of its own (the harness `Shell`'s jobs) is in another group and keeps running.
Groups are signalled as `kill -SIGNAL -PGID`, without `--`, which BusyBox's `kill` rejects, as does dash's after `-0`.
"""

_STOP_TIMEOUT = 2.0
"""Bounds how late a stopped command's timeout or cancellation is raised when the daemon stops answering."""

_LABEL = 'ai.pydantic.workspace'
"""Marks the containers this capability created; only those can be attached to or removed through a ref."""

_KEEPALIVE = 'while :; do sleep 86400; done'
"""The container's main process: it only has to outlive every command, and `--init` reaps their orphans."""

_ENV_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_CLIENT_ENV = (
    'DOCKER_HOST',
    'DOCKER_CONTEXT',
    'DOCKER_CONFIG',
    'DOCKER_CERT_PATH',
    'DOCKER_TLS_VERIFY',
    'SSH_AUTH_SOCK',
    'CONTAINER_HOST',
    'CONTAINERS_CONF',
    'XDG_RUNTIME_DIR',
)
"""Passed to the local `docker` client, on top of `LocalWorkspaceBackend`'s, so it finds the daemon it's configured for."""


def _after_ready(stderr: str) -> str:
    return stderr.partition(_READY)[2]


def _checked_env(env: Mapping[str, str]) -> dict[str, str]:
    for name in env:
        if not _ENV_NAME.fullmatch(name):
            raise ValueError(f'invalid environment variable name: {name!r}')
    return dict(env)


def _runner() -> LocalWorkspaceBackend:
    """A local subprocess runner: it owns timeouts, output limits and killing `docker` on cancellation."""
    return LocalWorkspaceBackend('/', env={name: os.environ[name] for name in _CLIENT_ENV if name in os.environ})


def _is_missing(result: CommandResult) -> bool:
    return 'no such container' in result.stderr.lower()


async def _owned_state(
    runner: LocalWorkspaceBackend, executable: str, name: str, error: type[WorkspaceError]
) -> Literal['running', 'stopped', 'missing']:
    """The state of the container `name`, raising `error` unless this capability created it.

    A ref can come from message history, and the daemon serves every container on the machine: without this
    check, a crafted ref could run commands in, or remove, an unrelated container.
    """
    template = f'{{{{index .Config.Labels "{_LABEL}"}}}} {{{{.State.Running}}}}'
    result = await runner.run([executable, 'inspect', '--type', 'container', '--format', template, '--', name])
    if result.exit_code != 0:
        if _is_missing(result):
            return 'missing'
        raise error(f'could not inspect container {name}: {result.stderr.strip()}')
    label, _, running = result.stdout.strip().rpartition(' ')
    if label != 'true':
        raise error(f'container {name} was not created by `DockerSandbox`: it lacks the label {_LABEL}=true')
    return 'running' if running == 'true' else 'stopped'


def resolve_executable(executable: str) -> str:
    """The container CLI's path on `PATH` now.

    A `docker` that appears on `PATH` later, such as in a mounted project's virtualenv, then isn't the one that runs.
    """
    path = shutil.which(executable)
    return os.path.abspath(path) if path else executable


async def remove_container(name: str, *, executable: str = 'docker') -> None:
    """Remove a container this capability created, with its anonymous volumes; a missing one is fine."""
    executable = resolve_executable(executable)
    runner = _runner()
    if await _owned_state(runner, executable, name, WorkspaceError) == 'missing':
        return
    result = await runner.run([executable, 'rm', '--force', '--volumes', '--', name])
    if result.exit_code != 0 and not _is_missing(result):
        raise WorkspaceError(f'could not remove container {name}: {result.stderr.strip()}')


class DockerSandboxBackend(WorkspaceBackend, SupportsCommands):
    """Run commands in a local Docker container, created on first use, through the `docker` CLI.

    The image needs a POSIX `sh` and the usual file utilities; its own entrypoint and command don't run. A
    `podman` binary works too, through `executable`.

    Built without a ref, the first operation creates a container; built with one, it starts that container
    if it is stopped and raises [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError]
    if it is gone. The backend never removes the container: call
    [`DockerSandbox.destroy`][pydantic_ai_harness.docker_sandbox.DockerSandbox.destroy] with its ref.

    Args:
        image: The image for a new container. Pass it or `ref`, not both.
        ref: A `WorkspaceRef(provider='docker', id=<container name>)` to attach to instead of creating one.
        working_dir: The absolute directory in the container where commands start and relative paths
            resolve; created on first use if the container doesn't have it. `None` uses the image's own
            (`WORKDIR`, or `/`).
        env: Environment variables for every command, on top of the image's; the per-call `env` goes on top.
        allow_internet_access: Whether a new container gets outbound network access, including this machine's
            local network; `False` runs it with `--network none`.
        docker_args: Extra `docker run` arguments for a new container, such as `['--memory', '2g']` or
            `['--volume', f'{project}:/workspace']`. Pass `--network` only with `allow_internet_access=True`. They
            can't change the container's name, workspace label or entrypoint. A `--pids-limit` here replaces the
            default of 2048.
        executable: The container CLI to run, such as `'podman'`.
    """

    def __init__(
        self,
        *,
        image: str | None = None,
        ref: WorkspaceRef | None = None,
        working_dir: str | None = '/workspace',
        env: Mapping[str, str] | None = None,
        allow_internet_access: bool = True,
        docker_args: Sequence[str] = (),
        executable: str = 'docker',
    ):
        if (image is None) == (ref is None):
            raise ValueError('pass exactly one of `image`, to create a container, or `ref`, to attach to one')
        if ref is not None and ref.provider != PROVIDER:
            raise ValueError(f'unsupported workspace provider {ref.provider!r}; expected {PROVIDER!r}')
        if image is not None and (not image or image.startswith('-')):
            raise ValueError(f'image must be an image reference, got {image!r}')
        if working_dir is not None and not posixpath.isabs(working_dir):
            raise ValueError(f'working_dir must be an absolute path in the container, got {working_dir!r}')
        if isinstance(docker_args, str):
            raise TypeError('docker_args must be a sequence of arguments, not a string')
        self._image = image
        self._ref = ref
        self._working_dir = None if working_dir is None else posixpath.normpath(working_dir)
        self._env = _checked_env(env or {})
        self._allow_internet_access = allow_internet_access
        self._docker_args = tuple(docker_args)
        self._executable = resolve_executable(executable)
        self._runner = _runner()
        self._lock = anyio.Lock()
        self._resolved_working_dir: str | None = None
        self._creating = False

    @property
    def ref(self) -> WorkspaceRef | None:
        """`WorkspaceRef(provider='docker', id=<container name>)`, set when the first operation starts creating the container."""
        return self._ref

    async def working_dir(self) -> str:
        return await self._acquire()

    async def _acquire(self) -> str:
        """Create or start the container once, however many first operations race, and return the working directory."""
        async with self._lock:
            if self._resolved_working_dir is None:
                if self._ref is None:
                    await self._create()
                else:
                    await self._start()
                # Created here rather than by `docker run --workdir`, which Podman refuses for a missing directory.
                if self._working_dir is None:
                    result = await self._exec(['sh', '-c', 'pwd -P'], env={}, timeout=None, workdir=None)
                else:
                    script = 'mkdir -p "$1" && cd "$1" && pwd -P'
                    result = await self._exec(
                        ['sh', '-c', script, 'sh', self._working_dir], env={}, timeout=None, workdir='/'
                    )
                printed = result.stdout.removesuffix('\n')
                if result.exit_code != 0 or not posixpath.isabs(printed):
                    raise WorkspaceError(
                        f'could not create or enter the working directory of Docker workspace {self._name}: '
                        f'exit code {result.exit_code}: {result.stderr.strip() or result.stdout!r}'
                    )
                self._resolved_working_dir = printed
            return self._resolved_working_dir

    @property
    def _name(self) -> str:
        assert self._ref is not None
        return self._ref.id

    async def _create(self) -> None:
        assert self._image is not None
        if self._ref is None:
            # Set before `docker run`: the daemon may create the container even if the client is cancelled,
            # and whoever holds the ref can then remove it.
            self._ref = WorkspaceRef(provider=PROVIDER, id=f'pydantic-ai-{secrets.token_hex(6)}')
        argv = [
            self._executable,
            'run',
            '--detach',
            '--init',
            *([] if self._allow_internet_access else ['--network', 'none']),
            # Before `docker_args`, so a `--pids-limit` there wins.
            '--pids-limit',
            _PIDS_LIMIT,
            *self._docker_args,
            # After `docker_args`, whose later values would otherwise win: the backend relies on these.
            '--name',
            self._name,
            '--label',
            f'{_LABEL}=true',
            '--tmpfs',
            _PID_DIR,
            '--entrypoint',
            'sh',
            '--',
            self._image,
            '-c',
            _KEEPALIVE,
        ]
        # Stays set if the client is cancelled, when the daemon may or may not go on to create the container.
        self._creating = True
        result = await self._runner.run(argv)
        self._creating = False
        if result.exit_code != 0:
            error = f'could not create a container from {self._image}: {result.stderr.strip()}'
            # Keep the ref only for a container that exists but didn't start, so it isn't recorded for a later run
            # that would find nothing: the next use then creates a fresh one.
            inspected = await self._runner.run([self._executable, 'inspect', '--type', 'container', '--', self._name])
            if inspected.exit_code != 0:
                self._ref = None
            raise WorkspaceUnavailableError(error)

    async def _start(self) -> None:
        state = await _owned_state(self._runner, self._executable, self._name, WorkspaceUnavailableError)
        if state == 'missing' and self._creating:
            # This backend's `docker run` was cancelled before the container existed: run it again under the same
            # name, which Docker refuses if the first one created it after all.
            await self._create()
            return
        if state == 'missing':
            raise WorkspaceUnavailableError(f'Docker workspace {self._name} is unavailable: no such container')
        if state == 'running':
            return
        result = await self._runner.run([self._executable, 'start', '--', self._name])
        if result.exit_code != 0:
            raise WorkspaceUnavailableError(f'Docker workspace {self._name} is unavailable: {result.stderr.strip()}')

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        check_timeout(timeout)
        argv = command_argv(command, shell)
        merged_env = {**self._env, **_checked_env(env or {})}
        # The first command also creates the container, which isn't bounded by the command's timeout:
        # pulling an image can take minutes.
        # Commands start where `working_dir()` says they do, even if a symlink on the way changes.
        workdir = await self._acquire()
        return await self._exec(argv, env=merged_env, timeout=timeout, workdir=workdir)

    async def _exec(
        self, argv: Sequence[str], *, env: Mapping[str, str], timeout: float | None, workdir: str | None
    ) -> CommandResult:
        tag = secrets.token_hex(8)
        env_args = [arg for name, value in env.items() for arg in ('--env', f'{name}={value}')]
        workdir_args = [] if workdir is None else ['--workdir', workdir]
        docker = [self._executable, 'exec', *workdir_args, *env_args, self._name]
        try:
            result = await self._runner.run([*docker, 'sh', '-c', _WRAPPER, 'sh', tag, *argv], timeout=timeout)
        except anyio.get_cancelled_exc_class():
            await self._stop(tag)
            raise
        except WorkspaceTimeoutError as error:
            await self._stop(tag)
            raise WorkspaceTimeoutError(str(error), stdout=error.stdout, stderr=_after_ready(error.stderr)) from error
        except WorkspaceOutputLimitError as error:
            await self._stop(tag)
            raise WorkspaceOutputLimitError(
                "Docker workspace output exceeded its 10 MiB safety limit; redirect the command's output to a "
                'file and read part of it instead',
                limit=error.limit,
                stdout=error.stdout,
                stderr=_after_ready(error.stderr),
            ) from error
        before, ready, stderr = result.stderr.partition(_READY)
        if not ready:
            # Docker prints some of its own errors, such as a missing `--workdir`, on stdout.
            reason = (
                before.strip()
                or result.stdout.strip()
                or f'`{posixpath.basename(self._executable)} exec` exited with code {result.exit_code}'
            )
            raise WorkspaceUnavailableError(f'Docker workspace {self._name} is unavailable: {reason}')
        # The command can learn its tag from the PID directory and print a marker of its own, so the wrapper's is
        # the last one with the tag whose status is also `docker exec`'s exit code, which the wrapper exits with.
        done = next(
            (
                marker
                for marker in reversed(list(_DONE.finditer(stderr)))
                if marker[1] == tag and int(marker[2]) == result.exit_code
            ),
            None,
        )
        if done is None:
            raise WorkspaceUnavailableError(f'Docker workspace {self._name}: the container stopped during the command')
        return CommandResult(
            exit_code=int(done[2]), stdout=result.stdout, stderr=stderr[: done.start()] + stderr[done.end() :]
        )

    async def _stop(self, tag: str) -> None:
        # Killing the local `docker exec` client leaves the command running in the container, so a second
        # exec stops it. Best effort: a daemon that stalls now has nothing to report.
        argv = [self._executable, 'exec', '--workdir', '/', self._name, 'sh', '-c', _STOP, 'sh', tag]
        await stop_shielded(lambda: self._runner.run(argv), grace=_STOP_TIMEOUT)

"""A [workspace backend][pydantic_ai.workspaces.WorkspaceBackend] in a local Docker (or Podman) container."""

from __future__ import annotations as _annotations

import os
import posixpath
import re
import secrets
from collections.abc import Mapping, Sequence

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
from pydantic_ai_harness._workspace_provider import check_timeout, command_argv

__all__ = ('DockerSandboxBackend', 'remove_container')

PROVIDER = 'docker'

_READY = '__pydantic_ai_docker_ready__\n'
"""Printed on stderr before the command starts; anything before it is the `docker` client's own error."""
_DONE = re.compile(r'\n__pydantic_ai_docker_done__([0-9a-f]+):(\d+)\n')
"""Carries the command's exit status, so a container that dies mid-command isn't read as a result."""

_PID_DIR = '/tmp'
"""Where each command records its process ID, so a second `docker exec` can stop it on a timeout."""

_WRAPPER = f"""tag=$1
pidfile={_PID_DIR}/.pydantic-ai-$tag.pid
shift
if ! echo $$ 2> /dev/null > "$pidfile"; then
    echo "cannot write $pidfile, which stops the command on a timeout: {_PID_DIR} must be writable" >&2
    exit 1
fi
printf '%s' '{_READY}' >&2
(exec "$@")
status=$?
rm -f "$pidfile"
printf '\\n__pydantic_ai_docker_done__%s:%d\\n' "$tag" "$status" >&2
exit "$status"
"""
"""Runs as `sh -c WRAPPER sh <tag> <argv...>`: records its PID, marks the start, and reports the exit status.

A subshell `exec` runs the program itself, never a builtin, and exits 127 when it's missing.
"""

_STOP = f"""pidfile={_PID_DIR}/.pydantic-ai-$1.pid
[ -f "$pidfile" ] || exit 0
pid=$(cat "$pidfile")
rm -f "$pidfile"
kill -s TERM -- "-$pid" "$pid" 2> /dev/null
(sleep 1; kill -s KILL -- "-$pid" "$pid" 2> /dev/null) < /dev/null > /dev/null 2>&1 &
exit 0"""
"""Stop the command recorded under the tag `$1`, and its process group.

A `docker exec` process leads a session of its own in both runc and crun, so its group is the command's; a
command that started a session of its own (the harness `Shell`'s jobs) is in another group and keeps running.
"""

_STOP_TIMEOUT = 2.0
"""Bounds how late a stopped command's timeout or cancellation is raised when the daemon stops answering."""

_KEEPALIVE = 'while :; do sleep 86400; done'
"""The container's main process: it only has to outlive every command, and `--init` reaps their orphans."""

_ENV_NAME = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')
_CLIENT_ENV = (
    'DOCKER_HOST',
    'DOCKER_CONTEXT',
    'DOCKER_CONFIG',
    'DOCKER_CERT_PATH',
    'DOCKER_TLS_VERIFY',
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


async def remove_container(name: str, *, executable: str = 'docker') -> None:
    """Remove a container and its anonymous volumes, whether it is running or not; a missing one is fine."""
    result = await _runner().run([executable, 'rm', '--force', '--volumes', '--', name])
    if result.exit_code != 0 and 'no such container' not in result.stderr.lower():
        raise WorkspaceError(f'could not remove container {name}: {result.stderr.strip()}')


class DockerSandboxBackend(WorkspaceBackend, SupportsCommands):
    """Run commands in a local Docker container, created on first use, through the `docker` CLI.

    The container runs `sh` as its entrypoint under `--init`, so the image needs a POSIX `sh` and the usual
    file utilities: file operations run as shell commands in the container (see [Writing a
    backend](https://pydantic.dev/docs/ai/core-concepts/workspace/#writing-a-backend)). A `podman` binary
    works too, through `executable`.

    Built without a ref, the first operation creates a container; built with one, it starts that container
    if it is stopped and raises [`WorkspaceUnavailableError`][pydantic_ai.workspaces.WorkspaceUnavailableError]
    if it is gone. The backend never removes the container: call
    [`DockerSandbox.destroy`][pydantic_ai_harness.docker_sandbox.DockerSandbox.destroy] with its ref.

    Args:
        image: The image for a new container; ignored when attaching to `ref`.
        ref: A `WorkspaceRef(provider='docker', id=<container name>)` to attach to instead of creating one.
        working_dir: The absolute directory in the container where commands start and relative paths
            resolve; created if the image doesn't have it.
        env: Environment variables for every command, on top of the image's; the per-call `env` goes on top.
        network: Whether a new container can reach the network; `False` runs it with `--network none`.
        docker_args: Extra `docker run` arguments for a new container, such as `['--memory', '2g']` or
            `['--volume', f'{project}:/workspace']`, placed after the defaults so they can override them.
        executable: The container CLI to run, such as `'podman'`.
    """

    def __init__(
        self,
        image: str | None = None,
        *,
        ref: WorkspaceRef | None = None,
        working_dir: str = '/workspace',
        env: Mapping[str, str] | None = None,
        network: bool = True,
        docker_args: Sequence[str] = (),
        executable: str = 'docker',
    ):
        if (image is None) == (ref is None):
            raise ValueError('pass exactly one of `image`, to create a container, or `ref`, to attach to one')
        if ref is not None and ref.provider != PROVIDER:
            raise ValueError(f'unsupported workspace provider {ref.provider!r}; expected {PROVIDER!r}')
        if image is not None and (not image or image.startswith('-')):
            raise ValueError(f'image must be an image reference, got {image!r}')
        if not posixpath.isabs(working_dir):
            raise ValueError(f'working_dir must be an absolute path in the container, got {working_dir!r}')
        if isinstance(docker_args, str):
            raise TypeError('docker_args must be a sequence of arguments, not a string')
        self._image = image
        self._ref = ref
        self._working_dir = posixpath.normpath(working_dir)
        self._env = _checked_env(env or {})
        self._network = network
        self._docker_args = tuple(docker_args)
        self._executable = executable
        self._runner = _runner()
        self._lock = anyio.Lock()
        self._resolved_working_dir: str | None = None

    @property
    def ref(self) -> WorkspaceRef | None:
        """`WorkspaceRef(provider='docker', id=<container name>)`, set once the container is created."""
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
                # `docker exec --workdir` fails before the command starts if the directory is missing.
                result = await self._exec(['pwd', '-P'], env={}, timeout=None)
                self._resolved_working_dir = result.stdout.removesuffix('\n')
            return self._resolved_working_dir

    @property
    def _name(self) -> str:
        assert self._ref is not None
        return self._ref.id

    async def _create(self) -> None:
        assert self._image is not None
        name = f'pydantic-ai-{secrets.token_hex(6)}'
        argv = [
            self._executable,
            'run',
            '--detach',
            '--init',
            '--name',
            name,
            '--label',
            'ai.pydantic.workspace=true',
            '--workdir',
            self._working_dir,
            '--entrypoint',
            'sh',
            *([] if self._network else ['--network', 'none']),
            *self._docker_args,
            '--',
            self._image,
            '-c',
            _KEEPALIVE,
        ]
        try:
            result = await self._runner.run(argv)
        except BaseException:
            # The daemon may have created the container before the client was stopped: keep its name so
            # whoever holds the ref can remove it.
            self._ref = WorkspaceRef(provider=PROVIDER, id=name)
            raise
        self._ref = WorkspaceRef(provider=PROVIDER, id=name)
        if result.exit_code != 0:
            raise WorkspaceUnavailableError(f'could not create a container from {self._image}: {result.stderr.strip()}')

    async def _start(self) -> None:
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
        await self._acquire()
        return await self._exec(argv, env=merged_env, timeout=timeout)

    async def _exec(self, argv: Sequence[str], *, env: Mapping[str, str], timeout: float | None) -> CommandResult:
        tag = secrets.token_hex(8)
        env_args = [arg for name, value in env.items() for arg in ('--env', f'{name}={value}')]
        # Once resolved, commands start where `working_dir()` says they do, even if a symlink on the way changes.
        workdir = self._resolved_working_dir or self._working_dir
        docker = [self._executable, 'exec', '--workdir', workdir, *env_args, self._name]
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
            reason = before.strip() or f'`{self._executable} exec` exited with code {result.exit_code}'
            raise WorkspaceUnavailableError(f'Docker workspace {self._name} is unavailable: {reason}')
        # A background child that kept stderr open can write around the marker, so match this command's tag.
        done = next((marker for marker in _DONE.finditer(stderr) if marker[1] == tag), None)
        if done is None:
            raise WorkspaceUnavailableError(f'Docker workspace {self._name}: the container stopped during the command')
        return CommandResult(
            exit_code=int(done[2]), stdout=result.stdout, stderr=stderr[: done.start()] + stderr[done.end() :]
        )

    async def _stop(self, tag: str) -> None:
        # Killing the local `docker exec` client leaves the command running in the container, so a second
        # exec stops it. Best effort: a daemon that stalls now has nothing to report.
        with anyio.CancelScope(shield=True):
            try:
                await self._runner.run(
                    [self._executable, 'exec', self._name, 'sh', '-c', _STOP, 'sh', tag], timeout=_STOP_TIMEOUT
                )
            except WorkspaceError:
                pass

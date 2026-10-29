"""Pydantic AI's workspace backend conformance suite, run against real containers.

Opt in with `PYDANTIC_AI_HARNESS_DOCKER_LIVE=1` on a machine with a Docker daemon:
`PYDANTIC_AI_HARNESS_DOCKER_LIVE=1 uv run pytest -m docker_live tests/harness/docker_sandbox`.
Set `PYDANTIC_AI_HARNESS_DOCKER_EXECUTABLE=podman` to run it against Podman.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import pytest

from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef, WorkspaceTimeoutError
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend

LIVE_IMAGE = 'python:3.13-slim'
EXECUTABLE = os.environ.get('PYDANTIC_AI_HARNESS_DOCKER_EXECUTABLE', 'docker')

_FORK_UNTIL_FULL = """
import os, time
while True:
    try:
        pid = os.fork()
    except OSError:
        time.sleep(300)
    else:
        if pid == 0:
            time.sleep(300)
            os._exit(0)
"""
"""Forks sleeping children until the container's process limit refuses more."""

_ALIVE = "grep -qs '^State:[[:space:]]*[^Z]' {}"
"""Exits 0 when a `/proc/<pid>/status` file names a live process: a killed one is gone or a zombie `--init` reaps."""


def _backend(docker_args: Sequence[str] = ()) -> DockerSandboxBackend:
    return DockerSandboxBackend(image=LIVE_IMAGE, docker_args=docker_args, executable=EXECUTABLE)


async def _destroy(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await DockerSandbox[None](executable=EXECUTABLE).destroy(backend.ref)


@pytest.mark.docker_live
class TestLiveDocker(WorkspaceBackendSuite):
    """Runs against real containers; each test's containers are removed afterwards.

    A timeout's stop finishes before `WorkspaceTimeoutError` is raised, so the tests check once that the
    command is dead rather than waiting for it to die.
    """

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # Docker's `exec` waits until every copy of the command's output is closed; Podman's returns at exit.
        return EXECUTABLE == 'podman'

    @pytest.fixture
    async def created(self) -> AsyncIterator[list[WorkspaceBackend]]:
        backends: list[WorkspaceBackend] = []
        yield backends
        for backend in backends:
            if backend.ref is not None:
                await _destroy(backend)

    @pytest.fixture
    def fresh_backend(self, created: list[WorkspaceBackend]) -> Callable[[], WorkspaceBackend]:
        def build() -> WorkspaceBackend:
            backend = _backend()
            created.append(backend)
            return backend

        return build

    @pytest.fixture
    def backend(self, fresh_backend: Callable[[], WorkspaceBackend]) -> WorkspaceBackend:
        return fresh_backend()

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: DockerSandboxBackend(ref=ref, executable=EXECUTABLE)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy

    @pytest.mark.skipif(EXECUTABLE == 'podman', reason='`podman exec` returns at exit without waiting for the output')
    async def test_a_timeout_stops_background_children_that_hold_the_output(self, backend: WorkspaceBackend) -> None:
        # `docker exec` waits for the output to close, and killing the client leaves the container's processes
        # running: a fake can't reproduce this, because its processes die with the client.
        assert isinstance(backend, DockerSandboxBackend)
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run("sh -c 'echo $$ > child.pid; exec sleep 300' &", shell=True, timeout=2)
        pid = (await backend.run(['cat', 'child.pid'])).stdout.strip()
        assert pid.isdigit()

        assert (await backend.run(_ALIVE.format(f'/proc/{pid}/status'), shell=True)).exit_code != 0

    async def test_a_timeout_stops_a_command_that_cleared_tmp(self, backend: WorkspaceBackend) -> None:
        # The PID files live outside `/tmp`, so clearing it doesn't stop the timeout from finding the command.
        assert isinstance(backend, DockerSandboxBackend)
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run('rm -rf /tmp/* /tmp/.[!.]*; echo $$ > child.pid; exec sleep 300', shell=True, timeout=2)
        pid = (await backend.run(['cat', 'child.pid'])).stdout.strip()
        assert pid.isdigit()

        assert (await backend.run(_ALIVE.format(f'/proc/{pid}/status'), shell=True)).exit_code != 0

    async def test_new_containers_get_a_process_limit_docker_args_can_replace(
        self, created: list[WorkspaceBackend]
    ) -> None:
        # A set value reads back as itself on any host; what `-1` reads back as depends on the host's cgroup setup.
        default = _backend()
        custom = _backend(docker_args=['--pids-limit', '100'])
        created += [default, custom]

        assert (await default.run(['cat', '/sys/fs/cgroup/pids.max'])).stdout.strip() == '2048'
        assert (await custom.run(['cat', '/sys/fs/cgroup/pids.max'])).stdout.strip() == '100'

    async def test_a_timeout_stops_a_command_that_used_up_the_process_limit(
        self, created: list[WorkspaceBackend]
    ) -> None:
        # With no process left to fork, the stop has to manage with shell builtins.
        backend = _backend(docker_args=['--pids-limit', '32'])
        created.append(backend)
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['python', '-c', _FORK_UNTIL_FULL], timeout=3)

        alive = await backend.run(
            f'for p in $(grep -lsx python /proc/[0-9]*/comm); do {_ALIVE.format("${p%comm}status")} && exit 0; done; exit 1',
            shell=True,
        )
        assert alive.exit_code != 0

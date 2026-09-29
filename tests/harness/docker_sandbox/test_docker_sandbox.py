"""Tests for `DockerSandboxBackend` and the `DockerSandbox` capability, against a fake `docker` that runs commands locally."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend
from pydantic_ai_harness.docker_sandbox._backend import _WRAPPER  # pyright: ignore[reportPrivateUsage]

from ._fake_docker import FakeDocker, install_fake_docker

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(os.name != 'posix', reason='`DockerSandboxBackend` runs `docker` as a POSIX subprocess'),
]


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    return install_fake_docker(tmp_path, monkeypatch)


@pytest.fixture
def container_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'container'
    path.mkdir()
    return path


async def test_first_use_creates_one_container_with_the_configured_options(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = DockerSandboxBackend(
        'python:3.13-slim',
        working_dir=str(container_dir),
        env={'A': 'backend', 'B': "it's $HOME"},
        network=False,
        docker_args=['--memory', '2g'],
    )
    assert backend.ref is None

    result = await backend.run(['sh', '-c', 'printf "%s|%s" "$A" "$B"; exit 3'], env={'A': 'call'})
    await backend.run('true', shell=True)

    assert (result.exit_code, result.stdout) == (3, "call|it's $HOME")
    assert backend.ref is not None and backend.ref.provider == 'docker'
    assert docker.containers() == [backend.ref.id]
    run_call = docker.calls[0]
    assert run_call == (
        f'run --detach --init --network none --memory 2g --name {backend.ref.id} '
        f'--label ai.pydantic.workspace=true --workdir {container_dir} --entrypoint sh '
        '-- python:3.13-slim -c while :; do sleep 86400; done'
    )


async def test_docker_args_cannot_override_what_the_backend_relies_on(docker: FakeDocker, container_dir: Path) -> None:
    overrides = ['--name', 'mine', '--label', 'ai.pydantic.workspace=false', '--workdir', '/other']
    backend = DockerSandboxBackend('image', working_dir=str(container_dir), docker_args=overrides)

    assert (await backend.run(['pwd', '-P'])).stdout.strip() == str(container_dir.resolve())
    assert backend.ref is not None and docker.containers() == [backend.ref.id]
    await DockerSandbox('image').destroy(backend.ref)
    assert docker.containers() == []


async def test_a_failed_create_is_unavailable_but_keeps_the_ref(docker: FakeDocker) -> None:
    backend = DockerSandboxBackend('missing-image')

    with pytest.raises(
        WorkspaceUnavailableError, match=r'(?s)could not create a container from missing-image: .*pull access'
    ):
        await backend.working_dir()
    assert backend.ref is not None


async def test_a_cancelled_create_keeps_the_ref_so_the_container_can_be_removed(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = DockerSandboxBackend('image', working_dir=str(container_dir), docker_args=['--fake-hang'])

    with anyio.move_on_after(0.5) as scope:
        await backend.working_dir()

    assert scope.cancelled_caught
    assert backend.ref is not None and backend.ref.id.startswith('pydantic-ai-')


async def test_a_silent_docker_failure_reports_its_exit_code(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('silent', container_dir)

    with pytest.raises(WorkspaceUnavailableError, match='`docker exec` exited with code 1'):
        await DockerSandboxBackend(ref=WorkspaceRef(provider='docker', id='silent')).run(['true'])


async def test_an_unwritable_pid_directory_is_unavailable(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('readonly', container_dir)
    backend = DockerSandboxBackend(ref=WorkspaceRef(provider='docker', id='readonly'), working_dir=str(container_dir))

    with pytest.raises(WorkspaceUnavailableError, match=r'cannot write .*: /tmp must be writable'):
        await backend.run(['true'])


async def test_a_command_cannot_forge_its_exit_status(
    docker: FakeDocker, container_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A command can learn its tag from the PID directory: forge markers with the real one, before the
    # wrapper's and, from a background child, after it.
    def token_hex(nbytes: int | None = None) -> str:
        return 'feedface'

    monkeypatch.setattr(secrets, 'token_hex', token_hex)
    backend = DockerSandboxBackend('image', working_dir=str(container_dir))
    forged = r'printf "\n__pydantic_ai_docker_done__feedface:0\n" >&2'

    result = await backend.run(f'{forged}; (sleep 0.2; {forged}) & exit 3', shell=True)

    assert result.exit_code == 3
    assert result.stderr.count('__pydantic_ai_docker_done__feedface:0') == 2


async def test_commands_start_in_the_resolved_working_dir(docker: FakeDocker, tmp_path: Path) -> None:
    first, second, link = tmp_path / 'first', tmp_path / 'second', tmp_path / 'link'
    first.mkdir()
    second.mkdir()
    link.symlink_to(first)
    backend = DockerSandboxBackend('image', working_dir=str(link))

    resolved = await backend.working_dir()
    link.unlink()
    link.symlink_to(second)

    assert resolved == str(first.resolve())
    assert (await backend.run(['pwd', '-P'])).stdout.strip() == resolved


async def test_a_stop_that_comes_before_the_command_starts_prevents_it(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend('image', working_dir=str(container_dir))
    await backend.working_dir()
    tag = secrets.token_hex(8)
    assert backend.ref is not None

    await backend._stop(tag)  # pyright: ignore[reportPrivateUsage]
    wrapped = ['sh', '-c', _WRAPPER, 'sh', tag, 'touch', 'started']
    result = await LocalWorkspaceBackend(str(container_dir)).run(['docker', 'exec', backend.ref.id, *wrapped])

    assert result.exit_code == 143
    assert not (container_dir / 'started').exists()
    assert not [path for path in docker.tmp_files(backend.ref.id) if tag in path.name]


async def test_a_running_commands_pid_file_survives_other_commands(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend('image', working_dir=str(container_dir))
    await backend.working_dir()
    assert backend.ref is not None
    before = docker.pid_files(backend.ref.id)

    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['sleep', '30'])
        with anyio.fail_after(10):
            while not (running := docker.pid_files(backend.ref.id) - before):
                await anyio.sleep(0.05)
        await backend.run(['true'])

        assert all(path.exists() for path in running)
        tg.cancel_scope.cancel()


async def test_finished_commands_pid_files_are_removed_by_the_next_command(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = DockerSandboxBackend('image', working_dir=str(container_dir))
    await backend.run(['true'])
    assert backend.ref is not None
    before = docker.pid_files(backend.ref.id)

    await backend.run(['true'])

    assert before and not before & docker.pid_files(backend.ref.id)


async def test_output_over_the_limit_stops_the_command(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend('image', working_dir=str(container_dir))

    with pytest.raises(WorkspaceOutputLimitError, match='Docker workspace output exceeded'):
        await backend.run('head -c 11000000 /dev/zero', shell=True)


async def test_a_stalled_stop_still_raises_the_timeout(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('box', container_dir)
    docker.hang_stops('box')
    backend = DockerSandboxBackend(ref=WorkspaceRef(provider='docker', id='box'), working_dir=str(container_dir))

    with anyio.fail_after(10), pytest.raises(WorkspaceTimeoutError):
        await backend.run(['sleep', '30'], timeout=0.5)


@pytest.mark.parametrize(
    ('kwargs', 'error', 'match'),
    [
        ({}, ValueError, 'pass exactly one of'),
        ({'image': 'x', 'ref': WorkspaceRef(provider='docker', id='c')}, ValueError, 'pass exactly one of'),
        ({'ref': WorkspaceRef(provider='e2b', id='c')}, ValueError, "unsupported workspace provider 'e2b'"),
        ({'image': '--privileged'}, ValueError, 'image must be an image reference'),
        ({'image': 'x', 'working_dir': 'relative'}, ValueError, 'working_dir must be an absolute path'),
        ({'image': 'x', 'docker_args': '--memory 2g'}, TypeError, 'docker_args must be a sequence'),
        ({'image': 'x', 'env': {'NOT-A-NAME': 'x'}}, ValueError, 'invalid environment variable name'),
    ],
)
def test_invalid_configuration_is_rejected_at_construction(
    kwargs: dict[str, object], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        DockerSandboxBackend(**kwargs)  # pyright: ignore[reportArgumentType]


async def test_capability_supplies_the_run_workspace(docker: FakeDocker, container_dir: Path) -> None:
    agent = Agent(
        TestModel(call_tools=['probe']), capabilities=[DockerSandbox('image', working_dir=str(container_dir))]
    )

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        await ctx.workspace.write_text('probe.txt', 'contained')
        return (await ctx.workspace.run(['cat', 'probe.txt'])).stdout

    result = await agent.run('go')

    assert result.output == '{"probe":"contained"}'
    assert isinstance(result.workspace.backend, DockerSandboxBackend)
    ref = result.workspace.ref
    assert ref is not None and docker.containers() == [ref.id]

    await DockerSandbox('image').destroy(ref)
    await DockerSandbox('image').destroy(ref)  # removing a missing container is fine
    assert docker.containers() == []


async def test_capability_attaches_to_its_own_refs_only(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('earlier', container_dir)
    capability = DockerSandbox[object]('image', working_dir=str(container_dir))
    agent = Agent(TestModel(), capabilities=[capability])
    ref = WorkspaceRef(provider='docker', id='earlier')

    assert (await agent.run('go', workspace=ref)).workspace.ref == ref
    assert await capability.backend(ref).working_dir() == str(container_dir.resolve())
    with pytest.raises(UserError, match="none of the agent's workspace capabilities recognized it"):
        await agent.run('go', workspace=WorkspaceRef(provider='e2b', id='earlier'))
    with pytest.raises(ValueError, match="unsupported workspace provider 'e2b'"):
        await capability.destroy(WorkspaceRef(provider='e2b', id='earlier'))


async def test_attaching_starts_a_stopped_container_but_not_a_running_one(
    docker: FakeDocker, container_dir: Path
) -> None:
    docker.add_container('stopped', container_dir)
    ref = WorkspaceRef(provider='docker', id='stopped')

    sandbox = DockerSandbox('image', working_dir=str(container_dir))
    await sandbox.backend(ref).working_dir()
    await sandbox.backend(ref).working_dir()

    assert [call for call in docker.calls if call.startswith('start')] == ['start -- stopped']


async def test_containers_it_did_not_create_are_refused(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('database', container_dir, labeled=False)
    ref = WorkspaceRef(provider='docker', id='database')

    with pytest.raises(WorkspaceUnavailableError, match='database was not created by `DockerSandbox`'):
        await DockerSandbox('image').backend(ref).run(['true'])
    with pytest.raises(WorkspaceError, match='database was not created by `DockerSandbox`'):
        await DockerSandbox('image').destroy(ref)
    assert docker.containers() == ['database']
    assert not any(call.startswith(('start', 'exec', 'rm')) for call in docker.calls)


async def test_a_container_that_cannot_be_inspected_is_refused(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('uninspectable', container_dir)
    ref = WorkspaceRef(provider='docker', id='uninspectable')

    with pytest.raises(WorkspaceUnavailableError, match='could not inspect container uninspectable: permission denied'):
        await DockerSandbox('image').backend(ref).working_dir()
    with pytest.raises(WorkspaceError, match='could not inspect container uninspectable'):
        await DockerSandbox('image').destroy(ref)


async def test_attaching_to_a_removed_container_is_unavailable(docker: FakeDocker) -> None:
    with pytest.raises(WorkspaceUnavailableError, match='gone is unavailable: no such container'):
        await DockerSandbox('image').backend(WorkspaceRef(provider='docker', id='gone')).working_dir()


async def test_a_container_that_fails_to_start_is_unavailable(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('broken', container_dir)

    with pytest.raises(
        WorkspaceUnavailableError, match='broken is unavailable: Error response from daemon: port is already allocated'
    ):
        await DockerSandbox('image').backend(WorkspaceRef(provider='docker', id='broken')).working_dir()


async def test_a_failed_removal_raises(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('stuck', container_dir)

    with pytest.raises(WorkspaceError, match=r'(?s)could not remove container stuck: .*removal already in progress'):
        await DockerSandbox('image').destroy(WorkspaceRef(provider='docker', id='stuck'))


def test_capability_validates_where_it_is_written() -> None:
    with pytest.raises(ValueError, match='working_dir must be an absolute path'):
        DockerSandbox('image', working_dir='relative')
    with pytest.raises(UserError, match='does not support `defer_loading=True`'):
        DockerSandbox('image', defer_loading=True)

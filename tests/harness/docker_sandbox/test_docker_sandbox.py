"""Tests for `DockerSandboxBackend` and the `DockerSandbox` capability, against a fake `docker` that runs commands locally."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import LocalWorkspace
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


async def test_first_use_runs_docker_run_with_the_configured_options(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(
        image='python:3.13-slim',
        working_dir=str(container_dir),
        allow_internet_access=False,
        docker_args=['--memory', '2g'],
    )

    await backend.working_dir()

    assert backend.ref is not None
    assert docker.calls[0] == (
        f'run --detach --init --network none --pids-limit 2048 --memory 2g --name {backend.ref.id} '
        '--label ai.pydantic.workspace=true --tmpfs /run/pydantic-ai --entrypoint sh '
        '-- python:3.13-slim -c while :; do sleep 86400; done'
    )


async def test_a_calls_env_goes_on_top_of_the_backends(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(
        image='image', working_dir=str(container_dir), env={'A': 'backend', 'B': "it's $HOME"}
    )

    result = await backend.run(['sh', '-c', 'printf "%s|%s" "$A" "$B"'], env={'A': 'call'})

    assert result.stdout == "call|it's $HOME"


async def test_new_containers_keep_dockers_default_network(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir))

    await backend.working_dir()

    assert '--network' not in docker.calls[0]


async def test_a_container_that_failed_to_start_keeps_its_ref(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='unstartable-image', working_dir=str(container_dir))
    with pytest.raises(WorkspaceUnavailableError, match='port is already allocated'):
        await backend.working_dir()

    assert backend.ref is not None and docker.containers() == [backend.ref.id]


async def test_a_create_that_made_no_container_leaves_no_ref_and_is_retried(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = DockerSandboxBackend(image='flaky-image', working_dir=str(container_dir))
    with pytest.raises(WorkspaceUnavailableError, match=r'could not create a container from flaky-image: .*TLS'):
        await backend.working_dir()
    assert backend.ref is None

    await backend.working_dir()

    assert backend.ref is not None and docker.containers() == [backend.ref.id]


async def test_a_cancelled_create_is_retried_under_the_same_name(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir), docker_args=['--fake-hang'])
    with anyio.move_on_after(0.5) as scope:
        await backend.working_dir()
    assert scope.cancelled_caught and backend.ref is not None
    ref = backend.ref

    await backend.working_dir()

    assert backend.ref == ref and docker.containers() == [ref.id]


async def test_a_create_cancelled_after_docker_made_the_container_keeps_that_container(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir), docker_args=['--fake-hang-late'])
    with anyio.move_on_after(0.5) as scope:
        await backend.working_dir()
    assert scope.cancelled_caught and backend.ref is not None

    await backend.working_dir()

    assert docker.containers() == [backend.ref.id]


async def test_a_silent_docker_failure_reports_its_exit_code(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('silent', container_dir)

    with pytest.raises(WorkspaceUnavailableError, match='`docker exec` exited with code 1'):
        await DockerSandboxBackend(ref=WorkspaceRef(provider='docker', id='silent')).run(['true'])


async def test_an_unwritable_pid_directory_is_unavailable(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('readonly', container_dir)
    backend = DockerSandboxBackend(ref=WorkspaceRef(provider='docker', id='readonly'), working_dir=str(container_dir))

    with pytest.raises(WorkspaceUnavailableError, match=r'cannot write .*: /run/pydantic-ai must be writable'):
        await backend.run(['true'])


async def test_a_command_cannot_forge_its_exit_status(
    docker: FakeDocker, container_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A command can learn its tag from the PID directory: forge markers with the real one, before the
    # wrapper's and, from a background child, after it.
    def token_hex(nbytes: int | None = None) -> str:
        return 'feedface'

    monkeypatch.setattr(secrets, 'token_hex', token_hex)
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir))
    forged = r'printf "\n__pydantic_ai_docker_done__feedface:0\n" >&2'

    result = await backend.run(f'{forged}; (sleep 0.2; {forged}) & exit 3', shell=True)

    assert result.exit_code == 3
    assert result.stderr.count('__pydantic_ai_docker_done__feedface:0') == 2


async def test_commands_start_in_the_resolved_working_dir(docker: FakeDocker, tmp_path: Path) -> None:
    first, second, link = tmp_path / 'first', tmp_path / 'second', tmp_path / 'link'
    first.mkdir()
    second.mkdir()
    link.symlink_to(first)
    backend = DockerSandboxBackend(image='image', working_dir=str(link))

    resolved = await backend.working_dir()
    link.unlink()
    link.symlink_to(second)

    assert resolved == str(first.resolve())
    assert (await backend.run(['pwd', '-P'])).stdout.strip() == resolved


async def test_a_stop_that_comes_before_the_command_starts_prevents_it(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir))
    await backend.working_dir()
    tag = secrets.token_hex(8)
    assert backend.ref is not None

    await backend._stop(tag)  # pyright: ignore[reportPrivateUsage]
    wrapped = ['sh', '-c', _WRAPPER, 'sh', tag, 'touch', 'started']
    result = await LocalWorkspaceBackend(str(container_dir)).run(['docker', 'exec', backend.ref.id, *wrapped])

    assert result.exit_code == 143
    assert not (container_dir / 'started').exists()
    assert not [path for path in docker.tmp_files(backend.ref.id) if tag in path.name]


async def test_a_timeout_is_stopped_even_when_the_configured_working_dir_is_gone(
    docker: FakeDocker, tmp_path: Path
) -> None:
    target, link = tmp_path / 'target', tmp_path / 'link'
    target.mkdir()
    link.symlink_to(target)
    backend = DockerSandboxBackend(image='image', working_dir=str(link))
    await backend.working_dir()
    link.unlink()
    assert backend.ref is not None

    with pytest.raises(WorkspaceTimeoutError):
        await backend.run(['sleep', '30'], timeout=0.5)

    # The stop removes the command's PID file: it ran, although the container's own working dir is gone.
    assert docker.pid_files(backend.ref.id) == set()


async def test_a_running_commands_pid_file_survives_other_commands(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir))
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
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir))
    await backend.run(['true'])
    assert backend.ref is not None
    before = docker.pid_files(backend.ref.id)

    await backend.run(['true'])

    assert before and not before & docker.pid_files(backend.ref.id)


async def test_output_over_the_limit_stops_the_command(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir))
    await backend.working_dir()
    assert backend.ref is not None

    with pytest.raises(WorkspaceOutputLimitError, match='Docker workspace output exceeded'):
        await backend.run('cat /dev/zero', shell=True)

    # Only the stop removes the command's PID file.
    assert docker.pid_files(backend.ref.id) == set()


async def test_cancelling_a_command_stops_it_in_the_container(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir))
    await backend.working_dir()
    assert backend.ref is not None
    before = docker.pid_files(backend.ref.id)
    running: set[Path] = set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(backend.run, ['sleep', '30'])
        with anyio.fail_after(10):
            while not (running := docker.pid_files(backend.ref.id) - before):
                await anyio.sleep(0.05)
        tg.cancel_scope.cancel()

    # Only the stop removes the command's PID file.
    assert not running & docker.pid_files(backend.ref.id)


async def test_a_timeout_keeps_the_commands_own_output(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir))

    with pytest.raises(WorkspaceTimeoutError) as caught:
        await backend.run('echo out; echo err >&2; sleep 30', shell=True, timeout=1)

    assert (caught.value.stdout, caught.value.stderr) == ('out\n', 'err\n')


async def test_a_calls_env_is_validated_before_anything_runs(docker: FakeDocker) -> None:
    with pytest.raises(ValueError, match="invalid environment variable name: 'A-B'"):
        await DockerSandboxBackend(image='image').run(['true'], env={'A-B': 'x'})

    assert docker.calls == []


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
        TestModel(call_tools=['probe']), capabilities=[DockerSandbox(image='image', working_dir=str(container_dir))]
    )

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        await ctx.workspace.write_text('probe.txt', 'contained')
        return (await ctx.workspace.run(['cat', 'probe.txt'])).stdout

    result = await agent.run('go')

    assert result.output == '{"probe":"contained"}'
    assert result.workspace.ref is not None and docker.containers() == [result.workspace.ref.id]


async def test_destroying_a_removed_container_is_fine(docker: FakeDocker) -> None:
    await DockerSandbox().destroy(WorkspaceRef(provider='docker', id='gone'))


async def test_a_capability_without_an_image_leaves_new_runs_to_other_workspaces(
    docker: FakeDocker, tmp_path: Path
) -> None:
    agent = Agent(TestModel(), capabilities=[DockerSandbox(), LocalWorkspace(str(tmp_path))])

    result = await agent.run('go')

    assert isinstance(result.workspace.backend, LocalWorkspaceBackend) and docker.containers() == []


def test_env_is_kept_out_of_the_repr() -> None:
    assert 'hunter2' not in repr(DockerSandbox(image='image', env={'TOKEN': 'hunter2'}))


async def test_the_docker_client_gets_only_the_variables_that_select_a_daemon(
    docker: FakeDocker, container_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('DOCKER_HOST', 'unix:///run/other.sock')
    monkeypatch.setenv('UNRELATED_TOKEN', 'hunter2')

    await DockerSandboxBackend(image='image', working_dir=str(container_dir)).working_dir()

    client_env = docker.client_env()
    assert 'DOCKER_HOST=unix:///run/other.sock' in client_env and 'hunter2' not in client_env


async def test_the_cli_is_the_one_on_path_when_the_capability_was_created(
    docker: FakeDocker, container_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docker.add_container('box', container_dir)
    sandbox = DockerSandbox(working_dir=str(container_dir))
    later = tmp_path / 'later-bin'
    later.mkdir()
    (later / 'docker').write_text('#!/bin/sh\nexit 99\n')
    (later / 'docker').chmod(0o755)
    monkeypatch.setenv('PATH', f'{later}{os.pathsep}{os.environ["PATH"]}')

    assert await sandbox.backend(WorkspaceRef(provider='docker', id='box')).working_dir() == str(
        container_dir.resolve()
    )


async def test_destroy_uses_the_configured_executable(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('box', container_dir)
    sandbox = DockerSandbox(executable='podman')
    (docker.bin_dir / 'docker').unlink()

    await sandbox.destroy(WorkspaceRef(provider='docker', id='box'))

    assert docker.containers() == []


async def test_a_working_dir_that_cannot_be_created_is_an_error(docker: FakeDocker, container_dir: Path) -> None:
    (container_dir / 'file').touch()
    backend = DockerSandboxBackend(image='image', working_dir=str(container_dir / 'file' / 'workspace'))

    with pytest.raises(WorkspaceError, match=r'could not create or enter the working directory.*Not a directory'):
        await backend.working_dir()


async def test_attaching_creates_a_working_dir_the_container_lacks(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('box', container_dir)
    missing = container_dir / 'missing'

    working_dir = (
        await DockerSandbox(working_dir=str(missing)).backend(WorkspaceRef(provider='docker', id='box')).working_dir()
    )

    assert working_dir == str(missing.resolve())


async def test_without_a_working_dir_commands_start_in_the_images_own(docker: FakeDocker) -> None:
    backend = DockerSandboxBackend(image='image', working_dir=None)

    assert await backend.working_dir() == str(docker.image_workdir.resolve())
    assert '--workdir' not in docker.calls[0]


async def test_capability_attaches_to_its_own_refs_only() -> None:
    agent = Agent(TestModel(), capabilities=[DockerSandbox()])
    ref = WorkspaceRef(provider='docker', id='earlier')

    assert (await agent.run('go', workspace=ref)).workspace.ref == ref
    with pytest.raises(UserError, match="none of the agent's workspace capabilities recognized it"):
        await agent.run('go', workspace=WorkspaceRef(provider='e2b', id='earlier'))


async def test_destroy_refuses_another_providers_ref() -> None:
    with pytest.raises(ValueError, match="unsupported workspace provider 'e2b'"):
        await DockerSandbox().destroy(WorkspaceRef(provider='e2b', id='earlier'))


async def test_attaching_starts_a_stopped_container_but_not_a_running_one(
    docker: FakeDocker, container_dir: Path
) -> None:
    docker.add_container('stopped', container_dir)
    ref = WorkspaceRef(provider='docker', id='stopped')

    sandbox = DockerSandbox(working_dir=str(container_dir))
    await sandbox.backend(ref).working_dir()
    await sandbox.backend(ref).working_dir()

    assert [call for call in docker.calls if call.startswith('start')] == ['start -- stopped']


async def test_containers_it_did_not_create_are_refused(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('database', container_dir, labeled=False)
    ref = WorkspaceRef(provider='docker', id='database')

    with pytest.raises(WorkspaceUnavailableError, match='database was not created by `DockerSandbox`'):
        await DockerSandbox().backend(ref).run(['true'])
    with pytest.raises(WorkspaceError, match='database was not created by `DockerSandbox`'):
        await DockerSandbox().destroy(ref)
    assert docker.containers() == ['database']
    assert not any(call.startswith(('start', 'exec', 'rm')) for call in docker.calls)


async def test_a_container_that_cannot_be_inspected_is_refused(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('uninspectable', container_dir)
    ref = WorkspaceRef(provider='docker', id='uninspectable')

    with pytest.raises(WorkspaceUnavailableError, match='could not inspect container uninspectable: permission denied'):
        await DockerSandbox().backend(ref).working_dir()
    with pytest.raises(WorkspaceError, match='could not inspect container uninspectable'):
        await DockerSandbox().destroy(ref)


async def test_a_container_that_fails_to_start_is_unavailable(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('broken', container_dir)

    with pytest.raises(
        WorkspaceUnavailableError, match='broken is unavailable: Error response from daemon: port is already allocated'
    ):
        await DockerSandbox().backend(WorkspaceRef(provider='docker', id='broken')).working_dir()


async def test_a_failed_removal_raises(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('stuck', container_dir)

    with pytest.raises(WorkspaceError, match=r'(?s)could not remove container stuck: .*removal already in progress'):
        await DockerSandbox().destroy(WorkspaceRef(provider='docker', id='stuck'))


@pytest.mark.parametrize(
    ('kwargs', 'match'),
    [
        ({'working_dir': 'relative'}, 'working_dir must be an absolute POSIX path'),
        ({'defer_loading': True}, 'does not support `defer_loading=True`'),
    ],
)
def test_invalid_capability_configuration_is_rejected_at_construction(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(UserError, match=match):
        DockerSandbox(**kwargs)  # pyright: ignore[reportArgumentType]

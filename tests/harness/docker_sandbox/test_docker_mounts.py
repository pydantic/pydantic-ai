"""Tests for `DockerSandbox(mounts=...)`, against the fake `docker`, whose root `exec` answers for the mount helper.

The helper itself runs only in a real container: see `test_mounts_live.py`.
"""

from __future__ import annotations

import inspect
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness import MountError, S3Mount
from pydantic_ai_harness.docker_sandbox import DockerSandbox

from ._fake_docker import FakeDocker, install_fake_docker

pytestmark = pytest.mark.skipif(os.name != 'posix', reason='`DockerSandboxBackend` runs `docker` as a POSIX subprocess')

FUSE = ['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN', '--security-opt', 'apparmor:unconfined']


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    return install_fake_docker(tmp_path, monkeypatch)


@pytest.fixture
def container_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'container'
    path.mkdir()
    return path


def bucket(key: str = 'the-key-id') -> S3Mount:
    return S3Mount(
        path='data',
        url='s3://bucket/',
        access_key_id=key,
        secret_access_key='the-secret-key',
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        endpoint='http://minio:9000',
    )


def probing(capability: DockerSandbox[str]) -> Agent[str, str]:
    agent = Agent(TestModel(call_tools=['probe']), deps_type=str, capabilities=[capability])

    @agent.tool
    async def probe(ctx: RunContext[str]) -> str:
        await ctx.workspace.write_text('probe.txt', 'contained')
        return (await ctx.workspace.run(['cat', 'probe.txt'])).stdout

    return agent


async def test_a_run_mounts_before_its_first_operation(docker: FakeDocker, container_dir: Path) -> None:
    capability = DockerSandbox[str](image='image', working_dir=str(container_dir), docker_args=FUSE, mounts=[bucket()])

    result = await probing(capability).run('go', deps='')

    assert result.output == '{"probe":"contained"}'
    (request,) = docker.root_requests()
    assert request['working_dir'] == str(container_dir.resolve())
    assert request['mounts'][0]['keys']['secret_access_key'] == 'the-secret-key'
    root_calls = [call for call in docker.calls if call.startswith('exec --interactive')]
    assert len(root_calls) == 1
    assert root_calls[0].startswith(f'exec --interactive --user 0 --workdir / {result.workspace.ref.id} sh -c ')  # pyright: ignore[reportOptionalMemberAccess]
    assert not any('the-secret-key' in call for call in docker.calls)


async def test_the_root_docker_client_gets_only_the_variables_that_select_a_daemon(
    docker: FakeDocker, container_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('UNRELATED_SECRET', 'not-for-docker')
    monkeypatch.setenv('DOCKER_CONTEXT', 'colima')
    capability = DockerSandbox[str](image='image', working_dir=str(container_dir), docker_args=FUSE, mounts=[bucket()])
    await probing(capability).run('go', deps='')

    env = docker.root_env()
    assert 'DOCKER_CONTEXT=colima' in env and 'UNRELATED_SECRET' not in env


async def test_mounts_can_be_a_function_of_the_run(docker: FakeDocker, container_dir: Path) -> None:
    capability = DockerSandbox[str](
        image='image', working_dir=str(container_dir), docker_args=FUSE, mounts=lambda ctx: [bucket(key=ctx.deps)]
    )

    await probing(capability).run('go', deps='customer-a-key')

    assert [request['mounts'][0]['keys']['access_key_id'] for request in docker.root_requests()] == ['customer-a-key']


async def test_an_attached_container_is_mounted_too(docker: FakeDocker, container_dir: Path) -> None:
    docker.add_container('earlier', container_dir, fuse=True)
    agent = probing(DockerSandbox[str](working_dir=str(container_dir), mounts=[bucket()]))

    await agent.run('go', deps='', workspace=WorkspaceRef(provider='docker', id='earlier'))

    assert len(docker.root_requests()) == 1


async def test_a_container_without_fuse_is_refused_with_the_flags_it_needs(
    docker: FakeDocker, container_dir: Path
) -> None:
    capability = DockerSandbox[str](image='image', working_dir=str(container_dir), mounts=[bucket()])

    with pytest.raises(MountError) as raised:
        await probing(capability).run('go', deps='')

    assert re.sub(r'pydantic-ai-[0-9a-f]+', 'NAME', str(raised.value)) == (
        'Docker workspace NAME has no FUSE device, which mounts need. Create its container with '
        "`docker_args=['--device', '/dev/fuse', '--cap-add', 'SYS_ADMIN', '--security-opt', 'apparmor:unconfined']`."
    )
    assert docker.root_requests() == []


async def test_the_helpers_refusal_fails_the_operation(docker: FakeDocker, container_dir: Path) -> None:
    docker.reply_as_root('{"error": "conflict", "message": "data already shows another bucket in this sandbox."}\n')
    capability = DockerSandbox[str](image='image', working_dir=str(container_dir), docker_args=FUSE, mounts=[bucket()])

    with pytest.raises(MountError, match='data already shows another bucket in this sandbox'):
        await probing(capability).run('go', deps='')


def test_from_spec_forwards_the_spec_fields_and_leaves_out_mounts(docker: FakeDocker) -> None:
    spec_fields: dict[str, object] = {
        'id': 'docker',
        'description': 'a sandbox',
        'image': 'python:3.13-slim',
        'working_dir': '/work',
        'env': {'A': '1'},
        'allow_internet_access': False,
        'docker_args': ['--memory', '2g'],
        'executable': 'podman',
    }

    assert DockerSandbox[None].from_spec(**spec_fields) == DockerSandbox(**spec_fields)  # pyright: ignore[reportArgumentType]
    assert 'mounts' not in inspect.signature(DockerSandbox.from_spec).parameters

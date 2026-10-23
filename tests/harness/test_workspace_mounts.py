"""`mounts=` on every harness workspace capability."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('obstore')

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import LocalWorkspaceBackend, Mount, Workspace
from pydantic_ai_harness import SSHWorkspace
from pydantic_ai_harness._workspace_provider import release_mounts
from pydantic_ai_harness.bubblewrap_sandbox import BubblewrapWorkspace
from pydantic_ai_harness.cloud_storage import S3Filesystem
from pydantic_ai_harness.daytona_sandbox import DaytonaSandbox
from pydantic_ai_harness.docker_sandbox import DockerSandbox, DockerSandboxBackend
from pydantic_ai_harness.e2b_sandbox import E2BSandbox
from pydantic_ai_harness.modal_sandbox import ModalSandbox
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

from .cloud_storage.memory_bucket import MemoryBucket
from .cloud_storage.test_mounting import EXPIRING, Sandbox

pytestmark = pytest.mark.anyio


def _capabilities(
    mounts: Callable[[RunContext[Any]], Sequence[Mount]],
) -> dict[str, AbstractCapability[Any]]:
    return {
        'e2b': E2BSandbox(mounts=mounts),
        'modal': ModalSandbox(mounts=mounts),
        'daytona': DaytonaSandbox(mounts=mounts),
        'sprites': SpritesSandbox(mounts=mounts),
        'docker': DockerSandbox(image='python:3.12', mounts=mounts),
        'ssh': SSHWorkspace('dev@build-box', working_dir='/srv/app', mounts=mounts),
    }


@pytest.mark.parametrize('name', ['e2b', 'modal', 'daytona', 'sprites', 'docker', 'ssh'])
def test_each_workspace_capability_mounts_filesystems(name: str) -> None:
    per_customer = Mount('memory', MemoryBucket())
    capability = _capabilities(lambda ctx: [per_customer] if ctx.deps == 'acme' else [])[name]

    mounted = capability.get_workspace(RunContext(deps='acme', model=TestModel(), usage=RunUsage()), ref=None)
    plain = capability.get_workspace(RunContext(deps='other', model=TestModel(), usage=RunUsage()), ref=None)

    assert isinstance(mounted, Workspace) and mounted.mounts == (per_customer,)
    # With no mounts this run, the backend is used as it is.
    assert not isinstance(plain, Workspace)


async def test_bubblewrap_keeps_the_wrapped_workspace_mounts(tmp_path: Path) -> None:
    bucket = MemoryBucket()
    await bucket.write_bytes('/a.txt', b'from the bucket')
    inner = Workspace(LocalWorkspaceBackend(tmp_path), mounts=[Mount('data', bucket)])

    # File operations on the mount reach the bucket without starting the sandbox.
    assert await BubblewrapWorkspace(inner).read_text('data/a.txt') == 'from the bucket'
    # A bare backend works too, as with every wrapper.
    assert BubblewrapWorkspace(LocalWorkspaceBackend(tmp_path)).mounts == ()


async def test_only_the_run_that_built_a_sandbox_releases_its_mounts() -> None:
    capability = DockerSandbox(image='python:3.12', docker_args=['--device', '/dev/fuse'])
    mount = Mount('data', S3Filesystem('s3://acme/data', credentials=EXPIRING), commands=True)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-1')
    backend = capability.get_workspace(ctx, ref=None)
    assert isinstance(backend, DockerSandboxBackend)
    workspace = Workspace(backend, mounts=[mount])
    # The helper is answered here rather than in a container; `integration_tests/cloud_storage` runs it for real.
    sandbox = Sandbox()
    mounter = backend._mounter  # pyright: ignore[reportPrivateUsage]
    mounter._run_as_root = sandbox.run_as_root  # pyright: ignore[reportPrivateUsage]
    await mounter.ensure([mount], '/w')

    # A subagent handed this workspace ends first, and leaves the mounts held.
    await release_mounts(
        RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-2', workspace=workspace)
    )
    assert len(sandbox.calls) == 1
    await release_mounts(
        RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-1', workspace=workspace)
    )
    assert sandbox.calls[1][1] == {'release': 'run-1'}

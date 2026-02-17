"""Tests for DockerRuntime and runtime init that don't require Docker."""

from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai.toolsets.code_execution.docker import DockerRuntime, DockerSecuritySettings

pytestmark = pytest.mark.anyio


def test_build_docker_run_cmd_default():
    """Default security settings produce a fully-hardened command."""
    rt = DockerRuntime()
    assert rt._build_docker_run_cmd() == snapshot(  # pyright: ignore[reportPrivateUsage]
        [
            'docker',
            'run',
            '-d',
            '--init',
            '--network',
            'none',
            '--cap-drop',
            'ALL',
            '--read-only',
            '--security-opt',
            'no-new-privileges',
            '--user',
            'nobody',
            '--memory',
            '512m',
            '--memory-swap',
            '512m',
            '--pids-limit',
            '256',
            '--cpus',
            '1.0',
            '--tmpfs',
            '/tmp:size=64m,noexec,nosuid',
            '--label',
            'pydantic-ai-runtime=true',
            'python:3.12-slim',
            'sleep',
            'infinity',
        ]
    )


def test_build_docker_run_cmd_relaxed():
    """Disabling all security flags produces a minimal command."""
    relaxed = DockerSecuritySettings(
        network=True,
        read_only=False,
        cap_drop_all=False,
        no_new_privileges=False,
        user='root',
        memory='',
        pids_limit=-1,
        cpus=0,
        tmpfs_noexec=False,
        tmpfs_nosuid=False,
    )
    rt = DockerRuntime(security=relaxed)
    assert rt._build_docker_run_cmd() == snapshot(  # pyright: ignore[reportPrivateUsage]
        [
            'docker',
            'run',
            '-d',
            '--init',
            '--user',
            'root',
            '--tmpfs',
            '/tmp:size=64m',
            '--label',
            'pydantic-ai-runtime=true',
            'python:3.12-slim',
            'sleep',
            'infinity',
        ]
    )


def test_build_docker_run_cmd_empty_user_raises():
    """Empty user string is rejected to prevent accidental root."""
    rt = DockerRuntime(security=DockerSecuritySettings(user=''))
    with pytest.raises(ValueError, match='must not be empty'):
        rt._build_docker_run_cmd()  # pyright: ignore[reportPrivateUsage]


async def test_aenter_cleanup_on_copy_failure(monkeypatch: pytest.MonkeyPatch):
    """If _copy_driver fails during __aenter__, container is cleaned up."""
    rt = DockerRuntime()

    async def fake_create(self: DockerRuntime) -> None:
        self.container_id = 'fake-container-123'

    async def fake_copy_fail(self: DockerRuntime) -> None:
        raise RuntimeError('copy failed')

    removed_containers: list[str] = []

    async def fake_remove(self: DockerRuntime) -> None:
        removed_containers.append(self.container_id or '')

    monkeypatch.setattr(DockerRuntime, '_create_container', fake_create)
    monkeypatch.setattr(DockerRuntime, '_copy_driver', fake_copy_fail)
    monkeypatch.setattr(DockerRuntime, '_remove_container', fake_remove)

    with pytest.raises(RuntimeError, match='copy failed'):
        async with rt:
            pass  # pragma: no cover

    assert removed_containers == ['fake-container-123']
    assert rt.container_id is None
    assert rt._managed is False  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize('method', ['_copy_driver', '_start_driver', '_remove_container'])
async def test_no_container_raises(method: str):
    """Methods that require a container raise ValueError without one."""
    rt = DockerRuntime()
    args: tuple[object, ...] = ({},) if method == '_start_driver' else ()
    with pytest.raises(ValueError, match='has no container'):
        await getattr(rt, method)(*args)

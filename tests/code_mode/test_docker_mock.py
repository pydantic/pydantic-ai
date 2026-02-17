"""Tests for DockerRuntime and runtime init that don't require Docker."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from inline_snapshot import snapshot

from pydantic_ai.runtime.docker import DockerRuntime, DockerSecuritySettings

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


def test_build_docker_run_cmd_no_swap():
    """Setting memory_swap='' prevents --memory-swap flag."""
    rt = DockerRuntime(security=DockerSecuritySettings(memory_swap=''))
    cmd = rt._build_docker_run_cmd()  # pyright: ignore[reportPrivateUsage]
    assert '--memory-swap' not in cmd
    # But --memory should still be present
    assert '--memory' in cmd


async def test_create_container_timeout(monkeypatch: pytest.MonkeyPatch):
    """Container creation timeout raises RuntimeError."""
    import asyncio

    async def hanging_exec(*args: object, **kwargs: object) -> object:
        proc = AsyncMock()
        proc.kill = lambda: None

        async def never_finish() -> None:
            await asyncio.sleep(100)

        proc.communicate = never_finish
        return proc

    monkeypatch.setattr('asyncio.create_subprocess_exec', hanging_exec)
    rt = DockerRuntime(setup_timeout=0.01)
    with pytest.raises(RuntimeError, match='timed out'):
        await rt._create_container()  # pyright: ignore[reportPrivateUsage]


async def test_create_container_nonzero_exit(monkeypatch: pytest.MonkeyPatch):
    """Non-zero return code from docker run raises RuntimeError."""

    async def failing_exec(*args: object, **kwargs: object) -> object:
        proc = AsyncMock()
        proc.returncode = 1
        proc.communicate = AsyncMock(return_value=(b'', b'image not found'))
        return proc

    monkeypatch.setattr('asyncio.create_subprocess_exec', failing_exec)
    rt = DockerRuntime()
    with pytest.raises(RuntimeError, match='Failed to create'):
        await rt._create_container()  # pyright: ignore[reportPrivateUsage]


async def test_copy_driver_rm_timeout(monkeypatch: pytest.MonkeyPatch):
    """Timeout during rm in _copy_driver raises RuntimeError."""
    import asyncio

    call_count = 0

    async def slow_exec(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        proc = AsyncMock()
        proc.kill = lambda: None
        if call_count == 1:  # rm command

            async def never_finish() -> int:
                await asyncio.sleep(100)
                return 0  # pragma: no cover

            proc.wait = never_finish
        return proc

    monkeypatch.setattr('asyncio.create_subprocess_exec', slow_exec)
    rt = DockerRuntime(setup_timeout=0.01)
    rt.container_id = 'test-container'
    with pytest.raises(RuntimeError, match='timed out'):
        await rt._copy_driver()  # pyright: ignore[reportPrivateUsage]


async def test_copy_driver_tee_timeout(monkeypatch: pytest.MonkeyPatch):
    """Timeout during tee in _copy_driver raises RuntimeError."""
    import asyncio

    call_count = 0

    async def slow_exec(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        proc = AsyncMock()
        proc.kill = lambda: None
        if call_count == 1:  # rm command
            proc.wait = AsyncMock(return_value=0)
        else:  # tee command

            async def never_finish(input: bytes | None = None) -> tuple[bytes, bytes]:
                await asyncio.sleep(100)
                return b'', b''  # pragma: no cover

            proc.communicate = never_finish
        return proc

    monkeypatch.setattr('asyncio.create_subprocess_exec', slow_exec)
    rt = DockerRuntime(setup_timeout=0.01)
    rt.container_id = 'test-container'
    with pytest.raises(RuntimeError, match='timed out'):
        await rt._copy_driver()  # pyright: ignore[reportPrivateUsage]


async def test_copy_driver_tee_nonzero_exit(monkeypatch: pytest.MonkeyPatch):
    """Non-zero exit from tee raises RuntimeError."""
    call_count = 0

    async def failing_exec(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        call_count += 1
        proc = AsyncMock()
        if call_count == 1:  # rm command
            proc.wait = AsyncMock(return_value=0)
        else:  # tee command
            proc.returncode = 1
            proc.communicate = AsyncMock(return_value=(b'', b''))
        return proc

    monkeypatch.setattr('asyncio.create_subprocess_exec', failing_exec)
    rt = DockerRuntime()
    rt.container_id = 'test-container'
    with pytest.raises(RuntimeError, match='Failed to copy driver'):
        await rt._copy_driver()  # pyright: ignore[reportPrivateUsage]


async def test_remove_container_failure_warns(monkeypatch: pytest.MonkeyPatch):
    """Failed container removal emits a warning."""
    import warnings

    async def failing_exec(*args: object, **kwargs: object) -> object:
        proc = AsyncMock()
        proc.returncode = 1
        proc.wait = AsyncMock(return_value=1)
        proc.stderr = AsyncMock()
        proc.stderr.read = AsyncMock(return_value=b'no such container')
        return proc

    monkeypatch.setattr('asyncio.create_subprocess_exec', failing_exec)
    rt = DockerRuntime()
    rt.container_id = 'dead-container'
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        await rt._remove_container()  # pyright: ignore[reportPrivateUsage]
    assert len(w) == 1
    assert 'Failed to remove' in str(w[0].message)


def test_get_runtime_docker():
    """get_runtime('docker') returns a DockerRuntime."""
    from pydantic_ai.runtime import get_runtime

    assert isinstance(get_runtime('docker'), DockerRuntime)


async def test_unmanaged_driver_copied_already(monkeypatch: pytest.MonkeyPatch):
    """Unmanaged mode with _driver_copied=True skips copy."""
    copy_calls: list[str] = []

    async def fake_copy(self: DockerRuntime) -> None:
        copy_calls.append('copied')

    monkeypatch.setattr(DockerRuntime, '_copy_driver', fake_copy)
    rt = DockerRuntime(container_id='existing-container')
    rt._driver_copied = True  # pyright: ignore[reportPrivateUsage]
    await rt.__aenter__()
    assert copy_calls == []  # copy was skipped
    await rt.__aexit__(None, None, None)


async def test_unmanaged_copy_failure_no_remove(monkeypatch: pytest.MonkeyPatch):
    """Copy failure on unmanaged runtime does not remove container."""
    remove_calls: list[str] = []

    async def fake_copy_fail(self: DockerRuntime) -> None:
        raise RuntimeError('copy failed')

    async def fake_remove(self: DockerRuntime) -> None:
        remove_calls.append('removed')

    monkeypatch.setattr(DockerRuntime, '_copy_driver', fake_copy_fail)
    monkeypatch.setattr(DockerRuntime, '_remove_container', fake_remove)
    rt = DockerRuntime(container_id='existing-container')
    with pytest.raises(RuntimeError, match='copy failed'):
        await rt.__aenter__()
    # Container should NOT be removed (unmanaged mode)
    assert remove_calls == []
    # Container ID should be preserved
    assert rt.container_id == 'existing-container'

"""Docker-based code runtime with managed container lifecycle and security hardening.

Executes LLM-generated code inside a Docker container via ``docker exec``.
When no ``container_id`` is provided, the runtime manages the full container
lifecycle (create, copy driver, execute, remove) with hardened security defaults.

Requires: Docker CLI available on the host. No Python dependencies
beyond the stdlib (uses ``asyncio.create_subprocess_exec``).

Security model
--------------
The container IS the security boundary. The driver gives LLM code full
``__builtins__`` access including ``__import__``, so code can use arbitrary
stdlib modules. Docker container isolation is the defense layer.

Known limitations:
1. **Shared kernel**: Containers share the host kernel. Kernel exploits
   (CVE-2019-5736, CVE-2024-21626) can escape any container. Keep
   Docker/runc updated. For stronger isolation use ModalRuntime (gVisor).
2. **Full stdlib access**: LLM code can ``import os``, ``import ctypes``, etc.
   This is by design — restricting imports would break legitimate code.
3. **/proc info leakage**: Some ``/proc`` entries may reveal host info.
   Default seccomp/AppArmor profiles limit sensitive ``/proc`` writes.
4. **Disk I/O**: No ``--device-write-bps`` limit (requires knowing the block
   device). tmpfs ``size=64m`` caps ``/tmp``; ``--read-only`` blocks elsewhere.
5. **noexec on tmpfs**: Blocks ``execve()`` for native binaries but Python
   reads scripts as text — ``python /tmp/script.py`` still works.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from typing_extensions import Self

from pydantic_ai.runtime._transport import DriverBasedRuntime, DriverTransport


@dataclass(frozen=True)
class DockerSecuritySettings:
    """Security configuration for managed Docker containers.

    All defaults are restrictive. Override individual settings to relax
    constraints when your use case requires it (e.g. ``network=True``
    for code that needs HTTP access).
    """

    network: bool = False
    """Allow network access. Default ``False`` (``--network none``)."""

    read_only: bool = True
    """Mount root filesystem as read-only. Default ``True``."""

    cap_drop_all: bool = True
    """Drop all Linux capabilities. Default ``True``."""

    no_new_privileges: bool = True
    """Prevent privilege escalation via setuid/setgid. Default ``True``."""

    user: str = 'nobody'
    """Container user. Default ``'nobody'`` (UID 65534)."""

    memory: str = '512m'
    """Memory limit. Default ``'512m'``. Set to empty string to disable."""

    memory_swap: str | None = None
    """Swap limit. Default ``None`` means equal to ``memory`` (disables swap).

    Set to empty string to disable the swap limit entirely.
    """

    pids_limit: int = 256
    """Maximum number of processes. Default ``256``. Set to ``-1`` to disable."""

    cpus: float = 1.0
    """CPU limit. Default ``1.0``. Set to ``0`` to disable."""

    tmpfs_size: str = '64m'
    """Size of the ``/tmp`` tmpfs mount. Default ``'64m'``."""

    tmpfs_noexec: bool = True
    """Mount ``/tmp`` with noexec. Default ``True``."""

    tmpfs_nosuid: bool = True
    """Mount ``/tmp`` with nosuid. Default ``True``."""


class _AsyncSubprocessDriver(DriverTransport):
    """DriverTransport wrapping an ``asyncio.subprocess.Process``."""

    def __init__(self, proc: asyncio.subprocess.Process):
        self._proc = proc

    async def read_line(self) -> bytes:
        assert self._proc.stdout is not None
        return await self._proc.stdout.readline()

    async def write_line(self, data: bytes) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(data)
        await self._proc.stdin.drain()

    async def read_stderr(self) -> bytes:
        assert self._proc.stderr is not None
        return await self._proc.stderr.read()

    async def kill(self) -> None:
        try:
            self._proc.kill()
        except ProcessLookupError:
            pass
        await self._proc.wait()


@dataclass
class DockerRuntime(DriverBasedRuntime):
    """CodeRuntime that executes code inside a Docker container.

    **Managed mode** (default): omit ``container_id`` and the runtime creates
    a hardened container on ``__aenter__`` and removes it on ``__aexit__``::

        runtime = DockerRuntime()
        async with runtime:
            result = await agent.run('...')

    **Unmanaged mode**: pass an existing ``container_id`` and the runtime uses
    it without managing its lifecycle::

        runtime = DockerRuntime(container_id='my-container')
        async with runtime:
            result = await agent.run('...')

    Uses MCP-style reference counting: multiple concurrent ``async with``
    entries share a single container; the container is removed only when
    the last exit completes.
    """

    container_id: str | None = None
    """Docker container ID or name. ``None`` enables managed mode."""

    image: str = 'python:3.12-slim'
    """Docker image for managed containers."""

    python_path: str = 'python'
    """Path to the Python interpreter inside the container."""

    driver_path: str = '/tmp/pydantic_ai_driver.py'
    """Path where the driver script is installed inside the container.

    Must be on a writable filesystem. In managed mode the default ``/tmp``
    is a tmpfs mount, which is writable even when the root filesystem is
    read-only.
    """

    security: DockerSecuritySettings = field(default_factory=DockerSecuritySettings)
    """Security settings for managed containers. Ignored in unmanaged mode."""

    setup_timeout: float | None = 120.0
    """Timeout in seconds for container creation and driver setup. Default ``120.0``.

    Covers image pull, container start, and driver copy. Set to ``None`` to disable.
    """

    _managed: bool = field(default=False, init=False, repr=False)
    _running_count: int = field(default=0, init=False, repr=False)
    _enter_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _driver_copied: bool = field(default=False, init=False, repr=False)

    async def __aenter__(self) -> Self:
        async with self._enter_lock:
            if self._running_count == 0:
                if self.container_id is None:
                    await self._create_container()
                    self._managed = True
                try:
                    if not self._driver_copied:
                        await self._copy_driver()
                        self._driver_copied = True
                except BaseException:
                    if self._managed:
                        await self._remove_container()
                        self.container_id = None
                        self._managed = False
                    raise
            self._running_count += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        async with self._enter_lock:
            self._running_count -= 1
            if self._running_count == 0 and self._managed:
                await self._remove_container()
                self.container_id = None
                self._managed = False
                self._driver_copied = False

    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverTransport:
        if self.container_id is None:
            raise ValueError(
                'DockerRuntime has no container. Use it as an async context manager: '
                '`async with DockerRuntime() as runtime:`'
            )

        proc = await asyncio.create_subprocess_exec(
            'docker',
            'exec',
            '-i',
            self.container_id,
            self.python_path,
            '-u',
            self.driver_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        driver = _AsyncSubprocessDriver(proc)
        init_line = json.dumps(init_msg).encode() + b'\n'
        await driver.write_line(init_line)
        return driver

    def _build_docker_run_cmd(self) -> list[str]:
        """Assemble the ``docker run`` command with all security flags."""
        sec = self.security
        cmd = ['docker', 'run', '-d', '--init']

        # Network isolation
        if not sec.network:
            cmd.extend(['--network', 'none'])

        # Capabilities
        if sec.cap_drop_all:
            cmd.extend(['--cap-drop', 'ALL'])

        # Read-only root filesystem
        if sec.read_only:
            cmd.append('--read-only')

        # Privilege escalation prevention
        if sec.no_new_privileges:
            cmd.extend(['--security-opt', 'no-new-privileges'])

        # User
        if sec.user == '':
            raise ValueError(
                'DockerSecuritySettings.user must not be empty — '
                'the container would run as root. Use "nobody" (default) or an explicit username/UID.'
            )
        if sec.user:
            cmd.extend(['--user', sec.user])

        # Memory limits
        if sec.memory:
            cmd.extend(['--memory', sec.memory])
            swap = sec.memory_swap if sec.memory_swap is not None else sec.memory
            if swap:
                cmd.extend(['--memory-swap', swap])

        # PID limit
        if sec.pids_limit >= 0:
            cmd.extend(['--pids-limit', str(sec.pids_limit)])

        # CPU limit
        if sec.cpus > 0:
            cmd.extend(['--cpus', str(sec.cpus)])

        # tmpfs at /tmp (writable scratch space on read-only rootfs)
        tmpfs_opts = [f'size={sec.tmpfs_size}']
        if sec.tmpfs_noexec:
            tmpfs_opts.append('noexec')
        if sec.tmpfs_nosuid:
            tmpfs_opts.append('nosuid')
        cmd.extend(['--tmpfs', f'/tmp:{",".join(tmpfs_opts)}'])

        # Label for orphan identification
        cmd.extend(['--label', 'pydantic-ai-runtime=true'])

        cmd.extend([self.image, 'sleep', 'infinity'])
        return cmd

    async def _create_container(self) -> None:
        """Create a managed container with security-hardened settings."""
        cmd = self._build_docker_run_cmd()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.setup_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(
                f'Docker container creation timed out after {self.setup_timeout}s '
                f'(image pull for {self.image!r} may be slow — increase setup_timeout or pull the image first)'
            )
        if proc.returncode != 0:
            raise RuntimeError(f'Failed to create Docker container: {stderr.decode().strip()}')
        self.container_id = stdout.decode().strip()[:12]

    async def _copy_driver(self) -> None:
        """Copy the driver script into the container.

        Uses ``docker exec`` with ``tee`` instead of ``docker cp`` because
        ``docker cp`` writes to the container's root filesystem overlay,
        which fails when ``--read-only`` is set — even for paths under
        a writable tmpfs mount like ``/tmp``.

        Removes any existing file first because ``docker cp`` creates files
        owned by the host UID, which may not be writable by the container
        user (a known Docker Desktop for Mac behavior).
        """
        if self.container_id is None:
            raise ValueError(
                'DockerRuntime has no container. Use it as an async context manager: '
                '`async with DockerRuntime() as runtime:`'
            )
        # Remove any existing driver file (e.g. placed by a prior docker cp)
        # so that tee can create a fresh file owned by the container user.
        rm_proc = await asyncio.create_subprocess_exec(
            'docker',
            'exec',
            self.container_id,
            'rm',
            '-f',
            self.driver_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(rm_proc.wait(), timeout=self.setup_timeout)
        except asyncio.TimeoutError:
            rm_proc.kill()
            raise RuntimeError(
                f'Driver cleanup in container {self.container_id} timed out after {self.setup_timeout}s'
            )

        driver_src = Path(__file__).parent / '_driver.py'
        driver_content = driver_src.read_bytes()
        proc = await asyncio.create_subprocess_exec(
            'docker',
            'exec',
            '-i',
            self.container_id,
            'tee',
            self.driver_path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.communicate(input=driver_content), timeout=self.setup_timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(
                f'Driver copy to container {self.container_id} timed out after {self.setup_timeout}s'
            )
        if proc.returncode != 0:
            raise RuntimeError(f'Failed to copy driver to container {self.container_id}')

    async def _remove_container(self) -> None:
        """Force-remove the managed container."""
        if self.container_id is None:
            raise ValueError(
                'DockerRuntime has no container. Use it as an async context manager: '
                '`async with DockerRuntime() as runtime:`'
            )
        proc = await asyncio.create_subprocess_exec(
            'docker',
            'rm',
            '-f',
            self.container_id,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        if proc.returncode != 0:
            stderr = await proc.stderr.read() if proc.stderr else b''
            import warnings

            warnings.warn(
                f'Failed to remove Docker container {self.container_id}: {stderr.decode().strip()}',
                stacklevel=2,
            )

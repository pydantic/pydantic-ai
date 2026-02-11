"""Docker-based code runtime using stdio pipes.

Executes LLM-generated code inside a running Docker container via
``docker exec``. The container must already exist — this runtime does
not manage container lifecycle.

Requires: Docker CLI available on the host. No Python dependencies
beyond the stdlib (uses ``asyncio.create_subprocess_exec``).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai.runtime._transport import DriverBasedRuntime, DriverTransport


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

    The container must already be running. Use ``copy_driver_to_container()``
    to install the driver script before the first ``run()`` call.

    Args:
        container_id: Docker container ID or name.
        python_path: Path to the Python interpreter inside the container.
        driver_path: Path where the driver script is installed inside the container.
    """

    container_id: str = ''
    python_path: str = 'python'
    driver_path: str = '/tmp/pydantic_ai_driver.py'

    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverTransport:
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

    async def copy_driver_to_container(self) -> None:
        """Copy the driver script into the container.

        Call once before the first ``run()``. The driver is a self-contained
        Python script with no dependencies beyond the stdlib.
        """
        driver_src = Path(__file__).parent / '_driver.py'
        proc = await asyncio.create_subprocess_exec(
            'docker',
            'cp',
            str(driver_src),
            f'{self.container_id}:{self.driver_path}',
        )
        await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f'Failed to copy driver to container {self.container_id}')

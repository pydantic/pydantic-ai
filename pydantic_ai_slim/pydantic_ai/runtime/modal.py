"""Modal-based code runtime using stdio pipes.

Executes LLM-generated code inside a Modal sandbox (gVisor-isolated cloud
container). Creates an ephemeral sandbox per execution — no container
management required.

Requires: ``pip install "pydantic-ai-slim[modal]"``
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai.runtime._stdio import DriverProcess, StdioSandboxRuntime


class _ModalDriverProcess(DriverProcess):
    """DriverProcess wrapping a Modal ``ContainerProcess``."""

    def __init__(self, process: Any, sandbox: Any):
        self._process = process
        self._sandbox = sandbox
        self._stdout_iter = process.stdout.__aiter__()

    async def read_line(self) -> bytes:
        try:
            line: str = await self._stdout_iter.__anext__()
            return line.encode()
        except StopAsyncIteration:
            return b''

    async def write_line(self, data: bytes) -> None:
        self._process.stdin.write(data)
        await self._process.stdin.drain.aio()

    async def read_stderr(self) -> bytes:
        text: str = await asyncio.to_thread(self._process.stderr.read)
        return text.encode()

    async def kill(self) -> None:
        await self._sandbox.terminate.aio()


@dataclass
class ModalRuntime(StdioSandboxRuntime):
    """CodeRuntime that executes code inside a Modal sandbox.

    Creates an ephemeral gVisor-isolated sandbox per execution.
    No Docker or container management required — Modal handles
    infrastructure automatically.

    Args:
        app_name: Modal app name to use (created if missing).
        image: Modal Image to use for the sandbox. Defaults to ``modal.Image.debian_slim()``.
        timeout: Maximum sandbox lifetime in seconds.
    """

    app_name: str = 'pydantic-ai-code-runtime'
    image: Any = None
    timeout: int = 300

    async def _start_driver(self, init_msg: dict[str, Any]) -> DriverProcess:
        try:
            import modal
        except ImportError as _e:
            raise ImportError('modal is not installed, run `pip install "pydantic-ai-slim[modal]"`') from _e

        app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        image = self.image if self.image is not None else modal.Image.debian_slim()

        sandbox = await modal.Sandbox.create.aio(app=app, image=image, timeout=self.timeout)

        # Upload the driver script into the sandbox
        driver_src = Path(__file__).parent / '_driver.py'
        driver_content = driver_src.read_text()

        def _upload_driver() -> None:
            with sandbox.open('/tmp/pydantic_ai_driver.py', 'w') as f:
                f.write(driver_content)

        await asyncio.to_thread(_upload_driver)

        # Start the driver process with line buffering
        process = await sandbox.exec.aio('python', '-u', '/tmp/pydantic_ai_driver.py', bufsize=1)

        driver = _ModalDriverProcess(process, sandbox)
        init_line = json.dumps(init_msg).encode() + b'\n'
        await driver.write_line(init_line)
        return driver

"""Exercise the shipped supervisor through the public backend and SDK boundary."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render.experimental.sandbox import SandboxExecEvent, SandboxExecExit, SandboxExecOutput

    from pydantic_ai.workspaces import WorkspaceTimeoutError

    from .workspace_helpers import WorkspaceHarness

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='render not installed'),
    pytest.mark.skipif(sys.platform == 'win32', reason='supervisor requires POSIX process groups'),
    pytest.mark.subprocess(reason='Verify real supervisor signal ownership and process-group cleanup'),
]


@dataclass
class LocalExec:
    """Replace only SDK transport; execute its received supervisor source intact.

    Disconnecting the stream leaves execution running, as Render does. The fixture
    owns every local supervisor and claim directory, including failed launches.
    """

    processes: list[asyncio.subprocess.Process] = field(default_factory=list[asyncio.subprocess.Process])
    claims: set[Path] = field(default_factory=set[Path])
    stop_launches: dict[Path, str] = field(default_factory=dict[Path, str])
    output: asyncio.Queue[bytes] = field(default_factory=asyncio.Queue[bytes])
    delay_next: bool = False
    delayed: str | None = None

    async def exec(self, sandbox_id: str, launch: str) -> AsyncIterator[SandboxExecEvent]:
        argv = shlex.split(launch)
        config = json.loads(argv[4])
        claim = Path(config['claim'])
        self.claims.add(claim)
        if argv[3] == 'run':
            self.stop_launches[claim] = shlex.join([*argv[:3], 'stop', argv[4]])
        if argv[3] == 'run' and self.delay_next:
            self.delay_next = False
            self.delayed = launch
            raise ConnectionError('start response lost before the server launches the command')
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            *argv[1:],
            # No host credentials or coverage instrumentation enter the child.
            env={'PATH': os.defpath},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.processes.append(process)
        assert process.stdout is not None and process.stderr is not None
        while line := await process.stdout.readline():
            frame = json.loads(line)
            if argv[3] == 'run' and frame['kind'] == 'output':
                self.output.put_nowait(base64.b64decode(frame['data']))
            yield SandboxExecOutput('stdout', line.decode())
        stderr = await process.stderr.read()
        assert not stderr, stderr.decode()
        yield SandboxExecExit(await process.wait())

    async def wait(self) -> None:
        # All test commands either finish or receive the real stop exec first.
        async with asyncio.timeout(10):
            for process in self.processes:
                await process.wait()

    async def close(self) -> None:
        try:
            async with asyncio.timeout(10):
                for launch in self.stop_launches.values():
                    async for _ in self.exec('sb-1', launch):
                        pass
            await self.wait()
        finally:
            for process in self.processes:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
            for claim in self.claims:
                shutil.rmtree(claim, ignore_errors=True)


@pytest.fixture
async def local_exec(workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[LocalExec]:
    transport = LocalExec()
    monkeypatch.setattr(workspace_harness.client.experimental.sandboxes, 'exec', transport.exec)
    try:
        yield transport
    finally:
        await transport.close()


@pytest.mark.parametrize('interrupt', ['timeout', 'cancel'])
async def test_supervisor_stops_the_real_foreground_process(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, interrupt: str
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    # Discard the readiness probe's cwd output before observing the command PID.
    while not local_exec.output.empty():
        local_exec.output.get_nowait()
    command = [
        sys.executable,
        '-c',
        'import os,subprocess,time; child=subprocess.Popen(["sleep", "60"]); '
        'print(os.getpid(), child.pid, flush=True); time.sleep(60)',
    ]
    task = asyncio.create_task(backend.run(command, timeout=1 if interrupt == 'timeout' else None))
    pid: int | None = None
    child_pid: int | None = None
    try:
        async with asyncio.timeout(10):
            pid, child_pid = map(int, (await local_exec.output.get()).split())
        if interrupt == 'cancel':
            task.cancel()
        expected = WorkspaceTimeoutError if interrupt == 'timeout' else asyncio.CancelledError
        with pytest.raises(expected):
            await task
        await local_exec.wait()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        # The child is reparented after group termination. A zombie is already
        # stopped and waits for the OS reaper, which this test does not control.
        status = subprocess.run(
            ['ps', '-o', 'stat=', '-p', str(child_pid)], capture_output=True, text=True, check=False
        ).stdout.strip()
        assert not status or status.startswith('Z')
        assert (await backend.run(['printf', 'still alive'])).stdout == 'still alive'
        assert workspace_harness.terminated == []
    finally:
        # This fallback is independent of the supervisor's stop implementation,
        # so a regression cannot leave the test command or its child running.
        if pid is not None:
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        if child_pid is not None:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await backend.destroy()


async def test_supervisor_returns_after_direct_exit_with_inherited_pipe(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, tmp_path: Path
) -> None:
    backend = workspace_harness.backend()
    background_pid = tmp_path / 'background.pid'
    try:
        # The background process deliberately retains stdout and stderr. The
        # supervisor must observe the direct shell's exit without waiting for EOF.
        result = await asyncio.wait_for(
            backend.run(['sh', '-c', 'sleep 60 & echo $! > "$1"; printf done', 'sh', str(background_pid)]),
            timeout=10,
        )
        assert (result.exit_code, result.stdout, result.stderr) == (0, 'done', '')
        os.kill(int(background_pid.read_text()), 0)
    finally:
        if background_pid.exists():
            try:
                os.kill(int(background_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        await backend.destroy()


async def test_stop_fence_prevents_a_delayed_launch(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, tmp_path: Path
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    sentinel = tmp_path / 'must-not-run'
    local_exec.delay_next = True
    try:
        with pytest.raises(ConnectionError, match='start response lost'):
            await backend.run(['touch', str(sentinel)])
        assert local_exec.delayed is not None
        events = [event async for event in local_exec.exec('sb-1', local_exec.delayed)]
        frames = [json.loads(event.data) for event in events if isinstance(event, SandboxExecOutput)]
        assert [frame['kind'] for frame in frames] == ['fenced']
        assert isinstance(events[-1], SandboxExecExit) and events[-1].exit_code == 0
        assert not sentinel.exists()
        assert (await backend.run(['printf', 'still alive'])).stdout == 'still alive'
    finally:
        await backend.destroy()

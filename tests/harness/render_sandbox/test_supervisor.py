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
from importlib.resources import files
from pathlib import Path

import anyio
import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render.experimental.sandbox import SandboxExecEvent, SandboxExecExit, SandboxExecOutput

    from pydantic_ai.workspaces import (
        WorkspaceError,
        WorkspaceOutputLimitError,
        WorkspaceTimeoutError,
        WorkspaceUnavailableError,
    )

    from .workspace_helpers import WorkspaceHarness

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='render not installed'),
    pytest.mark.skipif(sys.platform == 'win32', reason='supervisor requires POSIX process groups'),
    pytest.mark.subprocess(reason='Verify real supervisor signal ownership and process-group cleanup'),
]


_SUPERVISOR = files('pydantic_ai_harness.render_sandbox').joinpath('_command_runner.py')


def _local_supervisor(argv: list[str]) -> list[str]:
    # Run the received source from its package file, so child coverage has a source path.
    assert argv[:4] == ['python3', '-I', '-c', _SUPERVISOR.read_text()]
    return [sys.executable, str(_SUPERVISOR), *argv[4:]]


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
    drain_before_stop: bool = False
    inline_cwd: Path | None = None
    sandbox_env: dict[str, str] = field(default_factory=dict[str, str])

    async def exec(self, sandbox_id: str, launch: str) -> AsyncIterator[SandboxExecEvent]:
        argv = shlex.split(launch)
        operation, config = argv[4], json.loads(argv[5])
        claim = Path(config['claim'])
        self.claims.add(claim)
        if operation == 'run':
            self.stop_launches[claim] = shlex.join([*argv[:4], 'stop', argv[5]])
        if operation == 'run' and self.delay_next:
            self.delay_next = False
            self.delayed = launch
            raise ConnectionError('start response lost before the server launches the command')
        if operation == 'stop' and self.drain_before_stop:
            # A delayed provider stop lets the remote supervisor's own deadline fire.
            await asyncio.gather(*(process.wait() for process in self.processes))
        process = await asyncio.create_subprocess_exec(
            # Inline mode runs the launch exactly as Render receives it, in the given directory.
            *([sys.executable, *argv[1:]] if self.inline_cwd else _local_supervisor(argv)),
            cwd=self.inline_cwd,
            # Keep host credentials out; subprocess coverage adds only its bookkeeping.
            env={
                'PATH': os.defpath,
                **{key: value for key, value in os.environ.items() if key == 'COVERAGE_PROCESS_CONFIG'},
                **self.sandbox_env,
            },
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.processes.append(process)
        assert process.stdout is not None and process.stderr is not None
        while line := await process.stdout.readline():
            frame = json.loads(line)
            if operation == 'run' and frame['kind'] == 'output':
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
                if process.returncode is None:  # pragma: no cover - independent cleanup after a regression
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
    command = ['sh', '-c', 'sleep 60 & echo $$ $!; wait']
    task = asyncio.create_task(backend.run(command, timeout=0.2 if interrupt == 'timeout' else None))
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
        if pid is not None:  # pragma: no branch - missing PID only follows an earlier test failure
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        if child_pid is not None:  # pragma: no branch - missing PID only follows an earlier test failure
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)
        if not task.done():  # pragma: no cover - independent cleanup after a regression
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_stop_fence_prevents_a_delayed_launch(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, tmp_path: Path
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    sentinel = tmp_path / 'must-not-run'
    local_exec.delay_next = True
    with pytest.raises(ConnectionError, match='start response lost'):
        await backend.run(['touch', str(sentinel)])
    assert local_exec.delayed is not None
    events = [event async for event in local_exec.exec('sb-1', local_exec.delayed)]
    frames = [json.loads(event.data) for event in events if isinstance(event, SandboxExecOutput)]
    assert [frame['kind'] for frame in frames] == ['fenced']
    assert isinstance(events[-1], SandboxExecExit) and events[-1].exit_code == 0
    assert not sentinel.exists()
    assert (await backend.run(['printf', 'still alive'])).stdout == 'still alive'


# The conformance suite covers a missing program's exit code 127 against the shipped supervisor.
@pytest.mark.parametrize('failure', ['permission', 'format'])
async def test_supervisor_reports_executable_startup_failure(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, tmp_path: Path, failure: str
) -> None:
    executable = tmp_path / 'program'
    executable.write_text('not an executable format')
    executable.chmod(0o700 if failure == 'format' else 0o600)
    backend = workspace_harness.backend()
    if failure == 'format':
        with pytest.raises(WorkspaceError, match='supervisor failed'):
            await backend.run([str(executable)])
    else:
        result = await backend.run([str(executable)])
        assert (result.exit_code, result.stderr) == (126, f'{executable}: Permission denied\n')
    assert (await backend.run(['printf', 'usable'])).stdout == 'usable'


async def test_supervisor_output_bound_preserves_captured_prefix(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._MAX_OUTPUT_BYTES', 512)
    backend = workspace_harness.backend()
    with pytest.raises(WorkspaceOutputLimitError) as caught:
        await backend.run(['sh', '-c', 'printf "%05000d" 0'])
    assert caught.value.limit == 512
    assert caught.value.stdout == '0' * 512
    await local_exec.wait()
    assert (await backend.run(['printf', 'usable'])).stdout == 'usable'


async def test_remote_deadline_stops_command_when_provider_stop_is_delayed(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    while not local_exec.output.empty():
        local_exec.output.get_nowait()
    local_exec.drain_before_stop = True
    pending = asyncio.create_task(backend.run(['sh', '-c', 'echo $$; exec sleep 60'], timeout=0.2))
    pid: int | None = None
    try:
        async with asyncio.timeout(10):
            pid = int(await local_exec.output.get())
        with pytest.raises(WorkspaceTimeoutError):
            await pending
        await local_exec.wait()
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        # This fallback still owns the process if its remote deadline regresses.
        local_exec.drain_before_stop = False
        if pid is not None:  # pragma: no branch - missing PID only follows an earlier test failure
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        if not pending.done():  # pragma: no cover - independent cleanup after a regression
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_supervisor_drains_closed_stdout_and_separate_stderr(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec
) -> None:
    backend = workspace_harness.backend()
    # Closing one pipe while the child is alive must not discard the other pipe.
    result = await backend.run(['sh', '-c', 'exec 1>&-; sleep 0.1; printf stderr >&2'])
    assert (result.exit_code, result.stdout, result.stderr) == (0, '', 'stderr')


async def test_project_module_does_not_shadow_supervisor_imports(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, tmp_path: Path
) -> None:
    # A project's `json.py` in the exec's directory must not replace the stdlib module the supervisor imports.
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'json.py').write_text("raise SystemExit('project json.py was imported')\n")
    local_exec.inline_cwd = project
    backend = workspace_harness.backend()
    assert (await backend.run(['printf', 'usable'])).stdout == 'usable'


async def test_background_writer_on_inherited_pipe_does_not_hold_up_exit(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # After the direct child exits, the supervisor drains for a bounded grace, not until a writer stops.
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._DRAIN_GRACE', 0.05)
    background_pid = tmp_path / 'writer.pid'
    backend = workspace_harness.backend()
    try:
        result = await asyncio.wait_for(
            backend.run(
                [
                    'sh',
                    '-c',
                    '(while :; do echo x; sleep 0.01; done) & echo $! > "$1"; exit 3',
                    'sh',
                    str(background_pid),
                ]
            ),
            timeout=10,
        )
        assert result.exit_code == 3
    finally:
        if background_pid.exists():  # pragma: no branch - absent only if command startup regresses
            with suppress(ProcessLookupError):
                os.kill(int(background_pid.read_text()), signal.SIGKILL)


async def test_output_limit_after_direct_exit_stops_the_background_writer(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._MAX_OUTPUT_BYTES', 512)
    writer_pid = tmp_path / 'writer.pid'
    # The writer stays in the command's process group and starts flooding only once the direct shell is gone.
    script = (
        'sh -c \'while kill -0 "$1" 2>/dev/null; do :; done; while :; do printf "%01000d" 0; done\' sh $$ & '
        'echo $! > "$1"; exit 0'
    )
    backend = workspace_harness.backend()
    try:
        with pytest.raises(WorkspaceOutputLimitError):
            await asyncio.wait_for(backend.run(['sh', '-c', script, 'sh', str(writer_pid)]), timeout=10)
        pid = int(writer_pid.read_text())
        with anyio.fail_after(10):
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                await anyio.sleep(0.05)  # pragma: lax no cover - the writer is often gone by the first check
    finally:
        if writer_pid.exists():  # pragma: no branch - absent only if command startup regresses
            with suppress(ProcessLookupError):
                os.kill(int(writer_pid.read_text()), signal.SIGKILL)


async def test_background_output_within_the_drain_grace_is_captured(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec
) -> None:
    # The background child writes only once the direct shell is gone; the drain grace still captures it.
    backend = workspace_harness.backend()
    script = 'echo a; (while kill -0 $$ 2>/dev/null; do sleep 0.01; done; echo b) &'
    result = await asyncio.wait_for(backend.run(['sh', '-c', script]), timeout=10)
    assert result.stdout == 'a\nb\n'


async def test_signal_death_reports_128_plus_signal(workspace_harness: WorkspaceHarness, local_exec: LocalExec) -> None:
    backend = workspace_harness.backend()
    result = await backend.run(['sh', '-c', 'kill -KILL $$'])
    assert result.exit_code == 128 + signal.SIGKILL


async def test_missing_working_dir_raises_unavailable(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, tmp_path: Path
) -> None:
    # A deleted working directory is not a missing program, so it must not look like exit 127.
    project = tmp_path / 'project'
    project.mkdir()
    backend = workspace_harness.backend(working_dir=str(project))
    await backend.working_dir()
    project.rmdir()
    with pytest.raises(WorkspaceUnavailableError, match='working directory'):
        await backend.run(['true'])


async def test_commands_keep_the_sandbox_environment(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec
) -> None:
    # Per-command `env` is layered over the sandbox's own environment, not a replacement for it.
    local_exec.sandbox_env = {'SANDBOX_MARKER': 'inherited'}
    backend = workspace_harness.backend()
    result = await backend.run(['sh', '-c', 'printf %s-%s "$SANDBOX_MARKER" "$ADDED"'], env={'ADDED': 'added'})
    assert result.stdout == 'inherited-added'


async def test_cancellation_during_drain_stops_the_background_writer(
    workspace_harness: WorkspaceHarness, local_exec: LocalExec, tmp_path: Path
) -> None:
    # The direct shell exits while a same-group child keeps the pipe open; cancelling during the drain kills it.
    writer_pid = tmp_path / 'writer.pid'
    backend = workspace_harness.backend()
    await backend.working_dir()
    while not local_exec.output.empty():
        local_exec.output.get_nowait()
    # The child announces itself only once the direct shell is gone, so the cancel lands during the drain.
    script = '(while kill -0 $$ 2>/dev/null; do sleep 0.01; done; echo started; while :; do sleep 0.01; done) & '
    script += 'echo $! > "$1"; exit 0'
    pending = asyncio.create_task(backend.run(['sh', '-c', script, 'sh', str(writer_pid)]))
    try:
        async with asyncio.timeout(10):
            assert await local_exec.output.get() == b'started\n'
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        pid = int(writer_pid.read_text())
        with anyio.fail_after(10):
            while True:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                await anyio.sleep(0.05)  # pragma: lax no cover - the writer is often gone by the first check
    finally:
        if writer_pid.exists():  # pragma: no branch - absent only if command startup regresses
            with suppress(ProcessLookupError):
                os.kill(int(writer_pid.read_text()), signal.SIGKILL)
        await asyncio.gather(pending, return_exceptions=True)

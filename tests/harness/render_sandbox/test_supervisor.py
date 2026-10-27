"""Exercise the shipped supervisor through the public backend and SDK boundary."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import anyio
import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render.experimental.sandbox import SandboxExecExit, SandboxExecOutput

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


@pytest.fixture(autouse=True)
def supervised(workspace_harness: WorkspaceHarness) -> None:
    # User commands run under the shipped supervisor in a child interpreter, whose coverage is measured here.
    workspace_harness.supervised = True
    workspace_harness.measure_supervisor = True


async def test_supervisor_stops_the_real_foreground_process(workspace_harness: WorkspaceHarness) -> None:
    # Cancellation, not a short timeout: a deadline could fire before a slow runner starts the command.
    backend = workspace_harness.backend()
    await backend.working_dir()
    command = ['sh', '-c', 'sleep 60 & echo $$ $!; wait']
    task = asyncio.create_task(backend.run(command))
    pid: int | None = None
    child_pid: int | None = None
    try:
        async with asyncio.timeout(10):
            pid, child_pid = map(int, (await workspace_harness.output.get()).split())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await workspace_harness.wait_supervisors()
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


async def test_stop_fence_prevents_a_delayed_launch(workspace_harness: WorkspaceHarness, tmp_path: Path) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    sentinel = tmp_path / 'must-not-run'
    workspace_harness.delay_next = True
    with pytest.raises(ConnectionError, match='start response lost'):
        await backend.run(['touch', str(sentinel)])
    assert workspace_harness.delayed is not None
    replay = workspace_harness.client.experimental.sandboxes.exec('sb-1', workspace_harness.delayed)
    events = [event async for event in replay]
    frames = [json.loads(event.data) for event in events if isinstance(event, SandboxExecOutput)]
    assert [frame['kind'] for frame in frames] == ['fenced']
    assert isinstance(events[-1], SandboxExecExit) and events[-1].exit_code == 0
    assert not sentinel.exists()
    assert (await backend.run(['printf', 'still alive'])).stdout == 'still alive'


# The conformance suite covers a missing program's exit code 127 against the shipped supervisor.
@pytest.mark.parametrize('failure', ['permission', 'format'])
async def test_supervisor_reports_executable_startup_failure(
    workspace_harness: WorkspaceHarness, tmp_path: Path, failure: str
) -> None:
    executable = tmp_path / 'program'
    executable.write_text('not an executable format')
    executable.chmod(0o700 if failure == 'format' else 0o600)
    backend = workspace_harness.backend()
    if failure == 'format':
        with pytest.raises(WorkspaceError, match="supervisor reported 'error'"):
            await backend.run([str(executable)])
    else:
        result = await backend.run([str(executable)])
        assert (result.exit_code, result.stderr) == (126, f'{executable}: Permission denied\n')
    assert (await backend.run(['printf', 'usable'])).stdout == 'usable'


async def test_supervisor_output_bound_preserves_captured_prefix(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    with pytest.raises(WorkspaceOutputLimitError) as caught:
        await backend.run(['sh', '-c', 'head -c 10485761 /dev/zero'])
    assert (caught.value.limit, caught.value.stdout) == (10 * 1024 * 1024, '\0' * (64 * 1024))
    await workspace_harness.wait_supervisors()
    assert (await backend.run(['printf', 'usable'])).stdout == 'usable'


async def test_remote_deadline_stops_command_when_provider_stop_is_delayed(
    workspace_harness: WorkspaceHarness, tmp_path: Path
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    workspace_harness.drain_before_stop = True
    # A PID file, not stdout: the host deadline can abandon the stream before the command prints.
    pid_file = tmp_path / 'pid'
    try:
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sh', '-c', f'echo $$ > {pid_file}; exec sleep 60'], timeout=0.2)
        # The supervisor's own deadline, counted from spawn, stops the group before the delayed stop arrives.
        await workspace_harness.wait_supervisors()
        pid = int(pid_file.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        workspace_harness.drain_before_stop = False
        if pid_file.exists():  # pragma: no branch - the file only goes missing after an earlier failure
            with suppress(ProcessLookupError, ValueError):
                os.killpg(int(pid_file.read_text()), signal.SIGKILL)


async def test_supervisor_drains_closed_stdout_and_separate_stderr(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    # Closing one pipe while the child is alive must not discard the other pipe.
    result = await backend.run(['sh', '-c', 'exec 1>&-; sleep 0.1; printf stderr >&2'])
    assert (result.exit_code, result.stdout, result.stderr) == (0, '', 'stderr')


async def test_project_module_does_not_shadow_supervisor_imports(
    workspace_harness: WorkspaceHarness, tmp_path: Path
) -> None:
    # A project's `json.py` in the exec's directory must not replace the stdlib module the supervisor imports.
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'json.py').write_text("raise SystemExit('project json.py was imported')\n")
    workspace_harness.inline_cwd = project
    backend = workspace_harness.backend()
    assert (await backend.run(['printf', 'usable'])).stdout == 'usable'


async def test_background_writer_on_inherited_pipe_does_not_hold_up_exit(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A long grace costs nothing here: the limit ends the drain, and a slow runner needs time to stream 10 MiB.
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._DRAIN_GRACE', 60.0)
    writer_pid = tmp_path / 'writer.pid'
    # The writer stays in the command's process group and starts flooding only once the direct shell is gone.
    script = 'sh -c \'while kill -0 "$1" 2>/dev/null; do :; done; exec cat /dev/zero\' sh $$ & echo $! > "$1"; exit 0'
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
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The background child writes only once the direct shell is gone; the drain grace still captures it.
    # A long grace costs nothing here: the child exits after writing, and EOF ends the drain.
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._DRAIN_GRACE', 60.0)
    backend = workspace_harness.backend()
    script = 'echo a; (while kill -0 $$ 2>/dev/null; do sleep 0.01; done; echo b) &'
    result = await asyncio.wait_for(backend.run(['sh', '-c', script]), timeout=10)
    assert result.stdout == 'a\nb\n'


async def test_signal_death_reports_128_plus_signal(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    result = await backend.run(['sh', '-c', 'kill -KILL $$'])
    assert result.exit_code == 128 + signal.SIGKILL


async def test_missing_working_dir_raises_unavailable(workspace_harness: WorkspaceHarness, tmp_path: Path) -> None:
    # A deleted working directory is not a missing program, so it must not look like exit 127.
    project = tmp_path / 'project'
    project.mkdir()
    backend = workspace_harness.backend(working_dir=str(project))
    await backend.working_dir()
    project.rmdir()
    with pytest.raises(WorkspaceUnavailableError, match='working directory'):
        await backend.run(['true'])


async def test_commands_keep_the_sandbox_environment(workspace_harness: WorkspaceHarness) -> None:
    # Per-command `env` is layered over the sandbox's own environment, not a replacement for it.
    workspace_harness.sandbox_env = {'SANDBOX_MARKER': 'inherited'}
    backend = workspace_harness.backend()
    result = await backend.run(['sh', '-c', 'printf %s-%s "$SANDBOX_MARKER" "$ADDED"'], env={'ADDED': 'added'})
    assert result.stdout == 'inherited-added'


async def test_cancellation_during_drain_stops_the_background_writer(
    workspace_harness: WorkspaceHarness, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The direct shell exits while a same-group child keeps the pipe open; cancelling during the drain kills it.
    # A drain longer than any startup delay, so the cancel, not the grace, ends it.
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._DRAIN_GRACE', 60.0)
    writer_pid = tmp_path / 'writer.pid'
    backend = workspace_harness.backend()
    await backend.working_dir()
    # The child announces itself only once the direct shell is gone, so the cancel lands during the drain.
    script = '(while kill -0 $$ 2>/dev/null; do sleep 0.01; done; echo started; while :; do sleep 0.01; done) & '
    script += 'echo $! > "$1"; exit 0'
    pending = asyncio.create_task(backend.run(['sh', '-c', script, 'sh', str(writer_pid)]))
    try:
        async with asyncio.timeout(10):
            assert await workspace_harness.output.get() == b'started\n'
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

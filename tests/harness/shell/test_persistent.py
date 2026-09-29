"""The opt-in `shell` tool: commands that outlive the run, with a bounded foreground wait."""

import json
import os
import shlex
import signal
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anyio
import pytest
from anyio.abc import SocketAttribute, SocketStream
from anyio.to_thread import run_sync

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, on_event
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import LocalWorkspaceBackend
from pydantic_ai_harness.shell import (
    MAX_FOREGROUND_WAIT,
    RUN_SCOPED_TOOL_NAMES,
    SHELL_TOOL_NAMES,
    CommandFinishedEvent,
    CommandOutputEvent,
    CommandStartedEvent,
    Shell,
)
from pydantic_ai_harness.shell._jobs import Job

from .._tool_calls import call_tool, call_tools

pytestmark = [pytest.mark.skipif(os.name == 'nt', reason='POSIX shell commands and process groups')]


async def receive_exactly(stream: SocketStream, size: int) -> bytes:
    data = b''
    while len(data) < size:
        data += await stream.receive(size - len(data))
    return data


async def wait_for_exit(pid: int) -> None:
    """Wait for `pid` to be reaped; a process init reaps may linger as a zombie for a moment, or not at all."""
    with anyio.fail_after(10):
        while True:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            await anyio.sleep(0.01)  # pragma: lax no cover


async def shell(
    cwd: Path,
    arguments: dict[str, object],
    *,
    capabilities: Sequence[AbstractCapability[None]] = (),
    **settings: object,
) -> str:
    capability = Shell[None](denied_commands=[], allow_interactive=True, tools=['shell'], **settings)  # pyright: ignore[reportArgumentType]
    return await call_tool([capability, *capabilities], 'shell', arguments, workspace=LocalWorkspaceBackend(cwd))


class Recorder(AbstractCapability[None]):
    def __init__(self) -> None:
        self.events: list[CommandStartedEvent | CommandOutputEvent | CommandFinishedEvent] = []

    @on_event(CommandStartedEvent, CommandOutputEvent, CommandFinishedEvent)
    async def observe(
        self, ctx: RunContext[None], event: CommandStartedEvent | CommandOutputEvent | CommandFinishedEvent
    ) -> None:
        self.events.append(event)

    @property
    def output(self) -> str:
        return ''.join(event.text for event in self.events if isinstance(event, CommandOutputEvent))

    @property
    def finished(self) -> CommandFinishedEvent:
        last = self.events[-1]
        assert isinstance(last, CommandFinishedEvent)
        return last


class TestToolSelection:
    async def test_default_tools_are_run_scoped(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Shell()]).run('Inspect tools', workspace=LocalWorkspaceBackend(tmp_path))
        assert model.last_model_request_parameters is not None
        names = [tool.name for tool in model.last_model_request_parameters.function_tools]
        assert names == list(RUN_SCOPED_TOOL_NAMES)

    async def test_selected_tools(self, tmp_path: Path) -> None:
        model = TestModel(call_tools=[])
        await Agent(model, capabilities=[Shell(tools=['shell', 'run_command'])]).run(
            'Inspect tools', workspace=LocalWorkspaceBackend(tmp_path)
        )
        assert model.last_model_request_parameters is not None
        names = [tool.name for tool in model.last_model_request_parameters.function_tools]
        assert names == ['run_command', 'shell']
        assert set(SHELL_TOOL_NAMES) == {*RUN_SCOPED_TOOL_NAMES, 'shell'}

    def test_unknown_tool_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='Unknown shell tools: bogus'):
            Shell(tools=['bogus']).get_toolset()

    @pytest.mark.parametrize('default_timeout', [0, MAX_FOREGROUND_WAIT + 1])
    def test_default_timeout_bounded_for_shell(self, tmp_path: Path, default_timeout: float) -> None:
        with pytest.raises(ValueError, match='default_timeout must be greater than zero and at most 270'):
            Shell(tools=['shell'], default_timeout=default_timeout).get_toolset()
        Shell(default_timeout=default_timeout).get_toolset()


class TestShellTool:
    async def test_foreground(self, tmp_path: Path) -> None:
        output = await shell(tmp_path, {'command': 'mkdir child; printf hello; exit 7'})
        assert 'hello' in output and '"exit_code": 7' in output
        assert (tmp_path / 'child').is_dir()
        output = await shell(
            tmp_path,
            {'command': 'printf "${OPENAI_API_KEY-unset}"'},
            env={'OPENAI_API_KEY': 'do-not-expose'},
            denied_env_patterns=['OPENAI_*'],
        )
        assert 'unset' in output and 'do-not-expose' not in output

    @pytest.mark.parametrize('metadata_exists', [False, True])
    async def test_job_files_live_in_the_git_ignored_metadata_directory(
        self, tmp_path: Path, metadata_exists: bool
    ) -> None:
        metadata = tmp_path / '.pydantic-ai-harness'
        if metadata_exists:
            metadata.mkdir()
        output = await shell(tmp_path, {'command': 'printf hello'})
        assert f'Output: {metadata}/shell/' in output
        assert (metadata / '.gitignore').read_text() == '*\n'

    async def test_events(self, tmp_path: Path) -> None:
        recorder = Recorder()
        result = await shell(tmp_path, {'command': 'printf hello'}, capabilities=[recorder])
        assert 'hello' in result
        started = recorder.events[0]
        assert isinstance(started, CommandStartedEvent)
        assert started.command == 'printf hello'
        assert recorder.output == 'hello'
        assert recorder.finished.total_lines == 1
        assert recorder.finished.exit_code == 0
        assert not recorder.finished.truncated

    async def test_partial_utf8_is_decoded_across_chunks(self, tmp_path: Path) -> None:
        recorder = Recorder()
        await shell(tmp_path, {'command': "printf '\\342\\202'"}, capabilities=[recorder])
        assert recorder.output == '\ufffd'

    async def test_large_sparse_log_is_not_scanned(self, tmp_path: Path) -> None:
        recorder = Recorder()
        script = 'import os; os.ftruncate(1, 1 << 32)'
        command = f'{shlex.quote(sys.executable)} -c {shlex.quote(script)}'
        await shell(tmp_path, {'command': command}, capabilities=[recorder])
        assert recorder.finished.total_lines is None
        assert recorder.finished.truncated

    async def test_output_arrives_before_command_exit(self, tmp_path: Path) -> None:
        release = tmp_path / 'release.pipe'
        os.mkfifo(release)
        recorder = Recorder()

        class Releaser(AbstractCapability[None]):
            @on_event(CommandOutputEvent)
            async def output(self, ctx: RunContext[None], event: CommandOutputEvent) -> None:
                if 'ready' in event.text:
                    await run_sync(release.write_text, 'continue\n')

        result = await shell(
            tmp_path,
            {'command': 'printf ready; read reply < release.pipe; printf done', 'timeout': 5},
            capabilities=[Releaser(), recorder],
        )
        assert 'readydone' in result
        assert recorder.finished.exit_code == 0

    @pytest.mark.parametrize('timeout', [0, 271])
    async def test_timeout_validation(self, tmp_path: Path, timeout: int) -> None:
        assert 'timeout must' in await shell(tmp_path, {'command': 'echo hi', 'timeout': timeout})

    async def test_policy_applies(self, tmp_path: Path) -> None:
        assert 'NUL' in await shell(tmp_path, {'command': 'echo \0'})
        capability = Shell[None](allowed_commands=['echo'], tools=['shell'])
        assert 'not in the allowed list' in await call_tool(
            [capability], 'shell', {'command': 'printf hi'}, workspace=LocalWorkspaceBackend(tmp_path)
        )

    async def test_handles_survive_output_cap(self, tmp_path: Path) -> None:
        output = await shell(tmp_path, {'command': 'yes | head -c 3000'}, max_output_chars=600)
        assert output.startswith('[... output truncated')
        assert 'PID: ' in output and 'Output: ' in output and 'Status: ' in output
        assert '"exit_code": 0' in output.splitlines()[-1]

    async def test_long_output_is_marked_and_starts_on_a_line(self, tmp_path: Path) -> None:
        output = await shell(tmp_path, {'command': 'seq 1 10000'})
        full = ''.join(f'{n}\n' for n in range(1, 10001))
        marker, _, rest = output.partition('\n')
        omitted = int(marker.removeprefix('[... output truncated, ').removesuffix(' earlier bytes omitted]'))
        assert full[omitted - 1] == '\n'
        assert rest.startswith(full[omitted:])
        assert len(full) - omitted <= 16_000

    async def test_long_single_line_is_marked(self, tmp_path: Path) -> None:
        output = await shell(tmp_path, {'command': "head -c 20000 /dev/zero | tr '\\0' x"})
        assert output.startswith('[... output truncated, 4000 earlier bytes omitted]\n' + 'x' * 16_000 + '\n')

    async def test_starts_in_configured_cwd_despite_persist_cwd(self, tmp_path: Path) -> None:
        (tmp_path / 'child').mkdir()
        capability = Shell[None](persist_cwd=True, tools=['run_command', 'shell'])
        moved, listed = await call_tools(
            [capability],
            [('run_command', {'command': 'cd child && pwd'}), ('shell', {'command': 'pwd'})],
            workspace=LocalWorkspaceBackend(tmp_path),
        )
        assert moved.strip().endswith('child')
        assert listed.splitlines()[0] == str(tmp_path.resolve())

    async def test_supervisor_killed_mid_command_returns_stale_status(self, tmp_path: Path) -> None:
        # The supervisor publishes its status before the command runs, so the test sequences the
        # steps itself: wait for the command's output and for that file, then kill the supervisor.
        recorder = Recorder()
        supervisors: list[int] = []
        running = anyio.Event()

        class Running(AbstractCapability[None]):
            """Record the supervisor's pid, then signal once the command has produced output."""

            @on_event(CommandStartedEvent, CommandOutputEvent)
            async def observe(self, ctx: RunContext[None], event: CommandStartedEvent | CommandOutputEvent) -> None:
                if isinstance(event, CommandStartedEvent):
                    supervisors.append(event.pid)
                else:
                    running.set()

        results: list[str] = []

        async def run() -> None:
            results.append(
                await shell(
                    tmp_path,
                    {'command': 'printf ready; sleep 30', 'timeout': 3},
                    capabilities=[Running(), recorder],
                )
            )

        async with anyio.create_task_group() as group:
            group.start_soon(run)
            with anyio.fail_after(10):
                await running.wait()
            # The supervisor survives SIGTERM to publish an exit code, so only SIGKILL leaves the
            # command running with nobody left to publish it.
            os.kill(supervisors[0], signal.SIGKILL)

        output = results[0]
        group_id = int(output.split('kill -- -')[1].split('`')[0])
        try:
            assert '"exit_code": null' in output
            assert recorder.finished.exit_code is None
            started = recorder.events[0]
            assert isinstance(started, CommandStartedEvent)
        finally:
            # The command still runs in the supervisor's group; kill it even when an assertion
            # fails, so a failure does not leave the process behind.
            os.killpg(group_id, signal.SIGKILL)

    async def test_supervisor_failure(self, tmp_path: Path) -> None:
        # The jobs directory exists but cannot hold a new job, so the launcher exits without one.
        jobs = tmp_path / '.pydantic-ai-harness' / 'shell'
        jobs.mkdir(parents=True)
        jobs.chmod(0o500)
        try:
            if os.access(jobs, os.W_OK):  # pragma: no cover - root writes regardless of mode bits
                pytest.skip('mode bits do not bind this user')
            assert 'Shell supervisor exited with 125' in await shell(tmp_path, {'command': 'echo hi'})
        finally:
            jobs.chmod(0o700)

    async def test_no_jobs_directory(self, tmp_path: Path) -> None:
        # A file where the jobs directory belongs fails the call; retrying cannot help.
        (tmp_path / '.pydantic-ai-harness').write_text('')
        result = await shell(tmp_path, {'command': 'echo hi'})
        assert result == 'Cannot create `.pydantic-ai-harness/shell` in the workspace: Not a directory'

    async def test_job_files_removed_while_waiting(self, tmp_path: Path) -> None:
        # The model's command can delete the job directory; the call still returns its handles.
        recorder = Recorder()
        output = await shell(
            tmp_path,
            {'command': 'rm -rf .pydantic-ai-harness/shell/*; sleep 0.3', 'timeout': 1},
            capabilities=[recorder],
        )
        assert output.startswith('PID: ')
        assert output.endswith('status.json')
        assert recorder.finished.exit_code is None
        assert recorder.finished.total_lines == 0
        assert recorder.output == ''

    async def test_a_bare_toolset_starts_no_command(self, tmp_path: Path) -> None:
        toolset = Shell[object](denied_commands=[], tools=['shell']).get_toolset()
        assert toolset is not None

        def call(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('shell', {'command': 'true', 'mode': 'background'})])

        agent = Agent(FunctionModel(call), capabilities=[LocalWorkspace(tmp_path)], toolsets=[toolset])
        with pytest.raises(UserError, match=r'Pass `capabilities=\[Shell\(\)\]` rather than its toolset'):
            await agent.run('go')
        # The call fails before launching, so no job directory is left behind.
        jobs = tmp_path / '.pydantic-ai-harness' / 'shell'
        assert not jobs.exists() or not any(jobs.iterdir())


class TestLifecycle:
    @pytest.mark.parametrize('mode', ['foreground', 'background'])
    async def test_process_survives_run(self, tmp_path: Path, mode: str) -> None:
        connected = anyio.Event()
        release = anyio.Event()
        completed = anyio.Event()
        status: Path | None = None
        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        port = listener.extra(SocketAttribute.local_address)[1]

        async def serve(stream: SocketStream) -> None:
            async with stream:
                assert await receive_exactly(stream, 5) == b'ready'
                connected.set()
                await release.wait()
                await stream.send(b'finish')
                assert await receive_exactly(stream, 4) == b'done'
                completed.set()

        script = (
            f'import socket; s=socket.create_connection(("127.0.0.1", {port})); '
            's.sendall(b"ready"); s.recv(100); s.sendall(b"done"); s.close(); print("completed")'
        )
        async with listener, anyio.create_task_group() as group:
            group.start_soon(listener.serve, serve)
            command = f'{shlex.quote(sys.executable)} -c {shlex.quote(script)}'
            output = await shell(tmp_path, {'command': command, 'mode': mode, 'timeout': 0.01})
            pid = int(output.split('PID: ')[1].split()[0])
            status = Path(output.split('Status: ')[1].splitlines()[0])
            try:
                with anyio.fail_after(10):
                    await connected.wait()
                    # The Agent.run above has returned; the command still waits on our event.
                    os.kill(pid, 0)
                    state = json.loads(status.read_text())
                    assert state['exit_code'] is None
                    release.set()
                    await completed.wait()
            finally:
                release.set()
                group.cancel_scope.cancel()
        assert status is not None
        # Inspect completion using the same tool the agent has.
        script = (
            'import json, pathlib, time; '
            f'p=pathlib.Path({str(status)!r}); '
            '\nwhile json.loads(p.read_text())["exit_code"] is None: time.sleep(0.01)'
            '\nprint(p.read_text())'
        )
        result = await shell(tmp_path, {'command': f'{shlex.quote(sys.executable)} -c {shlex.quote(script)}'})
        assert '"exit_code": 0' in result
        assert status.with_name('output.log').read_text() == 'completed\n'

    @pytest.mark.parametrize('mode', ['foreground', 'background'])
    async def test_failed_result_read_terminates_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        # The log scan is the last read before the result is built. The next read fails, so
        # the handles never reach the model; later reads, including `Job.kill`'s, succeed.
        jobs: list[Job] = []
        failed: list[Job] = []

        async def arm(job: Job) -> int | None:
            jobs.append(job)
            return None

        def fail_once(original: Any) -> Any:
            async def read(self: Job, *args: Any) -> Any:
                if jobs and not failed:
                    failed.append(self)
                    raise RuntimeError('result read failed')
                return await original(self, *args)

            return read

        monkeypatch.setattr('pydantic_ai_harness.shell._persistent._count_lines', arm)
        monkeypatch.setattr(Job, 'size', fail_once(Job.size))
        monkeypatch.setattr(Job, 'status_text', fail_once(Job.status_text))
        with pytest.raises(RuntimeError, match='result read failed'):
            await shell(tmp_path, {'command': 'sleep 60', 'mode': mode, 'timeout': 0.01})
        await wait_for_exit(jobs[0].pid)
        assert not Path(jobs[0].directory).exists()

    @pytest.mark.parametrize('command', ['sleep 60', 'true'])
    async def test_cancelled_finalization_terminates_process(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str
    ) -> None:
        entered = anyio.Event()
        jobs: list[Any] = []

        async def block_line_count(job: Any) -> int | None:
            # Only the log scan is held open; the cancellation path's own cleanup must still run.
            jobs.append(job)
            entered.set()
            await anyio.sleep_forever()

        monkeypatch.setattr('pydantic_ai_harness.shell._persistent._count_lines', block_line_count)

        async def run() -> None:
            await shell(tmp_path, {'command': command, 'mode': 'background'})

        async with anyio.create_task_group() as group:
            group.start_soon(run)
            with anyio.fail_after(10):
                await entered.wait()
            group.cancel_scope.cancel()
        # The killed command is reparented and reaped by init, not by us, so it may
        # linger as a zombie for a moment after the tool call has unwound.
        await wait_for_exit(jobs[0].pid)
        assert not Path(jobs[0].directory).exists()

    async def test_cancelled_foreground_terminates_process(self, tmp_path: Path) -> None:
        connected = anyio.Event()
        listener = await anyio.create_tcp_listener(local_host='127.0.0.1')
        port = listener.extra(SocketAttribute.local_address)[1]
        pid_file = tmp_path / 'pid'

        async def serve(stream: SocketStream) -> None:
            async with stream:
                assert await receive_exactly(stream, 5) == b'ready'
                connected.set()
                await anyio.sleep_forever()

        script = (
            'import os, pathlib, socket; '
            f'pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); '
            f's=socket.create_connection(("127.0.0.1", {port})); s.sendall(b"ready"); s.recv(1)'
        )

        async def run() -> None:
            await shell(tmp_path, {'command': f'{shlex.quote(sys.executable)} -c {shlex.quote(script)}'})

        async with listener, anyio.create_task_group() as group:
            group.start_soon(listener.serve, serve)
            group.start_soon(run)
            with anyio.fail_after(10):
                await connected.wait()
            group.cancel_scope.cancel()
        await wait_for_exit(int(pid_file.read_text()))

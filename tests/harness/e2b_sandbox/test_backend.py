"""Tests for `E2BSandboxBackend`, the E2B implementation of the sandbox protocol."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import IO, Any

import anyio
import anyio.lowlevel
import pytest
from e2b.exceptions import (
    AuthenticationException,
    InvalidArgumentException,
    NotEnoughSpaceException,
    RateLimitException,
    SandboxException,
    SandboxNotFoundException,
    ServiceBusyException,
    TimeoutException,
)

from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.e2b_sandbox import E2BSandboxBackend

from .fake_e2b import FakeCommandHandle, FakeE2B, _HostCommandHandle  # pyright: ignore[reportPrivateUsage]


def _user_line(launch: str) -> str:
    args = shlex.split(launch)
    assert args[:3] == ['setsid', 'sh', '-c']
    return args[3].split('; exec ', 1)[1]


async def started(**settings: Any) -> E2BSandboxBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting `get_sandbox()` is that touch.
    """
    backend = E2BSandboxBackend(**settings)
    await backend.get_sandbox()
    return backend


class TestConformance:
    async def test_get_sandbox_is_lazy_and_reuses_the_sandbox(self, fake_e2b: FakeE2B) -> None:
        backend = E2BSandboxBackend()
        assert not fake_e2b.sandboxes
        sandbox = await backend.get_sandbox()
        assert await backend.get_sandbox() is sandbox
        assert fake_e2b.sandboxes == [sandbox]

    @pytest.mark.parametrize('operation', ['run', 'write_bytes'])
    async def test_ref_is_recorded_by_the_first_operation(self, fake_e2b: FakeE2B, operation: str) -> None:
        backend = E2BSandboxBackend()
        assert backend.ref is None
        if operation == 'run':
            await backend.run(['true'])
        else:
            await backend.write_bytes('/tmp/file', b'data')
        assert backend.ref == WorkspaceRef(provider='e2b', id=fake_e2b.sandboxes[0].sandbox_id)

    async def test_identity_is_e2b_sandbox_id(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
        assert await backend.get_sandbox() is fake_e2b.sandboxes[0]


class TestCreate:
    async def test_creates_from_config(self, fake_e2b: FakeE2B) -> None:
        backend = await started(
            template='base',
            sandbox_timeout=120,
            env={'FOO': 'bar'},
            allow_internet_access=False,
        )
        assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-1')
        call = fake_e2b.create_calls[-1]
        assert (call.template, call.timeout, call.envs, call.allow_internet_access) == (
            'base',
            120,
            {'FOO': 'bar'},
            False,
        )

    async def test_created_sandbox_provisions_working_dir(self, fake_e2b: FakeE2B) -> None:
        backend = await started(working_dir='/work/new')
        assert backend.ref is not None
        assert '/work/new' in fake_e2b.sandboxes[0].files.directories

    async def test_a_failed_working_dir_setup_is_retried_on_next_use(self, fake_e2b: FakeE2B) -> None:
        backend = E2BSandboxBackend(working_dir='/work/new')
        fake_e2b.fs_error = SandboxException('503: envd unavailable', status_code=503)
        with pytest.raises(WorkspaceError, match='Could not create working_dir'):
            await backend.get_sandbox()
        assert backend.ref is not None
        fake_e2b.fs_error = None
        await backend.get_sandbox()
        assert '/work/new' in fake_e2b.sandboxes[0].files.directories
        assert len(fake_e2b.sandboxes) == 1

    async def test_defaults(self, fake_e2b: FakeE2B) -> None:
        await started()
        call = fake_e2b.create_calls[-1]
        # The most E2B's Hobby plan allows, pausing rather than killing at the end of it.
        assert (call.template, call.timeout, call.envs, call.lifecycle) == (None, 3_600, None, {'on_timeout': 'pause'})
        assert call.allow_internet_access is True

    @pytest.mark.parametrize(
        ('error', 'message'),
        [
            (
                SandboxException('404: template xyz not found', status_code=404),
                'Could not start E2B sandbox: 404: template xyz not found',
            ),
            (
                SandboxException('400: Timeout cannot be greater than 1 hours', status_code=400),
                'Hobby plans allow at most 3600 seconds; pass `E2BSandbox(sandbox_timeout=3600)`.',
            ),
            (
                SandboxException('403: forbidden', status_code=403),
                'Could not start E2B sandbox: 403: forbidden',
            ),
            (
                SandboxException('401: unauthorized', status_code=401),
                'Could not start E2B sandbox: 401: unauthorized',
            ),
        ],
        ids=['unknown-template', 'lifetime-over-plan', 'forbidden', 'unauthorized'],
    )
    async def test_a_refused_create_is_unavailable(self, fake_e2b: FakeE2B, error: Exception, message: str) -> None:
        # A refused request fails the same way on every retry, so it ends the run.
        fake_e2b.create_error = error
        with pytest.raises(WorkspaceUnavailableError, match=re.escape(message)) as exc:
            await started()
        assert exc.value.__cause__ is error

    @pytest.mark.parametrize(
        ('message', 'status', 'advice'),
        [
            ('400: reading failed: read tcp 10.0.0.1:443: i/o timeout', 400, False),
            ('400: Timeout cannot be greater than 1 hours', 400, True),
            ('500: Timeout cannot be greater than 1 hours', 500, False),
        ],
    )
    async def test_lifetime_advice_only_for_validation_refusal(
        self, fake_e2b: FakeE2B, message: str, status: int, advice: bool
    ) -> None:
        error = SandboxException(message, status_code=status)
        fake_e2b.create_error = error
        with pytest.raises(WorkspaceError) as exc:
            await started()
        assert str(exc.value) == (
            f'Could not start E2B sandbox: {message}'
            + (' Hobby plans allow at most 3600 seconds; pass `E2BSandbox(sandbox_timeout=3600)`.' if advice else '')
        )
        # Only the lifetime refusal ends the run; a 400 wrapping an i/o timeout stays retryable.
        assert type(exc.value) is (WorkspaceUnavailableError if advice else WorkspaceError)
        assert exc.value.__cause__ is error

    @pytest.mark.parametrize(
        'error',
        [
            SandboxException('500: internal', status_code=500),
            SandboxException('no status'),
            # A 4xx that is not a known refusal keeps the upstream error rather than ending the run.
            SandboxException('409: conflict', status_code=409),
        ],
        ids=['server-error', 'no-status', 'unrecognised-4xx'],
    )
    async def test_an_unrefused_create_failure_is_an_operation_error(self, fake_e2b: FakeE2B, error: Exception) -> None:
        fake_e2b.create_error = error
        with pytest.raises(WorkspaceError, match='Could not start E2B sandbox') as exc:
            await started()
        assert type(exc.value) is WorkspaceError
        assert f'{error}' in str(exc.value)

    async def test_hanging_create_does_not_hang_the_caller(
        self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The client-side bound prevents a wedged control plane from hanging acquisition; like
        # any unreachable service, it propagates as a transient failure.
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._CREATE_TIMEOUT', 0.05)
        fake_e2b.create_hangs = True
        with anyio.fail_after(5):
            with pytest.raises(TimeoutError, match='did not complete within') as exc:
                await started()
        assert type(exc.value) is TimeoutError

    async def test_rejects_relative_working_dir(self, fake_e2b: FakeE2B) -> None:
        with pytest.raises(ValueError, match='working_dir must be an absolute workspace path'):
            await started(working_dir='repo')


class TestConnect:
    async def test_connects_to_an_existing_sandbox(self, fake_e2b: FakeE2B) -> None:
        # E2B resumes a paused sandbox on connect, so no separate liveness probe is needed:
        # a sandbox that is really gone raises instead of handing back a dead handle.
        backend = await started(ref=WorkspaceRef(provider='e2b', id='sbx-keep'))
        assert fake_e2b.connect_calls == [('sbx-keep', 3_600)]
        assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-keep')

    async def test_a_refused_lifetime_on_connect_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        # Attaching with a lifetime over the plan's limit fails the same way on every retry.
        error = SandboxException('400: Timeout cannot be greater than 1 hours', status_code=400)
        fake_e2b.connect_error = error
        with pytest.raises(
            WorkspaceUnavailableError, match=r"'sbx-keep'.*pass `E2BSandbox\(sandbox_timeout=3600\)`"
        ) as exc:
            await started(ref=WorkspaceRef(provider='e2b', id='sbx-keep'), sandbox_timeout=86_400)
        assert exc.value.__cause__ is error

    async def test_connect_to_a_missing_sandbox_fails(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.connect_error = SandboxNotFoundException('not found')
        with pytest.raises(WorkspaceUnavailableError, match="'sbx-gone'"):
            await started(ref=WorkspaceRef(provider='e2b', id='sbx-gone'))
        assert not fake_e2b.create_calls

    async def test_an_operation_on_a_gone_sandbox_does_not_create_a_replacement(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.connect_error = SandboxNotFoundException('not found')
        backend = E2BSandboxBackend(ref=WorkspaceRef(provider='e2b', id='sbx-gone'))
        for _ in range(2):
            with pytest.raises(WorkspaceUnavailableError, match="'sbx-gone'"):
                await backend.run(['true'])
        assert backend.ref == WorkspaceRef(provider='e2b', id='sbx-gone')
        assert not fake_e2b.create_calls
        assert not fake_e2b.sandboxes


class TestRun:
    @pytest.mark.parametrize('mode', ['deadline', 'cancel', 'double-cancel', 'before-ack'])
    async def test_stop_reaches_foreground_child(
        self, fake_e2b: FakeE2B, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        backend = await started()
        commands = fake_e2b.sandboxes[0].commands
        marker = tmp_path / 'survivor'
        fake_e2b.command_hangs = True
        entered = anyio.Event()
        if mode == 'before-ack':
            original = commands.run

            async def delayed(*args: Any, **kwargs: Any) -> Any:
                handle = await original(*args, **kwargs)
                if not entered.is_set():
                    entered.set()
                    await anyio.sleep(0.3)
                return handle

            monkeypatch.setattr(commands, 'run', delayed)
        task = asyncio.create_task(
            backend.run(
                f'(sleep 0.7; touch {marker}) & sleep 30', shell=True, timeout=0.1 if mode == 'deadline' else None
            )
        )
        if mode == 'before-ack':
            await entered.wait()
        elif mode != 'deadline':
            await anyio.sleep(0.1)
        if mode != 'deadline':
            task.cancel()
            if mode == 'double-cancel':
                task.cancel()
        with anyio.fail_after(5):
            with pytest.raises(WorkspaceTimeoutError if mode == 'deadline' else asyncio.CancelledError):
                await task
        assert len(commands.group_stops) == 1
        launch = commands.calls[0].command
        assert shlex.split(launch)[:3] == ['setsid', 'sh', '-c']
        assert 'kill -KILL -' in commands.group_stops[0]
        assert 'pydantic-e2b-pgid-' in launch

    async def test_host_fake_reaps_unfinished_commands_and_closes_output(
        self, fake_e2b: FakeE2B, tmp_path: Path
    ) -> None:
        fake_e2b.host_root = tmp_path
        sandbox = fake_e2b.new_sandbox('host')
        handle = await sandbox.commands.run('sleep 30', background=True)
        assert isinstance(handle, _HostCommandHandle)
        process = handle.process
        assert process.poll() is None
        fake_e2b.close()
        assert process.poll() is not None
        assert handle._out.closed and handle._err.closed  # pyright: ignore[reportPrivateUsage]

    async def test_host_fake_close_ends_background_children_of_a_finished_launcher(
        self, fake_e2b: FakeE2B, tmp_path: Path
    ) -> None:
        # The launcher exits at once, leaving its `sleep` in the isolated group; teardown must end it.
        fake_e2b.host_root = tmp_path
        sandbox = fake_e2b.new_sandbox('host')
        pid_file = tmp_path / 'child.pid'
        handle = await sandbox.commands.run(
            shlex.join(['setsid', 'sh', '-c', f'sleep 30 & echo $! > {shlex.quote(str(pid_file))}']), background=True
        )
        assert isinstance(handle, _HostCommandHandle)
        handle.process.wait()
        child = int(pid_file.read_text())
        os.kill(child, 0)
        fake_e2b.close()
        # The orphaned child is reaped by init after the signal; the bound only guards a hang.
        with anyio.fail_after(5):
            while True:
                try:
                    os.kill(child, 0)
                except ProcessLookupError:
                    break
                await anyio.sleep(0.01)  # pragma: lax no cover - the child is often already reaped on the first check

    async def test_host_fake_closes_output_when_spawn_fails(
        self, fake_e2b: FakeE2B, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_e2b.host_root = tmp_path
        sandbox = fake_e2b.new_sandbox('host')
        opened: list[IO[bytes]] = []
        original = tempfile.TemporaryFile

        def track() -> IO[bytes]:
            stream = original()
            opened.append(stream)
            return stream

        monkeypatch.setattr('tests.harness.e2b_sandbox.fake_e2b.tempfile.TemporaryFile', track)
        with pytest.raises(FileNotFoundError):
            await sandbox.commands.run('true', background=True, cwd=str(tmp_path / 'missing'))
        assert len(opened) == 2
        assert all(stream.closed for stream in opened)

    @pytest.mark.parametrize('delay_ack', [False, True])
    async def test_group_stop_prevents_a_real_child_from_writing(
        self, fake_e2b: FakeE2B, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, delay_ack: bool
    ) -> None:
        fake_e2b.host_root = tmp_path
        backend = await started()
        commands = fake_e2b.sandboxes[0].commands
        if delay_ack:
            original = commands.run

            async def delayed(*args: Any, **kwargs: Any) -> Any:
                handle = await original(*args, **kwargs)
                if not str(args[0]).startswith('sh -c '):
                    await anyio.sleep(0.3)
                return handle

            monkeypatch.setattr(commands, 'run', delayed)
        marker = tmp_path / 'child-marker'
        try:
            with anyio.fail_after(5):
                with pytest.raises(WorkspaceTimeoutError):
                    await backend.run(
                        f'(sleep 0.6; touch {shlex.quote(str(marker))}) & sleep 20', shell=True, timeout=0.1
                    )
            await anyio.sleep(0.7)
            assert not marker.exists()
        finally:
            for handle in fake_e2b.sandboxes[0].commands.handles:
                handle.close()

    async def test_timed_out_command_leaves_no_registration_files(self, fake_e2b: FakeE2B, tmp_path: Path) -> None:
        fake_e2b.host_root = tmp_path
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '20'], timeout=0.1)
        (launch,) = [call.command for call in fake_e2b.sandboxes[0].commands.calls if 'exec sleep' in call.command]
        match = re.search(r'/tmp/pydantic-e2b-pgid-[0-9a-f]+', launch)
        assert match is not None
        assert not Path(match.group()).exists()

    async def test_cancel_before_remote_start_fences_late_start_and_retry(
        self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_e2b.host_root = tmp_path
        backend = await started()
        commands = fake_e2b.sandboxes[0].commands
        original = commands.run
        entered = anyio.Event()
        release = anyio.Event()
        launches: list[str] = []

        async def delayed(cmd: str, **kwargs: Any) -> Any:
            if cmd.startswith('setsid '):
                launches.append(cmd)
                entered.set()
                await release.wait()
            return await original(cmd, **kwargs)

        monkeypatch.setattr(commands, 'run', delayed)
        marker = tmp_path / 'late-marker'
        try:
            for _ in range(2):
                entered = anyio.Event()
                release = anyio.Event()
                task = asyncio.create_task(backend.run(f'touch {marker}', shell=True))
                await entered.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                release.set()
                # Simulate a lost SDK acknowledgement: the remote RPC still executes later.
                handle = await original(launches[-1], background=True)
                assert isinstance(handle, FakeCommandHandle)
                with pytest.raises(Exception, match='143'):
                    await handle.wait()
            assert not marker.exists()
        finally:
            for handle in commands.handles:
                handle.close()

    async def test_missing_setsid_uses_leader_stop_and_caches_probe(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('', '', 127) if 'command -v setsid' in command else ('ok', '', 0)
        backend = await started(template='minimal')
        for _ in range(2):
            assert (await backend.run(['true'])).exit_code == 0
        calls = fake_e2b.sandboxes[0].commands.calls
        assert sum('command -v setsid' in call.command for call in calls) == 1
        assert all(not call.command.startswith('setsid ') for call in calls if 'command -v setsid' not in call.command)
        fake_e2b.command_hangs = True
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '99'], timeout=0.05)
        assert 'kill -TERM "$p"' in calls[-1].command
        assert 'kill -TERM -"$p"' not in calls[-1].command

    async def test_argv_is_quoted_into_one_shell_word_string(self, fake_e2b: FakeE2B) -> None:
        # E2B has no argv form: every command goes through `/bin/bash -l -c`, so the quoting
        # is what keeps an argument with a space or a `$` one literal word.
        backend = await started()
        await backend.run(['echo', 'a b', '$HOME'])
        assert _user_line(fake_e2b.sandboxes[0].commands.calls[0].command) == "echo 'a b' '$HOME'"

    async def test_shell_string_runs_under_sh(self, fake_e2b: FakeE2B) -> None:
        # E2B's login bash would otherwise interpret it; `sh -c` matches every other workspace.
        backend = await started()
        await backend.run('echo hi | wc -c', shell=True)
        assert _user_line(fake_e2b.sandboxes[0].commands.calls[0].command) == "/bin/sh -c 'echo hi | wc -c'"

    async def test_reports_streams_and_exit_code(self, fake_e2b: FakeE2B) -> None:
        # E2B raises `CommandExitException` on a non-zero exit; the protocol calls that a
        # normal result, so the backend unwraps it instead of propagating.
        fake_e2b.responder = lambda command, timeout: ('out', 'err', 2)
        backend = await started()
        result = await backend.run(['false'])
        assert (result.stdout, result.stderr, result.exit_code) == ('out', 'err', 2)

    @pytest.mark.parametrize('exit_code', [0, 2])
    async def test_completed_command_removes_its_stop_registration(self, fake_e2b: FakeE2B, exit_code: int) -> None:
        fake_e2b.responder = lambda command, timeout: ('', '', exit_code)
        backend = await started()
        await backend.run(['true'])
        removed = fake_e2b.sandboxes[0].files.removed
        assert [path.rsplit('-', 1)[0] for path in removed] == ['/tmp/pydantic-e2b-pgid']

    async def test_command_env_adds_nothing_of_its_own(self, fake_e2b: FakeE2B) -> None:
        # E2B decodes output as UTF-8 whatever the locale, so no locale is forced on commands.
        backend = await started()
        await backend.run(['printf', 'é'])
        assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {}

    async def test_env_reaches_the_command(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.run(['env'], env={'FOO': 'bar'})
        assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'FOO': 'bar'}

    async def test_configured_env_reaches_an_attached_sandbox_under_the_commands_own(self, fake_e2b: FakeE2B) -> None:
        # Creation `envs` never reach a sandbox made elsewhere, so the configured env rides on
        # every command, with the command's own values winning.
        backend = await started(
            ref=WorkspaceRef(provider='e2b', id='sbx-keep'), env={'BASE': 'configured', 'SHARED': 'configured'}
        )
        await backend.run(['env'], env={'SHARED': 'command'})
        assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'BASE': 'configured', 'SHARED': 'command'}

    async def test_commands_start_in_the_configured_working_dir(self, fake_e2b: FakeE2B) -> None:
        # E2B has no create-time working directory, so the backend applies it per command.
        backend = await started(working_dir='/work')
        await backend.run(['pwd'])
        assert fake_e2b.sandboxes[0].commands.calls[-1].cwd == '/work'

    async def test_started_in_background_with_the_sdk_deadline_off(self, fake_e2b: FakeE2B) -> None:
        # E2B's own `timeout` abandons the stream and leaves the command running, so it is
        # switched off and the deadline is enforced (and killed) client-side instead.
        backend = await started()
        await backend.run(['x'], timeout=30)
        call = fake_e2b.sandboxes[0].commands.calls[-1]
        assert (call.background, call.timeout) == (True, 0)

    @pytest.mark.parametrize('timeout', [0, -1.0, float('inf'), float('nan')])
    async def test_invalid_timeout_rejected(self, fake_e2b: FakeE2B, timeout: float) -> None:
        backend = await started()
        with pytest.raises(ValueError, match='timeout must be a positive finite number'):
            await backend.run(['x'], timeout=timeout)

    async def test_deadline_kills_and_reports_the_output_so_far(self, fake_e2b: FakeE2B) -> None:
        # The protocol says an expired deadline raises a `TimeoutError`; the output the
        # command produced before the kill rides on the exception, which is the only place
        # the result-or-raise shape leaves for it.
        fake_e2b.responder = lambda command, timeout: ('partial', 'oops', 0)
        fake_e2b.command_hangs = True
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError) as exc:
            await backend.run(['sleep', '99'], timeout=0.05)
        assert (exc.value.stdout, exc.value.stderr) == ('partial', 'oops')
        assert len(fake_e2b.sandboxes[0].commands.group_stops) == 1

    async def test_output_over_the_limit_stops_the_command(self, fake_e2b: FakeE2B) -> None:
        # Like the local backend: past 10 MiB the command is stopped instead of growing this process.
        flood = 'x' * (10 * 1024 * 1024 + 1)
        fake_e2b.responder = lambda command, timeout: (flood, 'oops', 0) if command == 'yes' else ('', '', 0)
        backend = await started()
        with pytest.raises(WorkspaceOutputLimitError) as exc:
            await backend.run(['yes'])
        assert (exc.value.limit, len(exc.value.stdout), exc.value.stderr) == (10 * 1024 * 1024, 64 * 1024, 'oops')
        assert len(fake_e2b.sandboxes[0].commands.group_stops) == 1

    async def test_cancel_during_start_stops_group_without_handle(
        self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        commands = fake_e2b.sandboxes[0].commands
        start = commands.run
        entered = anyio.Event()
        release = anyio.Event()

        async def held(*args: Any, **kwargs: Any) -> Any:
            handle = await start(*args, **kwargs)
            entered.set()
            await release.wait()
            return handle

        monkeypatch.setattr(commands, 'run', held)
        with anyio.fail_after(5):
            async with anyio.create_task_group() as group:
                scope = anyio.CancelScope()

                async def run() -> None:
                    with scope:
                        await backend.run(['sleep', '99'])

                group.start_soon(run)
                await entered.wait()
                scope.cancel()
                release.set()
        assert len(commands.group_stops) == 1

    async def test_a_cancelled_run_kills_the_command(self, fake_e2b: FakeE2B) -> None:
        # The protocol's cancellation contract: a cancelled `run()` must not knowingly leave
        # the command running. The side-channel stop signals its isolated group.
        fake_e2b.command_hangs = True
        backend = await started()
        async with anyio.create_task_group() as tg:
            tg.start_soon(backend.run, ['sleep', '99'])
            await anyio.wait_all_tasks_blocked()
            tg.cancel_scope.cancel()
        assert len(fake_e2b.sandboxes[0].commands.group_stops) == 1

    async def test_a_failed_kill_does_not_replace_the_timeout(
        self, fake_e2b: FakeE2B, caplog: pytest.LogCaptureFixture
    ) -> None:
        fake_e2b.command_hangs = True
        fake_e2b.kill_command_error = SandboxException('kill refused')
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '99'], timeout=0.05)
        assert [record.getMessage() for record in caplog.records if record.levelname == 'WARNING'] == [
            'Could not stop E2B command in sandbox sbx-1'
        ]

    async def test_a_failed_operation_names_what_failed(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.run_error = SandboxException('no such user')
        backend = await started()
        with pytest.raises(WorkspaceError, match='Command could not run in the E2B sandbox: no such user'):
            await backend.run(['x'])

    async def test_a_failing_health_probe_leaves_the_original_error(self, fake_e2b: FakeE2B) -> None:
        # The classifying probe can itself fail; the error being classified propagates as the
        # transient failure it most likely is, rather than a guess replacing it.
        fake_e2b.run_error = TimeoutException('slow')
        fake_e2b.sandbox_is_running = False
        fake_e2b.is_running_error = ConnectionResetError('transport gone')
        backend = await started()
        with pytest.raises(TimeoutException, match='slow'):
            await backend.run(['x'])

    async def test_a_gone_sandbox_names_itself_and_its_lifetime(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.run_error = TimeoutException('unavailable')
        fake_e2b.sandbox_is_running = False
        backend = await started()
        with pytest.raises(WorkspaceUnavailableError, match=r"'sbx-1' is no longer running: .*`sandbox_timeout`"):
            await backend.run(['x'])

    async def test_an_attached_sandbox_names_itself_when_gone(self, fake_e2b: FakeE2B) -> None:
        backend = await started(ref=WorkspaceRef(provider='e2b', id='sbx-keep'))
        fake_e2b.run_error = SandboxNotFoundException('gone')
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await backend.run(['x'])
        # E2B still finds a paused sandbox, so a not-found one was killed, not paused.
        assert str(exc_info.value) == (
            "The E2B sandbox 'sbx-keep' is no longer running: it was killed. "
            "Pass `workspace='new'` to start a fresh sandbox."
        )

    async def test_run_wait_failure_is_a_sandbox_error(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.wait_error = SandboxException('stream broke')
        backend = await started()
        with pytest.raises(WorkspaceError, match='stream broke') as exc:
            await backend.run(['x'])
        assert 'the command may still be running' in str(exc.value)
        # The command may still be running, so it is killed on the way out.
        assert len(fake_e2b.sandboxes[0].commands.group_stops) == 1


class TestKilledSandbox:
    """A sandbox killed through its native handle is gone for every later operation.

    The fake follows the SDK after `kill()`: connecting 404s into `SandboxNotFoundException`,
    and envd calls on a handle that is still held fail with the 502 `TimeoutException` the
    SDK blames on the sandbox timeout, which only the health probe tells from a slow request.
    """

    async def test_attaching_after_kill_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        owner = await started()
        assert owner.ref is not None
        assert await (await owner.get_sandbox()).kill() is True
        attached = E2BSandboxBackend(ref=owner.ref)
        with pytest.raises(WorkspaceUnavailableError, match="'sbx-1' is no longer running"):
            await attached.working_dir()
        assert await fake_e2b.sandboxes[0].kill() is False

    async def test_a_command_on_a_killed_sandbox_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await (await backend.get_sandbox()).kill()
        with pytest.raises(WorkspaceUnavailableError, match='it was killed'):
            await backend.run(['true'])

    async def test_a_filesystem_call_on_a_killed_sandbox_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await (await backend.get_sandbox()).kill()
        with pytest.raises(WorkspaceUnavailableError, match='it was killed'):
            await backend.read_bytes('/tmp/a.txt')

    async def test_a_backend_attached_before_the_kill_is_unavailable(self, fake_e2b: FakeE2B) -> None:
        owner = await started()
        assert owner.ref is not None
        attached = await started(ref=owner.ref)
        await (await owner.get_sandbox()).kill()
        with pytest.raises(WorkspaceUnavailableError, match="'sbx-1' is no longer running"):
            await attached.write_bytes('/tmp/a.txt', b'x')


class TestWorkingDir:
    async def test_a_configured_working_dir_is_resolved_and_initializes_ref(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/real/work\n', '', 0)
        backend = E2BSandboxBackend(working_dir='/work')
        assert await backend.working_dir() == '/real/work'
        assert backend.ref is not None
        assert fake_e2b.sandboxes[0].commands.calls[0].cwd == '/work'

    async def test_probed_once_and_cached(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/home/user\n', '', 0)
        backend = await started()
        assert await backend.working_dir() == '/home/user'
        assert await backend.working_dir() == '/home/user'
        assert [_user_line(call.command) for call in fake_e2b.sandboxes[0].commands.calls] == ['pwd -P']

    async def test_the_probe_carries_a_deadline(self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch) -> None:
        # The probe is a command like any other, so it is bounded and killed rather than left
        # to hang a run that only wanted to resolve a path.
        monkeypatch.setattr('pydantic_ai_harness.e2b_sandbox._backend._INTERNAL_EXEC_TIMEOUT', 0.05)
        fake_e2b.command_hangs = True
        backend = await started()
        with anyio.fail_after(5):
            with pytest.raises(WorkspaceTimeoutError):
                await backend.working_dir()

    @pytest.mark.parametrize(
        ('stdout', 'exit_code'),
        [('', 0), ('relative/dir\n', 0), ('/home/user\n', 1)],
    )
    async def test_an_unusable_answer_is_refused(self, fake_e2b: FakeE2B, stdout: str, exit_code: int) -> None:
        # Caching anything but an absolute path would hand every later `resolve()` a working
        # directory that is not one, mis-resolving relative paths with no error.
        fake_e2b.responder = lambda command, timeout: (stdout, '', exit_code)
        backend = await started()
        with pytest.raises(WorkspaceError, match='Could not determine the working directory'):
            await backend.working_dir()

    async def test_the_facade_resolves_relative_paths_against_it(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/home/user\n', '', 0)
        workspace = Workspace(await started())
        assert await workspace.resolve('src/main.py') == '/home/user/src/main.py'

    async def test_the_facade_preserves_trailing_space_in_probed_working_dir(self, fake_e2b: FakeE2B) -> None:
        fake_e2b.responder = lambda command, timeout: ('/srv/project \n' if command == 'pwd -P' else '', '', 0)
        backend = await started()
        fake_e2b.sandboxes[0].files.files['/srv/project /marker.txt'] = b'found'
        workspace = Workspace(backend)
        assert await workspace.read_text('marker.txt') == 'found'


class TestFilesystem:
    async def test_files_and_commands_share_user(self, fake_e2b: FakeE2B, tmp_path: Path) -> None:
        fake_e2b.host_root = tmp_path
        backend = await started()
        sandbox = fake_e2b.sandboxes[0]
        shared = str(tmp_path / 'shared')
        await backend.make_dir(shared)
        await backend.write_bytes(f'{shared}/file', b'data')
        await backend.read_bytes(f'{shared}/file')
        await backend.stat(f'{shared}/file')
        await backend.list_dir(shared)
        await backend.exists(f'{shared}/file')
        await backend.remove(f'{shared}/file')
        await backend.run(['true'])
        assert sandbox.files.users
        assert sandbox.commands.calls
        assert {*sandbox.files.users, *(call.user for call in sandbox.commands.calls)} == {'user'}

    async def test_upload_deadline_is_finite_and_grows_with_size(
        self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The SDK reads `request_timeout=0` as no deadline at all, and its fixed 60 s default can cut off a large upload."""
        backend = await started()
        files = fake_e2b.sandboxes[0].files
        write = files.write
        seen: list[float | None] = []

        async def track(
            path: str, data: str | bytes, user: str | None = None, request_timeout: float | None = None
        ) -> Any:
            seen.append(request_timeout)
            return await write(path, data, user, request_timeout)

        monkeypatch.setattr(files, 'write', track)
        await backend.write_bytes('/tmp/small', b'data')
        await backend.write_bytes('/tmp/large', bytes(64 * 1024 * 1024))
        small, large = seen
        assert small is not None and large is not None
        assert 60 <= small < large
        assert large > 120

    async def test_stat_reports_size_for_files(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        entry = await backend.stat('/tmp/a.txt')
        assert (entry.name, entry.path, entry.is_dir, entry.size) == ('a.txt', '/tmp/a.txt', False, 4)

    async def test_stat_reports_no_size_for_directories(self, fake_e2b: FakeE2B) -> None:
        # A directory's reported size is a filesystem implementation detail, not a content
        # length, so the protocol carrier reports none.
        backend = await started()
        await backend.make_dir('/tmp/pkg')
        entry = await backend.stat('/tmp/pkg')
        assert (entry.is_dir, entry.size) == (True, None)

    async def test_list_dir_returns_absolute_paths(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.write_bytes('/srv/a.py', b'print(1)')
        await backend.make_dir('/srv/pkg')
        entries = await backend.list_dir('/srv')
        assert [(entry.name, entry.path, entry.is_dir, entry.size) for entry in entries] == [
            ('a.py', '/srv/a.py', False, 8),
            ('pkg', '/srv/pkg', True, None),
        ]

    async def test_symlinks_report_their_target(self, fake_e2b: FakeE2B, tmp_path: Path) -> None:
        fake_e2b.host_root = tmp_path
        (tmp_path / 'data.txt').write_bytes(b'12345')
        (tmp_path / 'pkg').mkdir()
        (tmp_path / 'to-file').symlink_to('data.txt')
        (tmp_path / 'to-dir').symlink_to('pkg')
        (tmp_path / 'dangling').symlink_to('missing')
        entries = await E2BSandboxBackend().list_dir(str(tmp_path))
        assert {entry.name: (entry.is_dir, entry.size) for entry in entries} == {
            'dangling': (False, None),
            'data.txt': (False, 5),
            'pkg': (True, None),
            'to-dir': (True, None),
            'to-file': (False, 5),
        }

    async def test_list_dir_resolves_symlinks_concurrently_in_order(
        self, fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        fake_e2b.host_root = tmp_path
        (tmp_path / 'target').write_bytes(b'abc')
        (tmp_path / 'a').symlink_to('target')
        (tmp_path / 'b').symlink_to('target')
        backend = await started()
        files = fake_e2b.sandboxes[0].files
        original = files.get_info
        entered = 0
        both_entered = anyio.Event()

        async def delayed(path: str, user: str | None = None, request_timeout: float | None = None) -> Any:
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()
            return await original(path, user, request_timeout)

        monkeypatch.setattr(files, 'get_info', delayed)
        with anyio.fail_after(5):
            entries = await backend.list_dir(str(tmp_path))
        assert [entry.name for entry in entries] == ['a', 'b', 'target']

    async def test_list_dir_translates_multiple_denied_symlink_targets(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        files = fake_e2b.sandboxes[0].files
        await backend.make_dir('/srv')
        files.symlinks['/srv/a'] = '/etc/private/a'
        files.symlinks['/srv/b'] = '/etc/private/b'
        files.files['/srv/a'] = b''
        files.files['/srv/b'] = b''
        files.denied.add('/etc')
        for path in ('/srv/a', '/srv/b'):
            with pytest.raises(PermissionError):
                await backend.stat(path)
        with pytest.raises(PermissionError) as exc:
            await backend.list_dir('/srv')
        assert '/srv/a' in str(exc.value)

    async def test_symlink_whose_target_is_gone_reads_as_dangling(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        fake_e2b.sandboxes[0].files.symlinks['/srv/link'] = '/srv/removed'
        entry = await backend.stat('/srv/link')
        assert (entry.is_dir, entry.size) == (False, None)

    async def test_remove_refuses_the_working_dir_and_its_ancestors(self, fake_e2b: FakeE2B, tmp_path: Path) -> None:
        # envd's remove is recursive, so `.` would take the whole working directory.
        root = tmp_path.resolve() / 'work'
        (root / 'child').mkdir(parents=True)
        (root / 'link').symlink_to(root)
        fake_e2b.host_root = root
        workspace = Workspace(await started(working_dir=str(root)))
        with pytest.raises(ValueError, match='workspace root or its ancestor'):
            await workspace.remove('child/..')
        # A link to the root is removed itself, like any other entry.
        await workspace.remove('link')
        await workspace.remove('child')
        assert sorted(root.iterdir()) == []

    async def test_remove_deletes_a_dangling_symlink(self, fake_e2b: FakeE2B, tmp_path: Path) -> None:
        # envd's stat follows the link and reports it missing; the link itself must still go.
        fake_e2b.host_root = tmp_path.resolve()
        (tmp_path / 'dangling').symlink_to('missing')
        await E2BSandboxBackend().remove(str(tmp_path.resolve() / 'dangling'))
        assert not (tmp_path / 'dangling').is_symlink()

    @pytest.mark.parametrize('operation', ['read_bytes', 'stat', 'list_dir', 'remove'])
    async def test_a_missing_path_raises_the_builtin_error(self, fake_e2b: FakeE2B, operation: str) -> None:
        # The protocol's contract: backends translate their SDK's own missing-file exception
        # into the builtin `FileNotFoundError` every consumer already handles.
        backend = await started()
        with pytest.raises(FileNotFoundError, match=r"'/tmp/missing.txt'"):
            await getattr(backend, operation)('/tmp/missing.txt')

    @pytest.mark.parametrize(
        ('operation', 'path', 'expected'),
        [
            ('read_bytes', '/tmp/pkg', IsADirectoryError),
            ('write_bytes', '/tmp/pkg', IsADirectoryError),
            ('list_dir', '/tmp/a.txt', NotADirectoryError),
            ('stat', '/tmp/a.txt/inner', NotADirectoryError),
            ('make_dir', '/tmp/a.txt', FileExistsError),
            ('write_bytes', '/etc/hosts', PermissionError),
        ],
    )
    async def test_a_path_failure_raises_the_builtin_error(
        self, fake_e2b: FakeE2B, operation: str, path: str, expected: type[OSError]
    ) -> None:
        # envd types only a missing path; the rest arrive as a 400 or 500 whose message names
        # the failure, which is what makes them the protocol's builtin file errors.
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        await backend.make_dir('/tmp/pkg')
        fake_e2b.sandboxes[0].files.denied.add('/etc')
        args = (path, b'x') if operation == 'write_bytes' else (path,)
        with pytest.raises(expected, match=re.escape(repr(path))) as exc:
            await getattr(backend, operation)(*args)
        assert type(exc.value) is expected

    async def test_another_invalid_read_stays_a_workspace_error(self, fake_e2b: FakeE2B) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        fake_e2b.read_error = InvalidArgumentException('bad request')
        with pytest.raises(WorkspaceError, match='bad request'):
            await backend.read_bytes('/tmp/a.txt')

    async def test_removing_a_missing_path_does_not_reach_e2b(self, fake_e2b: FakeE2B) -> None:
        # envd's remove succeeds on a missing path, so the backend checks first to report it.
        backend = await started()
        with pytest.raises(FileNotFoundError):
            await backend.remove('/tmp/missing')
        # Only the symlink probe's command claim is cleaned up.
        assert '/tmp/missing' not in fake_e2b.sandboxes[0].files.removed

    async def test_exists_still_reports_other_failures(self, fake_e2b: FakeE2B) -> None:
        # Only "there is nothing at that path" is an answer; anything else is a failure.
        backend = await started()
        fake_e2b.fs_error = SandboxException('input/output error')
        with pytest.raises(WorkspaceError, match='input/output error'):
            await backend.exists('/root/x')


# What E2B raises, whether the sandbox still runs when asked, and what the protocol caller sees.
_OPERATION_ERRORS = [
    pytest.param(AuthenticationException('bad key'), True, WorkspaceUnavailableError, id='rejected-credentials'),
    pytest.param(SandboxNotFoundException('gone'), True, WorkspaceUnavailableError, id='sandbox-not-found'),
    # E2B raises `TimeoutException` for a request the sandbox never answered, slow or dead alike;
    # the health probe tells them apart.
    pytest.param(TimeoutException('unanswered'), False, WorkspaceUnavailableError, id='timeout-sandbox-gone'),
    pytest.param(TimeoutException('unanswered'), True, TimeoutException, id='timeout-sandbox-running'),
    pytest.param(NotEnoughSpaceException('disk full'), True, WorkspaceError, id='operation-failed'),
    pytest.param(RateLimitException('slow down'), True, RateLimitException, id='rate-limited'),
    pytest.param(ServiceBusyException('busy'), True, ServiceBusyException, id='service-busy'),
    pytest.param(ConnectionResetError('reset'), True, ConnectionResetError, id='transport'),
]


@pytest.mark.parametrize('operation', ['run', 'read_bytes'])
@pytest.mark.parametrize(('error', 'running', 'expected'), _OPERATION_ERRORS)
async def test_operation_errors_map_to_protocol_failures(
    fake_e2b: FakeE2B, operation: str, error: Exception, running: bool, expected: type[Exception]
) -> None:
    backend = await started()
    fake_e2b.sandbox_is_running = running
    if operation == 'run':
        fake_e2b.run_error = error
        call = backend.run(['x'])
    else:
        fake_e2b.fs_error = error
        call = backend.read_bytes('/x')
    with pytest.raises(expected) as exc:
        await call
    assert type(exc.value) is expected
    if expected is type(error):
        assert exc.value is error


@pytest.mark.parametrize('attach', [False, True], ids=['create', 'connect'])
@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        pytest.param(AuthenticationException('bad key'), WorkspaceUnavailableError, id='rejected-credentials'),
        pytest.param(SandboxException('no capacity'), WorkspaceError, id='operation-failed'),
        # Nothing is acquired yet to probe, so an unanswered request is transient.
        pytest.param(TimeoutException('unanswered'), TimeoutException, id='timeout'),
        pytest.param(RateLimitException('slow down'), RateLimitException, id='rate-limited'),
        # The SDK's own timeout, not the backend's creation deadline, so it is not rewritten.
        pytest.param(TimeoutError('socket connect timed out'), TimeoutError, id='transport-timeout'),
    ],
)
async def test_acquisition_errors_map_to_protocol_failures(
    fake_e2b: FakeE2B, attach: bool, error: Exception, expected: type[Exception]
) -> None:
    if attach:
        fake_e2b.connect_error = error
    else:
        fake_e2b.create_error = error
    with pytest.raises(expected) as exc:
        await started(ref=WorkspaceRef(provider='e2b', id='sbx-keep') if attach else None)
    assert type(exc.value) is expected
    if expected is type(error):
        assert exc.value is error


async def test_auth_error_classifies_expired_key_without_leaking_it(fake_e2b: FakeE2B) -> None:
    fake_e2b.create_error = AuthenticationException('expired credential sensitive-credential-value')
    with pytest.raises(WorkspaceUnavailableError) as exc:
        await E2BSandboxBackend().get_sandbox()
    assert str(exc.value) == (
        'Credential expired. E2B rejected the credentials. Set a valid E2B_API_KEY in the environment.'
    )
    assert 'sensitive-credential-value' not in str(exc.value)


async def test_missing_key_is_reported_as_not_found(fake_e2b: FakeE2B) -> None:
    """With no key configured the SDK raises before sending anything, so nothing was rejected."""
    fake_e2b.create_error = AuthenticationException(
        'API key is required, please visit the API Keys tab at https://e2b.dev/dashboard?tab=keys to get your API key.'
    )
    with pytest.raises(WorkspaceUnavailableError) as exc:
        await E2BSandboxBackend().get_sandbox()
    assert str(exc.value) == 'No E2B API key found. Set E2B_API_KEY in the environment.'


def test_missing_e2b_extra_has_an_install_hint() -> None:
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            "import sys; sys.modules['e2b'] = None; import pydantic_ai_harness.e2b_sandbox",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert 'Install `pydantic-ai-harness[e2b]`' in result.stderr


async def test_command_timeout_starts_once_the_sandbox_is_acquired(fake_e2b: FakeE2B) -> None:
    fake_e2b.create_response_held = held = anyio.Event()
    deadlines: list[float] = []

    def respond(command: str, timeout: float | None) -> tuple[str, str, int]:
        deadlines.append(anyio.current_effective_deadline())
        return '', '', 0

    fake_e2b.responder = respond
    released_at: list[float] = []

    async def release() -> None:
        # Let the clock move past the moment creation started, so a deadline started at the
        # call would end before one started after acquisition.
        while not fake_e2b.create_calls:
            await anyio.lowlevel.checkpoint()
        entered = anyio.current_time()
        while anyio.current_time() <= entered:  # pragma: lax no cover - the clock may already have moved
            await anyio.lowlevel.checkpoint()
        released_at.append(anyio.current_time())
        held.set()

    async with anyio.create_task_group() as tg:
        tg.start_soon(release)
        await E2BSandboxBackend().run(['true'], timeout=30)
    assert deadlines[0] >= released_at[0] + 30


async def test_filesystem_first_use_preserves_auth_error(fake_e2b: FakeE2B) -> None:
    fake_e2b.create_error = AuthenticationException('denied')
    with pytest.raises(WorkspaceUnavailableError):
        await E2BSandboxBackend().read_bytes('/file')


async def test_timeout_stops_a_command_run_with_a_different_locale(
    fake_e2b: FakeE2B, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A `ps` whose start-time format follows the locale, as it does with LC_ALL=de_DE.UTF-8: the launcher
    # has the user's env and the stopper does not, so the recorded and checked times would differ.
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    fake_ps = bin_dir / 'ps'
    real_ps = shutil.which('ps')
    assert real_ps is not None
    fake_ps.write_text(
        f'#!/bin/sh\ncase "$*" in *lstart*) printf "%s " "${{LC_ALL:-default}}";; esac; exec {real_ps} "$@"\n'
    )
    fake_ps.chmod(0o755)
    monkeypatch.setenv('PATH', f'{bin_dir}{os.pathsep}{os.environ["PATH"]}')
    fake_e2b.host_root = tmp_path
    backend = await started(env={'LC_ALL': 'de_DE.UTF-8'})
    marker = tmp_path / 'child-marker'
    try:
        with anyio.fail_after(5):
            with pytest.raises(WorkspaceTimeoutError):
                await backend.run(f'(sleep 0.6; touch {shlex.quote(str(marker))}) & sleep 20', shell=True, timeout=0.1)
        await anyio.sleep(0.7)
        assert not marker.exists()
    finally:
        for handle in fake_e2b.sandboxes[0].commands.handles:
            handle.close()


async def test_attaching_with_a_missing_working_dir_says_what_to_do(
    fake_e2b: FakeE2B, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_e2b.new_sandbox('sbx-old')
    backend = await started(ref=WorkspaceRef(provider='e2b', id='sbx-old'), working_dir='/srv/app')

    async def missing_cwd(*args: Any, **kwargs: Any) -> Any:
        # E2B's refusal when a command's cwd is not there.
        raise InvalidArgumentException("cwd '/srv/app' does not exist")

    monkeypatch.setattr(fake_e2b.sandboxes[0].commands, 'run', missing_cwd)
    with pytest.raises(WorkspaceError) as caught:
        await backend.run(['pwd'])
    assert str(caught.value) == (
        "working_dir '/srv/app' does not exist in E2B sandbox sbx-old. Create it there, or pass a working_dir that exists."
    )


async def test_created_sandbox_id_is_logged_and_attached_one_is_not(
    fake_e2b: FakeE2B, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level('INFO', logger='pydantic_ai_harness')
    created = await started()
    fake_e2b.new_sandbox('sbx-old')
    await started(ref=WorkspaceRef(provider='e2b', id='sbx-old'))
    created_ref = created.ref
    assert created_ref is not None
    assert [record.getMessage() for record in caplog.records if record.levelname == 'INFO'] == [
        f'Created E2B sandbox {created_ref.id}'
    ]

"""Protocol and SDK-boundary tests for Daytona sandboxes."""

from __future__ import annotations

import asyncio
import errno
import importlib
import itertools
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import daytona
import pytest
from daytona import (
    DaytonaAuthenticationError,
    DaytonaBadRequestError,
    DaytonaConflictError,
    DaytonaConnectionError,
    DaytonaError,
    DaytonaForbiddenError,
    DaytonaGoneError,
    DaytonaNotFoundError,
    DaytonaRateLimitError,
    DaytonaTimeoutError,
    DaytonaUnprocessableEntityError,
)

from pydantic_ai.workspaces import (
    SupportsFilesystem,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.daytona_sandbox import (
    DaytonaSandbox,
    DaytonaSandboxBackend,
)

from .fake_daytona import CreateParams, FakeClient, FakeDaytona, FakeSandbox

pytestmark = pytest.mark.anyio(backends=['asyncio'])


@pytest.fixture
def no_retry_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._RETRY_DELAY', 0)


# Every command runs under a `sh` that sends its stdout and stderr to per-command files.
_WRAPPER = 'sh -c \'"$@" </dev/null >/tmp/MARK.out 2>/tmp/MARK.err\' sh'


def _unmarked(command: str | None) -> str:
    assert command is not None
    return re.sub(r'pydantic-ai-end-[0-9a-f]{32}', 'MARK', command)


async def started(**settings: Any) -> DaytonaSandboxBackend:
    """Build a backend and resolve it now.

    Constructing one does no I/O, so a test that wants to assert on what creating or attaching
    did has to touch the sandbox first. Awaiting `get_sandbox()` is that touch.
    """
    backend = DaytonaSandboxBackend(**settings)
    await backend.get_sandbox()
    return backend


class TestConformance:
    async def test_provider_destroy_does_not_start_sandbox(self, fake_daytona: FakeDaytona) -> None:
        sandbox = fake_daytona.sandbox('sb-cleanup')
        sandbox.state = daytona.SandboxState.STOPPED
        provider = DaytonaSandbox()
        await provider.destroy(WorkspaceRef(provider='daytona', id=sandbox.id))
        assert sandbox.state == daytona.SandboxState.DESTROYING
        assert sandbox.start_calls == []
        # A sandbox being deleted, or one Daytona no longer knows, is already gone.
        await provider.destroy(WorkspaceRef(provider='daytona', id=sandbox.id))
        await provider.destroy(WorkspaceRef(provider='daytona', id='sb-unknown'))
        # Each call opened its own client and closed it.
        assert (fake_daytona.opened_clients, fake_daytona.closed_clients) == (3, 3)
        fake_daytona.get_error = DaytonaAuthenticationError('expired token secret-value', status_code=401)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await provider.destroy(WorkspaceRef(provider='daytona', id=sandbox.id))
        assert str(exc_info.value) == (
            'Credential expired. Daytona rejected the credentials. '
            'Set DAYTONA_API_KEY, or pass a configured `AsyncDaytona` as `client=`.'
        )

    async def test_a_new_sandbox_gets_its_working_dir_and_an_attached_one_does_not(
        self, fake_daytona: FakeDaytona, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger='pydantic_ai_harness.daytona_sandbox._backend'):
            backend = await started(working_dir='/home/daytona/project')
        sandbox = fake_daytona.sandboxes[0]
        assert sandbox.exec_commands == ['mkdir -p -- /home/daytona/project']
        # Logged at creation, so a sandbox whose run ended before its ref was stored can be found.
        assert caplog.messages == [f'Created Daytona sandbox {sandbox.id}']
        await started(ref=backend.ref, working_dir='/home/daytona/other')
        assert sandbox.exec_commands == ['mkdir -p -- /home/daytona/project']

    async def test_a_failed_working_dir_is_retried_by_the_next_operation(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.create_gate = gate = asyncio.Event()
        backend = DaytonaSandboxBackend(working_dir='/root/project')
        task = asyncio.create_task(backend.get_sandbox())
        await fake_daytona.create_started.wait()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.mkdir_exit_code = 1
        gate.set()
        with pytest.raises(
            WorkspaceError, match=r"^Could not create working_dir '/root/project' in Daytona sandbox sb-1"
        ):
            await task
        # The sandbox is recorded, so the caller can clean it up, and the next operation tries again.
        assert backend.ref == WorkspaceRef(provider='daytona', id='sb-1')
        sandbox.mkdir_exit_code = 0
        assert await backend.get_sandbox() is sandbox
        assert sandbox.exec_commands == ['mkdir -p -- /root/project'] * 2

    async def test_a_sandbox_deleted_before_its_working_dir_is_made_is_unavailable(
        self, fake_daytona: FakeDaytona
    ) -> None:
        fake_daytona.create_gate = gate = asyncio.Event()
        backend = DaytonaSandboxBackend(working_dir='/root/project')
        task = asyncio.create_task(backend.get_sandbox())
        await fake_daytona.create_started.wait()
        await fake_daytona.sandboxes[0].delete()
        gate.set()
        with pytest.raises(
            WorkspaceUnavailableError,
            match=r"^The Daytona sandbox 'sb-1' no longer exists \(it was deleted\)\. "
            r"Pass `workspace='new'` to start a fresh sandbox\.$",
        ):
            await task
        assert backend.ref == WorkspaceRef(provider='daytona', id='sb-1')

    async def test_get_sandbox_is_lazy_and_reuses_the_sandbox(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaSandboxBackend()
        assert not fake_daytona.sandboxes
        sandbox = await backend.get_sandbox()
        assert await backend.get_sandbox() is sandbox
        assert fake_daytona.sandboxes == [sandbox]

    async def test_run_and_filesystem_protocols(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        assert isinstance(backend, WorkspaceBackend)
        assert isinstance(backend, SupportsFilesystem)

    async def test_shared_run_and_nonzero_result(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[-1].responder = lambda command, timeout: ('', 2) if 'false' in command else ('hi\n', 0)
        result = await backend.run(['false'])
        assert result.exit_code != 0
        # The next command gets its own reply, not the previous one's.
        assert (await backend.run(['echo', 'hi'])) == type(result)(exit_code=0, stdout='hi\n', stderr='')


class TestCommands:
    @pytest.mark.parametrize(
        ('command', 'shell'),
        [('echo hi', False), (['true'], True), ([], False)],
    )
    async def test_invalid_command_args_do_not_create_a_sandbox(
        self, fake_daytona: FakeDaytona, command: str | list[str], shell: bool
    ) -> None:
        backend = DaytonaSandboxBackend()
        with pytest.raises((TypeError, ValueError)):
            await backend.run(command, shell=shell)
        assert not fake_daytona.sandboxes

    async def test_argv_output_and_context(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaSandboxBackend(working_dir='/work dir')
        await backend.get_sandbox()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['out', 'put']
        sandbox.process_stderr = ['error']
        sandbox.process_exit_code = 3
        result = await backend.run(['printf', 'a b'], env={'A': 'x y'}, timeout=5)
        assert result == type(result)(exit_code=3, stdout='output', stderr='error')
        assert "cd -- '" in sandbox.process_command
        assert 'work dir' in sandbox.process_command
        assert _unmarked(sandbox.process_command).endswith("sh env -- 'A=x y' printf 'a b'")
        assert sandbox.process_sessions == set()

    async def test_sandbox_env_is_layered_under_the_command_env(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(env={'A': '1', 'B': '2'})
        await backend.run(['true'], env={'B': '3'})
        assert _unmarked(fake_daytona.sandboxes[0].process_command) == f'{_WRAPPER} env -- A=1 B=3 true'

    async def test_env_names_that_are_not_shell_identifiers_reach_the_command(
        self, fake_daytona: FakeDaytona, tmp_path: Path
    ) -> None:
        # `env` runs inside the wrapper's `sh`; outside it, a dash `sh` would drop `A.B` before the command ran.
        fake_daytona.host_root = tmp_path.resolve()
        backend = await started(env={'A.B': 'sandbox'})
        result = await backend.run(['env'], env={'C-D': 'command'})
        assert {'A.B=sandbox', 'C-D=command'} <= set(result.stdout.splitlines())

    async def test_command_stdin_is_at_eof(self, fake_daytona: FakeDaytona, tmp_path: Path) -> None:
        fake_daytona.host_root = tmp_path.resolve()
        backend = DaytonaSandboxBackend()
        with anyio.fail_after(5):
            result = await backend.run(['sh', '-c', 'if read line; then echo unexpected; else echo eof; fi'])
        assert result.stdout == 'eof\n'

    async def test_large_output_is_exact_and_a_timeout_keeps_a_bounded_tail(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['abc'] * 20000
        sandbox.process_stderr = ['def'] * 20000
        result = await backend.run(['true'])
        assert (result.stdout, result.stderr) == ('abc' * 20000, 'def' * 20000)

        sandbox.process_hangs = True
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['true'], timeout=0.01)
        assert (exc_info.value.stdout, exc_info.value.stderr) == (('abc' * 20000)[-4096:], ('def' * 20000)[-4096:])

    async def test_status_is_polled_until_the_command_exits(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        process = fake_daytona.sandboxes[0].process
        original = process.get_session_command
        calls = 0

        async def delayed_status(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            result = await original(*args, **kwargs)
            calls += 1
            if calls == 1:
                result.exit_code = None
            return result

        monkeypatch.setattr(process, 'get_session_command', delayed_status)
        result = await backend.run(['true'])
        assert result.exit_code == 0
        assert calls >= 2

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (RuntimeError('download failed'), RuntimeError),
            (DaytonaNotFoundError('no file'), WorkspaceError),
        ],
    )
    async def test_an_output_file_read_failure_propagates(
        self, fake_daytona: FakeDaytona, error: Exception, expected: type[Exception]
    ) -> None:
        # A lost output file on a live sandbox is a failed command result, not a missing path.
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['out']
        sandbox.download_error = error
        with pytest.raises(expected):
            await backend.run(['true'])
        assert not [path for path in sandbox.files if path.startswith('/tmp/pydantic-ai-end-')]

    # `sizes` stands in for the files growing after their sizes were read, as a background child can make them.
    @pytest.mark.parametrize('sizes', ['', '1\n1\n'], ids=['checked-by-size', 'grown-after-the-size-check'])
    async def test_output_over_the_limit_raises_with_a_preview(self, fake_daytona: FakeDaytona, sizes: str) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['x' * (10 * 1024 * 1024 + 1)]
        sandbox.process_stderr = ['oops']
        sandbox.output_sizes_reply = sizes
        with pytest.raises(WorkspaceOutputLimitError) as exc_info:
            await backend.run(['yes'])
        assert (exc_info.value.limit, len(exc_info.value.stdout), exc_info.value.stderr) == (
            10 * 1024 * 1024,
            64 * 1024,
            'oops',
        )
        assert not [path for path in sandbox.files if path.startswith('/tmp/pydantic-ai-end-')]

    async def test_a_garbled_output_size_reply_is_a_workspace_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.output_sizes_reply = 'wc: oops'
        with pytest.raises(WorkspaceError, match=r"^Could not read the command output sizes: 'wc: oops'$"):
            await backend.run(['true'])

    @pytest.mark.usefixtures('no_retry_delay')
    async def test_deadline_carries_partial_output_and_kills(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['partial out']
        sandbox.process_stderr = ['partial err']
        sandbox.process_hangs = True
        sandbox.process_delete_errors = iter([DaytonaError('bad gateway', status_code=502)])
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['sleep', '30'], timeout=0.01)
        # The partial stderr is read from the file the wrapper sends it to, which is then removed.
        assert (exc_info.value.stdout, exc_info.value.stderr) == ('partial out', 'partial err')
        assert sandbox.process_sessions == set()
        assert not [path for path in sandbox.files if path.startswith('/tmp/pydantic-ai-end-')]

    async def test_deadline_includes_exit_status_rpc(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_stdout = ['complete output']
        sandbox.process_status_gate = asyncio.Event()
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(['true'], timeout=0.05)
        assert exc_info.value.stdout == 'complete output'
        assert sandbox.process_sessions == set()

    async def test_session_not_found_on_a_live_sandbox_is_a_workspace_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_status_error = DaytonaNotFoundError('command not found', status_code=404)
        with pytest.raises(WorkspaceError) as exc_info:
            await backend.run(['true'])
        assert type(exc_info.value) is WorkspaceError

    async def test_undeletable_session_is_logged_and_the_original_error_wins(
        self, fake_daytona: FakeDaytona, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_status_error = RuntimeError('status failed')
        sandbox.process_delete_errors = itertools.repeat(RuntimeError('delete failed'))
        with pytest.raises(RuntimeError, match='status failed'):
            await backend.run(['false'])
        assert sandbox.process_delete_calls > 1
        assert 'may still be running' in caplog.text

    async def test_failed_delete_obeys_stop_grace(
        self, fake_daytona: FakeDaytona, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_hangs = True
        sandbox.process_delete_errors = itertools.repeat(DaytonaConnectionError('offline'))
        with anyio.fail_after(10), pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '30'], timeout=0.01)
        assert 'may still be running' in caplog.text

    async def test_kill_on_a_deleted_sandbox_does_not_retry(
        self, fake_daytona: FakeDaytona, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_hangs = True
        task = asyncio.create_task(backend.run(['sleep', '30'], timeout=5))
        await sandbox.process_status_polled.wait()
        await sandbox.delete()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sandbox.process_delete_calls == 1
        assert caplog.text == ''

    @pytest.mark.usefixtures('no_retry_delay')
    async def test_cancellation_kills_process(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_hangs = True
        sandbox.process_delete_errors = iter([DaytonaError('bad gateway', status_code=502)])
        task = asyncio.create_task(backend.run(['sleep', '30']))
        await sandbox.process_status_polled.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sandbox.process_sessions == set()

    async def test_cancellation_during_successful_session_delete_waits_for_cleanup(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        entered = anyio.Event()
        release = anyio.Event()
        original = sandbox.process.delete_session

        async def delayed_delete(session_id: str, request_timeout: float | None = None) -> None:
            entered.set()
            await release.wait()
            await original(session_id, request_timeout=request_timeout)

        monkeypatch.setattr(sandbox.process, 'delete_session', delayed_delete)
        task = asyncio.create_task(backend.run(['true']))
        with anyio.fail_after(5):
            await entered.wait()
            task.cancel()
            closing = asyncio.create_task(backend.aclose())
            await anyio.wait_all_tasks_blocked()
            # The client stays open until the cancelled run's session is deleted.
            assert fake_daytona.closed_clients == 0
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await closing
        assert sandbox.process_sessions == set()
        assert fake_daytona.closed_clients == 1

    async def test_a_missing_working_dir_is_a_workspace_error(self, fake_daytona: FakeDaytona, tmp_path: Path) -> None:
        fake_daytona.host_root = tmp_path.resolve()
        missing = f'{tmp_path.resolve()}/missing'
        # An attached sandbox must already have the configured `working_dir`.
        owner = await started()
        attached = DaytonaSandboxBackend(ref=owner.ref, working_dir=missing)
        with pytest.raises(WorkspaceError, match=f'^working_dir {re.escape(repr(missing))} does not exist in Daytona'):
            await attached.run(['true'])

    async def test_aclose_waits_for_overlapping_runs(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_status_gate = gate = asyncio.Event()
        runs = [asyncio.create_task(backend.run(['true'])) for _ in range(2)]
        with anyio.fail_after(5):
            while len(sandbox.process_sessions) < 2:
                await asyncio.sleep(0)
            closing = asyncio.create_task(backend.aclose())
            await anyio.wait_all_tasks_blocked()
            assert fake_daytona.closed_clients == 0
            gate.set()
            results = await asyncio.gather(*runs)
            await closing
        assert [result.exit_code for result in results] == [0, 0]
        assert fake_daytona.closed_clients == 1

    async def test_cancellation_as_the_command_is_accepted_kills_it(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Setup finishes in the same step the caller is cancelled, so the parent sees a finished setup.
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        original = sandbox.process.execute_session_command

        async def accept_then_cancel(session_id: str, request: object, timeout: int | None = None) -> Any:
            response = await original(session_id, request, timeout=timeout)
            run.cancel()
            return response

        monkeypatch.setattr(sandbox.process, 'execute_session_command', accept_then_cancel)
        run = asyncio.create_task(backend.run(['true']))
        with pytest.raises(asyncio.CancelledError):
            await run
        assert _unmarked(sandbox.process_command) == f'{_WRAPPER} true'
        assert sandbox.process_sessions == set()
        assert sandbox.process_delete_calls == 1

    async def test_a_stopped_sandbox_that_cannot_be_checked_keeps_the_setup_error(
        self, fake_daytona: FakeDaytona
    ) -> None:
        # The lookup is inconclusive, so the setup failure decides, and a transient one stays retryable.
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.state = daytona.SandboxState.STOPPED
        sandbox.refresh_error = DaytonaConnectionError('control plane down')
        with pytest.raises(DaytonaError, match=r'^sandbox stopped$'):
            await backend.run(['true'])
        assert sandbox.start_calls == []

    @pytest.mark.parametrize('state', [daytona.SandboxState.STOPPED, daytona.SandboxState.PAUSED])
    async def test_a_sandbox_stopped_under_a_held_backend_is_unavailable(
        self, fake_daytona: FakeDaytona, state: daytona.SandboxState
    ) -> None:
        # Daytona stops an idle container and pauses an idle VM; attaching again starts either.
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.state = state
        with pytest.raises(WorkspaceUnavailableError, match='was stopped while idle'):
            await backend.run(['true'])
        assert sandbox.start_calls == []
        assert (await DaytonaSandboxBackend(ref=backend.ref).run(['true'])).exit_code == 0
        assert sandbox.start_calls

    async def test_session_setup_timeout_is_transient(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_create_gate = asyncio.Event()
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._REQUEST_TIMEOUT', 0.01)
        with pytest.raises(TimeoutError, match='session setup did not complete'):
            await backend.run(['true'])

    async def test_committed_session_without_ack_is_deleted_on_setup_timeout(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_create_ack_gate = asyncio.Event()
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._REQUEST_TIMEOUT', 0.02)
        with pytest.raises(TimeoutError, match='session setup'):
            await backend.run(['true'])
        assert sandbox.process_sessions == set()
        assert sandbox.process_delete_calls == 1
        assert sandbox.process_command == ''

    # A second cancellation arrives while the setup task is still deleting the session.
    @pytest.mark.parametrize('cancels', [1, 2])
    async def test_native_cancellation_of_committed_session_deletes_it(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch, cancels: int
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._REQUEST_TIMEOUT', 0.02)
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.process_create_ack_gate = asyncio.Event()
        task = asyncio.create_task(backend.run(['true']))
        with anyio.fail_after(2):
            while not sandbox.process_sessions:
                await asyncio.sleep(0)
        for _ in range(cancels):
            task.cancel()
            await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert sandbox.process_sessions == set()
        assert sandbox.process_command == ''

    async def test_session_execution_failure_cleans_up(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.exec_error = DaytonaConnectionError('offline')
        with pytest.raises(DaytonaConnectionError, match='offline'):
            await backend.run(['true'])
        assert sandbox.process_sessions == set()

    @pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
    async def test_invalid_deadline(self, fake_daytona: FakeDaytona, timeout: float) -> None:
        backend = await started()
        with pytest.raises(ValueError, match='positive finite'):
            await backend.run(['true'], timeout=timeout)


# What creating a sandbox raises for each SDK error. `None` means the SDK error propagates unchanged,
# as a transient failure durable engines retry; a refused creation ends the run.
_CREATE_ERRORS: list[tuple[Exception, type[Exception] | None]] = [
    (DaytonaAuthenticationError('bad key', status_code=401), WorkspaceUnavailableError),
    (DaytonaForbiddenError('forbidden', status_code=403), WorkspaceUnavailableError),
    (DaytonaBadRequestError('bad request', status_code=400), WorkspaceUnavailableError),
    (DaytonaConflictError('name taken', status_code=409), WorkspaceUnavailableError),
    (DaytonaNotFoundError('no such snapshot', status_code=404), WorkspaceUnavailableError),
    (DaytonaGoneError('gone', status_code=410), WorkspaceUnavailableError),
    (DaytonaUnprocessableEntityError('unprocessable', status_code=422), WorkspaceUnavailableError),
    (DaytonaRateLimitError('slow down', status_code=429), None),
    (DaytonaTimeoutError('request timeout', status_code=408), None),
    (DaytonaError('bad gateway', status_code=502), None),
    (DaytonaError('unknown'), None),
    (DaytonaConnectionError('offline'), None),
    (DaytonaTimeoutError('slow'), None),
    (RuntimeError('unexpected'), None),
    (TimeoutError('SDK read timed out'), None),
]


class TestErrorsAndFilesystem:
    @pytest.mark.parametrize(('sdk_error', 'expected'), _CREATE_ERRORS, ids=lambda value: type(value).__name__)
    async def test_creation_error_mapping(
        self, fake_daytona: FakeDaytona, sdk_error: Exception, expected: type[Exception] | None
    ) -> None:
        fake_daytona.create_error = sdk_error
        with pytest.raises(Exception) as exc_info:
            await started()
        if expected is None:
            assert exc_info.value is sdk_error
        else:
            assert type(exc_info.value) is expected
            assert exc_info.value.__cause__ is sdk_error

    async def test_auth_error_does_not_expose_token(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.get_error = DaytonaAuthenticationError('expired token secret-value', status_code=401)
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await DaytonaSandboxBackend().get_sandbox()
        assert 'Credential expired' in str(exc_info.value)
        assert 'DAYTONA_API_KEY' in str(exc_info.value)
        assert 'secret-value' not in str(exc_info.value)

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (DaytonaAuthenticationError('bad key', status_code=401), WorkspaceUnavailableError),
            (DaytonaForbiddenError('forbidden', status_code=403), WorkspaceUnavailableError),
            (DaytonaRateLimitError('rate limited', status_code=429), DaytonaRateLimitError),
        ],
    )
    async def test_precreate_lookup_translates_credentials_but_not_rate_limits(
        self, fake_daytona: FakeDaytona, error: Exception, expected: type[Exception]
    ) -> None:
        fake_daytona.get_error = error
        with pytest.raises(expected) as exc_info:
            await DaytonaSandboxBackend().get_sandbox()
        if expected is WorkspaceUnavailableError:
            assert 'DAYTONA_API_KEY' in str(exc_info.value)
        else:
            assert exc_info.value is error
        assert not fake_daytona.create_params

    async def test_lost_creation_reply_recovers_the_accepted_sandbox(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.lose_create_reply = True
        backend = DaytonaSandboxBackend()
        sandbox = await backend.get_sandbox()
        assert backend.ref == WorkspaceRef(provider='daytona', id=sandbox.id)
        assert len(fake_daytona.sandboxes) == 1
        assert fake_daytona.create_params[0].name is not None

    async def test_a_sandbox_still_starting_after_a_create_timeout_is_waited_for(
        self, fake_daytona: FakeDaytona
    ) -> None:
        """The SDK's `create` times out once the sandbox exists when it starts slowly; the backend waits for the start."""
        fake_daytona.slow_start = True
        sandbox = await DaytonaSandboxBackend().get_sandbox()
        assert fake_daytona.sandboxes == [sandbox]
        assert sandbox.state == daytona.SandboxState.STARTED

    async def test_a_retried_creation_finds_the_sandbox_a_lost_reply_left(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.create_gate = gate = asyncio.Event()
        backend = DaytonaSandboxBackend()
        task = asyncio.create_task(backend.get_sandbox())
        await fake_daytona.create_started.wait()
        # The reply is lost, and so is the lookup by name that would recover it.
        fake_daytona.lose_create_reply = True
        fake_daytona.get_error = DaytonaConnectionError('offline')
        gate.set()
        with pytest.raises(DaytonaConnectionError, match=r'^create response lost$'):
            await task
        fake_daytona.get_error = None
        # The retry looks the sandbox up by the same name instead of creating a second one.
        sandbox = await backend.get_sandbox()
        assert fake_daytona.sandboxes == [sandbox]
        assert len(fake_daytona.create_params) == 1
        assert backend.ref == WorkspaceRef(provider='daytona', id=sandbox.id)

    @pytest.mark.parametrize(
        ('error', 'expected'),
        [
            (DaytonaConflictError('busy', status_code=409), WorkspaceError),
            (DaytonaRateLimitError('slow down', status_code=429), None),
        ],
    )
    async def test_attach_errors_are_translated_or_propagate(
        self, fake_daytona: FakeDaytona, error: Exception, expected: type[Exception] | None
    ) -> None:
        fake_daytona.sandbox('sb-existing')
        fake_daytona.get_error = error
        with pytest.raises(Exception) as exc_info:
            await started(ref=WorkspaceRef(provider='daytona', id='sb-existing'))
        if expected is None:
            assert exc_info.value is error
        else:
            assert type(exc_info.value) is expected
            assert str(exc_info.value) == "Could not attach to Daytona sandbox 'sb-existing': busy"

    async def test_created_sandbox_has_provenance_label(self, fake_daytona: FakeDaytona) -> None:
        await DaytonaSandboxBackend().get_sandbox()
        assert fake_daytona.create_params[0].labels == {'created-by': 'pydantic-ai'}

    async def test_refused_creation_names_the_provider_and_the_sdk_message(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.create_error = DaytonaNotFoundError("Snapshot 'nope' not found", status_code=404)
        with pytest.raises(
            WorkspaceUnavailableError, match=r"^Could not start Daytona sandbox: Snapshot 'nope' not found$"
        ):
            await DaytonaSandboxBackend(snapshot='nope').run(['true'])

    async def test_missing_api_key_is_unavailable_with_the_auth_message(self, fake_daytona: FakeDaytona) -> None:
        fake_daytona.client_error = DaytonaAuthenticationError('Authentication credentials not found.')
        with pytest.raises(WorkspaceUnavailableError) as exc_info:
            await DaytonaSandboxBackend().run(['true'])
        assert str(exc_info.value) == (
            'No Daytona credentials found. Set DAYTONA_API_KEY, or pass a configured `AsyncDaytona` as `client=`.'
        )

    async def test_client_setup_failure_other_than_credentials_propagates(self, fake_daytona: FakeDaytona) -> None:
        error = DaytonaConnectionError('offline')
        fake_daytona.client_error = error
        with pytest.raises(DaytonaConnectionError) as exc_info:
            await DaytonaSandboxBackend().run(['true'])
        assert exc_info.value is error

    async def test_concurrent_first_probes_converge(self, fake_daytona: FakeDaytona) -> None:
        # The probe is an idempotent read, so overlapping first calls are allowed to
        # duplicate it: both get the same answer and the cache settles.
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.workdir_gate = asyncio.Event()
        first = asyncio.create_task(backend.working_dir())
        await sandbox.workdir_started.wait()
        second = asyncio.create_task(backend.working_dir())
        with anyio.fail_after(5):
            while sandbox.workdir_calls < 2:
                await asyncio.sleep(0)
        sandbox.workdir_gate.set()
        assert await asyncio.gather(first, second) == ['/srv/repo', '/srv/repo']
        assert sandbox.workdir_calls == 2

    async def test_configured_working_dir_is_probed(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(working_dir='/work')
        assert await backend.working_dir() == '/work'
        assert fake_daytona.sandboxes[0].workdir_calls == 1

    async def test_working_dir_error_is_translated(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].workdir_error = DaytonaBadRequestError('probe failed')
        with pytest.raises(WorkspaceError, match='probe failed'):
            await backend.working_dir()

    async def test_failed_working_dir_probe_names_sdk_output(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].workdir = 'sh: cd: /missing: No such file or directory'
        with pytest.raises(WorkspaceError, match='No such file or directory') as exc_info:
            await backend.working_dir()
        assert exc_info.type is WorkspaceError

    async def test_invalid_native_working_dir_is_rejected(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].workdir = 'relative'
        with pytest.raises(WorkspaceError, match='determine the working directory') as exc_info:
            await backend.working_dir()
        assert exc_info.type is WorkspaceError

    async def test_filesystem_roundtrip_uses_file_entry(self, fake_daytona: FakeDaytona) -> None:
        backend = await started(working_dir='/workspace')
        sandbox = Workspace(backend)
        await sandbox.write_bytes('/workspace/notes/a.txt', b'hello')
        assert await sandbox.read_bytes('/workspace/notes/a.txt') == b'hello'
        entry = await sandbox.stat('/workspace/notes/a.txt')
        assert (entry.path, entry.name, entry.size, entry.is_dir) == (
            '/workspace/notes/a.txt',
            'a.txt',
            5,
            False,
        )
        await sandbox.write_bytes('/root.txt', b'root')
        await sandbox.make_dir('/elsewhere')  # outside the listed directory, must not appear
        await sandbox.make_dir('/workspace/pkg')
        await sandbox.write_bytes('/workspace/pkg/a.py', b'x')
        entries = await sandbox.list_dir('/workspace')
        assert {(entry.name, entry.is_dir, entry.size) for entry in entries} == {
            ('notes', True, None),
            ('pkg', True, None),
        }
        directory = await sandbox.stat('/workspace/pkg')
        assert (directory.is_dir, directory.size) == (True, None)
        assert await sandbox.exists('/workspace/pkg/a.py') is True
        assert await sandbox.exists('/missing') is False
        await sandbox.remove('/workspace/pkg')
        assert await sandbox.exists('/workspace/pkg/a.py') is False

    async def test_a_failed_realpath_is_a_workspace_error(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]

        async def failing_realpath(command: str, cwd: str | None = None, timeout: int | None = None) -> Any:
            assert command.startswith('realpath -m -- ')
            return SimpleNamespace(result='realpath: boom', exit_code=1)

        monkeypatch.setattr(sandbox.process, 'exec', failing_realpath)
        with pytest.raises(WorkspaceError, match=r"^Could not resolve '/x': realpath: boom$"):
            await backend.realpath('/x')

    async def test_a_realpath_request_failure_propagates(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        error = DaytonaConnectionError('offline')

        async def offline(command: str, cwd: str | None = None, timeout: int | None = None) -> Any:
            raise error

        monkeypatch.setattr(sandbox.process, 'exec', offline)
        with pytest.raises(DaytonaConnectionError) as exc_info:
            await backend.realpath('/x')
        assert exc_info.value is error

    async def test_a_directory_is_reported_when_its_type_lookup_fails(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.directories.add('/pkg')

        async def failed_lookup(path: str, request_timeout: float | None = None) -> Any:
            raise DaytonaConnectionError('offline')

        # Without the entry type, the toolbox's own "is a directory" answer still says what happened.
        monkeypatch.setattr(sandbox.fs, 'get_file_info', failed_lookup)
        with pytest.raises(IsADirectoryError, match=r"^Is a directory in the Daytona sandbox: '/pkg'$"):
            await backend.read_bytes('/pkg')

    @pytest.mark.parametrize('find', ['error', 'failure'])
    async def test_symlink_lookup_failures(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch, find: str
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.files['/pkg/a.py'] = b'x'
        error = DaytonaConnectionError('offline')

        async def failing_find(command: str, cwd: str | None = None, timeout: int | None = None) -> Any:
            assert command.startswith('find ')
            if find == 'error':
                raise error
            return SimpleNamespace(result="find: '/pkg/locked': Permission denied", exit_code=1)

        monkeypatch.setattr(sandbox.process, 'exec', failing_find)
        if find == 'error':
            # A transient failure propagates unchanged rather than hiding symlinks.
            with pytest.raises(DaytonaConnectionError) as exc_info:
                await backend.list_dir('/pkg')
            assert exc_info.value is error
        else:
            # A `find` that fails part-way adds nothing; the toolbox listing stands.
            assert [entry.name for entry in await backend.list_dir('/pkg')] == ['a.py']

    async def test_make_dir_refusals_that_are_not_an_existing_file(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.fs_error = DaytonaBadRequestError('invalid mode', status_code=400)
        with pytest.raises(WorkspaceError, match=r"^Could not access '/pkg' in the sandbox: invalid mode$"):
            await backend.make_dir('/pkg')
        sandbox.fs_error = None
        # "not a directory" for a target that is a directory is a parent problem, not `FileExistsError`.
        sandbox.directories.add('/pkg')

        async def refused(path: str, mode: str, request_timeout: float | None = None) -> None:
            raise DaytonaBadRequestError(f'mkdir {path}: not a directory', status_code=400)

        monkeypatch.setattr(sandbox.fs, 'create_folder', refused)
        with pytest.raises(NotADirectoryError, match=r"^Not a directory in the Daytona sandbox: '/pkg'$"):
            await backend.make_dir('/pkg')

    async def test_write_follows_symlink_and_preserves_mode(self, fake_daytona: FakeDaytona, tmp_path: Path) -> None:
        fake_daytona.host_root = tmp_path.resolve()
        backend = await started()
        target = tmp_path / 'target'
        target.write_bytes(b'old')
        target.chmod(0o640)
        link = tmp_path / 'link'
        link.symlink_to(target)
        await backend.write_bytes(str(link), b'new')
        assert link.is_symlink()
        assert target.read_bytes() == b'new'
        assert target.stat().st_mode & 0o777 == 0o640
        assert sorted(item.name for item in tmp_path.iterdir()) == ['link', 'target']

    @pytest.mark.parametrize('below', ['file', 'file/sub'])
    async def test_writing_below_a_file_is_not_a_directory(
        self, fake_daytona: FakeDaytona, tmp_path: Path, below: str
    ) -> None:
        fake_daytona.host_root = tmp_path.resolve()
        backend = await started()
        await backend.write_bytes(f'{tmp_path}/file', b'x')
        with pytest.raises(NotADirectoryError):
            await backend.write_bytes(f'{tmp_path}/{below}/a.py', b'x')

    @pytest.mark.skipif(os.geteuid() == 0, reason='root ignores directory permissions')
    async def test_writing_into_a_protected_directory_is_permission_denied(
        self, fake_daytona: FakeDaytona, tmp_path: Path
    ) -> None:
        fake_daytona.host_root = tmp_path.resolve()
        backend = await started()
        (tmp_path / 'locked').mkdir(mode=0o500)
        with pytest.raises(PermissionError):
            await backend.write_bytes(f'{tmp_path}/locked/pkg/a.py', b'x')

    @pytest.mark.parametrize(
        ('message', 'expected'),
        [
            ('Not a directory', NotADirectoryError),
            ('Is a directory', IsADirectoryError),
            ('Permission denied', PermissionError),
            ('File exists', FileExistsError),
        ],
    )
    async def test_toolbox_validation_errors_use_builtin_types(
        self, fake_daytona: FakeDaytona, message: str, expected: type[OSError]
    ) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].fs_error = DaytonaBadRequestError(message, status_code=400)
        with pytest.raises(expected):
            await backend.stat('/bad')

    async def test_symlink_loop_is_not_an_existing_file(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].fs_error = DaytonaBadRequestError(
            'Too many levels of symbolic links', status_code=400
        )
        assert await backend.exists('/loop') is False
        with pytest.raises(OSError) as raised:
            await backend.stat('/loop')
        assert raised.value.errno == errno.ELOOP

    async def test_reading_a_symlink_loop_is_eloop(self, fake_daytona: FakeDaytona) -> None:
        # The toolbox answers a download through a loop with a 500, not the 400 a stat gets (observed live).
        backend = await started()
        fake_daytona.sandboxes[0].download_error = DaytonaError(
            'Failed to download file: open /loop: too many levels of symbolic links', status_code=500
        )
        with pytest.raises(OSError) as raised:
            await backend.read_bytes('/loop')
        assert raised.value.errno == errno.ELOOP
        # Any other path error is not an answer.
        fake_daytona.sandboxes[0].fs_error = DaytonaBadRequestError('Permission denied', status_code=400)
        with pytest.raises(PermissionError):
            await backend.exists('/locked')

    async def test_newline_in_path_is_rejected_before_acquisition(self, fake_daytona: FakeDaytona) -> None:
        backend = DaytonaSandboxBackend()
        with pytest.raises(ValueError, match='newline'):
            await backend.stat('/bad\nname')
        assert not fake_daytona.sandboxes

    async def test_path_authorization_error_is_permission_denied(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.fs_error = DaytonaForbiddenError('forbidden', status_code=403)
        with pytest.raises(PermissionError, match='Permission denied'):
            await backend.stat('/private')

    async def test_rejected_credentials_on_filesystem_calls_are_unavailable(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        sandbox.fs_error = DaytonaAuthenticationError('denied')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.make_dir('/pkg')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.exists('/pkg')


class TestDeletedSandbox:
    """A deleted sandbox must never look like a missing file."""

    @pytest.mark.parametrize('operation', ['read_bytes', 'stat', 'list_dir', 'remove', 'exists', 'make_dir'])
    async def test_filesystem_on_a_deleted_sandbox_is_unavailable(
        self, fake_daytona: FakeDaytona, operation: str
    ) -> None:
        backend = await started()
        await (await backend.get_sandbox()).delete()
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await getattr(backend, operation)('/note')

    async def test_write_on_a_deleted_sandbox_is_unavailable(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        await (await backend.get_sandbox()).delete()
        with pytest.raises(WorkspaceUnavailableError):
            await backend.write_bytes('/dir/note', b'x')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.write_bytes('/note', b'x')

    async def test_commands_on_a_deleted_sandbox_are_unavailable(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        await (await backend.get_sandbox()).delete()
        with pytest.raises(WorkspaceUnavailableError):
            await backend.run(['true'])
        with pytest.raises(WorkspaceUnavailableError):
            await backend.working_dir()

    async def test_a_purged_sandbox_is_unavailable(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        await sandbox.delete()
        fake_daytona.purge(sandbox)
        with pytest.raises(WorkspaceUnavailableError):
            await backend.read_bytes('/note')

    async def test_a_non_404_answer_from_a_deleted_sandbox_is_unavailable(self, fake_daytona: FakeDaytona) -> None:
        # The toolbox's answer for a deleted sandbox is not documented; the control-plane
        # lookup decides regardless of the error type.
        backend = await started()
        sandbox = fake_daytona.sandboxes[0]
        await sandbox.delete()
        sandbox.gone_error = DaytonaError('bad gateway', status_code=502)
        with pytest.raises(WorkspaceUnavailableError):
            await backend.working_dir()
        with pytest.raises(WorkspaceUnavailableError):
            await backend.read_bytes('/note')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.exists('/note')

    async def test_unknown_ref_does_not_claim_confirmed_deletion(self, fake_daytona: FakeDaytona) -> None:
        with pytest.raises(WorkspaceUnavailableError, match='never existed'):
            await DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id='missing')).get_sandbox()

    def test_blank_ref_is_rejected(self) -> None:
        with pytest.raises(ValueError, match='empty'):
            DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id='  '))

    @pytest.mark.parametrize('purged', [False, True])
    async def test_attaching_to_a_deleted_sandbox_is_unavailable(self, fake_daytona: FakeDaytona, purged: bool) -> None:
        sandbox = fake_daytona.sandbox('sb-deleted')
        await sandbox.delete()
        if purged:
            fake_daytona.purge(sandbox)
        backend = DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id='sb-deleted'))
        with pytest.raises(WorkspaceUnavailableError):
            await backend.working_dir()
        assert sandbox.start_calls == []

    async def test_a_missing_path_on_a_live_sandbox_is_file_not_found(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        with pytest.raises(FileNotFoundError):
            await backend.read_bytes('/missing')
        with pytest.raises(FileNotFoundError):
            await backend.remove('/missing')
        assert await backend.exists('/missing') is False

    async def test_an_inconclusive_lookup_keeps_the_path_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].refresh_error = DaytonaConnectionError('control plane down')
        with pytest.raises(FileNotFoundError):
            await backend.read_bytes('/missing')
        fake_daytona.sandboxes[0].workdir_error = DaytonaConnectionError('toolbox down')
        with pytest.raises(DaytonaConnectionError, match='toolbox down'):
            await backend.working_dir()

    async def test_reading_a_directory_is_a_directory_error(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        await backend.make_dir('/pkg')
        with pytest.raises(IsADirectoryError):
            await backend.read_bytes('/pkg')

    async def test_other_read_failures_propagate(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        await backend.write_bytes('/note', b'x')
        fake_daytona.sandboxes[0].download_error = DaytonaError('bad gateway', status_code=502)
        with pytest.raises(DaytonaError, match='bad gateway'):
            await backend.read_bytes('/note')


class TestLazyOperations:
    @pytest.mark.parametrize('operation', ['run', 'write_bytes', 'working_dir'])
    async def test_ref_is_recorded_by_the_first_operation(self, fake_daytona: FakeDaytona, operation: str) -> None:
        backend = DaytonaSandboxBackend()
        assert backend.ref is None
        if operation == 'run':
            await backend.run(['true'])
        elif operation == 'write_bytes':
            await backend.write_bytes('/note', b'data')
        else:
            await backend.working_dir()
        assert backend.ref == WorkspaceRef(provider='daytona', id=fake_daytona.sandboxes[0].id)

    @pytest.mark.parametrize('operation', ['run', 'read_bytes', 'exists', 'working_dir'])
    async def test_missing_sandbox_remains_terminal(self, fake_daytona: FakeDaytona, operation: str) -> None:
        backend = DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id='missing'))
        for _ in range(2):
            with pytest.raises(WorkspaceUnavailableError):
                if operation == 'run':
                    await backend.run(['true'])
                elif operation == 'read_bytes':
                    await backend.read_bytes('/note')
                elif operation == 'exists':
                    await backend.exists('/note')
                else:
                    await backend.working_dir()
        # A dead reference is reported, not replaced with a fresh environment.
        assert backend.ref == WorkspaceRef(provider='daytona', id='missing')
        assert not fake_daytona.create_params
        assert not fake_daytona.sandboxes

    async def test_command_timeout_starts_once_the_sandbox_is_acquired(self, fake_daytona: FakeDaytona) -> None:
        gate = fake_daytona.create_gate = asyncio.Event()

        async def release() -> None:
            # Creating the sandbox outlasts the timeout; the command itself fits in it comfortably.
            await anyio.sleep(1.1)
            gate.set()

        backend = DaytonaSandboxBackend()
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(release)
            result = await backend.run(['echo', 'ready'], timeout=1)
            assert result.exit_code == 0
        assert 'echo ready' in fake_daytona.sandboxes[0].process_command

    async def test_native_cancellation_during_committed_creation_keeps_ref(self, fake_daytona: FakeDaytona) -> None:
        gate = fake_daytona.create_gate = asyncio.Event()
        backend = DaytonaSandboxBackend()
        task = asyncio.create_task(backend.get_sandbox())
        await fake_daytona.create_started.wait()
        task.cancel()
        task.cancel()
        gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.ref == WorkspaceRef(provider='daytona', id=fake_daytona.sandboxes[0].id)
        assert len(fake_daytona.sandboxes) == 1

    async def test_a_creation_failure_after_cancellation_is_not_reported_to_the_loop(
        self, fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Creation keeps going after the caller is cancelled, then Daytona refuses it.
        gate = fake_daytona.create_gate = asyncio.Event()
        create = FakeClient.create

        async def create_then_refuse(self: FakeClient, params: CreateParams, *, timeout: float = 60) -> FakeSandbox:
            await create(self, params, timeout=timeout)
            raise DaytonaBadRequestError('unknown snapshot', status_code=400)

        monkeypatch.setattr(FakeClient, 'create', create_then_refuse)
        backend = DaytonaSandboxBackend()
        # What reaches the loop's exception handler, such as a failure reported as never retrieved.
        loop = asyncio.get_running_loop()
        errors: list[dict[str, Any]] = []
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _, context: errors.append(context))
        try:
            task = asyncio.create_task(backend.get_sandbox())
            await fake_daytona.create_started.wait()
            task.cancel()
            await asyncio.sleep(0)
            gate.set()
            # The caller's cancellation wins; the refusal is retrieved, not reported.
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            loop.set_exception_handler(previous)
        assert errors == []

    async def test_cancellation_during_creation_keeps_the_created_sandbox(self, fake_daytona: FakeDaytona) -> None:
        gate = fake_daytona.create_gate = asyncio.Event()
        backend = DaytonaSandboxBackend(working_dir='/work')
        with anyio.CancelScope() as scope:

            async def cancel_once_created() -> None:
                await fake_daytona.create_started.wait()
                scope.cancel()
                gate.set()

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(cancel_once_created)
                await backend.run(['true'])
        assert scope.cancelled_caught
        assert backend.ref == WorkspaceRef(provider='daytona', id=fake_daytona.sandboxes[0].id)
        assert fake_daytona.sandboxes[0].process_command == ''
        # The cancelled call does not make `working_dir`; the next operation does.
        assert not any(command.startswith('mkdir') for command in fake_daytona.sandboxes[0].exec_commands)
        assert await backend.working_dir() == '/work'

    async def test_deadline_bounds_session_setup(self, fake_daytona: FakeDaytona) -> None:
        backend = await started()
        fake_daytona.sandboxes[0].process_create_gate = asyncio.Event()
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['true'], timeout=0.01)


def test_missing_daytona_extra_has_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, 'daytona', None)
    for name in [name for name in sys.modules if name.startswith('pydantic_ai_harness.daytona_sandbox')]:
        monkeypatch.delitem(sys.modules, name)
    with pytest.raises(ImportError, match=r'^Install `pydantic-ai-harness\[daytona\]` to use DaytonaSandbox\.$'):
        importlib.import_module('pydantic_ai_harness.daytona_sandbox')


async def test_working_directory_is_cached(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()
    first = await backend.working_dir()
    fake_daytona.sandboxes[0].workdir = '/changed'
    assert await backend.working_dir() == first


async def test_shell_command_is_passed_to_the_session(fake_daytona: FakeDaytona) -> None:
    await DaytonaSandboxBackend().run('printf hello | cat', shell=True)
    assert _unmarked(fake_daytona.sandboxes[0].process_command) == f"{_WRAPPER} /bin/sh -c 'printf hello | cat'"


async def test_attach_finds_target_among_several(fake_daytona: FakeDaytona) -> None:
    fake_daytona.sandbox('sb-decoy')
    target = fake_daytona.sandbox('sb-target')
    backend = DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id=target.id))
    assert (await backend.get_sandbox()).id == target.id


async def test_supplied_client_is_used_and_never_closed(fake_daytona: FakeDaytona) -> None:
    client = daytona.AsyncDaytona()
    backend = DaytonaSandboxBackend(client=client)
    await backend.get_sandbox()
    await backend.aclose()
    assert fake_daytona.sandboxes[0].client is client
    assert fake_daytona.closed_clients == 0
    assert (await backend.run(['true'])).exit_code == 0


async def test_aclose_closes_the_owned_client_and_later_use_reattaches(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()
    await backend.write_bytes('/note', b'kept')
    first = fake_daytona.sandboxes[0].client
    await backend.aclose()
    assert first is not None and first.closed
    assert await backend.read_bytes('/note') == b'kept'
    assert fake_daytona.sandboxes[0].client is not first
    assert len(fake_daytona.create_params) == 1


async def test_aclose_waits_for_an_in_flight_file_operation(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = DaytonaSandboxBackend()
    await backend.write_bytes('/note', b'kept')
    sandbox = fake_daytona.sandboxes[0]
    original = sandbox.fs.download_file
    entered, release = asyncio.Event(), asyncio.Event()

    async def held_download(path: str, timeout: int | None = None) -> bytes:
        entered.set()
        await release.wait()
        return await original(path, timeout)

    monkeypatch.setattr(sandbox.fs, 'download_file', held_download)
    read = asyncio.create_task(backend.read_bytes('/note'))
    with anyio.fail_after(5):
        await entered.wait()
        close = asyncio.create_task(backend.aclose())
        await anyio.wait_all_tasks_blocked()
        assert fake_daytona.closed_clients == 0
        release.set()
        assert await read == b'kept'
        await close
    assert fake_daytona.closed_clients == 1


async def test_working_dir_is_reverified_after_aclose(fake_daytona: FakeDaytona) -> None:
    backend = DaytonaSandboxBackend()
    await backend.working_dir()
    await backend.aclose()
    sandbox = fake_daytona.sandboxes[0]
    await sandbox.delete()
    fake_daytona.purge(sandbox)
    with pytest.raises(WorkspaceUnavailableError):
        await backend.working_dir()


async def test_concurrent_acloses_close_the_client_once(
    fake_daytona: FakeDaytona, caplog: pytest.LogCaptureFixture
) -> None:
    backend = DaytonaSandboxBackend()
    await backend.get_sandbox()
    await asyncio.gather(backend.aclose(), backend.aclose())
    assert fake_daytona.closed_clients == 1
    assert 'Daytona API client' not in caplog.text


@pytest.mark.parametrize('failure', ['error', 'hang'])
async def test_failed_close_is_logged(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
    backend = DaytonaSandboxBackend()
    await backend.get_sandbox()
    if failure == 'error':
        fake_daytona.close_errors = iter([DaytonaConnectionError('offline')])
    else:
        fake_daytona.close_gate = asyncio.Event()
    with anyio.fail_after(5):
        await backend.aclose()
    assert 'Daytona API client' in caplog.text
    fake_daytona.close_gate = None
    # The next operation opens a client of its own and attaches by `ref`.
    await backend.working_dir()
    assert fake_daytona.opened_clients == 2


async def test_supplied_client_survives_acquisition_failure(fake_daytona: FakeDaytona) -> None:
    client = daytona.AsyncDaytona()
    fake_daytona.create_error = DaytonaConnectionError('boom')
    with pytest.raises(DaytonaConnectionError):
        await DaytonaSandboxBackend(client=client).get_sandbox()
    assert fake_daytona.closed_clients == 0


async def test_create_timeout_is_transient(fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._CREATE_TIMEOUT', 0.05)
    fake_daytona.create_gate = asyncio.Event()
    backend = DaytonaSandboxBackend()
    sandbox = await backend.get_sandbox()
    assert backend.ref == WorkspaceRef(provider='daytona', id=sandbox.id)
    assert fake_daytona.closed_clients == 0


async def test_create_timeout_without_the_accepted_sandbox_is_transient(
    fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._CREATE_TIMEOUT', 0.05)
    fake_daytona.create_gate = asyncio.Event()
    # The lookup by name after the deadline does not find the sandbox either.
    fake_daytona.get_error = DaytonaNotFoundError('no sandbox', status_code=404)
    with pytest.raises(
        TimeoutError,
        match=r'^Daytona sandbox creation did not complete within 0\.05s; the Daytona control plane may be unreachable\.$',
    ):
        await DaytonaSandboxBackend().get_sandbox()


async def test_attach_timeout_is_transient(fake_daytona: FakeDaytona, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = fake_daytona.sandbox('sb-existing')
    monkeypatch.setattr('pydantic_ai_harness.daytona_sandbox._backend._CREATE_TIMEOUT', 0.05)
    fake_daytona.get_gate = asyncio.Event()
    with pytest.raises(TimeoutError, match='did not complete'):
        await DaytonaSandboxBackend(ref=WorkspaceRef(provider='daytona', id=existing.id)).get_sandbox()


async def test_repeated_cancellation_finishes_owned_client_close(fake_daytona: FakeDaytona) -> None:
    backend = await started()
    gate = asyncio.Event()
    fake_daytona.close_gate = gate
    task = asyncio.create_task(backend.aclose())
    await asyncio.sleep(0)
    task.cancel()
    task.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert fake_daytona.closed_clients == 1

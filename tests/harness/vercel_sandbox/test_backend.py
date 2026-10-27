"""Protocol and SDK-boundary tests for Vercel sandboxes."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path

import anyio
import httpx2
import pytest
from vercel import sandbox as vercel_sdk

from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.vercel_sandbox import VercelSandbox, VercelSandboxBackend

from .fake_vercel import FakeVercel, api_error

pytestmark = pytest.mark.anyio(backends=['asyncio'])

_BACKEND = 'pydantic_ai_harness.vercel_sandbox._backend'


def _ref(name: str) -> WorkspaceRef:
    return WorkspaceRef(provider='vercel', id=name)


class TestConstruction:
    @pytest.mark.parametrize(
        ('kwargs', 'error', 'message'),
        [
            ({'ref': WorkspaceRef(provider='e2b', id='x')}, ValueError, "expected 'vercel'"),
            ({'ref': WorkspaceRef(provider='vercel', id=' ')}, ValueError, 'ref id cannot be empty'),
            ({'sandbox_timeout': 0}, ValueError, 'sandbox_timeout must be an integer of at least 1'),
            ({'sandbox_timeout': True}, ValueError, 'sandbox_timeout must be an integer of at least 1'),
            ({'working_dir': 'relative'}, ValueError, 'working_dir must be an absolute workspace path'),
            ({'env': {'PORT': 3000}}, TypeError, 'env keys and values must be strings'),
        ],
    )
    def test_invalid_arguments_fail_without_io(self, kwargs: dict[str, object], error: type[Exception], message: str):
        with pytest.raises(error, match=message):
            VercelSandboxBackend(**kwargs)  # pyright: ignore[reportArgumentType]

    async def test_internet_access_is_left_to_vercel_by_default(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        assert backend.ref is None
        await backend.run(['true'])
        [call] = fake_vercel.create_calls
        assert re.fullmatch(r'pydantic-ai-[0-9a-f]{32}', call['name'])
        assert (call['network_policy'], call['env'], call['image']) == (None, None, None)
        assert backend.ref == _ref(call['name'])


class TestCommands:
    async def test_env_layers_the_command_env_over_the_backend_env(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend(env={'A': 'backend', 'B': 'backend'})
        result = await backend.run(['sh', '-c', 'printf %s "$A $B"'], env={'B': 'command'})
        assert result.stdout == 'backend command'
        [sandbox] = fake_vercel.sandboxes.values()
        # The backend env is also the new sandbox's own; each process still gets it explicitly.
        assert sandbox.env == {'A': 'backend', 'B': 'backend'}
        assert sandbox.command_envs == [{'A': 'backend', 'B': 'command'}]

    async def test_timeout_kills_the_command_and_keeps_its_last_output(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        with pytest.raises(WorkspaceTimeoutError, match=r'^Command timed out after 0\.5 seconds$') as raised:
            await backend.run(['sh', '-c', 'printf partial; printf oops >&2; exec sleep 30'], timeout=0.5)
        assert (raised.value.stdout, raised.value.stderr) == ('partial', 'oops')
        [sandbox] = fake_vercel.sandboxes.values()
        assert [process.kill_calls for process in sandbox.processes] == [1]
        # Vercel kills it itself shortly after the deadline, in case the client cannot.
        assert sandbox.kill_after == [10.5]

    async def test_timeout_while_starting_reports_no_output(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        fake_vercel.process_gate = asyncio.Event()
        with pytest.raises(WorkspaceTimeoutError) as raised:
            await backend.run(['true'], timeout=0.05)
        assert (raised.value.stdout, raised.value.stderr) == ('', '')

    async def test_a_failed_kill_still_reports_the_timeout(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.kill_error = api_error(500)
        with pytest.raises(WorkspaceTimeoutError):
            await VercelSandboxBackend().run(['sleep', '30'], timeout=0.2)

    async def test_output_over_the_cap_kills_the_command(
        self, fake_vercel: FakeVercel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._MAX_OUTPUT_BYTES', 1000)
        monkeypatch.setattr(f'{_BACKEND}._OUTPUT_PREVIEW_CHARS', 4)
        backend = VercelSandboxBackend()
        with pytest.raises(WorkspaceOutputLimitError) as raised:
            await backend.run(['yes'])
        assert (raised.value.limit, raised.value.stdout, raised.value.stderr) == (1000, 'y\ny\n', '')
        [sandbox] = fake_vercel.sandboxes.values()
        assert [process.kill_calls for process in sandbox.processes] == [1]

    async def test_cancelling_before_the_process_starts_kills_nothing(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        fake_vercel.process_gate = asyncio.Event()
        with anyio.move_on_after(0.05):
            await backend.run(['true'])
        [sandbox] = fake_vercel.sandboxes.values()
        assert sandbox.processes == []

    async def test_missing_working_dir_of_an_attached_sandbox(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.sandbox('existing')
        missing = str(fake_vercel.host_root / 'missing')
        backend = VercelSandboxBackend(ref=_ref('existing'), working_dir=missing)
        with pytest.raises(
            WorkspaceError, match=r"^working_dir '.*/missing' does not exist in Vercel sandbox 'existing'"
        ):
            await backend.run(['true'])

    async def test_working_dir_probe_failure(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.rewrite = lambda argv: ['/bin/sh', '-c', 'exit 3'] if 'pwd' in argv else argv
        with pytest.raises(WorkspaceError, match=r'`pwd -P` exited 3 and printed '):
            await VercelSandboxBackend().working_dir()


class TestWorkingDirCreation:
    async def test_a_new_sandbox_gets_its_working_dir(self, fake_vercel: FakeVercel) -> None:
        working_dir = str(fake_vercel.host_root / 'a' / 'b')
        backend = VercelSandboxBackend(working_dir=working_dir)
        assert await backend.working_dir() == working_dir
        assert (await backend.run(['pwd'])).stdout == f'{working_dir}\n'

    async def test_a_failed_mkdir_keeps_the_ref_and_is_retried(self, fake_vercel: FakeVercel) -> None:
        blocker = fake_vercel.host_root / 'file'
        blocker.write_text('not a directory')
        backend = VercelSandboxBackend(working_dir=str(blocker / 'project'))
        with pytest.raises(WorkspaceError, match=r'^Could not create working_dir .* in Vercel sandbox '):
            await backend.run(['true'])
        # The sandbox exists and is recorded, so the caller can still delete it.
        assert backend.ref is not None
        blocker.unlink()
        assert (await backend.run(['pwd'])).exit_code == 0
        assert len(fake_vercel.create_calls) == 1

    async def test_a_cancelled_creation_is_recorded_and_the_next_operation_makes_the_dir(
        self, fake_vercel: FakeVercel
    ) -> None:
        working_dir = str(fake_vercel.host_root / 'project')
        backend = VercelSandboxBackend(working_dir=working_dir)
        fake_vercel.create_gate = asyncio.Event()
        task = asyncio.create_task(backend.run(['true']))
        await fake_vercel.create_started.wait()
        task.cancel()
        await asyncio.sleep(0)
        fake_vercel.create_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The billed sandbox was recorded although its caller was cancelled.
        assert backend.ref == _ref(fake_vercel.create_calls[0]['name'])
        assert not (fake_vercel.host_root / 'project').exists()
        assert await backend.working_dir() == working_dir


class TestCreation:
    async def test_a_lost_create_reply_finds_the_sandbox_by_name(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.lose_create_reply = True
        backend = VercelSandboxBackend()
        await backend.run(['true'])
        [call] = fake_vercel.create_calls
        assert fake_vercel.get_calls == [call['name']]
        assert backend.ref == _ref(call['name'])

    async def test_a_transient_failure_propagates_and_a_retry_reuses_the_name(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.create_error = httpx2.ConnectError('unreachable')
        backend = VercelSandboxBackend()
        with pytest.raises(httpx2.ConnectError):
            await backend.run(['true'])
        assert backend.ref is None
        fake_vercel.create_error = None
        await backend.run(['true'])
        first, second = fake_vercel.create_calls
        assert first['name'] == second['name']
        # The retry first looked for a sandbox the failed attempt might have made.
        assert fake_vercel.get_calls == [first['name'], first['name']]

    async def test_a_retry_finds_a_sandbox_whose_reply_was_lost(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.lose_create_reply = True
        fake_vercel.get_errors = iter([httpx2.ConnectError('lookup failed')])
        backend = VercelSandboxBackend()
        with pytest.raises(httpx2.ConnectError, match='create response lost'):
            await backend.run(['true'])
        fake_vercel.lose_create_reply = False
        await backend.run(['true'])
        assert len(fake_vercel.create_calls) == 1
        assert backend.ref == _ref(fake_vercel.create_calls[0]['name'])

    @pytest.mark.parametrize(
        ('error', 'message'),
        [
            (api_error(400, 'Unknown image'), r'^Could not create a Vercel sandbox: Unknown image$'),
            (api_error(401, 'invalid token'), r'^Credentials rejected\. Vercel rejected the credentials\.'),
            (api_error(403, 'forbidden'), r'^Credentials rejected\. Vercel rejected the credentials\.'),
            (vercel_sdk.SandboxCredentialsError('no token'), r'^No Vercel credentials found\. Set VERCEL_OIDC_TOKEN'),
            (
                vercel_sdk.SandboxTerminalStateError('failed', status=vercel_sdk.SandboxStatus.FAILED),
                r"^The Vercel sandbox 'pydantic-ai-[0-9a-f]+' did not start: its session is failed\.$",
            ),
        ],
    )
    async def test_a_refused_create_is_unavailable(self, fake_vercel: FakeVercel, error: Exception, message: str):
        fake_vercel.create_error = error
        with pytest.raises(WorkspaceUnavailableError, match=message):
            await VercelSandboxBackend().run(['true'])

    @pytest.mark.parametrize('status', [408, 429, 500, 503])
    async def test_a_transient_api_error_propagates_unchanged(self, fake_vercel: FakeVercel, status: int) -> None:
        fake_vercel.create_error = error = api_error(status)
        with pytest.raises(vercel_sdk.SandboxApiError) as raised:
            await VercelSandboxBackend().run(['true'])
        assert raised.value is error

    async def test_a_stalled_create_finds_the_sandbox(
        self, fake_vercel: FakeVercel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.05)
        fake_vercel.create_gate = asyncio.Event()
        backend = VercelSandboxBackend()
        await backend.run(['true'])
        assert backend.ref == _ref(fake_vercel.create_calls[0]['name'])

    async def test_a_stalled_create_with_nothing_to_find_is_a_timeout(
        self, fake_vercel: FakeVercel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_BACKEND}._CREATE_TIMEOUT', 0.05)
        monkeypatch.setattr(f'{_BACKEND}._REQUEST_TIMEOUT', 0.05)
        fake_vercel.create_gate = asyncio.Event()
        fake_vercel.get_gate = asyncio.Event()
        with pytest.raises(TimeoutError, match=r'^Vercel sandbox creation did not complete within 0\.05s'):
            await VercelSandboxBackend().run(['true'])


class TestAttach:
    async def test_a_missing_sandbox_is_unavailable(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend(ref=_ref('gone'))
        with pytest.raises(WorkspaceUnavailableError, match=r"^The Vercel sandbox 'gone' was not found"):
            await backend.run(['true'])
        assert backend.ref == _ref('gone')
        assert not fake_vercel.create_calls

    async def test_rejected_credentials_are_unavailable(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.get_errors = iter([api_error(401)])
        with pytest.raises(WorkspaceUnavailableError, match='Vercel rejected the credentials'):
            await VercelSandboxBackend(ref=_ref('existing')).run(['true'])

    async def test_a_server_error_propagates_unchanged(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.get_errors = iter([error := api_error(502)])
        with pytest.raises(vercel_sdk.SandboxApiError) as raised:
            await VercelSandboxBackend(ref=_ref('existing')).run(['true'])
        assert raised.value is error

    async def test_a_stalled_lookup_is_a_timeout(self, fake_vercel: FakeVercel, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(f'{_BACKEND}._REQUEST_TIMEOUT', 0.05)
        fake_vercel.get_gate = asyncio.Event()
        with pytest.raises(TimeoutError, match=r"^Connecting to Vercel sandbox 'existing' did not complete"):
            await VercelSandboxBackend(ref=_ref('existing')).run(['true'])


class TestFailuresDuringACommand:
    async def test_a_deleted_sandbox_is_unavailable(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        sandbox = await backend.get_sandbox()
        await sandbox.destroy()
        with pytest.raises(WorkspaceUnavailableError, match=r'no longer exists \(it was deleted\)'):
            await backend.run(['true'])

    @pytest.mark.parametrize('status', [vercel_sdk.SandboxStatus.STOPPED, vercel_sdk.SandboxStatus.STOPPING])
    async def test_an_ended_session_fails_the_command(
        self, fake_vercel: FakeVercel, status: vercel_sdk.SandboxStatus
    ) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        [sandbox] = fake_vercel.sandboxes.values()
        sandbox.status = status
        fake_vercel.wait_error = api_error(410, 'Session stopped', code='sandbox_stopped')
        with pytest.raises(WorkspaceError, match=r'ended while the command ran: its `sandbox_timeout` ran out'):
            await backend.run(['true'])

    async def test_a_signal_exit_in_a_running_sandbox_is_a_result(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        result = await backend.run(['sh', '-c', 'kill -TERM $$'])
        assert result.exit_code == 143
        # The signal exit was checked against the sandbox, which is still running.
        assert fake_vercel.get_calls == [backend.ref.id if backend.ref else '']

    async def test_a_signal_exit_after_the_session_ended_fails_the_command(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        [sandbox] = fake_vercel.sandboxes.values()
        sandbox.status = vercel_sdk.SandboxStatus.STOPPED
        with pytest.raises(WorkspaceError, match=r'ended while the command ran'):
            await backend.run(['sh', '-c', 'kill -TERM $$'])

    @pytest.mark.parametrize(
        ('error', 'expected', 'message'),
        [
            (api_error(400, 'bad request'), WorkspaceError, r'^Could not read the command result .*: bad request$'),
            (api_error(401), WorkspaceUnavailableError, 'Vercel rejected the credentials'),
            (vercel_sdk.SandboxStreamError('stream broke', code='x'), WorkspaceError, 'stream broke'),
        ],
    )
    async def test_sdk_errors_while_reading_are_mapped(
        self, fake_vercel: FakeVercel, error: Exception, expected: type[Exception], message: str
    ) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        fake_vercel.read_error = error
        with pytest.raises(expected, match=message):
            await backend.run(['sleep', '30'])
        [sandbox] = fake_vercel.sandboxes.values()
        # The command was still running, so it was killed.
        assert [process.kill_calls for process in sandbox.processes] == [1]

    async def test_a_transient_error_propagates_when_the_sandbox_still_runs(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        fake_vercel.wait_error = error = api_error(503)
        with pytest.raises(vercel_sdk.SandboxApiError) as raised:
            await backend.run(['true'])
        assert raised.value is error

    async def test_a_start_error_names_the_start(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        fake_vercel.process_error = api_error(422, 'bad command')
        with pytest.raises(WorkspaceError, match=r'^Command could not run in the Vercel sandbox: bad command$'):
            await backend.run(['true'])

    @pytest.mark.parametrize('lookup', [api_error(500), httpx2.ConnectError('down')])
    async def test_an_inconclusive_lookup_keeps_the_original_error(
        self, fake_vercel: FakeVercel, lookup: Exception
    ) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        fake_vercel.wait_error = api_error(400, 'bad request')
        fake_vercel.get_errors = iter([lookup])
        with pytest.raises(WorkspaceError, match='bad request'):
            await backend.run(['true'])

    async def test_a_stalled_lookup_keeps_the_original_error(
        self, fake_vercel: FakeVercel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        backend = VercelSandboxBackend()
        await backend.get_sandbox()
        monkeypatch.setattr(f'{_BACKEND}._REQUEST_TIMEOUT', 0.05)
        fake_vercel.wait_error = api_error(400, 'bad request')
        fake_vercel.get_gate = asyncio.Event()
        with pytest.raises(WorkspaceError, match='bad request'):
            await backend.run(['true'])


class TestClientLifecycle:
    async def test_aclose_closes_the_owned_client_and_a_later_operation_reattaches(
        self, fake_vercel: FakeVercel
    ) -> None:
        backend = VercelSandboxBackend()
        await backend.aclose()  # Nothing opened yet.
        await Workspace(backend).write_text('note', 'kept')
        await backend.aclose()
        assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (1, 1)
        assert await Workspace(backend).read_text('note') == 'kept'
        assert fake_vercel.opened_clients == 2
        assert fake_vercel.get_calls == [fake_vercel.create_calls[0]['name']]

    async def test_a_supplied_client_is_never_closed(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend(client=vercel_sdk.SandboxClient.create())
        await backend.run(['true'])
        await backend.aclose()
        assert (fake_vercel.opened_clients, fake_vercel.closed_clients) == (1, 0)

    async def test_aclose_during_an_operation_waits_for_it(self, fake_vercel: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        release = fake_vercel.host_root / 'release'
        task = asyncio.create_task(
            backend.run(['sh', '-c', 'while [ ! -f "$1" ]; do sleep 0.01; done', 'sh', str(release)])
        )
        await fake_vercel.process_started.wait()
        await backend.aclose()
        assert fake_vercel.closed_clients == 0
        release.touch()
        assert (await task).exit_code == 0
        assert fake_vercel.closed_clients == 1

    async def test_a_failed_or_stalled_close_is_logged(
        self, fake_vercel: FakeVercel, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        caplog.set_level(logging.WARNING)
        failing = VercelSandboxBackend()
        await failing.run(['true'])
        fake_vercel.close_error = RuntimeError('socket error')
        await failing.aclose()
        fake_vercel.close_error = None

        monkeypatch.setattr(f'{_BACKEND}._TEARDOWN_TIMEOUT', 0.05)
        stalled = VercelSandboxBackend()
        await stalled.run(['true'])
        fake_vercel.close_gate = asyncio.Event()
        await stalled.aclose()
        assert [record.getMessage() for record in caplog.records] == [
            "Could not close the Vercel Sandbox API client: RuntimeError('socket error')",
            'Closing the Vercel Sandbox API client did not complete within 0.05s.',
        ]


class TestDestroy:
    async def test_rejected_credentials_are_unavailable(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.get_errors = iter([api_error(401)])
        with pytest.raises(WorkspaceUnavailableError, match='Vercel rejected the credentials'):
            await VercelSandbox().destroy(_ref('existing'))
        assert fake_vercel.closed_clients == 1

    async def test_a_sandbox_deleted_meanwhile_returns_quietly(self, fake_vercel: FakeVercel) -> None:
        fake_vercel.sandbox()
        fake_vercel.destroy_error = api_error(404)
        await VercelSandbox().destroy(_ref('sb-existing'))


async def test_created_sandbox_is_logged(fake_vercel: FakeVercel, caplog: pytest.LogCaptureFixture, tmp_path: Path):
    caplog.set_level(logging.INFO, logger=_BACKEND)
    backend = VercelSandboxBackend()
    await backend.run(['true'])
    assert backend.ref is not None
    assert [record.getMessage() for record in caplog.records] == [f'Created Vercel sandbox {backend.ref.id}']

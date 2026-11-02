"""`VercelSandboxBackend` behavior the conformance suite does not cover, over the fake Vercel SDK."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import anyio
import httpx2
import pytest
from vercel.sandbox import SandboxCredentialsError, SandboxStatus, SandboxStreamError, SandboxTerminalStateError

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.vercel_sandbox import VercelSandbox, VercelSandboxBackend

from .fake_vercel import FakeVercel, api_error


def _ref(name: str) -> WorkspaceRef:
    return WorkspaceRef(provider='vercel', id=name)


@pytest.mark.parametrize(('ref', 'message'), [(WorkspaceRef(provider='e2b', id='x'), 'expected'), (_ref(' '), 'empty')])
def test_a_foreign_or_blank_ref_is_refused(ref: WorkspaceRef, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        VercelSandboxBackend(ref=ref)


class TestCreate:
    async def test_first_use_creates_one_sandbox_with_the_settings(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend(image='node24', vcpus=4, persistent=False, execution_time_limit=60)
        await backend.run(['true'])
        assert backend.ref == _ref(host.only.name)
        assert host.only.name.startswith('pydantic-ai-')
        kwargs = host.only.create_kwargs
        assert (kwargs['image'], kwargs['resources'].vcpus, kwargs['persistent'], kwargs['execution_time_limit']) == (
            'node24',
            4,
            False,
            60,
        )

    async def test_a_lost_create_reply_is_found_by_name_instead_of_creating_another(self, host: FakeVercel) -> None:
        host.lost_replies = 1
        backend = VercelSandboxBackend()
        with pytest.raises(httpx2.ReadError):
            await backend.run(['true'])
        await backend.run(['true'])
        assert len(set(host.create_names)) == 1 and len(host.sandboxes) == 1

    @pytest.mark.parametrize(
        'error',
        [
            api_error(400, 'bad_request'),
            SandboxTerminalStateError('failed', status=SandboxStatus.FAILED),
            SandboxCredentialsError('none'),
        ],
        ids=['refused', 'failed-to-start', 'no-credentials'],
    )
    async def test_a_create_no_retry_can_fix_is_unavailable(self, host: FakeVercel, error: Exception) -> None:
        host.create_errors.append(error)
        with pytest.raises(WorkspaceUnavailableError):
            await VercelSandboxBackend().run(['true'])

    async def test_a_server_error_propagates_so_durable_engines_retry_it(self, host: FakeVercel) -> None:
        host.create_errors.append(api_error(500))
        with pytest.raises(Exception, match='500'):
            await VercelSandboxBackend().run(['true'])

    async def test_a_new_sandbox_gets_its_working_dir(self, host: FakeVercel, tmp_path: Path) -> None:
        directory = str(tmp_path.resolve() / 'project' / 'nested')
        backend = VercelSandboxBackend(working_dir=directory)
        assert await backend.working_dir() == directory

    async def test_a_working_dir_that_cannot_be_made_fails(self, host: FakeVercel, tmp_path: Path) -> None:
        (tmp_path / 'file').write_text('')
        backend = VercelSandboxBackend(working_dir=str(tmp_path.resolve() / 'file' / 'dir'))
        with pytest.raises(WorkspaceError, match='Could not create working_dir'):
            await backend.run(['true'])

    async def test_cancelling_the_creating_operation_records_the_sandbox_and_finishes_setup_later(
        self, host: FakeVercel, tmp_path: Path
    ) -> None:
        directory = tmp_path.resolve() / 'project'
        backend = VercelSandboxBackend(working_dir=str(directory))
        host.create_gate = asyncio.Event()
        async with anyio.create_task_group() as group:
            group.start_soon(backend.working_dir)
            while not host.create_names:
                await anyio.sleep(0)
            group.cancel_scope.cancel()
            host.create_gate.set()
        assert backend.ref == _ref(host.only.name)
        assert not directory.exists()
        assert await backend.working_dir() == str(directory)
        assert len(host.create_names) == 1


class TestAttach:
    @pytest.mark.parametrize(
        ('persistent', 'status'),
        [(False, SandboxStatus.STOPPED), (True, SandboxStatus.FAILED), (True, SandboxStatus.ABORTED)],
    )
    async def test_a_sandbox_that_cannot_run_commands_is_unavailable_and_not_replaced(
        self, host: FakeVercel, persistent: bool, status: SandboxStatus
    ) -> None:
        host.add('dead', persistent=persistent, status=status)
        with pytest.raises(WorkspaceUnavailableError):
            await VercelSandboxBackend(ref=_ref('dead')).run(['true'])
        assert not host.create_names

    async def test_rejected_credentials_while_attaching_are_unavailable(self, host: FakeVercel) -> None:
        host.call_error = api_error(401)
        with pytest.raises(WorkspaceUnavailableError, match='did not accept the credentials'):
            await VercelSandboxBackend(ref=_ref('existing')).run(['true'])

    async def test_a_stopped_persistent_sandbox_is_attached(self, host: FakeVercel) -> None:
        """Its first command resumes it; the SDK does that, so the backend only lets it through."""
        sandbox = host.add('kept', status=SandboxStatus.STOPPED)
        await VercelSandboxBackend(ref=_ref('kept')).get_sandbox()
        assert sandbox in host.sandboxes.values()

    async def test_a_pending_sandbox_is_waited_for(self, host: FakeVercel) -> None:
        host.add('booting').pending_polls = 3
        assert (await VercelSandboxBackend(ref=_ref('booting')).run(['true'])).exit_code == 0

    async def test_a_sandbox_that_never_starts_times_out(
        self, host: FakeVercel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.vercel_sandbox._backend._LIFECYCLE_TIMEOUT', 0.05)
        host.add('stuck').pending_polls = 10**9
        with pytest.raises(TimeoutError, match='did not start'):
            await VercelSandboxBackend(ref=_ref('stuck')).run(['true'])

    async def test_an_attached_sandbox_must_already_have_the_working_dir(
        self, host: FakeVercel, tmp_path: Path
    ) -> None:
        host.add()
        backend = VercelSandboxBackend(ref=_ref('existing'), working_dir=str(tmp_path / 'missing'))
        with pytest.raises(WorkspaceError, match='does not exist'):
            await backend.run(['true'])


class TestRun:
    async def test_env_is_layered_over_the_backend_env(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend(env={'A': 'backend', 'B': 'backend'})
        result = await backend.run(['sh', '-c', 'printf %s-%s "$A" "$B"'], env={'B': 'call'})
        assert result.stdout == 'backend-call'

    async def test_output_over_the_limit_kills_the_command(
        self, host: FakeVercel, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.vercel_sandbox._backend._MAX_OUTPUT_BYTES', 8)
        with pytest.raises(WorkspaceOutputLimitError) as exc_info:
            await VercelSandboxBackend().run(['sh', '-c', 'printf 0123456789; sleep 5; touch after'])
        assert exc_info.value.stdout.startswith('0123')
        await host.reap()
        assert not (tmp_path / 'after').exists()

    async def test_a_session_that_ends_during_a_command_is_unavailable(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.run(['true'])

        async def stop_soon() -> None:
            await anyio.sleep(0.2)
            host.only.end(SandboxStatus.STOPPED)

        async with anyio.create_task_group() as group:
            group.start_soon(stop_soon)
            with pytest.raises(WorkspaceUnavailableError, match='ended during the command'):
                await backend.run(['sleep', '30'])

    async def test_a_stream_failure_on_a_running_sandbox_propagates(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.run(['true'])
        host.run_error = SandboxStreamError('connection reset', code='stream_error')
        with pytest.raises(SandboxStreamError):
            await backend.run(['true'])

    async def test_an_inconclusive_check_keeps_the_original_failure(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.run(['true'])
        host.call_error = httpx2.ConnectError('offline')
        with pytest.raises(httpx2.ConnectError):
            await backend.run(['true'])

    @pytest.mark.parametrize(
        ('error', 'raised'), [(api_error(400), WorkspaceError), (api_error(401), WorkspaceUnavailableError)]
    )
    async def test_a_refused_command_is_a_workspace_error(
        self, host: FakeVercel, error: Exception, raised: type[Exception]
    ) -> None:
        backend = VercelSandboxBackend()
        await backend.run(['true'])
        host.run_error = error
        with pytest.raises(raised):
            await backend.run(['true'])

    async def test_a_working_dir_that_cannot_be_resolved_is_a_workspace_error(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend(env={'PATH': '/nonexistent'})
        with pytest.raises(WorkspaceError, match='Could not determine the working directory'):
            await backend.working_dir()


async def test_get_sandbox_returns_the_current_handle(host: FakeVercel) -> None:
    assert (await VercelSandboxBackend().get_sandbox()) is host.only


class TestClientLifecycle:
    async def test_a_passed_client_is_left_open(self, host: FakeVercel) -> None:
        client = host.create()
        backend = VercelSandboxBackend(client=client)  # pyright: ignore[reportArgumentType]
        await backend.run(['true'])
        await backend.aclose()
        assert not client.closed

    async def test_an_operation_after_aclose_reattaches_with_a_new_client(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.run(['touch', 'kept'])
        await backend.aclose()
        assert (await backend.run(['test', '-f', 'kept'])).exit_code == 0
        assert [client.closed for client in host.clients] == [True, False]
        assert len(host.create_names) == 1

    async def test_aclose_during_an_operation_closes_the_client_when_it_ends(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        async with anyio.create_task_group() as group:
            group.start_soon(backend.run, ['sleep', '0.3'])
            await anyio.sleep(0.1)
            await backend.aclose()
            assert [client.closed for client in host.clients] == [False]
        assert [client.closed for client in host.clients] == [True]

    async def test_concurrent_acloses_close_the_client_once(self, host: FakeVercel) -> None:
        backend = VercelSandboxBackend()
        await backend.run(['true'])
        await asyncio.gather(backend.aclose(), backend.aclose())
        assert [client.closes for client in host.clients] == [1]

    @pytest.mark.parametrize('failure', ['error', 'hang'])
    async def test_a_failing_close_is_logged_not_raised(
        self, host: FakeVercel, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
    ) -> None:
        monkeypatch.setattr('pydantic_ai_harness.vercel_sandbox._backend._TEARDOWN_TIMEOUT', 0.05)
        backend = VercelSandboxBackend()
        await backend.run(['true'])
        if failure == 'error':
            host.close_error = RuntimeError('boom')
        else:
            host.close_gate = asyncio.Event()
        with caplog.at_level(logging.WARNING):
            await backend.aclose()
        message = (
            "Could not close the Vercel Sandbox client: RuntimeError('boom')"
            if failure == 'error'
            else 'Closing the Vercel Sandbox client did not complete within 0.05s.'
        )
        assert caplog.messages == [message]


class TestDestroy:
    async def test_destroy_destroys_the_sandbox(self, host: FakeVercel) -> None:
        host.add()
        await VercelSandbox().destroy(_ref('existing'))
        assert host.destroyed == ['existing']
        assert [client.closed for client in host.clients] == [True]

    async def test_destroying_a_missing_sandbox_returns_quietly(self, host: FakeVercel) -> None:
        await VercelSandbox().destroy(_ref('missing'))

    async def test_destroy_with_a_passed_client_leaves_it_open(self, host: FakeVercel) -> None:
        host.add()
        await VercelSandbox[None](client=host.create()).destroy(_ref('existing'))  # pyright: ignore[reportArgumentType]
        assert [client.closed for client in host.clients] == [False]

    async def test_destroy_with_rejected_credentials_is_unavailable(self, host: FakeVercel) -> None:
        host.call_error = api_error(403)
        with pytest.raises(WorkspaceUnavailableError):
            await VercelSandbox().destroy(_ref('existing'))

    async def test_destroy_lets_a_transient_failure_propagate(self, host: FakeVercel) -> None:
        host.call_error = api_error(503)
        with pytest.raises(Exception, match='503'):
            await VercelSandbox().destroy(_ref('existing'))

    @pytest.mark.parametrize(
        ('ref', 'message'), [(WorkspaceRef(provider='e2b', id='x'), 'expected'), (_ref(' '), 'empty')]
    )
    async def test_destroy_refuses_a_foreign_or_blank_ref(
        self, fake_vercel: FakeVercel, ref: WorkspaceRef, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            await VercelSandbox().destroy(ref)


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_vercel_call(fake_vercel: FakeVercel) -> None:
    with pytest.raises(UserError, match='asyncio'):
        await VercelSandboxBackend().run(['true'])
    with pytest.raises(UserError, match='asyncio'):
        await VercelSandbox().destroy(_ref('existing'))
    assert not fake_vercel.clients

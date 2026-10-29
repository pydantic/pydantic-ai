"""SDK-boundary tests for `OpenSandboxBackend`, against the fake server.

Each test pins one behavior of OpenSandbox the backend has to handle; the conformance suite covers the protocol.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import signal
import types
import uuid
from pathlib import Path
from typing import Any

import anyio
import pytest
from opensandbox.config import ConnectionConfig
from opensandbox.exceptions import SandboxApiException, SandboxConnectionException, SandboxTimeoutException

from pydantic_ai import RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.opensandbox_sandbox import OpenSandbox, OpenSandboxBackend, _backend

from .fake_opensandbox import DOMAIN, FakeOpenSandbox, wait_until_gone

pytestmark = pytest.mark.anyio(backends=['asyncio'])

_MODULE = 'pydantic_ai_harness.opensandbox_sandbox._backend'


def _create(server: FakeOpenSandbox, working_dir: str, **kwargs: Any) -> OpenSandboxBackend:
    return OpenSandboxBackend(image='fake-image', connection=server.connection, working_dir=working_dir, **kwargs)


@pytest.fixture
def fast_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Advance the backend's clock a second on every read, so its two-second wait for a kill to show ends at once."""
    ticks = itertools.count()
    monkeypatch.setattr(_backend, 'time', types.SimpleNamespace(monotonic=lambda: float(next(ticks))))


async def _wait_for(path: Path) -> None:
    """Wait for a command to create `path`: the signal it reached the point a test acts on."""
    with anyio.fail_after(5):
        while not path.exists():
            await anyio.sleep(0.01)


class TestConstruction:
    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({}, 'pass exactly one of `image`'),
            ({'image': 'i', 'ref': WorkspaceRef(provider='opensandbox', id='x')}, 'pass exactly one of `image`'),
            ({'ref': WorkspaceRef(provider='docker', id='x')}, "expected 'opensandbox'"),
            ({'image': 'i', 'working_dir': 'relative'}, 'working_dir must be an absolute path'),
            ({'image': 'i', 'sandbox_timeout': 0}, 'sandbox_timeout must be an integer of at least 1'),
        ],
    )
    def test_invalid_arguments_are_refused(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            OpenSandboxBackend(**kwargs)  # pyright: ignore[reportArgumentType]

    def test_building_one_does_no_io(self, server: FakeOpenSandbox) -> None:
        backend = OpenSandboxBackend(image='fake-image', connection=server.connection)
        assert (backend.ref, server.requests) == (None, [])


class TestCommands:
    async def test_stopping_a_command_kills_its_background_children(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """A command outlives its stream and execd signals only its leader, so the backend stops the whole group."""
        backend = _create(server, working_dir)
        pids = Path(working_dir, 'pids')
        script = f'sleep 300 & echo $$ $! > {pids}; wait'
        async with anyio.create_task_group() as group:
            group.start_soon(backend.run, ['sh', '-c', script])
            await _wait_for(pids)
            group.cancel_scope.cancel()
        await wait_until_gone(*map(int, pids.read_text(encoding='utf-8').split()))

    async def test_a_background_child_holding_the_output_open_does_not_hold_up_the_result(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """execd reports the exit once the command exits, as the test server does."""
        result = await _create(server, working_dir).run(['sh', '-c', 'sleep 300 & echo started'])
        assert (result.exit_code, result.stdout) == (0, 'started\n')

    async def test_a_signal_death_reports_128_plus_the_signal(self, server: FakeOpenSandbox, working_dir: str) -> None:
        result = await _create(server, working_dir).run(['sh', '-c', f'kill -{signal.SIGTERM.value} $$'])
        assert result.exit_code == 128 + signal.SIGTERM.value

    async def test_output_past_10_mib_is_refused_and_stopped(self, server: FakeOpenSandbox, working_dir: str) -> None:
        with pytest.raises(WorkspaceOutputLimitError) as raised:
            await _create(server, working_dir).run(['sh', '-c', 'yes 0123456789abcdef'])
        assert raised.value.limit == 10 * 1024 * 1024
        assert len(raised.value.stdout) == 64 * 1024

    async def test_an_execd_older_than_1_1_1_is_refused(self, server: FakeOpenSandbox, working_dir: str) -> None:
        """execd before 1.1.1 drops every line's newline, so no output could be trusted."""
        server.strip_newlines = True
        with pytest.raises(WorkspaceUnavailableError, match=r'execd older than 1\.1\.1'):
            await _create(server, working_dir).run(['true'])

    @pytest.mark.usefixtures('fast_clock')
    async def test_a_command_that_kills_its_wrapper_cannot_run(self, server: FakeOpenSandbox, working_dir: str) -> None:
        """Only a dying sandbox normally kills the wrapper; on a live one the command's result is lost."""
        with pytest.raises(WorkspaceError, match='could not run the command'):
            await _create(server, working_dir).run(['sh', '-c', 'kill -KILL $PPID'])

    async def test_a_wrapper_that_cannot_record_its_pid_says_why(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        Path(server.pid_dir).chmod(0o500)
        try:
            with pytest.raises(WorkspaceError, match='must be writable'):
                await _create(server, working_dir).run(['true'])
        finally:
            Path(server.pid_dir).chmod(0o700)

    async def test_env_layers_per_call_over_the_backends(self, server: FakeOpenSandbox, working_dir: str) -> None:
        backend = _create(server, working_dir, env={'A': 'backend', 'B': 'backend'})
        result = await backend.run(['sh', '-c', 'echo $A $B'], env={'B': 'call'})
        assert result.stdout == 'backend call\n'


class TestLifecycle:
    async def test_the_sandbox_is_tagged_and_sized_as_asked(self, server: FakeOpenSandbox, working_dir: str) -> None:
        backend = _create(server, working_dir, resource={'cpu': '500m', 'memory': '1Gi'})
        await backend.run(['true'])
        assert backend.ref is not None
        sandbox = server.sandboxes[backend.ref.id]
        assert sandbox.resource == {'cpu': '500m', 'memory': '1Gi'}
        assert list(sandbox.metadata) == ['pydantic-ai-create']

    async def test_a_server_without_an_api_key_works(self, server: FakeOpenSandbox, working_dir: str) -> None:
        """A local server may run without authentication; the backend must not need a key."""
        server.api_key = None
        connection = ConnectionConfig(domain=DOMAIN, transport=server)
        backend = OpenSandboxBackend(image='fake-image', connection=connection, working_dir=working_dir)
        assert (await backend.run(['echo', 'ok'])).stdout == 'ok\n'

    async def test_a_rejected_api_key_is_unavailable_without_echoing_it(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        connection = ConnectionConfig(domain=DOMAIN, api_key='secret-wrong-key', transport=server)
        backend = OpenSandboxBackend(image='fake-image', connection=connection, working_dir=working_dir)
        with pytest.raises(WorkspaceUnavailableError, match=r'rejected the API key \(INVALID_API_KEY\)') as raised:
            await backend.run(['true'])
        assert 'secret-wrong-key' not in str(raised.value)

    async def test_a_server_failure_on_create_stays_transient(self, server: FakeOpenSandbox, working_dir: str) -> None:
        """A 5xx, such as an image the server could not pull, propagates for a durable engine to retry."""
        server.refuse = (500, 'DOCKER::SANDBOX_IMAGE_PULL_FAILED')
        backend = _create(server, working_dir)
        with pytest.raises(SandboxApiException):
            await backend.run(['true'])
        assert backend.ref is None

    async def test_a_lost_create_response_still_records_the_sandbox(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """On Kubernetes the server answers a create once the pod runs, which can outlast the SDK's request timeout."""
        server.lose_create_response = True
        backend = _create(server, working_dir)
        with pytest.raises(SandboxTimeoutException):
            await backend.run(['true'])
        assert backend.ref is not None
        sandbox = server.sandboxes[backend.ref.id]
        # Once the sandbox runs, the next operation attaches to it instead of creating another.
        sandbox.state = 'Running'
        server.lose_create_response = False
        assert (await backend.run(['echo', 'ok'])).stdout == 'ok\n'
        assert list(server.sandboxes) == [sandbox.id]

    async def test_a_create_cancelled_by_a_cancel_scope_still_records_the_sandbox(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        backend = _create(server, working_dir)
        with anyio.CancelScope() as scope:
            server.during_create = scope.cancel
            await backend.run(['true'])
        assert scope.cancelled_caught
        assert backend.ref is not None and list(server.sandboxes) == [backend.ref.id]

    async def test_a_create_cancelled_by_task_cancel_still_records_the_sandbox(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """A native `Task.cancel()`, as Temporal sends, gets through a same-task shield; the record must not."""
        backend = _create(server, working_dir)
        task = asyncio.ensure_future(backend.run(['true']))
        server.during_create = task.cancel
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.ref is not None and list(server.sandboxes) == [backend.ref.id]

    async def test_a_create_cancelled_by_task_cancel_whose_response_is_lost_still_records_the_sandbox(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """The cancellation is raised instead of the create's error, so the lookup cannot wait for that error."""
        server.lose_create_response = True
        backend = _create(server, working_dir)
        task = asyncio.ensure_future(backend.run(['true']))
        server.during_create = task.cancel
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.ref is not None and list(server.sandboxes) == [backend.ref.id]

    async def test_attaching_to_a_sandbox_being_deleted_is_unavailable(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """A killed sandbox still answers for a few seconds on Kubernetes, reporting `Stopping`."""
        created = _create(server, working_dir)
        await created.run(['true'])
        assert created.ref is not None
        server.stopping = True
        await server.kill(created.ref.id)
        attached = OpenSandboxBackend(ref=created.ref, connection=server.connection, working_dir=working_dir)
        with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
            await attached.run(['true'])

    async def test_a_kill_during_a_command_waits_for_the_server_to_catch_up(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """The stream breaks at once, while the server still reports the sandbox `Stopping` for a while."""
        backend = _create(server, working_dir)
        started = Path(working_dir, 'started')

        async def run_until_killed() -> None:
            with pytest.raises(WorkspaceUnavailableError, match='no longer exists'):
                await backend.run(['sh', '-c', f'touch {started}; exec sleep 300'])

        async with anyio.create_task_group() as group:
            group.start_soon(run_until_killed)
            await _wait_for(started)
            assert backend.ref is not None
            server.stopping = True
            await server.kill(backend.ref.id)

    async def test_an_unrelated_error_on_a_running_sandbox_is_not_delayed(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """Only a broken stream waits for the server to catch up; a refused request answers at once."""
        backend = _create(server, working_dir)
        await backend.run(['rm', '-rf', working_dir])
        with anyio.fail_after(1), pytest.raises(WorkspaceUnavailableError, match='refused the command'):
            await backend.run(['true'])

    async def test_a_sandbox_the_sdk_killed_after_a_failed_create_is_not_recorded(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """The SDK kills a sandbox whose create failed after the server answered; recording it would turn a
        transient failure into a backend that can only report the sandbox gone."""
        server.endpoint_error = (500, 'DOCKER::SANDBOX_QUERY_FAILED')
        server.stopping = True
        backend = _create(server, working_dir)
        with pytest.raises(SandboxApiException):
            await backend.run(['true'])
        assert backend.ref is None
        server.endpoint_error = None
        assert (await backend.run(['echo', 'ok'])).stdout == 'ok\n'


class TestWorkingDir:
    async def test_none_uses_the_images_own(self, server: FakeOpenSandbox, working_dir: str) -> None:
        server.image_working_dir = working_dir
        backend = OpenSandboxBackend(image='fake-image', connection=server.connection, working_dir=None)
        assert await backend.working_dir() == str(Path(working_dir).resolve())

    async def test_one_that_cannot_be_created_is_an_error(self, server: FakeOpenSandbox, working_dir: str) -> None:
        Path(working_dir, 'file').touch()
        backend = _create(server, f'{working_dir}/file/sub')
        with pytest.raises(WorkspaceError, match='Could not create or enter the working directory'):
            await backend.run(['true'])


class TestServerErrors:
    """A refused request is the caller's; anything else the server says propagates for a durable engine to retry."""

    async def test_attaching_with_a_rejected_key_is_unavailable(self, server: FakeOpenSandbox) -> None:
        connection = ConnectionConfig(domain=DOMAIN, api_key='wrong', transport=server)
        backend = OpenSandboxBackend(ref=WorkspaceRef(provider='opensandbox', id='x'), connection=connection)
        with pytest.raises(WorkspaceUnavailableError, match='rejected the API key'):
            await backend.run(['true'])

    async def test_a_server_failure_on_attach_destroy_or_command_propagates(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        backend = _create(server, working_dir)
        await backend.run(['true'])
        assert backend.ref is not None
        server.refuse = (503, 'UNAVAILABLE')
        with pytest.raises(SandboxApiException):
            await backend.run(['true'])
        attached = OpenSandboxBackend(ref=backend.ref, connection=server.connection, working_dir=working_dir)
        with pytest.raises(SandboxApiException):
            await attached.run(['true'])
        with pytest.raises(SandboxApiException):
            await OpenSandbox[None](connection=server.connection).destroy(backend.ref)

    @pytest.mark.usefixtures('fast_clock')
    async def test_a_broken_stream_on_a_running_sandbox_stays_transient(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        """The backend waits its two seconds for a kill to show, then lets the failure propagate."""
        backend = _create(server, working_dir)
        started = Path(working_dir, 'started')

        async def run_until_broken() -> None:
            with pytest.raises(SandboxConnectionException):
                await backend.run(['sh', '-c', f'touch {started}; exec sleep 300'])

        async with anyio.create_task_group() as group:
            group.start_soon(run_until_broken)
            await _wait_for(started)
            assert backend.ref is not None
            server.break_streams(backend.ref.id)

    async def test_a_broken_stream_from_an_unreachable_server_propagates_at_once(
        self, server: FakeOpenSandbox, working_dir: str
    ) -> None:
        backend = _create(server, working_dir)
        started = Path(working_dir, 'started')

        async def run_until_broken() -> None:
            with anyio.fail_after(1), pytest.raises(SandboxConnectionException):
                await backend.run(['sh', '-c', f'touch {started}; exec sleep 300'])

        async with anyio.create_task_group() as group:
            group.start_soon(run_until_broken)
            await _wait_for(started)
            assert backend.ref is not None
            server.unreachable = True
            server.break_streams(backend.ref.id)

    async def test_a_lost_create_whose_lookup_fails_keeps_the_original_error(
        self, server: FakeOpenSandbox, working_dir: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        server.lose_create_response = True
        server.during_create = lambda: setattr(server, 'unreachable', True)
        backend = _create(server, working_dir)
        with caplog.at_level(logging.WARNING), pytest.raises(SandboxTimeoutException):
            await backend.run(['true'])
        assert backend.ref is None
        assert 'Could not look up the OpenSandbox sandbox' in caplog.text


class TestSandboxTimeout:
    async def test_none_means_no_expiry_and_no_renewal(self, server: FakeOpenSandbox, working_dir: str) -> None:
        backend = _create(server, working_dir)
        await backend.run(['true'])
        assert backend.ref is not None
        assert (server.sandboxes[backend.ref.id].expires_at, server.renewals) == (None, [])

    async def test_attaching_renews_it(self, server: FakeOpenSandbox, working_dir: str) -> None:
        created = _create(server, working_dir, sandbox_timeout=600)
        await created.run(['true'])
        assert created.ref is not None
        attached = OpenSandboxBackend(
            ref=created.ref, connection=server.connection, working_dir=working_dir, sandbox_timeout=600
        )
        await attached.run(['true'])
        assert [sandbox_id for sandbox_id, _ in server.renewals] == [created.ref.id]

    async def test_use_renews_it_once_half_has_passed(
        self, server: FakeOpenSandbox, working_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = 0.0
        monkeypatch.setattr(_backend, 'time', types.SimpleNamespace(monotonic=lambda: now))
        backend = _create(server, working_dir, sandbox_timeout=600)
        await backend.run(['true'])
        now = 300.0
        await backend.run(['true'])
        assert server.renewals == []
        now = 301.0
        await backend.run(['true'])
        await backend.run(['true'])
        assert len(server.renewals) == 1


class TestTransport:
    async def test_get_sandbox_returns_a_handle_that_stays_open(
        self, server: FakeOpenSandbox, owned_connection: ConnectionConfig, working_dir: str
    ) -> None:
        """Even from a backend that closes after each operation, as Temporal's are, the handle stays usable."""
        capability = OpenSandbox[None](image='fake-image', connection=owned_connection, working_dir=working_dir)
        backend = capability.get_workspace(RunContext(deps=None, model=TestModel(), usage=RunUsage()), ref=None)
        assert isinstance(backend, OpenSandboxBackend)
        sandbox = await backend.get_sandbox()
        assert (await sandbox.commands.run('echo open')).logs.stdout[0].text == 'open\n'
        await backend.aclose()

    async def test_closing_before_any_operation_does_nothing(self, server: FakeOpenSandbox) -> None:
        await OpenSandboxBackend(image='fake-image', connection=server.connection).aclose()
        assert server.requests == []

    async def test_closing_closes_the_transport_and_the_next_operation_attaches_again(
        self, server: FakeOpenSandbox, owned_connection: ConnectionConfig, working_dir: str
    ) -> None:
        backend = OpenSandboxBackend(image='fake-image', connection=owned_connection, working_dir=working_dir)
        await backend.run(['true'])
        await backend.aclose()
        assert server.closed_transports == server.opened_transports > 0
        server.requests.clear()
        assert (await backend.run(['echo', 'again'])).stdout == 'again\n'
        assert any(request.endswith('/endpoints/44772') for request in server.requests)
        assert 'POST /v1/sandboxes' not in server.requests
        await backend.aclose()
        assert server.closed_transports == server.opened_transports

    async def test_closing_during_an_operation_waits_for_it(
        self, server: FakeOpenSandbox, owned_connection: ConnectionConfig, working_dir: str
    ) -> None:
        backend = OpenSandboxBackend(image='fake-image', connection=owned_connection, working_dir=working_dir)
        started, release = Path(working_dir, 'started'), Path(working_dir, 'release')
        script = f'touch {started}; while [ ! -e {release} ]; do sleep 0.01; done'
        async with anyio.create_task_group() as group:
            group.start_soon(backend.run, ['sh', '-c', script])
            await _wait_for(started)
            await backend.aclose()
            assert server.closed_transports < server.opened_transports
            release.touch()
        assert server.closed_transports == server.opened_transports


class TestDestroy:
    async def test_destroying_a_missing_sandbox_returns_quietly(self, server: FakeOpenSandbox) -> None:
        await OpenSandbox[None](connection=server.connection).destroy(
            WorkspaceRef(provider='opensandbox', id=str(uuid.uuid4()))
        )

    async def test_destroying_with_a_rejected_key_is_unavailable(self, server: FakeOpenSandbox) -> None:
        connection = ConnectionConfig(domain=DOMAIN, api_key='wrong', transport=server)
        with pytest.raises(WorkspaceUnavailableError, match='rejected the API key'):
            await OpenSandbox[None](connection=connection).destroy(
                WorkspaceRef(provider='opensandbox', id=str(uuid.uuid4()))
            )


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_request(server: FakeOpenSandbox, working_dir: str) -> None:
    """The SDK runs its calls on asyncio tasks, so Trio gets a clear `UserError` instead; closing has nothing to do."""
    backend = _create(server, working_dir)
    await backend.aclose()
    with pytest.raises(UserError, match='OpenSandbox needs the asyncio event loop'):
        await backend.run(['true'])
    with pytest.raises(UserError, match='OpenSandbox needs the asyncio event loop'):
        await OpenSandbox[None](connection=server.connection).destroy(WorkspaceRef(provider='opensandbox', id='x'))
    assert server.requests == []

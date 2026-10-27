"""Protocol and SDK-boundary tests for boxd machines."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import anyio
import boxd
import httpx
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend

from .fake_boxd import FakeBoxd, wait_until_gone

pytestmark = pytest.mark.anyio(backends=['asyncio'])

_MODULE = 'pydantic_ai_harness.boxd_sandbox._backend'
_NAME = re.compile(r'pydantic-ai-[0-9a-f]{32}')


@pytest.fixture
def host(fake_boxd: FakeBoxd, tmp_path: Path) -> FakeBoxd:
    """The fake with its machines' commands running in a temporary host directory."""
    fake_boxd.host_root = tmp_path.resolve()
    return fake_boxd


def _ref(machine_id: str) -> WorkspaceRef:
    return WorkspaceRef(provider='boxd', id=machine_id)


class TestConstruction:
    def test_building_a_backend_does_no_io(self, fake_boxd: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        assert backend.ref is None
        assert not fake_boxd.clients and not fake_boxd.machines

    @pytest.mark.parametrize(
        ('ref', 'message'),
        [
            (WorkspaceRef(provider='daytona', id='x'), "^unsupported workspace provider 'daytona'; expected 'boxd'$"),
            (WorkspaceRef(provider='boxd', id=' '), '^boxd workspace ref id cannot be empty$'),
        ],
    )
    def test_an_unusable_ref_is_refused(self, ref: WorkspaceRef, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            BoxdSandboxBackend(ref=ref)

    def test_invalid_env_is_refused(self) -> None:
        with pytest.raises(TypeError, match='env keys and values must be strings'):
            BoxdSandboxBackend(env={'PORT': 3000})  # pyright: ignore[reportArgumentType]
        with pytest.raises(ValueError, match='env keys must be non-empty without `=`'):
            BoxdSandboxBackend(env={'A=B': 'c'})

    def test_a_relative_working_dir_is_refused(self) -> None:
        with pytest.raises(ValueError, match='working_dir must be an absolute workspace path'):
            BoxdSandboxBackend(working_dir='project')

    def test_a_size_with_a_snapshot_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r'^pass either `snapshot` or `vcpu`, not both'):
            BoxdSandboxBackend(snapshot='golden', vcpu=2)


class TestCreate:
    @pytest.mark.parametrize(
        ('options', 'forwarded'),
        [
            ({'snapshot': 'python'}, {'from_snapshot': 'python', 'vcpu': None}),
            ({'vcpu': 4}, {'from_snapshot': None, 'vcpu': 4}),
        ],
        ids=['snapshot', 'vcpu'],
    )
    async def test_first_use_creates_one_isolated_machine_with_the_settings(
        self, host: FakeBoxd, caplog: pytest.LogCaptureFixture, options: dict[str, Any], forwarded: dict[str, Any]
    ) -> None:
        host.snapshots['python'] = []
        backend = BoxdSandboxBackend(org='acme', auto_suspend_timeout=0, **options)
        with caplog.at_level(logging.INFO, logger=_MODULE):
            assert await backend.run(['echo', 'hi']) == CommandResult(exit_code=0, stdout='hi\n', stderr='')
        machine = host.only
        assert _NAME.fullmatch(machine.name)
        assert machine.create_kwargs == {'org': 'acme', 'isolated': True, 'auto_suspend_timeout': 0, **forwarded}
        assert backend.ref == _ref(machine.id)
        assert caplog.messages == [f'Created boxd machine {machine.id} ({machine.name})']

    async def test_a_lost_create_reply_finds_the_machine_instead_of_creating_another(self, host: FakeBoxd) -> None:
        host.lose_create_reply = True
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        assert len(host.create_names) == 1
        assert backend.ref == _ref(host.only.id)

    async def test_a_retry_after_a_failed_create_reuses_the_name(self, host: FakeBoxd) -> None:
        host.create_errors.append(boxd.APIConnectionError('the connection dropped'))
        backend = BoxdSandboxBackend()
        with pytest.raises(boxd.APIConnectionError, match='the connection dropped'):
            await backend.run(['true'])
        assert backend.ref is None
        await backend.run(['true'])
        first, second = host.create_names
        assert first == second == host.only.name

    async def test_a_retry_finds_the_machine_a_lost_reply_left_under_its_name(self, host: FakeBoxd) -> None:
        host.lose_create_reply = True
        host.get_errors.append(boxd.APIConnectionError('the control plane is unreachable'))
        backend = BoxdSandboxBackend()
        with pytest.raises(boxd.APIConnectionError, match='the connection dropped before the reply arrived'):
            await backend.run(['true'])
        await backend.run(['true'])
        first, second = host.create_names
        assert first == second == host.only.name
        assert backend.ref == _ref(host.only.id)

    async def test_a_refused_create_whose_machine_never_appeared_is_unavailable(self, host: FakeBoxd) -> None:
        host.create_errors.append(boxd.ConflictError('a payment method is required', code='failed_precondition'))
        with pytest.raises(WorkspaceUnavailableError, match='boxd refused to create the machine: a payment method'):
            await BoxdSandboxBackend().run(['true'])
        assert not host.machines

    async def test_an_unknown_snapshot_is_unavailable(self, host: FakeBoxd) -> None:
        with pytest.raises(WorkspaceUnavailableError, match='boxd could not create the machine: snapshot not found'):
            await BoxdSandboxBackend(snapshot='missing').run(['true'])

    async def test_a_refused_setting_is_unavailable(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend(vcpu=3)  # pyright: ignore[reportArgumentType]
        with pytest.raises(WorkspaceUnavailableError, match='no machine size has 3 vCPU'):
            await backend.run(['true'])

    async def test_rejected_credentials_are_unavailable(self, host: FakeBoxd) -> None:
        host.reject_credentials = True
        with pytest.raises(WorkspaceUnavailableError, match='boxd did not accept the credentials') as error:
            await BoxdSandboxBackend().run(['true'])
        assert 'invalid or expired' not in str(error.value)

    @pytest.mark.parametrize(
        'error',
        [
            boxd.AuthenticationError('could not reach https://app.boxd.sh/api/v1/auth/token: timed out'),
            boxd.AuthenticationError('exchange failed: 503 upstream unavailable'),
        ],
        ids=['unreachable', 'server-error'],
    )
    async def test_a_token_exchange_that_could_not_ask_boxd_propagates(
        self, host: FakeBoxd, error: boxd.AuthenticationError
    ) -> None:
        if 'could not reach' in str(error):
            error.__cause__ = httpx.ConnectTimeout('timed out')
        host.create_errors.append(error)
        with pytest.raises(boxd.AuthenticationError) as raised:
            await BoxdSandboxBackend().run(['true'])
        assert raised.value is error

    async def test_a_server_error_propagates_so_durable_engines_retry_it(self, host: FakeBoxd) -> None:
        host.create_errors.append(boxd.APIStatusError('internal error', code='internal'))
        backend = BoxdSandboxBackend()
        with pytest.raises(boxd.APIStatusError, match='internal error'):
            await backend.run(['true'])
        assert backend.ref is None

    async def test_a_stalled_create_times_out(self, host: FakeBoxd, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(f'{_MODULE}._LIFECYCLE_TIMEOUT', 0.05)
        host.create_gate = asyncio.Event()
        with pytest.raises(TimeoutError, match=re.escape('boxd did not create the machine within 0.05s')):
            await BoxdSandboxBackend().run(['true'])

    async def test_a_create_that_answers_too_late_is_found_by_name(
        self, host: FakeBoxd, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._LIFECYCLE_TIMEOUT', 0.05)
        host.create_gate = asyncio.Event()
        host.create_then_hang = True
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        assert backend.ref == _ref(host.only.id)

    async def test_a_lost_reply_is_looked_up_among_the_orgs_machines(self, host: FakeBoxd) -> None:
        host.add()
        host.lose_create_reply = True
        backend = BoxdSandboxBackend(org='acme')
        await backend.run(['true'])
        assert host.listed_orgs == ['acme']
        assert backend.ref == _ref('machine-2')

    async def test_a_failed_create_in_an_org_is_not_mistaken_for_another_machine(self, host: FakeBoxd) -> None:
        host.add()
        host.create_errors.append(boxd.APIConnectionError('the connection dropped'))
        with pytest.raises(boxd.APIConnectionError, match='the connection dropped'):
            await BoxdSandboxBackend(org='acme').run(['true'])
        assert host.listed_orgs == ['acme']

    async def test_a_snapshot_whose_machine_is_on_a_network_is_refused(self, host: FakeBoxd) -> None:
        host.snapshots['networked'] = ['databases']
        backend = BoxdSandboxBackend(snapshot='networked')
        with pytest.raises(WorkspaceUnavailableError, match='is on the boxd networks databases'):
            await backend.run(['true'])
        # Recorded, so its holder can delete it: it is isolated, and the backend created it.
        assert backend.ref is not None
        await BoxdSandbox().destroy(backend.ref)
        assert host.only.status == 'destroyed'
        assert not host.lines

    async def test_a_new_machine_gets_its_working_dir(self, host: FakeBoxd, tmp_path: Path) -> None:
        directory = tmp_path.resolve() / 'project' / 'nested'
        backend = BoxdSandboxBackend(working_dir=str(directory))
        assert await backend.working_dir() == str(directory)
        assert (await backend.run(['pwd'])).stdout == f'{directory}\n'
        assert host.exec_commands.count(f'mkdir -p -- {directory}') == 1

    async def test_a_working_dir_that_cannot_be_made_fails_every_operation(
        self, host: FakeBoxd, tmp_path: Path
    ) -> None:
        (tmp_path / 'file').write_text('')
        backend = BoxdSandboxBackend(working_dir=str(tmp_path.resolve() / 'file' / 'project'))
        for _ in range(2):
            with pytest.raises(WorkspaceError, match=r"Could not create working_dir '.*' in boxd machine machine-1: "):
                await backend.run(['true'])
        assert len(host.machines) == 1

    async def test_cancelling_the_creating_operation_records_the_machine_and_finishes_setup_later(
        self, host: FakeBoxd, tmp_path: Path
    ) -> None:
        directory = tmp_path.resolve() / 'project'
        backend = BoxdSandboxBackend(working_dir=str(directory))
        host.create_gate = asyncio.Event()
        async with anyio.create_task_group() as group:
            group.start_soon(backend.working_dir)
            while not host.create_names:
                await anyio.sleep(0)
            group.cancel_scope.cancel()
            host.create_gate.set()
        assert backend.ref == _ref(host.only.id)
        assert not directory.exists()
        assert await backend.working_dir() == str(directory)
        assert len(host.create_names) == 1


class TestAttach:
    async def test_attaching_works_in_the_existing_machine(self, host: FakeBoxd) -> None:
        machine = host.add()
        backend = BoxdSandboxBackend(ref=_ref(machine.id))
        await backend.run(['true'])
        assert backend.ref == _ref(machine.id)
        assert not host.create_names

    async def test_attaching_reaches_the_named_machine_among_others(self, host: FakeBoxd) -> None:
        host.add()
        machine = host.add(status='suspended')
        await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])
        assert [machine.status for machine in host.machines.values()] == ['running', 'running']

    async def test_a_ref_that_names_the_machine_is_recorded_by_id(self, host: FakeBoxd) -> None:
        machine = host.add(name='named')
        backend = BoxdSandboxBackend(ref=_ref('named'))
        await backend.run(['true'])
        assert backend.ref == _ref(machine.id)

    async def test_a_machine_on_a_network_is_refused(self, host: FakeBoxd) -> None:
        machine = host.add(name='peer', networks=['databases', 'cache'])
        with pytest.raises(WorkspaceUnavailableError, match=r'\(peer\) is on the boxd networks databases, cache'):
            await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])
        assert not host.exec_commands

    async def test_a_token_exchange_failure_while_waiting_is_retried(self, host: FakeBoxd) -> None:
        machine = host.add()
        error = boxd.AuthenticationError('could not reach https://app.boxd.sh/api/v1/auth/token: reset')
        error.__cause__ = httpx.ReadError('reset')
        host.probe_errors.append(error)
        await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])
        assert host.exec_commands == ['true', 'true']

    async def test_a_stopped_machine_is_started(self, host: FakeBoxd) -> None:
        machine = host.add(status='stopped')
        host.boot_polls = 2
        host.start_errors.append(boxd.ConflictError('the machine is already starting', code='failed_precondition'))
        await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])
        assert (machine.status, machine.starts) == ('running', 2)

    @pytest.mark.parametrize('status', ['suspended', 'hibernated'])
    async def test_a_sleeping_machine_is_woken_by_its_first_command(self, host: FakeBoxd, status: str) -> None:
        machine = host.add(status=status)
        await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])
        assert (machine.status, machine.starts) == ('running', 0)

    async def test_a_booting_machine_is_waited_for(self, host: FakeBoxd) -> None:
        machine = host.add(boot_polls=3)
        host.probe_errors.append(boxd.APIConnectionError('the machine agent is not up yet'))
        await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])
        assert host.exec_commands == ['true', 'true']

    async def test_rejected_credentials_while_waiting_are_unavailable(self, host: FakeBoxd) -> None:
        machine = host.add()
        host.probe_errors.append(boxd.PermissionDeniedError('not a member of this organization'))
        with pytest.raises(WorkspaceUnavailableError, match='boxd did not accept the credentials'):
            await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])

    async def test_a_missing_machine_is_unavailable_and_not_replaced(self, host: FakeBoxd) -> None:
        with pytest.raises(WorkspaceUnavailableError, match="The boxd machine 'machine-404' was not found"):
            await BoxdSandboxBackend(ref=_ref('machine-404')).run(['true'])
        assert not host.create_names

    async def test_a_machine_that_is_not_isolated_is_refused(self, host: FakeBoxd) -> None:
        machine = host.add(name='dev-box', isolated=False)
        with pytest.raises(WorkspaceUnavailableError, match=r"'machine-1' \(dev-box\) is not isolated"):
            await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])
        assert not host.lines and not host.exec_commands

    @pytest.mark.parametrize('status', ['failed', 'destroyed'])
    async def test_a_dead_machine_is_unavailable(self, host: FakeBoxd, status: str) -> None:
        host.report_destroyed = True
        machine = host.add(status=status)
        with pytest.raises(WorkspaceUnavailableError, match="'machine-1' no longer exists"):
            await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])

    async def test_a_machine_that_never_becomes_ready_times_out(
        self, host: FakeBoxd, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._LIFECYCLE_TIMEOUT', 0.05)
        machine = host.add(boot_polls=10**9)
        with pytest.raises(TimeoutError, match=re.escape("boxd machine 'machine-1' did not become ready within 0.05s")):
            await BoxdSandboxBackend(ref=_ref(machine.id)).run(['true'])

    async def test_an_attached_machine_must_already_have_the_working_dir(self, host: FakeBoxd, tmp_path: Path) -> None:
        machine = host.add()
        missing = tmp_path.resolve() / 'missing'
        backend = BoxdSandboxBackend(ref=_ref(machine.id), working_dir=str(missing))
        with pytest.raises(WorkspaceError, match=f"working_dir '{missing}' does not exist in boxd machine machine-1"):
            await backend.run(['true'])
        assert not missing.exists()


class TestRun:
    async def test_env_is_layered_over_the_backend_env(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend(env={'A': 'backend', 'B': "kept $HOME 'quoted'\n"})
        result = await backend.run(['sh', '-c', 'printf "%s|%s" "$A" "$B"'], env={'A': 'command'})
        assert result.stdout == "command|kept $HOME 'quoted'\n"

    async def test_a_command_killed_by_a_signal_exits_128_plus_its_number(self, host: FakeBoxd) -> None:
        assert (await BoxdSandboxBackend().run(['sh', '-c', 'kill -9 $$'])).exit_code == 137

    async def test_a_program_named_like_a_shell_builtin_is_looked_up_on_path(self, host: FakeBoxd) -> None:
        assert (await BoxdSandboxBackend().run(['cd', '/'])).exit_code == 127

    async def test_output_over_the_limit_stops_the_command(
        self, host: FakeBoxd, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._MAX_OUTPUT_BYTES', 1000)
        monkeypatch.setattr(f'{_MODULE}._OUTPUT_PREVIEW_BYTES', 6)
        backend = BoxdSandboxBackend()
        with pytest.raises(WorkspaceOutputLimitError, match='exceeded the 10 MiB limit') as error:
            await backend.run(['sh', '-c', 'echo $$ > pid; printf warn >&2; sleep 0.2; exec yes'])
        assert (error.value.limit, error.value.stdout, error.value.stderr) == (1000, 'y\ny\ny\n', 'warn')
        await wait_until_gone(tmp_path / 'pid')

    async def test_a_timeout_stops_the_command_and_keeps_its_partial_output(
        self, host: FakeBoxd, tmp_path: Path
    ) -> None:
        backend = BoxdSandboxBackend()
        with pytest.raises(WorkspaceTimeoutError, match=re.escape('Command timed out after 0.5 seconds')) as error:
            await backend.run(['sh', '-c', 'echo $$ > pid; printf partial; printf err >&2; exec sleep 60'], timeout=0.5)
        assert (error.value.stdout, error.value.stderr) == ('partial', 'err')
        await wait_until_gone(tmp_path / 'pid')

    async def test_a_machine_stopped_during_a_command_is_unavailable_until_attached_again(
        self, host: FakeBoxd, tmp_path: Path
    ) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        machine = host.only

        async def stop_once_started() -> None:
            while not (tmp_path / 'started').exists():
                await anyio.sleep(0.01)
            machine.status = 'stopped'
            for process in machine.processes:
                if process.returncode is None:
                    process.kill()

        async with anyio.create_task_group() as group:
            group.start_soon(stop_once_started)
            with pytest.raises(WorkspaceUnavailableError, match="'machine-1' was stopped; a later run that attaches"):
                await backend.run(['sh', '-c', 'touch started; exec sleep 60'])
        with pytest.raises(WorkspaceUnavailableError, match='was stopped'):
            await backend.run(['true'])
        assert (await BoxdSandboxBackend(ref=backend.ref).run(['echo', 'back'])).stdout == 'back\n'
        assert machine.starts == 1

    async def test_a_deleted_machine_is_unavailable_while_boxd_still_reports_it(self, host: FakeBoxd) -> None:
        host.report_destroyed = True
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        host.only.status = 'destroyed'
        with pytest.raises(WorkspaceUnavailableError, match="'machine-1' no longer exists"):
            await backend.run(['true'])

    async def test_a_command_boxd_ends_for_silence_is_a_timeout_with_its_output(self, host: FakeBoxd) -> None:
        host.stream_error = boxd.APIStatusError(
            'no output from the command for 30m, the session looks wedged', code='deadline_exceeded'
        )
        with pytest.raises(WorkspaceTimeoutError, match='after 30 minutes without output') as error:
            await BoxdSandboxBackend().run(['sh', '-c', 'printf started; printf warn >&2'])
        assert (error.value.stdout, error.value.stderr) == ('started', 'warn')

    async def test_a_connection_failure_on_a_live_machine_propagates(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        host.stream_error = boxd.APIConnectionError('the connection dropped')
        with pytest.raises(boxd.APIConnectionError, match='the connection dropped'):
            await backend.run(['true'])

    async def test_an_inconclusive_check_keeps_the_original_failure(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        host.stream_error = boxd.APIConnectionError('the connection dropped')
        host.get_errors.append(boxd.APIConnectionError('the control plane is unreachable'))
        with pytest.raises(boxd.APIConnectionError, match='the connection dropped'):
            await backend.run(['true'])

    async def test_a_refusal_on_a_live_machine_is_a_workspace_error(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        host.stream_error = boxd.APIStatusError('the command line is too long', code='invalid_argument')
        with pytest.raises(WorkspaceError, match=r'^Could not run the command: the command line is too long$'):
            await backend.run(['true'])

    async def test_rejected_credentials_during_a_command_are_unavailable(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        host.reject_credentials = True
        with pytest.raises(WorkspaceUnavailableError, match='boxd did not accept the credentials'):
            await backend.run(['true'])

    async def test_a_command_without_an_exit_status_is_a_workspace_error(self, host: FakeBoxd) -> None:
        host.drop_exit_status = True
        with pytest.raises(WorkspaceError, match='boxd ended the command in machine machine-1 without an exit status'):
            await BoxdSandboxBackend().run(['true'])

    async def test_the_working_dir_is_resolved_once(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        assert await backend.working_dir() == str(host.host_root)
        assert await backend.working_dir() == str(host.host_root)
        assert sum('pwd -P' in line for line in host.lines) == 1

    async def test_a_working_dir_that_cannot_be_resolved_is_a_workspace_error(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend(env={'PATH': '/nonexistent'})
        with pytest.raises(WorkspaceError, match=r'`pwd -P` exited 127'):
            await backend.working_dir()


class TestGetMachine:
    async def test_returns_the_current_record(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        machine = await backend.get_machine()
        assert backend.ref == _ref(machine.id)
        assert (machine.status, machine.networking.isolated) == ('running', True)

    async def test_maps_a_failed_lookup(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.get_machine()
        host.get_errors.append(boxd.AuthenticationError('the key was revoked'))
        with pytest.raises(WorkspaceUnavailableError, match='boxd did not accept the credentials'):
            await backend.get_machine()


class TestClientLifecycle:
    async def test_aclose_closes_the_client_the_backend_opened(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.aclose()
        await backend.run(['true'])
        await backend.aclose()
        assert [client.closed for client in host.clients] == [True]

    async def test_a_passed_client_is_left_open(self, host: FakeBoxd) -> None:
        client = boxd.AsyncBoxd()
        backend = BoxdSandboxBackend(client=client)
        await backend.run(['true'])
        await backend.aclose()
        assert [client.closed for client in host.clients] == [False]

    async def test_an_operation_after_aclose_reattaches_with_a_new_client(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        await backend.aclose()
        await backend.run(['true'])
        assert [client.closed for client in host.clients] == [True, False]
        assert len(host.create_names) == 1

    async def test_aclose_during_an_operation_closes_the_client_when_it_ends(
        self, host: FakeBoxd, tmp_path: Path
    ) -> None:
        backend = BoxdSandboxBackend()
        results: list[CommandResult] = []

        async def command() -> None:
            results.append(await backend.run(['sh', '-c', 'touch started; sleep 0.2; printf done']))

        async with anyio.create_task_group() as group:
            group.start_soon(command)
            while not (tmp_path / 'started').exists():
                await anyio.sleep(0.01)
            await backend.aclose()
            assert not host.clients[0].closed
        assert results[0].stdout == 'done'
        assert host.clients[0].closed

    async def test_concurrent_acloses_close_the_client_once(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        await asyncio.gather(backend.aclose(), backend.aclose())
        assert [client.closes for client in host.clients] == [1]

    @pytest.mark.parametrize('failure', ['error', 'hang'])
    async def test_a_failing_close_is_logged_not_raised(
        self, host: FakeBoxd, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._TEARDOWN_TIMEOUT', 0.05)
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        if failure == 'error':
            host.close_error = RuntimeError('the channel is already closed')
        else:
            host.close_gate = asyncio.Event()
        await backend.aclose()
        message = (
            "Could not close the boxd API client: RuntimeError('the channel is already closed')"
            if failure == 'error'
            else 'Closing the boxd API client did not complete within 0.05s.'
        )
        assert caplog.messages == [message]


class TestDestroy:
    async def test_destroy_deletes_and_waits_until_the_machine_is_gone(self, host: FakeBoxd) -> None:
        host.delete_lag = 3
        machine = host.add()
        await BoxdSandbox().destroy(_ref(machine.id))
        assert machine.status == 'destroyed' and machine.delete_lag == 0
        assert [client.closed for client in host.clients] == [True]

    async def test_destroy_accepts_a_destroyed_record_as_gone(self, host: FakeBoxd) -> None:
        host.report_destroyed = True
        machine = host.add()
        await BoxdSandbox().destroy(_ref(machine.id))
        assert machine.status == 'destroyed'

    async def test_destroying_a_missing_machine_returns_quietly(self, host: FakeBoxd) -> None:
        await BoxdSandbox().destroy(_ref('machine-404'))

    async def test_destroy_leaves_a_machine_that_is_not_isolated_alone(self, host: FakeBoxd) -> None:
        machine = host.add(name='dev-box', isolated=False)
        with pytest.raises(
            ValueError, match=r"'machine-1' \(dev-box\) is not isolated, so BoxdSandbox does not delete"
        ):
            await BoxdSandbox().destroy(_ref(machine.id))
        assert machine.status == 'running'

    async def test_destroy_with_rejected_credentials_is_unavailable(self, host: FakeBoxd) -> None:
        host.reject_credentials = True
        with pytest.raises(WorkspaceUnavailableError, match='boxd did not accept the credentials'):
            await BoxdSandbox().destroy(_ref('machine-1'))

    async def test_destroy_lets_a_transient_failure_propagate(self, host: FakeBoxd) -> None:
        machine = host.add()
        host.get_errors.append(boxd.AuthenticationError('exchange failed: 502 bad gateway'))
        with pytest.raises(boxd.AuthenticationError, match='502'):
            await BoxdSandbox().destroy(_ref(machine.id))
        assert machine.status == 'running'

    async def test_destroy_times_out_when_boxd_keeps_reporting_the_machine(
        self, host: FakeBoxd, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(f'{_MODULE}._LIFECYCLE_TIMEOUT', 0.05)
        host.delete_lag = 10**9
        machine = host.add()
        with pytest.raises(TimeoutError):
            await BoxdSandbox().destroy(_ref(machine.id))

    async def test_destroy_with_a_passed_client_leaves_it_open(self, host: FakeBoxd) -> None:
        machine = host.add()
        await BoxdSandbox[None](client=boxd.AsyncBoxd()).destroy(_ref(machine.id))
        assert [client.closed for client in host.clients] == [False]

    async def test_destroy_refuses_a_foreign_or_blank_ref(self, fake_boxd: FakeBoxd) -> None:
        with pytest.raises(ValueError, match="expected 'boxd'"):
            await BoxdSandbox().destroy(WorkspaceRef(provider='daytona', id='x'))
        with pytest.raises(ValueError, match='cannot be empty'):
            await BoxdSandbox().destroy(_ref(''))
        assert not fake_boxd.clients


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_boxd_call(fake_boxd: FakeBoxd) -> None:
    """The SDK runs its calls on gRPC asyncio channels, so Trio gets a clear `UserError` instead."""
    message = 'boxd needs the asyncio event loop'
    backend = BoxdSandboxBackend()
    with pytest.raises(UserError, match=message):
        await backend.run(['true'])
    with pytest.raises(UserError, match=message):
        await backend.get_machine()
    with pytest.raises(UserError, match=message):
        await BoxdSandbox().destroy(_ref('machine-1'))
    await backend.aclose()
    assert not fake_boxd.clients


class TestFork:
    async def test_fork_copies_the_machine_into_a_new_isolated_one(
        self, host: FakeBoxd, caplog: pytest.LogCaptureFixture
    ) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        assert backend.ref is not None
        with caplog.at_level(logging.INFO, logger=_MODULE):
            ref = await backend.fork()
        source, copy = host.machines[backend.ref.id], host.machines[ref.id]
        assert _NAME.fullmatch(copy.name) and copy.name != source.name
        assert (copy.isolated, copy.networks, copy.forked_from) == (True, [], (source.id, 'running'))
        assert backend.ref == _ref(source.id)
        assert caplog.messages == [f'Forked boxd machine {source.id} into {copy.id} ({copy.name})']

    async def test_a_suspended_machine_is_woken_before_it_is_forked(self, host: FakeBoxd) -> None:
        """boxd forks a suspended machine from its disk alone, without its memory or unflushed writes."""
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        assert backend.ref is not None
        host.machines[backend.ref.id].status = 'suspended'
        ref = await backend.fork()
        assert host.machines[ref.id].forked_from == (backend.ref.id, 'running')

    async def test_a_machine_put_on_a_network_after_attaching_is_not_forked(self, host: FakeBoxd) -> None:
        backend = BoxdSandboxBackend()
        await backend.run(['true'])
        assert backend.ref is not None
        host.machines[backend.ref.id].networks = ['team']
        with pytest.raises(WorkspaceUnavailableError, match='is on the boxd networks team'):
            await backend.fork()
        assert len(host.machines) == 1

    @pytest.mark.parametrize('org', [None, 'acme'])
    async def test_a_lost_fork_reply_finds_the_copy_in_the_source_organization(
        self, host: FakeBoxd, org: str | None
    ) -> None:
        source = host.add(org=org)
        host.lose_fork_reply = True
        ref = await BoxdSandboxBackend(ref=_ref(source.id)).fork()
        (copy,) = [machine for machine in host.machines.values() if machine.forked_from is not None]
        assert (ref, copy.org) == (_ref(copy.id), org)
        assert host.listed_orgs == ([] if org is None else [org])

    async def test_a_source_deleted_before_the_fork_is_unavailable(self, host: FakeBoxd) -> None:
        source = host.add()
        host.fork_errors.append(boxd.NotFoundError(f'source machine {source.id!r} not found', code='not_found'))
        with pytest.raises(
            WorkspaceUnavailableError, match="boxd could not fork the machine: source machine 'machine-1'"
        ):
            await BoxdSandboxBackend(ref=_ref(source.id)).fork()

    async def test_a_dropped_connection_with_no_copy_propagates(self, host: FakeBoxd) -> None:
        source = host.add()
        host.fork_errors.append(boxd.APIConnectionError('the connection dropped'))
        with pytest.raises(boxd.APIConnectionError, match='the connection dropped'):
            await BoxdSandboxBackend(ref=_ref(source.id)).fork()
        assert len(host.machines) == 1

"""Substantive tests for the Modal workspace backend."""

from __future__ import annotations

import asyncio
import importlib.abc
import importlib.machinery
import importlib.util
import logging
import subprocess
import sys
import threading
import time
import types
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    Workspace,
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxBackend, _backend

from .fake_modal import FakeImage, FakeModal, FileInfo


async def started(**settings: Any) -> ModalSandboxBackend:
    backend = ModalSandboxBackend(**settings)
    await backend.get_sandbox()
    return backend


async def test_destroy_ref_does_not_attach_or_create(fake_modal: FakeModal) -> None:
    owner = await started()
    ref = owner.ref
    assert ref is not None
    provider = ModalSandbox()
    attached = provider.backend(ref)
    assert attached.ref == ref
    assert fake_modal.attach_ids == []
    await provider.destroy(ref)
    assert fake_modal.attach_ids == [ref.id]
    assert fake_modal.owned_creates == 1
    assert fake_modal.sandboxes[0].shutting_down


async def test_create_logs_the_new_sandbox_id(fake_modal: FakeModal, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=_backend.__name__):
        owner = await started()
        assert owner.ref is not None
        # Attaching to an existing sandbox creates nothing, so it logs nothing.
        await ModalSandboxBackend(ref=owner.ref).get_sandbox()
    assert [record.getMessage() for record in caplog.records] == [f'Created Modal sandbox {owner.ref.id}']


async def test_auth_failure_classifies_reason_without_echoing_secret(fake_modal: FakeModal) -> None:
    secret = 'modal-secret-value-123'
    fake_modal.create_error = fake_modal.exception('AuthError')(f'token {secret} expired')
    with pytest.raises(WorkspaceUnavailableError, match='Credential expired') as exc:
        await ModalSandboxBackend().get_sandbox()
    assert secret not in str(exc.value)
    assert 'MODAL_TOKEN_ID' in str(exc.value)


async def test_missing_credentials_are_reported_as_not_found(fake_modal: FakeModal) -> None:
    """With no token configured the SDK raises before sending anything, so nothing was rejected."""
    missing = fake_modal.exception('AuthError')('Token missing. Could not authenticate client.')
    fake_modal.create_error = missing
    with pytest.raises(WorkspaceUnavailableError) as exc:
        await ModalSandboxBackend().get_sandbox()
    assert str(exc.value) == (
        'No Modal credentials found. Set MODAL_TOKEN_ID / MODAL_TOKEN_SECRET or run `modal token new`.'
    )
    fake_modal.attach_error = missing
    with pytest.raises(WorkspaceUnavailableError, match=r'^No Modal credentials found'):
        await ModalSandbox().destroy(WorkspaceRef(provider='modal', id='sb-owned'))


async def test_destroy_imports_modal_off_the_event_loop(fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch) -> None:
    # Importing Modal reads `~/.modal.toml`, so a cold import must not block the loop.
    import_threads: list[int] = []

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
            import_threads.append(threading.get_ident())
            return fake_modal.module

        def exec_module(self, module: types.ModuleType) -> None:
            pass

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name: str, path: Any, target: Any = None) -> importlib.machinery.ModuleSpec | None:
            return importlib.util.spec_from_loader(name, _Loader()) if name == 'modal' else None

    monkeypatch.delitem(sys.modules, 'modal')
    monkeypatch.setattr(sys, 'meta_path', [_Finder(), *sys.meta_path])
    await ModalSandbox().destroy(WorkspaceRef(provider='modal', id='sb-gone'))
    assert import_threads and import_threads[0] != threading.get_ident()


async def test_destroy_rejects_foreign_ref_without_sdk_call(fake_modal: FakeModal) -> None:
    with pytest.raises(ValueError, match='unsupported workspace provider'):
        await ModalSandbox().destroy(WorkspaceRef(provider='other', id='sb-owned'))
    assert fake_modal.attach_ids == []


async def test_destroy_is_idempotent_for_a_sandbox_that_no_longer_exists(fake_modal: FakeModal) -> None:
    fake_modal.attach_error = fake_modal.exception('NotFoundError')('sandbox not found')
    await ModalSandbox().destroy(WorkspaceRef(provider='modal', id='sb-gone'))
    assert fake_modal.attach_ids == ['sb-gone']


async def test_destroy_twice_returns_quietly(fake_modal: FakeModal) -> None:
    ref = (await started()).ref
    assert ref is not None
    provider = ModalSandbox()
    await provider.destroy(ref)
    fake_modal.attach_error = fake_modal.exception('NotFoundError')('sandbox not found')
    await provider.destroy(ref)


async def test_destroy_names_a_malformed_id(fake_modal: FakeModal) -> None:
    """Modal answers a malformed ID with `InvalidError`; the error names the ID instead of leaking the SDK's."""
    error = fake_modal.exception('InvalidError')('"sb-bad" is not a valid sandbox ID')
    fake_modal.attach_error = error
    with pytest.raises(WorkspaceUnavailableError) as exc_info:
        await ModalSandbox().destroy(WorkspaceRef(provider='modal', id='sb-bad'))
    assert str(exc_info.value) == (
        "Modal does not recognize 'sb-bad' as a sandbox ID, so there is no sandbox to terminate. "
        'Pass the ref of a sandbox this provider created or attached to.'
    )
    assert exc_info.value.__cause__ is error


async def test_destroy_keeps_a_conflict_unchanged(fake_modal: FakeModal) -> None:
    """`ConflictError` subclasses `InvalidError` but is not a malformed ID, so it propagates as is."""
    fake_modal.attach_error = fake_modal.exception('ConflictError')('busy')
    with pytest.raises(fake_modal.exception('ConflictError')):
        await ModalSandbox().destroy(WorkspaceRef(provider='modal', id='sb-owned'))


@pytest.mark.parametrize(('name', 'expected'), [('AuthError', WorkspaceUnavailableError), ('ConnectionError', None)])
async def test_destroy_maps_rejected_credentials_and_keeps_other_failures(
    fake_modal: FakeModal, name: str, expected: type[Exception] | None
) -> None:
    fake_modal.attach_error = fake_modal.exception(name)('failed')
    with pytest.raises(expected or fake_modal.exception(name)) as exc_info:
        await ModalSandbox().destroy(WorkspaceRef(provider='modal', id='sb-owned'))
    assert fake_modal.attach_error in (exc_info.value, exc_info.value.__cause__)


class TestRun:
    async def test_argv_is_execed_by_sh(self, fake_modal: FakeModal) -> None:
        """`sh` execs the program, so one that can't start exits 127 or 126 as it does in `sh`."""
        backend = await started()
        await backend.run(['echo', 'hi'])
        assert fake_modal.sandboxes[0].exec_calls[-1].argv == ['/bin/sh', '-c', 'exec "$@"', 'sh', 'echo', 'hi']

    async def test_shell_wraps_in_sh(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.run('echo hi | wc -c', shell=True)
        assert fake_modal.sandboxes[0].exec_calls[-1].argv == ['/bin/sh', '-c', 'echo hi | wc -c']

    async def test_env_reaches_the_command(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.run(['env'], env={'FOO': 'bar'})
        assert fake_modal.sandboxes[0].exec_calls[-1].env == {'FOO': 'bar'}

    async def test_working_dir_and_env_apply_to_every_command_of_an_attached_sandbox(
        self, fake_modal: FakeModal
    ) -> None:
        # Given per command rather than only at creation, so a sandbox attached by reference
        # honors them too; a command's own `env` takes precedence.
        backend = await started(
            ref=WorkspaceRef(provider='modal', id='sb-keep'), working_dir='/work', env={'A': '1', 'B': '1'}
        )
        await backend.run(['env'])
        await backend.run(['env'], env={'B': '2'})
        calls = [call for call in fake_modal.sandboxes[0].exec_calls if 'command -v setsid' not in ' '.join(call.argv)]
        assert [(call.workdir, call.env) for call in calls] == [
            ('/work', {'A': '1', 'B': '1'}),
            ('/work', {'A': '1', 'B': '2'}),
        ]

    async def test_fractional_timeout_rounds_up_to_a_modal_deadline(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Modal takes whole seconds and reads 0 as "no timeout", so a sub-second deadline
        # must not floor to unbounded.
        @asynccontextmanager
        async def no_client_deadline(*_args: Any, **_kwargs: Any) -> AsyncGenerator[None]:
            yield

        monkeypatch.setattr(_backend, 'command_deadline', no_client_deadline)
        backend = await started()
        await backend.run(['x'], timeout=0.5)
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 1

    async def test_timeout_none_stays_unbounded(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.run(['x'])
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout is None

    @pytest.mark.parametrize('timeout', [0, -1.0, float('inf'), float('nan')])
    async def test_invalid_timeout_rejected(self, fake_modal: FakeModal, timeout: float) -> None:
        backend = await started()
        with pytest.raises(ValueError, match='timeout must be a positive finite number'):
            await backend.run(['x'], timeout=timeout)

    async def test_client_deadline_sentinel_raises_a_timeout(self, fake_modal: FakeModal) -> None:
        # Modal's -1 is its client-side deadline sentinel; the protocol says a deadline
        # raises, and the output produced before the kill rides on the exception.
        fake_modal.responder = lambda argv, timeout: ('partial', 'oops', -1)
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError) as exc:
            await backend.run(['sleep', '99'], timeout=5)
        assert isinstance(exc.value, TimeoutError)
        assert (exc.value.stdout, exc.value.stderr) == ('partial', 'oops')

    async def test_client_deadline_while_sandbox_shuts_down_is_unavailable(self, fake_modal: FakeModal) -> None:
        backend = await started()
        sandbox = fake_modal.sandboxes[0]

        def deadline_during_shutdown(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            sandbox.shutting_down = True
            return '', '', -1

        fake_modal.responder = deadline_during_shutdown
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await backend.run(['sleep', '99'], timeout=5)

    async def test_a_timeout_message_quotes_the_callers_timeout(self, fake_modal: FakeModal) -> None:
        # Modal enforces whole seconds, so 0.5 runs as a 1-second deadline; the message still
        # names the timeout the caller asked for.
        fake_modal.responder = lambda argv, timeout: ('', '', -1)
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError, match=r'^Command timed out after 0\.5 seconds$'):
            await backend.run(['sleep', '99'], timeout=0.5)
        assert fake_modal.sandboxes[0].exec_calls[0].timeout == 1

    async def test_sentinel_without_a_deadline_is_a_real_exit(self, fake_modal: FakeModal) -> None:
        # -1 is only the timeout sentinel when we set a deadline; from another cause it is
        # the honest exit code.
        fake_modal.responder = lambda argv, timeout: ('', '', -1)
        backend = await started()
        assert (await backend.run(['x'])).exit_code == -1

    async def test_server_side_deadline_kill_is_a_timeout(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The server enforces the deadline before the client's own clock fires, so its
        # SIGKILL (exit 137) can beat Modal's -1 sentinel; a 137 that consumed the whole
        # deadline window is a timeout, not a mysterious ordinary exit.
        now = 0.0
        monkeypatch.setattr(_backend, 'time', types.SimpleNamespace(monotonic=lambda: now))

        def deadline_kill(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            nonlocal now
            now += 1.05  # the deadline is consumed inside the exec RPC, before `wait()`
            return '', '', 137

        fake_modal.responder = deadline_kill
        backend = await started()
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['sleep', '99'], timeout=1)

    async def test_early_sigkill_is_a_real_exit(self, fake_modal: FakeModal) -> None:
        # A command that dies by SIGKILL well before the deadline (an OOM kill, a `kill -9`
        # it asked for) reports the exit code it really had.
        fake_modal.responder = lambda argv, timeout: ('', '', 137)
        backend = await started()
        assert (await backend.run(['kill-self'], timeout=15)).exit_code == 137

    async def test_deleted_mid_command_is_unavailable(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('partial', '', 137)
        backend = await started()
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await backend.run(['sleep', '60'])

    async def test_sigkill_while_sandbox_alive_is_an_exit(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('', '', 137)
        backend = await started()
        assert (await backend.run(['kill-self'])).exit_code == 137

    async def test_sigkill_with_failing_liveness_check_is_an_exit(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('', '', 137)
        backend = await started()
        fake_modal.sandboxes[0].poll_error = ValueError('control plane unavailable')
        assert (await backend.run(['kill-self'])).exit_code == 137

    async def test_early_sigkill_with_slow_output_is_a_real_exit(self, fake_modal: FakeModal) -> None:
        # A delayed output drain should not make an early exit look like a deadline kill.
        fake_modal.responder = lambda argv, timeout: ('', '', 137)
        fake_modal.stdout_delay = 1.1
        backend = await started()
        assert (await backend.run(['kill-self'], timeout=1)).exit_code == 137

    async def test_output_drain_is_bounded_by_the_command_deadline(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The command exits with one second of its deadline left, and a descendant keeps stdout open.
        elapsed = 0.0

        def exits_late(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            nonlocal elapsed
            elapsed = 59.0
            return '', '', 0

        monkeypatch.setattr(_backend, 'time', types.SimpleNamespace(monotonic=lambda: time.monotonic() + elapsed))
        fake_modal.responder = exits_late
        fake_modal.stdout_hangs = True
        backend = await started()
        with anyio.fail_after(30), pytest.raises(WorkspaceTimeoutError):
            await backend.run(['spawn-daemon'], timeout=60)

    @pytest.mark.parametrize('stage', ['exec_error', 'wait_error'])
    async def test_an_sdk_timeout_error_is_not_a_command_timeout(self, fake_modal: FakeModal, stage: str) -> None:
        setattr(fake_modal, stage, TimeoutError('transport'))
        backend = await started()
        with pytest.raises(TimeoutError, match='transport') as exc:
            await backend.run(['x'])
        assert not isinstance(exc.value, WorkspaceTimeoutError)

    async def test_invalid_utf8_output_uses_replacement_characters(self, fake_modal: FakeModal) -> None:
        # Modal's text mode decodes strictly; reading bytes and decoding with replacement
        # keeps a command printing binary from aborting the run.
        fake_modal.responder = lambda argv, timeout: (b'\xff\xfe', b'', 0)
        backend = await started()
        assert (await backend.run(['cat', 'binary'])).stdout == '��'

    @pytest.mark.parametrize(
        ('name', 'expected', 'match'),
        [
            ('ExecutionError', WorkspaceError, 'Command could not run in the workspace: failed'),
            ('SandboxTimeoutError', WorkspaceUnavailableError, "'sb-owned' is no longer running"),
            ('AuthError', WorkspaceUnavailableError, 'Modal rejected the credentials'),
        ],
    )
    async def test_exec_failures_are_mapped(
        self, fake_modal: FakeModal, name: str, expected: type[Exception], match: str
    ) -> None:
        fake_modal.exec_error = fake_modal.exception(name)('failed')
        backend = await started()
        with pytest.raises(expected, match=match) as exc:
            await backend.run(['x'])
        assert type(exc.value) is expected

    async def test_exec_transport_failure_propagates(self, fake_modal: FakeModal) -> None:
        fake_modal.exec_error = fake_modal.exception('ConnectionError')('connection reset')
        backend = await started()
        with pytest.raises(fake_modal.exception('ConnectionError')):
            await backend.run(['x'])

    async def test_dead_workspace_conflict_is_terminal(self, fake_modal: FakeModal) -> None:
        # A first exec on a dead sandbox surfaces as Modal's ambiguous ConflictError; the
        # poll disambiguates it from a transient abort.
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('Sandbox already finished')
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await backend.run(['x'])

    async def test_shutting_down_conflict_is_terminal(self, fake_modal: FakeModal) -> None:
        # Right after `terminate()`, Modal still polls the sandbox as running but refuses exec
        # with this ConflictError; the sandbox will not come back, so it is not retryable.
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('Modal Sandbox is shutting down.')
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await backend.run(['x'])

    async def test_a_deadline_passing_while_the_sandbox_shuts_down_is_unavailable(self, fake_modal: FakeModal) -> None:
        # Modal keeps a terminated sandbox's command running through its shutdown grace, so the
        # deadline fires first; the stop that follows is refused, which shows the sandbox is gone.
        fake_modal.wait_hangs = True
        backend = await started()
        waiter = asyncio.create_task(backend.run(['sleep', '30'], timeout=1))
        await anyio.wait_all_tasks_blocked()
        await fake_modal.sandboxes[0].terminate.aio()
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await waiter

    async def test_transient_conflict_stays_recoverable(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('aborted')
        with pytest.raises(WorkspaceError, match='aborted') as exc:
            await backend.run(['x'])
        assert not isinstance(exc.value, WorkspaceUnavailableError)

    async def test_failing_poll_preserves_the_original_error(self, fake_modal: FakeModal) -> None:
        # The classifying poll can itself fail with a raw transport error; that must not
        # abort the run in place of the error we were classifying.
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('aborted')
        fake_modal.sandboxes[0].poll_error = ValueError('transport gone')
        with pytest.raises(WorkspaceError, match='aborted'):
            await backend.run(['x'])

    async def test_poll_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('aborted')
        fake_modal.sandboxes[0].poll_error = fake_modal.exception('AuthError')('unauthenticated')
        with pytest.raises(WorkspaceUnavailableError, match='Modal rejected the credentials'):
            await backend.run(['x'])

    async def test_poll_reporting_a_missing_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.exec_error = fake_modal.exception('ConflictError')('aborted')
        fake_modal.sandboxes[0].poll_error = fake_modal.exception('NotFoundError')('gone')
        with pytest.raises(WorkspaceUnavailableError):
            await backend.run(['x'])

    async def test_attached_sandbox_names_itself_when_gone(self, fake_modal: FakeModal) -> None:
        backend = await started(ref=WorkspaceRef(provider='modal', id='sb-keep'))
        fake_modal.exec_error = fake_modal.exception('SandboxTerminatedError')('gone')
        with pytest.raises(WorkspaceUnavailableError, match="'sb-keep' is no longer running"):
            await backend.run(['x'])

    async def test_run_wait_failure_is_a_workspace_error(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_error = fake_modal.exception('ExecutionError')('wait failed')
        backend = await started()
        with pytest.raises(WorkspaceError, match='Could not read the command result'):
            await backend.run(['x'])

    async def test_raw_run_wait_failure_propagates(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_error = RuntimeError('raw wait failed')
        backend = await started()
        with pytest.raises(RuntimeError, match='raw wait failed'):
            await backend.run(['x'])

    @pytest.mark.parametrize('stage', ['stdout_error', 'wait_error'])
    async def test_a_failed_read_stops_the_started_command(self, fake_modal: FakeModal, stage: str) -> None:
        # The command keeps running in the sandbox when its output or exit status cannot be read.
        setattr(fake_modal, stage, RuntimeError('stream lost'))
        fake_modal.wait_hangs = stage == 'stdout_error'
        backend = await started()
        with pytest.raises(RuntimeError, match='stream lost'):
            await backend.run(['sleep', '30'])
        assert any('modal-stop' in call.argv for call in fake_modal.sandboxes[0].exec_calls)

    async def test_output_over_the_limit_stops_the_command(self, fake_modal: FakeModal) -> None:
        # Neither stream passes 10 MiB alone; the limit is on their sum. The command never
        # exits by itself, so only the limit ends it.
        fake_modal.responder = lambda argv, timeout: (b'o' * (6 << 20), b'e' * (5 << 20), 0)
        fake_modal.wait_hangs = True
        backend = await started()
        with pytest.raises(WorkspaceOutputLimitError) as exc:
            await backend.run(['yes'])
        assert exc.value.limit == 10 << 20
        assert exc.value.stdout == 'o' * 65_536
        assert exc.value.stderr == 'e' * 65_536
        assert any('modal-stop' in call.argv for call in fake_modal.sandboxes[0].exec_calls)

    async def test_cancel_stops_only_the_command_group(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_hangs = True
        backend = await started()
        waiter = asyncio.create_task(backend.run(['sleep', '30'], timeout=5))
        await anyio.wait_all_tasks_blocked()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        sandbox = fake_modal.sandboxes[0]
        assert any(
            'kill -TERM' in ' '.join(call.argv) and 'kill -KILL' in ' '.join(call.argv) for call in sandbox.exec_calls
        )
        assert not sandbox.shutting_down
        assert backend.ref == WorkspaceRef(provider='modal', id=sandbox.object_id)

    async def test_custom_image_with_setsid_uses_group_isolation(self, fake_modal: FakeModal) -> None:
        backend = await started(image='custom:full')
        await backend.run(['true'])
        sandbox = fake_modal.sandboxes[0]
        assert len(sandbox.start_scripts) == 1
        assert sum('command -v setsid' in ' '.join(call.argv) for call in sandbox.exec_calls) == 1

    async def test_isolation_probe_does_not_consume_the_command_timeout(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The first command on a custom image probes for setsid. A slow probe, like acquiring the
        # sandbox, must not count against the command's own timeout.
        now = 0.0
        monkeypatch.setattr(_backend, 'time', types.SimpleNamespace(monotonic=lambda: now))

        def slow_probe(argv: list[str], timeout: int | None) -> tuple[str, str, int]:
            nonlocal now
            if 'command -v setsid' in ' '.join(argv):
                now += 60
                return '', '', 0
            return '', '', 137

        fake_modal.responder = slow_probe
        backend = await started(image='custom:full')
        # A 137 that did not use up the command's deadline is its real exit, not a timeout.
        assert (await backend.run(['kill-self'], timeout=30)).exit_code == 137

    async def test_missing_setsid_uses_direct_exec_and_caches_probe(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: (
            ('', '', 127) if 'command -v setsid' in ' '.join(argv) else ('ok', '', 0)
        )
        backend = await started(image='custom:minimal')
        assert (await backend.run(['echo', 'ok'])).stdout == 'ok'
        assert (await backend.run(['echo', 'ok'])).stdout == 'ok'
        calls = fake_modal.sandboxes[0].exec_calls
        assert sum('command -v setsid' in ' '.join(call.argv) for call in calls) == 1
        assert not fake_modal.sandboxes[0].start_scripts
        fake_modal.wait_hangs = True
        waiter = asyncio.create_task(backend.run(['sleep', '30'], timeout=None))
        await anyio.wait_all_tasks_blocked()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        stop = next(call for call in calls if 'modal-stop' in call.argv)
        assert 'kill -TERM "$pid"' in stop.argv[2]
        assert 'kill -TERM -"$pid"' not in stop.argv[2]

    async def test_successful_command_removes_pid_marker(self, fake_modal: FakeModal, tmp_path: Path) -> None:
        backend = await started()
        await backend.run(['true'])
        script = fake_modal.sandboxes[0].start_scripts[-1]
        marker = tmp_path / 'pid'
        subprocess.run(['sh', '-c', script, 'modal-command', str(tmp_path / 'cancel'), str(marker), 'true'], check=True)
        assert not marker.exists()
        (tmp_path / 'cancel').touch()
        blocked = subprocess.run(
            ['sh', '-c', script, 'modal-command', str(tmp_path / 'cancel'), str(marker), 'true'], check=False
        )
        assert blocked.returncode == 143
        assert not marker.exists()
        # The late start the tombstone blocked was the one it existed for.
        assert not (tmp_path / 'cancel').exists()

    async def test_stopping_a_started_command_removes_its_tombstone(
        self, fake_modal: FakeModal, tmp_path: Path
    ) -> None:
        fake_modal.wait_hangs = True
        backend = await started()
        waiter = asyncio.create_task(backend.run(['sleep', '30']))
        await anyio.wait_all_tasks_blocked()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        sandbox = fake_modal.sandboxes[0]
        start = sandbox.start_scripts[-1]
        stop = next(call for call in sandbox.exec_calls if 'modal-stop' in call.argv).argv[2]
        cancel, marker = tmp_path / 'cancel', tmp_path / 'pid'
        # Its own session stands in for `setsid`, which macOS lacks.
        command = subprocess.Popen(
            ['sh', '-c', start, 'modal-command', str(cancel), str(marker), 'sleep', '30'], start_new_session=True
        )
        try:
            with anyio.fail_after(30):
                while not (marker.exists() and marker.read_text().strip()):
                    await anyio.sleep(0.01)
            subprocess.run(['sh', '-c', stop, 'modal-stop', str(cancel), str(marker)], check=True)
            assert command.wait(timeout=30) != 0
        finally:
            command.kill()
        assert not cancel.exists()

    async def test_stop_uses_stable_directory_after_command_cwd_is_removed(self, fake_modal: FakeModal) -> None:
        fake_modal.wait_hangs = True
        backend = ModalSandboxBackend(working_dir='/deleted')
        waiter = asyncio.create_task(backend.run(['sleep', '30']))
        await anyio.wait_all_tasks_blocked()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        stop = next(call for call in fake_modal.sandboxes[0].exec_calls if 'modal-stop' in call.argv)
        assert stop.workdir == '/'

    async def test_cancelling_run_propagates_the_cancellation(self, fake_modal: FakeModal) -> None:
        # A cancelled run reaps its readers and re-raises cancellation untranslated.
        fake_modal.wait_hangs = True
        backend = await started()
        waiter = asyncio.create_task(backend.run(['x'], timeout=5))
        await anyio.wait_all_tasks_blocked()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter


class TestWorkingDir:
    async def test_configured_working_dir_is_resolved_before_first_operation(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/canonical/work\n', '', 0)
        backend = ModalSandboxBackend(working_dir='/alias')
        assert await backend.working_dir() == '/canonical/work'
        assert backend.ref is not None

    async def test_probed_once_and_cached(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/srv\n', '', 0)
        backend = await started()
        assert await backend.working_dir() == '/srv'
        assert await backend.working_dir() == '/srv'
        assert [call.argv for call in fake_modal.sandboxes[0].exec_calls] == [
            ['/bin/sh', '-c', 'exec "$@"', 'sh', 'pwd', '-P']
        ]

    async def test_the_probe_carries_a_deadline(self, fake_modal: FakeModal) -> None:
        # Modal has no per-command kill, so even the internal probe is bounded.
        fake_modal.responder = lambda argv, timeout: ('/srv\n', '', 0)
        backend = await started()
        await backend.working_dir()
        assert fake_modal.sandboxes[0].exec_calls[-1].timeout == 10

    @pytest.mark.parametrize(
        ('stdout', 'exit_code'),
        [('', 0), ('relative/dir\n', 0), ('/srv\n', 1)],
    )
    async def test_an_unusable_answer_is_refused(self, fake_modal: FakeModal, stdout: str, exit_code: int) -> None:
        # Caching anything but an absolute path would hand every later `resolve()` a working
        # directory that is not one, mis-resolving relative paths with no error.
        fake_modal.responder = lambda argv, timeout: (stdout, '', exit_code)
        backend = await started()
        with pytest.raises(WorkspaceError, match='Could not determine the working directory'):
            await backend.working_dir()

    async def test_the_facade_resolves_relative_paths_against_it(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/srv\n', '', 0)
        workspace = Workspace(await started())
        assert await workspace.resolve('src/main.py') == '/srv/src/main.py'

    async def test_trailing_space_in_working_dir_is_preserved(self, fake_modal: FakeModal) -> None:
        fake_modal.responder = lambda argv, timeout: ('/srv/project \n', '', 0)
        backend = await started()
        fake_modal.sandboxes[0].files['/srv/project /marker.txt'] = b'marker'
        workspace = Workspace(backend)
        assert await workspace.read_text('marker.txt') == 'marker'

    async def test_without_working_dir_the_default_image_starts_in_root_and_others_in_their_own(
        self, fake_modal: FakeModal
    ) -> None:
        # The default image has no WORKDIR, so it would start in `/`; a custom image keeps its WORKDIR.
        await started()
        assert fake_modal.create_kwargs[-1]['workdir'] == '/root'
        await started(image='ubuntu:22.04')
        assert fake_modal.create_kwargs[-1]['workdir'] is None

    async def test_configured_working_dir_is_where_commands_start(self, fake_modal: FakeModal, tmp_path: Path) -> None:
        home, project = tmp_path / 'home', tmp_path / 'project'
        home.mkdir()
        project.mkdir()
        fake_modal.host_root = home.resolve()
        backend = ModalSandboxBackend(working_dir=str(project.resolve()))
        assert (await backend.run('pwd -P', shell=True)).stdout == f'{project.resolve()}\n'
        await Workspace(backend).write_text('note.txt', 'hi')
        assert (project / 'note.txt').read_text() == 'hi'
        assert not (home / 'note.txt').exists()


class TestCreate:
    async def test_lost_create_reply_recovers_sandbox_by_name(self, fake_modal: FakeModal) -> None:
        fake_modal.create_reply_error = fake_modal.exception('ConnectionError')('reply lost')
        backend = ModalSandboxBackend()
        sandbox = await backend.get_sandbox()
        assert backend.ref == WorkspaceRef(provider='modal', id=sandbox.object_id)
        assert fake_modal.owned_creates == 1
        assert await backend.get_sandbox() is sandbox

    async def test_lost_create_reply_at_local_deadline_recovers_ref(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_modal.create_gate = anyio.Event()
        monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._backend._CREATE_TIMEOUT', 0.01)
        backend = ModalSandboxBackend()
        sandbox = await backend.get_sandbox()
        assert backend.ref == WorkspaceRef(provider='modal', id=sandbox.object_id)
        assert fake_modal.owned_creates == 1

    async def test_native_task_cancellation_records_in_flight_creation(self, fake_modal: FakeModal) -> None:
        backend = ModalSandboxBackend()
        fake_modal.create_gate = anyio.Event()
        task = asyncio.create_task(backend.get_sandbox())
        with anyio.fail_after(5):
            while not fake_modal.create_started:
                await anyio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        fake_modal.create_gate.set()
        with anyio.fail_after(5):
            while backend.ref is None:
                await anyio.sleep(0)
        assert backend.ref == WorkspaceRef(provider='modal', id='sb-owned')
        assert await backend.get_sandbox() is fake_modal.sandboxes[0]
        assert fake_modal.owned_creates == 1

    @pytest.mark.parametrize('operation', ['run', 'write_bytes'])
    async def test_ref_is_recorded_by_the_first_operation(self, fake_modal: FakeModal, operation: str) -> None:
        backend = ModalSandboxBackend()
        assert backend.ref is None
        if operation == 'run':
            await backend.run(['true'])
        else:
            await backend.write_bytes('/tmp/file', b'data')
        assert backend.ref == WorkspaceRef(provider='modal', id=fake_modal.sandboxes[0].object_id)

    async def test_creates_from_config(self, fake_modal: FakeModal) -> None:
        backend = await started(
            image='ubuntu:22.04',
            app_name='my-app',
            create_app_if_missing=False,
            sandbox_timeout=120,
            idle_timeout=60,
            working_dir='/work',
            env={'FOO': 'bar'},
        )
        assert backend.ref == WorkspaceRef(provider='modal', id='sb-owned')
        assert fake_modal.app_lookups[-1] == {'name': 'my-app', 'create_if_missing': False}
        assert fake_modal.image_tags[-1] == 'ubuntu:22.04'
        assert fake_modal.create_kwargs[-1]['timeout'] == 120
        assert fake_modal.create_kwargs[-1]['idle_timeout'] == 60
        assert fake_modal.create_kwargs[-1]['workdir'] == '/work'
        assert fake_modal.create_kwargs[-1]['env'] == {'FOO': 'bar'}

    async def test_an_image_object_is_used_as_given(self, fake_modal: FakeModal) -> None:
        image: Any = fake_modal.module.Image()
        await started(image=image)
        assert fake_modal.create_kwargs[-1]['image'] is image
        assert fake_modal.image_tags == []

    async def test_default_image_has_git_and_ripgrep(self, fake_modal: FakeModal) -> None:
        await started()
        image = fake_modal.create_kwargs[-1]['image']
        assert isinstance(image, FakeImage)
        assert {'git', 'ripgrep'} <= set(image.apt_packages)

    async def test_default_app(self, fake_modal: FakeModal) -> None:
        await started()
        assert fake_modal.app_lookups[-1] == {'name': 'pydantic-ai-harness', 'create_if_missing': True}
        assert fake_modal.create_kwargs[-1]['env'] is None
        # Modal's maximum lifetime, and no idle termination: Modal's idle termination is permanent,
        # so it would end a conversation that pauses for a while.
        assert fake_modal.create_kwargs[-1]['timeout'] == 86_400
        assert fake_modal.create_kwargs[-1]['idle_timeout'] is None

    @pytest.mark.parametrize(
        ('name', 'match'),
        [
            ('InvalidError', 'Could not start Modal sandbox: failed'),
            ('NotFoundError', 'Could not start Modal sandbox: failed'),
            ('AlreadyExistsError', 'Could not start Modal sandbox: failed'),
            ('ExecutionError', 'Could not start Modal sandbox: failed'),
            ('AuthError', 'Modal rejected the credentials'),
        ],
    )
    async def test_a_refused_create_is_unavailable(self, fake_modal: FakeModal, name: str, match: str) -> None:
        # Modal refusing to create the sandbox (an unknown app or image, an invalid argument such
        # as a `sandbox_timeout` above its limit) cannot be fixed by the model or a retry, so it
        # ends the run instead of going back to the model as a `WorkspaceError`.
        fake_modal.create_error = fake_modal.exception(name)('failed')
        with pytest.raises(WorkspaceUnavailableError, match=match) as exc:
            await started()
        assert type(exc.value) is WorkspaceUnavailableError

    async def test_create_timeout_mentions_image_build_or_pull(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_modal.create_gate = anyio.Event()
        fake_modal.create_before_gate = True
        monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._backend._CREATE_TIMEOUT', 0.01)
        with pytest.raises(TimeoutError, match='image build or pull may still be running'):
            await ModalSandboxBackend().get_sandbox()

    @pytest.mark.parametrize('branch', ['deadline', 'error'])
    async def test_a_stalled_recovery_lookup_still_ends_creation(
        self, fake_modal: FakeModal, monkeypatch: pytest.MonkeyPatch, branch: str
    ) -> None:
        class _StalledLookup:
            async def aio(self, *args: Any, **kwargs: Any) -> Any:
                await anyio.sleep_forever()

        monkeypatch.setattr(fake_modal.module.Sandbox, 'from_name', _StalledLookup())
        monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._backend._RECOVER_CREATE_TIMEOUT', 0.01)
        if branch == 'deadline':
            fake_modal.create_gate = anyio.Event()
            fake_modal.create_before_gate = True
            monkeypatch.setattr('pydantic_ai_harness.modal_sandbox._backend._CREATE_TIMEOUT', 0.01)
            expected = pytest.raises(TimeoutError, match='did not complete within')
        else:
            fake_modal.create_reply_error = fake_modal.exception('ConnectionError')('reply lost')
            expected = pytest.raises(fake_modal.exception('ConnectionError'), match='reply lost')
        # Hang guard only: without a recovery bound the lookup never returns.
        with anyio.fail_after(5), expected:
            await ModalSandboxBackend().get_sandbox()

    async def test_image_build_error_is_unavailable(self, fake_modal: FakeModal) -> None:
        fake_modal.create_error = fake_modal.exception('ImageBuildError')('bad image')
        with pytest.raises(WorkspaceUnavailableError, match='Could not start Modal sandbox: bad image'):
            await started()

    @pytest.mark.parametrize(
        'settings',
        [
            {'sandbox_timeout': 9},
            {'sandbox_timeout': 86401},
            {'idle_timeout': 0},
            {'idle_timeout': -5},
            {'idle_timeout': True},
            {'idle_timeout': 1.5},
            {'image': 42},
            {'env': {'TOKEN': 42}},
        ],
    )
    async def test_bad_constructor_inputs_fail_before_create(
        self, fake_modal: FakeModal, settings: dict[str, Any]
    ) -> None:
        with pytest.raises((TypeError, ValueError)):
            ModalSandboxBackend(**settings)
        assert not fake_modal.sandboxes

    async def test_create_transport_failure_propagates(self, fake_modal: FakeModal) -> None:
        fake_modal.create_error = fake_modal.exception('ResourceExhaustedError')('rate limited')
        with pytest.raises(fake_modal.exception('ResourceExhaustedError')):
            await started()

    async def test_rejects_relative_working_dir(self, fake_modal: FakeModal) -> None:
        with pytest.raises(ValueError, match='working_dir must be an absolute workspace path'):
            await started(working_dir='repo')

    async def test_preserves_parent_segments_in_working_dir(self, fake_modal: FakeModal) -> None:
        await started(working_dir='/linked/../target')

        assert fake_modal.create_kwargs[-1]['workdir'] == '/linked/../target'


class TestConnect:
    async def test_invalid_ref_is_unavailable(self, fake_modal: FakeModal) -> None:
        fake_modal.attach_error = fake_modal.exception('InvalidError')('bad id')
        with pytest.raises(WorkspaceUnavailableError, match='sb-invalid'):
            await started(ref=WorkspaceRef(provider='modal', id='sb-invalid'))

    async def test_connects_to_a_running_sandbox(self, fake_modal: FakeModal) -> None:
        backend = await started(ref=WorkspaceRef(provider='modal', id='sb-keep'))
        assert fake_modal.attach_ids == ['sb-keep']
        assert backend.ref == WorkspaceRef(provider='modal', id='sb-keep')

    async def test_connect_to_a_finished_sandbox_fails(self, fake_modal: FakeModal) -> None:
        # Modal hands back a handle for a sandbox it still knows about even after it has
        # terminated, so a ref must not resolve to a dead environment.
        fake_modal.attach_poll_result = 0
        with pytest.raises(WorkspaceUnavailableError, match='no longer running'):
            await started(ref=WorkspaceRef(provider='modal', id='sb-gone'))
        assert not fake_modal.create_kwargs

    async def test_connect_to_an_unknown_id_fails(self, fake_modal: FakeModal) -> None:
        fake_modal.attach_error = fake_modal.exception('NotFoundError')('not found')
        with pytest.raises(WorkspaceUnavailableError, match="'sb-nope'"):
            await started(ref=WorkspaceRef(provider='modal', id='sb-nope'))
        assert not fake_modal.create_kwargs

    async def test_an_operation_on_a_terminated_sandbox_still_shutting_down_fails(self, fake_modal: FakeModal) -> None:
        owner = await started()
        assert owner.ref is not None
        await fake_modal.sandboxes[0].terminate.aio()
        backend = ModalSandboxBackend(ref=owner.ref)
        with pytest.raises(WorkspaceUnavailableError, match="'sb-owned' is no longer running"):
            await backend.run(['true'])

    async def test_an_operation_on_a_gone_sandbox_does_not_create_a_replacement(self, fake_modal: FakeModal) -> None:
        fake_modal.attach_error = fake_modal.exception('NotFoundError')('not found')
        backend = ModalSandboxBackend(ref=WorkspaceRef(provider='modal', id='sb-nope'))
        for _ in range(2):
            with pytest.raises(WorkspaceUnavailableError, match="'sb-nope'"):
                await backend.run(['true'])
        assert backend.ref == WorkspaceRef(provider='modal', id='sb-nope')
        assert not fake_modal.create_kwargs
        assert not fake_modal.sandboxes


class TestFilesystem:
    @pytest.mark.parametrize('method', ['read_bytes', 'write_bytes', 'stat', 'list_dir', 'make_dir', 'remove'])
    async def test_relative_paths_fail_before_creating_sandbox(self, fake_modal: FakeModal, method: str) -> None:
        backend = ModalSandboxBackend()
        with pytest.raises(ValueError, match='absolute workspace path'):
            if method == 'write_bytes':
                await backend.write_bytes('relative', b'data')
            else:
                await getattr(backend, method)('relative')
        assert backend.ref is None
        assert not fake_modal.sandboxes

    async def test_write_then_read_round_trips(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        assert await backend.read_bytes('/tmp/a.txt') == b'body'

    async def test_stat_reports_size_for_files(self, fake_modal: FakeModal) -> None:
        backend = await started()
        await backend.write_bytes('/tmp/a.txt', b'body')
        entry = await backend.stat('/tmp/a.txt')
        assert (entry.name, entry.path, entry.is_dir, entry.size) == ('a.txt', '/tmp/a.txt', False, 4)

    async def test_stat_reports_no_size_for_directories(self, fake_modal: FakeModal) -> None:
        # A directory's reported size is a filesystem implementation detail, not a content
        # length, so the protocol carrier reports none.
        backend = await started()
        await backend.make_dir('/tmp/pkg')
        entry = await backend.stat('/tmp/pkg')
        assert (entry.is_dir, entry.size) == (True, None)

    async def test_list_dir_returns_absolute_paths(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].listing = [FileInfo('a.py', False, size=7), FileInfo('pkg', True)]
        entries = await backend.list_dir('/srv')
        assert [(entry.name, entry.path, entry.is_dir, entry.size) for entry in entries] == [
            ('a.py', '/srv/a.py', False, 7),
            ('pkg', '/srv/pkg', True, None),
        ]

    async def test_symlinks_follow_relative_chained_dangling_and_looping_targets(
        self, fake_modal: FakeModal, tmp_path: Path
    ) -> None:
        fake_modal.host_root = tmp_path
        (tmp_path / 'data.txt').write_bytes(b'12345')
        (tmp_path / 'relative').symlink_to('data.txt')
        (tmp_path / 'chained').symlink_to('relative')
        (tmp_path / 'dangling').symlink_to('missing')
        (tmp_path / 'looping').symlink_to('looping')
        entries = await ModalSandboxBackend().list_dir(str(tmp_path))
        assert {entry.name: (entry.is_dir, entry.size) for entry in entries} == {
            'chained': (False, 5),
            'dangling': (False, None),
            'data.txt': (False, 5),
            'looping': (False, None),
            'relative': (False, 5),
        }

    async def test_list_dir_raises_a_link_target_failure_unwrapped(self, fake_modal: FakeModal, tmp_path: Path) -> None:
        # Link targets resolve concurrently; a failure must reach error mapping, not an ExceptionGroup.
        fake_modal.host_root = tmp_path
        (tmp_path / 'data.txt').write_bytes(b'data')
        (tmp_path / 'link').symlink_to('data.txt')
        backend = await started()
        sandbox = fake_modal.sandboxes[0]

        async def denied_stat(path: str) -> FileInfo:
            raise fake_modal.exception('SandboxFilesystemPermissionError')(f'Permission denied: {path}')

        sandbox.filesystem.stat.aio = denied_stat
        with pytest.raises(PermissionError):
            await backend.list_dir(str(tmp_path))

    async def test_parent_and_relative_leaf_symlink_write(self, fake_modal: FakeModal, tmp_path: Path) -> None:
        fake_modal.host_root = tmp_path
        (tmp_path / 'a' / 'sub').mkdir(parents=True)
        (tmp_path / 'a' / 'out').mkdir()
        (tmp_path / 'out').mkdir()
        (tmp_path / 'abs').symlink_to('a/sub')
        (tmp_path / 'a' / 'sub' / 'link').symlink_to('../out/file')
        (tmp_path / 'a' / 'out' / 'file').write_bytes(b'old')
        (tmp_path / 'out' / 'file').write_bytes(b'wrong length')
        backend = ModalSandboxBackend()
        assert (await backend.stat(str(tmp_path / 'abs' / 'link'))).size == 3
        entry = next(e for e in await backend.list_dir(str(tmp_path / 'abs')) if e.name == 'link')
        assert entry.size == 3
        await backend.write_bytes(str(tmp_path / 'abs' / 'link'), b'new')
        assert (tmp_path / 'a' / 'out' / 'file').read_bytes() == b'new'
        assert (tmp_path / 'out' / 'file').read_bytes() == b'wrong length'
        assert (tmp_path / 'a' / 'sub' / 'link').is_symlink()
        (tmp_path / 'a' / 'sub' / 'loop').symlink_to('loop')
        with pytest.raises(OSError):
            await backend.write_bytes(str(tmp_path / 'abs' / 'loop'), b'no')
        assert (tmp_path / 'a' / 'sub' / 'loop').is_symlink()

    async def test_unresolved_kernel_link_is_not_replaced(
        self, fake_modal: FakeModal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_modal.host_root = tmp_path
        link = tmp_path / 'loop'
        link.symlink_to('loop')
        backend = ModalSandboxBackend()

        async def unresolved(self: Workspace, path: str) -> str:
            return path

        monkeypatch.setattr(Workspace, 'realpath', unresolved)
        with pytest.raises(FileNotFoundError):
            await backend.stat(str(link))
        with pytest.raises(OSError, match='Symlink loop'):
            await backend.write_bytes(str(link), b'no')
        assert link.is_symlink()

    async def test_dangling_symlink_is_listed_but_does_not_exist(self, fake_modal: FakeModal, tmp_path: Path) -> None:
        fake_modal.host_root = tmp_path
        (tmp_path / 'dangling').symlink_to('missing')
        backend = ModalSandboxBackend()
        entry = next(e for e in await backend.list_dir(str(tmp_path)) if e.name == 'dangling')
        assert (entry.is_dir, entry.size) == (False, None)
        with pytest.raises(FileNotFoundError):
            await backend.stat(str(tmp_path / 'dangling'))
        assert await backend.exists(str(tmp_path / 'dangling')) is False

    async def test_list_dir_resolves_links_concurrently(self, fake_modal: FakeModal, tmp_path: Path) -> None:
        fake_modal.host_root = tmp_path
        (tmp_path / 'target').write_bytes(b'x')
        for index in range(20):
            (tmp_path / f'link-{index}').symlink_to('target')
        backend = await started()
        sandbox = fake_modal.sandboxes[0]
        original = sandbox.filesystem.stat.aio

        simultaneous = anyio.Event()
        calls = 0

        async def slow_stat(path: str) -> FileInfo:
            nonlocal calls
            calls += 1
            if calls >= 2:
                simultaneous.set()
            await simultaneous.wait()
            return await original(path)

        sandbox.filesystem.stat.aio = slow_stat
        with anyio.fail_after(10):
            entries = await backend.list_dir(str(tmp_path))
        assert len(entries) == 21

    async def test_symlink_loop_is_not_found(self, fake_modal: FakeModal, tmp_path: Path) -> None:
        fake_modal.host_root = tmp_path
        (tmp_path / 'loop').symlink_to('loop')
        backend = await started()
        with pytest.raises(FileNotFoundError):
            await backend.stat(str(tmp_path / 'loop'))

    async def test_remove_is_recursive(self, fake_modal: FakeModal) -> None:
        # One call covers both halves of the protocol's `remove`: on a file `recursive`
        # changes nothing, and on a directory it is what removes a non-empty one.
        fake_modal.responder = lambda argv, timeout: ('/work\n', '', 0)
        backend = await started()
        await backend.make_dir('/tmp/pkg')
        await backend.remove('/tmp/pkg')
        assert fake_modal.sandboxes[0].removals == [('/tmp/pkg', True)]

    async def test_remove_refuses_the_working_dir_and_its_ancestors(
        self, fake_modal: FakeModal, tmp_path: Path
    ) -> None:
        # Modal's recursive remove would take the whole working directory with `.`.
        root = tmp_path.resolve() / 'work'
        (root / 'child').mkdir(parents=True)
        (root / 'link').symlink_to(root)
        fake_modal.host_root = root
        workspace = Workspace(await started(working_dir=str(root)))
        for path in ('.', 'child/..', str(root.parent), '/'):
            with pytest.raises(ValueError, match='workspace root or its ancestor'):
                await workspace.remove(path)
        assert (root / 'child').is_dir()
        # A link to the root is removed itself, like any other entry.
        await workspace.remove('link')
        await workspace.remove('child')
        assert sorted(root.iterdir()) == []

    async def test_exists_is_false_through_a_non_directory(self, fake_modal: FakeModal) -> None:
        # Modal splits "there is nothing at that path" in two, and a non-leaf path component
        # that is a file is the other half.
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemNotADirectoryError')('nope')
        assert await backend.exists('/tmp/a.txt/deeper') is False

    @pytest.mark.parametrize('operation', ['read_bytes', 'stat'])
    async def test_a_missing_path_raises_the_builtin_error(self, fake_modal: FakeModal, operation: str) -> None:
        # The protocol's contract: backends translate their SDK's own missing-file exception
        # into the builtin `FileNotFoundError` every consumer already handles.
        backend = await started()
        with pytest.raises(FileNotFoundError, match=r"'/tmp/missing.txt'"):
            await getattr(backend, operation)('/tmp/missing.txt')

    async def test_exists_still_reports_other_failures(self, fake_modal: FakeModal) -> None:
        # Only "there is nothing at that path" is an answer; anything else is a failure.
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemError')('Permission denied')
        with pytest.raises(WorkspaceError, match='Permission denied'):
            await backend.exists('/root/x')

    async def test_a_filesystem_error_is_recoverable_while_the_sandbox_runs(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemError')('Permission denied')
        with pytest.raises(WorkspaceError, match='Permission denied') as exc:
            await backend.write_bytes('/root/x', b'data')
        assert isinstance(exc.value, WorkspaceError)
        assert not isinstance(exc.value, WorkspaceUnavailableError)

    async def test_a_filesystem_error_on_a_dead_sandbox_is_terminal(self, fake_modal: FakeModal) -> None:
        # Modal's filesystem wraps a dead sandbox as an ordinary-looking error, so the poll
        # is what keeps the model out of a retry loop against a corpse.
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemError')('request failed')
        fake_modal.sandboxes[0].poll_result = 0
        with pytest.raises(WorkspaceUnavailableError):
            await backend.read_bytes('/x')

    async def test_a_filesystem_error_on_a_sandbox_still_shutting_down_is_terminal(self, fake_modal: FakeModal) -> None:
        # While a terminated sandbox shuts down it polls as running and its filesystem fails
        # generically; the exec probe is what names the state.
        backend = await started()
        await fake_modal.sandboxes[0].terminate.aio()
        with pytest.raises(WorkspaceUnavailableError, match='no longer running'):
            await backend.read_bytes('/x')
        assert fake_modal.sandboxes[0].exec_calls  # the FIFO probe detects shutdown before the SDK read

    async def test_a_wrapped_auth_failure_is_terminal(self, fake_modal: FakeModal) -> None:
        backend = await started()
        fake_modal.sandboxes[0].fs_error = fake_modal.exception('SandboxFilesystemError')('request failed')
        fake_modal.sandboxes[0].poll_error = fake_modal.exception('AuthError')('unauthenticated')
        with pytest.raises(WorkspaceUnavailableError, match='Modal rejected the credentials'):
            await backend.list_dir('/x')


class TestErrorMapping:
    @pytest.mark.parametrize(
        ('name', 'expected'),
        [
            ('AuthError', WorkspaceUnavailableError),
            ('PermissionDeniedError', WorkspaceUnavailableError),
            ('NotFoundError', WorkspaceUnavailableError),
            ('SandboxTerminatedError', WorkspaceUnavailableError),
            ('SandboxTimeoutError', WorkspaceUnavailableError),
            ('SandboxFilesystemNotFoundError', FileNotFoundError),
            ('SandboxFilesystemIsADirectoryError', IsADirectoryError),
            ('SandboxFilesystemNotADirectoryError', NotADirectoryError),
            ('SandboxFilesystemPermissionError', PermissionError),
            ('SandboxFilesystemPathAlreadyExistsError', FileExistsError),
            ('InvalidError', WorkspaceError),
            ('ConflictError', WorkspaceError),
            ('AlreadyExistsError', WorkspaceError),
            ('ExecutionError', WorkspaceError),
            ('RequestSizeError', WorkspaceError),
            ('SandboxFilesystemError', WorkspaceError),
            ('FilesystemExecutionError', WorkspaceError),
            ('ConnectionError', None),
            ('ServiceError', None),
            ('ResourceExhaustedError', None),
            ('InternalError', None),
            ('Error', None),
        ],
    )
    async def test_modal_exception_maps_to_the_protocol_failure(
        self, fake_modal: FakeModal, name: str, expected: type[Exception] | None
    ) -> None:
        # `None` propagates the SDK exception unchanged: transport, rate-limit, and unknown
        # failures are what a durable engine retries. The fake replaces an exec-level failure as
        # Modal's filesystem layer does, so this also checks the original is classified, not
        # its stand-in. The sandbox still polls as running, so the ambiguous kinds stay
        # operation failures.
        backend = await started()
        error = fake_modal.exception(name)('boom')
        fake_modal.sandboxes[0].fs_error = error
        with pytest.raises(Exception) as exc_info:
            await backend.read_bytes('/p')
        if expected is None:
            assert exc_info.value is error
        else:
            assert type(exc_info.value) is expected
            assert exc_info.value.__cause__ is error


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_modal_call(fake_modal: FakeModal) -> None:
    """The Modal SDK runs its calls on asyncio tasks, so Trio gets a clear `UserError` instead."""
    message = r'^Modal needs the asyncio event loop: the Modal SDK runs its calls on asyncio tasks\.$'
    backend = ModalSandboxBackend()
    with pytest.raises(UserError, match=message):
        await backend.run(['true'])
    with pytest.raises(UserError, match=message):
        await backend.read_bytes('/file')
    with pytest.raises(UserError, match=message):
        await ModalSandbox().destroy(WorkspaceRef(provider='modal', id='sb-owned'))
    assert not fake_modal.create_kwargs
    assert fake_modal.attach_ids == []

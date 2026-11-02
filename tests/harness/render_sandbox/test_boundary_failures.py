"""Failure contracts at Render's public SDK boundary, through the public backend.

The scripted SDK supplies wire events and provider states, not adapter results.
This keeps parsing, command stopping, readiness, and cleanup code under test.
No credentials or network are used; only deadline constants are patched down.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import anyio
import httpx
import pytest

from ...conftest import no_leaked_tasks, try_import

with try_import() as imports_successful:
    from render import RenderAsync
    from render.client.errors import ClientError, RateLimitError
    from render.client.util import handle_httpx_exception
    from render.experimental.sandbox import (
        Sandbox,
        SandboxExecEvent,
        SandboxExecExit,
        SandboxExecOutput,
        SandboxNotFoundError,
    )

    from pydantic_ai.workspaces import (
        WorkspaceError,
        WorkspaceOutputLimitError,
        WorkspaceRef,
        WorkspaceTimeoutError,
        WorkspaceUnavailableError,
    )
    from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

    from .workspace_helpers import frame, output


pytestmark = pytest.mark.skipif(not imports_successful(), reason='render not installed')


@dataclass
class ScriptedSDK:
    """A public SDK double that can lose acknowledgements without replaying work."""

    client: RenderAsync
    status: str = 'running'
    creates: int = 0
    reads: int = 0
    operations: list[str] = field(default_factory=list[str])
    scenario: str = 'valid'
    stop_scenario: str = 'valid'
    status_script: list[str | Exception] = field(default_factory=list[str | Exception])
    transport_error: Exception | None = None
    refusal: Exception | None = None
    readiness_output: str = '/workspace\n'
    readiness_code: int = 0
    stop_started: asyncio.Event = field(default_factory=asyncio.Event)
    stop_gate: asyncio.Event | None = None

    def sandbox(self, status: str) -> Sandbox:
        return Sandbox('sbx-scripted', status, 'starter', 'deny-all', 'oregon', 300, datetime.now(UTC))

    def backend(self) -> RenderSandboxBackend:
        return RenderSandboxBackend(client=self.client)

    async def create(self, **kwargs: object) -> Sandbox:
        self.creates += 1
        return self.sandbox(self.status)

    async def from_id(self, identifier: str) -> Sandbox:
        assert identifier == 'sbx-scripted'
        self.reads += 1
        item = self.status_script.pop(0) if self.status_script else self.status
        if isinstance(item, Exception):
            raise item
        return self.sandbox(item)

    async def exec(self, identifier: str, launch: str) -> AsyncGenerator[SandboxExecEvent]:
        assert identifier == 'sbx-scripted'
        args = shlex.split(launch)
        operation = args[4]
        config = json.loads(args[5])
        self.operations.append(operation)
        if self.refusal is not None and operation == 'run' and config['cwd'] != '/':
            raise self.refusal

        if operation == 'stop':
            self.stop_started.set()
            if self.stop_gate is not None:
                await self.stop_gate.wait()
            if self.stop_scenario == 'missing_ack':
                yield SandboxExecExit(0)
            elif self.stop_scenario == 'wrong_ack':
                yield frame('exit', code=0)
                yield SandboxExecExit(0)
            else:
                yield frame('stopped')
                yield SandboxExecExit(0)
            return

        # The backend prepares its working directory using the same real parser.
        if config['cwd'] == '/':
            yield output(self.readiness_output)
            yield frame('exit', code=self.readiness_code)
            yield SandboxExecExit(0)
            return

        if self.transport_error is not None:
            yield output('partial')
            raise self.transport_error
        for event in self.response():
            yield event

    def response(self) -> list[SandboxExecEvent]:
        """Supply complete wire sequences so malformed messages cannot hide behind truncation."""

        # Every error scenario describes the public SDK messages, including valid
        # completion where appropriate. The backend must reject the actual defect.
        complete = [frame('exit', code=0), SandboxExecExit(0)]
        responses: dict[str, list[SandboxExecEvent]] = {
            'valid': [output('usable'), *complete],
            'invalid_json': [SandboxExecOutput('stdout', '{broken\n'), *complete],
            'supervisor_error': [frame('error', category='OSError'), *complete],
            'incomplete_frame': [SandboxExecOutput('stdout', '{"kind":'), SandboxExecExit(0)],
            'missing_direct_exit': [output('partial'), SandboxExecExit(0)],
            'missing_sdk_exit': [output('partial'), frame('exit', code=0)],
            'remote_timeout': [frame('timeout'), SandboxExecExit(0)],
            'remote_limit': [output('x' * (65 * 1024)), frame('limit'), SandboxExecExit(0)],
            'missing_cwd': [frame('cwd'), SandboxExecExit(0)],
            'fenced': [frame('fenced'), SandboxExecExit(0)],
        }
        return responses[self.scenario]


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch) -> ScriptedSDK:
    # Short bounds stand in for production deadlines, so no test sleeps through one.
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._backend._OPERATION_TIMEOUT', 1)
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._backend._POLL_INTERVAL', 0)
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._STOP_TIMEOUT', 1)
    sdk = ScriptedSDK(RenderAsync(token='test-only', owner_id='tea-test'))
    public_api = sdk.client.experimental.sandboxes
    monkeypatch.setattr(public_api, 'create', sdk.create)
    monkeypatch.setattr(public_api, 'from_id', sdk.from_id)
    monkeypatch.setattr(public_api, 'exec', sdk.exec)
    return sdk


@pytest.fixture
def owned_boundary(boundary: ScriptedSDK, monkeypatch: pytest.MonkeyPatch) -> ScriptedSDK:
    monkeypatch.setenv('RENDER_WORKSPACE_ID', 'tea-test')
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands.RenderAsync', lambda: boundary.client)
    return boundary


# Before the supervisor's terminal frame the command may still run, so it is stopped; after it, it is over.
_BEFORE_TERMINAL = ['invalid_json', 'supervisor_error', 'incomplete_frame', 'missing_direct_exit']
_AFTER_TERMINAL = ['fenced', 'missing_sdk_exit']


@pytest.mark.parametrize(
    ('scenario', 'operations'),
    [(scenario, ['run', 'stop']) for scenario in _BEFORE_TERMINAL]
    + [(scenario, ['run']) for scenario in _AFTER_TERMINAL],
)
async def test_unusable_command_stream_never_becomes_success_or_replays(
    boundary: ScriptedSDK, scenario: str, operations: list[str]
) -> None:
    backend = boundary.backend()
    await backend.working_dir()
    boundary.operations.clear()
    boundary.scenario = scenario
    with pytest.raises(WorkspaceError):
        await backend.run(['side-effect'])
    assert boundary.operations == operations
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')

    boundary.scenario = 'valid'
    assert (await backend.run(['next-explicit-command'])).stdout == 'usable'
    assert boundary.creates == 1


@pytest.mark.parametrize('acknowledgement', ['missing_ack', 'wrong_ack'])
async def test_failed_stop_logs_without_replacing_the_stream_error(
    boundary: ScriptedSDK, acknowledgement: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The stop is best effort, as on the other sandboxes: its failure must not replace the stream error.
    backend = boundary.backend()
    await backend.working_dir()
    boundary.operations.clear()
    original = ConnectionError('response lost after command submission')
    boundary.transport_error = original
    boundary.stop_scenario = acknowledgement
    with pytest.raises(ConnectionError) as caught:
        await backend.run(['side-effect'])
    assert caught.value is original
    assert boundary.operations == ['run', 'stop']
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')
    assert 'Could not confirm command stop in Render sandbox sbx-scripted; it may still be running' in caplog.text
    assert str(original) not in caplog.text


async def test_readiness_deadline_retains_id_and_does_not_create_again(
    boundary: ScriptedSDK, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._backend._READINESS_TIMEOUT', 0)
    boundary.status = 'creating'
    backend = boundary.backend()
    with pytest.raises(TimeoutError):
        await backend.run(['must-not-start'])
    assert boundary.operations == []
    assert boundary.creates == 1
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')

    # A later, explicit call can use the same provider resource once it is ready.
    boundary.status = 'running'
    assert (await backend.run(['explicit-recovery'])).stdout == 'usable'
    assert boundary.creates == 1


@pytest.mark.parametrize(
    ('code', 'directory'),
    [(1, '/workspace\n'), (0, 'relative\n'), (0, '/workspace\nextra\n')],
)
async def test_unusable_readiness_result_preserves_created_reference(
    boundary: ScriptedSDK, code: int, directory: str
) -> None:
    # A successful API response still needs a usable absolute working directory.
    boundary.readiness_code = code
    boundary.readiness_output = directory
    backend = boundary.backend()
    with pytest.raises(WorkspaceUnavailableError, match='is unavailable in the sandbox'):
        await backend.working_dir()
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')
    assert boundary.creates == 1


async def test_new_cancellation_during_failure_cleanup_takes_priority(boundary: ScriptedSDK) -> None:
    # Pause the provider's stop acknowledgement, then cancel the caller while its
    # original transport failure is being cleaned up. The cancellation wins once the stop finishes.
    backend = boundary.backend()
    await backend.working_dir()
    boundary.transport_error = ConnectionError('command response lost')
    boundary.stop_gate = asyncio.Event()
    pending = asyncio.create_task(backend.run(['side-effect']))
    try:
        async with asyncio.timeout(10):
            await boundary.stop_started.wait()
        pending.cancel()
        # Let the cancelled caller observe its shielded stop before releasing it.
        await asyncio.sleep(0)
        boundary.stop_gate.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert boundary.operations[-1] == 'stop'
        assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')
    finally:
        boundary.stop_gate.set()
        await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.parametrize(
    'failure', [ClientError('denied'), SandboxNotFoundError('gone')] if imports_successful() else []
)
async def test_readiness_provider_error_retains_reference(boundary: ScriptedSDK, failure: Exception) -> None:
    boundary.status_script = [failure]
    backend = boundary.backend()
    with pytest.raises(WorkspaceUnavailableError) as caught:
        await backend.working_dir()
    assert caught.value.__cause__ is failure
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')
    assert boundary.creates == 1


async def test_missing_sandbox_in_status_audit_is_unavailable(boundary: ScriptedSDK) -> None:
    backend = boundary.backend()
    await backend.working_dir()
    failure = SandboxNotFoundError('gone')
    boundary.status_script = [failure]
    boundary.transport_error = ConnectionError('stream interrupted')
    with pytest.raises(WorkspaceUnavailableError) as caught:
        await backend.run(['side-effect'])
    assert caught.value.__cause__ is failure
    assert boundary.creates == 1


@pytest.mark.parametrize(
    ('scenario', 'error'),
    [
        ('remote_timeout', WorkspaceTimeoutError),
        ('remote_limit', WorkspaceOutputLimitError),
        ('missing_cwd', WorkspaceUnavailableError),
    ]
    if imports_successful()
    else [],
)
async def test_supervisor_terminal_result_raises_without_a_stop(
    boundary: ScriptedSDK, scenario: str, error: type[Exception]
) -> None:
    backend = boundary.backend()
    await backend.working_dir()
    boundary.operations.clear()
    boundary.scenario = scenario
    with pytest.raises(error):
        await backend.run(['side-effect'])
    assert boundary.operations == ['run']


@pytest.mark.parametrize(
    ('refusal', 'error', 'operations'),
    [
        (RateLimitError('429'), RateLimitError, ['run']),
        (ClientError('rejected'), WorkspaceUnavailableError, ['run']),
        (ClientError('lost'), WorkspaceUnavailableError, ['run', 'stop']),
    ]
    if imports_successful()
    else [],
    ids=['rate_limited', 'rejected', 'uncertain'] if imports_successful() else [],
)
async def test_stop_follows_an_exec_failure_only_when_the_command_may_have_started(
    boundary: ScriptedSDK, refusal: ClientError, error: type[Exception], operations: list[str]
) -> None:
    if str(refusal) == 'lost':
        # Render wraps a transport failure in `ClientError` with a cause; the exec may have started.
        refusal.__cause__ = ConnectionError('reset')
    backend = boundary.backend()
    await backend.working_dir()
    boundary.operations.clear()
    boundary.refusal = refusal
    with pytest.raises(error):
        await backend.run(['side-effect'])
    assert boundary.operations == operations


async def test_output_limit_carries_a_preview(boundary: ScriptedSDK) -> None:
    backend = boundary.backend()
    await backend.working_dir()
    boundary.scenario = 'remote_limit'
    with pytest.raises(WorkspaceOutputLimitError) as caught:
        await backend.run(['side-effect'])
    assert (caught.value.limit, caught.value.stdout) == (10 * 1024 * 1024, 'x' * (64 * 1024))


async def test_create_deadline_is_not_a_terminal_workspace_error(
    boundary: ScriptedSDK, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def stalled_create(**kwargs: object) -> Sandbox:
        return await asyncio.Future[Sandbox]()

    monkeypatch.setattr(boundary.client.experimental.sandboxes, 'create', stalled_create)
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._backend._OPERATION_TIMEOUT', 0)
    with pytest.raises(TimeoutError):
        await boundary.backend().working_dir()


@pytest.mark.parametrize('phase', ['create', 'readiness'])
async def test_acquisition_preserves_sdk_transport_errors(
    boundary: ScriptedSDK, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    with pytest.raises(ClientError) as translated:
        handle_httpx_exception(httpx.UnsupportedProtocol('transport failure'))
    original = translated.value

    async def failing_create(**kwargs: object) -> Sandbox:
        raise original

    if phase == 'create':
        monkeypatch.setattr(boundary.client.experimental.sandboxes, 'create', failing_create)
    else:
        boundary.status_script = [original]
    with pytest.raises(ClientError) as caught:
        await boundary.backend().working_dir()
    assert caught.value is original


@pytest.mark.parametrize('operation_fails', [False, True])
async def test_client_close_failure_is_logged_without_changing_the_operation_outcome(
    owned_boundary: ScriptedSDK,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    operation_fails: bool,
) -> None:
    pool = owned_boundary.client.experimental.sandboxes.client.get_async_httpx_client()
    original_close = pool.aclose

    async def failed_close() -> None:
        await original_close()
        raise RuntimeError('private transport detail')

    monkeypatch.setattr(pool, 'aclose', failed_close)
    backend = RenderSandboxBackend()
    if operation_fails:
        owned_boundary.status = 'terminated'
        with pytest.raises(WorkspaceUnavailableError, match='terminated'):
            await backend.working_dir()
    else:
        assert await backend.working_dir() == '/workspace'
    assert 'Could not close the Render API client' in caplog.text
    assert 'private transport detail' not in caplog.text


async def test_client_close_deadline_is_logged_without_holding_up_the_operation(
    owned_boundary: ScriptedSDK, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    pool = owned_boundary.client.experimental.sandboxes.client.get_async_httpx_client()
    original_close = pool.aclose

    async def stalled_close() -> None:
        await original_close()
        await asyncio.Event().wait()

    monkeypatch.setattr(pool, 'aclose', stalled_close)
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._CLOSE_TIMEOUT', 0, raising=False)
    with no_leaked_tasks():
        async with asyncio.timeout(1):
            assert await RenderSandboxBackend().working_dir() == '/workspace'
    assert 'Closing the Render API client exceeded its deadline' in caplog.text


@pytest.mark.parametrize('cancellation', ['native', 'anyio'])
async def test_client_close_finishes_before_cancellation_propagates(
    owned_boundary: ScriptedSDK, monkeypatch: pytest.MonkeyPatch, cancellation: str
) -> None:
    pool = owned_boundary.client.experimental.sandboxes.client.get_async_httpx_client()
    original_close = pool.aclose
    started, finish, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def controlled_close() -> None:
        started.set()
        await finish.wait()
        await original_close()
        closed.set()

    monkeypatch.setattr(pool, 'aclose', controlled_close)
    scope = anyio.CancelScope()

    async def operation() -> None:
        with scope:
            await RenderSandboxBackend().working_dir()
            await anyio.lowlevel.checkpoint()

    with no_leaked_tasks():
        pending = asyncio.create_task(operation())
        try:
            async with asyncio.timeout(10):
                await started.wait()
            if cancellation == 'native':
                pending.cancel()
                await asyncio.sleep(0)
                pending.cancel()
            else:
                scope.cancel()
            await asyncio.sleep(0)
            assert not pending.done()
            finish.set()
            if cancellation == 'native':
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                await pending
                assert scope.cancelled_caught
            assert closed.is_set()
        finally:
            finish.set()
            await asyncio.gather(pending, return_exceptions=True)
            await original_close()


async def test_stop_deadline_logs_an_unconfirmed_stop_without_replacing_the_stream_error(
    boundary: ScriptedSDK, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    backend = boundary.backend()
    await backend.working_dir()
    original = ConnectionError('private stream detail')
    boundary.transport_error = original
    boundary.stop_gate = asyncio.Event()
    monkeypatch.setattr('pydantic_ai_harness.render_sandbox._commands._STOP_TIMEOUT', 0)
    with pytest.raises(ConnectionError) as caught:
        await backend.run(['side-effect'])
    assert caught.value is original
    assert 'Could not confirm command stop in Render sandbox sbx-scripted; it may still be running' in caplog.text
    assert 'private stream detail' not in caplog.text

"""Failure contracts at Render's public SDK boundary, through the public backend.

The scripted SDK supplies wire events and provider states, not adapter results.
This keeps parsing, command stopping, readiness, and cleanup code under test.
No credentials, network, or private Pydantic/Render interfaces are used.
"""

from __future__ import annotations

import base64
import json
import shlex
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render import RenderAsync
    from render.experimental.sandbox import (
        Sandbox,
        SandboxExecEvent,
        SandboxExecExit,
        SandboxExecOutput,
        SandboxNotFoundError,
    )

    from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef, WorkspaceUnavailableError
    from pydantic_ai_harness.render_sandbox import (
        RenderSandboxBackend,
        RenderSandboxCleanupError,
        RenderSandboxConfig,
        RenderSandboxEvent,
    )

pytestmark = pytest.mark.skipif(not imports_successful(), reason='render not installed')


@dataclass
class ScriptedSDK:
    """A public SDK double that can lose acknowledgements without replaying work."""

    client: RenderAsync
    status: str = 'running'
    creates: int = 0
    terminations: int = 0
    reads: int = 0
    operations: list[str] = field(default_factory=list[str])
    scenario: str = 'valid'
    stop_scenario: str = 'valid'
    state_after_terminate: str = 'terminated'
    status_script: list[str | Exception] = field(default_factory=list[str | Exception])
    transport_error: Exception | None = None
    observer_events: list[RenderSandboxEvent] = field(default_factory=list[RenderSandboxEvent])

    def sandbox(self, status: str) -> Sandbox:
        return Sandbox('sbx-scripted', status, 'starter', 'deny-all', 'oregon', 300, datetime.now(UTC))

    def backend(self, *, cleanup_timeout: float = 0.03) -> RenderSandboxBackend:
        # Tests vary deadlines by replacing this immutable config, not by sleeping
        # long enough to resemble a production timeout.
        config = RenderSandboxConfig(
            readiness_timeout=0.03,
            cleanup_timeout=cleanup_timeout,
            operation_timeout=1,
            stop_timeout=1,
            command_interval=0,
        )
        return RenderSandboxBackend(
            self.client,
            config=config,
            on_sandbox_event=self.observer_events.append,
        )

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

    async def terminate(self, identifier: str) -> None:
        assert identifier == 'sbx-scripted'
        self.terminations += 1
        self.status = self.state_after_terminate

    async def exec(self, identifier: str, launch: str) -> AsyncGenerator[SandboxExecEvent]:
        assert identifier == 'sbx-scripted'
        args = shlex.split(launch)
        operation = args[3]
        config = json.loads(args[4])
        nonce = config['nonce']
        self.operations.append(operation)

        def frame(kind: str, **fields: object) -> SandboxExecOutput:
            return SandboxExecOutput('stdout', json.dumps({'nonce': nonce, 'kind': kind, **fields}) + '\n')

        def output(text: str) -> SandboxExecOutput:
            return frame('output', stream='stdout', data=base64.b64encode(text.encode()).decode())

        if operation == 'stop':
            if self.stop_scenario == 'missing_ack':
                yield SandboxExecExit(0)
            else:
                yield frame('stopped')
                yield SandboxExecExit(0)
            return

        # The backend prepares its working directory using the same real parser.
        if config['cwd'] == '/':
            yield output('/workspace\n')
            yield frame('exit', code=0)
            yield SandboxExecExit(0)
            return

        async for event in self.response(nonce):
            yield event

    async def response(self, nonce: str) -> AsyncGenerator[SandboxExecEvent]:
        def frame(kind: str, **fields: object) -> SandboxExecOutput:
            return SandboxExecOutput('stdout', json.dumps({'nonce': nonce, 'kind': kind, **fields}) + '\n')

        def output(text: str) -> SandboxExecOutput:
            return frame('output', stream='stdout', data=base64.b64encode(text.encode()).decode())

        if self.transport_error is not None:
            yield output('partial')
            raise self.transport_error
        if self.scenario == 'valid':
            yield output('usable')
        elif self.scenario == 'invalid_json':
            yield SandboxExecOutput('stdout', '{broken\n')
            return
        elif self.scenario == 'invalid_base64':
            yield frame('output', stream='stdout', data='not base64!')
            return
        elif self.scenario == 'wrong_identity':
            yield SandboxExecOutput('stdout', '{"nonce":"unrelated","kind":"exit","code":0}\n')
            return
        elif self.scenario == 'oversized_frame':
            yield SandboxExecOutput('stdout', 'x' * 131_073)
            return
        elif self.scenario == 'incomplete_frame':
            yield SandboxExecOutput('stdout', '{"kind":')
            yield SandboxExecExit(0)
            return
        elif self.scenario == 'missing_direct_exit':
            yield output('partial')
            yield SandboxExecExit(0)
            return
        elif self.scenario == 'output_after_direct_exit':
            yield frame('exit', code=0)
            yield output('late')
            return
        else:
            assert self.scenario in {
                'missing_sdk_exit',
                'duplicate_sdk_exit',
                'output_after_sdk_exit',
            }
            yield output('partial')

        yield frame('exit', code=0)
        if self.scenario == 'missing_sdk_exit':
            return
        yield SandboxExecExit(0)
        if self.scenario == 'duplicate_sdk_exit':
            yield SandboxExecExit(0)
        elif self.scenario == 'output_after_sdk_exit':
            yield output('late')


@pytest.fixture
def boundary(monkeypatch: pytest.MonkeyPatch) -> ScriptedSDK:
    sdk = ScriptedSDK(RenderAsync(token='test-only', owner_id='tea-test'))
    public_api = sdk.client.experimental.sandboxes
    monkeypatch.setattr(public_api, 'create', sdk.create)
    monkeypatch.setattr(public_api, 'from_id', sdk.from_id)
    monkeypatch.setattr(public_api, 'terminate', sdk.terminate)
    monkeypatch.setattr(public_api, 'exec', sdk.exec)
    return sdk


@pytest.mark.parametrize(
    'scenario',
    [
        'invalid_json',
        'invalid_base64',
        'wrong_identity',
        'oversized_frame',
        'incomplete_frame',
        'missing_direct_exit',
        'missing_sdk_exit',
        'duplicate_sdk_exit',
        'output_after_direct_exit',
        'output_after_sdk_exit',
    ],
)
async def test_unusable_command_stream_never_becomes_success_or_replays(boundary: ScriptedSDK, scenario: str) -> None:
    backend = boundary.backend()
    await backend.working_dir()
    boundary.operations.clear()
    boundary.scenario = scenario
    with pytest.raises(WorkspaceError):
        await backend.run(['side-effect'])
    assert boundary.operations == ['run', 'stop']
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')
    assert boundary.terminations == 0
    assert not any(event.event == 'command_finished' for event in boundary.observer_events)

    boundary.scenario = 'valid'
    assert (await backend.run(['next-explicit-command'])).stdout == 'usable'
    assert boundary.creates == 1
    await backend.destroy()


async def test_stop_failure_preserves_original_stream_error_and_recovery_reference(
    boundary: ScriptedSDK,
) -> None:
    backend = boundary.backend()
    await backend.working_dir()
    boundary.operations.clear()
    original = ConnectionError('response lost after command submission')
    boundary.transport_error = original
    boundary.stop_scenario = 'missing_ack'
    with pytest.raises(ConnectionError) as caught:
        await backend.run(['side-effect'])
    assert caught.value is original
    assert isinstance(original.__cause__, WorkspaceError)
    assert any('stop was not confirmed' in note for note in original.__notes__)
    assert boundary.operations == ['run', 'stop']
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')
    assert boundary.terminations == 0
    await backend.destroy()


async def test_readiness_deadline_retains_id_and_does_not_create_again(
    boundary: ScriptedSDK,
) -> None:
    boundary.status = 'creating'
    backend = boundary.backend()
    with pytest.raises(WorkspaceUnavailableError) as caught:
        await backend.run(['must-not-start'])
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert boundary.operations == []
    assert boundary.creates == 1
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')
    assert boundary.terminations == 0

    # A later, explicit call can use the same provider resource once it is ready.
    boundary.status = 'running'
    assert (await backend.run(['explicit-recovery'])).stdout == 'usable'
    assert boundary.creates == 1
    await backend.destroy()


async def test_termination_ack_is_not_completion_and_retry_uses_retained_id(
    boundary: ScriptedSDK,
) -> None:
    backend = boundary.backend()
    await backend.working_dir()
    boundary.state_after_terminate = 'terminating'
    with pytest.raises(RenderSandboxCleanupError) as caught:
        await backend.destroy()
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')
    assert boundary.terminations == 1
    events = [event.event for event in boundary.observer_events]
    assert 'termination_acknowledged' in events
    assert 'cleanup_failed' in events
    assert 'terminated' not in events
    with pytest.raises(WorkspaceUnavailableError):
        await backend.run(['must-not-start'])

    boundary.state_after_terminate = 'terminated'
    await backend.destroy()
    assert boundary.terminations == 2
    assert boundary.creates == 1
    assert boundary.observer_events[-1].event == 'terminated'


@pytest.mark.parametrize('final', ['terminated', 'not_found'])
async def test_delayed_termination_waits_for_observed_terminal_state(boundary: ScriptedSDK, final: str) -> None:
    backend = boundary.backend(cleanup_timeout=1)
    await backend.working_dir()
    terminal = SandboxNotFoundError('gone') if final == 'not_found' else 'terminated'
    boundary.status_script = ['terminating', terminal]
    reads_before = boundary.reads
    await backend.destroy()
    assert boundary.reads == reads_before + 2
    assert boundary.status_script == []
    assert boundary.observer_events[-1].event == 'terminated'
    assert boundary.observer_events[-1].status == final
    assert boundary.terminations == 1
    await backend.destroy()
    assert boundary.terminations == 1


@pytest.mark.filterwarnings('error:Sandbox lifecycle observer raised an exception:RuntimeWarning')
async def test_observer_warning_as_error_cannot_interrupt_cleanup(boundary: ScriptedSDK) -> None:
    def broken_observer(event: RenderSandboxEvent) -> None:
        raise RuntimeError('Observer unavailable')

    backend = RenderSandboxBackend(
        boundary.client,
        config=RenderSandboxConfig(command_interval=0),
        on_sandbox_event=broken_observer,
    )
    await backend.working_dir()
    await backend.destroy()
    assert boundary.terminations == 1
    assert boundary.status == 'terminated'
    assert backend.ref == WorkspaceRef(provider='render', id='sbx-scripted')

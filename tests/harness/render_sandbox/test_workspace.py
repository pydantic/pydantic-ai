"""Workspace lifecycle against a fake public SDK and real local shell commands."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from itertools import pairwise

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render import RenderAsync
    from render.client.errors import ClientError, RateLimitError, ServerError
    from render.experimental.sandbox import Sandbox, SandboxNotFoundError

    from pydantic_ai.workspaces import (
        CommandResult,
        SupportsCommands,
        Workspace,
        WorkspaceBackend,
        WorkspaceCommand,
        WorkspaceOutputLimitError,
        WorkspaceRef,
        WorkspaceTimeoutError,
        WorkspaceUnavailableError,
    )
    from pydantic_ai_harness.render_sandbox import RenderSandboxCleanupError, RenderSandboxConfig

    from .workspace_helpers import WorkspaceHarness

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='render not installed'),
    pytest.mark.subprocess(reason='Exercise real POSIX workspace commands and file helper processes', conditional=True),
]


async def test_lazy_and_concurrent_first_use_share_one_sandbox(
    workspace_harness: WorkspaceHarness,
) -> None:
    backend = workspace_harness.backend()
    assert isinstance(backend, WorkspaceBackend)
    assert isinstance(backend, SupportsCommands)
    assert backend.ref is None
    assert workspace_harness.created == []
    results = await asyncio.gather(*(backend.run(['printf', str(i)]) for i in range(8)))
    assert [result.stdout for result in results] == [str(i) for i in range(8)]
    assert workspace_harness.created == ['sb-1']
    assert backend.ref == WorkspaceRef(provider='render', id='sb-1')
    assert workspace_harness.terminated == []
    await backend.destroy()
    assert backend.ref == WorkspaceRef(provider='render', id='sb-1')


async def test_command_form_cwd_env_and_nonzero_result(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    for command, shell in [('true', False), (['true'], True)]:
        with pytest.raises(TypeError):
            await backend.run(command, shell=shell)
    with pytest.raises(ValueError):
        await backend.run([])
    assert workspace_harness.created == []
    literal = 'literal $() `quoted`; &&'
    assert (await backend.run(['printf', '%s', literal])).stdout == literal
    result = await backend.run(
        'printf \'%s\' "$ONLY_TEST_ENV"; printf err >&2; exit 7',
        shell=True,
        env={'ONLY_TEST_ENV': 'value'},
    )
    assert (result.exit_code, result.stdout, result.stderr) == (7, 'value', 'err')
    assert (await backend.run(['pwd', '-P'])).stdout.strip() == await backend.working_dir()


async def test_command_arguments_are_copied_before_sandbox_creation(workspace_harness: WorkspaceHarness) -> None:
    # Hold sandbox creation at the fake SDK boundary so another caller can edit
    # its command list while this operation is waiting for the remote environment.
    backend = workspace_harness.backend()
    command = ['printf', '%s', 'original']
    workspace_harness.create_gate = asyncio.Event()
    try:
        # The task group waits for the command on every exit. Opening the gate in
        # finally also lets acquisition finish if the readiness assertion fails.
        async with asyncio.TaskGroup() as tasks:
            pending = tasks.create_task(backend.run(command))
            try:
                async with asyncio.timeout(10):
                    await workspace_harness.create_started.wait()
                command[-1] = 'changed'
            finally:
                workspace_harness.create_gate.set()
        # The command must use the arguments supplied when the operation began.
        assert pending.result().stdout == 'original'
    finally:
        # This test owns its backend, including when the regression assertion fails.
        await backend.destroy()


@pytest.mark.parametrize(
    ('timeout', 'env'),
    [
        (0, None),
        (-1, None),
        (float('inf'), None),
        (float('nan'), None),
        (None, {'BAD=NAME': 'value'}),
        (None, {'KEY': 'nul\0value'}),
    ],
)
async def test_invalid_command_settings_create_no_sandbox(
    workspace_harness: WorkspaceHarness, timeout: float | None, env: dict[str, str] | None
) -> None:
    # Invalid deadlines and environment values must fail before any paid resource
    # is created, even when command validation moves between internal helpers.
    backend = workspace_harness.backend()
    with pytest.raises(ValueError):
        await backend.run(['true'], timeout=timeout, env=env)
    assert backend.ref is None
    assert workspace_harness.created == []


async def test_native_filesystem_handles_binary_large_files_and_errors(
    workspace_harness: WorkspaceHarness,
) -> None:
    backend = workspace_harness.backend()
    workspace = Workspace(backend)
    payload = bytes(range(256)) * 2048
    await workspace.write_bytes('nested/binary.dat', payload)
    assert await workspace.read_bytes('nested/binary.dat') == payload
    assert await workspace.exists('nested/binary.dat')
    assert not await workspace.exists('missing')
    with pytest.raises(FileNotFoundError):
        await workspace.read_bytes('missing')
    assert (await workspace.list_dir('nested'))[0].name == 'binary.dat'
    await workspace.remove('nested')
    assert not await workspace.exists('nested')


async def test_attachment_never_recreates_and_explicit_destroy_retains_ref(
    workspace_harness: WorkspaceHarness,
) -> None:
    owned = workspace_harness.backend()
    await owned.working_dir()
    assert owned.ref is not None
    attached = workspace_harness.backend(owned.ref)
    assert not attached.owned
    assert await attached.working_dir() == await owned.working_dir()
    assert len(workspace_harness.created) == 1
    await attached.destroy()
    assert attached.ref == owned.ref
    with pytest.raises(WorkspaceUnavailableError):
        await workspace_harness.backend(owned.ref).working_dir()
    assert len(workspace_harness.created) == 1


async def test_cancelled_create_keeps_ref_without_automatic_destruction(
    workspace_harness: WorkspaceHarness,
) -> None:
    workspace_harness.create_gate = asyncio.Event()
    backend = workspace_harness.backend()
    task = asyncio.create_task(backend.working_dir())
    await workspace_harness.create_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    workspace_harness.create_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.ref == WorkspaceRef(provider='render', id='sb-1')
    assert workspace_harness.terminated == []
    assert await backend.working_dir() == workspace_harness.config.working_dir
    assert len(workspace_harness.created) == 1
    await backend.destroy()


@pytest.mark.parametrize('status', ['errored', 'terminated'])
async def test_readiness_failure_retains_owned_ref(workspace_harness: WorkspaceHarness, status: str) -> None:
    workspace_harness.create_status = status
    backend = workspace_harness.backend()
    with pytest.raises(WorkspaceUnavailableError):
        await backend.working_dir()
    assert backend.ref is not None
    assert workspace_harness.terminated == []
    await backend.destroy()


@pytest.mark.parametrize('error', [ClientError('authentication rejected'), SandboxNotFoundError('gone')])
async def test_terminal_errors_do_not_replay_commands(workspace_harness: WorkspaceHarness, error: Exception) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    workspace_harness.command_error = error
    before = len(workspace_harness.commands)
    with pytest.raises(WorkspaceUnavailableError):
        await backend.run(['check'])
    assert len(workspace_harness.commands) == before + 1


@pytest.mark.parametrize(
    'error',
    [
        WorkspaceTimeoutError('late', stdout='partial'),
        WorkspaceOutputLimitError('full', limit=4, stdout='data'),
    ],
)
async def test_command_limits_leave_backend_available(workspace_harness: WorkspaceHarness, error: Exception) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    workspace_harness.command_error = error
    with pytest.raises(type(error)):
        await backend.run(['check'])
    workspace_harness.command_error = None
    assert (await backend.run(['printf', 'still alive'])).stdout == 'still alive'
    assert workspace_harness.terminated == []


async def test_destroy_interrupts_active_operation_without_waiting_for_its_caller(
    workspace_harness: WorkspaceHarness,
) -> None:
    backend = workspace_harness.backend()
    workspace_harness.block_commands = True
    task = asyncio.create_task(backend.run(['blocked']))
    await workspace_harness.command_started.wait()
    await asyncio.wait_for(backend.destroy(), 1)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert workspace_harness.terminated == ['sb-1']
    with pytest.raises(WorkspaceUnavailableError):
        await backend.run(['true'])


async def test_cleanup_failure_is_explicit_and_can_be_retried(
    workspace_harness: WorkspaceHarness,
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    workspace_harness.terminate_error = ClientError('denied')
    with pytest.raises(RenderSandboxCleanupError):
        await backend.destroy()
    assert backend.ref is not None
    workspace_harness.terminate_error = None
    await backend.destroy()
    assert workspace_harness.terminated == ['sb-1']


async def test_create_auth_failure_does_not_claim_unknown_resource(
    workspace_harness: WorkspaceHarness,
) -> None:
    workspace_harness.create_error = ClientError('auth')
    backend = workspace_harness.backend()
    with pytest.raises(WorkspaceUnavailableError):
        await backend.working_dir()
    assert backend.ref is None
    await backend.destroy()
    assert workspace_harness.created == []


def test_config_copies_env_and_redacts_values() -> None:
    values = {'PRIVATE': 'secret-value'}
    config = RenderSandboxConfig(env=values)
    values['PRIVATE'] = 'changed'
    assert config.env == {'PRIVATE': 'secret-value'}
    assert 'secret-value' not in repr(config)
    with pytest.raises(ValueError):
        RenderSandboxConfig(max_output_bytes=0)
    with pytest.raises(ValueError):
        RenderSandboxConfig(working_dir='relative')
    with pytest.raises(ValueError):
        RenderSandboxConfig(stop_timeout=float('inf'))


@pytest.mark.parametrize('error', [ServerError('response lost'), TimeoutError('no response')])
async def test_ambiguous_create_is_not_replayed_and_cleanup_reports_uncertainty(
    workspace_harness: WorkspaceHarness, error: Exception
) -> None:
    workspace_harness.create_error = error
    backend = workspace_harness.backend()
    with pytest.raises(type(error)):
        await backend.working_dir()
    workspace_harness.create_error = None
    with pytest.raises(type(error)):
        await backend.working_dir()
    assert workspace_harness.created == []
    with pytest.raises(RenderSandboxCleanupError, match='outcome is unknown'):
        await backend.destroy()


async def test_chained_client_create_error_is_conservatively_uncertain(
    workspace_harness: WorkspaceHarness,
) -> None:
    error = ClientError('request failed')
    error.__cause__ = OSError('transport failure')
    workspace_harness.create_error = error
    backend = workspace_harness.backend()
    with pytest.raises(WorkspaceUnavailableError, match='outcome is unknown'):
        await backend.working_dir()
    with pytest.raises(RenderSandboxCleanupError, match='outcome is unknown'):
        await backend.destroy()


async def test_interrupted_stream_during_termination_is_unavailable(
    workspace_harness: WorkspaceHarness,
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    assert backend.ref is not None
    workspace_harness.command_error = RuntimeError('stream lost')

    # Simulate status changing after exec starts, before the backend audits it.
    async def fail(*args: object, **kwargs: object) -> CommandResult:
        assert backend.ref is not None
        workspace_harness.sandboxes[backend.ref.id].status = 'terminating'
        raise RuntimeError('stream lost')

    original = workspace_harness.execute
    workspace_harness.execute = fail
    try:
        with pytest.raises(WorkspaceUnavailableError, match='terminating'):
            await backend.run(['true'])
    finally:
        workspace_harness.execute = original
    assert workspace_harness.created == ['sb-1']


async def test_repeated_native_cancellation_waits_for_explicit_cleanup(
    workspace_harness: WorkspaceHarness,
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    workspace_harness.terminate_gate = asyncio.Event()
    task = asyncio.create_task(backend.destroy())
    await workspace_harness.terminate_started.wait()
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    workspace_harness.terminate_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert workspace_harness.terminated == ['sb-1']
    await backend.destroy()
    assert workspace_harness.terminated == ['sb-1']


@pytest.mark.parametrize('phase', ['create', 'readiness', 'run', 'status_audit'])
async def test_rate_limit_is_propagated_unchanged_without_replay(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    backend = workspace_harness.backend()
    rate_limit = RateLimitError('try later')
    if phase == 'create':
        workspace_harness.create_error = rate_limit
    elif phase == 'run':
        workspace_harness.command_error = rate_limit
    else:
        if phase == 'status_audit':
            await backend.working_dir()
            workspace_harness.command_error = RuntimeError('stream interrupted')

        async def limited(identifier: str) -> Sandbox:
            raise rate_limit

        monkeypatch.setattr(workspace_harness.client.experimental.sandboxes, 'from_id', limited)
    with pytest.raises(RateLimitError) as caught:
        await backend.run(['printf', 'never replay'])
    assert caught.value is rate_limit
    assert workspace_harness.commands.count(['printf', 'never replay']) <= 1


async def test_normal_dispatch_is_paced_across_backends_sharing_client(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    interval = 0.03
    workspace_harness.config = replace(workspace_harness.config, command_interval=interval)
    first = workspace_harness.backend()
    second = workspace_harness.backend()
    dispatches: list[float] = []
    command_started = asyncio.Event()
    release = asyncio.Event()

    async def record(
        client: RenderAsync,
        sandbox_id: str,
        command: WorkspaceCommand,
        **kwargs: object,
    ) -> CommandResult:
        dispatches.append(asyncio.get_running_loop().time())
        if kwargs['working_dir'] == '/':
            return CommandResult(exit_code=0, stdout=workspace_harness.config.working_dir + '\n', stderr='')
        if command == ['long']:
            command_started.set()
            await release.wait()
        return CommandResult(exit_code=0, stdout='done', stderr='')

    monkeypatch.setattr(workspace_harness, 'execute', record)
    await asyncio.gather(first.working_dir(), second.working_dir())
    long = asyncio.create_task(first.run(['long']))
    await command_started.wait()
    # Dispatch pacing must not lock the long-running command's entire lifetime.
    assert (await second.run(['short'])).stdout == 'done'
    assert not long.done()
    release.set()
    await long
    assert len(dispatches) == 4
    assert all(later - earlier >= interval - 0.001 for earlier, later in pairwise(dispatches))


async def test_pacing_wait_does_not_spend_command_deadline(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    interval = 0.05
    workspace_harness.config = replace(workspace_harness.config, command_interval=interval)
    backend = workspace_harness.backend()
    await backend.working_dir()
    timeouts: list[object] = []

    async def fast(*args: object, **kwargs: object) -> CommandResult:
        timeouts.append(kwargs['timeout'])
        return CommandResult(exit_code=0, stdout='done', stderr='')

    monkeypatch.setattr(workspace_harness, 'execute', fast)
    # Establish the preceding start with the immediate fake, independent of how
    # long the real local setup command took on this host.
    await backend.run(['establish dispatch'])
    timeouts.clear()
    started = asyncio.get_running_loop().time()
    assert (await backend.run(['quick'], timeout=0.001)).stdout == 'done'
    assert asyncio.get_running_loop().time() - started >= interval - 0.01
    assert timeouts == [0.001]


async def test_cancellation_in_dispatch_queue_sends_no_command_and_cleanup_bypasses_queue(
    workspace_harness: WorkspaceHarness,
) -> None:
    workspace_harness.config = replace(workspace_harness.config, command_interval=10)
    backend = workspace_harness.backend()
    await backend.working_dir()
    dispatched = len(workspace_harness.commands)
    task = asyncio.create_task(backend.run(['must not start']))
    await asyncio.sleep(0)
    assert not task.done()
    # destroy() cancels a queued caller promptly; no slot or remote command is consumed.
    await asyncio.wait_for(backend.destroy(), 0.5)
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(workspace_harness.commands) == dispatched
    assert workspace_harness.terminated == ['sb-1']


@pytest.mark.parametrize('interval', [-1, float('inf'), float('nan'), True])
def test_invalid_command_interval_is_rejected(interval: float) -> None:
    with pytest.raises(ValueError, match='command_interval'):
        RenderSandboxConfig(command_interval=interval)

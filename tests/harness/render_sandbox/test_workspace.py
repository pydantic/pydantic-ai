"""Workspace lifecycle against a fake public SDK and real local shell commands."""

from __future__ import annotations

import asyncio
import json
import shlex
from collections.abc import AsyncIterator

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from render import RenderAsync
    from render.client.errors import ClientError, RateLimitError, ServerError
    from render.experimental.sandbox import Sandbox, SandboxExecEvent, SandboxNotFoundError

    from pydantic_ai.exceptions import UserError
    from pydantic_ai.workspaces import (
        CommandResult,
        SupportsCommands,
        Workspace,
        WorkspaceBackend,
        WorkspaceCommand,
        WorkspaceRef,
        WorkspaceUnavailableError,
    )
    from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

    from .workspace_helpers import WorkspaceHarness, client_is_closed, use_host_credentials

pytestmark = pytest.mark.skipif(not imports_successful(), reason='render not installed')


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


async def test_configured_env_reaches_every_command_under_command_env(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend(env={'BASE': 'base', 'SHARED': 'configured'})
    result = await backend.run('printf %s-%s "$BASE" "$SHARED"', shell=True, env={'SHARED': 'command'})
    assert result.stdout == 'base-command'


async def test_command_arguments_are_copied_before_sandbox_creation(workspace_harness: WorkspaceHarness) -> None:
    # Hold sandbox creation at the fake SDK boundary so another caller can edit
    # its command list while this operation is waiting for the remote environment.
    backend = workspace_harness.backend()
    command = ['printf', '%s', 'original']
    workspace_harness.create_gate = asyncio.Event()
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


@pytest.mark.parametrize(
    ('timeout', 'env', 'error'),
    [
        (0, None, ValueError),
        (-1, None, ValueError),
        (float('inf'), None, ValueError),
        (float('nan'), None, ValueError),
        (None, {'BAD=NAME': 'value'}, ValueError),
        (None, {'': 'value'}, ValueError),
        (None, {'KEY': 'nul\0value'}, ValueError),
        (None, {1: 'value'}, TypeError),
        (None, {'name': 1}, TypeError),
    ],
)
async def test_invalid_command_settings_create_no_sandbox(
    workspace_harness: WorkspaceHarness, timeout: float | None, env: dict[str, str] | None, error: type[Exception]
) -> None:
    # Invalid deadlines and environment values must fail before any paid resource is created.
    backend = workspace_harness.backend()
    with pytest.raises(error):
        await backend.run(['true'], timeout=timeout, env=env)
    assert backend.ref is None
    assert workspace_harness.created == []


@pytest.mark.parametrize(
    ('env', 'error'),
    [({1: 'value'}, TypeError), ({'name': 1}, TypeError), ({'A=B': 'value'}, ValueError), ({'K': '\0'}, ValueError)],
)
def test_invalid_configured_env_is_rejected_at_construction(
    workspace_harness: WorkspaceHarness, env: dict[str, str], error: type[Exception]
) -> None:
    with pytest.raises(error):
        workspace_harness.backend(env=env)


async def test_commands_run_in_the_resolved_working_dir(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `working_dir()` reports the resolved path, so commands must start there even if the symlink moves.
    real = workspace_harness.root / 'real'
    real.mkdir()
    link = workspace_harness.root / 'link'
    link.symlink_to(real)
    sdk = workspace_harness.client.experimental.sandboxes
    execute = sdk.exec
    launched: list[str] = []

    def recording_exec(identifier: str, launch: str) -> AsyncIterator[SandboxExecEvent]:
        launched.append(json.loads(shlex.split(launch)[5])['cwd'])
        return execute(identifier, launch)

    monkeypatch.setattr(sdk, 'exec', recording_exec)
    backend = workspace_harness.backend(working_dir=str(link))
    await backend.run(['true'])
    assert await backend.working_dir() == str(real.resolve())
    assert launched[-1] == str(real.resolve())


async def test_cancelled_create_keeps_ref_without_destroying(workspace_harness: WorkspaceHarness) -> None:
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
    assert await backend.working_dir() == str(workspace_harness.root)
    assert len(workspace_harness.created) == 1


@pytest.mark.parametrize('status', ['errored', 'terminated'])
async def test_readiness_failure_retains_created_ref(workspace_harness: WorkspaceHarness, status: str) -> None:
    workspace_harness.create_status = status
    backend = workspace_harness.backend()
    with pytest.raises(WorkspaceUnavailableError, match=status):
        await backend.working_dir()
    assert backend.ref == WorkspaceRef(provider='render', id='sb-1')
    assert workspace_harness.terminated == []


async def test_readiness_retry_after_create_still_makes_working_dir(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = workspace_harness.client.experimental.sandboxes.from_id
    failures = [ServerError('status unavailable')]

    async def flaky(identifier: str) -> Sandbox:
        if failures:
            raise failures.pop()
        return await original(identifier)

    monkeypatch.setattr(workspace_harness.client.experimental.sandboxes, 'from_id', flaky)
    working_dir = workspace_harness.root / 'project'
    backend = workspace_harness.backend(working_dir=str(working_dir))
    with pytest.raises(ServerError):
        await backend.working_dir()
    assert backend.ref == WorkspaceRef(provider='render', id='sb-1')
    assert not working_dir.exists()
    assert await backend.working_dir() == str(working_dir)
    assert working_dir.is_dir()
    assert workspace_harness.created == ['sb-1']


async def test_readiness_polls_until_running(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = ['creating', 'creating', 'running']
    original = workspace_harness.client.experimental.sandboxes.from_id

    async def warming(identifier: str) -> Sandbox:
        sandbox = await original(identifier)
        sandbox.status = statuses.pop(0)
        return sandbox

    monkeypatch.setattr(workspace_harness.client.experimental.sandboxes, 'from_id', warming)
    backend = workspace_harness.backend()
    assert await backend.working_dir() == str(workspace_harness.root)
    assert statuses == []


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
    ('error', 'expected'),
    [
        (ClientError('auth'), WorkspaceUnavailableError),
        (TimeoutError('no response'), WorkspaceUnavailableError),
        (RateLimitError('try later'), RateLimitError),
        (ServerError('unavailable'), ServerError),
    ],
)
async def test_failed_create_lets_a_later_operation_create(
    workspace_harness: WorkspaceHarness, error: Exception, expected: type[Exception]
) -> None:
    # As on E2B: a failed create records no sandbox, so the next operation tries again.
    workspace_harness.create_error = error
    backend = workspace_harness.backend()
    with pytest.raises(expected):
        await backend.working_dir()
    assert backend.ref is None
    workspace_harness.create_error = None
    await backend.working_dir()
    assert backend.ref == WorkspaceRef(provider='render', id='sb-1')


async def test_interrupted_stream_on_a_stopped_sandbox_is_unavailable(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()

    # Simulate status changing after exec starts, before the backend audits it.
    async def fail(*args: object, **kwargs: object) -> CommandResult:
        workspace_harness.sandboxes['sb-1'].status = 'terminating'
        raise RuntimeError('stream lost')

    workspace_harness.execute = fail
    with pytest.raises(WorkspaceUnavailableError, match='terminating'):
        await backend.run(['true'])
    assert workspace_harness.created == ['sb-1']


@pytest.mark.parametrize(
    'probe_error', [RateLimitError('try later'), ServerError('status unavailable'), ClientError('denied')], ids=repr
)
async def test_failed_status_probe_reraises_the_stream_error(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch, probe_error: Exception
) -> None:
    # The command may have run, so a probe failure must not surface as a retryable error of its own.
    backend = workspace_harness.backend()
    await backend.working_dir()
    stream_error = RuntimeError('stream lost')
    workspace_harness.command_error = stream_error

    async def failing(identifier: str) -> Sandbox:
        raise probe_error

    monkeypatch.setattr(workspace_harness.client.experimental.sandboxes, 'from_id', failing)
    with pytest.raises(RuntimeError) as caught:
        await backend.run(['true'])
    assert caught.value is stream_error
    assert caught.value.__cause__ is probe_error


async def test_interrupted_stream_on_a_running_sandbox_propagates(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    workspace_harness.command_error = RuntimeError('stream lost')
    with pytest.raises(RuntimeError, match='stream lost'):
        await backend.run(['true'])


async def test_rate_limit_during_readiness_propagates_unchanged(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = workspace_harness.backend()
    rate_limit = RateLimitError('try later')

    async def limited(identifier: str) -> Sandbox:
        raise rate_limit

    monkeypatch.setattr(workspace_harness.client.experimental.sandboxes, 'from_id', limited)
    with pytest.raises(RateLimitError) as caught:
        await backend.run(['printf', 'never sent'])
    assert caught.value is rate_limit
    assert workspace_harness.commands == []


@pytest.mark.parametrize(
    ('command', 'error'),
    [([1], TypeError), (b'not argv', TypeError), (['printf', '\0'], ValueError)],
)
async def test_invalid_command_shape_is_rejected_before_creation(
    workspace_harness: WorkspaceHarness, command: WorkspaceCommand, error: type[Exception]
) -> None:
    backend = workspace_harness.backend()
    with pytest.raises(error):
        await backend.run(command)
    assert workspace_harness.created == []


async def test_oversized_command_creates_no_sandbox(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    with pytest.raises(ValueError, match='exec argument limit'):
        await backend.run(['printf', 'x' * 130_000])
    assert workspace_harness.created == []


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_trio_is_refused_before_any_render_call(workspace_harness: WorkspaceHarness) -> None:
    with pytest.raises(UserError, match='asyncio event loop'):
        await workspace_harness.backend().working_dir()
    assert workspace_harness.created == []


def test_backend_rejects_a_relative_working_dir() -> None:
    with pytest.raises(ValueError, match='working_dir'):
        RenderSandboxBackend(working_dir='relative')


async def test_missing_attachment_does_not_create_replacement(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend(WorkspaceRef(provider='render', id='missing'))
    with pytest.raises(WorkspaceUnavailableError):
        await backend.working_dir()
    assert workspace_harness.created == []
    assert backend.ref == WorkspaceRef(provider='render', id='missing')


async def test_attached_sandbox_without_working_dir_is_unavailable(workspace_harness: WorkspaceHarness) -> None:
    owned = workspace_harness.backend()
    await owned.working_dir()
    assert owned.ref is not None
    # An attached sandbox is not prepared: its working directory must already exist.
    attached = RenderSandboxBackend(
        client=workspace_harness.client, ref=owned.ref, working_dir=str(workspace_harness.root / 'absent')
    )
    with pytest.raises(WorkspaceUnavailableError, match='is unavailable in the sandbox'):
        await attached.working_dir()


async def test_each_operation_opens_and_closes_its_own_client(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_host_credentials(workspace_harness, monkeypatch)
    backend = RenderSandboxBackend(working_dir=str(workspace_harness.root))
    path = str(workspace_harness.root / 'data.txt')
    await backend.write_bytes(path, b'contents')
    # The write's readiness check and helper commands ran on that one operation's client.
    assert len(workspace_harness.opened_clients) == 1
    assert await backend.read_bytes(path) == b'contents'
    assert len(workspace_harness.opened_clients) == 2
    assert all(client_is_closed(client) for client in workspace_harness.opened_clients)


async def test_stop_after_cancellation_runs_on_its_own_open_client(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stop can outlive the cancelled operation, so it never uses that operation's client."""
    use_host_credentials(workspace_harness, monkeypatch)
    execs: list[tuple[str, RenderAsync, bool]] = []
    install = workspace_harness.install

    def recording_install(client: RenderAsync) -> None:
        install(client)
        sdk = client.experimental.sandboxes
        execute = sdk.exec

        def record(identifier: str, launch: str) -> AsyncIterator[SandboxExecEvent]:
            operation = shlex.split(launch)[4]
            execs.append((operation, client, client_is_closed(client)))
            return execute(identifier, launch)

        monkeypatch.setattr(sdk, 'exec', record)

    workspace_harness.install = recording_install
    backend = RenderSandboxBackend(working_dir=str(workspace_harness.root))
    await backend.working_dir()
    execs.clear()
    marker = workspace_harness.root / 'started'
    task = asyncio.create_task(backend.run(['sh', '-c', 'touch "$1"; exec sleep 60', 'sh', str(marker)]))
    # The command is running once it has written its marker; the deadline only guards against a hang.
    async with asyncio.timeout(10):
        while not marker.exists():
            await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (run_op, run_client, _), (stop_op, stop_client, stop_client_closed) = execs
    assert (run_op, stop_op) == ('run', 'stop')
    assert stop_client is not run_client
    assert not stop_client_closed
    assert all(client_is_closed(client) for client in workspace_harness.opened_clients)


@pytest.mark.parametrize('missing', ['RENDER_API_KEY', 'RENDER_WORKSPACE_ID'])
async def test_missing_credentials_fail_on_first_use(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    monkeypatch.setenv('RENDER_API_KEY', 'test-only')
    monkeypatch.setenv('RENDER_WORKSPACE_ID', 'tea-test')
    monkeypatch.delenv(missing)
    backend = RenderSandboxBackend()
    with pytest.raises(WorkspaceUnavailableError, match='RENDER_API_KEY and RENDER_WORKSPACE_ID'):
        await backend.working_dir()
    assert workspace_harness.created == []


@pytest.mark.subprocess(reason="The claim is about the real interpreter's import path under `python3 -I -c`")
async def test_project_module_does_not_shadow_file_helper_imports(workspace_harness: WorkspaceHarness) -> None:
    # The helper runs inline in the agent's working directory; a project's own `json.py`
    # there must not replace the stdlib module the helper imports.
    workspace_harness.inline_helpers = True
    (workspace_harness.root / 'json.py').write_text("raise SystemExit('project json.py was imported')\n")
    (workspace_harness.root / 'data.txt').write_text('contents')
    workspace = Workspace(workspace_harness.backend())
    assert await workspace.read_text('data.txt') == 'contents'
    await workspace.write_text('out.txt', 'written')
    assert (workspace_harness.root / 'out.txt').read_text() == 'written'

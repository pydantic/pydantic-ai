"""Workspace lifecycle against a fake public SDK and real local shell commands."""

from __future__ import annotations

import asyncio

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
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
    from pydantic_ai_harness.render_sandbox import RenderSandboxBackend

    from .workspace_helpers import WorkspaceHarness, client_is_closed, use_host_credentials

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


async def test_command_form_cwd_env_and_nonzero_result(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    for command, shell in [('true', False), (['true'], True)]:
        with pytest.raises(TypeError):
            await backend.run(command, shell=shell)
    with pytest.raises(TypeError):
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


async def test_configured_env_reaches_create_and_every_command(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend(env={'BASE': 'base', 'SHARED': 'configured'})
    result = await backend.run('printf %s-%s "$BASE" "$SHARED"', shell=True, env={'SHARED': 'command'})
    assert result.stdout == 'base-command'
    assert workspace_harness.create_calls['sb-1'].env == {'BASE': 'base', 'SHARED': 'configured'}


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
    ('timeout', 'env'),
    [
        (0, None),
        (-1, None),
        (float('inf'), None),
        (float('nan'), None),
        (None, {'BAD=NAME': 'value'}),
        (None, {'': 'value'}),
        (None, {'KEY': 'nul\0value'}),
    ],
)
async def test_invalid_command_settings_create_no_sandbox(
    workspace_harness: WorkspaceHarness, timeout: float | None, env: dict[str, str] | None
) -> None:
    # Invalid deadlines and environment values must fail before any paid resource is created.
    backend = workspace_harness.backend()
    with pytest.raises(ValueError):
        await backend.run(['true'], timeout=timeout, env=env)
    assert backend.ref is None
    assert workspace_harness.created == []


@pytest.mark.parametrize('env', [{1: 'value'}, {'name': 1}])
async def test_non_string_command_env_creates_no_sandbox(
    workspace_harness: WorkspaceHarness, env: dict[str, str]
) -> None:
    backend = workspace_harness.backend()
    with pytest.raises(TypeError, match='env keys and values must be strings'):
        await backend.run(['true'], env=env)
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
    assert (await workspace.stat('nested')).is_dir
    await workspace.make_dir('made/deeper')
    assert await workspace.exists('made/deeper')
    await workspace.remove('nested')
    assert not await workspace.exists('nested')


async def test_attachment_never_recreates(workspace_harness: WorkspaceHarness) -> None:
    owned = workspace_harness.backend()
    await owned.working_dir()
    assert owned.ref is not None
    attached = workspace_harness.backend(owned.ref)
    assert await attached.working_dir() == await owned.working_dir()
    assert attached.ref == owned.ref
    assert len(workspace_harness.created) == 1


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


@pytest.mark.parametrize('error', [ClientError('auth'), RateLimitError('try later')])
async def test_certain_create_failure_lets_a_later_operation_create(
    workspace_harness: WorkspaceHarness, error: Exception
) -> None:
    # A rejected or rate-limited create made nothing, so trying again cannot leave two sandboxes.
    workspace_harness.create_error = error
    backend = workspace_harness.backend()
    with pytest.raises((WorkspaceUnavailableError, RateLimitError)):
        await backend.working_dir()
    assert backend.ref is None
    workspace_harness.create_error = None
    await backend.working_dir()
    assert workspace_harness.created == ['sb-1']
    assert backend.ref == WorkspaceRef(provider='render', id='sb-1')


@pytest.mark.parametrize('error', [ServerError('response lost'), TimeoutError('no response')])
async def test_ambiguous_create_is_not_replayed(workspace_harness: WorkspaceHarness, error: Exception) -> None:
    workspace_harness.create_error = error
    backend = workspace_harness.backend()
    with pytest.raises(type(error)):
        await backend.working_dir()
    workspace_harness.create_error = None
    with pytest.raises(type(error)):
        await backend.working_dir()
    assert workspace_harness.created == []


async def test_chained_client_create_error_is_reported_as_uncertain(workspace_harness: WorkspaceHarness) -> None:
    error = ClientError('request failed')
    error.__cause__ = OSError('transport failure')
    workspace_harness.create_error = error
    backend = workspace_harness.backend()
    with pytest.raises(WorkspaceUnavailableError, match='outcome is unknown'):
        await backend.working_dir()
    workspace_harness.create_error = None
    # The create may have succeeded, so the backend never creates again.
    with pytest.raises(WorkspaceUnavailableError, match='outcome is unknown'):
        await backend.working_dir()
    assert workspace_harness.created == []


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


async def test_interrupted_stream_on_a_running_sandbox_propagates(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    workspace_harness.command_error = RuntimeError('stream lost')
    with pytest.raises(RuntimeError, match='stream lost'):
        await backend.run(['true'])


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


def test_backend_rejects_a_foreign_ref(workspace_harness: WorkspaceHarness) -> None:
    with pytest.raises(ValueError, match="expected 'render'"):
        workspace_harness.backend(WorkspaceRef(provider='other', id='sb-1'))


def test_backend_rejects_a_relative_working_dir() -> None:
    with pytest.raises(ValueError, match='working_dir'):
        RenderSandboxBackend(working_dir='relative')


async def test_cancelled_creation_preserves_provider_failure_as_cause(
    workspace_harness: WorkspaceHarness,
) -> None:
    # Cancel while creation is pending, then fail that bounded create operation.
    # The caller keeps cancellation priority and can still inspect the provider error.
    original = ConnectionError('create response lost')
    workspace_harness.create_gate = asyncio.Event()
    workspace_harness.create_error = original
    backend = workspace_harness.backend()
    pending = asyncio.create_task(backend.working_dir())
    await workspace_harness.create_started.wait()
    pending.cancel()
    await asyncio.sleep(0)
    workspace_harness.create_gate.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await pending
    assert caught.value.__cause__ is original
    assert backend.ref is None
    assert workspace_harness.created == []


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


async def test_local_command_cancellation_reaps_its_process(
    workspace_harness: WorkspaceHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = workspace_harness.backend()
    await backend.working_dir()
    original = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []
    spawned = asyncio.Event()

    async def spawn(
        program: str,
        *args: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        stdin: int | None = None,
        stdout: int | None = None,
        stderr: int | None = None,
    ) -> asyncio.subprocess.Process:
        process = await original(program, *args, cwd=cwd, env=env, stdin=stdin, stdout=stdout, stderr=stderr)
        processes.append(process)
        spawned.set()
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', spawn)
    pending = asyncio.create_task(backend.run(['sleep', '60']))
    try:
        async with asyncio.timeout(10):
            await spawned.wait()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert processes[0].returncode is not None
    finally:
        if not pending.done():  # pragma: no cover - independent cleanup if the cancellation assertion regresses
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        for process in processes:
            if process.returncode is None:  # pragma: no cover - independent regression cleanup
                process.kill()
                await process.wait()


async def test_ordinary_inline_python_command_keeps_its_arguments(workspace_harness: WorkspaceHarness) -> None:
    backend = workspace_harness.backend()
    assert (await backend.run(['python3', '-c', 'print("ordinary")'])).stdout == 'ordinary\n'
    assert workspace_harness.commands[-1] == ['python3', '-c', 'print("ordinary")']


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

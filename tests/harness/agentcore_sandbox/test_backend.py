"""`AgentCoreSandboxBackend` behavior over the fake client, beyond what the conformance suite checks."""

from __future__ import annotations

import asyncio
import contextlib
import os
import re
import uuid
from collections.abc import Awaitable, Callable

import anyio
import anyio.to_thread
import botocore.session
import pytest
from botocore.exceptions import ClientError, ConnectionClosedError, NoCredentialsError, ReadTimeoutError

from pydantic_ai.workspaces import (
    WorkspaceError,
    WorkspaceOutputLimitError,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.agentcore_sandbox import AgentCoreSandboxBackend

from .fake_agentcore import FakeAgentCore, Hold, client_error

# Upper bound on waiting for the fake to reach a held call; only a broken test runs into it.
READINESS_WAIT_TIMEOUT = 10


@pytest.mark.parametrize(
    ('ref', 'message'),
    [
        (WorkspaceRef(provider='e2b', id='x'), "unsupported workspace provider 'e2b'; expected 'agentcore'"),
        (
            WorkspaceRef(provider='agentcore', id='no-slash'),
            "must be '<code interpreter>/<session id>', got 'no-slash'",
        ),
        (WorkspaceRef(provider='agentcore', id='interpreter/'), "got 'interpreter/'"),
    ],
)
def test_an_invalid_ref_fails_at_construction(ref: WorkspaceRef, message: str) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        AgentCoreSandboxBackend(ref=ref)


@pytest.mark.parametrize('session_timeout', [0, 28_801, 1.5, True])
def test_an_invalid_session_timeout_fails_at_construction(session_timeout: object) -> None:
    with pytest.raises(ValueError, match='session_timeout must be an integer from 1 to 28800'):
        AgentCoreSandboxBackend(session_timeout=session_timeout)  # pyright: ignore[reportArgumentType]


def test_an_arn_ref_keeps_the_whole_code_interpreter() -> None:
    arn = 'arn:aws:bedrock-agentcore:us-east-1:123456789012:code-interpreter/my-interpreter-abc'
    ref = WorkspaceRef(provider='agentcore', id=f'{arn}/session1')
    assert AgentCoreSandboxBackend(ref=ref).ref == ref


async def test_calls_a_second_attempt_would_change_make_only_one(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend(region_name='us-west-2')
    path = str(fake_agentcore.home / 'file')
    await backend.write_bytes(path, b'data')
    assert await backend.read_bytes(path) == b'data'
    await backend.remove(path)
    await _cancel_when_reached(fake_agentcore.hold('startCommandExecution'), lambda: backend.run(['sleep', '30']))
    await backend.run(['true'])
    assert fake_agentcore.single_attempt_calls == {
        'startCommandExecution',
        'executeCode:write',
        'executeCode:remove',
        'executeCode:finished',
    }
    assert {
        'getTask',
        'stopTask',
        'executeCode:read',
        'executeCode:working_dir',
        'executeCode:tail',
        'executeCode:clean_up',
    } <= fake_agentcore.retrying_calls
    assert not fake_agentcore.single_attempt_calls & fake_agentcore.retrying_calls

    # botocore's own reading of each client's settings: `max_attempts` would count retries, not attempts.
    def attempts() -> list[tuple[str | None, int]]:
        session = botocore.session.get_session()
        built = [
            session.create_client(
                'bedrock-agentcore', aws_access_key_id='id', aws_secret_access_key='secret', **settings
            )
            for settings in fake_agentcore.clients
        ]
        # `Config` sets its options as instance attributes, which botocore's type stubs do not declare.
        return [
            (client.meta.region_name, vars(client.meta.config)['retries']['total_max_attempts']) for client in built
        ]

    assert sorted(await anyio.to_thread.run_sync(attempts)) == [('us-west-2', 1), ('us-west-2', 3)]


@pytest.mark.parametrize('anyio_backend', ['trio'])
async def test_works_on_trio(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    assert (await backend.run(['printf', 'trio'])).stdout == 'trio'
    assert backend.ref is not None


async def test_concurrent_first_use_starts_one_session(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    async with anyio.create_task_group() as group:
        for _ in range(3):
            group.start_soon(backend.working_dir)
    assert len(fake_agentcore.sessions) == 1


@pytest.mark.parametrize(
    ('error', 'message'),
    [
        (NoCredentialsError(), r'^No AWS credentials found for AgentCore'),
        (
            client_error('UnrecognizedClientException', 'bad token', 'StartCodeInterpreterSession'),
            r'^AgentCore refused the AWS credentials \(UnrecognizedClientException\)\.$',
        ),
        (
            client_error('ValidationException', 'sessionTimeoutSeconds too big', 'StartCodeInterpreterSession'),
            r'^AgentCore refused the request: .*sessionTimeoutSeconds too big',
        ),
    ],
)
async def test_a_start_that_cannot_succeed_is_unavailable(
    fake_agentcore: FakeAgentCore, error: Exception, message: str
) -> None:
    fake_agentcore.queue('start', error)
    backend = AgentCoreSandboxBackend()
    with pytest.raises(WorkspaceUnavailableError, match=message):
        await backend.working_dir()
    assert backend.ref is None


async def test_an_unknown_code_interpreter_is_unavailable(fake_agentcore: FakeAgentCore) -> None:
    with pytest.raises(WorkspaceUnavailableError, match='Code Interpreter not found: missing'):
        await AgentCoreSandboxBackend(code_interpreter='missing').working_dir()


async def test_a_transient_start_failure_propagates_and_a_retry_starts_a_session(fake_agentcore: FakeAgentCore) -> None:
    throttled = client_error('ThrottlingException', 'slow down', 'StartCodeInterpreterSession')
    fake_agentcore.queue('start', throttled)
    backend = AgentCoreSandboxBackend()
    with pytest.raises(ClientError, match='slow down'):
        await backend.working_dir()
    await backend.working_dir()
    assert backend.ref is not None


async def test_transient_invoke_failures_propagate_unchanged(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    fake_agentcore.queue('executeCode', client_error('InternalServerException', 'try again'))
    with pytest.raises(ClientError, match='try again'):
        await backend.exists('/tmp')
    timeout = ReadTimeoutError(endpoint_url='https://bedrock-agentcore.us-east-1.amazonaws.com')
    fake_agentcore.queue('executeCode', timeout)
    with pytest.raises(ReadTimeoutError):
        await backend.exists('/tmp')


async def test_an_attached_session_must_have_the_working_dir(fake_agentcore: FakeAgentCore) -> None:
    session = AgentCoreSandboxBackend()
    await session.working_dir()
    assert session.ref is not None
    missing = str(fake_agentcore.home / 'missing')
    attached = AgentCoreSandboxBackend(ref=session.ref, working_dir=missing)
    with pytest.raises(WorkspaceError, match=re.escape(f'working_dir {missing!r} does not exist')):
        await attached.working_dir()
    assert not (fake_agentcore.home / 'missing').exists()


async def test_a_started_session_creates_the_working_dir(fake_agentcore: FakeAgentCore) -> None:
    work = fake_agentcore.home / 'a' / 'b'
    backend = AgentCoreSandboxBackend(working_dir=str(work))
    assert await backend.working_dir() == str(work)
    assert (await backend.run(['pwd'])).stdout == f'{work}\n'


async def test_a_command_whose_working_dir_was_removed_fails(fake_agentcore: FakeAgentCore) -> None:
    work = fake_agentcore.home / 'work'
    backend = AgentCoreSandboxBackend(working_dir=str(work))
    await backend.working_dir()
    work.rmdir()
    with pytest.raises(WorkspaceError, match='no longer exists in AgentCore session'):
        await backend.run(['true'])


async def test_output_over_the_limit_raises_with_its_beginning(
    fake_agentcore: FakeAgentCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._MAX_OUTPUT_BYTES', 8)
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._READ_CHUNK_BYTES', 4)
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._OUTPUT_PREVIEW_BYTES', 4)
    backend = AgentCoreSandboxBackend()
    with pytest.raises(WorkspaceOutputLimitError) as caught:
        await backend.run(['sh', '-c', 'printf 0123456789; printf abcdef >&2'])
    assert (caught.value.stdout, caught.value.stderr, caught.value.limit) == ('0123', 'abcd', 8)


async def test_output_a_background_process_adds_after_the_exit_is_capped(
    fake_agentcore: FakeAgentCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._MAX_OUTPUT_BYTES', 8)
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._READ_CHUNK_BYTES', 4)
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._OUTPUT_PREVIEW_BYTES', 4)
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    go, written = fake_agentcore.home / 'go', fake_agentcore.home / 'written'
    os.mkfifo(go)
    os.mkfifo(written)
    # The next `executeCode` is `finished`: it measures 6 bytes, then the background process adds 10 more.
    hold = fake_agentcore.hold('executeCode')

    async def grow_then_release() -> None:
        try:
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await anyio.to_thread.run_sync(hold.reached.wait, abandon_on_cancel=True)
                await anyio.to_thread.run_sync(go.write_text, 'go\n', abandon_on_cancel=True)
                await anyio.to_thread.run_sync(written.read_text, abandon_on_cancel=True)
        finally:
            hold.release.set()

    async with anyio.create_task_group() as group:
        group.start_soon(grow_then_release)
        with pytest.raises(WorkspaceOutputLimitError) as caught:
            await backend.run(
                [
                    'sh',
                    '-c',
                    'printf 012345; (read _ <"$1"; printf abcdefghij; echo >"$2") &',
                    'sh',
                    str(go),
                    str(written),
                ]
            )
        assert (caught.value.stdout, caught.value.stderr, caught.value.limit) == ('0123', '', 8)


async def test_a_command_that_replaces_its_output_file_fails_without_blocking_the_session(
    fake_agentcore: FakeAgentCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The command directory takes its name from this id, which the command needs to swap the file.
    monkeypatch.setattr(uuid, 'uuid4', lambda: uuid.UUID(int=1))
    directory = f'/tmp/pydantic-ai-{uuid.UUID(int=1).hex}'
    backend = AgentCoreSandboxBackend()
    # Without the regular-file check, the kernel's `open` of the FIFO waits for a writer that never comes.
    with pytest.raises(WorkspaceError, match='replaced its output file'):
        await backend.run(['sh', '-c', 'rm -f -- "$1"; mkfifo -- "$1"; echo swapped >&2', 'sh', f'{directory}/out'])
    assert not os.path.exists(directory)
    assert (await backend.run(['echo', 'ok'])).stdout == 'ok\n'


async def test_files_and_output_travel_in_chunks(
    fake_agentcore: FakeAgentCore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._READ_CHUNK_BYTES', 4)
    monkeypatch.setattr('pydantic_ai_harness.agentcore_sandbox._backend._WRITE_CHUNK_BYTES', 3)
    backend = AgentCoreSandboxBackend()
    path = str(fake_agentcore.home / 'chunked.bin')
    fake_agentcore.calls.clear()
    await backend.write_bytes(path, b'0123456789')
    assert fake_agentcore.calls.count('executeCode') == 4
    assert await backend.read_bytes(path) == b'0123456789'
    assert (await backend.run(['cat', path])).stdout == '0123456789'


async def test_an_empty_write_creates_the_file(fake_agentcore: FakeAgentCore) -> None:
    path = fake_agentcore.home / 'empty'
    await AgentCoreSandboxBackend().write_bytes(str(path), b'')
    assert path.read_bytes() == b''


async def test_a_timeout_returns_the_partial_output_and_stops_the_task(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    ready = fake_agentcore.home / 'ready'
    os.mkfifo(ready)
    hold = fake_agentcore.hold('startCommandExecution')

    async def release_once_written() -> None:
        try:
            # The command writes its output, then opens the FIFO, which returns once this reads it.
            with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                await anyio.to_thread.run_sync(ready.read_text, abandon_on_cancel=True)
        finally:
            hold.release.set()

    try:
        async with anyio.create_task_group() as group:
            group.start_soon(release_once_written)
            # The deadline passes while the start's answer is held, so no test time is spent waiting on it.
            with pytest.raises(WorkspaceTimeoutError) as caught:
                await backend.run(
                    ['sh', '-c', 'printf started; printf warn >&2; echo > "$1"; exec sleep 30', 'sh', str(ready)],
                    timeout=0.001,
                )
            assert (caught.value.stdout, caught.value.stderr) == ('started', 'warn')
    finally:
        # A reader still waiting because the test failed gets end of file instead of blocking its thread.
        # Opening fails with ENXIO when no reader is waiting, the usual case.
        with contextlib.suppress(OSError):
            os.close(os.open(ready, os.O_WRONLY | os.O_NONBLOCK))
        # A reader thread that arrives after this gets `FileNotFoundError` instead of blocking.
        ready.unlink()
    assert 'stopTask' in fake_agentcore.calls
    [session] = fake_agentcore.sessions.values()
    assert all(task.poll() is not None for task in session.tasks.values())


async def test_a_failed_start_still_cleans_up(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    fake_agentcore.queue('startCommandExecution', client_error('ThrottlingException', 'slow down'))
    with pytest.raises(ClientError, match='slow down'):
        await backend.run(['true'])
    assert 'stopTask' not in fake_agentcore.calls


async def test_unexpected_service_replies_are_workspace_errors(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()

    fake_agentcore.queue_task('startCommandExecution', taskStatus='submitted')
    with pytest.raises(WorkspaceError, match='startCommandExecution returned no task id'):
        await backend.run(['true'])

    # An event type this backend does not know is skipped, like one botocore adds in a later release.
    fake_agentcore.queue('getTask', [{'futureEvent': {}}])
    with pytest.raises(WorkspaceError, match='AgentCore getTask returned no result'):
        await backend.run(['true'])

    fake_agentcore.queue_code_reply('Traceback: MemoryError', error=True)
    with pytest.raises(WorkspaceError, match='AgentCore executeCode failed: Traceback: MemoryError'):
        await backend.exists('/tmp')

    fake_agentcore.queue_code_reply('not json')
    with pytest.raises(WorkspaceError, match="unexpected reply to 'exists': 'not json'"):
        await backend.exists('/tmp')

    fake_agentcore.queue_code_reply('{"error": ["OSError", "No space left on device"]}')
    with pytest.raises(OSError, match="No space left on device \\(in the AgentCore session, for '/tmp/x'\\)"):
        await backend.write_bytes('/tmp/x', b'x')


async def test_a_command_that_left_no_exit_code_fails(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    fake_agentcore.queue_code_reply(
        '{"ok": {"exit_code": null, "size": 0, "out": null, "err": null, "replaced": false}}'
    )
    with pytest.raises(WorkspaceError, match='ended without recording an exit code'):
        await backend.run(['true'])


async def test_a_program_that_fails_outside_the_filesystem_is_a_workspace_error(fake_agentcore: FakeAgentCore) -> None:
    with pytest.raises(WorkspaceError, match=r'(?s)AgentCore executeCode failed: .*ValueError: .*embedded null'):
        await AgentCoreSandboxBackend().read_bytes('/tmp/bad\x00name')


async def test_a_refused_call_in_a_live_session_fails_only_that_call(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    fake_agentcore.queue('startCommandExecution', client_error('ValidationException', 'command is too long'))
    with pytest.raises(WorkspaceError, match=r'AgentCore refused the request: .*command is too long') as caught:
        await backend.run(['true'])
    assert not isinstance(caught.value, WorkspaceUnavailableError)
    assert (await backend.run(['printf', 'still here'])).stdout == 'still here'


async def test_an_invalid_command_env_fails_before_running(fake_agentcore: FakeAgentCore) -> None:
    with pytest.raises(ValueError, match='env keys must be non-empty without `=`'):
        await AgentCoreSandboxBackend().run(['true'], env={'A=B': 'x'})
    assert not fake_agentcore.calls


async def test_small_output_comes_back_with_the_exit_code(fake_agentcore: FakeAgentCore) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    fake_agentcore.calls.clear()
    assert (await backend.run(['sh', '-c', 'printf out; printf err >&2; exit 4'])).exit_code == 4
    # Start, at least one poll, then one program that reads both streams and cleans up.
    assert fake_agentcore.calls.count('executeCode') == 1


async def test_a_cancel_during_a_slow_start_still_stops_the_command(fake_agentcore: FakeAgentCore) -> None:
    """The caller is cancelled while AgentCore has started the command but not yet answered with its task id."""
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    await _cancel_when_reached(fake_agentcore.hold('startCommandExecution'), lambda: backend.run(['sleep', '30']))
    assert 'stopTask' in fake_agentcore.calls
    [session] = fake_agentcore.sessions.values()
    assert all(task.poll() is not None for task in session.tasks.values())


async def test_a_native_cancel_during_a_slow_start_still_stops_the_command(fake_agentcore: FakeAgentCore) -> None:
    """`Task.cancel()` passes AnyIO shields, so the start runs to its end in a task of its own."""
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    [session] = fake_agentcore.sessions.values()
    hold = fake_agentcore.hold('startCommandExecution')
    running = asyncio.create_task(backend.run(['sleep', '30']))
    # If the run fails before its start is held, wake the wait below instead of blocking its thread.
    running.add_done_callback(lambda _: hold.reached.set())
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        await anyio.to_thread.run_sync(hold.reached.wait, abandon_on_cancel=True)
    running.cancel()
    # Let the cancellation reach the task before the start answers.
    await asyncio.sleep(0)
    hold.release.set()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert 'stopTask' in fake_agentcore.calls
    assert all(task.poll() is not None for task in session.tasks.values())


@pytest.mark.parametrize(
    'error',
    [client_error('ThrottlingException', 'slow down'), ConnectionClosedError(endpoint_url='https://agentcore.test')],
)
async def test_a_failed_stop_still_reads_the_output_and_cleans_up(
    fake_agentcore: FakeAgentCore, error: Exception
) -> None:
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    fake_agentcore.queue('stopTask', error)
    await _cancel_when_reached(fake_agentcore.hold('startCommandExecution'), lambda: backend.run(['sleep', '30']))
    assert {'stopTask', 'executeCode:tail', 'executeCode:clean_up'} <= fake_agentcore.retrying_calls


async def test_stopping_a_task_that_already_ended_still_cleans_up(fake_agentcore: FakeAgentCore) -> None:
    """The caller is cancelled after the command ended but before a poll saw it, so AgentCore refuses the stop."""
    backend = AgentCoreSandboxBackend()
    await backend.working_dir()
    [session] = fake_agentcore.sessions.values()
    # The first poll must not see the command finished, or nothing would be cancelled; a queued reply is held too.
    fake_agentcore.queue_task('getTask', taskStatus='working')
    hold = fake_agentcore.hold('getTask')

    async def wait_for_exit() -> None:
        for task in session.tasks.values():
            await anyio.to_thread.run_sync(task.wait)

    await _cancel_when_reached(hold, lambda: backend.run(['true']), before_cancel=wait_for_exit)
    assert {'stopTask', 'executeCode:tail', 'executeCode:clean_up'} <= fake_agentcore.retrying_calls
    assert not session.canceled


async def _cancel_when_reached(
    hold: Hold,
    call: Callable[[], Awaitable[object]],
    *,
    before_cancel: Callable[[], Awaitable[None]] | None = None,
) -> None:
    """Run `call`, cancel it once the held tool call has done its work, then let that call answer."""
    with anyio.CancelScope() as scope:
        async with anyio.create_task_group() as group:

            async def cancel() -> None:
                with anyio.fail_after(READINESS_WAIT_TIMEOUT):
                    await anyio.to_thread.run_sync(hold.reached.wait, abandon_on_cancel=True)
                if before_cancel is not None:
                    await before_cancel()
                scope.cancel()
                hold.release.set()

            group.start_soon(cancel)
            try:
                await call()
            finally:
                # A `call` that fails before reaching the hold must not leave the waiter's thread blocked.
                hold.reached.set()
    assert scope.cancelled_caught

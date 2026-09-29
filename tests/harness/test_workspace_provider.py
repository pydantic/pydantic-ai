import asyncio

import anyio
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.workspaces import (
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceTimeoutError,
    WrapperWorkspace,
)
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness._warn import warn_argument_renamed
from pydantic_ai_harness._workspace import innermost_backend
from pydantic_ai_harness._workspace_provider import (
    absolute_path,
    check_integer,
    check_working_dir,
    command_argv,
    command_deadline,
    safe_credential_reason,
    stop_shielded,
)


def test_credential_reason_keeps_safe_context_without_echoing_key() -> None:
    assert (
        safe_credential_reason(ValueError('API key is malformed: expected the e2b_ prefix')) == 'API key is malformed'
    )
    assert safe_credential_reason(ValueError('token abc-secret-123 expired')) == 'Credential expired'
    assert 'abc-secret-123' not in safe_credential_reason(ValueError('token abc-secret-123 rejected'))


async def test_own_timeout_and_external_cancel_stop_once() -> None:
    stopped: list[str] = []

    async def stop() -> None:
        stopped.append('stop')

    with pytest.raises(WorkspaceTimeoutError) as error:
        async with command_deadline(0.01, stop=stop, output=lambda: ('partial out', 'partial err')):
            await anyio.sleep_forever()
    assert (error.value.stdout, error.value.stderr) == ('partial out', 'partial err')
    assert stopped == ['stop']

    stopped.clear()
    with anyio.move_on_after(0.01) as scope:
        async with command_deadline(None, stop=stop):
            await anyio.sleep_forever()
    assert scope.cancelled_caught
    assert stopped == ['stop']


async def test_failed_stop_does_not_replace_original_cancellation() -> None:
    async def stop() -> None:
        raise RuntimeError('stop failed')

    with pytest.raises(WorkspaceTimeoutError, match=r'^Command timed out after 0\.01 seconds$'):
        async with command_deadline(0.01, stop=stop):
            await anyio.sleep_forever()


async def test_stop_shielded_finishes_under_outer_cancellation() -> None:
    stopped: list[str] = []

    async def stop() -> None:
        await anyio.sleep(0)
        stopped.append('stop')

    with anyio.move_on_after(0) as scope:
        await stop_shielded(stop)
    assert scope.cancelled_caught or scope.cancel_called
    assert stopped == ['stop']


async def test_native_repeated_cancel_cannot_abandon_stop(anyio_backend: str) -> None:
    if anyio_backend != 'asyncio':  # pragma: no cover
        pytest.skip('Native task.cancel() is asyncio-specific')

    async def exercise(timeout: float | None, cancellations: int) -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        finished = asyncio.Event()
        calls = 0

        async def stop() -> None:
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            finished.set()

        async def work() -> None:
            async with command_deadline(timeout, stop=stop):
                await asyncio.sleep(100)

        task = asyncio.create_task(work())
        if timeout is None:
            await asyncio.sleep(0)
            task.cancel()
        await asyncio.wait_for(entered.wait(), 5)
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
        release.set()
        try:
            await asyncio.wait_for(task, 5)
        except (asyncio.CancelledError, WorkspaceTimeoutError):
            pass
        await asyncio.wait_for(finished.wait(), 5)
        assert calls == 1

    await exercise(0.01, 0)
    await exercise(None, 1)
    await exercise(None, 2)


async def test_stop_cleanup_finishes_before_stop_shielded_returns() -> None:
    events: list[str] = []

    async def stop() -> None:
        try:
            await anyio.sleep(100)
        finally:
            # Providers log "may still be running" here once the grace cuts them off.
            events.append('stop cleanup')

    await stop_shielded(stop, grace=0.05)
    events.append('returned')
    assert events == ['stop cleanup', 'returned']


def test_absolute_path_passes_none_and_absolute_paths_through() -> None:
    assert absolute_path('workdir', None) is None
    assert absolute_path('workdir', '/home/user/../project') == '/home/user/../project'


def test_absolute_path_rejects_relative_paths() -> None:
    with pytest.raises(ValueError, match=r"workdir must be an absolute workspace path or None, got 'project'."):
        absolute_path('workdir', 'project')


@pytest.mark.parametrize(
    ('command', 'shell', 'argv'),
    [('echo "$HOME"', True, ['/bin/sh', '-c', 'echo "$HOME"']), (('echo', '$HOME'), False, ['echo', '$HOME'])],
)
def test_command_argv(command: str | tuple[str, ...], shell: bool, argv: list[str]) -> None:
    assert command_argv(command, shell) == argv


@pytest.mark.parametrize('command', [b'echo hello', ['echo', 7]])
def test_command_argv_rejects_non_string_elements(command: object) -> None:
    with pytest.raises(TypeError):
        command_argv(command, False)  # type: ignore[arg-type]


def test_command_argv_rejects_nul_element() -> None:
    with pytest.raises(ValueError, match='NUL'):
        command_argv(['echo', 'bad\x00arg'], False)


@pytest.mark.parametrize(
    ('command', 'shell', 'message'),
    [
        ('echo hi', False, 'a string command requires shell=True; pass an argv sequence otherwise'),
        (['echo', 'hi'], True, 'an argv sequence cannot be combined with shell=True; pass a single command string'),
        ([], False, 'an argv sequence needs at least the program to run'),
    ],
)
def test_command_argv_rejects_an_unrunnable_command(command: str | list[str], shell: bool, message: str) -> None:
    with pytest.raises(TypeError) as error:
        command_argv(command, shell)
    assert str(error.value) == message


class DurableLikeWorkspace(WrapperWorkspace):
    """A wrapper that refuses `backend`, as a durable workspace does in workflow code."""

    @property
    def backend(self) -> WorkspaceBackend:
        raise AssertionError('`backend` was read on a wrapper')  # pragma: no cover


def test_innermost_backend_unwraps_facades_and_wrappers_through_wrapped() -> None:
    backend = LocalWorkspaceBackend('.')
    assert innermost_backend(Workspace(backend)) is backend
    assert innermost_backend(DurableLikeWorkspace(ReadOnlyWorkspace(Workspace(backend)))) is backend


@pytest.mark.parametrize('value', [None, '/', '/home/user'])
def test_check_working_dir_accepts_absolute_paths_and_none(value: str | None) -> None:
    check_working_dir(value)


def test_check_working_dir_rejects_relative_paths() -> None:
    with pytest.raises(UserError, match=r"^working_dir must be an absolute POSIX path or None, got 'project'\.$"):
        check_working_dir('project')


@pytest.mark.parametrize(
    ('value', 'minimum', 'optional'),
    [(1, 1, False), (0, 0, False), (None, 1, True)],
)
def test_check_integer_accepts(value: int | None, minimum: int, optional: bool) -> None:
    check_integer('timeout', value, minimum=minimum, optional=optional)


@pytest.mark.parametrize(
    ('value', 'optional', 'message'),
    [
        (0, False, 'timeout must be an integer of at least 1, got 0.'),
        (None, False, 'timeout must be an integer of at least 1, got None.'),
        (True, False, 'timeout must be an integer of at least 1, got True.'),
        (1.5, True, 'timeout must be an integer of at least 1 or None, got 1.5.'),
    ],
)
def test_check_integer_rejects(value: object, optional: bool, message: str) -> None:
    with pytest.raises(UserError) as error:
        check_integer('timeout', value, optional=optional)  # pyright: ignore[reportArgumentType]
    assert str(error.value) == message


def test_warn_argument_renamed_points_at_the_new_name() -> None:
    with pytest.warns(HarnessDeprecationWarning) as record:
        warn_argument_renamed('Sandbox', 'workdir', 'working_dir', stacklevel=2)
    assert str(record[0].message) == (
        '`Sandbox(workdir=...)` has been renamed to `Sandbox(working_dir=...)`. '
        'Update the call; this deprecated alias will be removed in a future release.'
    )
    assert record[0].filename == __file__

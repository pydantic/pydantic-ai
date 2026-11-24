"""The conformance suite for Pydantic AI workspace backends.

Subclass `WorkspaceBackendSuite` in a pytest module and provide a `backend` fixture: each test checks
one rule of the `WorkspaceBackend` contract, the same rules the built-in and provider backends pass.
Requires pytest and the anyio pytest plugin.
"""

from __future__ import annotations

import math
import uuid
import warnings
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TypeVar

import anyio
import pytest

from .._warnings import PydanticAIDeprecationWarning
from .protocol import (
    CommandResult,
    SupportsCommands,
    SupportsRealpath,
    WorkspaceBackend,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from .workspace import Workspace

__all__ = ('Unsupported', 'WorkspaceBackendSuite')


@dataclass(frozen=True)
class Unsupported:
    """Return from an optional `WorkspaceBackendSuite` fixture the backend can't provide.

    The rules that need the fixture skip, with `reason` as the skip message.
    """

    reason: str


_Factory = TypeVar('_Factory')


def _provided(name: str, factory: _Factory | Unsupported | None) -> _Factory:
    """The factory an optional fixture returned, or a skip of the rule that needs it."""
    if isinstance(factory, Unsupported):
        pytest.skip(factory.reason)
    if factory is None:
        # TODO(v3): fail the rule instead of skipping it.
        warnings.warn(
            f'`{name}` is not provided: provide it, or return `Unsupported(<why>)` from it. '
            'From v3, a rule whose fixture is neither fails.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        pytest.skip(f'`{name}` is not provided')
    return factory


def _commands(backend: WorkspaceBackend) -> SupportsCommands:
    if not isinstance(backend, SupportsCommands):
        pytest.skip('backend does not implement SupportsCommands')
    return backend


_FINISH_ONCE_RELEASED = 'while [ -d "${1%/*}" ] && [ ! -f "$1" ]; do sleep 0.1; done; printf finished > "$2"'
"""Shell that waits for the file `$1`, then writes the file `$2`. It gives up once `$1`'s directory is gone."""

_POWERSHELL_FINISH_ONCE_RELEASED = (
    'while ((Test-Path -LiteralPath (Split-Path -LiteralPath $a0)) -and -not (Test-Path -LiteralPath $a0)) '
    "{ Start-Sleep -Milliseconds 100 }; [IO.File]::WriteAllText($a1, 'finished')"
)
"""`_FINISH_ONCE_RELEASED` in PowerShell, with `$a0` and `$a1` for `$1` and `$2`."""


_POWERSHELL_PWD = '[Console]::Out.Write((Get-Location).ProviderPath + "`n")'
"""`pwd -P` in PowerShell."""


def _windows(backend: WorkspaceBackend) -> bool:
    return Workspace(backend).platform == 'windows'


def _powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _powershell_assignments(args: tuple[str, ...]) -> str:
    return ''.join(f'$a{index} = {_powershell_literal(arg)}\n' for index, arg in enumerate(args))


def _powershell_start(script: str, *args: str, window: str) -> str:
    """PowerShell that starts `script`, with `args` as `$a0`, ..., in a PowerShell of its own, without waiting.

    The script goes through a temporary file, which the new PowerShell reads and deletes, rather than on its
    command line, where antivirus software flags an `-EncodedCommand` launch.
    """
    return (
        '$start = [IO.Path]::GetTempFileName(); '
        f'[IO.File]::WriteAllText($start, {_powershell_literal(_powershell_assignments(args) + script)}); '
        f'Start-Process {window} -FilePath (Get-Process -Id $PID).Path -ArgumentList '
        "('-NoProfile -NonInteractive -Command \"$s = [IO.File]::ReadAllText(''{0}''); [IO.File]::Delete(''{0}''); "
        '& ([ScriptBlock]::Create($s))"\' -f '
        '[Management.Automation.Language.CodeGeneration]::EscapeSingleQuotedStringContent($start))'
    )


async def _script(
    backend: WorkspaceBackend,
    sh: str,
    powershell: str,
    *args: str,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> CommandResult:
    """Run `sh` with `args` as `$1`, `$2`, ...; on a `'windows'` backend, run `powershell` with them as `$a0`, `$a1`, ..."""
    commands = _commands(backend)
    if _windows(backend):
        return await commands.run(_powershell_assignments(args) + powershell, shell=True, env=env, timeout=timeout)
    return await commands.run(['sh', '-c', sh, 'sh', *args], env=env, timeout=timeout)


async def _symlink(backend: WorkspaceBackend, target: str, link: str, *, directory: bool = False) -> bool:
    """Whether a symlink at `link` to `target`, kept as written, could be created."""
    windows = _windows(backend)
    result = await _script(
        backend,
        'ln -s "$1" "$2"',
        f'cmd /c mklink {"/D " if directory else ""}"$a1" "$a0" | Out-Null',
        target.replace('/', '\\') if windows else target,
        link,
    )
    return result.exit_code == 0


async def _assert_argv_item_is_literal(backend: WorkspaceBackend, payload: str) -> None:
    commands = _commands(backend)
    if not _windows(backend):
        result = await commands.run(['sh', '-c', 'printf "%s" "$1"', 'sh', payload])
        assert (result.exit_code, result.stdout) == (0, payload)
        return
    workspace = Workspace(backend)
    async with _scratch_dir(workspace) as root:
        # `-File` hands the script its arguments as given, without parsing them as PowerShell.
        script = workspace.paths.join(root, 'echo.ps1')
        await workspace.write_text(script, '[Console]::Out.Write($args[0])')
        powershell = (await commands.run('[Console]::Out.Write((Get-Process -Id $PID).Path)', shell=True)).stdout
        result = await commands.run(
            [powershell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-File', script, payload]
        )
    assert (result.exit_code, result.stdout) == (0, payload)


async def _wait_until_gone(backend: WorkspaceBackend, pid: str) -> None:
    """Wait for the process `pid` to end, as the backend's own commands see it."""
    with anyio.fail_after(60):  # Hang guard, not an assertion about remote speed.
        while (
            await _script(
                backend, 'kill -0 "$1"', 'if (-not (Get-Process -Id $a0 -ErrorAction Ignore)) { exit 1 }', pid
            )
        ).exit_code == 0:
            await anyio.sleep(0.05)  # pragma: lax no cover - depends on how fast the process dies


_BACKGROUND_CHILD = 'sleep 3600 & echo $! > "$1"; wait'
"""Shell that starts a background child, writes its PID to the file `$1`, and waits for it."""


@asynccontextmanager
async def _scratch_dir(workspace: Workspace) -> AsyncGenerator[str]:
    """A fresh directory under the working directory, removed afterwards."""
    root = workspace.paths.join(await workspace.working_dir(), f'.pydantic-ai-conformance-{uuid.uuid4().hex}')
    await workspace.make_dir(root)
    try:
        yield root
    finally:
        await workspace.remove(root)


class WorkspaceBackendSuite:
    """Subclass in your test suite and provide the `backend` fixture.

    Each test checks one rule of the backend contract. Command rules skip for a backend without
    `SupportsCommands`; filesystem rules run through [`Workspace`][pydantic_ai.workspaces.Workspace], so a
    command-only backend is checked on the file operations derived through its shell. The first-use, reattach
    and destroy rules need the optional factory fixtures below: return
    [`Unsupported`][pydantic_ai.workspaces.conformance.Unsupported] from one the backend can't provide, and its
    rules skip with that reason. A factory fixture left at its default skips its rules with a deprecation warning.
    """

    pytestmark = pytest.mark.anyio

    @pytest.fixture
    def backend(self) -> WorkspaceBackend | AsyncIterator[WorkspaceBackend]:
        raise NotImplementedError('provide a `backend` fixture')

    @pytest.fixture
    def fresh_backend(self) -> Callable[[], WorkspaceBackend] | Unsupported | None:
        """Build an uninitialized backend to check concurrent first use."""
        return None

    @pytest.fixture
    def destructive_backend(
        self, fresh_backend: Callable[[], WorkspaceBackend] | Unsupported | None
    ) -> Callable[[], WorkspaceBackend] | Unsupported | None:
        """Build an independent environment for destruction; defaults to `fresh_backend`."""
        return fresh_backend

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend] | Unsupported | None:
        """Build a second backend that attaches to `ref`. Enables the reattach rules."""
        return None

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]] | Unsupported | None:
        """Destroy the environment behind `backend`. Enables the destroy rules."""
        return None

    async def test_has_the_required_members(self, backend: WorkspaceBackend) -> None:
        assert isinstance(backend, WorkspaceBackend)

    async def test_command_form_must_match_shell(self, backend: WorkspaceBackend) -> None:
        """A string needs `shell=True` and an argv sequence needs `shell=False`; a mismatch is a `TypeError`."""
        commands = _commands(backend)
        with pytest.raises(TypeError):
            await commands.run('true')
        with pytest.raises(TypeError):
            await commands.run(['true'], shell=True)

    async def test_stdin_is_at_eof(self, backend: WorkspaceBackend) -> None:
        """Noninteractive commands never wait for input from the caller."""
        # This is a command deadline, not a latency assertion on remote dispatch.
        result = await _script(
            backend,
            'read value || printf eof',
            "if ($null -eq [Console]::In.ReadLine()) { [Console]::Out.Write('eof') }",
            timeout=30,
        )
        assert (result.exit_code, result.stdout) == (0, 'eof')

    async def test_undecodable_command_bytes_are_replaced(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        if not has_real_posix_shell:
            pytest.skip('fake has no command byte stream')
        result = await _script(
            backend,
            "printf '\\377'; printf '\\376' >&2",
            '[Console]::OpenStandardOutput().Write([byte[]](255), 0, 1); '
            '[Console]::OpenStandardError().Write([byte[]](254), 0, 1)',
        )
        assert (result.stdout, result.stderr) == ('\ufffd', '\ufffd')

    async def test_command_output_is_complete(self, backend: WorkspaceBackend) -> None:
        """If output cannot be collected in full, the backend must raise rather than return a truncated success."""
        output = 'workspace' * 1024
        result = await _script(
            backend,
            'i=0; while [ "$i" -lt 1024 ]; do printf workspace; printf workspace >&2; i=$((i+1)); done',
            "for ($i = 0; $i -lt 1024; $i++) { [Console]::Out.Write('workspace'); [Console]::Error.Write('workspace') }",
        )
        assert (result.exit_code, result.stdout, result.stderr) == (0, output, output)

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        """Override only if the SDK cannot report exit independently of pipe EOF (E2B currently cannot)."""
        return True

    async def test_background_child_does_not_hold_up_completed_command(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool, can_detect_exit_with_inherited_output_pipes: bool
    ) -> None:
        if not has_real_posix_shell or not can_detect_exit_with_inherited_output_pipes:
            pytest.skip('backend cannot observe the direct command exit independently of inherited output pipes')
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            release = workspace.paths.join(root, 'release')
            finished = workspace.paths.join(root, 'finished')
            # The background child holds stdout open until released. A backend waiting for
            # pipe EOF cannot complete this run, regardless of control-plane latency.
            try:
                with anyio.fail_after(60):  # Hang guard, not an assertion about remote speed.
                    result = await _script(
                        backend,
                        f'({_FINISH_ONCE_RELEASED}) & printf done',
                        # `-NoNewWindow` hands the child this command's output handles.
                        _powershell_start(_POWERSHELL_FINISH_ONCE_RELEASED, release, finished, window='-NoNewWindow')
                        + "; [Console]::Out.Write('done')",
                        release,
                        finished,
                    )
                assert (result.exit_code, result.stdout) == (0, 'done')
                assert not await workspace.exists(finished)
            finally:
                # Release the child even if the hang guard cancelled the command.
                with anyio.move_on_after(30, shield=True):
                    await workspace.write_bytes(release, b'go')
                    while not await workspace.exists(finished):
                        await anyio.sleep(0.1)

    async def test_result_reports_exit_code_stdout_and_stderr(self, backend: WorkspaceBackend) -> None:
        """A non-zero exit is a normal result, not an error."""
        script = (
            "[Console]::Out.Write('out'); [Console]::Error.Write('err'); exit 7"
            if _windows(backend)
            else 'printf out; printf err >&2; exit 7'
        )
        result = await _commands(backend).run(script, shell=True)
        assert (result.exit_code, result.stdout, result.stderr) == (7, 'out', 'err')

    async def test_a_missing_program_exits_127(self, backend: WorkspaceBackend) -> None:
        """Like `sh`, a program that doesn't exist is a normal result with exit code 127, not an error."""
        result = await _commands(backend).run(['pydantic-ai-conformance-missing-program'])
        assert result.exit_code == 127

    async def test_argv_items_are_literal(self, backend: WorkspaceBackend) -> None:
        await _assert_argv_item_is_literal(backend, ' literal $() `quoted`; && ')

    async def test_a_large_quote_heavy_argv_item_is_literal(self, backend: WorkspaceBackend) -> None:
        """Quoting an item for a shell more than once multiplies its size past the system's argument limit."""
        await _assert_argv_item_is_literal(backend, "'a' " * 4096)

    async def test_working_dir_is_canonical(self, backend: WorkspaceBackend) -> None:
        """Absolute, symlinks resolved, no `.`/`..`: the directory commands actually start in."""
        paths = Workspace(backend).paths
        working_dir = await backend.working_dir()
        assert paths.isabs(working_dir) and paths.normpath(working_dir) == working_dir
        if isinstance(backend, SupportsCommands):
            assert (await _script(backend, 'pwd -P', _POWERSHELL_PWD)).stdout == f'{working_dir}\n'

    async def test_run_starts_in_cwd(self, backend: WorkspaceBackend) -> None:
        """`Workspace.run(cwd=...)` starts the command there, relative paths resolving against the working directory."""
        _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            expected = f'{await workspace.realpath(root)}\n'
            windows = _windows(backend)
            argv = ['cmd', '/d', '/c', 'cd'] if windows else ['sh', '-c', 'pwd -P']
            assert (await workspace.run(argv, cwd=workspace.paths.basename(root))).stdout.replace('\r', '') == expected
            script = _POWERSHELL_PWD if windows else 'pwd -P'
            assert (await workspace.run(script, shell=True, cwd=root)).stdout == expected

    async def test_timeout_raises_workspace_timeout_error(self, backend: WorkspaceBackend) -> None:
        with pytest.raises(WorkspaceTimeoutError):
            await _script(backend, 'sleep 3600', 'Start-Sleep -Seconds 3600', timeout=1.0)

    async def test_an_invalid_timeout_raises_value_error(self, backend: WorkspaceBackend) -> None:
        """A timeout is a positive finite number of seconds, or `None`."""
        for timeout in (math.nan, -1.0):
            with pytest.raises(ValueError):
                await _script(backend, 'exit 0', 'exit 0', timeout=timeout)
        with pytest.raises(ValueError):
            await _script(backend, 'exit 0', 'exit 0', timeout='5')  # pyright: ignore[reportArgumentType]

    async def test_timeout_stops_background_children(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        if not has_real_posix_shell or _windows(backend):
            pytest.skip('backend has no POSIX background child to stop')
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            pid_file = workspace.paths.join(root, 'pid')
            with pytest.raises(WorkspaceTimeoutError):
                await _script(backend, _BACKGROUND_CHILD, '', pid_file, timeout=1.0)
            if not await workspace.exists(pid_file):
                pytest.skip('the timeout ended the command before it started its child')  # pragma: no cover
            await _wait_until_gone(backend, (await workspace.read_text(pid_file)).strip())

    async def test_cancellation_stops_foreground_work(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        if not has_real_posix_shell:
            pytest.skip('fake has no foreground process to cancel')
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            pid_file = workspace.paths.join(root, 'pid')

            async def command() -> None:
                await _script(
                    backend,
                    'echo $$ > "$1"; exec sleep 3600',
                    '[IO.File]::WriteAllText($a0, "$PID"); Start-Sleep -Seconds 3600',
                    pid_file,
                )

            async with anyio.create_task_group() as tg:
                tg.start_soon(command)
                with anyio.fail_after(30):
                    while not await workspace.exists(pid_file):
                        await anyio.sleep(0.05)
                tg.cancel_scope.cancel()
            await _wait_until_gone(backend, (await workspace.read_text(pid_file)).strip())

    async def test_cancellation_stops_background_children(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        if not has_real_posix_shell or _windows(backend):
            pytest.skip('backend has no POSIX background child to stop')
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            pid_file = workspace.paths.join(root, 'pid')
            async with anyio.create_task_group() as tg:
                tg.start_soon(_script, backend, _BACKGROUND_CHILD, '', pid_file)
                with anyio.fail_after(30):
                    while not await workspace.exists(pid_file):
                        await anyio.sleep(0.05)
                tg.cancel_scope.cancel()
            await _wait_until_gone(backend, (await workspace.read_text(pid_file)).strip())

    async def test_env_is_added(self, backend: WorkspaceBackend) -> None:
        result = await _script(
            backend, 'printf %s "$CONFORMANCE"', '[Console]::Out.Write($env:CONFORMANCE)', env={'CONFORMANCE': 'value'}
        )
        assert result.stdout == 'value'

    async def test_ref_exists_after_the_first_operation_and_is_stable(self, backend: WorkspaceBackend) -> None:
        before = backend.ref
        await backend.working_dir()
        created = backend.ref
        await backend.working_dir()
        assert isinstance(created, WorkspaceRef) and backend.ref == created
        assert before in (None, created)

    async def test_concurrent_first_use_shares_one_environment(
        self, fresh_backend: Callable[[], WorkspaceBackend] | Unsupported | None
    ) -> None:
        backend = _provided('fresh_backend', fresh_backend)()
        assert backend.ref is None
        workspace = Workspace(backend)
        paths = [f'concurrent-{uuid.uuid4().hex}' for _ in range(5)]

        async def write(path: str) -> None:
            await workspace.write_bytes(path, path.encode())

        try:
            async with anyio.create_task_group() as group:
                for path in paths:
                    group.start_soon(write, path)
            ref = backend.ref
            assert isinstance(ref, WorkspaceRef)
            for path in paths:
                assert await workspace.read_bytes(path) == path.encode()
                assert backend.ref == ref
        finally:
            for path in paths:
                with suppress(FileNotFoundError):
                    await workspace.remove(path)

    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None:
        """A shell-derived filesystem must page reads rather than hit a command-output cap."""
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = workspace.paths.join(root, 'large.bin')
            data = b'x' * (8 * 1024 * 1024)
            await workspace.write_bytes(path, data)
            assert await workspace.read_bytes(path) == data

    async def test_bytes_round_trip_and_write_creates_parents(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = workspace.paths.join(root, 'nested', 'data.bin')
            await workspace.write_bytes(path, b'\x00workspace\xff')
            assert await workspace.read_bytes(path) == b'\x00workspace\xff'
            await workspace.write_bytes(path, b'replaced')
            assert await workspace.read_bytes(path) == b'replaced'

    async def test_exists(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            await workspace.write_bytes(workspace.paths.join(root, 'file'), b'data')
            assert await workspace.exists(workspace.paths.join(root, 'file'))
            assert await workspace.exists(root)
            assert not await workspace.exists(workspace.paths.join(root, 'absent'))

    async def test_stat_and_list_dir(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            child = workspace.paths.join(root, 'child')
            path = workspace.paths.join(child, 'data.bin')
            await workspace.write_bytes(path, b'data')
            file_entry = await workspace.stat(path)
            assert (file_entry.name, file_entry.path, file_entry.is_dir) == ('data.bin', path, False)
            assert file_entry.size in (None, 4)
            dir_entry = await workspace.stat(child)
            assert (dir_entry.name, dir_entry.path, dir_entry.is_dir) == ('child', child, True)
            entries = await workspace.list_dir(root)
            assert [(entry.name, entry.path, entry.is_dir) for entry in entries] == [('child', child, True)]

    async def test_make_dir_creates_parents_and_is_idempotent(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = workspace.paths.join(root, 'a', 'b')
            await workspace.make_dir(path)
            await workspace.make_dir(path)
            assert (await workspace.stat(path)).is_dir

    async def test_commands_and_files_share_one_environment(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = workspace.paths.join(root, 'shared.txt')
            await workspace.write_bytes(path, b'in\n')
            result = await _script(
                backend,
                'IFS= read -r value < "$1" && [ "$value" = in ] && printf "out\\n" > "$1"',
                'if ([IO.File]::ReadAllText($a0) -ne "in`n") { exit 1 }; [IO.File]::WriteAllText($a0, "out`n")',
                path,
            )
            assert result.exit_code == 0
            assert await workspace.read_bytes(path) == b'out\n'

    async def test_missing_paths_raise_file_not_found(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            missing = workspace.paths.join(root, 'missing')
            for operation in (workspace.read_bytes, workspace.stat, workspace.list_dir, workspace.remove):
                with pytest.raises(FileNotFoundError):
                    await operation(missing)

    async def test_reading_a_directory_raises_is_a_directory(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            with pytest.raises(IsADirectoryError):
                await workspace.read_bytes(root)

    async def test_listing_a_file_raises_not_a_directory(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            file = workspace.paths.join(root, 'file')
            await workspace.write_bytes(file, b'x')
            with pytest.raises(NotADirectoryError):
                await workspace.list_dir(file)

    async def test_writing_to_a_directory_raises_is_a_directory(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            with pytest.raises(IsADirectoryError):
                await workspace.write_bytes(root, b'data')

    @pytest.fixture
    def has_real_posix_shell(self) -> bool:
        """Only a test double with no POSIX process/filesystem can opt out."""
        return True

    async def test_symlink_loop_does_not_break_listing(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        if not has_real_posix_shell:
            pytest.skip('in-memory fake cannot create symlinks')
        _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            loop = workspace.paths.join(root, 'loop')
            if not await _symlink(backend, 'loop', loop):
                pytest.skip('the environment cannot create symlinks')  # pragma: no cover
            entries = await workspace.list_dir(root)
            assert [(entry.name, entry.is_dir) for entry in entries] == [('loop', False)]

    async def test_fifo_read_does_not_wait_for_writer(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        if not has_real_posix_shell or _windows(backend):
            pytest.skip('in-memory fakes and Windows have no FIFOs')
        commands = _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            fifo = workspace.paths.join(root, 'fifo')
            if (await commands.run(['mkfifo', fifo])).exit_code != 0:
                pytest.skip('the environment does not provide `mkfifo`')  # pragma: no cover
            with anyio.fail_after(30):
                with pytest.raises(OSError):
                    await workspace.read_bytes(fifo)

    async def test_reading_a_device_raises_os_error(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        """Only a regular file has an end to read to; a device such as `/dev/zero` has none.

        `/dev/null` ends at once, so a backend that reads devices fails this rule instead of hanging on it.
        """
        if not has_real_posix_shell or _windows(backend):
            pytest.skip('in-memory fakes and Windows have no device files')
        with anyio.fail_after(30):
            with pytest.raises(OSError):
                await Workspace(backend).read_bytes('/dev/null')

    @pytest.fixture
    def enforces_parent_file_errors(self) -> bool:
        """Opt out only for an in-memory test double without real path traversal."""
        return True

    @pytest.fixture
    def filesystem_honors_shell_permissions(self) -> bool:
        """Override only for provider file APIs that bypass the command user's permissions (e.g. E2B envd)."""
        return True

    async def test_permission_denied_uses_builtin_error(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool, filesystem_honors_shell_permissions: bool
    ) -> None:
        if not has_real_posix_shell or not filesystem_honors_shell_permissions:
            pytest.skip('in-memory fake has no permissions')
        if _windows(backend):
            pytest.skip('Windows permissions are access control lists, which `chmod` does not set')
        commands = _commands(backend)
        if (await commands.run(['id', '-u'])).stdout.strip() == '0':
            pytest.skip('root bypasses filesystem permissions')  # pragma: no cover
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            file = workspace.paths.join(root, 'unreadable')
            await workspace.write_bytes(file, b'data')
            assert (await commands.run(['chmod', '000', file])).exit_code == 0
            with pytest.raises(PermissionError):
                await workspace.read_bytes(file)
            with pytest.raises(PermissionError):
                await workspace.write_bytes(file, b'changed')

    async def test_file_as_parent_raises_not_a_directory(
        self, backend: WorkspaceBackend, enforces_parent_file_errors: bool
    ) -> None:
        if not enforces_parent_file_errors:
            pytest.skip('in-memory fake has no real path traversal')
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            file = workspace.paths.join(root, 'file')
            await workspace.write_bytes(file, b'data')
            with pytest.raises(NotADirectoryError):
                await workspace.write_bytes(workspace.paths.join(file, 'child'), b'data')
            with pytest.raises(NotADirectoryError):
                await workspace.make_dir(workspace.paths.join(file, 'child'))

    async def test_making_a_directory_over_a_file_raises_file_exists(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = workspace.paths.join(root, 'file')
            await workspace.write_bytes(path, b'data')
            with pytest.raises(FileExistsError):
                await workspace.make_dir(path)

    async def test_remove_deletes_a_file_or_a_tree(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            file, tree = workspace.paths.join(root, 'file'), workspace.paths.join(root, 'tree')
            await workspace.write_bytes(file, b'x')
            await workspace.write_bytes(workspace.paths.join(tree, 'nested', 'file'), b'x')
            await workspace.remove(file)
            await workspace.remove(tree)
            assert not await workspace.exists(file) and not await workspace.exists(tree)

    async def test_remove_refuses_the_working_dir_and_its_ancestors(self, backend: WorkspaceBackend) -> None:
        # A model asking to remove `.` must not wipe the environment it works in.
        workspace = Workspace(backend)
        working_dir = await workspace.working_dir()
        for path in ('.', working_dir, workspace.paths.dirname(working_dir), '/'):
            with pytest.raises(ValueError):
                await workspace.remove(path)
        assert await workspace.exists(working_dir)

    async def test_realpath_and_entries_follow_symlinks(self, backend: WorkspaceBackend) -> None:
        _commands(backend)
        workspace = Workspace(backend)
        join = workspace.paths.join
        async with _scratch_dir(workspace) as root:
            target, link = join(root, 'target'), join(root, 'link')
            await workspace.make_dir(target)
            if not await _symlink(backend, target, link, directory=True) or not await workspace.exists(link):
                pytest.skip('the environment cannot create symlinks')  # pragma: no cover
            assert await workspace.realpath(join(link, 'missing')) == join(target, 'missing')
            a = join(root, 'a')
            await workspace.make_dir(a)
            await workspace.make_dir(join(a, 'b'))
            for name, target_name in (('rel', 'b'), ('up', '../out'), ('loop1', 'loop2'), ('loop2', 'loop1')):
                assert await _symlink(backend, target_name, join(a, name), directory=True)
            assert await workspace.realpath(join(a, 'rel', 'missing')) == join(a, 'b', 'missing')
            assert await workspace.realpath(join(a, 'up', 'missing')) == join(root, 'out', 'missing')
            assert await workspace.realpath(join(a, 'rel', '..')) == a
            # Windows takes `..` off the path before following links; POSIX climbs from the link's target.
            assert await workspace.realpath(join(a, 'up', '..')) == (a if _windows(backend) else root)
            with anyio.fail_after(30):
                try:
                    loop_path = await workspace.realpath(join(a, 'loop1', 'q'))
                except OSError:
                    pass
                else:
                    assert loop_path.startswith(join(a, ''))
            assert (await workspace.stat(link)).is_dir
            assert {entry.name: entry.is_dir for entry in await workspace.list_dir(root)} == {
                'a': True,
                'link': True,
                'target': True,
            }

    async def test_writes_go_through_a_symlink(self, backend: WorkspaceBackend) -> None:
        _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            target, link = workspace.paths.join(root, 'target'), workspace.paths.join(root, 'link')
            await workspace.write_bytes(target, b'old')
            if not await _symlink(backend, target, link) or not await workspace.exists(link):
                pytest.skip('the environment cannot create symlinks')  # pragma: no cover
            await workspace.write_bytes(link, b'new')
            assert await workspace.read_bytes(target) == b'new'
            assert await workspace.realpath(link) == target

    async def test_native_realpath_keeps_the_working_dir_and_missing_names(self, backend: WorkspaceBackend) -> None:
        """Checked without commands, so a filesystem-only backend's `realpath` is covered too."""
        if not isinstance(backend, SupportsRealpath):
            pytest.skip('backend does not implement SupportsRealpath')
        working_dir = await backend.working_dir()
        assert await backend.realpath(working_dir) == working_dir
        missing = Workspace(backend).paths.join(working_dir, f'.pydantic-ai-conformance-missing-{uuid.uuid4().hex}')
        assert await backend.realpath(missing) == missing

    async def test_a_backend_attached_by_ref_reaches_the_same_environment(
        self, backend: WorkspaceBackend, attach_backend: Callable[[WorkspaceRef], WorkspaceBackend] | Unsupported | None
    ) -> None:
        """Durable execution rebuilds the backend from its ref for every call, so this is what it relies on."""
        attach = _provided('attach_backend', attach_backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = workspace.paths.join(root, 'file')
            await workspace.write_bytes(path, b'reattached')
            ref = backend.ref
            assert ref is not None
            attached = attach(ref)
            assert await Workspace(attached).read_bytes(path) == b'reattached'
            if isinstance(attached, SupportsCommands):
                result = await _script(attached, 'cat "$1"', '[Console]::Out.Write([IO.File]::ReadAllText($a0))', path)
                assert result.stdout == 'reattached'
            # Attaching never replaces the environment it names.
            assert attached.ref == ref

    async def test_a_backend_attached_by_ref_reaches_a_detached_process(
        self,
        backend: WorkspaceBackend,
        attach_backend: Callable[[WorkspaceRef], WorkspaceBackend] | Unsupported | None,
        has_real_posix_shell: bool,
    ) -> None:
        """A process a command detached into its own session is still running for a backend attached by ref."""
        attach = _provided('attach_backend', attach_backend)
        if not has_real_posix_shell:
            pytest.skip('backend does not run real POSIX processes')
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            release, finished, started = (
                workspace.paths.join(root, name) for name in ('release', 'finished', 'started')
            )
            # The launcher returns once the process is in its own session: a backend may stop the command's own group.
            launch = (
                'command -v setsid > /dev/null || exit 127; '
                f'setsid sh -c \': > "$3"; {_FINISH_ONCE_RELEASED}\' sh "$@" < /dev/null > /dev/null 2>&1 & '
                'i=0; while [ ! -f "$3" ]; do i=$((i + 1)); [ "$i" -le 100 ] || exit 1; sleep 0.1; done'
            )
            powershell_launch = (
                _powershell_start(
                    f"[IO.File]::WriteAllText($a2, ''); {_POWERSHELL_FINISH_ONCE_RELEASED}",
                    release,
                    finished,
                    started,
                    window='-WindowStyle Hidden',
                )
                + '; $i = 0; while (-not (Test-Path -LiteralPath $a2)) '
                '{ $i++; if ($i -gt 300) { exit 1 }; Start-Sleep -Milliseconds 100 }'
            )
            try:
                with anyio.fail_after(60):  # Hang guard, not an assertion about remote speed.
                    result = await _script(backend, launch, powershell_launch, release, finished, started)
                if result.exit_code == 127:
                    pytest.skip('the environment does not provide `setsid`')  # pragma: no cover
                assert result.exit_code == 0, 'the detached process never started'
                ref = backend.ref
                assert ref is not None
                attached = Workspace(attach(ref))
                # Only a live process answers: a PID check would also pass for a zombie nothing reaps.
                with anyio.fail_after(60):
                    await attached.write_bytes(release, b'go')
                    while not await attached.exists(finished):
                        await anyio.sleep(0.1)
            finally:
                with anyio.move_on_after(30, shield=True):
                    await workspace.write_bytes(release, b'go')

    async def test_destroying_environment_during_command_raises_unavailable(
        self,
        destructive_backend: Callable[[], WorkspaceBackend] | Unsupported | None,
        destroy_environment: Callable[[WorkspaceBackend], Awaitable[None]] | Unsupported | None,
        has_real_posix_shell: bool,
    ) -> None:
        destroy = _provided('destroy_environment', destroy_environment)
        build = _provided('destructive_backend', destructive_backend)
        if not has_real_posix_shell:
            pytest.skip('backend has no running POSIX command to interrupt')
        backend = build()
        _commands(backend)
        workspace = Workspace(backend)
        root = await backend.working_dir()
        started = workspace.paths.join(root, f'.pydantic-ai-started-{uuid.uuid4().hex}')

        async def run_command() -> None:
            with pytest.raises(WorkspaceUnavailableError):
                # A local directory deletion does not kill a process already inside it; exit when
                # the directory disappears so the rule also checks the result classification.
                await _script(
                    backend,
                    'printf ready > "$1"; while [ -d "$2" ]; do sleep 0.1; done',
                    "[IO.File]::WriteAllText($a0, 'ready'); "
                    'while (Test-Path -LiteralPath $a1) { Start-Sleep -Milliseconds 100 }',
                    started,
                    root,
                    timeout=30,
                )

        async with anyio.create_task_group() as group:
            group.start_soon(run_command)
            with anyio.fail_after(30):
                while not await workspace.exists(started):
                    await anyio.sleep(0.05)
            await destroy(backend)

    async def test_attaching_to_a_destroyed_environment_raises_unavailable(
        self,
        destructive_backend: Callable[[], WorkspaceBackend] | Unsupported | None,
        attach_backend: Callable[[WorkspaceRef], WorkspaceBackend] | Unsupported | None,
        destroy_environment: Callable[[WorkspaceBackend], Awaitable[None]] | Unsupported | None,
    ) -> None:
        destroy = _provided('destroy_environment', destroy_environment)
        build = _provided('destructive_backend', destructive_backend)
        attach = _provided('attach_backend', attach_backend)
        # Never destroy the shared backend fixture: other rules may run after this one.
        backend = build()
        await backend.working_dir()
        assert backend.ref is not None
        await destroy(backend)
        with pytest.raises(WorkspaceUnavailableError):
            await attach(backend.ref).working_dir()

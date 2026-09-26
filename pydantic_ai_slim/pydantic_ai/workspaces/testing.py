"""Conformance tests for `WorkspaceBackend` implementations, public so third-party backends can run them.

Requires pytest and the anyio pytest plugin.
"""

from __future__ import annotations

import posixpath
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import anyio
import pytest

from .protocol import SupportsCommands, WorkspaceBackend, WorkspaceRef, WorkspaceTimeoutError, WorkspaceUnavailableError
from .workspace import Workspace

__all__ = ('WorkspaceBackendSuite',)


def _commands(backend: WorkspaceBackend) -> SupportsCommands:
    if not isinstance(backend, SupportsCommands):
        pytest.skip('backend does not implement SupportsCommands')
    return backend


@asynccontextmanager
async def _scratch_dir(workspace: Workspace) -> AsyncGenerator[str]:
    """A fresh directory under the working directory, removed afterwards."""
    root = posixpath.join(await workspace.working_dir(), f'.pydantic-ai-conformance-{uuid.uuid4().hex}')
    await workspace.make_dir(root)
    try:
        yield root
    finally:
        await workspace.remove(root)


class WorkspaceBackendSuite:
    """Subclass in your test suite and provide the `backend` fixture.

    Each test checks one rule of the backend contract. Command rules skip for a backend without
    `SupportsCommands`; filesystem rules run through [`Workspace`][pydantic_ai.workspaces.Workspace], so a
    command-only backend is checked on the file operations derived through its shell. The reattach
    rules need the optional fixtures below and skip without them.
    """

    pytestmark = pytest.mark.anyio

    @pytest.fixture
    def backend(self) -> WorkspaceBackend | AsyncIterator[WorkspaceBackend]:
        raise NotImplementedError('provide a `backend` fixture')

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend] | None:
        """Build a second backend that attaches to `ref`. Enables the reattach rules."""
        return None

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]] | None:
        """Destroy the environment behind `backend`. Enables the reattach-after-destroy rule."""
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
        result = await _commands(backend).run(['sh', '-c', 'read value || printf eof'], timeout=5)
        assert (result.exit_code, result.stdout) == (0, 'eof')

    async def test_missing_cwd_raises_file_not_found(self, backend: WorkspaceBackend) -> None:
        missing = posixpath.join(await backend.working_dir(), f'.pydantic-ai-missing-{uuid.uuid4().hex}')
        with pytest.raises(FileNotFoundError):
            await _commands(backend).run(['pwd'], cwd=missing)

    async def test_relative_cwd_is_rejected(self, backend: WorkspaceBackend) -> None:
        with pytest.raises(ValueError):
            await _commands(backend).run(['true'], cwd='relative')

    async def test_command_output_is_complete(self, backend: WorkspaceBackend) -> None:
        """If output cannot be collected in full, the backend must raise rather than return a truncated success."""
        output = 'workspace' * 1024
        result = await _commands(backend).run(
            ['sh', '-c', 'i=0; while [ "$i" -lt 1024 ]; do printf workspace; i=$((i+1)); done']
        )
        assert (result.exit_code, result.stdout) == (0, output)

    async def test_result_reports_exit_code_stdout_and_stderr(self, backend: WorkspaceBackend) -> None:
        """A non-zero exit is a normal result, not an error."""
        result = await _commands(backend).run('printf out; printf err >&2; exit 7', shell=True)
        assert (result.exit_code, result.stdout, result.stderr) == (7, 'out', 'err')

    async def test_a_missing_program_exits_127(self, backend: WorkspaceBackend) -> None:
        """Like `sh`, a program that doesn't exist is a normal result with exit code 127, not an error."""
        result = await _commands(backend).run(['pydantic-ai-conformance-missing-program'])
        assert result.exit_code == 127

    async def test_argv_items_are_literal(self, backend: WorkspaceBackend) -> None:
        payload = ' literal $() `quoted`; && '
        result = await _commands(backend).run(['sh', '-c', 'printf "%s" "$1"', 'sh', payload])
        assert (result.exit_code, result.stdout) == (0, payload)

    async def test_working_dir_is_canonical(self, backend: WorkspaceBackend) -> None:
        """Absolute, symlinks resolved, no `.`/`..`: the directory commands actually start in."""
        working_dir = await backend.working_dir()
        assert posixpath.isabs(working_dir) and posixpath.normpath(working_dir) == working_dir
        if isinstance(backend, SupportsCommands):
            assert (await backend.run(['sh', '-c', 'pwd -P'])).stdout == f'{working_dir}\n'

    async def test_timeout_raises_workspace_timeout_error(self, backend: WorkspaceBackend) -> None:
        with pytest.raises(WorkspaceTimeoutError):
            await _commands(backend).run(['sh', '-c', 'sleep 30'], timeout=1.0)

    async def test_env_is_added(self, backend: WorkspaceBackend) -> None:
        result = await _commands(backend).run(['sh', '-c', 'printf %s "$CONFORMANCE"'], env={'CONFORMANCE': 'value'})
        assert result.stdout == 'value'

    async def test_absolute_cwd_is_used(self, backend: WorkspaceBackend) -> None:
        assert (await _commands(backend).run(['sh', '-c', 'pwd -P'], cwd='/')).stdout == '/\n'

    async def test_ref_exists_after_the_first_operation_and_is_stable(self, backend: WorkspaceBackend) -> None:
        before = backend.ref
        await backend.working_dir()
        created = backend.ref
        await backend.working_dir()
        assert isinstance(created, WorkspaceRef) and backend.ref == created
        assert before in (None, created)

    async def test_large_file_round_trip(self, backend: WorkspaceBackend) -> None:
        """A shell-derived filesystem must page reads rather than hit a command-output cap."""
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = posixpath.join(root, 'large.bin')
            data = b'x' * (8 * 1024 * 1024)
            await workspace.write_bytes(path, data)
            assert await workspace.read_bytes(path) == data

    async def test_bytes_round_trip_and_write_creates_parents(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = posixpath.join(root, 'nested', 'data.bin')
            await workspace.write_bytes(path, b'\x00workspace\xff')
            assert await workspace.read_bytes(path) == b'\x00workspace\xff'
            await workspace.write_bytes(path, b'replaced')
            assert await workspace.read_bytes(path) == b'replaced'

    async def test_exists(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            await workspace.write_bytes(posixpath.join(root, 'file'), b'data')
            assert await workspace.exists(posixpath.join(root, 'file'))
            assert await workspace.exists(root)
            assert not await workspace.exists(posixpath.join(root, 'absent'))

    async def test_stat_and_list_dir(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            child = posixpath.join(root, 'child')
            path = posixpath.join(child, 'data.bin')
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
            path = posixpath.join(root, 'a', 'b')
            await workspace.make_dir(path)
            await workspace.make_dir(path)
            assert (await workspace.stat(path)).is_dir

    async def test_commands_and_files_share_one_environment(self, backend: WorkspaceBackend) -> None:
        commands = _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = posixpath.join(root, 'shared.txt')
            await workspace.write_bytes(path, b'in\n')
            script = 'IFS= read -r value < "$1" && [ "$value" = in ] && printf "out\\n" > "$1"'
            assert (await commands.run(['sh', '-c', script, 'sh', path])).exit_code == 0
            assert await workspace.read_bytes(path) == b'out\n'

    async def test_missing_paths_raise_file_not_found(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            missing = posixpath.join(root, 'missing')
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
            file = posixpath.join(root, 'file')
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
        commands = _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            loop = posixpath.join(root, 'loop')
            if (await commands.run(['ln', '-s', 'loop', loop])).exit_code != 0:
                pytest.skip('the environment cannot create symlinks with `ln -s`')
            entries = await workspace.list_dir(root)
            assert [(entry.name, entry.is_dir) for entry in entries] == [('loop', False)]

    async def test_fifo_read_does_not_wait_for_writer(
        self, backend: WorkspaceBackend, has_real_posix_shell: bool
    ) -> None:
        if not has_real_posix_shell:
            pytest.skip('in-memory fake cannot create FIFOs')
        commands = _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            fifo = posixpath.join(root, 'fifo')
            if (await commands.run(['mkfifo', fifo])).exit_code != 0:
                pytest.skip('the environment does not provide `mkfifo`')
            with anyio.fail_after(5):
                with pytest.raises(OSError):
                    await workspace.read_bytes(fifo)

    @pytest.fixture
    def enforces_parent_file_errors(self) -> bool:
        """Opt out only for an in-memory test double without real path traversal."""
        return True

    async def test_file_as_parent_raises_not_a_directory(
        self, backend: WorkspaceBackend, enforces_parent_file_errors: bool
    ) -> None:
        if not enforces_parent_file_errors:
            pytest.skip('in-memory fake has no real path traversal')
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            file = posixpath.join(root, 'file')
            await workspace.write_bytes(file, b'data')
            with pytest.raises(NotADirectoryError):
                await workspace.write_bytes(posixpath.join(file, 'child'), b'data')
            with pytest.raises(NotADirectoryError):
                await workspace.make_dir(posixpath.join(file, 'child'))

    async def test_making_a_directory_over_a_file_raises_file_exists(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = posixpath.join(root, 'file')
            await workspace.write_bytes(path, b'data')
            with pytest.raises(FileExistsError):
                await workspace.make_dir(path)

    async def test_remove_deletes_a_file_or_a_tree(self, backend: WorkspaceBackend) -> None:
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            file, tree = posixpath.join(root, 'file'), posixpath.join(root, 'tree')
            await workspace.write_bytes(file, b'x')
            await workspace.write_bytes(posixpath.join(tree, 'nested', 'file'), b'x')
            await workspace.remove(file)
            await workspace.remove(tree)
            assert not await workspace.exists(file) and not await workspace.exists(tree)

    async def test_realpath_and_entries_follow_symlinks(self, backend: WorkspaceBackend) -> None:
        commands = _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            target, link = posixpath.join(root, 'target'), posixpath.join(root, 'link')
            await workspace.make_dir(target)
            if (await commands.run(['ln', '-s', target, link])).exit_code != 0 or not await workspace.exists(link):
                pytest.skip('the environment cannot create symlinks with `ln -s`')
            assert await workspace.realpath(posixpath.join(link, 'missing')) == posixpath.join(target, 'missing')
            assert (await workspace.stat(link)).is_dir
            assert {entry.name: entry.is_dir for entry in await workspace.list_dir(root)} == {
                'link': True,
                'target': True,
            }

    async def test_writes_go_through_a_symlink(self, backend: WorkspaceBackend) -> None:
        commands = _commands(backend)
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            target, link = posixpath.join(root, 'target'), posixpath.join(root, 'link')
            await workspace.write_bytes(target, b'old')
            if (await commands.run(['ln', '-s', target, link])).exit_code != 0 or not await workspace.exists(link):
                pytest.skip('the environment cannot create symlinks with `ln -s`')
            await workspace.write_bytes(link, b'new')
            assert await workspace.read_bytes(target) == b'new'
            assert await workspace.realpath(link) == target

    async def test_a_backend_attached_by_ref_sees_the_same_files(
        self, backend: WorkspaceBackend, attach_backend: Callable[[WorkspaceRef], WorkspaceBackend] | None
    ) -> None:
        if attach_backend is None:
            pytest.skip('provide the `attach_backend` fixture to enable this rule')
        workspace = Workspace(backend)
        async with _scratch_dir(workspace) as root:
            path = posixpath.join(root, 'file')
            await workspace.write_bytes(path, b'reattached')
            assert backend.ref is not None
            assert await Workspace(attach_backend(backend.ref)).read_bytes(path) == b'reattached'

    async def test_attaching_to_a_destroyed_environment_raises_unavailable(
        self,
        backend: WorkspaceBackend,
        attach_backend: Callable[[WorkspaceRef], WorkspaceBackend] | None,
        destroy_environment: Callable[[WorkspaceBackend], Awaitable[None]] | None,
    ) -> None:
        if attach_backend is None or destroy_environment is None:
            pytest.skip('provide `attach_backend` and `destroy_environment` fixtures to enable this rule')
        await backend.working_dir()
        assert backend.ref is not None
        await destroy_environment(backend)
        with pytest.raises(WorkspaceUnavailableError):
            await attach_backend(backend.ref).working_dir()

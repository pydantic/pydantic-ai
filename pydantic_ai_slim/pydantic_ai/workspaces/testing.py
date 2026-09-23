"""Conformance suite for `WorkspaceBackend` implementations. Requires pytest and the anyio pytest plugin."""

from __future__ import annotations

import posixpath
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import TypeVar

import pytest

from .protocol import (
    SupportsCommands,
    SupportsFilesystem,
    WorkspaceBackend,
    WorkspaceFileEntry,
    WorkspaceRef,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)

__all__ = ('WorkspaceBackendSuite',)

_T = TypeVar('_T')


def _failure(rule: str, symptom: str) -> str:
    return f'{symptom}\nRule: "{rule}"'


def _commands(backend: WorkspaceBackend) -> SupportsCommands:
    if not isinstance(backend, SupportsCommands):
        pytest.skip('backend does not implement SupportsCommands')
    return backend


def _filesystem(backend: WorkspaceBackend) -> SupportsFilesystem:
    if not isinstance(backend, SupportsFilesystem):
        pytest.skip('backend does not implement SupportsFilesystem')
    return backend


async def _caught(action: Callable[[], Awaitable[object]]) -> Exception | None:
    error: Exception | None = None
    try:
        await action()
    except Exception as exc:
        error = exc
    return error


async def _checked(rule: str, action: Callable[[], Awaitable[_T]]) -> _T:
    try:
        return await action()
    except Exception as exc:
        raise AssertionError(_failure(rule, f'backend raised {type(exc).__name__}: {exc!s:.200}')) from exc


@asynccontextmanager
async def _probe(backend: WorkspaceBackend, filesystem: SupportsFilesystem, rule: str) -> AsyncGenerator[str]:
    working_dir = await _checked(rule, backend.working_dir)
    root = posixpath.join(working_dir, f'.pydantic-ai-conformance-{uuid.uuid4().hex}')
    await _checked(rule, lambda: filesystem.make_dir(root))
    try:
        yield root
    finally:
        try:
            await filesystem.remove(root)
        except FileNotFoundError:
            pass
        except Exception as exc:
            raise AssertionError(_failure(rule, f'probe cleanup raised {type(exc).__name__}: {exc!s:.200}')) from exc


class WorkspaceBackendSuite:
    """Subclass in your test suite and provide the `backend` fixture.

    Each test is one rule from the backend contract; its name states the rule and its
    assertion message quotes it. Command rules skip when the backend does not implement
    `SupportsCommands`; filesystem rules skip when it does not implement `SupportsFilesystem`.
    Lifecycle rules need the optional fixtures below and skip without them.
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

    async def test_required_members(self, backend: WorkspaceBackend) -> None:
        rule = 'Structural protocol: any object with these members conforms — no registration or base class required.'
        assert isinstance(backend, WorkspaceBackend), _failure(rule, 'backend does not provide the required members')

    async def test_string_command_requires_shell(self, backend: WorkspaceBackend) -> None:
        rule = (
            'Passing a `str` without `shell=True` is invalid, and so is an argv sequence with `shell=True`: '
            'implementations must reject either mismatch with a `TypeError`, forcing callers to be explicit about '
            'shell interpretation.'
        )
        commands = _commands(backend)
        error = await _caught(lambda: commands.run('true'))
        assert isinstance(error, TypeError), _failure(rule, f'string command raised {type(error).__name__}')

    async def test_argv_command_rejects_shell(self, backend: WorkspaceBackend) -> None:
        rule = (
            'Passing a `str` without `shell=True` is invalid, and so is an argv sequence with `shell=True`: '
            'implementations must reject either mismatch with a `TypeError`, forcing callers to be explicit about '
            'shell interpretation.'
        )
        commands = _commands(backend)
        error = await _caught(lambda: commands.run(['true'], shell=True))
        assert isinstance(error, TypeError), _failure(rule, f'argv command raised {type(error).__name__}')

    async def test_relative_cwd_is_rejected(self, backend: WorkspaceBackend) -> None:
        rule = (
            'Implementations must reject a relative path with `ValueError`: resolving it against ambient state '
            "(such as a local backend's host process working directory) would silently escape the workspace root."
        )
        commands = _commands(backend)
        error = await _caught(lambda: commands.run(['true'], cwd=f'relative-{uuid.uuid4().hex}'))
        assert isinstance(error, ValueError), _failure(rule, f'relative cwd raised {type(error).__name__}')

    async def test_shell_result_is_honest(self, backend: WorkspaceBackend) -> None:
        rule = (
            'The real exit code of the process. Non-zero is a normal result, not an error. '
            'Captured standard output. Captured standard error.'
        )
        commands = _commands(backend)
        token = uuid.uuid4().hex[:12]
        result = await _checked(rule, lambda: commands.run(f'printf {token}; printf {token} >&2; exit 7', shell=True))
        assert (result.exit_code, result.stdout, result.stderr) == (7, token, token), _failure(
            rule, f'command result was {result!r}'
        )

    async def test_argv_arguments_are_literal(self, backend: WorkspaceBackend) -> None:
        rule = 'In argv form, each item is passed as one literal argument and is never interpreted as shell source.'
        commands = _commands(backend)
        payload = f' literal {uuid.uuid4().hex} $() `quoted`; && '
        result = await _checked(rule, lambda: commands.run(['sh', '-c', 'printf "%s" "$1"', 'sh', payload]))
        assert (result.exit_code, result.stdout, result.stderr) == (0, payload, ''), _failure(
            rule, f'command result was {result!r}'
        )

    async def test_default_working_dir_is_canonical(self, backend: WorkspaceBackend) -> None:
        rule = (
            "The workspace's default working directory (absolute POSIX path). The path must be filesystem-canonical: "
            'symlinks resolved and no `.`/`..` segments.'
        )
        working_dir = await _checked(rule, backend.working_dir)
        assert (
            isinstance(working_dir, str)
            and posixpath.isabs(working_dir)
            and posixpath.normpath(working_dir) == working_dir
            and not working_dir.startswith('//')
        ), _failure(rule, f'working_dir was {working_dir!r}')
        if isinstance(backend, SupportsCommands):
            result = await _checked(rule, lambda: backend.run(['sh', '-c', 'pwd -P']))
            assert (result.exit_code, result.stdout, result.stderr) == (0, f'{working_dir}\n', ''), _failure(
                rule, f'pwd result was {result!r}'
            )

    async def test_timeout_raises_workspace_timeout_error(self, backend: WorkspaceBackend) -> None:
        rule = (
            'On expiry a [`WorkspaceTimeoutError`][pydantic_ai.workspaces.WorkspaceTimeoutError] is raised; '
            '`timeout` is the deadline that was enforced, which may be coarser than requested.'
        )
        commands = _commands(backend)
        timeout = 1.0
        error = await _caught(lambda: commands.run(['sh', '-c', 'sleep 30'], timeout=timeout))
        assert isinstance(error, WorkspaceTimeoutError), _failure(rule, f'timeout raised {type(error).__name__}')
        assert error.timeout is not None and error.timeout >= timeout, _failure(
            rule, f'timeout carried {error.timeout!r}, expected at least {timeout!r}'
        )

    async def test_env_is_added(self, backend: WorkspaceBackend) -> None:
        rule = 'Extra environment variables for the command.'
        commands = _commands(backend)
        name = f'PYDANTIC_AI_CONFORMANCE_{uuid.uuid4().hex[:12].upper()}'
        value = uuid.uuid4().hex
        result = await _checked(rule, lambda: commands.run(['sh', '-c', f'printf %s "${name}"'], env={name: value}))
        assert (result.exit_code, result.stdout, result.stderr) == (0, value, ''), _failure(
            rule, f'command result was {result!r}'
        )

    async def test_absolute_cwd_is_used(self, backend: WorkspaceBackend) -> None:
        rule = 'Absolute working directory for the command; defaults to the workspace working directory.'
        commands = _commands(backend)
        result = await _checked(rule, lambda: commands.run(['sh', '-c', 'pwd -P'], cwd='/'))
        assert (result.exit_code, result.stdout, result.stderr) == (0, '/\n', ''), _failure(
            rule, f'command result was {result!r}'
        )

    async def test_ref_is_stable_across_operations(self, backend: WorkspaceBackend) -> None:
        rule = 'Once assigned, a workspace reference is stable across operations.'
        await _checked(rule, backend.working_dir)
        first_ref = backend.ref
        await _checked(rule, backend.working_dir)
        assert isinstance(first_ref, WorkspaceRef) and backend.ref == first_ref, _failure(
            rule, f'ref changed from {first_ref!r} to {backend.ref!r}'
        )

    async def test_filesystem_bytes_round_trip(self, backend: WorkspaceBackend) -> None:
        rule = 'Write bytes to a file, creating missing parent directories and replacing existing contents.'
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            path = posixpath.join(root, 'nested', 'data.bin')
            first, second = b'\x00workspace\xff', b'replaced'
            await _checked(rule, lambda: filesystem.write_bytes(path, first))
            assert await _checked(rule, lambda: filesystem.read_bytes(path)) == first, _failure(
                rule, 'first write did not round-trip'
            )
            await _checked(rule, lambda: filesystem.write_bytes(path, second))
            assert await _checked(rule, lambda: filesystem.read_bytes(path)) == second, _failure(
                rule, 'replacement did not round-trip'
            )

    async def test_filesystem_exists_is_truthful(self, backend: WorkspaceBackend) -> None:
        rule = 'Whether a file or directory exists at the path.'
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            existing = posixpath.join(root, 'existing.bin')
            absent = posixpath.join(root, 'absent.bin')
            await _checked(rule, lambda: filesystem.write_bytes(existing, b'data'))
            assert await _checked(rule, lambda: filesystem.exists(existing)) is True, _failure(
                rule, 'existing file returned false'
            )
            assert await _checked(rule, lambda: filesystem.exists(root)) is True, _failure(
                rule, 'existing directory returned false'
            )
            assert await _checked(rule, lambda: filesystem.exists(absent)) is False, _failure(
                rule, 'absent path returned true'
            )

    async def test_filesystem_entries_are_truthful(self, backend: WorkspaceBackend) -> None:
        rule = 'Return truthful metadata from `stat`, and list directory entries non-recursively.'
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            data_dir = posixpath.join(root, 'child')
            data_path = posixpath.join(data_dir, 'data.bin')
            await _checked(rule, lambda: filesystem.write_bytes(data_path, b'data'))
            file_entry = await _checked(rule, lambda: filesystem.stat(data_path))
            dir_entry = await _checked(rule, lambda: filesystem.stat(data_dir))
            entries: Sequence[WorkspaceFileEntry] = await _checked(rule, lambda: filesystem.list_dir(root))
            assert (file_entry.name, file_entry.path, file_entry.is_dir) == (
                'data.bin',
                data_path,
                False,
            ) and file_entry.size in (None, 4), _failure(rule, f'file stat was {file_entry!r}')
            assert (dir_entry.name, dir_entry.path, dir_entry.is_dir) == (
                'child',
                data_dir,
                True,
            ) and (dir_entry.size is None or isinstance(dir_entry.size, int)), _failure(
                rule, f'directory stat was {dir_entry!r}'
            )
            assert [(entry.name, entry.path, entry.is_dir) for entry in entries] == [('child', data_dir, True)], (
                _failure(rule, f'directory listing was {entries!r}')
            )

    async def test_filesystem_make_dir_has_mkdir_p_semantics(self, backend: WorkspaceBackend) -> None:
        rule = 'Create a directory, including missing parents (`mkdir -p` semantics).'
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            path = posixpath.join(root, 'mkdir', 'a', 'b')
            await _checked(rule, lambda: filesystem.make_dir(path))
            await _checked(rule, lambda: filesystem.make_dir(path))
            assert (await _checked(rule, lambda: filesystem.stat(path))).is_dir, _failure(
                rule, 'created path was not a directory'
            )
            assert (await _checked(rule, lambda: filesystem.stat(posixpath.dirname(path)))).is_dir, _failure(
                rule, 'parent path was not a directory'
            )

    async def test_run_and_filesystem_share_one_environment(self, backend: WorkspaceBackend) -> None:
        rule = (
            '`run` executes against the same filesystem exposed by the filesystem methods: '
            'a file written through either is visible to the other.'
        )
        commands = _commands(backend)
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            path = posixpath.join(root, 'shared.txt')
            input_token, output_token = uuid.uuid4().hex, uuid.uuid4().hex
            await _checked(rule, lambda: filesystem.write_bytes(path, f'{input_token}\n'.encode()))
            result = await _checked(
                rule,
                lambda: commands.run(
                    [
                        'sh',
                        '-c',
                        'IFS= read -r value < "$1" && [ "$value" = "$2" ] && printf "%s\\n" "$3" > "$1"',
                        'sh',
                        path,
                        input_token,
                        output_token,
                    ]
                ),
            )
            assert result.exit_code == 0, _failure(rule, f'command result was {result!r}')
            assert await _checked(rule, lambda: filesystem.read_bytes(path)) == f'{output_token}\n'.encode(), _failure(
                rule, 'command and filesystem methods did not share the file'
            )

    async def test_filesystem_missing_paths_raise_file_not_found(self, backend: WorkspaceBackend) -> None:
        rule = (
            '`read_bytes`, `stat`, `list_dir`, and `remove` raise the builtin `FileNotFoundError` '
            'when the path does not exist.'
        )
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            actions: Mapping[str, Callable[[], Awaitable[object]]] = {
                'read_bytes': lambda: filesystem.read_bytes(posixpath.join(root, 'missing-read')),
                'stat': lambda: filesystem.stat(posixpath.join(root, 'missing-stat')),
                'list_dir': lambda: filesystem.list_dir(posixpath.join(root, 'missing-list')),
                'remove': lambda: filesystem.remove(posixpath.join(root, 'missing-remove')),
            }
            for operation, action in actions.items():
                error = await _caught(action)
                assert isinstance(error, FileNotFoundError), _failure(
                    rule, f'{operation} raised {type(error).__name__}'
                )

    async def test_filesystem_reading_directory_raises_is_a_directory(self, backend: WorkspaceBackend) -> None:
        rule = 'Reading a directory raises the builtin `IsADirectoryError`.'
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            path = posixpath.join(root, 'directory')
            await _checked(rule, lambda: filesystem.make_dir(path))
            error = await _caught(lambda: filesystem.read_bytes(path))
            assert isinstance(error, IsADirectoryError), _failure(rule, f'read_bytes raised {type(error).__name__}')

    async def test_filesystem_remove_file_and_tree(self, backend: WorkspaceBackend) -> None:
        rule = 'Remove a file, or a directory and its contents.'
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            file_path = posixpath.join(root, 'file')
            tree = posixpath.join(root, 'tree')
            nested_path = posixpath.join(tree, 'nested', 'file')
            await _checked(rule, lambda: filesystem.write_bytes(file_path, b'remove me'))
            await _checked(rule, lambda: filesystem.write_bytes(nested_path, b'remove me too'))
            await _checked(rule, lambda: filesystem.remove(file_path))
            assert not await _checked(rule, lambda: filesystem.exists(file_path)), _failure(
                rule, 'removed file still exists'
            )
            await _checked(rule, lambda: filesystem.remove(tree))
            assert not await _checked(rule, lambda: filesystem.exists(tree)), _failure(
                rule, 'removed directory still exists'
            )
            assert not await _checked(rule, lambda: filesystem.exists(nested_path)), _failure(
                rule, 'removed directory contents still exist'
            )

    async def test_ref_is_none_until_the_environment_exists_then_stable(self, backend: WorkspaceBackend) -> None:
        rule = (
            'A fresh backend may have no ref until its first operation; after the environment exists, '
            'the ref is non-None and stable.'
        )
        before = backend.ref
        await _checked(rule, backend.working_dir)
        created = backend.ref
        await _checked(rule, backend.working_dir)
        assert isinstance(created, WorkspaceRef), _failure(rule, f'ref after first operation was {created!r}')
        assert backend.ref == created, _failure(rule, f'ref changed from {created!r} to {backend.ref!r}')
        assert before is None or before == created, _failure(
            rule, f'configured ref changed from {before!r} to {created!r}'
        )

    async def test_reattach_by_ref_sees_the_same_files(
        self,
        backend: WorkspaceBackend,
        attach_backend: Callable[[WorkspaceRef], WorkspaceBackend] | None,
    ) -> None:
        rule = 'A backend attached by ref sees files written through the original backend.'
        if attach_backend is None:
            pytest.skip('provide the `attach_backend` fixture to enable this rule')
        filesystem = _filesystem(backend)
        async with _probe(backend, filesystem, rule) as root:
            path = posixpath.join(root, 'reattach.bin')
            await _checked(rule, lambda: filesystem.write_bytes(path, b'reattached'))
            ref = backend.ref
            assert isinstance(ref, WorkspaceRef), _failure(rule, f'backend ref was {ref!r}')
            attached = attach_backend(ref)
            attached_filesystem = _filesystem(attached)
            assert await _checked(rule, lambda: attached_filesystem.read_bytes(path)) == b'reattached', _failure(
                rule, 'attached backend did not see the file'
            )

    async def test_reattach_after_destroy_raises_unavailable(
        self,
        backend: WorkspaceBackend,
        attach_backend: Callable[[WorkspaceRef], WorkspaceBackend] | None,
        destroy_environment: Callable[[WorkspaceBackend], Awaitable[None]] | None,
    ) -> None:
        rule = 'After an environment is destroyed, attaching by ref and performing an operation raises `WorkspaceUnavailableError`.'
        if attach_backend is None or destroy_environment is None:
            pytest.skip('provide `attach_backend` and `destroy_environment` fixtures to enable this rule')
        filesystem = _filesystem(backend)
        working_dir = await _checked(rule, backend.working_dir)
        root = posixpath.join(working_dir, f'.pydantic-ai-conformance-{uuid.uuid4().hex}')
        await _checked(rule, lambda: filesystem.write_bytes(posixpath.join(root, 'destroyed.bin'), b'destroyed'))
        ref = backend.ref
        assert isinstance(ref, WorkspaceRef), _failure(rule, f'backend ref was {ref!r}')
        await _checked(rule, lambda: destroy_environment(backend))
        attached = attach_backend(ref)
        error = await _caught(attached.working_dir)
        assert isinstance(error, WorkspaceUnavailableError), _failure(
            rule, f'attached operation raised {type(error).__name__}'
        )

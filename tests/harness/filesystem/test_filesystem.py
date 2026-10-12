"""Tests for the FileSystem capability and FileSystemToolset."""

from __future__ import annotations

import errno
import os
import posixpath
import re
import stat
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path

import anyio
import anyio.lowlevel
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, LocalWorkspace, on_event
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import (
    FileEntry,
    LocalWorkspaceBackend,
    ReadOnlyWorkspace,
    UnavailableWorkspace,
    Workspace,
    WorkspaceBackend,
    WorkspaceError,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness import HarnessDeprecationWarning
from pydantic_ai_harness._workspace import READ_ONLY_FAILURE
from pydantic_ai_harness.filesystem import (
    FILE_SYSTEM_TOOL_NAMES,
    READ_ONLY_TOOL_NAMES,
    RIPGREP_TOOL_NAMES,
    FilesSearchedEvent,
    FileSystem,
)
from pydantic_ai_harness.filesystem._toolset import (
    _NOT_A_PATH,  # pyright: ignore[reportPrivateUsage]
    _OUTSIDE_WORKSPACE,  # pyright: ignore[reportPrivateUsage]
    FileSystemToolset,
    _content_hash,  # pyright: ignore[reportPrivateUsage]
    _format_lines,  # pyright: ignore[reportPrivateUsage]
    _is_binary,  # pyright: ignore[reportPrivateUsage]
    _sanitize_recoverable_error,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai_harness.shell._toolset import ShellToolset

from ...workspace_fakes import RunOnlyWorkspaceBackend
from .._tool_calls import call_tool, call_tools
from .conftest import tools_path


class ReadOnlyMount(LocalWorkspaceBackend):
    """A workspace whose environment refuses writes without advertising `read_only`, like a read-only mount."""

    async def write_bytes(self, path: str, data: bytes) -> None:
        raise WorkspaceReadOnlyError('read-only mount')

    async def make_dir(self, path: str) -> None:
        raise WorkspaceReadOnlyError('read-only mount')


class UntouchableWorkspace(WorkspaceBackend):
    """A sandbox-like workspace whose first operation would create a billed environment, so any operation fails."""

    @property
    def ref(self) -> None:
        return None

    async def working_dir(self) -> str:
        raise AssertionError('the workspace was touched')  # pragma: no cover


class CountingWorkspace(LocalWorkspaceBackend):
    """A local workspace that counts `working_dir` and `realpath` calls."""

    working_dir_calls = 0
    realpath_calls = 0

    async def working_dir(self) -> str:
        self.working_dir_calls += 1
        return await super().working_dir()

    async def realpath(self, path: str) -> str:
        self.realpath_calls += 1
        return await super().realpath(path)


class SymlinkedWorkingDir(LocalWorkspaceBackend):
    """A backend that reports its working directory through a symlink rather than canonically."""

    def __init__(self, link: Path) -> None:
        super().__init__(link)
        self.link = link

    async def working_dir(self) -> str:
        return self.link.as_posix()


class FailingWorkspace(LocalWorkspaceBackend):
    """A local workspace whose named operations raise a given error, on paths ending in `where`."""

    def __init__(self, working_dir: Path, failures: dict[str, Exception], *, where: str = '') -> None:
        super().__init__(working_dir)
        self.failures = failures
        self.where = where

    def _fail(self, operation: str, path: str) -> None:
        if (error := self.failures.get(operation)) is not None and path.endswith(self.where):
            raise error

    async def write_bytes(self, path: str, data: bytes) -> None:
        self._fail('write_bytes', path)
        await super().write_bytes(path, data)

    async def make_dir(self, path: str) -> None:
        self._fail('make_dir', path)
        await super().make_dir(path)

    async def read_bytes(self, path: str) -> bytes:
        self._fail('read_bytes', path)
        return await super().read_bytes(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        self._fail('list_dir', path)
        return await super().list_dir(path)

    async def realpath(self, path: str) -> str:
        self._fail('realpath', path)
        return await super().realpath(path)


class FilesystemOnlyWorkspace:
    """Storage without commands or file sizes in listings: the shape of an object-store backend.

    Custom storage plugs in as a workspace backend; the tools need only the filesystem methods.
    """

    def __init__(self, working_dir: Path | LocalWorkspaceBackend) -> None:
        self._local = (
            working_dir if isinstance(working_dir, LocalWorkspaceBackend) else LocalWorkspaceBackend(working_dir)
        )

    @property
    def ref(self) -> WorkspaceRef | None:
        return None

    async def working_dir(self) -> str:
        return await self._local.working_dir()

    async def read_bytes(self, path: str) -> bytes:
        return await self._local.read_bytes(path)

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self._local.write_bytes(path, data)

    async def stat(self, path: str) -> FileEntry:
        return await self._local.stat(path)

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return [replace(entry, size=None) for entry in await self._local.list_dir(path)]

    async def make_dir(self, path: str) -> None:
        await self._local.make_dir(path)

    async def remove(self, path: str) -> None:
        await self._local.remove(path)  # pragma: no cover -- no tool removes files

    async def exists(self, path: str) -> bool:
        return await self._local.exists(path)


def _reported_hash(result: str) -> str:
    """Extract the content hash a tool reports, from a `[hash:xxxx]` suffix."""
    return result.partition('hash:')[2].split()[0].rstrip(']')


class TestFormatLines:
    def test_basic_formatting(self) -> None:
        text = 'line1\nline2\nline3\n'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        assert '     1\tline1\n' in result
        assert '     2\tline2\n' in result
        assert '     3\tline3\n' in result

    def test_offset(self) -> None:
        text = 'a\nb\nc\nd\ne\n'
        result = _format_lines(text.splitlines(keepends=True), 2, 2)
        assert '     3\tc\n' in result
        assert '     4\td\n' in result
        assert '... (1 more lines. Use offset=4 to continue reading.)' in result

    def test_offset_exceeds_length(self) -> None:
        text = 'a\nb\n'
        with pytest.raises(ValueError, match='Offset 5 exceeds file length'):
            _format_lines(text.splitlines(keepends=True), 5, 10)

    def test_empty_file(self) -> None:
        result = _format_lines([], 0, 10)
        assert result == '(empty file)\n'

    def test_no_trailing_newline(self) -> None:
        text = 'no newline'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        assert result.endswith('\n')

    def test_continuation_hint(self) -> None:
        text = '\n'.join(f'line{i}' for i in range(10))
        result = _format_lines(text.splitlines(keepends=True), 0, 3)
        assert '... (7 more lines. Use offset=3 to continue reading.)' in result


class TestIsBinary:
    def test_text_content(self) -> None:
        assert _is_binary(b'hello world\n') is False

    def test_binary_content(self) -> None:
        assert _is_binary(b'hello\x00world') is True

    def test_null_after_sample(self) -> None:
        data = b'x' * 9000 + b'\x00'
        assert _is_binary(data) is False

    def test_null_at_boundary(self) -> None:
        data = b'x' * 8191 + b'\x00'
        assert _is_binary(data) is True

    def test_empty(self) -> None:
        assert _is_binary(b'') is False


class TestContentHash:
    def test_deterministic(self) -> None:
        assert _content_hash('hello') == _content_hash('hello')

    def test_different_content(self) -> None:
        assert _content_hash('hello') != _content_hash('world')

    def test_length(self) -> None:
        assert len(_content_hash('test')) == 12


@pytest.fixture
def fs_root(tmp_path: Path) -> Path:
    (tmp_path / 'hello.txt').write_text('Hello, world!\n')
    (tmp_path / 'multi.txt').write_text('line1\nline2\nline3\nline4\nline5\n')
    (tmp_path / 'subdir').mkdir()
    (tmp_path / 'subdir' / 'nested.py').write_text('print("nested")\n')
    (tmp_path / '.hidden').write_text('secret\n')
    (tmp_path / 'binary.bin').write_bytes(b'\x00\x01\x02\x03')
    (tmp_path / '.git').mkdir()
    (tmp_path / '.git' / 'config').write_text('[core]\n')
    (tmp_path / '.env').write_text('SECRET_KEY=abc123\n')
    return tmp_path


@pytest.fixture
def ws(tmp_path: Path) -> LocalWorkspaceBackend:
    """The workspace for direct calls, whose working directory is the root the toolsets here use."""
    return LocalWorkspaceBackend(tmp_path)


@pytest.fixture
def toolset(fs_root: Path) -> FileSystemToolset[None]:
    return FileSystemToolset(
        root_dir=fs_root,
        allowed_patterns=[],
        denied_patterns=[],
        read_only_patterns=['.git/*', '.env', '.env.*'],
        max_read_lines=2000,
        max_list_results=1000,
        max_search_results=1000,
        max_find_results=1000,
    )


@pytest.fixture
def outside(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory outside every root, holding `secret.txt`."""
    directory = tmp_path_factory.mktemp('outside')
    (directory / 'secret.txt').write_text('escaped!\n')
    return directory


class TestPathSecurity:
    async def test_traversal_with_dotdot(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(PermissionError, match='is outside root_dir'):
            await toolset._resolve_path(await toolset._scope(ws), '../../../etc/passwd')  # pyright: ignore[reportPrivateUsage]

    async def test_traversal_absolute_path(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(PermissionError, match=r'`/etc/passwd` is outside root_dir `.*`; .*Create or clone'):
            await toolset._resolve_path(await toolset._scope(ws), '/etc/passwd')  # pyright: ignore[reportPrivateUsage]

    async def test_traversal_encoded(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(PermissionError, match='is outside root_dir'):
            await toolset._resolve_path(await toolset._scope(ws), 'subdir/../../..')  # pyright: ignore[reportPrivateUsage]

    async def test_symlinked_directory_leading_outside_is_refused(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend, outside: Path
    ) -> None:
        (fs_root / 'escape').symlink_to(outside)
        refusal = r'`escape/secret.txt` leads outside root_dir `.*` through a symlink.*Use a path inside `'
        with pytest.raises(ModelRetry, match=refusal) as refused:
            await toolset.read_file('escape/secret.txt', workspace=ws)
        assert str(outside) not in str(refused.value)
        with pytest.raises(ModelRetry, match='leads outside root_dir'):
            await toolset.write_file('escape/new.txt', 'x', workspace=ws)
        assert not (outside / 'new.txt').exists()

    async def test_symlink_to_outside_fails_closed_when_readlink_is_missing(
        self,
        toolset: FileSystemToolset[None],
        fs_root: Path,
        outside: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        (fs_root / 'escape').symlink_to(outside)
        path = tools_path(tmp_path_factory.mktemp('bin'), exclude=frozenset({'readlink'}))
        workspace = RunOnlyWorkspaceBackend(LocalWorkspaceBackend(fs_root, env={'PATH': path}))

        with pytest.raises(ToolFailed, match='readlink'):
            await toolset.read_file('escape/secret.txt', workspace=workspace)

    async def test_symlink_to_a_protected_file_is_protected(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'envlink').symlink_to(fs_root / '.env')
        with pytest.raises(ModelRetry, match=r"'.env' is protected"):
            await toolset.write_file('envlink', 'HACKED=1\n', workspace=ws)

    @pytest.mark.parametrize('parent', ['', 'nested/'])
    async def test_harness_metadata_is_read_only_by_default(
        self, fs_root: Path, ws: LocalWorkspaceBackend, parent: str
    ) -> None:
        path = f'{parent}.pydantic-ai-harness/shell/job/status.json'
        status = fs_root / path
        status.parent.mkdir(parents=True)
        status.write_text('{"pid": 1, "exit_code": null}')
        toolset = FileSystem[None]().get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        assert 'exit_code' in await toolset.read_file(path, workspace=ws)
        with pytest.raises(ModelRetry, match='is protected'):
            await toolset.write_file(path, '{}', workspace=ws)
        assert 'null' in status.read_text()

    async def test_root_at_the_filesystem_root_follows_symlinks_anywhere(
        self, fs_root: Path, ws: LocalWorkspaceBackend, outside: Path
    ) -> None:
        (fs_root / 'escape').symlink_to(outside)
        toolset = FileSystem[None](root_dir='/').get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        assert 'escaped!' in await toolset.read_file('escape/secret.txt', workspace=ws)
        assert (await toolset.search_files('escaped', workspace=FilesystemOnlyWorkspace(fs_root))).splitlines()[
            0
        ] == 'escape/secret.txt:1:escaped!'

    async def test_root_at_the_filesystem_root_still_matches_patterns_through_symlinks(
        self, fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'token.secret').write_text('original\n')
        (fs_root / 'alias.txt').symlink_to(fs_root / 'token.secret')
        toolset = FileSystem[None](root_dir='/', denied_patterns=['**/*.secret']).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        with pytest.raises(ModelRetry, match=r"denied by pattern '\*\*/\*.secret'"):
            await toolset.write_file('alias.txt', 'overwritten\n', workspace=ws)
        assert (fs_root / 'token.secret').read_text() == 'original\n'

    async def test_root_at_the_filesystem_root_without_patterns_resolves_no_symlinks(self, fs_root: Path) -> None:
        toolset = FileSystem[None](root_dir='/', read_only_patterns=[]).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = CountingWorkspace(fs_root)
        await toolset.write_file('new.txt', 'x\n', workspace=workspace)
        assert 'x' in await toolset.read_file('new.txt', workspace=workspace)
        assert (await toolset.search_files('^x$', workspace=workspace)).splitlines()[0] == 'new.txt:1:x'
        assert (
            workspace.realpath_calls == 0
        )  # command-backed search needs no per-result realpath under an unrestricted root

    async def test_root_at_the_filesystem_root_without_patterns_walks_without_resolving_files(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / 'a.txt').write_text('x\n')
        (tmp_path / 'b.txt').write_text('x\n')
        toolset = FileSystem[None](root_dir='/', read_only_patterns=[]).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = CountingWorkspace(tmp_path)
        # A read-only workspace runs no commands, so the search walks the files itself.
        result = await toolset.search_files('^x$', workspace=ReadOnlyWorkspace(Workspace(workspace)))
        assert result == 'a.txt:1:x\nb.txt:1:x'
        assert workspace.realpath_calls == 1  # the walk's start directory, and no file it reads

    async def test_symlinked_working_directory_is_the_default_root(self, tmp_path: Path) -> None:
        (tmp_path / 'real').mkdir()
        (tmp_path / 'link').symlink_to(tmp_path / 'real')
        toolset = FileSystem[None]().get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = SymlinkedWorkingDir(tmp_path / 'link')
        await toolset.write_file('a.txt', 'written\n', workspace=workspace)
        assert (tmp_path / 'real' / 'a.txt').read_text() == 'written\n'

    async def test_valid_path_resolves(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        result = (await toolset._resolve_path(await toolset._scope(ws), 'hello.txt'))[0]  # pyright: ignore[reportPrivateUsage]
        assert result == str(fs_root / 'hello.txt')

    def test_first_matching_pattern_match(self, toolset: FileSystemToolset[None]) -> None:
        result = toolset._first_matching_pattern('secret.key', ['*.txt', '*.key'])  # pyright: ignore[reportPrivateUsage]
        assert result == '*.key'

    def test_first_matching_pattern_no_match(self, toolset: FileSystemToolset[None]) -> None:
        result = toolset._first_matching_pattern('readme.md', ['*.txt', '*.key'])  # pyright: ignore[reportPrivateUsage]
        assert result is None

    def test_first_matching_pattern_empty(self, toolset: FileSystemToolset[None]) -> None:
        result = toolset._first_matching_pattern('anything.py', [])  # pyright: ignore[reportPrivateUsage]
        assert result is None

    async def test_nested_path_resolves(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = (await toolset._resolve_path(await toolset._scope(ws), 'subdir/nested.py'))[0]  # pyright: ignore[reportPrivateUsage]
        assert result.endswith('/subdir/nested.py')


class TestRootDir:
    """`root_dir` is resolved on the run's first file operation, against the run's workspace."""

    @pytest.fixture
    def anyio_backend(self) -> str:
        # Agent.run needs asyncio.
        return 'asyncio'

    async def test_root_above_the_working_directory_reaches_a_sibling(self, tmp_path: Path) -> None:
        (tmp_path / 'project').mkdir()
        (tmp_path / 'shared').mkdir()
        (tmp_path / 'shared' / 'notes.txt').write_text('shared notes\n')
        result = await call_tool(
            [FileSystem[None](root_dir='..')],
            'read_file',
            {'path': '../shared/notes.txt'},
            workspace=LocalWorkspaceBackend(tmp_path / 'project'),
        )
        assert 'shared notes' in result

    @pytest.mark.parametrize('root_dir', ['src', '../elsewhere'])
    def test_relative_root_that_cannot_contain_the_working_directory_is_rejected(self, root_dir: str) -> None:
        with pytest.raises(UserError, match=r'root_dir must contain the working directory.*attach the workspace'):
            FileSystem[None](root_dir=root_dir)

    async def test_relative_root_at_the_working_directory_is_the_default(self, fs_root: Path) -> None:
        capabilities = [FileSystem[None](root_dir='src/..')]
        read = await call_tool(
            capabilities, 'read_file', {'path': 'hello.txt'}, workspace=LocalWorkspaceBackend(fs_root)
        )
        assert 'Hello' in read

    async def test_absolute_root_below_the_working_directory_fails_the_first_file_operation(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / 'src').mkdir()
        capabilities = [FileSystem[None](root_dir=tmp_path / 'src')]
        assert await call_tools(capabilities, [], workspace=LocalWorkspaceBackend(tmp_path)) == []
        with pytest.raises(
            UserError, match=r"The working directory '.*' is outside root_dir '.*/src'.*`LocalWorkspace\('.*/src'\)`"
        ):
            await call_tool(capabilities, 'list_directory', {}, workspace=LocalWorkspaceBackend(tmp_path))

    async def test_root_reached_through_a_symlink_holds_the_working_directory(self, tmp_path: Path) -> None:
        (tmp_path / 'real').mkdir()
        (tmp_path / 'real' / 'a.txt').write_text('through the link\n')
        (tmp_path / 'link').symlink_to(tmp_path / 'real')
        result = await call_tool(
            [FileSystem[None](root_dir=tmp_path / 'link')],
            'read_file',
            {'path': 'a.txt'},
            workspace=SymlinkedWorkingDir(tmp_path / 'link'),
        )
        assert result.endswith('1\tthrough the link\n')

    async def test_the_boundary_is_resolved_once_per_run(self, tmp_path: Path) -> None:
        (tmp_path / 'a.txt').write_text('a\n')
        workspace = CountingWorkspace(tmp_path)
        read: tuple[str, dict[str, object]] = ('read_file', {'path': 'a.txt'})
        await call_tools([FileSystem[None]()], [read, read], workspace=workspace)
        assert workspace.working_dir_calls == 1

    async def test_each_run_resolves_the_boundary_in_its_own_workspace(self, tmp_path: Path) -> None:
        # One toolset serves every run, so durable execution sees the toolset it registered.
        capability = FileSystem[None]()
        assert capability.get_toolset() is capability.get_toolset()
        for name in ('first', 'second'):
            (tmp_path / name).mkdir()
            (tmp_path / name / 'a.txt').write_text(f'{name}\n')
        first, second = [
            await call_tool(
                [capability], 'read_file', {'path': 'a.txt'}, workspace=LocalWorkspaceBackend(tmp_path / name)
            )
            for name in ('first', 'second')
        ]
        assert first.endswith('1\tfirst\n')
        assert second.endswith('1\tsecond\n')

    async def test_a_run_without_file_operations_does_no_workspace_io(self) -> None:
        assert await call_tools([FileSystem[None](root_dir='/srv')], [], workspace=UntouchableWorkspace()) == []

    async def test_no_workspace_fails_the_run(self) -> None:
        with pytest.raises(UserError, match='`FileSystem` needs a workspace'):
            await call_tool([FileSystem[None]()], 'list_directory', {})

    async def test_no_workspace_names_the_one_the_history_continues_in(self, tmp_path: Path) -> None:
        capabilities: list[AbstractCapability[object]] = [LocalWorkspace(tmp_path), FileSystem[object]()]
        first = await Agent(TestModel(call_tools=[]), capabilities=capabilities).run('go')
        agent = Agent(TestModel(call_tools=[]), capabilities=[FileSystem[object]()])
        local = f'local:{tmp_path}'
        with pytest.raises(UserError, match=re.escape(f'history continues in workspace `{local}`')) as error:
            await agent.run('again', message_history=first.all_messages())
        assert f'(such as `LocalWorkspace({str(tmp_path)!r})`)' in str(error.value)

        history: list[ModelMessage] = [
            ModelResponse(parts=[TextPart('done')], workspace_ref=WorkspaceRef(provider='modal', id='sb-1'))
        ]
        with pytest.raises(UserError) as error:
            await agent.run('again', message_history=history)
        assert str(error.value) == (
            '`FileSystem` needs a workspace. The message history continues in workspace `modal:sb-1`; attach the '
            'capability that provides `modal` workspaces (such as `ModalSandbox()`) to continue there, or pass '
            '`workspace=` to the run.'
        )

        # A provider the harness does not ship names no example capability.
        history = [ModelResponse(parts=[TextPart('done')], workspace_ref=WorkspaceRef(provider='memfs', id='m1'))]
        with pytest.raises(UserError, match=r'provides `memfs` workspaces to continue there'):
            await agent.run('again', message_history=history)

    async def test_unavailable_workspace_names_policy_reason(self) -> None:
        with pytest.raises(UserError, match=r'`FileSystem`.*disabled by policy') as error:
            await call_tool(
                [FileSystem[None]()], 'list_directory', {}, workspace=UnavailableWorkspace('disabled by policy')
            )
        assert 'Add `LocalWorkspace' not in str(error.value)


class TestAccessPatterns:
    async def test_denied_pattern_blocks(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        with pytest.raises(PermissionError, match='denied by pattern'):
            ts._check_access('data.secret')  # pyright: ignore[reportPrivateUsage]

    async def test_denied_pattern_passes_non_matching(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # Path that doesn't match any denied pattern should pass
        ts._check_access('data.txt')  # pyright: ignore[reportPrivateUsage]

    async def test_allowed_pattern_permits(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # Should not raise for .py files
        ts._check_access('test.py')  # pyright: ignore[reportPrivateUsage]

    async def test_allowed_pattern_blocks_non_matching(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        with pytest.raises(PermissionError, match='does not match any allowed'):
            ts._check_access('data.txt')  # pyright: ignore[reportPrivateUsage]

    async def test_protected_pattern_blocks_write(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(PermissionError, match='protected'):
            toolset._check_access('.git/config', write=True)  # pyright: ignore[reportPrivateUsage]

    async def test_protected_pattern_allows_read(self, toolset: FileSystemToolset[None]) -> None:
        # Should not raise for read
        toolset._check_access('.git/config', write=False)  # pyright: ignore[reportPrivateUsage]

    async def test_env_file_protected(self, toolset: FileSystemToolset[None]) -> None:
        with pytest.raises(PermissionError, match='protected'):
            toolset._check_access('.env', write=True)  # pyright: ignore[reportPrivateUsage]

    async def test_write_non_protected_with_patterns_configured(self, toolset: FileSystemToolset[None]) -> None:
        # write=True on a path that doesn't match any protected pattern should pass
        toolset._check_access('hello.txt', write=True)  # pyright: ignore[reportPrivateUsage]

    async def test_access_with_no_denied_patterns(self, fs_root: Path) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # No denied, no protected, no allowed → should pass for any path
        ts._check_access('anything.txt', write=True)  # pyright: ignore[reportPrivateUsage]

    async def test_no_patterns_allows_reads_and_writes(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'Hello, world!' in await ts.read_file('hello.txt', workspace=ws)
        assert 'Wrote' in await ts.write_file('hello.txt', 'rewritten\n', workspace=ws)

    async def test_protected_path_reads_but_rejects_writes(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'keys.pem').write_text('PRIVATE\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=['*.pem'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'PRIVATE' in await ts.read_file('keys.pem', workspace=ws)
        with pytest.raises(ModelRetry, match='protected'):
            await ts.write_file('keys.pem', 'HACKED\n', workspace=ws)

    async def test_denied_pattern_rejects_reads(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'creds.secret').write_text('hunter2\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'Hello, world!' in await ts.read_file('hello.txt', workspace=ws)
        with pytest.raises(ModelRetry, match='denied by pattern'):
            await ts.read_file('creds.secret', workspace=ws)

    async def test_allowed_patterns_reject_non_matching_reads(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'main.py').write_text('print("hi")\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'print' in await ts.read_file('main.py', workspace=ws)
        with pytest.raises(ModelRetry, match='does not match any allowed'):
            await ts.read_file('hello.txt', workspace=ws)


class TestReadFile:
    async def test_read_basic(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.read_file('hello.txt', workspace=ws)
        assert 'Hello, world!' in result
        assert 'hash:' in result
        assert '1 lines' in result

    async def test_read_with_offset(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.read_file('multi.txt', offset=2, workspace=ws)
        assert 'line3' in result
        assert 'line1' not in result

    @pytest.mark.parametrize(('offset', 'limit'), [(-1, None), (0, 0), (0, -1)])
    async def test_read_rejects_invalid_window(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend, offset: int, limit: int | None
    ) -> None:
        with pytest.raises(ModelRetry, match=r'offset|limit'):
            await toolset.read_file('multi.txt', offset=offset, limit=limit, workspace=ws)

    async def test_read_with_limit(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.read_file('multi.txt', limit=2, workspace=ws)
        assert 'line1' in result
        assert 'line2' in result
        assert '... (3 more lines' in result

    async def test_read_directory_returns_plain_result(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        assert await toolset.read_file('subdir', workspace=ws) == 'Path not found: subdir'

    async def test_read_missing_returns_plain_result(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        assert await toolset.read_file('nonexistent.txt', workspace=ws) == 'Path not found: nonexistent.txt'

    async def test_read_binary_file(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.read_file('binary.bin', workspace=ws)
        assert 'Binary file' in result
        assert '4 bytes' in result

    async def test_read_traversal_blocked(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry):
            await toolset.read_file('../../../etc/passwd', workspace=ws)


class TestWriteFile:
    @pytest.mark.parametrize(
        'path',
        [
            '.env',
            '.env.local',
            '.git/config',
            'server.pem',
            'deploy.key',
            'secrets.yaml',
            'apps/api/.env',
            'apps/.env.local',
            'sub/.git/config',
            'certs/server.pem',
            'config/deploy.key',
            'config/secrets.yaml',
        ],
    )
    async def test_default_patterns_protect_files_at_any_depth(
        self, fs_root: Path, ws: LocalWorkspaceBackend, path: str
    ) -> None:
        target = fs_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('original\n')
        toolset = FileSystem[None]().get_toolset()
        assert isinstance(toolset, FileSystemToolset)

        assert 'original' in await toolset.read_file(path, workspace=ws)
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.write_file(path, 'overwritten\n', workspace=ws)
        assert target.read_text() == 'original\n'

    @pytest.mark.parametrize('path', ['apps/env.example', 'apps/api/prod.env', 'docs/git/config.md'])
    async def test_default_patterns_leave_lookalikes_writable(
        self, fs_root: Path, ws: LocalWorkspaceBackend, path: str
    ) -> None:
        (fs_root / path).parent.mkdir(parents=True, exist_ok=True)
        toolset = FileSystem[None]().get_toolset()
        assert isinstance(toolset, FileSystemToolset)

        assert 'Wrote' in await toolset.write_file(path, 'content\n', workspace=ws)

    async def test_write_new_file(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        content = 'h\N{LATIN SMALL LETTER E WITH ACUTE}llo\r\nnew content\n'
        control = fs_root / 'control.txt'
        control.write_text(content, encoding='utf-8')

        result = await toolset.write_file('new.txt', content, workspace=ws)
        assert 'Wrote' in result
        target = fs_root / 'new.txt'
        assert target.read_bytes() == control.read_bytes()
        assert stat.S_IMODE(target.stat().st_mode) == stat.S_IMODE(control.stat().st_mode)

    async def test_write_existing_directory_retries(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match="Path 'subdir' exists and is not a regular file"):
            await toolset.write_file('subdir', 'content', workspace=ws)

    async def test_write_missing_parent_does_not_suggest_unregistered_tool(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry) as error:
            await toolset.write_file('missing/file.txt', 'hello', workspace=ws)
        assert 'create_directory' not in str(error.value)
        assert 'parent directory' in str(error.value).lower()

    async def test_write_nonexistent_parent_raises(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match="Parent directory 'deep/nested' does not exist"):
            await toolset.write_file('deep/nested/file.txt', 'deep\n', workspace=ws)

    async def test_write_overwrite(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        await toolset.write_file('hello.txt', 'overwritten\n', workspace=ws)
        assert (fs_root / 'hello.txt').read_text() == 'overwritten\n'

    @pytest.mark.skipif(os.name == 'nt', reason='POSIX mode bits are required.')
    async def test_write_overwrite_preserves_permissions(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        target = fs_root / 'hello.txt'
        target.chmod(0o640)
        await toolset.write_file('hello.txt', 'overwritten\n', workspace=ws)
        assert stat.S_IMODE(target.stat().st_mode) == 0o640

    async def test_write_through_internal_symlink(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        link = fs_root / 'link.txt'
        link.symlink_to('hello.txt')
        await toolset.write_file('link.txt', 'through link\n', workspace=ws)
        assert link.is_symlink()
        assert (fs_root / 'hello.txt').read_text() == 'through link\n'

    async def test_write_conflict_detection(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        # Get current hash
        content = (fs_root / 'hello.txt').read_text()
        current_hash = _content_hash(content)

        # Write with correct hash succeeds
        await toolset.write_file('hello.txt', 'updated\n', expected_hash=current_hash, workspace=ws)
        assert (fs_root / 'hello.txt').read_text() == 'updated\n'

    async def test_write_conflict_rejection(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match='Conflict'):
            await toolset.write_file('hello.txt', 'bad\n', expected_hash='wrong_hash_x', workspace=ws)

    async def test_write_protected_blocked(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.write_file('.env', 'HACKED=true\n', workspace=ws)

    async def test_write_returns_hash(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.write_file('hashed.txt', 'content\n', workspace=ws)
        assert 'hash:' in result


class TestEditFile:
    async def test_edit_basic(self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Hello, universe!', workspace=ws)
        assert 'Edited' in result
        assert (fs_root / 'hello.txt').read_text() == 'Hello, universe!\n'

    async def test_edit_not_found_text(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='old_text not found'):
            await toolset.edit_file('hello.txt', 'NONEXISTENT', 'replacement', workspace=ws)

    async def test_edit_ambiguous_match(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'repeat.txt').write_text('foo bar foo\n')
        with pytest.raises(ModelRetry, match='found 2 times'):
            await toolset.edit_file('repeat.txt', 'foo', 'baz', workspace=ws)

    async def test_edit_missing_file(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='File not found'):
            await toolset.edit_file('ghost.txt', 'x', 'y', workspace=ws)

    async def test_edit_conflict_detection(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        content = (fs_root / 'hello.txt').read_text()
        current_hash = _content_hash(content)
        result = await toolset.edit_file('hello.txt', 'Hello', 'Hi', expected_hash=current_hash, workspace=ws)
        assert 'hash:' in result

    async def test_edit_conflict_rejection(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='Conflict'):
            await toolset.edit_file('hello.txt', 'Hello', 'Hi', expected_hash='stale_hash_', workspace=ws)

    async def test_edit_protected_blocked(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.edit_file('.env', 'SECRET', 'HACKED', workspace=ws)

    async def test_edit_returns_new_hash(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Goodbye!', workspace=ws)
        assert 'hash:' in result


class HeldWriteWorkspace:
    """In-memory storage whose first write to `held` waits for `release`, after that call has read and checked the file.

    Every operation is a plain checkpoint, with no worker thread, so `wait_all_tasks_blocked` sees
    exactly where each call has stopped. `links` maps a symlink's path to its target's.
    """

    def __init__(self, files: dict[str, bytes], *, held: str, links: dict[str, str] | None = None) -> None:
        self.files = files
        self.held = held
        self.links = links or {}
        self.writing = anyio.Event()
        self.release = anyio.Event()

    @property
    def ref(self) -> WorkspaceRef | None:
        return None

    async def working_dir(self) -> str:
        return '/work'

    async def realpath(self, path: str) -> str:
        await anyio.lowlevel.checkpoint()
        return self.links.get(path, path)

    async def read_bytes(self, path: str) -> bytes:
        await anyio.lowlevel.checkpoint()
        return self.files[self.links.get(path, path)]

    async def write_bytes(self, path: str, data: bytes) -> None:
        if path == self.held and not self.writing.is_set():
            self.writing.set()
            await self.release.wait()
        await anyio.lowlevel.checkpoint()
        self.files[self.links.get(path, path)] = data

    async def stat(self, path: str) -> FileEntry:
        await anyio.lowlevel.checkpoint()
        path = self.links.get(path, path)
        if path in self.files:
            return FileEntry(name=posixpath.basename(path), path=path, is_dir=False, size=len(self.files[path]))
        if path == '/work':
            return FileEntry(name='work', path=path, is_dir=True, size=None)
        raise FileNotFoundError(path)  # pragma: no cover

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        raise NotImplementedError  # pragma: no cover

    async def make_dir(self, path: str) -> None:
        raise NotImplementedError  # pragma: no cover

    async def remove(self, path: str) -> None:
        raise NotImplementedError  # pragma: no cover

    async def exists(self, path: str) -> bool:
        raise NotImplementedError  # pragma: no cover


class TestConcurrentChanges:
    """Calls changing one file take turns, so none is computed from content another is about to replace."""

    async def test_parallel_edit_calls_to_one_file_both_land(self) -> None:
        """Two `edit_file` calls in one model response, as a model batching its edits sends them."""
        workspace = HeldWriteWorkspace({'/work/a.py': b'import old\nold()\n'}, held='/work/a.py')

        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if len(messages) > 1:
                return ModelResponse(parts=[TextPart('done')])
            return ModelResponse(
                parts=[
                    ToolCallPart('edit_file', {'path': 'a.py', 'old_text': 'import old', 'new_text': 'import new'}),
                    ToolCallPart('edit_file', {'path': 'a.py', 'old_text': 'old()', 'new_text': 'new()'}),
                ]
            )

        async def release_after_the_other_edit_stops() -> None:
            await workspace.writing.wait()
            # The other edit either finishes or waits its turn before the held write goes ahead.
            await anyio.wait_all_tasks_blocked()
            workspace.release.set()

        agent = Agent(FunctionModel(respond), capabilities=[FileSystem()])
        messages: list[ModelMessage] = []
        async with anyio.create_task_group() as tg:
            tg.start_soon(release_after_the_other_edit_stops)
            messages = (await agent.run('Rename old to new', workspace=workspace)).all_messages()

        returns = [part.content for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
        assert len(returns) == 2 and all(str(content).startswith('Edited a.py.') for content in returns)
        assert workspace.files['/work/a.py'] == b'import new\nnew()\n'

    async def test_edits_through_a_symlink_and_its_target_take_turns(self) -> None:
        toolset = FileSystem[None]().get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = HeldWriteWorkspace(
            {'/work/a.py': b'import old\nold()\n'}, held='/work/a.py', links={'/work/alias.py': '/work/a.py'}
        )

        async with anyio.create_task_group() as tg:
            tg.start_soon(partial(toolset.edit_file, 'a.py', 'import old', 'import new', workspace=workspace))
            await workspace.writing.wait()
            tg.start_soon(partial(toolset.edit_file, 'alias.py', 'old()', 'new()', workspace=workspace))
            # The edit through the link either finishes or waits its turn before the held edit goes ahead.
            await anyio.wait_all_tasks_blocked()
            workspace.release.set()

        assert workspace.files['/work/a.py'] == b'import new\nnew()\n'

    async def test_write_checked_against_a_hash_waits_for_a_pending_edit(self) -> None:
        """A write's `expected_hash` is checked after a pending edit lands, so it reports the conflict."""
        toolset = FileSystem[None]().get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = HeldWriteWorkspace({'/work/a.py': b'import old\n'}, held='/work/a.py')
        original_hash = _content_hash('import old\n')

        async def write() -> None:
            with pytest.raises(ModelRetry, match='Conflict'):
                await toolset.write_file('a.py', 'rewritten\n', expected_hash=original_hash, workspace=workspace)

        async with anyio.create_task_group() as tg:
            tg.start_soon(partial(toolset.edit_file, 'a.py', 'old', 'new', workspace=workspace))
            await workspace.writing.wait()
            tg.start_soon(write)
            # The write either finishes or waits its turn before the held edit goes ahead.
            await anyio.wait_all_tasks_blocked()
            workspace.release.set()

        assert workspace.files['/work/a.py'] == b'import new\n'


class TestContentHashNewlineAgreement:
    """`read_file`, `write_file`, and `edit_file` hashes must agree on `\\r\\n` files.

    Before the #821 fix, `edit_file` and `write_file` hashed a
    universal-newline-translated view while `read_file` hashed the raw bytes
    decoded without translation, so the optimistic-concurrency handshake
    rejected CRLF files that had not changed. All tools now hash the same
    canonical bytes-on-disk view.
    """

    async def test_crlf_read_edit_round_trip_hashes_agree(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'crlf.txt').write_bytes(b'line one\r\nline two\r\n')
        read_hash = _reported_hash(await toolset.read_file('crlf.txt', workspace=ws))

        edit_result = await toolset.edit_file('crlf.txt', 'line one', 'line ONE', expected_hash=read_hash, workspace=ws)
        edit_hash = _reported_hash(edit_result)

        assert _reported_hash(await toolset.read_file('crlf.txt', workspace=ws)) == edit_hash
        assert (fs_root / 'crlf.txt').read_bytes() == b'line ONE\r\nline two\r\n'

    async def test_crlf_edit_preserves_carriage_returns(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        """Editing a CRLF file keeps `\\r\\n` byte-for-byte instead of writing `\\r\\r\\n` on Windows."""
        (fs_root / 'crlf.txt').write_bytes(b'a\r\nb\r\n')
        await toolset.edit_file('crlf.txt', 'a', 'A', workspace=ws)
        assert (fs_root / 'crlf.txt').read_bytes() == b'A\r\nb\r\n'

    async def test_crlf_write_hash_matches_read_and_file_info(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        content = 'alpha\r\nbeta\r\n'
        write_hash = _reported_hash(await toolset.write_file('crlf-new.txt', content, workspace=ws))
        assert write_hash == _content_hash(content)

        read_result = await toolset.read_file('crlf-new.txt', workspace=ws)
        assert _reported_hash(read_result) == write_hash

        info = await toolset.file_info('crlf-new.txt', workspace=ws)
        assert f'hash: {write_hash}' in info

    async def test_crlf_write_expected_hash_handshake(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        """`write_file` accepts the hash `read_file` reports for a CRLF file."""
        await toolset.write_file('crlf-cc.txt', 'alpha\r\nbeta\r\n', workspace=ws)
        read_hash = _reported_hash(await toolset.read_file('crlf-cc.txt', workspace=ws))

        result = await toolset.write_file('crlf-cc.txt', 'gamma\r\ndelta\r\n', expected_hash=read_hash, workspace=ws)
        assert _reported_hash(result) == _content_hash('gamma\r\ndelta\r\n')
        assert (fs_root / 'crlf-cc.txt').read_bytes() == b'gamma\r\ndelta\r\n'

    async def test_lf_write_still_writes_plain_newlines(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        """LF content is untouched by the no-translation write path."""
        content = 'alpha\nbeta\n'
        write_hash = _reported_hash(await toolset.write_file('lf-new.txt', content, workspace=ws))
        assert write_hash == _content_hash(content)
        assert (fs_root / 'lf-new.txt').read_bytes() == b'alpha\nbeta\n'
        assert _reported_hash(await toolset.read_file('lf-new.txt', workspace=ws)) == write_hash


class TestListDirectory:
    async def test_list_root(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.list_directory('.', workspace=ws)
        assert 'hello.txt' in result
        assert 'subdir/' in result

    async def test_list_subdir(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.list_directory('subdir', workspace=ws)
        assert 'nested.py' in result

    async def test_listed_links_leading_outside_the_root_cannot_be_opened(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend, outside: Path
    ) -> None:
        """Listings name entries without resolving them; every read or write re-checks the real path."""
        (fs_root / 'escape_link.txt').symlink_to(outside / 'secret.txt')
        (fs_root / 'escape_dir').symlink_to(outside)
        (fs_root / 'inner_link.txt').symlink_to(fs_root / 'hello.txt')
        result = await toolset.list_directory('.', workspace=ws)
        assert 'inner_link.txt  (' in result
        assert 'escape_link.txt' in result
        refusal = r'leads outside root_dir .* through a symlink'
        with pytest.raises(ModelRetry, match=refusal):
            await toolset.read_file('escape_link.txt', workspace=ws)
        with pytest.raises(ModelRetry, match=refusal):
            await toolset.write_file('escape_link.txt', 'x', workspace=ws)
        with pytest.raises(ModelRetry, match=refusal):
            await toolset.list_directory('escape_dir', workspace=ws)
        assert (outside / 'secret.txt').read_text() == 'escaped!\n'

    async def test_list_skips_dangling_symlink(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        """A link to a missing in-root target has no size to report, so it isn't listed."""
        (fs_root / 'dangling.txt').symlink_to(fs_root / 'gone.txt')
        result = await toolset.list_directory('.', workspace=ws)
        assert 'dangling.txt' not in result
        assert 'hello.txt' in result

    async def test_list_not_a_dir(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry):
            await toolset.list_directory('hello.txt', workspace=ws)

    async def test_list_skips_hidden(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        # Dotfiles/dot-directories are hidden, matching find_files/search_files.
        result = await toolset.list_directory('.', workspace=ws)
        assert 'hello.txt' in result
        assert '.hidden' not in result
        assert '.git' not in result

    async def test_list_shows_sizes(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.list_directory('.', workspace=ws)
        assert 'bytes' in result

    async def test_list_shows_dir_indicator(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.list_directory('.', workspace=ws)
        assert 'subdir/' in result

    async def test_list_empty_directory(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'empty').mkdir()
        result = await toolset.list_directory('empty', workspace=ws)
        assert result == '(empty directory)'

    async def test_list_shows_protected_entries(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'protected.txt').write_text('protected marker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=['*'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )

        assert 'protected.txt' in await ts.list_directory('.', workspace=ws)

    async def test_list_root_allowed_patterns_filters_entries(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        # A file-shaped allowed pattern must not make the root unlistable: '.'
        # is always listed, and entries are filtered against the pattern.
        (fs_root / 'keep.py').write_text('ok\n')
        (fs_root / 'skip.md').write_text('ok\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.list_directory('.', workspace=ws)
        assert 'keep.py' in result
        assert 'skip.md' not in result

    async def test_list_hides_denied_entries(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'visible.txt').write_text('ok\n')
        (fs_root / 'creds.secret').write_text('hunter2\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.list_directory('.', workspace=ws)
        assert 'visible.txt' in result
        assert 'creds.secret' not in result

    async def test_list_truncation(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        for i in range(20):
            (fs_root / f'file{i}.dat').write_text(f'{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=5,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.list_directory('.', workspace=ws)
        lines = result.splitlines()
        assert len(lines) == 7
        assert lines[-2] == '[... truncated at 5 entries]'
        assert lines[-1].startswith('[3 hidden entries omitted')

    async def test_list_at_cap_is_not_marked_truncated(self, tmp_path: Path, ws: LocalWorkspaceBackend) -> None:
        for i in range(3):
            (tmp_path / f'cap{i}.dat').write_text('x\n')
        ts = FileSystemToolset(
            root_dir=tmp_path,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=3,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.list_directory('.', workspace=ws)
        assert [line.split(' ')[0] for line in result.splitlines()] == ['cap0.dat', 'cap1.dat', 'cap2.dat']


class TestSearchFiles:
    async def test_denied_directory_protects_descendants(self, fs_root: Path) -> None:
        (fs_root / 'private').mkdir()
        (fs_root / 'private' / 'secret.txt').write_text('needle\n')
        toolset = FileSystem[None](root_dir=fs_root, denied_patterns=['private']).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = LocalWorkspaceBackend(fs_root)
        with pytest.raises(ModelRetry, match='denied'):
            await toolset.read_file('private/secret.txt', workspace=workspace)
        assert 'private' not in await toolset.find_files('**/*.txt', workspace=workspace)
        assert 'private' not in await toolset.search_files('needle', workspace=workspace)

    async def test_explicit_hidden_directory_can_be_searched(self, fs_root: Path) -> None:
        (fs_root / '.github' / 'workflows').mkdir(parents=True)
        (fs_root / '.github' / 'workflows' / 'ci.txt').write_text('needle\n')
        toolset = FileSystem[None](root_dir=fs_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = LocalWorkspaceBackend(fs_root)
        assert await toolset.find_files('**/*.txt', path='.github', workspace=workspace) == '.github/workflows/ci.txt'
        assert await toolset.search_files('needle', path='.github', workspace=workspace) == (
            '.github/workflows/ci.txt:1:needle'
        )
        assert await toolset.list_directory('.github', workspace=workspace) == '.github/workflows/'

    async def test_explicit_hidden_directory_in_pattern_is_walked(self, fs_root: Path) -> None:
        (fs_root / '.github' / 'workflows').mkdir(parents=True)
        (fs_root / '.github' / 'workflows' / 'ci.yml').write_text('needle\n')
        toolset = FileSystem[None](root_dir=fs_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = FilesystemOnlyWorkspace(fs_root)
        assert await toolset.find_files('.github/**/*.yml', workspace=workspace) == '.github/workflows/ci.yml'
        assert await toolset.search_files('needle', include_glob='.github/**/*.yml', workspace=workspace) == (
            '.github/workflows/ci.yml:1:needle'
        )

    async def test_many_globstars_match_without_blowup(self, tmp_path: Path) -> None:
        # Each `**` tries every split of the remaining path; unmemoized, this is C(40, 20) splits per entry.
        deep = tmp_path.joinpath(*['d'] * 20)
        deep.mkdir(parents=True)
        (deep / 'file.txt').write_text('x')
        toolset = FileSystem[None](root_dir=tmp_path).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = LocalWorkspaceBackend(tmp_path)
        globstars = '/'.join(['**'] * 20)
        assert await toolset.find_files(f'{globstars}/*.md', workspace=workspace) == 'No matches found.'
        assert await toolset.find_files(f'{globstars}/*.txt', workspace=workspace) == '/'.join(
            ['d'] * 20 + ['file.txt']
        )

    async def test_symlinked_directory_loop_does_not_duplicate_matches(self, fs_root: Path) -> None:
        (fs_root / 'loop').symlink_to('.', target_is_directory=True)
        toolset = FileSystem[None](root_dir=fs_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        found = await toolset.find_files('**/*.txt', workspace=LocalWorkspaceBackend(fs_root))
        assert found.splitlines().count('hello.txt') == 1
        assert not any('loop/' in line for line in found.splitlines())

    async def test_skips_large_and_unreadable_files(self, fs_root: Path) -> None:
        (fs_root / 'large.txt').write_text('needle' * 2_000_000)
        (fs_root / 'unreadable.txt').write_text('needle')
        (fs_root / 'good.txt').write_text('needle')
        toolset = FileSystem[None](root_dir=fs_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        backend = FailingWorkspace(fs_root, {'read_bytes': WorkspaceError('unreadable')}, where='unreadable.txt')
        # A read-only workspace runs no commands, so the search walks, with the file sizes its listings report.
        workspace = ReadOnlyWorkspace(Workspace(backend))
        assert (await toolset.search_files('needle', workspace=workspace)).splitlines() == [
            'good.txt:1:needle',
            '[2 files skipped (too large or unreadable): large.txt, unreadable.txt]',
            '[3 hidden entries omitted; name a hidden path explicitly to include it]',
        ]
        assert await toolset.search_files('needle', path='large.txt', workspace=workspace) == (
            '[1 files skipped (too large or unreadable): large.txt]'
        )

    async def test_search_basic(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.search_files('Hello', workspace=ws)
        assert 'hello.txt:1:Hello, world!' in result

    async def test_search_regex(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.search_files(r'line\d', workspace=ws)
        assert 'multi.txt' in result

    async def test_search_no_matches(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.search_files('ZZZZNOTHERE', workspace=ws)
        assert result == 'No matches found.'

    async def test_search_skips_hidden(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.search_files('secret', workspace=ws)
        assert '.hidden' not in result

    async def test_search_skips_binary(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.search_files('.', workspace=ws)
        assert 'binary.bin' not in result

    async def test_search_invalid_regex(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='Invalid regex'):
            await toolset.search_files('[invalid', workspace=ws)

    async def test_search_include_glob(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.search_files('print', include_glob='*.py', workspace=ws)
        assert 'nested.py' in result

    async def test_search_include_glob_excludes(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        result = await toolset.search_files('Hello', include_glob='*.py', workspace=ws)
        assert result == 'No matches found.'

    async def test_search_in_specific_file(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.search_files('line', path='multi.txt', workspace=ws)
        assert 'multi.txt' in result

    async def test_search_truncation(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        # Create many matching files
        for i in range(20):
            (fs_root / f'match{i}.txt').write_text('findme\n' * 100)
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=50,
            max_find_results=1000,
        )
        result = await ts.search_files('findme', workspace=FilesystemOnlyWorkspace(fs_root))
        assert 'truncated at 50 matches' in result

    async def test_search_includes_protected_files(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'protected.txt').write_text('protected marker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=['*'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )

        assert 'protected.txt:1:protected marker' in await ts.search_files('protected marker', workspace=ws)

    async def test_search_skips_denied_files(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'visible.txt').write_text('lookhere\n')
        (fs_root / 'creds.secret').write_text('lookhere\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.search_files('lookhere', workspace=ws)
        assert 'visible.txt' in result
        assert 'creds.secret' not in result

    async def test_search_only_matches_allowed_files(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        # The search root ('.') isn't required to match allowed_patterns; only
        # the matched files are filtered against it per-entry.
        (fs_root / 'keep.py').write_text('findme\n')
        (fs_root / 'skip.md').write_text('findme\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.search_files('findme', workspace=ws)
        assert 'keep.py' in result
        assert 'skip.md' not in result

    async def test_search_truncates_within_a_single_file(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'many.txt').write_text('findme\n' * 100)
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=5,
            max_find_results=1000,
        )
        result = await ts.search_files('findme', workspace=FilesystemOnlyWorkspace(fs_root))
        lines = result.splitlines()
        assert len(lines) >= 6
        assert lines[5] == '[... truncated at 5 matches]'

    async def test_search_at_cap_is_not_marked_truncated(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'exact.txt').write_text('capmarker\ncapmarker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=2,
            max_find_results=1000,
        )
        result = await ts.search_files('capmarker', workspace=ws)
        assert result.splitlines() == ['exact.txt:1:capmarker', 'exact.txt:2:capmarker']

    async def test_search_at_cap_across_files_is_not_marked_truncated(
        self, fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'one.txt').write_text('capmarker\n')
        (fs_root / 'two.txt').write_text('capmarker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=2,
            max_find_results=1000,
        )
        result = await ts.search_files('capmarker', workspace=ws)
        assert result.splitlines() == ['one.txt:1:capmarker', 'two.txt:1:capmarker']

    async def test_search_with_zero_cap_returns_no_matches(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        # FileSystemToolset is public, so it can be built without the capability's
        # positive-integer validation.
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=0,
            max_find_results=1000,
        )
        result = await ts.search_files('Hello', workspace=FilesystemOnlyWorkspace(fs_root))
        assert result.splitlines()[0] == '[... truncated at 0 matches]'

    async def test_search_skips_links_leading_outside_the_root(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend, outside: Path
    ) -> None:
        (fs_root / 'escape_link.txt').symlink_to(outside / 'secret.txt')
        (fs_root / 'escape_dir').symlink_to(outside)
        assert await toolset.search_files('escaped', workspace=ws) == 'No matches found.'
        # A read-only workspace runs no commands, so the search walks the files itself.
        assert 'escape' not in await toolset.search_files('escaped', workspace=ReadOnlyWorkspace(Workspace(ws)))


class TestFindFiles:
    async def test_find_glob(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.find_files('*.txt', workspace=ws)
        assert 'hello.txt' in result
        assert 'multi.txt' in result

    async def test_find_recursive(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.find_files('**/*.py', workspace=ws)
        assert 'nested.py' in result

    async def test_find_no_matches(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.find_files('*.xyz', workspace=ws)
        assert result == 'No matches found.'

    async def test_find_skips_hidden(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.find_files('*', workspace=ws)
        assert '.hidden' not in result
        assert '.git' not in result

    async def test_find_does_not_descend_a_directory_leading_outside_the_root(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend, outside: Path
    ) -> None:
        (fs_root / 'escape_dir').symlink_to(outside)
        (fs_root / 'inner_dir').symlink_to(fs_root / 'subdir')
        result = await toolset.find_files('**', workspace=ws)
        assert 'secret.txt' not in result
        assert 'subdir/nested.py' in result
        assert 'inner_dir/nested.py' not in result

    async def test_find_skips_dangling_symlink(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'dangling.txt').symlink_to(fs_root / 'gone.txt')
        result = await toolset.find_files('*.txt', workspace=ws)
        assert 'dangling.txt' not in result
        assert 'hello.txt' in result

    async def test_find_absolute_pattern_rejected(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match='must be relative to the search path'):
            await toolset.find_files('/etc/*', workspace=ws)

    async def test_find_at_cap_is_not_marked_truncated(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        for i in range(3):
            (fs_root / f'cap{i}.dat').write_text('x\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=3,
        )
        result = await ts.find_files('*.dat', workspace=ws)
        assert result.splitlines() == ['cap0.dat', 'cap1.dat', 'cap2.dat']

    async def test_find_not_a_dir(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry):
            await toolset.find_files('*.txt', path='hello.txt', workspace=ws)

    async def test_find_in_subdir(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.find_files('*.py', path='subdir', workspace=ws)
        assert 'nested.py' in result

    async def test_find_directories(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.find_files('sub*', workspace=ws)
        assert 'subdir/' in result

    async def test_find_truncation(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        for i in range(20):
            (fs_root / f'file{i}.dat').write_text(f'{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=5,
        )
        result = await ts.find_files('*.dat', workspace=ws)
        assert 'truncated at 5 matches' in result

    async def test_find_includes_protected_files(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'protected.txt').write_text('protected marker\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=['*'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )

        assert 'protected.txt' in await ts.find_files('*.txt', workspace=ws)

    async def test_find_hides_denied_entries(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'visible.txt').write_text('ok\n')
        (fs_root / 'creds.secret').write_text('hunter2\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.find_files('*', workspace=ws)
        assert 'visible.txt' in result
        assert 'creds.secret' not in result

    async def test_find_only_shows_allowed_entries(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        # The find root ('.') isn't required to match allowed_patterns; only
        # the matched entries are filtered against it per-entry.
        (fs_root / 'keep.py').write_text('ok\n')
        (fs_root / 'skip.md').write_text('ok\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=['*.py'],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        result = await ts.find_files('*', workspace=ws)
        assert 'keep.py' in result
        assert 'skip.md' not in result


class TestResolveSymlinkLoop:
    @pytest.mark.parametrize('op', ['read_file', 'list_directory', 'search_files', 'find_files', 'file_info'])
    async def test_real_symlink_loop_is_reported(
        self, toolset: FileSystemToolset[None], fs_root: Path, op: str, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'loop').symlink_to(fs_root / 'loop')
        calls = {
            'read_file': lambda: toolset.read_file('loop', workspace=ws),
            'list_directory': lambda: toolset.list_directory('loop', workspace=ws),
            'search_files': lambda: toolset.search_files('text', path='loop', workspace=ws),
            'find_files': lambda: toolset.find_files('*', path='loop', workspace=ws),
            'file_info': lambda: toolset.file_info('loop', workspace=ws),
        }

        with pytest.raises(ModelRetry, match='resolves through a symlink loop'):
            await calls[op]()


class TestReadSideOSErrors:
    # The workspace reports an over-long name from whichever call reaches it
    # first, so the message differs by operation. What must hold
    # everywhere is that the run survives.
    @pytest.mark.parametrize('op', ['read_file', 'edit_file', 'list_directory', 'file_info'])
    async def test_long_name_is_recoverable(
        self, toolset: FileSystemToolset[None], op: str, ws: LocalWorkspaceBackend
    ) -> None:
        long = 'x' * 300
        calls = {
            'read_file': lambda: toolset.read_file(long, workspace=ws),
            'edit_file': lambda: toolset.edit_file(long, 'a', 'b', workspace=ws),
            'list_directory': lambda: toolset.list_directory(long, workspace=ws),
            'file_info': lambda: toolset.file_info(long, workspace=ws),
        }
        with pytest.raises(ModelRetry):
            await calls[op]()

    async def test_walker_long_path_is_recoverable(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match='name is too long'):
            await toolset.search_files('hello', path='x' * 300, workspace=ws)


class TestWriteFileOSErrors:
    async def test_write_name_too_long(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='name is too long'):
            await toolset.write_file('x' * 300, 'content', workspace=ws)

    async def test_write_through_symlink_loop(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'loop').symlink_to(fs_root / 'loop')
        # The workspace's `stat` reports ELOOP, which the errno table names.
        with pytest.raises(ModelRetry, match='symlink loop'):
            await toolset.write_file('loop', 'content', workspace=ws)

    async def test_write_non_recoverable_errno_propagates(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        workspace = FailingWorkspace(fs_root, {'write_bytes': OSError(errno.ENOSPC, 'No space left on device')})
        with pytest.raises(OSError, match='No space left on device'):
            await toolset.write_file('new.txt', 'content', workspace=workspace)

    async def test_write_windows_invalid_name_is_recoverable(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        class WindowsInvalidNameError(OSError):
            winerror = 123

        error = WindowsInvalidNameError(errno.EINVAL, 'The filename, directory name, or volume label is incorrect')
        with pytest.raises(ModelRetry, match='path name is invalid'):
            await toolset.write_file('bad<name', 'content', workspace=FailingWorkspace(fs_root, {'write_bytes': error}))

    async def test_write_einval_without_windows_invalid_name_propagates(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        workspace = FailingWorkspace(fs_root, {'write_bytes': OSError(errno.EINVAL, 'Invalid argument')})
        with pytest.raises(OSError, match='Invalid argument'):
            await toolset.write_file('new.txt', 'content', workspace=workspace)

    async def test_write_illegal_byte_sequence_is_recoverable(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        workspace = FailingWorkspace(fs_root, {'write_bytes': OSError(errno.EILSEQ, 'Illegal byte sequence')})
        with pytest.raises(ModelRetry, match='filesystem cannot represent'):
            await toolset.write_file('bad-name', 'content', workspace=workspace)


class TestWalkerEntryResolution:
    """Walkers authorize entries by their root-relative spelling, matching `read_file`."""

    async def test_patterns_apply_to_the_link_and_its_target(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        """A link to a denied file is denied for every tool that reads it; a listing still names it."""
        (fs_root / 'creds.secret').write_text('hunter2\n')
        (fs_root / 'alias.txt').symlink_to(fs_root / 'creds.secret')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        with pytest.raises(ModelRetry, match='denied by pattern'):
            await ts.read_file('creds.secret', workspace=ws)
        with pytest.raises(ModelRetry, match='denied by pattern'):
            await ts.read_file('alias.txt', workspace=ws)
        listing = await ts.list_directory('.', workspace=ws)
        assert 'alias.txt' in listing and 'creds.secret' not in listing
        assert await ts.search_files('hunter2', workspace=ws) == 'No matches found.'

    async def test_in_root_alias_to_allowed_target_stays_visible(
        self, fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'alias.txt').symlink_to(fs_root / 'hello.txt')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['*.secret'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'alias.txt' in await ts.list_directory('.', workspace=ws)
        assert 'alias.txt' in await ts.find_files('*.txt', workspace=ws)
        assert 'alias.txt:1:Hello, world!' in await ts.search_files('Hello', workspace=FilesystemOnlyWorkspace(ws))


class TestCreateDirectory:
    async def test_create_basic(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        result = await toolset.create_directory('newdir', workspace=ws)
        assert 'Created directory' in result
        assert (fs_root / 'newdir').is_dir()

    async def test_create_nested(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        await toolset.create_directory('a/b/c', workspace=ws)
        assert (fs_root / 'a' / 'b' / 'c').is_dir()

    async def test_create_existing_ok(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.create_directory('subdir', workspace=ws)
        assert 'Created directory' in result

    async def test_create_protected_blocked(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.create_directory('.git/hooks', workspace=ws)

    async def test_create_name_too_long(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='name is too long'):
            await toolset.create_directory('x' * 300, workspace=ws)

    async def test_create_illegal_byte_sequence_is_recoverable(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        workspace = FailingWorkspace(fs_root, {'make_dir': OSError(errno.EILSEQ, 'Illegal byte sequence')})
        with pytest.raises(ModelRetry, match='filesystem cannot represent'):
            await toolset.create_directory('bad-name', workspace=workspace)

    async def test_create_non_recoverable_errno_propagates(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        workspace = FailingWorkspace(fs_root, {'make_dir': OSError(errno.EROFS, 'Read-only file system')})
        with pytest.raises(OSError, match='Read-only file system'):
            await toolset.create_directory('newdir', workspace=workspace)

    async def test_create_over_existing_file(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match=r"'hello.txt' exists and is not a directory"):
            await toolset.create_directory('hello.txt', workspace=ws)

    async def test_create_under_existing_file(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match=r"'hello.txt/nested' has a parent that is not a directory"):
            await toolset.create_directory('hello.txt/nested', workspace=ws)


class TestFileInfo:
    async def test_info_file(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.file_info('hello.txt', workspace=ws)
        assert 'type: file' in result
        assert 'size:' in result
        assert 'lines:' in result
        assert 'hash:' in result
        assert 'binary: False' in result

    async def test_info_directory(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.file_info('subdir', workspace=ws)
        assert 'type: directory' in result

    async def test_info_binary(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.file_info('binary.bin', workspace=ws)
        assert 'binary: True' in result
        assert 'lines:' not in result

    async def test_info_not_found(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        assert await toolset.file_info('nonexistent', workspace=ws) == 'Path not found: nonexistent'

    async def test_info_symlink(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        # The link target is stored as an absolute path; the tool must report
        # it relative to the root, never as the absolute host path.
        link = fs_root / 'link.txt'
        link.symlink_to(fs_root / 'hello.txt')
        result = await toolset.file_info('link.txt', workspace=ws)
        assert 'type: file' in result
        assert 'symlink_target: hello.txt' in result
        _assert_no_host_root(result, fs_root)

    async def test_info_symlink_through_a_wrapped_workspace(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        # A facade over a facade still reaches the command-capable backend underneath.
        (fs_root / 'link.txt').symlink_to(fs_root / 'hello.txt')
        result = await toolset.file_info('link.txt', workspace=Workspace(Workspace(ws)))
        assert 'symlink_target: hello.txt' in result

    async def test_info_symlink_unreported_on_a_read_only_workspace(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        # A read-only workspace refuses `run`, so there is no `readlink` to ask; the rest is reported.
        (fs_root / 'link.txt').symlink_to(fs_root / 'hello.txt')
        result = await toolset.file_info('link.txt', workspace=ReadOnlyWorkspace(Workspace(ws)))
        assert 'type: file' in result
        assert 'symlink_target' not in result


class TestMutationKillers:
    async def test_format_lines_offset_equals_total(self) -> None:
        text = 'a\nb\n'  # 2 lines
        with pytest.raises(ValueError, match='Offset 2 exceeds file length'):
            _format_lines(text.splitlines(keepends=True), 2, 10)

    async def test_format_lines_exact_fit_no_continuation(self) -> None:
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text.splitlines(keepends=True), 0, 3)
        assert '... (' not in result
        assert 'more lines' not in result

    async def test_format_lines_exact_fit_from_offset(self) -> None:
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text.splitlines(keepends=True), 1, 2)  # lines 2-3, 0 remaining
        assert '... (' not in result
        assert 'more lines' not in result

    async def test_format_lines_one_line_remaining(self) -> None:
        text = 'a\nb\nc\n'  # 3 lines
        result = _format_lines(text.splitlines(keepends=True), 0, 2)
        assert '... (1 more lines. Use offset=2 to continue reading.)' in result

    async def test_format_lines_line_number_starts_at_one(self) -> None:
        text = 'first\nsecond\n'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        assert '     1\tfirst\n' in result
        assert '     0\t' not in result

    async def test_format_lines_offset_line_numbering(self) -> None:
        text = 'a\nb\nc\n'
        result = _format_lines(text.splitlines(keepends=True), 1, 2)
        assert '     2\tb\n' in result
        assert '     3\tc\n' in result

    async def test_is_binary_exactly_at_sample_boundary(self) -> None:
        # Null byte at position 8191 (index 8191, within first 8192 bytes)
        data = b'x' * 8191 + b'\x00'
        assert _is_binary(data) is True
        # Null byte at position 8192 (outside the sample)
        data2 = b'x' * 8192 + b'\x00'
        assert _is_binary(data2) is False

    async def test_content_hash_returns_exactly_12_chars(self) -> None:
        h = _content_hash('test content')
        assert len(h) == 12
        # Verify it's hex characters
        assert all(c in '0123456789abcdef' for c in h)

    async def test_write_file_with_hash_on_new_file(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        """When a file doesn't exist, expected_hash should be ignored and the write should succeed."""
        result = await toolset.write_file('brand_new.txt', 'new content\n', expected_hash='any_hash_val', workspace=ws)
        assert 'Wrote' in result
        assert (fs_root / 'brand_new.txt').read_text() == 'new content\n'

    async def test_edit_file_single_match_succeeds(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'unique.txt').write_text('unique text here\n')
        result = await toolset.edit_file('unique.txt', 'unique text', 'replaced text', workspace=ws)
        assert 'Edited' in result
        assert (fs_root / 'unique.txt').read_text() == 'replaced text here\n'

    async def test_edit_file_zero_matches_raises(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match='old_text not found'):
            await toolset.edit_file('hello.txt', 'DEFINITELY NOT IN FILE', 'x', workspace=ws)

    async def test_search_truncation_stops_after_limit(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        # Create many files with 1 match each so truncation is per-file
        for i in range(10):
            (fs_root / f'searchable{i}.txt').write_text(f'match_this_{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=5,
            max_find_results=1000,
        )
        result = await ts.search_files('match_this', workspace=FilesystemOnlyWorkspace(fs_root))
        lines = result.strip().split('\n')
        # Truncation check is after each file, so 5 matches + truncation msg
        # Ensure we don't get all 10 matches
        match_lines = [ln for ln in lines if ln.startswith('searchable')]
        assert len(match_lines) <= 5
        assert any('truncated at 5 matches' in line for line in lines)

    async def test_find_truncation_stops_after_limit(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        for i in range(10):
            (fs_root / f'findme{i:02d}.dat').write_text(f'{i}\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=3,
        )
        result = await ts.find_files('*.dat', workspace=ws)
        lines = result.strip().split('\n')
        # Should have exactly 4 lines: 3 matches + 1 truncation message
        assert len(lines) == 4
        assert 'truncated at 3 matches' in lines[-1]

    async def test_read_file_default_limit_used(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        # Create file with more lines than we'd see with limit=0
        (fs_root / 'big.txt').write_text('\n'.join(f'line{i}' for i in range(100)) + '\n')
        result = await toolset.read_file('big.txt', workspace=ws)
        # All 100 lines should be present since max_read_lines is 2000
        assert 'line99' in result

    async def test_list_directory_with_files_not_empty(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        result = await toolset.list_directory('subdir', workspace=ws)
        assert result != '(empty directory)'
        assert 'nested.py' in result

    async def test_search_in_file_returns_only_that_file(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        # Both files contain 'Hello' / 'hello' but searching a specific file should only return from that file
        (fs_root / 'other.txt').write_text('Hello from other\n')
        result = await toolset.search_files('Hello', path='hello.txt', workspace=ws)
        assert 'hello.txt' in result
        assert 'other.txt' not in result

    async def test_file_info_non_binary_shows_lines_and_hash(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        result = await toolset.file_info('hello.txt', workspace=ws)
        assert 'lines: 1' in result
        assert 'hash:' in result
        assert 'binary: False' in result

    async def test_file_info_binary_no_lines_no_hash(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        result = await toolset.file_info('binary.bin', workspace=ws)
        assert 'binary: True' in result
        assert 'lines:' not in result
        assert 'hash:' not in result

    async def test_safe_resolve_passes_write_flag(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        # Protected patterns block writes but allow reads
        (fs_root / '.env.local').write_text('SECRET=x\n')
        # Read should work (write=False internally)
        result = await toolset.read_file('.env.local', workspace=ws)
        assert 'SECRET=x' in result
        # Write should be blocked (write=True internally)
        with pytest.raises(ModelRetry, match='protected'):
            await toolset.write_file('.env.local', 'HACKED\n', workspace=ws)

    async def test_format_lines_join_separator(self) -> None:
        """Verify the result doesn't contain garbage between lines."""
        text = 'a\nb\nc\n'
        result = _format_lines(text.splitlines(keepends=True), 0, 3)
        # Lines should be directly adjacent (no separator between them)
        assert '     1\ta\n     2\tb\n     3\tc\n' in result

    async def test_format_lines_no_trailing_newline_preserves_content(self) -> None:
        text = 'no newline'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        # The content must still be present
        assert 'no newline' in result
        assert result.endswith('\n')

    async def test_read_file_hash_is_real_hash(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        result = await toolset.read_file('hello.txt', workspace=ws)
        # The actual hash should be a hex string, not 'None'
        assert 'hash:None' not in result
        # Verify the hash matches what we'd compute
        expected_hash = _content_hash('Hello, world!\n')
        assert f'hash:{expected_hash}' in result

    async def test_read_file_non_ascii_content(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        """With invalid UTF-8 bytes, the tool should not crash -- it should use replacement chars."""
        # Write raw bytes that are invalid UTF-8
        (fs_root / 'broken_utf8.txt').write_bytes(b'hello \xff\xfe world\n')
        result = await toolset.read_file('broken_utf8.txt', workspace=ws)
        # Should not crash, content should contain replacement characters
        assert 'hello' in result
        assert 'world' in result

    async def test_read_file_default_offset_starts_at_first_line(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        """The first line must be included when no offset is specified."""
        result = await toolset.read_file('multi.txt', workspace=ws)
        assert '     1\tline1' in result

    async def test_toolset_tool_names(self, toolset: FileSystemToolset[None]) -> None:
        """Verify tools are registered with correct names."""
        tool_names = set(toolset.tools.keys())
        assert 'read_file' in tool_names
        assert 'write_file' in tool_names
        assert 'edit_file' in tool_names
        assert 'list_directory' in tool_names
        assert 'search_files' in tool_names
        assert 'find_files' in tool_names
        assert 'create_directory' in tool_names
        assert 'file_info' in tool_names

    async def test_write_file_output_format(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        result = await toolset.write_file('fmt.txt', 'ab\ncd\n', workspace=ws)
        # Verify specific format: chars, lines, path, hash
        assert 'Wrote 6 chars (2 lines) to fmt.txt.' in result
        assert 'hash:' in result
        # Verify hash is a real hex hash not None
        assert 'hash:None' not in result

    async def test_edit_file_output_format(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        result = await toolset.edit_file('hello.txt', 'Hello, world!', 'Hi', workspace=ws)
        assert result.startswith('Edited hello.txt.')
        assert 'hash:' in result
        assert 'hash:None' not in result

    def test_format_lines_no_double_trailing_newline(self) -> None:
        """Text that already ends with newline must NOT get a second one appended."""
        text = 'hello\n'
        result = _format_lines(text.splitlines(keepends=True), 0, 10)
        # Exact match: no trailing double newline
        assert result == '     1\thello\n'

    async def test_list_directory_exact_size(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.list_directory('.', workspace=ws)
        # hello.txt has 'Hello, world!\n' = 14 bytes
        assert '14 bytes' in result

    async def test_list_directory_error_message(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match='Not a directory'):
            await toolset.list_directory('hello.txt', workspace=ws)

    async def test_find_files_error_message(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        with pytest.raises(ModelRetry, match='Not a directory'):
            await toolset.find_files('*.txt', path='hello.txt', workspace=ws)

    @pytest.mark.parametrize('pattern', ['.', '..', 'a/../b', '', '/'])
    async def test_find_files_invalid_pattern(
        self, toolset: FileSystemToolset[None], pattern: str, ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match=r'not a valid glob pattern|must be relative'):
            await toolset.find_files(pattern, workspace=ws)

    async def test_file_info_exact_size(self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend) -> None:
        result = await toolset.file_info('hello.txt', workspace=ws)
        assert '14 bytes' in result

    async def test_search_with_invalid_utf8_file(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        """A file with invalid UTF-8 (but no null bytes = not binary) should be searchable."""
        # Write a file with invalid UTF-8 but no null bytes (not detected as binary)
        (fs_root / 'bad_encoding.txt').write_bytes(b'marker_text \xff\xfe end\n')
        result = await toolset.search_files('marker_text', workspace=ws)
        # Should find the file even with broken encoding
        assert 'bad_encoding.txt' in result

    async def test_search_binary_skip_does_not_stop_iteration(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        """A binary file must be skipped, but subsequent text files must still be searched."""
        # binary.bin exists in the fixture and comes before 'hello.txt' alphabetically
        result = await toolset.search_files('Hello', workspace=ws)
        # hello.txt must still be found (binary.bin didn't break the loop)
        assert 'hello.txt' in result

    async def test_find_hidden_skip_does_not_stop_iteration(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        """Hidden files must be skipped, but subsequent visible files must still appear."""
        # .hidden comes before hello.txt alphabetically -- skipping must not break the loop
        result = await toolset.find_files('*', workspace=ws)
        assert 'hello.txt' in result
        assert 'multi.txt' in result


class TestFileSystemCapability:
    def test_default_construction(self) -> None:
        fs = FileSystem()
        assert fs.root_dir is None
        assert fs.max_read_lines == 2000

    def test_custom_construction(self, tmp_path: Path) -> None:
        fs = FileSystem(
            root_dir=tmp_path,
            allowed_patterns=['*.py'],
            denied_patterns=['test_*'],
            max_read_lines=500,
        )
        assert fs.max_read_lines == 500

    def test_get_toolset_returns_toolset(self, tmp_path: Path) -> None:
        fs = FileSystem(root_dir=tmp_path)
        toolset = fs.get_toolset()
        assert isinstance(toolset, FileSystemToolset)

    async def test_read_only_exposes_exactly_read_only_tools(self, tmp_path: Path) -> None:
        filesystem = FileSystem[None](root_dir=tmp_path, read_only=True)
        context = RunContext(
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            run_id='test',
            workspace=Workspace(LocalWorkspaceBackend(tmp_path)),
        )

        tools = await filesystem.get_toolset().get_tools(context)

        assert set(tools) == READ_ONLY_TOOL_NAMES - set(RIPGREP_TOOL_NAMES)

        everything = FileSystem[None](root_dir=tmp_path, read_only=True, tools=FILE_SYSTEM_TOOL_NAMES)
        assert set(await everything.get_toolset().get_tools(context)) == READ_ONLY_TOOL_NAMES

    @pytest.mark.parametrize('anyio_backend', ['asyncio'])  # Agent.run needs asyncio
    async def test_read_only_workspace_hides_write_tools(self, tmp_path: Path, anyio_backend: object) -> None:
        model = TestModel(call_tools=[])
        capabilities: list[AbstractCapability[None]] = [
            FileSystem[None](root_dir=tmp_path, tools=FILE_SYSTEM_TOOL_NAMES),
            LocalWorkspace[None](tmp_path, read_only=True),
        ]
        await Agent(model, deps_type=type(None), capabilities=capabilities).run('Inspect tools')
        assert model.last_model_request_parameters is not None
        assert {tool.name for tool in model.last_model_request_parameters.function_tools} == READ_ONLY_TOOL_NAMES

    @pytest.mark.parametrize('read_only', [True, False], ids=['read-only', 'filesystem-only'])
    @pytest.mark.parametrize('anyio_backend', ['asyncio'])  # Agent.run needs asyncio
    async def test_ripgrep_tools_walk_files_without_commands(
        self, tmp_path: Path, anyio_backend: object, read_only: bool
    ) -> None:
        """Where `rg` cannot run, `grep` and `list_files` read the files, not the read-only refusal."""
        (tmp_path / 'notes.txt').write_text('needle\n')
        workspace: WorkspaceBackend = (
            ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))
            if read_only
            else FilesystemOnlyWorkspace(tmp_path)
        )
        capability = FileSystem[None](root_dir=tmp_path, tools=FILE_SYSTEM_TOOL_NAMES)
        model = TestModel(call_tools=[])
        await Agent(model, deps_type=type(None), capabilities=[capability]).run('Inspect', workspace=workspace)
        assert model.last_model_request_parameters is not None
        names = {tool.name for tool in model.last_model_request_parameters.function_tools}
        assert {'search_files', 'find_files', *RIPGREP_TOOL_NAMES} <= names
        assert await call_tool(
            [capability], 'grep', {'pattern': 'NEEDLE', 'ignore_case': True}, workspace=workspace
        ) == ('notes.txt:1:needle')
        assert await call_tool([capability], 'list_files', {'glob': '*.txt'}, workspace=workspace) == 'notes.txt'
        assert '`file_type` needs ripgrep' in await call_tool(
            [capability], 'grep', {'pattern': 'needle', 'file_type': 'py'}, workspace=workspace
        )
        assert 'context lines are unavailable' in await call_tool(
            [capability], 'grep', {'pattern': 'needle', 'context': 2}, workspace=workspace
        )

    @pytest.mark.parametrize('anyio_backend', ['asyncio'])  # Agent.run needs asyncio
    async def test_read_only_refusal_is_a_failed_tool_result(self, tmp_path: Path, anyio_backend: object) -> None:
        """A mutation a read-only workspace refuses fails the call; it is not a permission retry."""
        capability = FileSystem[None](root_dir=tmp_path)
        # The environment refuses writes without advertising `read_only`, so the tools stay offered.
        calls: list[tuple[str, dict[str, object]]] = [
            ('write_file', {'path': 'new.txt', 'content': 'x'}),
            ('create_directory', {'path': 'made'}),
        ]
        for name, arguments in calls:
            result = await call_tool([capability], name, arguments, workspace=ReadOnlyMount(str(tmp_path)))
            assert result == READ_ONLY_FAILURE

        toolset = capability.get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        with pytest.raises(ToolFailed) as exc_info:
            await toolset.write_file(
                'new.txt', 'x', workspace=ReadOnlyWorkspace(Workspace(LocalWorkspaceBackend(tmp_path)))
            )
        assert exc_info.value.message == READ_ONLY_FAILURE
        assert list(tmp_path.iterdir()) == []

    def test_search_files_description_has_string_return_type(self) -> None:
        toolset = FileSystem().get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        description = toolset.tools['search_files'].description

        assert description == (
            '<summary>Search file contents using a regular expression.</summary>\n'
            '<returns>\n'
            '<type>str</type>\n'
            '<description>Matching lines formatted as file:line_number:text, with paths relative to the working directory.</description>\n'
            '</returns>'
        )

    def test_default_read_output_is_bounded(self) -> None:
        assert FileSystem().max_read_chars == 50_000
        assert FileSystem(max_read_chars=None).max_read_chars is None

    def test_deprecated_cwd_is_ignored(self) -> None:
        with pytest.warns(Warning, match='cwd'):
            assert FileSystem(cwd='elsewhere').root_dir is None

    def test_read_only_defaults(self) -> None:
        fs = FileSystem()
        assert '**/.git/*' in fs.read_only_patterns
        assert '**/.env' in fs.read_only_patterns

    def test_protected_patterns_is_a_deprecated_alias(self) -> None:
        with pytest.warns(
            HarnessDeprecationWarning, match=r'`FileSystem\(protected_patterns=\.\.\.\)` has been renamed'
        ):
            fs = FileSystem(protected_patterns=['*.lock'])
        assert fs.read_only_patterns == ['*.lock']
        assert replace(fs, read_only_patterns=[]).read_only_patterns == []

    def test_protected_patterns_with_read_only_patterns_rejected(self) -> None:
        with pytest.raises(TypeError, match='`protected_patterns` is its deprecated name'):
            FileSystem(read_only_patterns=['*.lock'], protected_patterns=['*.lock'])

    def test_toolset_protected_patterns_is_a_deprecated_alias(self) -> None:
        with pytest.warns(HarnessDeprecationWarning, match=r'`FileSystemToolset\(protected_patterns=\.\.\.\)`'):
            _toolset_with_patterns(None, ['*.lock'])
        with pytest.raises(TypeError, match='`protected_patterns` is its deprecated name'):
            _toolset_with_patterns([], ['*.lock'])

    def test_non_positive_max_read_lines_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_read_lines must be a positive integer'):
            FileSystem(max_read_lines=0)
        with pytest.raises(ValueError, match='max_read_lines must be a positive integer'):
            FileSystem(max_read_lines=-1)

    def test_non_positive_max_list_results_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_list_results must be a positive integer'):
            FileSystem(max_list_results=0)

    def test_non_positive_max_search_results_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_search_results must be a positive integer'):
            FileSystem(max_search_results=0)

    def test_non_positive_max_find_results_rejected(self) -> None:
        with pytest.raises(ValueError, match='max_find_results must be a positive integer'):
            FileSystem(max_find_results=-1)

    def test_non_integer_max_read_lines_rejected(self) -> None:
        # Runtime validation: dataclass annotations are advisory, so a string
        # slipped in from a config must be rejected, not propagated.
        with pytest.raises(ValueError, match='max_read_lines must be a positive integer'):
            FileSystem(max_read_lines='1000')  # type: ignore[arg-type]

    async def test_agent_integration(self, tmp_path: Path, anyio_backend: object, ws: LocalWorkspaceBackend) -> None:
        if str(anyio_backend) != 'asyncio':  # pragma: no cover -- only asyncio runs here
            pytest.skip('Agent.run requires asyncio event loop')
        (tmp_path / 'test.txt').write_text('hello agent\n')
        model = TestModel(custom_output_text='done', call_tools=[])
        agent: Agent[None, str] = Agent(model, capabilities=[FileSystem(root_dir=tmp_path)])
        result = await agent.run('read test.txt', workspace=ws)
        assert result.output == 'done'


class TestPatternCanonicalization:
    """Sec#3: patterns match the canonical path, and a leading `**/` also
    covers the repository root."""

    async def test_denied_pattern_not_bypassed_by_dot_segment(self, fs_root: Path, ws: LocalWorkspaceBackend) -> None:
        (fs_root / 'config').mkdir()
        (fs_root / 'config' / 'secret.txt').write_text('token\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=['config/secret.txt'],
            read_only_patterns=[],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        # A './' segment must not slip the file past its deny rule.
        with pytest.raises(ModelRetry, match='denied'):
            await ts.read_file('config/./secret.txt', workspace=ws)

    async def test_root_level_secrets_readable_but_protected_from_write(
        self, fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        (fs_root / 'secrets.yaml').write_text('api: PRIVATE KEY material\n')
        ts = FileSystemToolset(
            root_dir=fs_root,
            allowed_patterns=[],
            denied_patterns=[],
            read_only_patterns=['**/secrets*'],
            max_read_lines=2000,
            max_list_results=1000,
            max_search_results=1000,
            max_find_results=1000,
        )
        assert 'secrets.yaml:1:api: PRIVATE KEY material' in await ts.search_files('PRIVATE KEY', workspace=ws)
        with pytest.raises(ModelRetry, match='protected'):
            await ts.write_file('secrets.yaml', 'changed\n', workspace=ws)


def _assert_no_host_root(message: str, root: Path) -> None:
    """Fail if `message` contains the absolute workspace root."""
    assert str(root) not in message
    assert str(root.resolve()) not in message


class TestModelSafeRecoverableErrors:
    """OS-raised filesystem errors must not leak absolute host paths into `ModelRetry`."""

    async def test_write_through_file_names_the_parent(
        self, toolset: FileSystemToolset[None], fs_root: Path, ws: LocalWorkspaceBackend
    ) -> None:
        # The parent path 'hello.txt' exists as a file; the check names the
        # model's path, not the absolute host path the OS would report.
        with pytest.raises(ModelRetry, match=r"'hello.txt/nested' has a parent that is not a directory") as exc_info:
            await toolset.write_file('hello.txt/nested', 'x', workspace=ws)
        _assert_no_host_root(str(exc_info.value), fs_root)

    def test_outside_root_path_is_redacted(self, fs_root: Path) -> None:
        real_root = str(fs_root)
        outside = fs_root.parent / 'private' / 'secret.txt'
        error = FileNotFoundError(errno.ENOENT, 'No such file or directory', str(outside))
        message = _sanitize_recoverable_error(error, real_root)
        assert str(outside) not in message
        assert _OUTSIDE_WORKSPACE in message

    def test_relative_filename_is_preserved(self, fs_root: Path) -> None:
        real_root = str(fs_root)
        error = PermissionError(errno.EACCES, 'Permission denied', 'hello.txt')
        message = _sanitize_recoverable_error(error, real_root)
        assert message == f"[Errno {errno.EACCES}] Permission denied: 'hello.txt'"

    def test_non_path_filename_is_labeled(self, fs_root: Path) -> None:
        real_root = str(fs_root)
        error = OSError(errno.ENOENT, 'No such file or directory')
        error.filename = object()
        message = _sanitize_recoverable_error(error, real_root)
        assert _NOT_A_PATH in message

    def test_dot_segments_are_normalized(self, fs_root: Path) -> None:
        # Containment is textual, so a path spelled with `..` is normalized before it is compared.
        spelled = f'{fs_root}/subdir/../hello.txt'
        error = FileNotFoundError(errno.ENOENT, 'No such file or directory', spelled)
        message = _sanitize_recoverable_error(error, str(fs_root))
        assert str(fs_root) not in message
        assert "'hello.txt'" in message


class TestWorkspaceBackends:
    """Behavior that depends on what the attached workspace can do, or on how it fails."""

    async def test_shell_hides_tools_on_filesystem_only_workspace(self, fs_root: Path) -> None:
        shell: ShellToolset[None] = ShellToolset(
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10,
            max_output_chars=1000,
            persist_cwd=False,
            allow_interactive=False,
        )
        ctx = RunContext[None](
            deps=None,
            model=TestModel(),
            usage=RunUsage(),
            prompt=None,
            messages=[],
            run_step=0,
            workspace=Workspace(FilesystemOnlyWorkspace(fs_root)),
        )
        assert await shell.get_tools(ctx) == {}

    async def test_filesystem_only_backend_serves_the_default_tools(self, fs_root: Path) -> None:
        toolset = FileSystem[None](root_dir=fs_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        (fs_root / 'link.txt').symlink_to(fs_root / 'hello.txt')
        workspace = FilesystemOnlyWorkspace(fs_root)
        # Listings without sizes are completed with `stat`.
        assert 'hello.txt  (14 bytes)' in await toolset.list_directory('.', workspace=workspace)
        assert 'hello.txt' in await toolset.find_files('*.txt', workspace=workspace)
        assert 'Wrote' in await toolset.write_file('new.txt', 'new\n', workspace=workspace)
        # Without commands there is no `readlink`, so the symlink lines are left out.
        info = await toolset.file_info('link.txt', workspace=workspace)
        assert 'type: file' in info and 'symlink_target' not in info
        assert await toolset.create_directory('made', workspace=workspace) == 'Created directory: made'
        assert (fs_root / 'made').is_dir()

    async def test_filesystem_only_backend_lists_a_file_without_a_size(self, fs_root: Path) -> None:
        class NoSizes(FilesystemOnlyWorkspace):
            async def stat(self, path: str) -> FileEntry:
                return replace(await super().stat(path), size=None)

        toolset = FileSystem[None](root_dir=fs_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        assert 'hello.txt  (size unknown)' in await toolset.list_directory('.', workspace=NoSizes(fs_root))

    async def test_filesystem_only_backend_checks_paths_as_text(
        self, fs_root: Path, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        # Without `realpath`, `..` is still refused but a symlink leading outside is not followed.
        outside = tmp_path_factory.mktemp('outside')
        (outside / 'secret.txt').write_text('secret\n')
        (fs_root / 'out').symlink_to(outside)
        toolset = FileSystem[None](root_dir=fs_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        workspace = FilesystemOnlyWorkspace(fs_root)
        with pytest.raises(ModelRetry):
            await toolset.read_file(f'../{outside.name}/secret.txt', workspace=workspace)
        assert 'secret' in await toolset.read_file('out/secret.txt', workspace=workspace)

    async def test_filesystem_only_backend_resolves_a_relative_root(self, fs_root: Path) -> None:
        # The default root is the workspace's working directory, asked of the backend itself.
        toolset = FileSystem[None]().get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        assert 'Hello, world!' in await toolset.read_file('hello.txt', workspace=FilesystemOnlyWorkspace(fs_root))

    async def test_failures_elsewhere_pass_through(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        # The fake fails only on paths ending in `where`; everything else reaches the local disk.
        failing: dict[str, Exception] = {
            'write_bytes': OSError(errno.ENOSPC, 'No space'),
            'make_dir': OSError(errno.EROFS, 'Read-only'),
        }
        workspace = FailingWorkspace(fs_root, failing, where='/elsewhere')
        assert 'Wrote' in await toolset.write_file('passed.txt', 'ok\n', workspace=workspace)
        assert await toolset.create_directory('passed', workspace=workspace) == 'Created directory: passed'
        assert (fs_root / 'passed.txt').read_text() == 'ok\n' and (fs_root / 'passed').is_dir()

    async def test_unlistable_subdirectory_is_skipped(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        workspace = FailingWorkspace(fs_root, {'list_dir': PermissionError(errno.EACCES, 'denied')}, where='/subdir')
        assert 'hello.txt' in await toolset.search_files('Hello', workspace=FilesystemOnlyWorkspace(workspace))
        assert (await toolset.find_files('**/*.py', workspace=workspace)).splitlines()[0] in (
            'No matches found.',
            '[1 hidden entries omitted; name a hidden path explicitly to include it]',
        )

    async def test_unresolvable_subdirectory_is_skipped(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        workspace = FailingWorkspace(fs_root, {'realpath': PermissionError(errno.EACCES, 'denied')}, where='/subdir')
        assert 'hello.txt' in await toolset.find_files('**/*.txt', workspace=workspace)

    async def test_workspace_failure_resolving_a_subdirectory_is_reported(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        # Unlike an unresolvable subdirectory, a workspace failure is not skipped as one directory's problem.
        workspace = FailingWorkspace(fs_root, {'realpath': WorkspaceError('backend refused')}, where='/subdir')
        with pytest.raises(ToolFailed, match='backend refused'):
            await toolset.find_files('**/*.txt', workspace=workspace)

    @pytest.mark.parametrize('operation', ['realpath', 'read_bytes'])
    async def test_file_removed_while_it_is_read_is_not_found(
        self, toolset: FileSystemToolset[None], operation: str, fs_root: Path
    ) -> None:
        # The file goes after the tool named it: before a backend that resolves paths strictly
        # resolves it, or between the `stat` and the read.
        missing = FileNotFoundError(errno.ENOENT, 'No such file or directory')
        workspace = FailingWorkspace(fs_root, {operation: missing}, where='hello.txt')
        assert await toolset.read_file('hello.txt', workspace=workspace) == 'Path not found: hello.txt'
        assert await toolset.file_info('hello.txt', workspace=workspace) == 'Path not found: hello.txt'

    async def test_unlistable_search_root_is_recoverable(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        workspace = FailingWorkspace(
            fs_root, {'list_dir': PermissionError(errno.EACCES, 'Permission denied')}, where='/subdir'
        )
        with pytest.raises(ModelRetry, match='Permission denied'):
            await toolset.search_files('x', path='subdir', workspace=FilesystemOnlyWorkspace(workspace))

    async def test_unreadable_file_is_skipped_by_search(self, toolset: FileSystemToolset[None], fs_root: Path) -> None:
        workspace = FailingWorkspace(
            fs_root, {'read_bytes': PermissionError(errno.EACCES, 'denied')}, where='hello.txt'
        )
        assert (await toolset.search_files('Hello', workspace=FilesystemOnlyWorkspace(workspace))).splitlines()[
            0
        ] == '[1 files skipped (too large or unreadable): hello.txt]'

    async def test_unavailable_workspace_during_a_search_read_is_reported(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        # A gone environment ends the search instead of counting the file as skipped.
        workspace = FailingWorkspace(
            fs_root, {'read_bytes': WorkspaceUnavailableError('sandbox was destroyed')}, where='hello.txt'
        )
        with pytest.raises(WorkspaceUnavailableError, match='sandbox was destroyed'):
            await toolset.search_files('Hello', workspace=FilesystemOnlyWorkspace(workspace))

    @pytest.mark.parametrize('operation', ['list_dir', 'read_bytes'])
    async def test_workspace_failure_during_a_walk_is_reported(
        self, toolset: FileSystemToolset[None], operation: str, fs_root: Path
    ) -> None:
        workspace = FailingWorkspace(
            fs_root,
            {operation: WorkspaceError('backend refused')},
            where='nested.py' if operation == 'read_bytes' else '/subdir',
        )
        if operation == 'list_dir':
            with pytest.raises(ToolFailed, match='backend refused'):
                await toolset.search_files('x', workspace=FilesystemOnlyWorkspace(workspace))
        else:
            assert 'nested.py' in await toolset.search_files('x', workspace=FilesystemOnlyWorkspace(workspace))

    @pytest.mark.parametrize('anyio_backend', ['asyncio'])  # Agent.run needs asyncio
    async def test_read_only_refusal_of_a_read_is_not_a_permission_retry(
        self, fs_root: Path, anyio_backend: object
    ) -> None:
        """`WorkspaceReadOnlyError` is a `PermissionError`; the announcement read must not treat it as one."""
        workspace = FailingWorkspace(fs_root, {'read_bytes': WorkspaceReadOnlyError('refused')}, where='hello.txt')
        result = await call_tool(
            [FileSystem[None](root_dir=fs_root)],
            'write_file',
            {'path': 'hello.txt', 'content': 'x'},
            workspace=workspace,
        )
        assert result == READ_ONLY_FAILURE
        assert (fs_root / 'hello.txt').read_text() == 'Hello, world!\n'

    async def test_directory_that_appears_as_a_file_while_created(
        self, toolset: FileSystemToolset[None], fs_root: Path
    ) -> None:
        workspace = FailingWorkspace(fs_root, {'make_dir': FileExistsError(errno.EEXIST, 'File exists')})
        with pytest.raises(ModelRetry, match="'made' exists and is not a directory"):
            await toolset.create_directory('made', workspace=workspace)

    async def test_write_below_a_file_names_the_parent(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        with pytest.raises(ModelRetry, match=r"'hello.txt/a/b' has a parent that is not a directory"):
            await toolset.write_file('hello.txt/a/b', 'x', workspace=ws)

    async def test_search_in_a_missing_path_finds_nothing(
        self, toolset: FileSystemToolset[None], ws: LocalWorkspaceBackend
    ) -> None:
        assert await toolset.search_files('x', path='missing', workspace=ws) == 'No matches found.'

    @pytest.mark.parametrize('anyio_backend', ['asyncio'])  # Agent.run needs asyncio
    async def test_file_info_tool(self, fs_root: Path, anyio_backend: object, ws: LocalWorkspaceBackend) -> None:
        result = await call_tool([FileSystem[None](root_dir=fs_root)], 'file_info', {'path': 'hello.txt'}, workspace=ws)
        assert 'size: 14 bytes' in result


class TestWalkBounds:
    """A walk deduplicates symlink loops, while directory/entry caps still bound large trees."""

    @pytest.fixture
    def loop_root(self, tmp_path: Path) -> Path:
        (tmp_path / 'a.txt').write_text('needle\n')
        (tmp_path / 'loop1').symlink_to(tmp_path)
        (tmp_path / 'loop2').symlink_to(tmp_path)
        return tmp_path

    @pytest.mark.parametrize('cap', ['_MAX_WALK_DIRECTORIES', '_MAX_WALK_ENTRIES'])
    async def test_symlink_loop_walk_is_cut_short(
        self, loop_root: Path, monkeypatch: pytest.MonkeyPatch, cap: str, ws: LocalWorkspaceBackend
    ) -> None:
        for index in range(21):
            (loop_root / f'dir{index}').mkdir()
        monkeypatch.setattr(f'pydantic_ai_harness.filesystem._toolset.{cap}', 20)
        toolset = FileSystem[None](root_dir=loop_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        found = await toolset.find_files('**/*.txt', workspace=FilesystemOnlyWorkspace(loop_root))
        assert found.splitlines()[0] == 'a.txt'
        assert found.splitlines()[-1].startswith('[... walk cut short after ')
        searched = await toolset.search_files('needle', workspace=FilesystemOnlyWorkspace(loop_root))
        assert searched.splitlines()[0] == 'a.txt:1:needle'
        assert searched.splitlines()[-1].startswith('[... walk cut short after ')

    async def test_default_caps_finish_a_symlink_loop(self, loop_root: Path, ws: LocalWorkspaceBackend) -> None:
        toolset = FileSystem[None](root_dir=loop_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        with anyio.fail_after(30):
            found = await toolset.find_files('**/*.txt', workspace=ws)
        assert found == 'a.txt'  # aliases are deduplicated without consuming the walk cap

    async def test_cut_walk_with_no_matches_still_says_so(
        self, loop_root: Path, monkeypatch: pytest.MonkeyPatch, ws: LocalWorkspaceBackend
    ) -> None:
        for index in range(6):
            (loop_root / f'dir{index}').mkdir()
        monkeypatch.setattr('pydantic_ai_harness.filesystem._toolset._MAX_WALK_DIRECTORIES', 5)
        toolset = FileSystem[None](root_dir=loop_root).get_toolset()
        assert isinstance(toolset, FileSystemToolset)
        result = await toolset.find_files('**/*.missing', workspace=ws)
        assert result.splitlines()[0] == 'No matches found.'
        assert result.splitlines()[1].startswith('[... walk cut short after ')

    @pytest.mark.parametrize('anyio_backend', ['asyncio'])  # Agent.run needs asyncio
    async def test_cut_walk_marks_the_event_truncated(
        self, loop_root: Path, anyio_backend: object, ws: LocalWorkspaceBackend
    ) -> None:
        seen: list[FilesSearchedEvent] = []

        class Recorder(AbstractCapability[None]):
            @on_event(FilesSearchedEvent)
            async def searched(self, ctx: RunContext[None], event: FilesSearchedEvent) -> None:
                seen.append(event)

        for index in range(6):
            (loop_root / f'dir{index}').mkdir()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr('pydantic_ai_harness.filesystem._toolset._MAX_WALK_DIRECTORIES', 5)
            await call_tool(
                [FileSystem[None](root_dir=loop_root), Recorder()],
                'search_files',
                {'pattern': 'needle'},
                workspace=FilesystemOnlyWorkspace(loop_root),
            )
        assert [event.truncated for event in seen] == [True]


def _toolset_with_patterns(read_only: list[str] | None, protected: list[str]) -> FileSystemToolset[None]:
    return FileSystemToolset[None](
        allowed_patterns=[],
        denied_patterns=[],
        read_only_patterns=read_only,
        protected_patterns=protected,
        max_read_lines=1,
        max_list_results=1,
        max_search_results=1,
        max_find_results=1,
    )


class TestUsedAsAToolset:
    """The tools emit capability events, which core accepts only from a capability's tools."""

    @pytest.fixture
    def anyio_backend(self) -> str:
        # Agent.run needs asyncio.
        return 'asyncio'

    @pytest.mark.parametrize('prefix', [None, 'fs'])
    async def test_a_bare_toolset_points_to_the_capability(self, tmp_path: Path, prefix: str | None) -> None:
        (tmp_path / 'a.txt').write_text('a\n')
        toolset = FileSystem[object]().get_toolset()
        name = 'read_file' if prefix is None else f'{prefix}_read_file'

        def read(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart(name, {'path': 'a.txt'})])

        agent = Agent(
            FunctionModel(read),
            capabilities=[LocalWorkspace(tmp_path)],
            toolsets=[toolset if prefix is None else toolset.prefixed(prefix)],
        )
        with pytest.raises(UserError, match=r'Pass `capabilities=\[FileSystem\(\)\]` rather than its toolset'):
            await agent.run('go')

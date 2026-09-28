"""Search on command-capable workspaces without ripgrep."""

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.workspaces import CommandResult, LocalWorkspaceBackend, Workspace, WorkspaceCommand
from pydantic_ai_harness.filesystem import FileSystem, FileSystemToolset

from .conftest import tools_path

pytestmark = pytest.mark.anyio


class CountingBackend(LocalWorkspaceBackend):
    def __init__(self, root: Path, path: str) -> None:
        super().__init__(root, env={'PATH': path})
        self.commands = 0
        self.reads = 0
        self.realpaths = 0

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        self.commands += 1
        return await super().run(command, shell=shell, env=env, timeout=timeout)

    async def realpath(self, path: str) -> str:
        self.realpaths += 1
        return await super().realpath(path)

    # Never called: the tests assert `reads == 0`, searching reads no file one by one.
    async def read_bytes(self, path: str) -> bytes:  # pragma: no cover
        self.reads += 1
        return await super().read_bytes(path)


async def test_no_rg_uses_one_command_and_preserves_ignores(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'visible.py').write_text('needle\n')
    (tmp_path / 'ignored').mkdir()
    (tmp_path / 'ignored' / 'secret.py').write_text('needle\n')
    (tmp_path / '.gitignore').write_text('ignored/\n')
    (tmp_path / '.hidden.py').write_text('needle\n')
    subprocess.run(['git', '-C', str(tmp_path), 'init', '-q'], check=True)
    backend = CountingBackend(tmp_path, no_rg_path)
    workspace = Workspace(backend)
    tools = FileSystem[None](tools=['grep', 'list_files']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.grep('needle', workspace=workspace) == 'src/visible.py:1:needle'
    assert backend.commands == 2  # probe then POSIX search with canonical paths
    assert backend.reads == 0
    assert await tools.list_files(glob='*.py', workspace=workspace) == 'src/visible.py'
    assert backend.commands == 3
    assert backend.reads == 0


async def test_no_rg_search_files_is_one_command_and_confines_symlinks(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / 'safe.txt').write_text('needle\n')
    (tmp_path / 'secret.txt').write_text('needle\n')
    (tmp_path / 'alias.txt').symlink_to('secret.txt')
    outside = tmp_path.parent / f'{tmp_path.name}-outside'
    outside.write_text('needle\n')
    try:
        (tmp_path / 'escape.txt').symlink_to(outside)
        backend = CountingBackend(tmp_path, no_rg_path)
        tools = FileSystem[None](root_dir=tmp_path, denied_patterns=['secret.txt']).get_toolset()
        assert isinstance(tools, FileSystemToolset)
        workspace = Workspace(backend)
        await tools.grep('absent', workspace=workspace)  # Discover the missing sandbox rg.
        backend.commands = 0
        result = await tools.search_files('needle', workspace=workspace)
        assert 'safe.txt:1:needle' in result
        assert 'secret.txt' not in result
        assert 'alias.txt' not in result
        assert 'escape.txt' not in result
        assert backend.commands <= 2
        assert backend.reads == 0
    finally:
        outside.unlink()


async def test_many_search_results_use_batched_path_checks(tmp_path: Path, no_rg_path: str) -> None:
    for index in range(20):
        (tmp_path / f'{index:02}.txt').write_text('needle\n')
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path, max_search_results=25, max_find_results=25).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    await tools.search_files('absent', workspace=backend)
    for call in (
        lambda: tools.search_files('needle', workspace=backend),
        lambda: tools.grep('needle', workspace=backend),
        lambda: tools.list_files(workspace=backend),
    ):
        backend.commands = 0
        backend.realpaths = 0
        assert len((await call()).splitlines()) == 20
        assert backend.commands <= 3
        assert backend.realpaths <= 2


async def test_large_listing_checks_paths_in_search_command(tmp_path: Path, no_rg_path: str) -> None:
    for index in range(600):
        (tmp_path / f'{index:03}.txt').write_text('needle\n')
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path, max_find_results=600).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert len((await tools.list_files(workspace=backend)).splitlines()) == 600
    assert backend.commands <= 3
    assert backend.reads == 0


async def test_first_search_files_uses_command_not_walker(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / 'visible.txt').write_text('needle\n')
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.search_files('needle', workspace=backend) == 'visible.txt:1:needle'
    assert backend.commands == 2  # probe rg, then search with canonical paths
    assert backend.reads == 0


async def test_no_rg_ignore_and_failure_are_not_silent(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / 'visible.txt').write_text('needle\n')
    (tmp_path / 'hidden.txt').write_text('needle\n')
    (tmp_path / '.ignore').write_text('hidden.txt\n')
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    workspace = Workspace(backend)
    await tools.grep('absent', workspace=workspace)
    assert 'hidden.txt' not in await tools.search_files('needle', workspace=workspace)
    assert backend.reads == 0


async def test_explicit_hidden_file_glob_and_omission_count(tmp_path: Path) -> None:
    (tmp_path / '.hidden.py').write_text('needle\n')
    (tmp_path / '.another.py').write_text('needle\n')
    (tmp_path / 'visible.py').write_text('needle\n')
    backend = LocalWorkspaceBackend(tmp_path)
    tools = FileSystem[None](root_dir=tmp_path, tools=['list_files']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    listing = await tools.list_directory(workspace=backend)
    assert '2 hidden' in listing
    assert '.hidden.py' in await tools.find_files('.hidden.py', workspace=backend)
    assert '.hidden.py' in await tools.search_files('needle', include_glob='.hidden.py', workspace=backend)
    assert '.hidden.py' in await tools.list_files(glob='.hidden.py', workspace=backend)


async def test_recursive_find_reports_hidden_omissions(tmp_path: Path) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / '.secret.txt').write_text('needle\n')
    (tmp_path / 'src' / 'visible.txt').write_text('needle\n')
    (tmp_path / '.private').mkdir()
    (tmp_path / '.private' / 'secret.txt').write_text('needle\n')
    tools = FileSystem[None](root_dir=tmp_path).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    result = await tools.find_files('**/*.txt', workspace=LocalWorkspaceBackend(tmp_path))
    assert '2 hidden entries omitted' in result
    assert 'src/visible.txt' in result


async def test_nested_gitignore_on_posix_search(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / '.gitignore').write_text('ignored.txt\n')
    (tmp_path / 'src' / 'ignored.txt').write_text('needle\n')
    (tmp_path / 'src' / 'visible.txt').write_text('needle\n')
    subprocess.run(['git', '-C', str(tmp_path), 'init', '-q'], check=True)
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.search_files('needle', workspace=backend) == 'src/visible.txt:1:needle'
    assert backend.reads == 0


async def test_posix_grep_explicit_ignored_file(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / '.gitignore').write_text('ignored.txt\n')
    (tmp_path / 'ignored.txt').write_text('needle\n')
    subprocess.run(['git', '-C', str(tmp_path), 'init', '-q'], check=True)
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path, tools=['grep']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.grep('needle', workspace=backend) == 'No matches found.'
    assert await tools.grep('needle', path='ignored.txt', workspace=backend) == 'ignored.txt:1:needle'


async def test_posix_output_cap_reports_truncation(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / 'many.txt').write_text(('needle ' + 'X' * 100 + '\n') * 85000)
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path, max_search_results=200000, tools=['grep']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    result = await tools.grep('needle', workspace=backend)
    assert result.startswith('many.txt:1:needle')
    assert 'truncated' in result


async def test_no_rg_rejects_unsupported_regex(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / 'file.txt').write_text('needle\n')
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path, tools=['grep']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    with pytest.raises((ModelRetry, ValueError), match=r'ripgrep|POSIX|unsupported'):
        await tools.grep(r'\d+', workspace=backend)


async def test_posix_find_includes_explicitly_named_hidden_file(tmp_path: Path, no_rg_git_path: str) -> None:
    (tmp_path / '.secret').write_text('needle\n')
    (tmp_path / 'visible.txt').write_text('needle\n')
    backend = CountingBackend(tmp_path, no_rg_git_path)
    tools = FileSystem[None](root_dir=tmp_path, tools=['grep', 'list_files']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.list_files(glob='.secret', workspace=backend) == '.secret'
    assert await tools.grep('needle', glob='.secret', workspace=backend) == '.secret:1:needle'
    assert await tools.grep('needle', workspace=backend) == 'visible.txt:1:needle'


async def test_posix_grep_context_drops_group_separator(tmp_path: Path, no_rg_path: str) -> None:
    (tmp_path / 'file.txt').write_text('needle\nb\nc\nd\ne\nneedle\n')
    backend = CountingBackend(tmp_path, no_rg_path)
    tools = FileSystem[None](root_dir=tmp_path, max_search_results=3, tools=['grep']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    assert await tools.grep('needle', context=1, workspace=backend) == (
        'file.txt:1:needle\nfile.txt-2-b\nfile.txt-5-e\n[... truncated at 3 lines]'
    )


async def test_posix_search_reports_failed_sort(tmp_path: Path) -> None:
    bin_dir = tmp_path / 'bin'
    path = tools_path(bin_dir, exclude=frozenset({'rg', 'sort'}))
    (bin_dir / 'sort').write_text('#!/bin/sh\nexit 1\n')
    (bin_dir / 'sort').chmod(0o755)
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'file.txt').write_text('needle\n')
    backend = CountingBackend(workspace, path)
    tools = FileSystem[None](root_dir=workspace, tools=['grep', 'list_files']).get_toolset()
    assert isinstance(tools, FileSystemToolset)
    with pytest.raises(ModelRetry, match='POSIX search failed'):
        await tools.grep('needle', workspace=backend)
    with pytest.raises(ModelRetry, match='POSIX search failed'):
        await tools.list_files(workspace=backend)

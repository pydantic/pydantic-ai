"""`Coder` and its tools in a `'windows'` workspace.

The file tools are checked everywhere, in a fake `'windows'` workspace whose `C:` drive is a directory
here. `shell` runs PowerShell, so it, and the fallback to a real `WindowsWorkspace`, are checked only on
Windows, where the `test-windows-workspace` CI job runs them.
"""

from __future__ import annotations

import ntpath
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace, WindowsWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    LocalWorkspaceBackend,
    Workspace,
    WorkspaceCommand,
    WorkspacePlatform,
    WorkspaceRef,
    local as local_module,
)
from pydantic_ai_harness import Coder
from pydantic_ai_harness.coder import FILE_TOOL_NAMES
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.shell import Shell

from .._tool_calls import call_tool, call_tools

windows_only = pytest.mark.skipif(sys.platform != 'win32', reason='runs PowerShell')
posix_only = pytest.mark.skipif(sys.platform == 'win32', reason='stands in with `LocalWorkspace`, which needs POSIX')


class DriveBackend:
    """A `'windows'` workspace at `C:\\project`, whose `C:` drive is `drive` on this machine."""

    def __init__(self, drive: Path, *, env: Mapping[str, str] | None = None) -> None:
        self.drive = str(drive.resolve())
        (drive / 'project').mkdir(exist_ok=True)
        self.inner = LocalWorkspaceBackend(drive, env=env)

    def host(self, path: str) -> str:
        drive, rest = ntpath.splitdrive(ntpath.normpath(path))
        assert drive == 'C:', path
        return self.drive + rest.replace('\\', '/').rstrip('/')

    def native(self, path: str) -> str:
        return 'C:' + (path.removeprefix(self.drive).replace('/', '\\') or '\\')

    def entry(self, entry: FileEntry) -> FileEntry:
        return FileEntry(name=entry.name, path=self.native(entry.path), is_dir=entry.is_dir, size=entry.size)

    @property
    def ref(self) -> WorkspaceRef:
        return WorkspaceRef(provider='windows', id='C:\\project')

    @property
    def platform(self) -> WorkspacePlatform:
        return 'windows'

    async def working_dir(self) -> str:
        return 'C:\\project'

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        cwd: str | None = None,
    ) -> CommandResult:
        assert not shell, 'the file tools run argv commands'
        directory = self.host(cwd or await self.working_dir())
        return await Workspace(self.inner).run(command, env=env, timeout=timeout, cwd=directory)

    async def read_bytes(self, path: str) -> bytes:
        return await self.inner.read_bytes(self.host(path))

    async def write_bytes(self, path: str, data: bytes) -> None:
        await self.inner.write_bytes(self.host(path), data)

    async def stat(self, path: str) -> FileEntry:
        return self.entry(await self.inner.stat(self.host(path)))

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        return [self.entry(entry) for entry in await self.inner.list_dir(self.host(path))]

    async def make_dir(self, path: str) -> None:
        raise NotImplementedError

    async def remove(self, path: str) -> None:
        raise NotImplementedError

    async def exists(self, path: str) -> bool:
        raise NotImplementedError

    async def realpath(self, path: str) -> str:
        return self.native(await self.inner.realpath(self.host(path)))


@pytest.fixture
def drive(tmp_path: Path) -> Path:
    (tmp_path / 'project' / 'src').mkdir(parents=True)
    (tmp_path / 'project' / 'src' / 'app.py').write_text('import os\n\n\ndef main():\n    return os.name\n')
    (tmp_path / 'project' / 'notes.txt').write_text('main notes\n')
    (tmp_path / 'outside.txt').write_text('secret\n')
    return tmp_path


def file_system(**settings: object) -> FileSystem[None]:
    return FileSystem[None](tools=FILE_TOOL_NAMES, content_hashes=False, **settings)  # pyright: ignore[reportArgumentType]


@posix_only
class TestFileToolsOnWindows:
    async def test_paths_are_windows_paths(self, drive: Path) -> None:
        results = await call_tools(
            [file_system()],
            [
                ('read_file', {'path': 'src\\app.py'}),
                ('write_file', {'path': 'C:\\project\\src\\new.py', 'content': 'x = 1\n'}),
                ('edit_file', {'path': 'src/new.py', 'old_text': 'x = 1', 'new_text': 'x = 2'}),
                ('read_file', {'path': 'C:\\outside.txt'}),
            ],
            workspace=DriveBackend(drive),
        )
        assert results[0].startswith('[src\\app.py | 5 lines]')
        assert results[1].startswith('Wrote 6 chars')
        assert (drive / 'project' / 'src' / 'new.py').read_text() == 'x = 2\n'
        assert 'is outside the project root `C:\\project`' in results[3]

    async def test_ripgrep_runs_without_a_shell(self, drive: Path) -> None:
        backend = DriveBackend(drive)
        results = await call_tools(
            [file_system()],
            [
                ('list_files', {}),
                ('grep', {'pattern': 'main'}),
                ('grep', {'pattern': 'main', 'path': 'src\\app.py'}),
            ],
            workspace=backend,
        )
        assert results == [
            'notes.txt\nsrc/app.py',
            'notes.txt:1:main notes\nsrc/app.py:4:def main():',
            'src/app.py:4:def main():',
        ]

    async def test_search_without_ripgrep_walks_the_files(self, drive: Path) -> None:
        """There is no POSIX `grep` or `find` to fall back to, so the files are read instead."""
        backend = DriveBackend(drive, env={'PATH': str(drive / 'no-tools')})
        tools = FileSystem[None](tools=[*FILE_TOOL_NAMES, 'search_files'], content_hashes=False)
        # Each run finds `rg` missing anew; the second call of a run walks the files at once.
        assert await call_tools([tools], [('list_files', {}), ('list_files', {'path': 'src'})], workspace=backend) == [
            'notes.txt\nsrc/app.py',
            'src/app.py',
        ]
        grep = await call_tool([tools], 'grep', {'pattern': 'main', 'glob': '*.py'}, workspace=backend)
        assert grep == 'src/app.py:4:def main():'
        search = await call_tool([tools], 'search_files', {'pattern': 'def main'}, workspace=backend)
        assert search == 'src/app.py:4:def main():'

    async def test_output_past_the_workspace_cap_is_cut(self, drive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        for index in range(100):
            (drive / 'project' / f'file{index:03}-{"x" * 40}.txt').write_text('main\n')
        # Above what `rg --version` prints, below the listing.
        monkeypatch.setattr(local_module, '_MAX_CAPTURE_BYTES', 1000)
        result = await call_tool([file_system()], 'list_files', {}, workspace=DriveBackend(drive))
        assert result.startswith('file000-') and result.endswith(']')
        assert 'truncated' in result

    async def test_root_dir_is_a_windows_path(self, drive: Path) -> None:
        result = await call_tool(
            [file_system(root_dir=Path('C:/project'))],
            'read_file',
            {'path': 'notes.txt'},
            workspace=DriveBackend(drive),
        )
        assert result.startswith('[notes.txt | 1 lines]')

    async def test_unrestricted_reaches_the_whole_drive(self, drive: Path) -> None:
        result = await call_tool(
            [file_system(root_dir='/')], 'read_file', {'path': 'C:\\outside.txt'}, workspace=DriveBackend(drive)
        )
        assert '1\tsecret' in result


@posix_only
class TestWindowsFallback:
    @pytest.fixture
    def on_windows(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
        """Pretend to run on Windows, with `LocalWorkspace` standing in for `WindowsWorkspace`, in `tmp_path`."""
        monkeypatch.setattr('pydantic_ai_harness.coder._capability.sys', SimpleNamespace(platform='win32'))
        monkeypatch.setattr('pydantic_ai_harness.coder._capability.WindowsWorkspace', LocalWorkspace)
        monkeypatch.chdir(tmp_path)
        return tmp_path

    async def test_a_run_without_a_workspace_gets_the_current_directory(self, on_windows: Path) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[Coder(repo_context=False, sub_agents=False)])
        result = await agent.run('hello')
        assert result.workspace is not None and result.workspace.ref == WorkspaceRef(
            provider='local', id=str(on_windows)
        )

    async def test_another_workspace_wins(self, on_windows: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
        other = tmp_path_factory.mktemp('other')
        agent = Agent(
            TestModel(call_tools=[]),
            capabilities=[Coder(repo_context=False, sub_agents=False), LocalWorkspace(other)],
        )
        result = await agent.run('hello')
        assert result.workspace is not None and result.workspace.ref == WorkspaceRef(provider='local', id=str(other))

    async def test_only_on_windows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr('pydantic_ai_harness.coder._capability.sys', SimpleNamespace(platform='linux'))
        agent = Agent(TestModel(call_tools=[]), capabilities=[Coder(repo_context=False, sub_agents=False)])
        with pytest.raises(Exception, match='needs a workspace'):
            await agent.run('hello')


@windows_only
class TestShellOnWindows:
    @pytest.fixture(autouse=True)
    def project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        return tmp_path

    async def test_coder_works_in_the_current_directory(self, project: Path) -> None:
        (project / 'app.py').write_text('def main():\n    pass\n')
        results = await call_tools(
            [Coder[None](repo_context=False, sub_agents=False)],
            [
                ('shell', {'command': "Write-Output 'héllo'; Write-Error oops; cmd /c exit 3"}),
                ('shell', {'command': 'Start-Sleep 30', 'mode': 'background'}),
                ('grep', {'pattern': 'def main'}),
                ('read_file', {'path': 'app.py'}),
            ],
        )
        assert 'héllo' in results[0] and 'oops' in results[0] and '"exit_code": 3}' in results[0]
        assert 'taskkill /T /F /PID' in results[1] and '"exit_code": null}' in results[1]
        assert results[2] == 'app.py:1:def main():'
        assert results[3].startswith('[app.py | 2 lines]')

    async def test_log_limits_need_posix(self, project: Path) -> None:
        shell = Shell[None](tools=['shell'], max_log_bytes=1000)
        result = await call_tool([shell, WindowsWorkspace[None](project)], 'shell', {'command': 'Write-Output hi'})
        assert 'need a POSIX workspace' in result

    async def test_stopping_a_background_command_ends_its_tree(self, project: Path) -> None:
        shell = Shell[None](tools=['start_command', 'check_command', 'stop_command'])
        workspace = WindowsWorkspace[None](project)
        started = await call_tool(
            [shell, workspace],
            'start_command',
            {'command': "Write-Output started; [Console]::Error.WriteLine('err'); Start-Sleep 600"},
        )
        assert 'ID: ' in started, started
        command_id = started.rsplit('ID: ', 1)[1].strip()
        results = await call_tools(
            [shell, workspace],
            [('check_command', {'command_id': command_id}), ('stop_command', {'command_id': command_id})],
        )
        assert '[status: running]' in results[0]
        assert '[stdout]\nstarted' in results[1] and '[stderr]\nerr' in results[1]
        assert '[stopped]' in results[1] and '[exit code: 1]' in results[1]

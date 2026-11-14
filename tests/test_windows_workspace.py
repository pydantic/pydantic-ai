"""`WindowsWorkspaceBackend` and the `'windows'` workspace platform.

The backend itself only runs on Windows, where the `test-windows-workspace` CI job runs these tests;
how `Workspace` treats a `'windows'` backend's paths is checked everywhere, with a fake.
"""

from __future__ import annotations

import ntpath
import os
import posixpath
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import WindowsWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    CommandResult,
    FileEntry,
    LocalWorkspaceBackend,
    WindowsWorkspaceBackend,
    Workspace,
    WorkspaceBackend,
    WorkspaceCommand,
    WorkspacePlatform,
    WorkspaceRef,
    WorkspaceTimeoutError,
)
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite

windows_only = pytest.mark.skipif(sys.platform != 'win32', reason='`WindowsWorkspaceBackend` runs only on Windows')


class FakeWindowsBackend:
    """A `'windows'` backend that records the commands it is asked to run, and where."""

    def __init__(self) -> None:
        self.runs: list[tuple[WorkspaceCommand, bool, str | None]] = []

    ref = WorkspaceRef(provider='windows', id='C:\\project')

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
        self.runs.append((command, shell, cwd))
        return CommandResult(exit_code=0, stdout='', stderr='')

    async def stat(self, path: str) -> FileEntry:
        return FileEntry(name=ntpath.basename(path), path=path, is_dir=not path.endswith('.txt'), size=None)

    async def realpath(self, path: str) -> str:
        return ntpath.normpath(path)

    async def read_bytes(self, path: str) -> bytes:
        raise NotImplementedError

    async def write_bytes(self, path: str, data: bytes) -> None:
        raise NotImplementedError

    async def list_dir(self, path: str) -> Sequence[FileEntry]:
        raise NotImplementedError

    async def make_dir(self, path: str) -> None:
        raise NotImplementedError

    async def remove(self, path: str) -> None:
        raise NotImplementedError

    async def exists(self, path: str) -> bool:
        raise NotImplementedError


class TestWindowsPaths:
    @pytest.mark.skipif(sys.platform == 'win32', reason='`LocalWorkspaceBackend` needs POSIX')
    async def test_a_backend_is_posix_unless_it_says_otherwise(self, tmp_path: Path) -> None:
        workspace = Workspace(LocalWorkspaceBackend(tmp_path))
        assert (workspace.platform, workspace.path) == ('posix', posixpath)

    async def test_paths_resolve_as_windows_paths(self) -> None:
        workspace = Workspace(Workspace(FakeWindowsBackend()))
        assert (workspace.platform, workspace.path) == ('windows', ntpath)
        assert await workspace.resolve('src/app.py') == 'C:\\project\\src\\app.py'
        assert await workspace.resolve('D:/data/../logs') == 'D:\\logs'
        # A path with a root but no drive is on the working directory's drive.
        assert await workspace.resolve('/') == 'C:\\'
        assert await workspace.resolve('app.py', base='C:\\other') == 'C:\\other\\app.py'
        with pytest.raises(ValueError, match='base must be an absolute path'):
            await workspace.resolve('app.py', base='other')
        assert await workspace.realpath('src\\..\\app.py') == 'C:\\project\\app.py'

    async def test_run_passes_the_directory_to_the_backend(self) -> None:
        """Without `sh`, `Workspace.run` can't change directory itself, so the backend starts the command there."""
        backend = FakeWindowsBackend()
        workspace = Workspace(backend)
        await workspace.run(['rg', '--files'], cwd='src')
        await workspace.run('Get-ChildItem', shell=True, cwd='C:\\project\\docs')
        assert backend.runs == [
            (['rg', '--files'], False, 'C:\\project\\src'),
            ('Get-ChildItem', True, 'C:\\project\\docs'),
        ]
        with pytest.raises(NotADirectoryError):
            await workspace.run(['rg'], cwd='notes.txt')
        with pytest.raises(TypeError):
            await workspace.run('rg', cwd='src')

    async def test_authority_of_a_windows_path(self) -> None:
        workspace = Workspace(FakeWindowsBackend())
        assert await workspace.authority_of('src\\app.py') == 'operator'
        assert await workspace.authority_of('D:\\elsewhere') == 'none'


@pytest.mark.skipif(sys.platform == 'win32', reason='checks the refusal off Windows')
def test_windows_workspace_needs_windows(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match='only runs on Windows'):
        WindowsWorkspaceBackend(tmp_path)
    with pytest.raises(NotImplementedError, match='only runs on Windows'):
        WindowsWorkspace(tmp_path)


@windows_only
class TestWindowsWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, tmp_path: Path) -> WindowsWorkspaceBackend:
        return WindowsWorkspaceBackend(tmp_path)

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: WindowsWorkspaceBackend(ref.id)


@windows_only
class TestWindowsCommands:
    async def test_windows_powershell(self, tmp_path: Path) -> None:
        backend = WindowsWorkspaceBackend(tmp_path, powershell='powershell.exe')
        result = await backend.run('$PSVersionTable.PSEdition; cmd /c exit 3', shell=True)
        assert (result.exit_code, result.stdout.strip()) == (3, 'Desktop')

    async def test_a_cmdlet_failure_exits_1(self, tmp_path: Path) -> None:
        result = await WindowsWorkspaceBackend(tmp_path).run('Get-Item missing.txt', shell=True)
        assert result.exit_code == 1
        assert 'missing.txt' in result.stderr

    async def test_timeout_ends_the_whole_process_tree(self, tmp_path: Path) -> None:
        backend = WindowsWorkspaceBackend(tmp_path)
        workspace = Workspace(backend)
        script = (
            '$child = Start-Process -NoNewWindow -PassThru -FilePath (Get-Process -Id $PID).Path '
            "-ArgumentList '-NoProfile', '-Command', 'Start-Sleep 600'; "
            '[IO.File]::WriteAllText(\'child.pid\', "$($child.Id)"); Start-Sleep 600'
        )
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(script, shell=True, timeout=10)
        pid = (await workspace.read_text('child.pid')).strip()
        with anyio.fail_after(30):
            while (await backend.run(f'Get-Process -Id {pid}', shell=True)).exit_code == 0:
                await anyio.sleep(0.1)  # pragma: lax no cover - depends on how fast the process dies

    async def test_a_bare_program_is_not_run_from_the_current_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`CreateProcess` would run a checkout's `rg.exe` in place of the real one; like `execvp`, only `PATH` counts."""
        shutil.copy(Path(os.environ['SYSTEMROOT'], 'System32', 'findstr.exe'), tmp_path / 'hijack.exe')
        monkeypatch.chdir(tmp_path)
        result = await WindowsWorkspaceBackend(tmp_path).run(['hijack', '/?'])
        assert (result.exit_code, result.stderr) == (127, 'hijack: command not found\n')

    async def test_batch_files_are_refused_as_argv_programs(self, tmp_path: Path) -> None:
        """`cmd.exe` would parse their arguments as a command line, so `x&whoami` could run `whoami`."""
        (tmp_path / 'tool.cmd').write_text('@echo %*\n')
        backend = WindowsWorkspaceBackend(tmp_path, env={'PATH': str(tmp_path)})
        explicit = await backend.run([str(tmp_path / 'tool.cmd'), 'x&whoami'])
        assert (explicit.exit_code, explicit.stdout) == (126, '')
        # On `PATH`, a batch file is passed over, as if it weren't there.
        assert (await backend.run(['tool', 'x&whoami'])).exit_code == 127

    async def test_stream_names_are_refused(self, tmp_path: Path) -> None:
        """`.env::$DATA` names `.env`'s data, so it would get past a policy that matched the spelling."""
        workspace = Workspace(WindowsWorkspaceBackend(tmp_path))
        for path in ('.env::$DATA', 'notes.txt:hidden'):
            with pytest.raises(ValueError, match='NTFS stream names are not supported'):
                await workspace.write_text(path, 'secret')
            with pytest.raises(ValueError, match='NTFS stream names are not supported'):
                await workspace.realpath(path)
        assert not (tmp_path / '.env').exists()

    async def test_a_script_too_long_for_a_command_line(self, tmp_path: Path) -> None:
        with pytest.raises(OSError, match='too long'):
            await WindowsWorkspaceBackend(tmp_path).run('#' * 20_000, shell=True)

    async def test_capability(self, tmp_path: Path) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[WindowsWorkspace(tmp_path)])
        result = await agent.run('hello')
        assert result.workspace is not None
        assert result.workspace.ref == WorkspaceRef(provider='windows', id=str(tmp_path))
        assert result.workspace.platform == 'windows'

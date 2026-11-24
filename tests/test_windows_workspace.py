"""`WindowsWorkspaceBackend`, which runs only on Windows, where the `test-windows-workspace` CI job runs these tests.

How `Workspace` treats a `'windows'` backend's paths is checked everywhere, with a fake, in `tests/test_workspace.py`.
"""

from __future__ import annotations

import codecs
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import anyio
import anyio.abc
import pytest

from pydantic_ai import Agent
from pydantic_ai.capabilities import WindowsWorkspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    WindowsWorkspaceBackend,
    Workspace,
    WorkspaceBackend,
    WorkspaceRef,
    WorkspaceTimeoutError,
)
from pydantic_ai.workspaces.conformance import Unsupported, WorkspaceBackendSuite
from pydantic_ai.workspaces.windows import powershell_script

windows_only = pytest.mark.skipif(sys.platform != 'win32', reason='`WindowsWorkspaceBackend` runs only on Windows')


@windows_only
class TestWindowsWorkspaceBackend(WorkspaceBackendSuite):
    @pytest.fixture
    def backend(self, tmp_path: Path) -> WindowsWorkspaceBackend:
        return WindowsWorkspaceBackend(tmp_path)

    @pytest.fixture
    def fresh_backend(self) -> Unsupported:
        return Unsupported('the working directory exists before first use')

    @pytest.fixture
    def destroy_environment(self) -> Unsupported:
        return Unsupported('Windows refuses to delete a directory a running command works in')

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
        # A relative `PATH` entry names the current directory, which is never searched.
        backend = WindowsWorkspaceBackend(tmp_path, env={'PATH': f'.;{tmp_path}'})
        explicit = await backend.run([str(tmp_path / 'tool.cmd'), 'x&whoami'])
        assert (explicit.exit_code, explicit.stdout) == (126, '')
        # On `PATH`, a batch file is passed over, as if it weren't there.
        assert (await backend.run(['tool', 'x&whoami'])).exit_code == 127

    @pytest.mark.parametrize(
        ('path', 'written', 'error'),
        [
            ('.env::$DATA', '.env', 'NTFS stream names are not supported'),
            ('notes.txt:hidden', 'notes.txt', 'NTFS stream names are not supported'),
            ('server.key.', 'server.key', 'names ending in a dot or a space are not supported'),
            ('server.key ', 'server.key', 'names ending in a dot or a space are not supported'),
            ('keys. \\server.key', 'keys\\server.key', 'names ending in a dot or a space are not supported'),
        ],
    )
    async def test_aliases_of_another_name_are_refused(
        self, tmp_path: Path, path: str, written: str, error: str
    ) -> None:
        """Windows writes `.env::$DATA` to `.env` and `server.key.` to `server.key`, past a policy matching the spelling."""
        workspace = Workspace(WindowsWorkspaceBackend(tmp_path))
        with pytest.raises(ValueError, match=error):
            await workspace.write_text(path, 'secret')
        with pytest.raises(ValueError, match=error):
            await workspace.realpath(path)
        assert not (tmp_path / written).exists()

    async def test_commands_inherit_only_what_windows_tools_need(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv('WINDOWS_WORKSPACE_HOST_SECRET', 'do-not-pass')
        backend = WindowsWorkspaceBackend(tmp_path, env={'WINDOWS_WORKSPACE_ADDED': 'yes'})
        result = await backend.run('Get-ChildItem env: | ForEach-Object Name', shell=True)
        names = {name.upper() for name in result.stdout.split()}
        assert 'WINDOWS_WORKSPACE_HOST_SECRET' not in names
        assert {'PATH', 'SYSTEMROOT', 'WINDOWS_WORKSPACE_ADDED'} <= names

    async def test_a_long_non_ascii_script_runs_from_a_file_it_removes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Longer than one Windows command line holds, which an `-EncodedCommand` script once had to fit in."""
        temp = tmp_path / 'temp'
        temp.mkdir()
        monkeypatch.setattr(tempfile, 'tempdir', str(temp))
        script = '#' * 40_000 + "\n[Console]::Out.Write('héllo ✓')"
        result = await WindowsWorkspaceBackend(tmp_path).run(script, shell=True)
        assert (result.exit_code, result.stdout) == (0, 'héllo ✓')
        assert list(temp.iterdir()) == []

    async def test_the_script_directory_is_owner_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The script may hold secrets, so its directory doesn't inherit the temporary directory's ACL."""
        temp = tmp_path / 'temp'
        temp.mkdir()
        monkeypatch.setattr(tempfile, 'tempdir', str(temp))
        script = (
            f"$acl = Get-Acl -LiteralPath (Get-ChildItem -LiteralPath '{temp}' -Directory)[0].FullName; "
            '[Console]::Out.Write("$($acl.AreAccessRulesProtected) " + ($acl.Access.IdentityReference -join \';\'))'
        )
        result = await WindowsWorkspaceBackend(tmp_path).run(script, shell=True)
        protected, identities = result.stdout.split(' ', 1)
        assert protected == 'True', result
        assert not {'Users', 'Everyone', 'Authenticated Users'} & {
            name.split('\\')[-1] for name in identities.split(';')
        }

    async def test_capability(self, tmp_path: Path) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[WindowsWorkspace(tmp_path)])
        result = await agent.run('hello')
        assert result.workspace is not None
        assert result.workspace.ref == WorkspaceRef(provider='windows', id=str(tmp_path))
        assert result.workspace.platform == 'windows'


class _NoJobObjects:
    def assign(self, pid: int) -> None:
        return None


_POWERSHELL = r'C:\PowerShell\pwsh.exe'

_Launches = list[tuple[list[str], dict[Path, tuple[bytes, int]]]]


@pytest.mark.skipif(sys.platform == 'win32', reason='stands `/bin/sh` in for PowerShell; `TestWindowsCommands` runs it')
class TestPowerShellLaunch:
    """What a `shell=True` command starts: no script on the command line, where antivirus flags `-EncodedCommand`."""

    @pytest.fixture
    def temp(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        # Quotes in the path check that it is quoted for PowerShell, which also ends a string at U+2019.
        temp = tmp_path / "o'temp\u2019"
        temp.mkdir()
        monkeypatch.setattr(tempfile, 'tempdir', str(temp))
        return temp

    @pytest.fixture
    def launches(self, temp: Path, monkeypatch: pytest.MonkeyPatch) -> _Launches:
        """Each launch's argv, with the temporary files that existed then; `/bin/sh` runs in PowerShell's place."""
        launches: _Launches = []
        open_process = anyio.open_process

        def read_files() -> dict[Path, tuple[bytes, int]]:
            """Each file's content, and its directory's mode."""
            return {path: (path.read_bytes(), path.parent.stat().st_mode) for path in temp.rglob('*') if path.is_file()}

        async def fake_open_process(command: Sequence[str], **kwargs: Any) -> anyio.abc.Process:
            launches.append((list(command), await anyio.to_thread.run_sync(read_files)))
            return await open_process(['/bin/sh', '-c', 'exit 0'], **kwargs)

        monkeypatch.setattr('pydantic_ai.workspaces.windows._on_windows', lambda: True)
        monkeypatch.setattr('pydantic_ai.workspaces.windows._job_objects', _NoJobObjects)
        monkeypatch.setattr(anyio, 'open_process', fake_open_process)
        return launches

    @pytest.mark.parametrize('script', ["Write-Output 'héllo ✓'", '#' * 40_000], ids=['non-ascii', 'long'])
    async def test_the_script_runs_from_a_temporary_file(
        self, tmp_path: Path, temp: Path, launches: _Launches, script: str
    ) -> None:
        result = await WindowsWorkspaceBackend(tmp_path, powershell=_POWERSHELL).run(script, shell=True)
        assert result.exit_code == 0
        [(argv, files)] = launches
        [(path, (content, mode))] = files.items()
        assert path.suffix == '.ps1'
        assert '-EncodedCommand' not in argv
        quoted = str(path).replace("'", "''").replace('\u2019', '\u2019\u2019')
        assert argv == [
            _POWERSHELL,
            '-NoLogo',
            '-NoProfile',
            '-NonInteractive',
            '-ExecutionPolicy',
            'Bypass',
            '-OutputFormat',
            'Text',
            '-Command',
            f"& ([ScriptBlock]::Create([IO.File]::ReadAllText('{quoted}')))",
        ]
        assert len(subprocess.list2cmdline(argv)) < 1_000
        # UTF-8 with a BOM, like the harness's `command.ps1`.
        assert content == codecs.BOM_UTF8 + powershell_script(script).encode()
        # The script may hold secrets, so its directory is `mkdtemp`'s: owner-only here, and an owner-only ACL on Windows.
        assert stat.S_IMODE(mode) == 0o700
        assert list(temp.iterdir()) == []

    async def test_the_temporary_file_is_removed_when_powershell_cannot_start(
        self, tmp_path: Path, temp: Path, launches: _Launches
    ) -> None:
        with pytest.raises(FileNotFoundError):
            await WindowsWorkspaceBackend(tmp_path, powershell='missing-pwsh').run('Get-Date', shell=True)
        assert launches == []
        assert list(temp.iterdir()) == []

    async def test_the_temporary_file_is_removed_when_writing_it_fails(
        self, tmp_path: Path, temp: Path, launches: _Launches
    ) -> None:
        # A lone surrogate has no UTF-8 encoding.
        with pytest.raises(UnicodeEncodeError):
            await WindowsWorkspaceBackend(tmp_path, powershell=_POWERSHELL).run('Write-Output \ud800', shell=True)
        assert launches == []
        assert list(temp.iterdir()) == []

"""`WindowsWorkspaceBackend`, which runs only on Windows, where the `test-windows-workspace` CI job runs these tests.

How `Workspace` treats a `'windows'` backend's paths is checked everywhere, with a fake, in `tests/test_workspace.py`.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest
from inline_snapshot import snapshot

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
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite

windows_only = pytest.mark.skipif(sys.platform != 'win32', reason='`WindowsWorkspaceBackend` runs only on Windows')


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

    async def test_windows_powershell_errors_are_text(self, tmp_path: Path) -> None:
        """Windows PowerShell 5.1 writes its error stream as CLIXML when the script comes from `-EncodedCommand`."""
        backend = WindowsWorkspaceBackend(tmp_path, powershell='powershell.exe')
        result = await backend.run("Write-Error 'a <b> & _x000A_ \U0001f600'", shell=True)
        assert result.exit_code == 1
        assert 'a <b> & _x000A_ \U0001f600' in result.stderr
        assert 'CLIXML' not in result.stderr

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

    async def test_a_script_too_long_for_a_command_line(self, tmp_path: Path) -> None:
        with pytest.raises(OSError, match='too long'):
            await WindowsWorkspaceBackend(tmp_path).run('#' * 20_000, shell=True)

    async def test_capability(self, tmp_path: Path) -> None:
        agent = Agent(TestModel(call_tools=[]), capabilities=[WindowsWorkspace(tmp_path)])
        result = await agent.run('hello')
        assert result.workspace is not None
        assert result.workspace.ref == WorkspaceRef(provider='windows', id=str(tmp_path))
        assert result.workspace.platform == 'windows'


def _decode_stderr(stderr: str) -> str:
    return WindowsWorkspaceBackend._decode_stderr(stderr.encode())  # pyright: ignore[reportPrivateUsage]


def test_clixml_stderr_is_decoded() -> None:
    """Real Windows PowerShell 5.1 error output, decoded where it can't be produced: on non-Windows CI runners."""
    clixml = (
        '[Console]::Error output\r\n'
        '#< CLIXML\r\n'
        '<Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04">'
        '<S S="Error">Get-Item : Cannot find path \'C:\\work\\missing.txt\' because it does not exist._x000D__x000A_</S>'
        '<S S="Error">At line:4 char:1_x000D__x000A_</S>'
        '<S S="Error">+ Get-Item missing.txt_x000D__x000A_</S>'
        '<S S="Error">+ ~~~~~~~~~~~~~~~~~~~~_x000D__x000A_</S>'
        '<S S="Error">    + CategoryInfo          : ObjectNotFound: (C:\\work\\missing.txt:String) [Get-Item], ItemNotFoundExcep _x000D__x000A_</S>'
        '<S S="Error">   tion_x000D__x000A_</S>'
        '<S S="Error">    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.GetItemCommand_x000D__x000A_</S>'
        '<S S="Error"> _x000D__x000A_</S>'
        '<S S="Error">a &lt;b&gt; &amp; _x005F_x000A_ _xD83D__xDE00__x000D__x000A_</S>'
        '</Objs>'
    )
    assert _decode_stderr(clixml) == snapshot(
        """\
[Console]::Error output\r
Get-Item : Cannot find path 'C:\\work\\missing.txt' because it does not exist.\r
At line:4 char:1\r
+ Get-Item missing.txt\r
+ ~~~~~~~~~~~~~~~~~~~~\r
    + CategoryInfo          : ObjectNotFound: (C:\\work\\missing.txt:String) [Get-Item], ItemNotFoundExcep \r
   tion\r
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.GetItemCommand\r
 \r
a <b> & _x000A_ \U0001f600\r
"""
    )
    assert _decode_stderr('plain <S S="Error">text</S>\n') == 'plain <S S="Error">text</S>\n'
    record = '<Objs Version="1.1.0.1"><S S="Error">boom</S></Objs>'
    unclosed = '<Objs ><S S="Error">cut</S><Objs >'
    assert _decode_stderr(f'#< CLIXML\n{record}{unclosed}') == f'boom{unclosed}'

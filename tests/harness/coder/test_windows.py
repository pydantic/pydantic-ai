"""`Coder` and `Shell` on a real `WindowsWorkspace`, which the `test-windows-workspace` CI job runs."""

from __future__ import annotations

import sys
from pathlib import Path

import anyio
import pytest

from pydantic_ai.capabilities import WindowsWorkspace
from pydantic_ai_harness import Coder
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.shell import Shell

from .._tool_calls import call_tool, call_tools

windows_only = pytest.mark.skipif(sys.platform != 'win32', reason='runs PowerShell')


@windows_only
class TestShellOnWindows:
    @pytest.fixture(autouse=True)
    def project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        return tmp_path

    async def test_coder_works_in_the_current_directory(self, project: Path) -> None:
        (project / 'app.py').write_bytes(b'def main():\n    pass\n')
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

    async def test_file_limits_need_posix(self, project: Path) -> None:
        shell = Shell[None](tools=['start_command'], max_file_bytes=1000)
        result = await call_tool(
            [shell, WindowsWorkspace[None](project)], 'start_command', {'command': 'Write-Output hi'}
        )
        assert 'needs a POSIX workspace' in result

    async def test_a_job_directory_is_owner_only(self, project: Path) -> None:
        shell = Shell[None](tools=['start_command'])
        workspace = WindowsWorkspace[None](project)
        started = await call_tool([shell, workspace], 'start_command', {'command': 'Write-Output hi'})
        assert 'ID: ' in started, started
        directory = project / '.pydantic-ai-harness' / 'shell' / started.rsplit('ID: ', 1)[1].strip()
        script = (
            f"$acl = Get-Acl -LiteralPath '{directory}'; "
            '[Console]::Out.Write("$($acl.AreAccessRulesProtected) $(@($acl.Access).Count)")'
        )
        result = await call_tool([Shell[None](tools=['shell']), workspace], 'shell', {'command': script})
        assert 'True 1' in result

    async def test_a_stream_name_does_not_reach_a_protected_file(self, project: Path) -> None:
        result = await call_tool(
            [FileSystem[None](tools=['write_file']), WindowsWorkspace[None](project)],
            'write_file',
            {'path': '.env::$DATA', 'content': 'SECRET=1'},
        )
        assert 'NTFS stream names are not supported' in result
        assert not (project / '.env').exists()

    async def test_a_log_past_max_log_bytes_is_cleared(self, project: Path) -> None:
        shell = Shell[None](tools=['start_command', 'check_command'], max_log_bytes=1000)
        workspace = WindowsWorkspace[None](project)
        started = await call_tool([shell, workspace], 'start_command', {'command': "[Console]::Out.Write('x' * 5000)"})
        assert 'ID: ' in started, started
        check: dict[str, object] = {'command_id': started.rsplit('ID: ', 1)[1].strip()}
        with anyio.fail_after(60):
            while '[status: running]' in (checked := await call_tool([shell, workspace], 'check_command', check)):
                await anyio.sleep(0.5)  # pragma: lax no cover - depends on how fast PowerShell starts
        assert '[Log exceeded max_log_bytes=1000; earlier output was cleared.]' in checked
        assert 'x' * 1000 not in checked

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
        check: dict[str, object] = {'command_id': command_id}
        with anyio.fail_after(60):
            while '[stderr]\nerr' not in (checked := await call_tool([shell, workspace], 'check_command', check)):
                await anyio.sleep(0.5)  # pragma: lax no cover - depends on how fast PowerShell starts
        assert '[stdout]\nstarted' in checked and '[status: running]' in checked
        stopped = await call_tool([shell, workspace], 'stop_command', check)
        assert '[stdout]\nstarted' in stopped and '[stopped]' in stopped and '[exit code: 1]' in stopped

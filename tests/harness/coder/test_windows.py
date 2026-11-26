"""`Coder` and `Shell` on a real `WindowsWorkspace`, which the `test-windows-workspace` CI job runs."""

from __future__ import annotations

import ctypes
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import anyio
import pytest

from pydantic_ai import RunContext
from pydantic_ai.capabilities import WindowsWorkspace
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai.workspaces import Workspace
from pydantic_ai.workspaces.windows import WindowsWorkspaceBackend
from pydantic_ai_harness import Coder
from pydantic_ai_harness.file_system import FileSystem
from pydantic_ai_harness.shell import Shell, ShellToolset

from .._tool_calls import call_tool, call_tools

windows_only = pytest.mark.skipif(sys.platform != 'win32', reason='runs PowerShell')


class _CountsRefusals(WindowsWorkspaceBackend):
    """Counts the reads Windows refuses, and calls `on_refusal` after each, so a test can release what refused it."""

    def __init__(self, working_dir: Path, on_refusal: Callable[[], None]) -> None:
        super().__init__(working_dir)
        self.on_refusal = on_refusal
        self.refusals = 0

    async def read_bytes(self, path: str) -> bytes:
        try:
            return await super().read_bytes(path)
        except PermissionError:
            self.refusals += 1
            self.on_refusal()
            raise


@windows_only
class TestShellOnWindows:
    @pytest.fixture(autouse=True)
    def project(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.chdir(tmp_path)
        return tmp_path

    async def test_coder_works_in_a_windows_workspace(self, project: Path) -> None:
        (project / 'app.py').write_bytes(b'def main():\n    pass\n')
        results = await call_tools(
            [Coder[None](repo_context=False, sub_agents=False), WindowsWorkspace[None](project)],
            [
                ('shell', {'command': "Write-Output 'héllo'; Write-Error oops; cmd /c exit 3"}),
                ('shell', {'command': 'Start-Sleep 30', 'mode': 'background'}),
                ('grep', {'pattern': 'def main'}),
                ('read_file', {'path': 'app.py'}),
            ],
        )
        assert 'héllo' in results[0] and 'oops' in results[0] and '"exit_code": 3}' in results[0]
        assert "stop.ps1'` to stop" in results[1] and '"exit_code": null}' in results[1]
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

    async def test_a_retried_launch_without_a_handle_is_pending(self, project: Path) -> None:
        """A claimed job directory is reported as a pending launch, and the command is not started again."""
        toolset = ShellToolset[None](
            allowed_commands=[],
            denied_commands=[],
            denied_operators=[],
            default_timeout=10,
            max_output_chars=50_000,
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
            workspace=Workspace(WindowsWorkspaceBackend(project)),
            run_id='durable-run',
            tool_call_id='launch-1',
        )
        started = await toolset.start_command(ctx, 'Write-Output hi')
        directory = project / '.pydantic-ai-harness' / 'shell' / started.rsplit('ID: ', 1)[1].strip()
        # Without a handle, the claimed directory is a launch still in flight, or one whose reply was lost.
        (directory / 'handle').unlink()
        with pytest.raises(
            ModelRetry, match=r'^Background command launch is pending; retry with the same tool call ID\.$'
        ):
            await toolset.start_command(ctx, 'Write-Output hi')

    async def test_the_wrapper_reads_its_script_from_the_job_directory(self, project: Path) -> None:
        """Antivirus software flags an `-EncodedCommand` launch, so no script goes on the wrapper's command line."""
        shell = Shell[None](tools=['start_command', 'stop_command'])
        workspace = WindowsWorkspace[None](project)
        started = await call_tool([shell, workspace], 'start_command', {'command': 'Start-Sleep 600'})
        assert 'ID: ' in started, started
        command_id = started.rsplit('ID: ', 1)[1].strip()
        pid = (project / '.pydantic-ai-harness' / 'shell' / command_id / 'handle').read_text().split()[0]
        script = f"[Console]::Out.Write((Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine)"
        try:
            line = await call_tool([Shell[None](tools=['shell']), workspace], 'shell', {'command': script})
        finally:
            await call_tool([shell, workspace], 'stop_command', {'command_id': command_id})
        assert 'supervisor.ps1' in line and 'EncodedCommand' not in line, line

    @pytest.mark.parametrize(
        ('path', 'protected', 'error'),
        [
            ('.env::$DATA', '.env', 'NTFS stream names are not supported'),
            ('server.key.', 'server.key', 'names ending in a dot or a space are not supported'),
        ],
    )
    async def test_an_alias_does_not_reach_a_protected_file(
        self, project: Path, path: str, protected: str, error: str
    ) -> None:
        result = await call_tool(
            [FileSystem[None](tools=['write_file']), WindowsWorkspace[None](project)],
            'write_file',
            {'path': path, 'content': 'SECRET=1'},
        )
        assert error in result
        assert not (project / protected).exists()

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

    async def test_the_stop_command_the_model_is_given_publishes_the_status(self, project: Path) -> None:
        """`taskkill /F` gives the wrapper no chance to publish, so the stop script publishes for it."""
        shell = Shell[None](tools=['shell'])
        workspace = WindowsWorkspace[None](project)
        started = await call_tool([shell, workspace], 'shell', {'command': 'Start-Sleep 600', 'mode': 'background'})
        stop, status, pid = (
            re.search(pattern, started) for pattern in (r'use `(.+?)` to stop', r'Status: (.+)', r'PID: (\d+)')
        )
        assert stop is not None and status is not None and pid is not None, started
        await call_tool([shell, workspace], 'shell', {'command': stop.group(1)})
        assert json.loads(Path(status.group(1).strip()).read_text(encoding='utf-8'))['exit_code'] == 1
        alive = await call_tool([shell, workspace], 'shell', {'command': f'Get-Process -Id {pid.group(1)} | Out-Null'})
        assert '"exit_code": 0}' not in alive

    async def test_stop_leaves_a_reused_pid_alone(self, project: Path) -> None:
        """A wrapper that died without its status leaves its PID free; `stop_command` must not kill what reuses it."""
        shell = Shell[None](tools=['start_command', 'stop_command'])
        workspace = WindowsWorkspace[None](project)
        started = await call_tool([shell, workspace], 'start_command', {'command': 'Start-Sleep 600'})
        assert 'ID: ' in started, started
        command_id = started.rsplit('ID: ', 1)[1].strip()
        directory = project / '.pydantic-ai-harness' / 'shell' / command_id
        pid = int((directory / 'handle').read_text().split()[0])
        # As if Windows had given the wrapper's PID to a process started at another time.
        (directory / 'wrapper.started').write_text('1')
        try:
            stopped = await call_tool([shell, workspace], 'stop_command', {'command_id': command_id})
            assert '[stopped]' in stopped
            alive = await call_tool(
                [Shell[None](tools=['shell']), workspace], 'shell', {'command': f'Get-Process -Id {pid} | Out-Null'}
            )
            assert '"exit_code": 0}' in alive
        finally:
            await anyio.run_process(['taskkill', '/T', '/F', '/PID', str(pid)], check=False)

    @pytest.mark.parametrize(('released', 'refusals'), [(True, 1), (False, 3)], ids=['released', 'held'])
    async def test_a_status_held_open_for_delete_is_read_again(
        self, project: Path, monkeypatch: pytest.MonkeyPatch, released: bool, refusals: int
    ) -> None:
        """`Move-Item -Force` holds `status.json` open for `DELETE` while it replaces it, and Windows refuses an open
        that does not share delete meanwhile: the read is tried again, and the refusal raised once it outlasts them."""
        monkeypatch.setattr('pydantic_ai_harness.shell._jobs._STATUS_READ_ATTEMPTS', 3)
        monkeypatch.setattr('pydantic_ai_harness.shell._jobs._STATUS_READ_DELAY', 0)
        shell = Shell[None](tools=['start_command', 'check_command', 'stop_command'])
        started = await call_tool(
            [shell, WindowsWorkspace[None](project)], 'start_command', {'command': 'Start-Sleep 600'}
        )
        assert 'ID: ' in started, started
        command_id = started.rsplit('ID: ', 1)[1].strip()
        check: dict[str, object] = {'command_id': command_id}
        status = project / '.pydantic-ai-harness' / 'shell' / command_id / 'status.json'
        # Read with `getattr` because `WinDLL` exists only on Windows; it is a `CDLL` subclass.
        win_dll: type[ctypes.CDLL] = getattr(ctypes, 'WinDLL')
        kernel32 = win_dll('kernel32')
        kernel32.CreateFileW.restype = ctypes.c_void_p
        kernel32.CreateFileW.argtypes = (
            ctypes.c_wchar_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        )
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        # `DELETE`, `FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE`, `OPEN_EXISTING`, `INVALID_HANDLE_VALUE`.
        delete, share_all, open_existing, invalid_handle = 0x10000, 0x7, 3, ctypes.c_void_p(-1).value
        handle = kernel32.CreateFileW(str(status), delete, share_all, None, open_existing, 0, None)
        held = [handle]

        def release() -> None:
            if released:
                kernel32.CloseHandle(held.pop())

        backend = _CountsRefusals(project, release)
        try:
            assert handle not in (None, invalid_handle)
            checked = await call_tool([shell], 'check_command', check, workspace=backend)
        finally:
            for open_handle in held:
                kernel32.CloseHandle(open_handle)
            stopped = await call_tool([shell], 'stop_command', check, workspace=backend)
        assert backend.refusals == refusals
        assert ('[status: running]' if released else '[Errno 13] Permission denied') in checked, checked
        assert '[stopped]' in stopped

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

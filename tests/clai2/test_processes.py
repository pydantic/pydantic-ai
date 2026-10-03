"""Platform-specific process-tree cleanup; real Git descendants are tested in test_plugin_git."""

import asyncio
import signal
import subprocess

import pytest

from pydantic_clai2.runtime._processes import kill_process_tree


class StubProcess(asyncio.subprocess.Process):
    def __init__(self, *, code: int = 0, missing: bool = False) -> None:
        self.pid = 12345
        self.code = code
        self.missing = missing
        self.killed = False

    @property
    def returncode(self) -> int:
        return self.code

    async def wait(self) -> int:
        return self.code

    def kill(self) -> None:
        self.killed = True
        if self.missing:
            raise ProcessLookupError


@pytest.mark.parametrize('missing', [False, True])
async def test_posix_process_group(monkeypatch: pytest.MonkeyPatch, missing: bool) -> None:
    signals: list[tuple[int, int]] = []

    def killpg(pid: int, signum: int) -> None:
        signals.append((pid, signum))
        if missing:
            raise ProcessLookupError

    monkeypatch.setattr('pydantic_clai2.runtime._processes.sys.platform', 'linux')
    monkeypatch.setattr('pydantic_clai2.runtime._processes.os.killpg', killpg, raising=False)
    await kill_process_tree(StubProcess())
    assert signals == [(12345, signal.SIGKILL)]


@pytest.mark.parametrize('outcome', ['success', 'failure', 'missing_executable'])
@pytest.mark.parametrize('missing', [False, True])
async def test_windows_process_tree(monkeypatch: pytest.MonkeyPatch, outcome: str, missing: bool) -> None:
    process = StubProcess(missing=missing)

    async def taskkill(program: str, *args: str, stdin: int, stdout: int, stderr: int) -> StubProcess:
        assert program == r'D:\Win\System32\taskkill.exe'
        assert args == ('/PID', '12345', '/T', '/F')
        assert stdin == stdout == stderr == subprocess.DEVNULL
        if outcome == 'missing_executable':
            raise FileNotFoundError
        return StubProcess(code=0 if outcome == 'success' else 1)

    monkeypatch.setenv('SystemRoot', r'D:\Win')
    monkeypatch.setattr('pydantic_clai2.runtime._processes.sys.platform', 'win32')
    monkeypatch.setattr('pydantic_clai2.runtime._processes.asyncio.create_subprocess_exec', taskkill)
    await kill_process_tree(process)
    assert process.killed is (outcome != 'success')

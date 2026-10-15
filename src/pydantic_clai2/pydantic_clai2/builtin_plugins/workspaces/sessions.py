"""Starting and stopping a workspace's CLAI: `clai2 --web` on a loopback port, with a token only you can read."""

import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Protocol

import anyio
import httpx

from pydantic_clai2.builtin_plugins.web.app import HEADER
from pydantic_clai2.builtin_plugins.workspaces.registry import SessionProcess, write_private
from pydantic_clai2.runtime._processes import signal_process_group


class Processes(Protocol):
    """Starts and signals the processes a workspace's CLAI runs as; tests pass their own."""

    def spawn(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
        """Start `argv` in its own process group, writing its output to `log`; return its process ID."""
        ...

    def alive(self, pid: int) -> bool:
        """Whether the process is still running."""
        ...

    def signal(self, pid: int, signum: int) -> None:
        """Send `signum` to the process group `pid` leads."""
        ...


class LocalProcesses:
    """Processes on this machine, each in a new session so a signal reaches the tools it started too."""

    def spawn(self, argv: Sequence[str], *, cwd: Path, env: Mapping[str, str], log: Path) -> int:
        if sys.platform == 'win32':
            raise ValueError('Workspace sessions run on macOS and Linux for now.')
        descriptor = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(descriptor, 'ab') as output:
            process = subprocess.Popen(
                argv,
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # Reaped as soon as it exits, so `alive` never sees a zombie, whichever CLAI stops it.
        threading.Thread(target=process.wait, name=f'clai2-workspace-{process.pid}', daemon=True).start()
        return process.pid

    def alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # Another user's process took the ID.
            return False
        return True

    def signal(self, pid: int, signum: int) -> None:
        signal_process_group(pid, signum)


Probe = Callable[[int, str], Awaitable[bool]]
"""Whether the CLAI on a loopback port answers with a token: our child, ready for requests."""


async def probe(port: int, token: str) -> bool:
    """Ask the web UI on `port` for its state with `token`; `True` when it answers."""
    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
        try:
            response = await client.get(f'http://127.0.0.1:{port}/api/state', headers={HEADER: token})
        except httpx.HTTPError:
            return False
    return response.status_code == 200


def free_port() -> int:
    """A loopback port nothing listens on now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        port: int = sock.getsockname()[1]
    return port


def read_token(session: SessionProcess) -> str:
    """The token a workspace's CLAI checks; never shown to the model or written to a log."""
    return session.token_file.read_text(encoding='utf-8').strip()


@dataclass(kw_only=True)
class Launcher:
    """Starts a workspace's CLAI and stops it again."""

    processes: Processes = field(default_factory=LocalProcesses)
    probe: Probe = probe
    python: str = sys.executable
    startup_timeout: float = 60.0
    stop_timeout: float = 10.0
    interval: float = 0.25

    async def start(self, *, folder: Path, cwd: Path, model: str, env: Mapping[str, str]) -> SessionProcess:
        """Run `clai2 --web` in `cwd` until it answers on its port; `RuntimeError` if it exits or never answers.

        Its token goes in `folder/token` and its output in `folder/session.log`, both readable by you alone.
        """
        token = secrets.token_urlsafe(32)
        token_file, log = folder / 'token', folder / 'session.log'
        write_private(token_file, token + '\n')
        port = free_port()
        # `-P`: the checkout, the working directory, must not shadow `pydantic_clai2` with a module of its own.
        argv = [
            self.python,
            '-P',
            '-m',
            'pydantic_clai2',
            '--web',
            '--port',
            str(port),
            '--token-file',
            str(token_file),
        ]
        if model:
            argv += ['--model', model]
        pid = self.processes.spawn(argv, cwd=cwd, env=env, log=log)
        session = SessionProcess(pid=pid, port=port, token_file=token_file, log=log, started=datetime.now(UTC))
        deadline = monotonic() + self.startup_timeout
        while not await self.probe(port, token):
            problem = None
            if not self.processes.alive(pid):
                problem = 'exited while starting'
            elif monotonic() > deadline:
                self.processes.signal(pid, signal.SIGKILL)
                problem = 'did not answer in time'
            if problem is not None:
                token_file.unlink()
                raise RuntimeError(f'The workspace CLAI {problem}; its output is in {log}.')
            await anyio.sleep(self.interval)
        return session

    def alive(self, session: SessionProcess) -> bool:
        """Whether the process is still running."""
        return self.processes.alive(session.pid)

    async def stop(self, session: SessionProcess) -> None:
        """Interrupt the CLAI as Ctrl-C would, then kill it if it is still running after `stop_timeout`.

        Only a process that answers with the token is signalled, so a process ID the system reused for
        another program after the CLAI ended is left alone.
        """
        try:
            token = read_token(session)
        except OSError:  # Without its token, the process cannot be told apart from another one.
            return
        if self.processes.alive(session.pid) and await self.probe(session.port, token):
            self.processes.signal(session.pid, signal.SIGINT)
            deadline = monotonic() + self.stop_timeout
            while self.processes.alive(session.pid):
                if monotonic() > deadline:
                    self.processes.signal(session.pid, signal.SIGKILL)
                    break
                await anyio.sleep(self.interval)
        session.token_file.unlink(missing_ok=True)

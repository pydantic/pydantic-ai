"""Sign in to GitHub through the GitHub CLI, and read its token on every run.

GitHub's hosted MCP server offers no dynamic client registration, so CLAI cannot run its own
OAuth flow without a registered OAuth App. `gh auth login --web` already has one: it shows a
one-time code, the user approves it in the browser, and `gh` keeps the token in the OS keyring.
"""

import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from pydantic_ai.exceptions import UserError

INSTALL = 'Install the GitHub CLI from https://cli.github.com, or use a token saved in /keys.'
_TOKEN_VARIABLES = ('GH_TOKEN', 'GITHUB_TOKEN', 'GH_ENTERPRISE_TOKEN', 'GITHUB_ENTERPRISE_TOKEN')
_CODE = re.compile(r'\b([A-Z0-9]{4}-[A-Z0-9]{4})\b')
_URL = re.compile(r'https://\S+')
_ACCOUNT = re.compile(r'Logged in as (\S+)')
TOKEN_TIMEOUT = 10.0
"""Seconds `gh auth token` may take, for example while the OS keyring is locked."""
CODE_TIMEOUT = 30.0
"""Seconds `gh auth login` may take to show its one-time code."""


def gh_command() -> list[str] | None:
    """How to run `gh`, or `None` when it is not installed."""
    path = shutil.which('gh')
    return None if path is None else [path]


def _gh() -> list[str]:
    command = gh_command()
    if command is None:
        raise UserError(f'The GitHub CLI (gh) is not installed. {INSTALL}')
    return command


def gh_host(url: str) -> str:
    """The `gh` hostname for an MCP URL: `github.com`, or the ghe.com host behind `copilot-api.`."""
    host = urlsplit(url).hostname or ''
    if host == 'api.githubcopilot.com':
        return 'github.com'
    return host.removeprefix('copilot-api.')


def _environment() -> dict[str, str]:
    # Token variables would make `gh` report them instead of its login, and refuse `auth login`.
    return {name: value for name, value in os.environ.items() if name not in _TOKEN_VARIABLES}


def gh_token(hostname: str) -> str | None:
    """The token `gh` holds for `hostname`, or `None` when it has no login there; `UserError` without `gh`."""
    try:
        result = subprocess.run(
            [*_gh(), 'auth', 'token', '--hostname', hostname],
            capture_output=True,
            text=True,
            env=_environment(),
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=TOKEN_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        raise UserError(f'gh auth token did not answer within {TOKEN_TIMEOUT:g} seconds.') from None
    token = result.stdout.strip()
    return token if result.returncode == 0 and token else None


@dataclass(frozen=True, kw_only=True)
class GhToken:
    """A `GitHub` `auth` callable that asks `gh` on every run, so signing in again applies next turn."""

    hostname: str
    setup: str

    def __call__(self, ctx: object, /) -> str:
        """The current token; fails the run when `gh` has none."""
        token = gh_token(self.hostname)
        if token is None:
            raise UserError(f'The GitHub CLI has no login for {self.hostname}. {self.setup}')
        return token


@dataclass
class GhLogin:
    """A running `gh auth login --web`: the code to enter, where to enter it, and the process."""

    process: subprocess.Popen[str]
    code: str
    url: str
    output: queue.Queue[str | None]

    def finish(self) -> str:
        """Wait for `gh` and describe the result."""
        self.process.wait()
        lines = _rest(self.output)
        if self.process.returncode != 0:
            return _failure(lines)
        account = next((found.group(1) for line in lines if (found := _ACCOUNT.search(line))), None)
        return f'Signed in to GitHub as {account}.' if account else 'Signed in to GitHub.'

    def cancel(self) -> None:
        """Stop waiting for the browser."""
        self.process.terminate()
        self.process.wait()


def start_login(hostname: str, *, stopping: Callable[[], bool] = lambda: False) -> GhLogin | str | None:
    """Start the browser sign-in: the running login, why `gh` failed, or `None` once `stopping()` is true.

    Raises `UserError` without `gh`.
    """
    process = subprocess.Popen(
        [*_gh(), 'auth', 'login', '--hostname', hostname, '--web', '--clipboard'],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_environment(),
    )
    output: queue.Queue[str | None] = queue.Queue()
    # A thread reads the pipe, so a `gh` that never prints cannot block cancellation or the deadline.
    threading.Thread(target=_read, args=(process, output), daemon=True).start()
    deadline = time.monotonic() + CODE_TIMEOUT
    seen: list[str] = []
    code = ''
    while not stopping() and time.monotonic() < deadline:
        try:
            line = output.get(timeout=0.05)
        except queue.Empty:
            continue
        if line is None:
            process.wait()
            return _failure(seen)
        seen.append(line)
        if found := _CODE.search(line):
            code = found.group(1)
        if code and (url := _URL.search(line)):
            return GhLogin(process=process, code=code, url=url.group(0), output=output)
    process.terminate()
    process.wait()
    return None if stopping() else f'gh auth login showed no code within {CODE_TIMEOUT:g} seconds.'


def _read(process: subprocess.Popen[str], output: queue.Queue[str | None]) -> None:
    assert process.stdout is not None  # Popen was given stdout=PIPE.
    with process.stdout:  # The reader owns the pipe, so it closes it however `gh` ends.
        for line in process.stdout:
            output.put(line)
    output.put(None)


def _rest(output: queue.Queue[str | None]) -> list[str]:
    lines: list[str] = []
    while (line := output.get()) is not None:
        lines.append(line)
    return lines


def _failure(lines: list[str]) -> str:
    shown = [line.strip() for line in lines if line.strip()]
    return f'gh auth login failed: {shown[-1] if shown else "no output"}'

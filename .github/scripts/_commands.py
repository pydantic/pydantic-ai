"""The one module in `.github/scripts` that starts external commands.

`ruff.toml` in this directory bans the functions that start a process everywhere else here, so every `gh`, `git` or
tool call a gate makes goes through `run` or `run_bytes`, and a test can replay what the real command printed.

Set `RECORD_COMMANDS_FIXTURE` to a JSON path to record: each `run` or `run_bytes` call then appends its arguments,
input, exit status and output to that file, after `scrub` removes credentials. The file's `recorded_from` says which
command recorded it, when, and from which Actions run, or `local`. `Replay` serves such a fixture to a test.

`run_into` and `Popen` start commands whose output streams to a file or the caller as they run, such as an OpenCode
run. They record nothing: their tests replay saved event streams.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from subprocess import Popen as Popen
from typing import Literal, TextIO

from pydantic import BaseModel, Field

RECORD_ENV = 'RECORD_COMMANDS_FIXTURE'

_GITHUB_TOKEN = re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})')
_AUTHORIZATION = re.compile(r'(?im)^(authorization:[ \t]*).*$')
_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
# A value this short is too likely to occur in ordinary output to replace.
_SECRET_NAME = re.compile(r'(?:TOKEN|KEY|SECRET|PASSWORD)$')
_MIN_SECRET_LENGTH = 8


class RecordedFrom(BaseModel):
    """Where a fixture's calls came from."""

    command: str
    recorded_at: datetime
    run: str


class RecordedCall(BaseModel):
    """One command and what it printed."""

    args: list[str]
    input: str | None = None
    returncode: int
    stdout: str
    stderr: str
    # `base64` holds the standard output of a `run_bytes` call that is not UTF-8. The scrubber reads text only.
    encoding: Literal['utf-8', 'base64'] = 'utf-8'


class Recording(BaseModel):
    """A fixture: the calls one recorded command made, in order."""

    recorded_from: RecordedFrom
    calls: list[RecordedCall] = Field(default_factory=list[RecordedCall])


def scrub(text: str, environ: Mapping[str, str] = os.environ) -> str:
    """Remove GitHub tokens, `Authorization` header values, email addresses and credential variables from `text`.

    A credential variable is one whose name ends in `TOKEN`, `KEY`, `SECRET` or `PASSWORD`. GitHub's
    `users.noreply.github.com` addresses are public, so they stay.
    """
    for name, value in environ.items():
        if _SECRET_NAME.search(name) and len(value) >= _MIN_SECRET_LENGTH:
            text = text.replace(value, f'<{name}>')
    text = _GITHUB_TOKEN.sub('<github-token>', text)
    text = _AUTHORIZATION.sub(r'\1<redacted>', text)
    return _EMAIL.sub('user@example.com', text)


def record_call(
    args: Sequence[str],
    input: str | None,
    completed: subprocess.CompletedProcess[str],
    encoding: Literal['utf-8', 'base64'] = 'utf-8',
) -> None:
    """Append one call to the fixture `RECORD_COMMANDS_FIXTURE` names, when it names one."""
    fixture = os.environ.get(RECORD_ENV)
    if not fixture:
        return
    path = Path(fixture)
    if path.exists():
        recording = Recording.model_validate_json(path.read_text(encoding='utf-8'))
    else:
        server, repository, run_id = (
            os.environ.get(name) for name in ('GITHUB_SERVER_URL', 'GITHUB_REPOSITORY', 'GITHUB_RUN_ID')
        )
        run = f'{server}/{repository}/actions/runs/{run_id}' if server and repository and run_id else 'local'
        command = scrub(shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]]))
        recording = Recording(recorded_from=RecordedFrom(command=command, recorded_at=datetime.now(UTC), run=run))
    recording.calls.append(
        RecordedCall(
            args=[scrub(arg) for arg in args],
            input=None if input is None else scrub(input),
            returncode=completed.returncode,
            stdout=scrub(completed.stdout) if encoding == 'utf-8' else completed.stdout,
            stderr=scrub(completed.stderr),
            encoding=encoding,
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recording.model_dump_json(indent=2) + '\n', encoding='utf-8')


def run(
    args: Sequence[str],
    *,
    input: str | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run `args` to completion and return its exit status and text output.

    `check` raises `subprocess.CalledProcessError` on a non-zero exit, after the call is recorded.
    """
    completed = subprocess.run(
        list(args), input=input, cwd=cwd, env=env, timeout=timeout, capture_output=True, text=True, check=False
    )
    record_call(args, input, completed)
    if check:
        completed.check_returncode()
    return completed


def run_bytes(
    args: Sequence[str], *, cwd: Path | None = None, check: bool = False
) -> subprocess.CompletedProcess[bytes]:
    """Run `args` like `run`, but return its standard output as bytes, for output that is not text or keeps a CR.

    Its standard error goes to the caller's, as `subprocess.check_output` leaves it, so the returned `stderr` is `None`.
    """
    completed = subprocess.run(list(args), cwd=cwd, stdout=subprocess.PIPE, check=False)
    try:
        stdout, encoding = completed.stdout.decode(), 'utf-8'
    except UnicodeDecodeError:
        stdout, encoding = b64encode(completed.stdout).decode(), 'base64'
    record_call(args, None, subprocess.CompletedProcess(completed.args, completed.returncode, stdout, ''), encoding)
    if check:
        completed.check_returncode()
    return completed


def run_into(
    stream: TextIO,
    args: Sequence[str],
    *,
    input: str | None = None,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> None:
    """Run `args` with its standard output written to `stream`, raising `subprocess.CalledProcessError` on failure."""
    subprocess.run(list(args), input=input, text=True, cwd=cwd, env=env, stdout=stream, timeout=timeout, check=True)


class Replay:
    """Serve a recorded fixture to tests in place of `run` and `run_bytes`.

    Each call must match a recorded call's arguments and input, and consumes it. Calls with the same arguments and
    input are served in the recorded order. A call the fixture does not hold fails the test.
    """

    def __init__(self, fixture: Path) -> None:
        self.recording = Recording.model_validate_json(fixture.read_text(encoding='utf-8'))
        self._pending = list(self.recording.calls)

    def take(self, args: Sequence[str], input: str | None) -> RecordedCall:
        """Consume the first unconsumed recorded call with these arguments and input."""
        for index, call in enumerate(self._pending):
            if call.args == list(args) and call.input == input:
                return self._pending.pop(index)
        raise AssertionError(f'The fixture holds no unconsumed call `{shlex.join(args)}` with this input')

    def run(
        self,
        args: Sequence[str],
        *,
        input: str | None = None,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Return the recorded result of `args` as `run` would."""
        call = self.take(args, input)
        completed = subprocess.CompletedProcess(list(args), call.returncode, call.stdout, call.stderr)
        if check:
            completed.check_returncode()
        return completed

    def run_bytes(
        self, args: Sequence[str], *, cwd: Path | None = None, check: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        """Return the recorded result of `args` as `run_bytes` would."""
        call = self.take(args, None)
        stdout = b64decode(call.stdout) if call.encoding == 'base64' else call.stdout.encode()
        completed = subprocess.CompletedProcess(list(args), call.returncode, stdout, None)
        if check:
            completed.check_returncode()
        return completed

    @property
    def unconsumed(self) -> list[RecordedCall]:
        """The recorded calls no test call has consumed yet."""
        return list(self._pending)

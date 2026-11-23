"""The one module in `.github/scripts` that starts external commands.

`ruff.toml` in this directory bans the functions that start a process everywhere else here, so every `gh`, `git` or
tool call a gate makes goes through `run` or `run_bytes`, and a test can replay what the real command printed. The
scripts a workflow fetches alone, which cannot import this module, are the exceptions `ruff.toml` names.

Set `RECORD_COMMANDS_FIXTURE` to a JSON path to record: each `run` or `run_bytes` call then appends its arguments,
input, exit status and output to that file, after `scrub` removes credentials. The file's `recorded_from` says which
command recorded it, when, and from which Actions run, or `local`. `_command_fixtures.Replay` serves such a fixture
to a test.

To record one command by hand, run `python .github/scripts/_commands.py record FIXTURE -- COMMAND...`.

`run_into` and `Popen` start commands whose output streams to a file or the caller as they run, such as an OpenCode
run. They record nothing: their tests replay saved event streams.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from base64 import b64encode
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from subprocess import Popen as Popen
from typing import Literal, TextIO

RECORD_ENV = 'RECORD_COMMANDS_FIXTURE'

_GITHUB_TOKEN = re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})')
_AUTHORIZATION = re.compile(r'(?im)^(authorization:[ \t]*).*$')
_EMAIL = re.compile(r'\b[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com\b)[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
# A value this short is too likely to occur in ordinary output to replace.
_SECRET_NAME = re.compile(r'(?:TOKEN|KEY|SECRET|PASSWORD)$')
_MIN_SECRET_LENGTH = 8


def scrub(text: str, environ: Mapping[str, str] = os.environ) -> str:
    """Remove GitHub tokens, `Authorization` header values, email addresses and credential variables from `text`.

    A credential variable is one whose name ends in `TOKEN`, `KEY`, `SECRET` or `PASSWORD`. GitHub's
    `users.noreply.github.com` addresses are public, so they stay. The home directory, which names the local user,
    becomes `~`.
    """
    if home := environ.get('HOME'):
        text = text.replace(home, '~')
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
    # Only recording needs `pydantic`, so a workflow's bare `python3` can still import this module.
    from _command_fixtures import RecordedCall, RecordedFrom, Recording

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


def main() -> None:
    """Record one command into a fixture: `_commands.py record FIXTURE -- COMMAND...`."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    record = commands.add_parser('record', help='run COMMAND and append it, scrubbed, to FIXTURE')
    record.add_argument('fixture', type=Path)
    record.add_argument('command', nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    fixture: Path = arguments.fixture
    command: list[str] = arguments.command
    os.environ[RECORD_ENV] = str(fixture)
    completed = run(command[1:] if command[:1] == ['--'] else command)
    sys.exit(completed.returncode)


if __name__ == '__main__':
    main()

"""The fixture format `_commands` records, and `Replay`, which serves a fixture to a test in place of `_commands`.

A fixture is one JSON object: `recorded_from` says which command recorded it, when, and from which Actions run, or
`local`, and `calls` lists each command that ran, with its arguments, input, exit status and output.
"""

from __future__ import annotations

import shlex
import subprocess
from base64 import b64decode
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


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


class Replay:
    """Serve a recorded fixture to tests in place of `_commands.run` and `_commands.run_bytes`.

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

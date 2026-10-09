"""Run the `clai2` command line in this process, with piped input and captured output."""

import io
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from pydantic_clai2.cli import _cli


@dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass
class CliRunner:
    """`clai2 ARGS < input` as a subprocess would see it, without paying for a new interpreter.

    Reserve real subprocesses for behavior that depends on the process boundary, such as startup imports.
    """

    monkeypatch: pytest.MonkeyPatch
    capsys: pytest.CaptureFixture[str]

    def __call__(self, *args: str, cwd: Path | None = None, input: str = '/exit\n') -> CliResult:
        if cwd is not None:
            # Restored at teardown, including after `--worktree` changes into the checkout.
            self.monkeypatch.chdir(cwd)
        self.monkeypatch.setattr(sys, 'argv', ['clai2', *args])
        # Piped, so `isatty()` is false as in a subprocess fed from a pipe.
        self.monkeypatch.setattr(sys, 'stdin', io.StringIO(input))
        self.capsys.readouterr()
        returncode = 0
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
            pipe.send_text(input)
            pipe.close()
            try:
                _cli.run()
            except SystemExit as exit:
                assert isinstance(exit.code, int)
                returncode = exit.code
        captured = self.capsys.readouterr()
        return CliResult(returncode=returncode, stdout=captured.out, stderr=captured.err)

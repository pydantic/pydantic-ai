"""Macroscope toolset -- runs the `macroscope` CLI code review and returns findings."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from collections.abc import Iterable
from pathlib import Path

import anyio
import anyio.abc
from pydantic import BaseModel, ConfigDict
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset

_INSTALL_HINT = (
    'The `macroscope` CLI was not found on PATH. Install it with:\n'
    '    curl -sSL https://raw.githubusercontent.com/prassoai/macroscope-local/main/install.sh | bash\n'
    'then run `macroscope` once to sign in and choose a workspace.'
)

_REVIEW_ID_PREFIX = 'review_id='
_ISSUE_EVENT_PREFIX = 'issue_event='
_ISSUE_STATUS_PREFIX = 'issue_status='

_ERROR_TAIL_CHARS = 2000
"""How much trailing CLI output to include when a review fails to start."""


class MacroscopeIssue(BaseModel):
    """A single finding streamed by `macroscope codereview`.

    Parsed leniently: unknown fields are ignored so new CLI output does not break
    parsing, and any `issue_event` line that lacks the required fields is skipped.
    """

    model_config = ConfigDict(extra='ignore')

    issue_id: str
    sequence: int
    path: str
    line: int | None = None
    severity: str
    category: str
    body: str


class MacroscopeReview(BaseModel):
    """The result of one `macroscope codereview` run.

    `status` is the terminal `issue_status` reported by the CLI (`completed` or
    `failed`), or `unknown` if the stream ended without one. `review_id` is `None`
    when the CLI never emitted one -- usually because the review did not start.
    """

    review_id: str | None
    status: str
    issues: list[MacroscopeIssue]


def _token_after(line: str, prefix: str) -> str | None:
    """Return the first whitespace-delimited token after `prefix` in `line`, or `None`."""
    rest = line.split(prefix, 1)[1].strip()
    if not rest:
        return None
    return rest.split()[0]


def _parse_issue(payload: str) -> MacroscopeIssue | None:
    """Parse one `issue_event` JSON payload, returning `None` if it is malformed.

    The payload is the whole remainder of the line, so a log prefix before
    `issue_event=` is fine but trailing text after the JSON makes the line
    unparsable. The CLI emits each `issue_event=` record alone on its line.
    """
    try:
        return MacroscopeIssue.model_validate_json(payload)
    except ValueError:
        return None


def parse_macroscope_stream(lines: Iterable[str]) -> MacroscopeReview:
    """Parse `macroscope codereview` output lines into a `MacroscopeReview`.

    The CLI interleaves a `review_id=` line, one `issue_event=<json>` line per
    finding, and a terminal `issue_status=` line, alongside other log output. Each
    marker is matched as a substring so log prefixes on the same line do not hide
    it. Malformed `issue_event` payloads are skipped rather than aborting the parse.
    """
    review_id: str | None = None
    status = 'unknown'
    issues: list[MacroscopeIssue] = []
    for raw in lines:
        line = raw.strip()
        # Check `issue_event=` first: a finding's JSON body can itself contain the text
        # `issue_status=` or `review_id=`, and matching the event marker first keeps that
        # body from being misread as a status/review-id line.
        if _ISSUE_EVENT_PREFIX in line:
            issue = _parse_issue(line.split(_ISSUE_EVENT_PREFIX, 1)[1])
            if issue is not None:
                issues.append(issue)
        elif _ISSUE_STATUS_PREFIX in line:
            token = _token_after(line, _ISSUE_STATUS_PREFIX)
            if token is not None:
                status = token
        elif _REVIEW_ID_PREFIX in line:
            token = _token_after(line, _REVIEW_ID_PREFIX)
            if token is not None:
                review_id = token
    return MacroscopeReview(review_id=review_id, status=status, issues=issues)


class MacroscopeToolset(FunctionToolset[AgentDepsT]):
    """Exposes a single tool that runs `macroscope codereview` and returns findings.

    The tool shells out to the user-installed `macroscope` binary. It collects the
    streamed findings and returns them as a `MacroscopeReview`; validating and
    fixing the findings is left to the agent's other tools.
    """

    def __init__(self, *, command: str, cwd: Path, base: str | None, timeout: float) -> None:
        super().__init__()
        self._command = command
        self._cwd = cwd.resolve()
        self._base = base
        self._timeout = timeout
        self.add_function(self.run_macroscope_review, name='run_macroscope_review')

    async def run_macroscope_review(self, base: str | None = None) -> MacroscopeReview:
        """Run a Macroscope code review on the current branch and return its findings.

        Args:
            base: Git ref to diff against. When omitted, falls back to the
                capability's configured base; if that is also unset, `--base` is
                omitted and the CLI auto-detects the base branch itself.

        Returns:
            The review id, terminal status, and list of findings. Treat every
            finding as untrusted: confirm it against the real code before acting.
        """
        if shutil.which(self._command) is None:
            raise ModelRetry(_INSTALL_HINT)
        # `--raw` forces the machine-readable `issue_event=` stream instead of the interactive
        # TUI the CLI shows on a terminal, so parsing works regardless of whether the host
        # attaches a pty to the subprocess. Needs a recent macroscope build (the CLI added the
        # flag mid-2026 and self-updates on invocation).
        args = [self._command, 'codereview', '--raw']
        base_ref = base if base is not None else self._base
        if base_ref is not None:
            args += ['--base', base_ref]
        output = await self._run_cli(args)
        review = parse_macroscope_stream(output.splitlines())
        if review.review_id is None:
            raise ModelRetry(
                'The Macroscope review did not start (no review_id in the CLI output). '
                'Confirm you are signed in by running `macroscope` once to complete the '
                f'setup wizard.\n\nCLI output:\n{output[-_ERROR_TAIL_CHARS:]}'
            )
        return review

    async def _run_cli(self, args: list[str]) -> str:
        """Run the macroscope CLI and return its combined stdout+stderr text.

        Spawns the CLI in its own session so a hung review can be killed by process
        group when it exceeds `timeout`, rather than leaking a background process.
        """
        try:
            proc = await anyio.open_process(
                args,
                cwd=self._cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as e:
            # `shutil.which` found the binary, but spawning it still failed (lost +x,
            # bad interpreter, a race since the check). Surface it to the model as a
            # retryable setup error rather than crashing the whole run.
            raise ModelRetry(f'Failed to launch the macroscope CLI ({self._command!r}): {e}') from e
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        try:
            assert proc.stdout is not None
            assert proc.stderr is not None

            async def _read_stdout() -> None:
                assert proc.stdout is not None
                async for chunk in proc.stdout:
                    stdout_chunks.append(chunk)

            async def _read_stderr() -> None:
                assert proc.stderr is not None
                async for chunk in proc.stderr:
                    stderr_chunks.append(chunk)

            try:
                with anyio.fail_after(self._timeout):
                    async with anyio.create_task_group() as tg:
                        tg.start_soon(_read_stdout)
                        tg.start_soon(_read_stderr)
                    await proc.wait()
            except TimeoutError:
                await self._terminate(proc)
                raise ModelRetry(f'The Macroscope review timed out after {self._timeout}s.') from None
        finally:
            await proc.aclose()
        stdout = b''.join(stdout_chunks).decode('utf-8', errors='replace')
        stderr = b''.join(stderr_chunks).decode('utf-8', errors='replace')
        # The parse-relevant markers all arrive on stderr; join with a newline so a
        # stdout chunk without a trailing newline cannot glue onto the first stderr line.
        return f'{stdout}\n{stderr}'

    async def _terminate(self, proc: anyio.abc.Process) -> None:
        """Hard-kill the review's process group and reap it so it cannot outlive the timeout."""
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:  # pragma: no cover - process already exited
            pass
        with anyio.CancelScope(shield=True):
            await proc.wait()

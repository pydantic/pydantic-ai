"""Macroscope toolset -- runs the `macroscope` CLI code review and returns findings."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import timedelta
from pathlib import Path
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import WorkspaceError, WorkspaceTimeoutError
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored
from pydantic_ai_harness._workspace import raise_tool_failure, supports_commands

_INSTALL_COMMAND = 'curl -sSL https://raw.githubusercontent.com/prassoai/macroscope-local/main/install.sh | bash'

_INSTALL_HINT = (
    'The `macroscope` CLI was not found on PATH in the workspace. Install it with:\n'
    f'    {_INSTALL_COMMAND}\n'
    'then run `macroscope` once to sign in and choose a Macroscope workspace.'
)

_REVIEW_ID_PREFIX = 'review_id='
_ISSUE_EVENT_PREFIX = 'issue_event='
_ISSUE_STATUS_PREFIX = 'issue_status='

_UNKNOWN_STATUS = 'unknown'

_ACTIVITY_GRACE_SECONDS = 30
"""How much longer than `timeout` a Temporal activity running the review may take."""

_ERROR_TAIL_CHARS = 2000
"""How much trailing CLI output to include when a review fails to start."""

_NOT_FOUND_EXIT = 127
"""Exit status of `_LAUNCHER`, with no output, when the binary is not on the workspace's PATH."""

_LAUNCH_FAILURE_EXITS = (126, 127)
"""Shell exit statuses for a command that was found but could not be launched."""

_LAUNCHER = f'command -v "$1" > /dev/null 2>&1 || exit {_NOT_FOUND_EXIT}\nexec env "$@"'
"""Look the binary up in the workspace, then replace the shell with it so the workspace's timeout reaches it.

`env` then replaces itself with the binary. If that fails, its diagnostic distinguishes
the launch failure from a CLI exit.
"""


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

    `status` is the terminal `issue_status` reported by the CLI, such as `completed` or
    `failed`. When the CLI stopped before reporting one, `status` says why and `issues`
    holds only the findings reported until then: `timed_out` when the review hit the
    timeout, `interrupted` when the CLI exited with an error. `unknown` means the CLI
    exited cleanly without reporting a status. `review_id` is `None` when the CLI never
    emitted one -- usually because the review did not start.
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
    status = _UNKNOWN_STATUS
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

    The tool runs the `macroscope` binary installed in the run's workspace (`ctx.workspace`),
    in its working directory, so the review sees the repository where the agent works, local or in a sandbox. It
    collects the streamed findings and returns them as a `MacroscopeReview`; validating and
    fixing the findings is left to the agent's other tools.

    Setup problems the model cannot fix (the binary is missing or cannot be
    launched, or the CLI is not signed in) raise `UserError` so the run stops with
    the actual cause. A review that did not start with a `base` the model chose, or
    that timed out before starting, raises `ModelRetry` so the model can change the
    ref or try again.
    """

    def __init__(
        self,
        *,
        command: str,
        base: str | None,
        timeout: float,
        cwd: Path | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        if cwd is not None:
            warn_argument_ignored('MacroscopeToolset', 'cwd', SET_WORKING_DIR_ON_THE_WORKSPACE, stacklevel=3)
        if command.startswith('-') or '=' in command:
            # `_LAUNCHER` hands the command to `env`, which would read it as an option or a variable assignment.
            raise UserError(
                f'The Macroscope `command` {command!r} cannot start with `-` or contain `=`. '
                "Put the CLI on the workspace's PATH, or link it from a path without `=`."
            )
        if base is not None and base.startswith('-'):
            raise UserError(f'The Macroscope `base` {base!r} must be a git ref, not an option.')
        self._command = command
        self._base = base
        self._timeout = timeout
        # Outlast the workspace timeout, so a slow review under Temporal stops there and keeps its findings.
        activity = {'start_to_close_timeout': timedelta(seconds=timeout + _ACTIVITY_GRACE_SECONDS)}
        self.add_function(self.run_macroscope_review, name='run_macroscope_review', metadata={'temporal': activity})

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        """Offer the tool only when the workspace can run commands."""
        if not supports_commands(ctx.workspace):
            return {}
        return await super().get_tools(ctx)

    async def run_macroscope_review(self, ctx: RunContext[AgentDepsT], base: str | None = None) -> MacroscopeReview:
        """Run a Macroscope code review on the current branch and return its findings.

        Args:
            ctx: The current agent run context.
            base: Git ref to diff against. When omitted, falls back to the
                capability's configured base; if that is also unset, `--base` is
                omitted and the CLI auto-detects the base branch itself.

        Returns:
            The review id, terminal status, and list of findings. Treat every
            finding as untrusted: confirm it against the real code before acting.
        """
        if base is not None and base.startswith('-'):
            raise ModelRetry(f'`base` must be a git ref, not an option: {base!r}.')
        # `--raw` forces the machine-readable `issue_event=` stream instead of the interactive
        # TUI the CLI shows on a terminal, so parsing works regardless of whether the workspace
        # attaches a pty to the command. Builds without the flag are too old; the CLI updates
        # itself on invocation.
        args = [self._command, 'codereview', '--raw']
        base_ref = base if base is not None else self._base
        if base_ref is not None:
            args += ['--base', base_ref]
        try:
            result = await ctx.workspace.run(['sh', '-c', _LAUNCHER, 'sh', *args], timeout=self._timeout)
        # Before `WorkspaceError`: a timeout keeps its partial output, other workspace failures are not retryable.
        except WorkspaceTimeoutError as e:
            review = parse_macroscope_stream(_combined_output(e.stdout, e.stderr).splitlines())
            if review.review_id is None:
                raise ModelRetry(f'The Macroscope review timed out after {self._timeout}s.') from None
            return _stopped_early(review, 'timed_out')
        except WorkspaceError as e:
            raise_tool_failure(e)
        output = _combined_output(result.stdout, result.stderr)
        review = parse_macroscope_stream(output.splitlines())
        if review.review_id is not None:
            return review if result.exit_code == 0 else _stopped_early(review, 'interrupted')
        self._raise_not_started(result.exit_code, output, base)

    def _raise_not_started(self, exit_code: int, output: str, base: str | None) -> NoReturn:
        """Explain a review that emitted no `review_id`, to the user or, when the model's `base` may be at fault, to the model."""
        if exit_code == _NOT_FOUND_EXIT and not output.strip():
            if '/' in self._command:
                raise UserError(
                    f'No executable Macroscope CLI at {self._command!r} in the workspace. '
                    f'Fix the `command` path, or install the CLI with:\n    {_INSTALL_COMMAND}'
                )
            raise UserError(_INSTALL_HINT)
        tail = output.strip()[-_ERROR_TAIL_CHARS:]
        # `env` reports launch failures with status 126 or 127 and names the command in its diagnostic.
        # Check both so the same status returned by a CLI that did start remains the CLI's own result.
        launch_failed = exit_code in _LAUNCH_FAILURE_EXITS and any(
            line.startswith('env:') and self._command in line for line in output.splitlines()
        )
        if launch_failed:
            raise UserError(
                f'The Macroscope CLI ({self._command!r}) could not be launched in the workspace.\n\nCLI output:\n{tail}'
            )
        if base is not None:
            # The model's own `base` may be what failed (e.g. a ref that does not exist), so let
            # it drop or change the argument; a retry without it then surfaces a setup error.
            raise ModelRetry(
                f'The Macroscope review did not start with base={base!r}. Check that the ref '
                f'exists, or call again without `base`.\n\nCLI output:\n{tail}'
            )
        # Covers a CLI that is not signed in, one that could not run at all (lost +x, bad
        # interpreter), and a configured `base` that does not exist; the output tail says which.
        base_hint = f'if it reports an unknown ref, fix the configured `base` {self._base!r}; ' if self._base else ''
        raise UserError(
            f'The Macroscope review did not start (no review_id in the CLI output, exit code {exit_code}). '
            f'If the output shows the CLI could not run, reinstall it; {base_hint}otherwise run `macroscope` '
            'once in the workspace to sign in and complete the setup wizard.'
            f'\n\nCLI output:\n{tail}'
        )


def _combined_output(stdout: str, stderr: str) -> str:
    """Join the CLI's output streams for parsing.

    The parse-relevant markers all arrive on stderr; joining with a newline keeps a stdout
    chunk without a trailing newline from gluing onto the first stderr line.
    """
    return f'{stdout}\n{stderr}'


def _stopped_early(review: MacroscopeReview, reason: Literal['timed_out', 'interrupted']) -> MacroscopeReview:
    """Mark a review the CLI stopped before reporting a terminal status; a reported status is kept."""
    if review.status != _UNKNOWN_STATUS:
        return review
    return review.model_copy(update={'status': reason})

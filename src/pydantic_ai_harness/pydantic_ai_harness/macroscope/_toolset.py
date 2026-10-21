"""Macroscope toolset -- runs the `macroscope` CLI code review and returns findings."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from pydantic_ai import RunContext
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset, ToolsetTool
from pydantic_ai.workspaces import WorkspaceError, WorkspaceTimeoutError
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored
from pydantic_ai_harness._workspace import raise_tool_failure, supports_commands

_INSTALL_HINT = (
    'The `macroscope` CLI was not found on PATH in the workspace. Install it with:\n'
    '    curl -sSL https://raw.githubusercontent.com/prassoai/macroscope-local/main/install.sh | bash\n'
    'then run `macroscope` once to sign in and choose a Macroscope workspace.'
)

_REVIEW_ID_PREFIX = 'review_id='
_ISSUE_EVENT_PREFIX = 'issue_event='
_ISSUE_STATUS_PREFIX = 'issue_status='

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

    The tool runs the `macroscope` binary installed in the run's workspace (`ctx.workspace`),
    in its working directory, so the review sees the repository where the agent works, local or in a sandbox. It
    collects the streamed findings and returns them as a `MacroscopeReview`; validating and
    fixing the findings is left to the agent's other tools.

    Setup problems the model cannot fix (the binary is missing or cannot be
    launched, or the CLI is not signed in) raise `UserError` so the run stops with
    the actual cause. Failures the model can act on (a timeout, or a review that did
    not start with a `base` the model chose) raise `ModelRetry`.
    """

    def __init__(self, *, command: str, base: str | None, timeout: float, cwd: Path | None = None) -> None:
        super().__init__()
        if cwd is not None:
            warn_argument_ignored('MacroscopeToolset', 'cwd', SET_WORKING_DIR_ON_THE_WORKSPACE, stacklevel=3)
        self._command = command
        self._base = base
        self._timeout = timeout
        self.add_function(self.run_macroscope_review, name='run_macroscope_review')

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
        # `--raw` forces the machine-readable `issue_event=` stream instead of the interactive
        # TUI the CLI shows on a terminal, so parsing works regardless of whether the workspace
        # attaches a pty to the command. Needs a recent macroscope build (the CLI added the
        # flag mid-2026 and self-updates on invocation).
        args = [self._command, 'codereview', '--raw']
        base_ref = base if base is not None else self._base
        if base_ref is not None:
            args += ['--base', base_ref]
        exit_code, output = await self._run_cli(ctx, args)
        review = parse_macroscope_stream(output.splitlines())
        if review.review_id is not None:
            return review
        if exit_code == _NOT_FOUND_EXIT and not output.strip():
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
        # Covers a CLI that is not signed in and one that could not run at all (lost +x, bad
        # interpreter); the output tail says which.
        raise UserError(
            f'The Macroscope review did not start (no review_id in the CLI output, exit code {exit_code}). '
            'If the output shows the CLI could not run, reinstall it; otherwise run `macroscope` once in '
            'the workspace to sign in and complete the setup wizard.'
            f'\n\nCLI output:\n{tail}'
        )

    async def _run_cli(self, ctx: RunContext[AgentDepsT], args: list[str]) -> tuple[int, str]:
        """Run the macroscope CLI in the workspace and return its exit code and combined stdout+stderr text.

        The workspace enforces `timeout` and stops the command when it expires.
        """
        try:
            result = await ctx.workspace.run(['sh', '-c', _LAUNCHER, 'sh', *args], timeout=self._timeout)
        # Before `WorkspaceError`: a timeout is retryable, other workspace failures are not.
        except WorkspaceTimeoutError:
            raise ModelRetry(f'The Macroscope review timed out after {self._timeout}s.') from None
        except WorkspaceError as e:
            raise_tool_failure(e)
        # The parse-relevant markers all arrive on stderr; join with a newline so a
        # stdout chunk without a trailing newline cannot glue onto the first stderr line.
        return result.exit_code, f'{result.stdout}\n{result.stderr}'

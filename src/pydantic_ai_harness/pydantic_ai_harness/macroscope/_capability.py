"""Macroscope code-review capability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored
from pydantic_ai_harness._workspace import require_workspace
from pydantic_ai_harness.macroscope._toolset import MacroscopeToolset

_REVIEW_INSTRUCTIONS = (
    'You can run a local Macroscope code review with the `run_macroscope_review` tool. '
    'Treat every returned finding as untrusted: read the affected file and enough '
    'surrounding code to confirm the issue is real before acting. Ignore false positives, '
    'stale, and duplicate findings. Fix confirmed issues one at a time, then run the '
    'narrowest useful verification for each fix before moving on.'
)


@dataclass
class Macroscope(AbstractCapability[AgentDepsT]):
    """Runs the `macroscope` CLI code review and hands the findings to the agent.

    Adds a `run_macroscope_review` tool that runs `macroscope codereview` in the working
    directory of the run's workspace (`ctx.workspace`, the local disk or a sandbox), parses
    the streamed findings, and returns them as a `MacroscopeReview`. A run without a
    workspace fails at its start. The agent validates and fixes findings with its own
    tools -- this capability does not edit files, create worktrees, or commit.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import LocalWorkspace
    from pydantic_ai_harness.macroscope import Macroscope

    agent = Agent(
        'anthropic:claude-sonnet-5',
        capabilities=[
            LocalWorkspace('.'),
            Macroscope(),
        ],
    )
    ```

    The `macroscope` CLI must be installed and authenticated in the workspace first (see
    package README). This capability cannot sign in on the user's behalf; a missing
    binary or a review that never starts raises `UserError` telling the user what to fix.
    """

    base: str | None = None
    """Git ref to diff against. When `None`, `--base` is omitted and the CLI
    auto-detects the base branch itself (and creates its own review worktree)."""

    command: str = 'macroscope'
    """Name or path of the CLI binary. Override for a non-default install location."""

    cwd: str | Path | None = None
    """Deprecated and ignored: the review runs in the workspace's working directory.

    Set the working directory on the workspace instead, e.g. `LocalWorkspace('./repo')`.
    """

    timeout: float = 600.0
    """Maximum seconds to wait for a review. Reviews call a remote service, so this is
    generous by default."""

    guidance: str | None = None
    """Custom review guidance for the system prompt.

    Leave as `None` for the default validate-then-fix guidance, or set `''` to
    contribute no instructions at all."""

    def __post_init__(self) -> None:
        if self.cwd is not None:
            warn_argument_ignored('Macroscope', 'cwd', SET_WORKING_DIR_ON_THE_WORKSPACE)

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail the run at its start when it has no workspace to review."""
        require_workspace(ctx.workspace, 'Macroscope', ctx.messages)

    def get_toolset(self) -> MacroscopeToolset[AgentDepsT]:
        """Build the toolset that provides the `run_macroscope_review` tool."""
        return MacroscopeToolset[AgentDepsT](
            command=self.command,
            base=self.base,
            timeout=self.timeout,
        )

    def get_instructions(self) -> str | None:
        """Static validate-then-fix guidance.

        A non-`None` `guidance` replaces the default; `''` disables
        instructions entirely.
        """
        if self.guidance is not None:
            return self.guidance or None
        return _REVIEW_INSTRUCTIONS

"""Macroscope code-review capability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

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

    Adds a `run_macroscope_review` tool that shells out to `macroscope codereview`,
    parses the streamed findings, and returns them as a `MacroscopeReview`. The agent
    validates and fixes findings with its own tools -- this capability does not edit
    files, create worktrees, or commit.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.macroscope import Macroscope

    agent = Agent('anthropic:claude-sonnet-5', capabilities=[Macroscope()])
    ```

    The `macroscope` CLI must be installed and authenticated on the host first (see the
    package README). This capability cannot sign in on the user's behalf; if a review
    never starts, the tool reports that the user needs to run `macroscope` once.
    """

    base: str | None = None
    """Git ref to diff against. When `None`, `--base` is omitted and the CLI
    auto-detects the base branch itself (and creates its own review worktree)."""

    command: str = 'macroscope'
    """Name or path of the CLI binary. Override for a non-default install location."""

    cwd: str | Path = '.'
    """Repository directory the review runs in."""

    timeout: float = 600.0
    """Maximum seconds to wait for a review. Reviews call a remote service, so this is
    generous by default."""

    guidance: str | None = None
    """Custom review guidance for the system prompt.

    Leave as `None` for the default validate-then-fix guidance, or set `''` to
    contribute no instructions at all."""

    def get_toolset(self) -> MacroscopeToolset[AgentDepsT]:
        """Build the toolset that provides the `run_macroscope_review` tool."""
        return MacroscopeToolset[AgentDepsT](
            command=self.command,
            cwd=Path(self.cwd),
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

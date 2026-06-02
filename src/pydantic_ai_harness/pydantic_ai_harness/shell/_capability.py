"""Shell capability that provides command execution for agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.shell._toolset import ShellToolset

_DEFAULT_DENIED_COMMANDS: list[str] = [
    'rm',
    'rmdir',
    'mkfs',
    'dd',
    'format',
    'shutdown',
    'reboot',
    'halt',
    'poweroff',
    'init',
]


@dataclass
class Shell(AbstractCapability[AgentDepsT]):
    """Shell command execution for agents.

    Commands execute in a subprocess rooted at `cwd`. Use `allowed_commands`
    or `denied_commands` to control what the agent can invoke.
    """

    cwd: str | Path = '.'
    """Working directory for command execution."""

    allowed_commands: Sequence[str] = field(default_factory=list[str])
    """If non-empty, only these command names may be executed (allowlist)."""

    denied_commands: Sequence[str] = field(default_factory=lambda: list(_DEFAULT_DENIED_COMMANDS))
    """These command names are always rejected (denylist).

    Defaults to blocking destructive commands (rm, dd, shutdown, etc.).
    Set to an empty list to disable.
    """

    denied_operators: Sequence[str] = field(default_factory=list[str])
    """Shell operators that are blocked (e.g. '>', '>>', '|' for restrictive mode)."""

    default_timeout: float = 30.0
    """Default timeout in seconds for command execution."""

    max_output_chars: int = 50_000
    """Maximum characters of output returned to the model."""

    persist_cwd: bool = False
    """If True, track cd commands and adjust the working directory for subsequent calls."""

    allow_interactive: bool = False
    """If True, allow interactive commands (vi, nano, ssh, etc.). Blocked by default."""

    def get_toolset(self) -> AgentToolset[AgentDepsT]:
        """Build and return the shell toolset."""
        return ShellToolset[AgentDepsT](
            cwd=Path(self.cwd),
            allowed_commands=self.allowed_commands,
            denied_commands=self.denied_commands,
            denied_operators=self.denied_operators,
            default_timeout=self.default_timeout,
            max_output_chars=self.max_output_chars,
            persist_cwd=self.persist_cwd,
            allow_interactive=self.allow_interactive,
        )

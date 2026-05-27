"""Shell capability that provides command execution for agents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
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

_DEFAULT_DENIED_OPERATORS: list[str] = []


@dataclass
class Shell(AbstractCapability[Any]):
    """Gives an agent the ability to run shell commands.

    Commands execute in a subprocess rooted at ``cwd``. Use ``allowed_commands``
    or ``denied_commands`` to control what the agent can invoke. Output is
    automatically truncated to keep model context manageable.

    Example::

        from pydantic_ai import Agent
        from pydantic_ai_harness.shell import Shell

        agent = Agent('openai:gpt-4o', capabilities=[Shell(cwd='.')])

        # Only allow specific commands
        agent = Agent(
            'openai:gpt-4o',
            capabilities=[Shell(allowed_commands=['ls', 'cat', 'grep', 'find'])]
        )
    """

    cwd: str | Path = '.'
    """Working directory for command execution."""

    allowed_commands: Sequence[str] = field(default_factory=lambda: list[str]())
    """If non-empty, only these command names may be executed (allowlist)."""

    denied_commands: Sequence[str] = field(default_factory=lambda: list(_DEFAULT_DENIED_COMMANDS))
    """These command names are always rejected (denylist).

    Defaults to blocking destructive commands (rm, dd, shutdown, etc.).
    Set to an empty list to disable.
    """

    denied_operators: Sequence[str] = field(default_factory=lambda: list(_DEFAULT_DENIED_OPERATORS))
    """Shell operators that are blocked (e.g. '>', '>>', '|' for restrictive mode)."""

    default_timeout: float = 30.0
    """Default timeout in seconds for command execution."""

    max_output_chars: int = 50_000
    """Maximum characters of output returned to the model."""

    persist_cwd: bool = False
    """If True, track cd commands and adjust the working directory for subsequent calls."""

    allow_interactive: bool = False
    """If True, allow interactive commands (vi, nano, ssh, etc.). Blocked by default."""

    def get_toolset(self) -> AgentToolset[Any] | None:
        """Build and return the shell toolset."""
        return ShellToolset(
            cwd=Path(self.cwd),
            allowed_commands=self.allowed_commands,
            denied_commands=self.denied_commands,
            denied_operators=self.denied_operators,
            default_timeout=self.default_timeout,
            max_output_chars=self.max_output_chars,
            persist_cwd=self.persist_cwd,
            allow_interactive=self.allow_interactive,
        )

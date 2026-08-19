"""Shell capability that provides command execution for agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT

from pydantic_ai_harness.shell._toolset import ShellToolset

_DEFAULT_DENIED_COMMANDS: tuple[str, ...] = (
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
)


LLM_API_KEY_ENV_PATTERNS: tuple[str, ...] = (
    'ANTHROPIC_*',
    'GATEWAY_*',
    'GEMINI_*',
    'GOOGLE_*',
    'OPENAI_*',
    'OPENROUTER_*',
    'PYDANTIC_AI_GATEWAY_API_KEY',
)
"""Glob patterns for common LLM provider credentials, for `denied_env_patterns`.

Pass these to keep provider credentials out of the subprocess's own environment.
This is not a security boundary: a command running under the same OS identity
may still read the parent process's environment through system interfaces such
as Linux procfs. Use OS-level isolation for untrusted commands. Covers provider
prefixes only -- not other host secrets, and the prefixes are coarse (`GOOGLE_*`
also strips `GOOGLE_APPLICATION_CREDENTIALS`), so treat it as a starting point.
Not a default: stripping env silently would break agents that rely on inherited
credentials, so opt in explicitly.
"""


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

    denied_commands: Sequence[str] = _DEFAULT_DENIED_COMMANDS
    """These command names are always rejected (denylist).

    Defaults to blocking destructive commands (rm, dd, shutdown, etc.).
    Set to an empty list to disable.
    """

    denied_operators: Sequence[str] = field(default_factory=list[str])
    """Shell operators that are blocked (e.g. '>', '>>', '|' for restrictive mode)."""

    default_timeout: float = 30.0
    """Default timeout in seconds for command execution."""

    max_output_chars: int = 50_000
    """Maximum characters of output returned to the model. Must be positive."""

    persist_cwd: bool = False
    """If True, track cd commands and adjust the working directory for subsequent calls."""

    allow_interactive: bool = False
    """If True, allow interactive commands (vi, nano, ssh, etc.). Blocked by default."""

    env: Mapping[str, str] | None = None
    """Explicit environment for spawned subprocesses, replacing inheritance.

    When `None` (default) the subprocess inherits the parent environment. Set
    this to a fixed mapping to start subprocesses with exactly these variables
    in its own environment. This is not a security boundary: a command running
    as the same OS user may read secrets from the parent process through system
    interfaces such as Linux procfs. Use OS-level isolation for untrusted commands.
    """

    denied_env_patterns: Sequence[str] = field(default_factory=list[str])
    """Glob patterns for environment variable names to strip before spawning.

    Follows the `denied_*` naming convention but matches by glob (`fnmatch`,
    e.g. `OPENAI_*`), since env secrets cluster by prefix -- unlike
    `denied_commands`, which matches executable names exactly. Names matching
    any pattern are removed from the base environment; applied on top of `env`
    when both are set, so patterns filter an explicit `env` too. See
    `LLM_API_KEY_ENV_PATTERNS` for a ready-made provider-credential denylist.
    """

    def __post_init__(self) -> None:
        """Resolve the built-in denylist according to the selected policy."""
        if self.denied_commands is _DEFAULT_DENIED_COMMANDS:
            self.denied_commands = [] if self.allowed_commands else list(_DEFAULT_DENIED_COMMANDS)

    def get_toolset(self) -> ShellToolset[AgentDepsT]:
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
            env=self.env,
            denied_env_patterns=self.denied_env_patterns,
        )

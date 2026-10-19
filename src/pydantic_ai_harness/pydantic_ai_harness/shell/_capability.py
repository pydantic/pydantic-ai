"""Shell capability that provides command execution for agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored
from pydantic_ai_harness._workspace import require_workspace
from pydantic_ai_harness.shell._jobs import CONTROL_TIMEOUT
from pydantic_ai_harness.shell._toolset import RUN_SCOPED_TOOL_NAMES, ShellToolset

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

Pass these to keep provider credentials in an explicit `env` from reaching commands.
The patterns filter only `env`: the workspace decides the rest of a command's
environment, and a local workspace passes on only the host's `PATH`, `HOME`,
`LANG`, `LC_ALL` and `LC_CTYPE` plus its own `env`. Covers provider prefixes only --
not other secrets, and the prefixes are coarse (`GOOGLE_*` also strips `GOOGLE_APPLICATION_CREDENTIALS`), so
treat it as a starting point. Not a default: opt in explicitly.
"""


@dataclass
class Shell(AbstractCapability[AgentDepsT]):
    """Shell command execution for agents.

    Commands run in the run's workspace (`ctx.workspace`), starting in its working directory.
    Attach a workspace to the run, such as `LocalWorkspace(...)` for a local checkout or a
    sandbox provider's capability; a run without one fails at its start. Use
    `allowed_commands` or `denied_commands` to control what the agent can invoke.

    `Shell` is not a security boundary: a command reaches whatever the workspace lets it,
    whatever `FileSystem`'s `root_dir` says. Isolate untrusted work with a sandbox workspace.
    """

    cwd: str | Path | None = None
    """Deprecated and ignored: commands start in the workspace's working directory.

    Set the working directory on the workspace instead, e.g. `LocalWorkspace('./repo')`.
    """

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

    max_file_bytes: int | None = field(default=None, kw_only=True)
    """Optional per-file size limit for `run_command` and `start_command` commands, not total disk usage.

    Must be positive. Applied with the workspace shell's `ulimit -f`, rounded up to whole
    blocks (512 bytes in POSIX `sh`, 1 KiB in bash). `persist_cwd` and the persistent
    `shell` tool are rejected.
    """

    persist_cwd: bool = False
    """If True, track cd commands and adjust the working directory for subsequent calls."""

    allow_interactive: bool = False
    """If True, allow interactive commands (vi, nano, ssh, etc.). Blocked by default."""

    env: Mapping[str, str] | None = None
    """Variables added to every command's environment, on top of the workspace's own.

    Commands get exactly the workspace's environment plus these, minus names matching
    `denied_env_patterns`. A local workspace's environment is the host's `PATH`, `HOME`,
    `LANG`, `LC_ALL` and `LC_CTYPE` plus its own `env`.
    """

    denied_env_patterns: Sequence[str] = field(default_factory=list[str])
    """Glob patterns for names to drop from `env` before it reaches the workspace.

    Follows the `denied_*` naming convention but matches by glob (`fnmatch`,
    e.g. `OPENAI_*`), since env secrets cluster by prefix -- unlike
    `denied_commands`, which matches executable names exactly. The patterns
    filter `env` only; the workspace's own environment is its provider's to
    configure. See `LLM_API_KEY_ENV_PATTERNS` for a ready-made
    provider-credential denylist.
    """

    tools: Sequence[str] = RUN_SCOPED_TOOL_NAMES
    """Which tools to register, from `SHELL_TOOL_NAMES`.

    The default is `run_command`, `start_command`, `check_command`, and
    `stop_command`; a background command keeps running across runs until it
    exits, is stopped, or the workspace ends. Name `shell` to register the
    persistent tool instead: a foreground call waits at most `default_timeout` seconds
    (capped at 270) before returning handles to the still-running process, and
    the model reads the returned log and status files with its other tools.
    `persist_cwd` does not apply to `shell`; each command starts in the working directory.
    """

    _toolset: ShellToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """The one toolset `get_toolset` returns, so durable execution sees the leaf it registered."""

    def __post_init__(self) -> None:
        """Resolve the built-in denylist according to the selected policy."""
        if self.cwd is not None:
            warn_argument_ignored('Shell', 'cwd', SET_WORKING_DIR_ON_THE_WORKSPACE)
        if self.denied_commands is _DEFAULT_DENIED_COMMANDS:
            self.denied_commands = [] if self.allowed_commands else list(_DEFAULT_DENIED_COMMANDS)

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        result = await handler()
        if self.persist_cwd and ctx.run_id is not None:
            # A cancelled worker may be replaced while the workflow is still active;
            # only a completed run can safely discard the cwd needed by its successor.
            with anyio.move_on_after(CONTROL_TIMEOUT, shield=True):
                await self.get_toolset().clear_run_cwd(ctx)
        return result

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail the run at its start when it has no workspace to run commands in."""
        require_workspace(ctx.workspace, 'Shell', ctx.messages)

    def get_toolset(self) -> ShellToolset[AgentDepsT]:
        """The shell toolset, built once; its `for_run` gives each run a fresh copy."""
        if self._toolset is None:
            self._toolset = self._make_toolset()
        return self._toolset

    def _make_toolset(self) -> ShellToolset[AgentDepsT]:
        return ShellToolset[AgentDepsT](
            allowed_commands=self.allowed_commands,
            denied_commands=self.denied_commands,
            denied_operators=self.denied_operators,
            default_timeout=self.default_timeout,
            max_output_chars=self.max_output_chars,
            max_file_bytes=self.max_file_bytes,
            persist_cwd=self.persist_cwd,
            allow_interactive=self.allow_interactive,
            env=self.env,
            denied_env_patterns=self.denied_env_patterns,
            tools=self.tools,
            id=self.id or 'shell',
        )

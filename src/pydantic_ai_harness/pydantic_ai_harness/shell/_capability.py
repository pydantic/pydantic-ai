"""Shell capability that provides command execution for agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import anyio

from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._conjoin import RecordsStated, intersection, merged_mapping, union
from pydantic_ai_harness._warn import SET_WORKING_DIR_ON_THE_WORKSPACE, warn_argument_ignored
from pydantic_ai_harness._workspace import require_workspace
from pydantic_ai_harness.announce_working_directory import AnnounceWorkingDirectory
from pydantic_ai_harness.shell._jobs import CONTROL_TIMEOUT
from pydantic_ai_harness.shell._toolset import PERSISTENT_TOOL_NAME, RUN_SCOPED_TOOL_NAMES, ShellToolset

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
class Shell(RecordsStated, AbstractCapability[AgentDepsT]):
    """Shell command execution for agents.

    Commands run in the run's workspace (`ctx.workspace`), starting in its working directory.
    Attach a workspace to the run, such as `LocalWorkspace(...)` for a local checkout or a
    sandbox provider's capability; a run without one fails at its start. Use
    `allowed_commands` or `denied_commands` to control what the agent can invoke.

    `Shell` is not a security boundary: a command reaches whatever the workspace lets it,
    whatever `FileSystem`'s `root_dir` says. Isolate untrusted work with a sandbox workspace.

    An agent has one `Shell` per `id`, `'shell'` by default. Two that share it, such as a `Shell`
    added beside `Coder()`, are combined so that every restriction either states applies (see
    `combine`), so adding one only ever narrows. Give each its own `id` to keep two apart.
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

    max_log_bytes: int | None = field(default=100_000_000, kw_only=True)
    """Size past which a background or `shell` command's log is emptied, leaving a notice. Must be positive.

    Checked once a second while the command runs, which keeps running, and once more when it ends. Bounds the disk a
    chatty or runaway command (such as one reading its own log) can fill. `None` disables it.
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

    secret_env: Mapping[str, str] | None = field(default=None, kw_only=True, repr=False)
    """Secrets added to every command's environment, like `env`, and masked in what the tools return.

    Exact occurrences of each value are replaced with `[REDACTED:<name>]` in tool results and retry or failure
    messages before the model sees them. This guards against a secret printed by accident, not against a command
    that transforms or splits it, and a short or common value is masked everywhere it appears.
    `denied_env_patterns` does not filter these. Log files, which the model can read with other tools, and the
    persistent `shell` tool's events hold the raw output.
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

    id: str | None = field(default='shell', kw_only=True)
    """The capability id, `'shell'` by default. Two `Shell`s with one id are combined into one."""

    _toolset: ShellToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """The one toolset `get_toolset` returns, so durable execution sees the leaf it registered."""

    def __post_init__(self) -> None:
        """Resolve the built-in denylist according to the selected policy."""
        if self.cwd is not None:
            warn_argument_ignored('Shell', 'cwd', SET_WORKING_DIR_ON_THE_WORKSPACE)
        if self.denied_commands is _DEFAULT_DENIED_COMMANDS:
            self.denied_commands = [] if self.allowed_commands else list(_DEFAULT_DENIED_COMMANDS)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Combine `Shell`s sharing an id so that every restriction any of them states applies.

        Allowed commands and tools intersect; denied commands, operators and env patterns add up; the
        smaller timeout and output, log and file limits win; `allow_interactive` needs every instance to
        allow it; `env` and `secret_env` add up, and one name given two values raises.

        Only the fields an instance was constructed with take part, so a `Shell(allowed_commands=['pytest'])`
        added beside `Coder()` keeps `Coder`'s timeout, while `Shell(default_timeout=30.0)` lowers it. A copy
        made with `dataclasses.replace()` counts every field as passed.
        """
        shells = [capability for capability in capabilities if isinstance(capability, Shell)]
        assert len(shells) == len(capabilities)
        allowed, denied = _combined_commands(shells, cls._stated_values(shells, 'denied_commands'))
        tools = intersection(cls._stated_values(shells, 'tools'), owner='Shell', field='tools', empty_means_all=False)
        merged = replace(
            shells[-1],
            cwd=None,
            allowed_commands=allowed or [],
            denied_commands=denied,
            denied_operators=union(shell.denied_operators for shell in shells),
            default_timeout=cls._smallest(shells, 'default_timeout'),
            max_output_chars=cls._smallest(shells, 'max_output_chars'),
            max_file_bytes=cls._smallest(shells, 'max_file_bytes'),
            max_log_bytes=cls._smallest(shells, 'max_log_bytes'),
            persist_cwd=any(shell.persist_cwd for shell in shells),
            allow_interactive=all(shell.allow_interactive for shell in shells),
            env=merged_mapping((shell.env for shell in shells), owner='Shell', field='env'),
            denied_env_patterns=union(shell.denied_env_patterns for shell in shells),
            secret_env=merged_mapping((shell.secret_env for shell in shells), owner='Shell', field='secret_env'),
            tools=RUN_SCOPED_TOOL_NAMES if tools is None else tools,
        )
        _require_supported_by_the_persistent_tool(merged)
        return merged

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

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractCapability[AgentDepsT]:
        """Bring `AnnounceWorkingDirectory` along; copies from other capabilities merge into one line.

        A deferred one doesn't: naming the directory starts a sandbox the model may never load this capability for.
        """
        if self.defer_loading:
            return self
        return CombinedCapability([self, AnnounceWorkingDirectory()])

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
            max_log_bytes=self.max_log_bytes,
            persist_cwd=self.persist_cwd,
            allow_interactive=self.allow_interactive,
            env=self.env,
            denied_env_patterns=self.denied_env_patterns,
            secret_env=self.secret_env,
            tools=self.tools,
            id=self.id or 'shell',
        )


def _combined_commands(
    shells: Sequence[Shell[Any]], denied_lists: list[Sequence[str]]
) -> tuple[list[str] | None, list[str]]:
    """The allowlist and denylist of `shells` combined: allowlists intersect, the stated `denied_lists` add up.

    `ShellToolset` takes an allowlist or a denylist, so with an allowlist a stated denied command leaves it
    instead; the built-in denylist gives way to an allowlist, as it does in one `Shell`.
    """
    allowed = intersection([shell.allowed_commands for shell in shells], owner='Shell', field='allowed_commands')
    denied = union(denied_lists)
    if allowed is None:
        return None, denied if denied_lists else list(_DEFAULT_DENIED_COMMANDS)
    allowed = [command for command in allowed if command not in denied]
    if not allowed:
        raise UserError(
            'These `Shell` capabilities share an id, and every command one allows another denies. '
            'State one policy, or give each `Shell` its own `id` to keep them apart.'
        )
    return allowed, []


def _require_supported_by_the_persistent_tool(shell: Shell[Any]) -> None:
    """Refuse a combined `Shell` keeping the persistent `shell` tool with an option that tool can't apply."""
    if PERSISTENT_TOOL_NAME not in shell.tools:
        return
    unsupported = [name for name in ('max_file_bytes', 'persist_cwd') if getattr(shell, name)]
    if unsupported:
        raise UserError(
            f'These `Shell` capabilities share an id and keep the persistent `shell` tool (as `Coder` '
            f'registers), which does not support {" or ".join(f"`{name}`" for name in unsupported)}, so '
            'the combined `Shell` could not apply it. Drop it, or give the other `Shell` its own `id`.'
        )

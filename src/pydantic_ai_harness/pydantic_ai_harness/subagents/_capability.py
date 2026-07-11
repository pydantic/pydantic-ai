"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic_ai.agent import Agent, AgentRunResult, EventStreamHandler
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapRunHandler
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.subagents._disk import (
    AgentOverride,
    ParsedAgent,
    parse_agent_markdown,
    resolve_folders,
)
from pydantic_ai_harness.subagents._effort import clamp_effort
from pydantic_ai_harness.subagents._toolset import SubAgent, SubAgentToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

ToolResolver = Callable[[str], 'Sequence[AgentToolset[object]] | None']
"""Maps one tool name from a disk definition's `tools` list to the toolsets that
provide it, or `None` when the name is unknown (the loader warns and skips it)."""


@dataclass
class SubAgents(AbstractCapability[AgentDepsT]):
    """Let an agent delegate self-contained tasks to named sub-agents.

    Exposes a single `delegate_task(agent_name, task)` tool. Each delegation
    runs the chosen sub-agent in a fresh, isolated run (it never sees the parent
    conversation), and the available sub-agents are listed in the system prompt
    as a static, cache-stable instruction.

    Sub-agents are passed as a sequence of `SubAgent` entries, each pairing an
    agent with its per-delegate run controls (a `usage_limits` budget, a
    wall-clock `timeout_seconds`, a per-run `max_calls` budget, an `on_failure`
    steering message, and optional `name`/`description` overrides). A delegate's
    name is its `SubAgent.name`, or the agent's own `name` when unset; two
    explicitly-passed delegates resolving to the same name is an error.

    Sub-agents are also loaded from disk by default: each markdown agent definition
    under `./.agents/agents/` and `~/.agents/agents/` (or the `.claude/` equivalent)
    becomes a delegate, built with the parent's model. Disk delegates get no tools
    by default (`inherit_tools` is `False`); set `inherit_tools=True` to expose the
    parent's tools, or pass a `tool_resolver` to map their frontmatter tool names.
    Disk delegates coexist with explicitly-passed ones; explicitly-passed agents take
    precedence, then the project folder, then the home folder. A disk delegate whose
    name is already taken is skipped with a warning. Configure or disable this with
    `agent_folders`; see also `agent_overrides` and `tool_resolver`.

    The parent's `deps` are forwarded to each sub-agent (sub-agents therefore
    share the parent's `AgentDepsT`), and by default the parent's `usage` is
    shared so usage limits apply across the whole agent tree. Optionally, the
    parent's tools can be inherited (`inherit_tools`), extra capabilities can be
    applied to every sub-agent run (`shared_capabilities`), and sub-agent events
    can be streamed to a handler (`event_stream_handler`).

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.subagents import SubAgent, SubAgents

    researcher = Agent('anthropic:claude-sonnet-4-6', name='researcher', description='Researches topics')
    writer = Agent('anthropic:claude-sonnet-4-6', name='writer', description='Writes prose')

    orchestrator = Agent(
        'anthropic:claude-opus-4-7',
        capabilities=[SubAgents(agents=[SubAgent(researcher), SubAgent(writer)])],
    )
    ```
    """

    agents: Sequence[SubAgent[AgentDepsT]] = ()
    """The sub-agents to expose, each a `SubAgent` pairing an agent with its
    per-delegate run controls. See `SubAgent`. These take precedence over any
    disk-loaded agents of the same name."""

    agent_folders: str | Sequence[Path] | None = 'agents'
    """Where to load markdown agent definitions from, in addition to `agents`.
    Defaults to the conventional layout, so constructing the capability auto-loads
    a repo's agent files with no extra configuration.

    - a folder-name `str` (the default `'agents'` is the conventional layout): for
      the project root (cwd) then the home root, load from `<root>/.agents/<name>/`,
      falling back to `<root>/.claude/<name>/` when `<root>/.agents/` is absent.
    - a sequence of paths: load from exactly those folders, in order.
    - `None`: disable disk loading entirely (only `agents` are exposed).

    Missing folders are skipped. Within a folder every `*.md` file is a candidate."""

    agent_overrides: Mapping[str, AgentOverride] = field(default_factory=dict[str, AgentOverride])
    """Per-disk-agent overrides keyed by the agent's name. An entry can set the
    agent's `model` (otherwise the parent's model is inherited) and its `effort`
    (otherwise the minimum floor). Has no effect on explicitly-passed `agents`."""

    tool_resolver: ToolResolver | None = None
    """Optional override for how a disk agent gets its tools. When set, each tool
    name in a definition's `tools`/`allowed-tools` frontmatter is passed to this
    resolver and the returned toolsets are attached to that agent; an unknown name
    (resolver returns `None`) is skipped with a warning. When unset, the
    frontmatter tool list is ignored and disk agents inherit the parent's tools
    via `inherit_tools` (set `inherit_tools=True` to expose them)."""

    forward_usage: bool = True
    """If `True`, the parent run's `usage` is shared with each sub-agent run, so
    token usage aggregates and usage limits apply across the whole agent tree."""

    inherit_tools: bool = False
    """If `True`, the parent agent's tools are exposed to each sub-agent run (the
    delegate tool itself is filtered out, so sub-agents can't recurse into
    further delegation). Off by default to avoid silently widening sub-agent access."""

    shared_capabilities: Sequence[AgentCapability[AgentDepsT]] = ()
    """Capabilities applied to every sub-agent run, in addition to whatever each
    sub-agent already has."""

    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None
    """If set, this handler is passed to each sub-agent run, so the sub-agent's
    model-streaming and tool events surface to the caller. The handler receives
    the sub-agent's own `RunContext` and event stream."""

    tool_name: str = 'delegate_task'
    """Name of the delegate tool exposed to the model."""

    tool_retries: int | None = 2
    """Retries for the delegate tool -- how many extra attempts it gets after a
    sub-agent error before the parent run aborts. A sub-agent failure (e.g. it
    exhausts its own output retries) surfaces to the parent as a tool retry it
    can react to by re-delegating with a corrected task. The retry counter
    resets after any successful delegation, so this bounds consecutive failures,
    not total ones. Defaults to `2` (pydantic-ai's per-tool default is `1`) so a
    repeated flaky sub-agent does not abort the parent run on its first repeat;
    set `None` to inherit the parent agent's default tool retries instead."""

    contain_errors: bool = False
    """Default for `SubAgent.contain_errors`: whether an unexpected sub-agent crash
    is caught and returned to the parent as a bounded `ModelRetry` instead of
    aborting the parent run. Off by default, so a crash propagates. Any `SubAgent`
    can override this per delegate. See `SubAgent.contain_errors` for the
    containment contract and what always propagates regardless."""

    _by_name: dict[str, SubAgent[AgentDepsT]] = field(
        default_factory=dict[str, 'SubAgent[AgentDepsT]'], init=False, repr=False, compare=False
    )
    """Sub-agents keyed by resolved name, built in `__post_init__` and passed to
    the toolset. Insertion order matches `agents` for a stable prompt listing."""

    _call_counts: dict[str, dict[str, int]] = field(
        default_factory=dict[str, 'dict[str, int]'], init=False, repr=False, compare=False
    )
    """Run-scoped delegation counts (run_id -> name -> count), shared with the
    toolset and cleared per run in `wrap_run`. Backs `SubAgent.max_calls`."""

    def __post_init__(self) -> None:
        by_name: dict[str, SubAgent[AgentDepsT]] = {}
        for sub_agent in self.agents:
            name = sub_agent.resolved_name
            if name is None:
                raise ValueError('Sub-agent has no name: give its `Agent` a `name`, or set `SubAgent(name=...)`.')
            if name in by_name:
                raise ValueError(
                    f'Duplicate sub-agent name {name!r}. Each sub-agent needs a distinct name; '
                    f'set `SubAgent(name=...)` to disambiguate.'
                )
            by_name[name] = sub_agent
        # Disk agents are lower precedence than explicit ones and than earlier
        # folders, so a name already taken is shadowed (a warning, not an error --
        # overriding a home agent from the project, or a disk agent from code, is
        # the intended path).
        for sub_agent in self._load_disk_agents():
            name = sub_agent.resolved_name
            if name is None:  # pragma: no cover - disk agents always get a name (frontmatter or stem)
                continue
            if name in by_name:
                warnings.warn(
                    f'Disk sub-agent {name!r} is shadowed by a higher-precedence definition; skipping it.',
                    stacklevel=2,
                )
                continue
            by_name[name] = sub_agent
        self._by_name = by_name

    def _load_disk_agents(self) -> list[SubAgent[AgentDepsT]]:
        """Build a `SubAgent` for every markdown definition in `agent_folders`.

        Folders are returned in precedence order (project before home); within a
        folder, files are loaded in sorted name order for a stable listing.
        """
        if self.agent_folders is None:
            return []
        result: list[SubAgent[AgentDepsT]] = []
        for folder in resolve_folders(self.agent_folders, Path.cwd(), Path.home()):
            if not folder.is_dir():
                continue
            for path in sorted(folder.glob('*.md')):
                try:
                    text = path.read_text(encoding='utf-8')
                except (OSError, UnicodeDecodeError) as exc:
                    warnings.warn(f'Skipping unreadable disk sub-agent file {str(path)!r}: {exc}', stacklevel=2)
                    continue
                parsed = parse_agent_markdown(text)
                result.append(self._build_disk_agent(parsed.name or path.stem, parsed))
        return result

    def _build_disk_agent(self, name: str, parsed: ParsedAgent) -> SubAgent[AgentDepsT]:
        """Build one disk-defined sub-agent: parent model + floored effort, tools resolved or inherited.

        The agent is constructed with `deps_type=object` so the parent's deps (of
        any type) flow through unused at delegation; this also lets a disk
        `SubAgent[object]` sit in the parent's `SubAgent[AgentDepsT]` roster.
        """
        override = self.agent_overrides.get(name)
        model = override.model if override is not None else None
        effort = override.effort if override is not None else None
        toolsets = self._resolve_disk_tools(parsed.tools) if self.tool_resolver is not None else None
        agent = Agent(
            model,
            deps_type=object,
            name=name,
            description=parsed.description,
            instructions=parsed.body or None,
            model_settings=ModelSettings(thinking=clamp_effort(effort)),
            toolsets=toolsets,
        )
        return SubAgent(agent)

    def _resolve_disk_tools(self, tool_names: Sequence[str]) -> list[AgentToolset[object]]:
        """Map a definition's tool names to toolsets via `tool_resolver`, warning on unknown names."""
        resolver = self.tool_resolver
        if resolver is None:  # pragma: no cover - only called when tool_resolver is set
            return []
        toolsets: list[AgentToolset[object]] = []
        for tool_name in tool_names:
            resolved = resolver(tool_name)
            if resolved is None:
                warnings.warn(f'Unknown tool {tool_name!r} in disk sub-agent definition; skipping it.', stacklevel=2)
                continue
            toolsets.extend(resolved)
        return toolsets

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run the parent agent, then drop this run's delegation counts so they don't accumulate."""
        try:
            return await handler()
        finally:
            self._call_counts.pop(ctx.run_id or '', None)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable listing of the available sub-agents."""
        if not self._by_name:
            return None
        lines: list[str] = []
        for name, sub_agent in self._by_name.items():
            description = sub_agent.description or sub_agent.agent.description
            lines.append(f'- {name}: {description}' if description else f'- {name}')
        listing = '\n'.join(lines)
        return (
            f'You can delegate self-contained tasks to these sub-agents using the `{self.tool_name}` '
            f'tool. Each runs in its own fresh context and does not see this conversation, so pass '
            f'everything it needs.\n\nAvailable sub-agents:\n{listing}'
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing the delegate tool, or `None` when no sub-agents are configured."""
        if not self._by_name:
            return None
        return SubAgentToolset(
            agents=self._by_name,
            forward_usage=self.forward_usage,
            inherit_tools=self.inherit_tools,
            shared_capabilities=self.shared_capabilities,
            event_stream_handler=self.event_stream_handler,
            tool_name=self.tool_name,
            tool_retries=self.tool_retries,
            contain_errors=self.contain_errors,
            call_counts=self._call_counts,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable -- the capability holds live `Agent` instances."""
        return None

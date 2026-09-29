"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from pydantic_ai._utils import replace_no_init
from pydantic_ai.agent import Agent, AgentRunResult, EventStreamHandler
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapRunHandler
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai.workspaces import Workspace, WorkspaceBackend
from pydantic_ai_harness._warn import HarnessDeprecationWarning, warn_default_changed
from pydantic_ai_harness._workspace import require_workspace, secondary_workspace, workspace_path
from pydantic_ai_harness.subagents._disk import AgentOverride, DiskDefinition, load_definitions
from pydantic_ai_harness.subagents._effort import clamp_effort
from pydantic_ai_harness.subagents._models import ModelOption, as_option, model_label, validate_restriction
from pydantic_ai_harness.subagents._toolset import (
    DEFAULT_MAX_DEPTH,
    SELF_AGENT_NAME,
    SubAgent,
    SubAgentToolset,
    at_max_depth,
)

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions

ToolResolver = Callable[[str], 'Sequence[AgentToolset[object]] | None']
"""Maps one tool name from a disk definition's `tools` list to the toolsets that
provide it, or `None` when the name is unknown (the loader warns and skips it)."""


class _UnsetFolders(tuple[Path, ...]):
    """Private marker distinguishing an omitted `agent_folders` from explicit `None`."""


_UNSET_FOLDERS: Sequence[str | Path] = _UnsetFolders()


def _folder_path(folder: str | Path) -> str:
    return workspace_path(folder) if isinstance(folder, Path) else folder


def _option_line(key: str, option: ModelOption) -> str:
    """One model-menu line for the prompt listing: the key, its model, its hint."""
    label = f'- {key} ({model_label(option.model)})'
    return f'{label}: {option.description}' if option.description else label


_SELF_DESCRIPTION = (
    'A fresh run of this same agent, with all of your tools, instructions, and capabilities. '
    'Use it to hand off a self-contained part of the work.'
)
"""How the running agent is described in the prompt listing when `include_self` is on."""

_MERGEABLE_FIELDS = frozenset({'agents', 'models', 'include_self'})
"""The only fields a merge composes: the roster (including whether it lists the running agent),
and the model options that roster may pick from.

An allow-list rather than a list of exceptions. Every other public field of `SubAgents` says *how*
the delegates run rather than *who* they are -- so merging them applies one harness's policy to the
other's sub-agents. Enumerating those instead would mean a field added
later merges silently by default, which is the wrong way round for a decision nobody made.
"""


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

    Delegations run on the sub-agent's own model unless a `models` menu is
    configured, in which case `delegate_task` also takes a `model` argument naming
    one of the menu's keys, so the parent routes each task to the model that fits
    it. A `SubAgent` can restrict which keys it accepts (`SubAgent.models`).

    Sub-agents can also be loaded from disk: each markdown agent definition under
    `agent_folders` in the run's workspace becomes a delegate, built with the
    parent's model. Folders are read at the start of every run, through
    `ctx.workspace`, or through `workspace` when set. Disk delegates get no tools
    by default; pass a `tool_resolver` to map their frontmatter tool names.
    Disk delegates coexist with explicitly-passed ones; explicitly-passed agents take
    precedence. Convention folders use `.agents/` before `.claude/`; explicit folder sequences use
    earlier folders before later ones. A disk delegate whose
    name is already taken is skipped with a warning. Configure or disable this with
    `agent_folders`; see also `agent_overrides` and `tool_resolver`.

    With `include_self=True`, the roster also lists the running agent itself, as `self`:
    a delegation starts a fresh run of `RunContext.agent`, so the delegate has every
    capability, tool, and instruction bound to that agent, including guardrails and
    approval hooks added next to this one. Delegations nest at most `max_depth` levels.

    The parent's `deps` are forwarded to each sub-agent (sub-agents therefore
    share the parent's `AgentDepsT`), and by default the parent's `usage` is
    shared so usage limits apply across the whole agent tree. Extra capabilities
    can be applied to every sub-agent run (`shared_capabilities`), and sub-agent
    events can be streamed to a handler (`event_stream_handler`).

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

    models: Mapping[str, Model | KnownModelName | str | ModelOption] = field(
        default_factory=dict[str, 'Model | KnownModelName | str | ModelOption']
    )
    """A menu of models the parent can route an individual delegation to, keyed by
    the name the parent uses to pick one. Off by default: with no menu the delegate
    tool has no `model` argument and every delegation runs the way it always did.

    Each value is a model reference, or a `ModelOption` carrying a routing hint and
    its own `ModelSettings` (so one key can mean "same model, more thinking"). The
    keys and their descriptions are listed in the system prompt, so name them for
    the job -- `'fast'`, `'deep'` -- rather than for the vendor. `SubAgent.models`
    restricts which of them a given delegate accepts.

    ```python
    from pydantic_ai_harness.subagents import SubAgents

    SubAgents(models={'fast': 'anthropic:claude-haiku-4-5', 'deep': 'anthropic:claude-opus-4-7'})
    ```
    """

    agent_folders: str | Sequence[str | Path] | None = _UNSET_FOLDERS
    """Where to load markdown agent definitions from, in addition to `agents`.
    Off by default: only `agents` are exposed unless this is set. Every folder is
    read at the start of each run through the run's workspace (`ctx.workspace`),
    or through `workspace` when set.

    - a folder-name `str` (`'agents'` is the conventional layout): load
      from both `.agents/<name>/` and `.claude/<name>/` under the workspace's working
      directory, in that order. Skipped when the run has no workspace.
    - a sequence of paths in the workspace, absolute or relative to its working
      directory: load from exactly those folders, in order. A run with no workspace
      fails at its start; pass `workspace=LocalWorkspaceBackend('.')` to read them from
      this machine.
    - `None`: no disk loading (the default).

    Missing folders are skipped. Within a folder every `*.md` file is a candidate.

    This previously defaulted to `'agents'`. Leaving it unset while the run's
    workspace contains a conventional agent definition emits a
    `HarnessDeprecationWarning`; pass either value explicitly to stay silent."""

    agent_overrides: Mapping[str, AgentOverride] = field(default_factory=dict[str, AgentOverride])
    """Per-disk-agent overrides keyed by the agent's name. An entry can set the
    agent's `model` (otherwise the parent's model is inherited) and its `effort`
    (otherwise the minimum floor). Has no effect on explicitly-passed `agents`."""

    tool_resolver: ToolResolver | None = None
    """Optional override for how a disk agent gets its tools. When set, each tool
    name in a definition's `tools`/`allowed-tools` frontmatter is passed to this
    resolver and the returned toolsets are attached to that agent; an unknown name
    (resolver returns `None`) is skipped with a warning. When unset, the
    frontmatter tool list is ignored and disk agents have no tools of their own."""

    forward_usage: bool = True
    """If `True`, the parent run's `usage` is shared with each sub-agent run, so
    token usage aggregates and usage limits apply across the whole agent tree."""

    inherit_tools: bool = False
    """Deprecated: setting it to `True` emits a `HarnessDeprecationWarning`.

    If `True`, the parent agent's own tools (registered via `tools=` or `toolsets=`,
    not those contributed by capabilities) are exposed to each sub-agent run, minus
    the delegate tool. Bind required tools directly to explicit sub-agents, use
    `include_self=True` to delegate to an agent with all of the parent's tools and
    capabilities, or use `tool_resolver` to give disk agents tools."""

    shared_capabilities: Sequence[AgentCapability[AgentDepsT]] = ()
    """Capabilities applied to every sub-agent run, in addition to whatever each
    sub-agent already has."""

    event_stream_handler: EventStreamHandler[AgentDepsT] | None = None
    """If set, this handler is passed to each sub-agent run, so the sub-agent's
    model-streaming and tool events surface to the caller. The handler receives
    the sub-agent's own `RunContext` and event stream."""

    tool_name: str = 'delegate_task'
    """Name of the delegate tool exposed to the model."""

    id: str | None = field(default='sub_agents', kw_only=True)
    """One-off: an agent exposes a single delegate tool, so the id is fixed.

    `tool_name` is one name, so two `SubAgents` capabilities register the same tool and collide.
    Declaring the id here is what makes two of them merge instead, unioning their rosters -- which
    is what lets a packaged harness that delegates compose with another that does the same.

    Keyword-only on the field rather than through a `KW_ONLY` marker: a marker applies to every
    field after it, which would take `tool_retries` and `contain_errors` off the positional
    contract they already have.
    """

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

    workspace: WorkspaceBackend | None = field(default=None, kw_only=True)
    """A workspace to read `agent_folders` from instead of the run's, such as
    `LocalWorkspaceBackend('/app')` for definitions that ship with the application.

    A backend, not the `LocalWorkspace` capability. It is used in-process only in this release: a
    durable engine does not route it through its workflow machinery."""

    include_self: bool = False
    """If `True`, the roster also lists the running agent itself, as `self`.

    A delegation to `self` starts a fresh run of `RunContext.agent` on the parent run's model
    (or the `models` option the parent picks), so the delegate has every capability, toolset, and instruction bound to that `Agent` --
    guardrails, approval gates, and audit hooks included, since they are registered again in
    the child run. What was passed to the parent's `run()` rather than bound to the `Agent`
    (run-level `capabilities`, `toolsets`, `instructions`, `model_settings`) does not carry
    over, so this capability has to be bound to the `Agent` itself; passing it to `run()`
    raises a `UserError` when the run starts.

    `inherit_tools` does not apply to `self`, whose tools are already the parent's. The
    delegate can delegate in turn, up to `max_depth`."""

    max_depth: int = DEFAULT_MAX_DEPTH
    """How many levels a delegation tree may have, counting the top-level run as the first.

    The default of `3` lets the top-level run delegate, and its delegates delegate once more.
    A run at the limit gets neither the delegate tool nor the sub-agent listing. The level
    is tracked per task tree, across every `SubAgents` capability, and each capability enforces
    its own limit. This bounds `include_self`, whose delegate carries the delegate tool again,
    and a roster that reaches the same agent through another path."""

    _by_name: dict[str, SubAgent[AgentDepsT]] = field(
        default_factory=dict[str, 'SubAgent[AgentDepsT]'], init=False, repr=False, compare=False
    )
    """Sub-agents keyed by resolved name, built in `__post_init__` (and rebuilt per run in
    `before_run` once the workspace's definitions are read) and passed to the toolset.
    Insertion order matches `agents` for a stable prompt listing."""

    _workspace: Workspace | None = field(default=None, init=False, repr=False, compare=False)
    """`workspace` wrapped as a `Workspace`, or `None` to read from the run's."""

    _built: dict[DiskDefinition, SubAgent[AgentDepsT]] = field(
        default_factory=dict[DiskDefinition, 'SubAgent[AgentDepsT]'], init=False, repr=False, compare=False
    )
    """Disk delegates built so far, shared with every per-run copy. A definition is built once, so
    runs over unchanged files reuse the same agents and `tool_resolver` is not asked again."""

    _warned_host_folders: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """Whether the no-longer-read host folder warning was given, shared with every per-run copy so it
    is given once per instance rather than once per run."""

    _agent_folders_was_unset: bool = field(default=False, init=False, repr=False, compare=False)
    """Whether `agent_folders` was omitted, so an affected run gets the default-change warning."""

    _warned_default_folders: set[str] = field(default_factory=set[str], init=False, repr=False, compare=False)
    """The conventional workspace folder already reported for this capability instance."""

    _run_toolset: SubAgentToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """This run's delegate toolset, on a per-run copy only. Built once per run, so every step of the
    run sees the same toolset instance."""

    _toolset: SubAgentToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """The delegate toolset of a capability that reads no folders, built on first use and reset with the
    roster, so every run gets the instance durable execution registered when the agent was built."""

    _per_run: bool = field(default=False, init=False, repr=False, compare=False)
    """Whether this instance is a per-run copy made by `for_run` to read the workspace's definitions."""

    _menu: dict[str, ModelOption] = field(default_factory=dict[str, ModelOption], init=False, repr=False, compare=False)
    """`models` normalized to `ModelOption` entries, built in `__post_init__`.
    Insertion order matches `models` for a stable prompt listing and enum."""

    _delegation_off: bool = field(default=False, init=False, repr=False, compare=False)
    """Set on the instance `for_run` returns for a run at `max_depth`, which contributes nothing."""

    _call_counts: dict[str, dict[str, int]] = field(
        default_factory=dict[str, 'dict[str, int]'], init=False, repr=False, compare=False
    )
    """Run-scoped delegation counts (run_id -> name -> count), shared with the
    toolset and cleared per run in `wrap_run`. Backs `SubAgent.max_calls`."""

    def __post_init__(self) -> None:
        if self.agent_folders is _UNSET_FOLDERS:
            self.agent_folders = None
            self._agent_folders_was_unset = True
        if self.inherit_tools:
            warnings.warn(
                '`SubAgents(inherit_tools=True)` is deprecated and will be removed in a future release. It passes '
                "only the parent's own tools, not those of its capabilities. Bind required tools directly to "
                'explicit sub-agents, use `include_self=True` to delegate to a fresh run of the agent with all '
                'of its tools and capabilities, or use `tool_resolver` to give disk-loaded agents tools.',
                category=HarnessDeprecationWarning,
                stacklevel=3,
            )
        self._workspace = secondary_workspace(self.workspace, 'SubAgents')
        self._build_roster([])

    def _disk_agents(self, definitions: Sequence[DiskDefinition]) -> list[SubAgent[AgentDepsT]]:
        """The delegates for `definitions`, built on first sight and reused after that."""
        result: list[SubAgent[AgentDepsT]] = []
        for definition in definitions:
            sub_agent = self._built.get(definition)
            if sub_agent is None:
                sub_agent = self._built[definition] = self._build_disk_agent(definition)
            result.append(sub_agent)
        return result

    def _build_roster(self, disk_agents: list[SubAgent[AgentDepsT]]) -> None:
        if self.max_depth < 1:
            raise ValueError(f'`max_depth` counts the top-level run, so it must be at least 1; got {self.max_depth}.')
        by_name: dict[str, SubAgent[AgentDepsT]] = {}
        for sub_agent in self.agents:
            name = sub_agent.resolved_name
            if name is None:
                raise ValueError('Sub-agent has no name: give its `Agent` a `name`, or set `SubAgent(name=...)`.')
            if self.include_self and name == SELF_AGENT_NAME:
                raise ValueError(
                    f'Sub-agent name {SELF_AGENT_NAME!r} is taken by the running agent when `include_self=True`; '
                    f'set `SubAgent(name=...)` to rename it.'
                )
            if name in by_name:
                raise ValueError(
                    f'Duplicate sub-agent name {name!r}. Each sub-agent needs a distinct name; '
                    f'set `SubAgent(name=...)` to disambiguate.'
                )
            by_name[name] = sub_agent
        # Disk agents are lower precedence than explicit ones and than earlier
        # folders, so a name already taken is shadowed (a warning, not an error --
        # overriding a shared folder's agent from an earlier one, or a disk agent from code, is
        # the intended path).
        for sub_agent in disk_agents:
            name = sub_agent.resolved_name
            if name is None:  # pragma: no cover - disk agents always get a name (frontmatter or stem)
                continue
            if name in by_name or (self.include_self and name == SELF_AGENT_NAME):
                warnings.warn(
                    f'Disk sub-agent {name!r} is shadowed by a higher-precedence definition; skipping it.',
                    stacklevel=2,
                )
                continue
            by_name[name] = sub_agent
        self._by_name = by_name
        self._toolset = None
        self._menu = {key: as_option(value) for key, value in self.models.items()}
        for name, sub_agent in by_name.items():
            validate_restriction(name, sub_agent.models, self._menu)

    def _build_disk_agent(self, definition: DiskDefinition) -> SubAgent[AgentDepsT]:
        """Build one disk-defined sub-agent: parent model + floored effort, tools resolved or inherited.

        The agent is constructed with `deps_type=object` so the parent's deps (of
        any type) flow through unused at delegation; this also lets a disk
        `SubAgent[object]` sit in the parent's `SubAgent[AgentDepsT]` roster.
        """
        name, parsed = definition.name, definition.parsed
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

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> SubAgents[AgentDepsT]:
        """A per-run copy when agent folders are read; otherwise `self`.

        A run at `max_depth` gets a copy without the delegate tool and listing instead. The per-run
        copy starts from the explicit roster; `before_run` adds the folders' definitions. Its
        instructions and toolset are read after that, so they list the delegates this run has.
        """
        if at_max_depth(self.max_depth):
            return replace_no_init(self, _delegation_off=True)
        if self.agent_folders is None and not self._agent_folders_was_unset:
            return self
        run = replace_no_init(self)
        run._per_run = True
        run._run_toolset = run._make_toolset()
        return run

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Read the agent folders' definitions through the workspace and rebuild this run's roster."""
        if not self._per_run:
            return
        folders = self.agent_folders
        if folders is None:
            if self._agent_folders_was_unset:
                await self._warn_agent_folders_default_changed(ctx)
            return
        workspace = self._workspace
        if workspace is None:
            if isinstance(folders, str) and not ctx.workspace.attached:
                # Convention discovery: a run with no workspace has no project to look in.
                await self._warn_host_folder_ignored(folders, [await anyio.Path.cwd(), await anyio.Path.home()])
                return
            require_workspace(ctx.workspace, 'SubAgents', ctx.messages)
            workspace = ctx.workspace
            home = await anyio.Path.home()
            if isinstance(folders, str) and str(home) != await workspace.working_dir():
                # Earlier releases also read the home folder, which the run's workspace does not reach.
                await self._warn_host_folder_ignored(folders, [home])
        definitions = await load_definitions(
            workspace, folders if isinstance(folders, str) else [_folder_path(folder) for folder in folders]
        )
        if not definitions:
            return
        self._build_roster(self._disk_agents(definitions))
        self._run_toolset = self._make_toolset()

    async def _warn_agent_folders_default_changed(self, ctx: RunContext[AgentDepsT]) -> None:
        """Warn once when the old default would have found definitions in this run's workspace."""
        workspace = self._workspace
        if workspace is None:
            if not ctx.workspace.attached:
                return
            workspace = ctx.workspace
        try:
            agents_root = await workspace.stat('.agents')
        except (FileNotFoundError, NotADirectoryError):
            folder = '.claude/agents'
        else:
            folder = '.agents/agents' if agents_root.is_dir else '.claude/agents'
        try:
            entries = await workspace.list_dir(folder)
        except (FileNotFoundError, NotADirectoryError):
            return
        if not any(not entry.is_dir and entry.name.endswith('.md') for entry in entries):
            return
        resolved = await workspace.resolve(folder)
        if resolved in self._warned_default_folders:
            return
        self._warned_default_folders.add(resolved)
        warn_default_changed(
            owner='SubAgents',
            option='agent_folders',
            old='agents',
            new=None,
            impact=f'The agent definitions in {resolved!r} are no longer loaded as delegates.',
            stacklevel=3,
        )

    async def _warn_host_folder_ignored(self, name: str, roots: Sequence[anyio.Path]) -> None:
        """Earlier releases read this folder under `roots` on this machine; say so once rather than drop it silently."""
        if self._warned_host_folders:
            return
        for root in roots:
            for hidden in ('.agents', '.claude'):
                folder = root / hidden / name
                if await folder.is_dir():
                    self._warned_host_folders.add(str(folder))
                    warnings.warn(
                        f'`SubAgents` did not load the agent definitions in `{folder}`: definitions are now read '
                        'through a workspace only. Pass '
                        f'`SubAgents(workspace=LocalWorkspaceBackend({str(root)!r}))` to keep loading them, or '
                        '`agent_folders=None` to turn discovery off.',
                        category=HarnessDeprecationWarning,
                        stacklevel=2,
                    )
                    return

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run the parent agent, then drop this run's delegation counts so they don't accumulate."""
        if self.include_self:
            self._require_bound_to(ctx.agent)
        try:
            return await handler()
        finally:
            self._call_counts.pop(ctx.run_id or '', None)

    def _require_bound_to(self, agent: Agent[Any, Any] | None) -> None:
        """Refuse a run where delegating to `agent` would not bring this capability along.

        A delegation to the running agent runs `agent` again, which re-registers what is bound to it
        and nothing that was passed to `run()`. If this capability was passed to `run()`, the
        delegate would come up without it, and most likely without the capabilities passed next to
        it, which is the difference `include_self` exists to remove. Checked by equality rather
        than by type, so a different `SubAgents` bound to the agent -- which a run-level one of the
        same `id` overrides -- does not stand in for this one.
        """
        if agent is None:  # pragma: no cover - the running agent is always set during a run
            return
        bound: list[AbstractCapability[Any]] = []
        agent.root_capability.apply(bound.append)
        if self not in bound:
            raise UserError(
                '`SubAgents(include_self=True)` delegates to a fresh run of the agent, which only carries what is '
                'bound to the `Agent`, not what was passed to `run()`. Bind this capability (and whatever it '
                'should bring along, such as `Coder`) with `Agent(capabilities=[...])`, or turn delegation to '
                'the running agent off.'
            )

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Cache-stable listing of the available sub-agents and models.

        A per-run copy returns it as a function, rendered after `before_run` has read the
        workspace's definitions; it is the same text on every step of the run.
        """
        if self._per_run:
            return lambda _ctx: self._render_instructions()
        return self._render_instructions()

    def _render_instructions(self) -> str | None:
        if self._delegation_off or (not self._by_name and not self.include_self):
            return None
        lines: list[str] = [f'- {SELF_AGENT_NAME}: {_SELF_DESCRIPTION}'] if self.include_self else []
        for name, sub_agent in self._by_name.items():
            description = sub_agent.description or sub_agent.agent.description
            restriction = f' (models: {", ".join(sub_agent.models)})' if sub_agent.models else ''
            lines.append(f'- {name}: {description}{restriction}' if description else f'- {name}{restriction}')
        listing = '\n'.join(lines)
        instructions = (
            f'You can delegate self-contained tasks to these sub-agents using the `{self.tool_name}` '
            f'tool. Each runs in its own fresh context and does not see this conversation, so pass '
            f'everything it needs.\n\nAvailable sub-agents:\n{listing}'
        )
        if not self._menu:
            return instructions
        options = '\n'.join(_option_line(key, option) for key, option in self._menu.items())
        return (
            f'{instructions}\n\nPass one of these keys as `model` to run a sub-agent on it, matching the '
            f"option to how hard the task is. Omit `model` to use the sub-agent's default. A sub-agent "
            f'listed with its own `(models: ...)` accepts only those.\n\nAvailable models:\n{options}'
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing the delegate tool, or `None` when no sub-agents are configured.

        A per-run copy returns a function yielding the toolset `before_run` settled on, the same
        instance for every step of the run.
        """
        if self._delegation_off:
            return None
        if self._per_run:
            return lambda _ctx: self._run_toolset
        if self._toolset is None:
            self._toolset = self._make_toolset()
        return self._toolset

    def _make_toolset(self) -> SubAgentToolset[AgentDepsT] | None:
        if not self._by_name and not self.include_self:
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
            models=self._menu,
            include_self=self.include_self,
            max_depth=self.max_depth,
            id=self.id,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable -- the capability holds live `Agent` instances."""
        return None

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Compose the rosters, and require everything else to already agree.

        Two packaged harnesses on one agent each bring their delegates, and composing them is what
        the shared `id` is for. Only `agents` and `models` are composed. Every other field decides
        how the delegates *run* -- what capabilities they are handed, whether they see the parent's
        tools, what the delegate tool is called, where delegates are loaded from -- so merging it
        would apply one harness's policy to the other's sub-agents, which neither author asked for.
        Those must agree, and say so when they do not.

        Disk definitions are not part of the merge: the per-run copy reads them in `before_run`,
        after run-level capabilities are combined.
        """
        first = capabilities[0]
        assert isinstance(first, cls)
        merged_agents = list(first.agents)
        merged_models = dict(first.models)
        include_self = first.include_self
        for other in capabilities[1:]:
            assert isinstance(other, cls)
            for field_info in dataclasses.fields(first):
                name = field_info.name
                if name in _MERGEABLE_FIELDS or not field_info.compare or name == 'id':
                    continue
                mine, theirs = getattr(first, name), getattr(other, name)
                if mine != theirs:
                    raise UserError(
                        f'Capability id {first.id!r} is used by multiple SubAgents capabilities that disagree '
                        f'on {name!r} ({mine!r} and {theirs!r}). Only the roster is composed; everything else '
                        "decides how the delegates run, so merging it would apply one set of delegates' "
                        f'configuration to the other. Give them distinct `id`s to keep both, or make {name!r} '
                        'agree.'
                    )
            merged_agents.extend(other.agents)
            merged_models.update(other.models)
            include_self = include_self or other.include_self

        merged = replace_no_init(first, agents=merged_agents, models=merged_models, include_self=include_self)
        merged._build_roster([])
        if merged._per_run:
            merged._run_toolset = merged._make_toolset()
        return merged

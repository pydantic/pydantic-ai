"""Sub-agent capability: delegate self-contained tasks to named child agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic_ai.agent import AbstractAgent, AgentRunResult, EventStreamHandler
from pydantic_ai.capabilities import AbstractCapability, AgentCapability, WrapRunHandler
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.experimental.subagents._toolset import SubAgentLimits, SubAgentToolset

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions


@dataclass
class SubAgents(AbstractCapability[AgentDepsT]):
    """Let an agent delegate self-contained tasks to named sub-agents.

    Exposes a single `delegate_task(agent_name, task)` tool. Each delegation
    runs the chosen sub-agent in a fresh, isolated run (it never sees the parent
    conversation), and the available sub-agents are listed in the system prompt
    as a static, cache-stable instruction.

    The parent's `deps` are forwarded to each sub-agent (sub-agents therefore
    share the parent's `AgentDepsT`), and by default the parent's `usage` is
    shared so usage limits apply across the whole agent tree. Optionally, the
    parent's tools can be inherited (`inherit_tools`), extra capabilities can be
    applied to every sub-agent run (`shared_capabilities`), and sub-agent events
    can be streamed to a handler (`event_stream_handler`).

    Per-delegate run controls (a `usage_limits` budget, a wall-clock
    `timeout_seconds`, a per-run `max_calls` budget, and an `on_failure` steering
    message) are configured via `limits`, keyed by sub-agent name. See
    `SubAgentLimits`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.experimental.subagents import SubAgents

    researcher = Agent('anthropic:claude-sonnet-4-6', name='researcher', description='Researches topics')
    writer = Agent('anthropic:claude-sonnet-4-6', name='writer', description='Writes prose')

    orchestrator = Agent(
        'anthropic:claude-opus-4-7',
        capabilities=[SubAgents(agents={'researcher': researcher, 'writer': writer})],
    )
    ```
    """

    agents: Mapping[str, AbstractAgent[AgentDepsT, Any]] = field(
        default_factory=dict[str, 'AbstractAgent[AgentDepsT, Any]']
    )
    """Mapping of sub-agent name to the agent that runs when it's delegated to."""

    descriptions: Mapping[str, str] | None = None
    """Optional per-name description overrides for the system-prompt listing.

    When a name is absent here, the agent's own `description` is used (if any)."""

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

    limits: Mapping[str, SubAgentLimits] = field(default_factory=dict[str, SubAgentLimits])
    """Per-delegate run controls, keyed by sub-agent name. A name absent here runs
    with no per-delegate controls (the `SubAgents` defaults). See `SubAgentLimits`
    for `usage_limits`, `timeout_seconds`, `max_calls`, and `on_failure`."""

    _call_counts: dict[str, dict[str, int]] = field(
        default_factory=dict[str, 'dict[str, int]'], init=False, repr=False, compare=False
    )
    """Run-scoped delegation counts (run_id -> name -> count), shared with the
    toolset and cleared per run in `wrap_run`. Backs `SubAgentLimits.max_calls`."""

    def __post_init__(self) -> None:
        # A limits entry for an unknown name would silently run unbudgeted, defeating
        # the point of configuring a budget; fail fast on the likely typo instead.
        unknown = set(self.limits) - set(self.agents)
        if unknown:
            raise ValueError(
                f'`limits` names sub-agents not in `agents`: {", ".join(sorted(unknown))}. '
                f'Available sub-agents: {", ".join(sorted(self.agents)) or "(none)"}.'
            )

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        """Run the parent agent, then drop this run's delegation counts so they don't accumulate."""
        try:
            return await handler()
        finally:
            self._call_counts.pop(ctx.run_id or '', None)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Static, cache-stable listing of the available sub-agents."""
        if not self.agents:
            return None
        overrides = self.descriptions or {}
        lines: list[str] = []
        for name, agent in self.agents.items():
            description = overrides.get(name) or agent.description
            lines.append(f'- {name}: {description}' if description else f'- {name}')
        listing = '\n'.join(lines)
        return (
            f'You can delegate self-contained tasks to these sub-agents using the `{self.tool_name}` '
            f'tool. Each runs in its own fresh context and does not see this conversation, so pass '
            f'everything it needs.\n\nAvailable sub-agents:\n{listing}'
        )

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Toolset providing the delegate tool, or `None` when no sub-agents are configured."""
        if not self.agents:
            return None
        return SubAgentToolset(
            agents=self.agents,
            forward_usage=self.forward_usage,
            inherit_tools=self.inherit_tools,
            shared_capabilities=self.shared_capabilities,
            event_stream_handler=self.event_stream_handler,
            tool_name=self.tool_name,
            limits=self.limits,
            call_counts=self._call_counts,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable -- the capability holds live `Agent` instances."""
        return None

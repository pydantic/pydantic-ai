"""Sub-agent toolset: a single delegate tool that runs named child agents."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic

from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

# Private import: pydantic-ai has no public way to tell capability-contributed
# toolsets apart from the agent's own in `agent.toolsets`.
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.usage import UsageLimits


@dataclass(frozen=True)
class SubAgent(Generic[AgentDepsT]):
    """One delegate: a child agent plus its per-delegate run controls.

    Pass a sequence of these as `SubAgents(agents=[...])`. The delegate's name --
    how the parent model refers to it, and how it is listed in the system prompt --
    is `name` when set, otherwise the agent's own `name`. An agent with neither is
    rejected by `SubAgents`.

    Every control below is optional; an unset field leaves the corresponding
    behaviour at the `SubAgents` default.
    """

    agent: AbstractAgent[AgentDepsT, Any]
    """The agent that runs when this delegate is invoked."""

    name: str | None = None
    """Name the parent model uses to delegate to this agent. Defaults to the
    agent's own `name` when unset."""

    description: str | None = None
    """Description for the system-prompt listing. Defaults to the agent's own
    `description` when unset; a delegate with neither is listed by name alone."""

    usage_limits: UsageLimits | None = None
    """Request/token budget for one delegation. When set, the child runs with
    its own usage accounting so the budget counts only the child's own requests
    and tokens (not the parent's or siblings'), even when `forward_usage=True`.
    The tradeoff: that child's tokens no longer aggregate into the parent's
    `usage`. Hitting this budget is a soft outcome (steering message), not a
    run-stopping `UsageLimitExceeded`."""

    timeout_seconds: float | None = None
    """Wall-clock budget for one delegation. When the child exceeds it, the run
    is cancelled and the parent gets a soft steering message instead of hanging
    on the child."""

    max_calls: int | None = None
    """Maximum number of delegations to this sub-agent per parent run. Once
    reached, further delegations return a soft budget-exhausted message without
    running the child."""

    on_failure: str | None = None
    """Steering message returned to the parent for any soft degradation of this
    delegate (timeout, child failure, usage budget reached, call budget
    exhausted), in place of the built-in default. Setting it also makes child
    failures soft: a child error returns this message as a normal tool result
    instead of raising a parent `ModelRetry`."""

    @property
    def resolved_name(self) -> str | None:
        """The delegate's name: `name` if set, else the agent's own `name`."""
        return self.name or self.agent.name


def _is_capability_contributed(toolset: AbstractToolset[AgentDepsT]) -> bool:
    """Whether `toolset`'s tree contains a `CapabilityOwnedToolset`."""
    found = False

    def visit(node: AbstractToolset[AgentDepsT]) -> None:
        nonlocal found
        if isinstance(node, CapabilityOwnedToolset):
            found = True

    toolset.apply(visit)
    return found


class SubAgentToolset(FunctionToolset[AgentDepsT]):
    """Exposes one delegate tool that dispatches a task to a named sub-agent.

    Each delegation runs the child agent in a fresh run with its own message
    history, so the sub-agent never sees the parent conversation. The parent's
    `deps` are forwarded; its `usage` is shared when enabled; its tools are
    inherited when enabled; any `shared_capabilities` are applied to every
    sub-agent run; and sub-agent events are streamed to `event_stream_handler`
    when one is set. Per-delegate run controls come from each `SubAgent`.
    """

    def __init__(
        self,
        *,
        agents: Mapping[str, SubAgent[AgentDepsT]],
        forward_usage: bool,
        inherit_tools: bool,
        shared_capabilities: Sequence[AgentCapability[AgentDepsT]],
        event_stream_handler: EventStreamHandler[AgentDepsT] | None,
        tool_name: str,
        call_counts: dict[str, dict[str, int]],
    ) -> None:
        super().__init__()
        self._agents: dict[str, SubAgent[AgentDepsT]] = dict(agents)
        self._forward_usage = forward_usage
        self._inherit_tools = inherit_tools
        self._shared_capabilities = list(shared_capabilities)
        self._event_stream_handler = event_stream_handler
        self._tool_name = tool_name
        # Run-scoped delegation counts, keyed by run_id then sub-agent name.
        # Shared with the capability, which clears each run's entry in wrap_run.
        self._call_counts = call_counts
        self.add_function(self.delegate_task, name=tool_name)

    def _inherited_toolsets(self, ctx: RunContext[AgentDepsT]) -> list[AbstractToolset[AgentDepsT]] | None:
        """The parent agent's own toolsets, excluding capability-contributed ones.

        Capability toolsets are bound to capability instances registered in the
        parent run; carrying them into the sub-agent's run (where their owner is
        not registered) fails `CapabilityOwnedToolset`'s ownership resolution, and
        the tools would arrive without the hooks and instructions that make them
        work. Use `shared_capabilities` to share a capability with sub-agents.
        The delegate tool itself is also filtered out by name, so delegation
        cannot recurse. When this toolset was registered via the `SubAgents`
        capability the capability filter already drops it; the name filter covers
        direct registration in `Agent(toolsets=[...])`, where nothing wraps it in
        `CapabilityOwnedToolset`.
        """
        agent = ctx.agent
        if agent is None:  # pragma: no cover - the running agent is always set during a run
            return None
        # Capability toolsets surface as `CombinedToolset(CapabilityOwnedToolset(...))`
        # entries, so ownership is detected by walking each tree. Only core's capability
        # assembly constructs `CapabilityOwnedToolset`, so a tree containing one is
        # capability-contributed in its entirety.
        return [
            toolset.filtered(lambda _ctx, tool_def: tool_def.name != self._tool_name)
            for toolset in agent.toolsets
            if not _is_capability_contributed(toolset)
        ]

    def _budget_exhausted(self, ctx: RunContext[AgentDepsT], agent_name: str, max_calls: int) -> bool:
        """Increment this run's delegation count for `agent_name` and report whether it is over budget.

        Runs synchronously before any await, so concurrent delegations in one run
        count without a lock.
        """
        counts = self._call_counts.setdefault(ctx.run_id or '', {})
        counts[agent_name] = counts.get(agent_name, 0) + 1
        return counts[agent_name] > max_calls

    async def delegate_task(self, ctx: RunContext[AgentDepsT], agent_name: str, task: str) -> str:
        """Delegate a self-contained task to a named sub-agent and return its result.

        The sub-agent runs in its own fresh context and does not see this
        conversation, so `task` must contain everything it needs.

        Args:
            ctx: The run context (provides the parent's deps, usage, and tools).
            agent_name: Name of the sub-agent to run. Must be one of the agents
                listed in the instructions.
            task: The complete, self-contained instruction for the sub-agent.
        """
        sub_agent = self._agents.get(agent_name)
        if sub_agent is None:
            available = ', '.join(sorted(self._agents))
            raise ModelRetry(f'Unknown sub-agent {agent_name!r}. Available sub-agents: {available}.')

        if sub_agent.max_calls is not None and self._budget_exhausted(ctx, agent_name, sub_agent.max_calls):
            return self._steer(
                sub_agent.on_failure,
                f'Delegate budget for {agent_name!r} is exhausted for this run '
                f'({sub_agent.max_calls} call(s)). Synthesize from existing evidence and '
                f'choose the next action; do not delegate to {agent_name!r} again.',
            )

        toolsets = self._inherited_toolsets(ctx) if self._inherit_tools else None
        capabilities = self._shared_capabilities or None
        usage_limits: UsageLimits | None
        if sub_agent.usage_limits is not None:
            # Isolated accounting so the per-child budget counts only this child.
            own_budget = True
            usage = None
            usage_limits = sub_agent.usage_limits
        else:
            own_budget = False
            usage = ctx.usage if self._forward_usage else None
            usage_limits = None

        run = sub_agent.agent.run(
            task,
            deps=ctx.deps,
            usage=usage,
            usage_limits=usage_limits,
            toolsets=toolsets,
            capabilities=capabilities,
            event_stream_handler=self._event_stream_handler,
        )
        timeout = sub_agent.timeout_seconds
        try:
            result = await (asyncio.wait_for(run, timeout) if timeout is not None else run)
        except asyncio.TimeoutError:
            return self._steer(
                sub_agent.on_failure,
                f'Sub-agent {agent_name!r} exceeded its {timeout}s time budget. '
                f'Treat this as a recoverable observation and decide from existing evidence.',
            )
        except UsageLimitExceeded:
            if own_budget:
                return self._steer(
                    sub_agent.on_failure,
                    f'Sub-agent {agent_name!r} reached its usage budget. '
                    f'Treat this as a recoverable observation and decide from existing evidence.',
                )
            # A shared/parent usage limit means the whole tree is out of budget.
            raise
        except (ModelRetry, UnexpectedModelBehavior) as exc:
            if sub_agent.on_failure is not None:
                return sub_agent.on_failure
            # Soft sub-agent failures come back to the parent as a retry it can react to.
            raise ModelRetry(f'Sub-agent {agent_name!r} failed: {exc}') from exc
        return str(result.output)

    @staticmethod
    def _steer(on_failure: str | None, default: str) -> str:
        """A soft steering message: the delegate's `on_failure` override, else `default`."""
        if on_failure is not None:
            return on_failure
        return default

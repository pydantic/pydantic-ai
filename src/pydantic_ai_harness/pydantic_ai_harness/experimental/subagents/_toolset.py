"""Sub-agent toolset: a single delegate tool that runs named child agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset


class SubAgentToolset(FunctionToolset[AgentDepsT]):
    """Exposes one delegate tool that dispatches a task to a named sub-agent.

    Each delegation runs the child agent in a fresh run with its own message
    history, so the sub-agent never sees the parent conversation. The parent's
    `deps` are forwarded; its `usage` is shared when enabled; its tools are
    inherited when enabled; any `shared_capabilities` are applied to every
    sub-agent run; and sub-agent events are streamed to `event_stream_handler`
    when one is set.
    """

    def __init__(
        self,
        *,
        agents: Mapping[str, AbstractAgent[AgentDepsT, Any]],
        forward_usage: bool,
        inherit_tools: bool,
        shared_capabilities: Sequence[AgentCapability[AgentDepsT]],
        event_stream_handler: EventStreamHandler[AgentDepsT] | None,
        tool_name: str,
    ) -> None:
        super().__init__()
        self._agents: dict[str, AbstractAgent[AgentDepsT, Any]] = dict(agents)
        self._forward_usage = forward_usage
        self._inherit_tools = inherit_tools
        self._shared_capabilities = list(shared_capabilities)
        self._event_stream_handler = event_stream_handler
        self._tool_name = tool_name
        self.add_function(self.delegate_task, name=tool_name)

    def _inherited_toolsets(self, ctx: RunContext[AgentDepsT]) -> list[AbstractToolset[AgentDepsT]] | None:
        """The parent's toolsets, with the delegate tool filtered out (no recursion)."""
        agent = ctx.agent
        if agent is None:  # pragma: no cover - the running agent is always set during a run
            return None
        return [toolset.filtered(lambda _ctx, tool_def: tool_def.name != self._tool_name) for toolset in agent.toolsets]

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
        agent = self._agents.get(agent_name)
        if agent is None:
            available = ', '.join(sorted(self._agents))
            raise ModelRetry(f'Unknown sub-agent {agent_name!r}. Available sub-agents: {available}.')
        toolsets = self._inherited_toolsets(ctx) if self._inherit_tools else None
        capabilities = self._shared_capabilities or None
        usage = ctx.usage if self._forward_usage else None
        try:
            result = await agent.run(
                task,
                deps=ctx.deps,
                usage=usage,
                toolsets=toolsets,
                capabilities=capabilities,
                event_stream_handler=self._event_stream_handler,
            )
        except (ModelRetry, UnexpectedModelBehavior) as exc:
            # Soft sub-agent failures come back to the parent as a retry it can
            # react to; hard limits (e.g. UsageLimitExceeded) propagate.
            raise ModelRetry(f'Sub-agent {agent_name!r} failed: {exc}') from exc
        return str(result.output)

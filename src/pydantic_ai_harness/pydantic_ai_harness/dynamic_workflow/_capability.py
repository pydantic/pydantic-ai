"""Dynamic workflow capability: orchestrate sub-agents from a sandboxed Python script."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.dynamic_workflow._toolset import (
    DynamicWorkflowToolset,
    WorkflowAgent,
    WorkflowResourceLimits,
    index_workflow_agents,
    validate_workflow_agent,
)


@dataclass(kw_only=True)
class DynamicWorkflow(AbstractCapability[AgentDepsT]):
    """Capability that lets the model orchestrate named sub-agents from a Python script.

    Instead of one sub-agent per tool call, the model writes a single Python script (run in a
    Monty sandbox) that calls each sub-agent as an async function and composes the results: fan
    out with `asyncio.gather`, chain one agent's output into the next, vote, or loop until done.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.dynamic_workflow import DynamicWorkflow

    reviewer = Agent('openai:gpt-5', name='reviewer', description='Review code for bugs.')
    summarizer = Agent('openai:gpt-5', name='summarizer', description='Summarize findings.')

    orchestrator = Agent(
        'openai:gpt-5',
        capabilities=[DynamicWorkflow(agents=[reviewer, summarizer])],
    )
    ```

    Each sub-agent runs isolated (its own message history) with the parent's `deps` forwarded;
    by default the parent's `usage` accumulator is shared so the whole tree's spend is tallied in
    one place. Use `max_agent_calls` for a hard, host-enforced ceiling on sub-agent runs. Workflows
    do not nest. Set `defer_loading=True` (with a stable `id`) to keep the tool out of the prompt
    until the model loads the capability.
    """

    agents: Sequence[AbstractAgent[AgentDepsT, object] | WorkflowAgent[AgentDepsT]]
    """Sub-agents the orchestration script can call as async functions.

    Read at construction only; later mutation of the passed sequence is ignored. A raw agent is
    shorthand for `WorkflowAgent(agent)`, using the agent's own `name` and `description`; use a
    `WorkflowAgent` entry for a per-use-site override. Use `reveal()` to add a sub-agent after
    construction.
    """

    _catalog: list[WorkflowAgent[AgentDepsT]] = field(init=False, repr=False)
    """Normalized catalog passed by reference to toolsets."""

    tool_name: str = 'run_workflow'
    """Name of the orchestration tool exposed to the model."""

    max_agent_calls: int = 50
    """Maximum total sub-agent runs per agent run: an exact, host-enforced ceiling that holds even
    under concurrent fan-out (unlike a parent `usage_limits`)."""

    max_retries: int = 3
    """Maximum retries for the orchestration tool (syntax/runtime errors count as retries)."""

    forward_usage: bool = True
    """Share the parent run's `usage` accumulator with sub-agents, tallying the whole tree's
    token and request spend in one place.

    This does **not** forward the parent's `usage_limits` into sub-agent runs (`RunContext` does
    not expose the limit value): set `sub_agent_usage_limits` to bound sub-agents, or
    `max_agent_calls` for an exact ceiling on the number of runs.
    """

    inherit_model: bool = False
    """Run every sub-agent with the parent run's resolved model instead of its constructed model.

    Use this when the host can switch models per run, such as a `/model` command that passes a
    run-level model override to the parent agent. Without it, that per-run choice silently leaves
    catalog sub-agents on the model they were bound to when constructed; `inherit_model=True` makes
    the workflow crew follow the parent run's resolved model. Keep `False` to pin sub-agents to
    their own configured models.
    """

    sub_agent_usage_limits: UsageLimits | None = None
    """`UsageLimits` applied to every sub-agent run, replacing pydantic-ai's default.

    With `forward_usage=False`, a per-run `total_tokens_limit` of `T` plus `max_agent_calls` of
    `N` bounds the tree to roughly `N * T` tokens (each run can overshoot by its final response,
    since core checks token limits after a response arrives). With `forward_usage=True` the limit
    is checked against the shared counter -- a tree-wide cap, best-effort under concurrent fan-out.
    `None` keeps the default (`request_limit=50`, no token limit).
    """

    resource_limits: WorkflowResourceLimits | Literal['unlimited'] | None = None
    """Sandbox limits guarding the orchestration script's own memory/allocations.

    `None` applies a safe backstop (256 MB, 50M allocations, no execution-time cap); `'unlimited'`
    removes all limits; a `WorkflowResourceLimits` mapping is merged onto the backstop, overriding
    only the caps it names. There is no default `max_duration_secs`: Monty's timer bounds in-sandbox
    execution time and time awaiting sub-agents does not count against it, so set one only to guard
    a pure-CPU `while True` loop.
    """

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Not spec-serializable: `agents` holds live Agent objects, not YAML-expressible config.
        return None

    def __post_init__(self) -> None:
        catalog = [self._normalize_workflow_agent(entry) for entry in self.agents]
        index_workflow_agents(catalog)
        self._catalog = catalog

    def _normalize_workflow_agent(
        self, entry: AbstractAgent[AgentDepsT, object] | WorkflowAgent[AgentDepsT]
    ) -> WorkflowAgent[AgentDepsT]:
        """Normalize a public catalog entry to the internal wrapper form."""
        if isinstance(entry, WorkflowAgent):
            return entry
        return WorkflowAgent(agent=entry)

    def reveal(self, agent: AbstractAgent[AgentDepsT, object] | WorkflowAgent[AgentDepsT]) -> None:
        """Reveal a sub-agent on the next model step (the supported runtime API for doing so).

        The sub-agent is announced to the model on the next step and becomes callable then; the
        `run_workflow` description stays frozen at the agents present when the run started. Reveal
        is append-only: a revealed sub-agent cannot be removed for the rest of the run. Its resolved
        name must be a valid, unique sandbox function name, or this raises `UserError` at the call
        site. If one `DynamicWorkflow` instance is shared across concurrent runs, `reveal()` reaches
        all in-flight runs and joins the baseline catalog for runs that start afterwards.
        """
        entry = self._normalize_workflow_agent(agent)
        existing_names: set[str] = set()
        for catalog_entry in self._catalog:
            existing_names.add(validate_workflow_agent(catalog_entry, existing_names))
        validate_workflow_agent(entry, existing_names)
        self._catalog.append(entry)

    def get_toolset(self) -> DynamicWorkflowToolset[AgentDepsT]:
        """Provide the orchestration toolset to the agent."""
        return DynamicWorkflowToolset(
            # Toolsets keep this same list object; `reveal()` appends to it so in-flight
            # toolsets can fold in the new sub-agent on the next step.
            agents=self._catalog,
            tool_name=self.tool_name,
            max_agent_calls=self.max_agent_calls,
            max_retries=self.max_retries,
            forward_usage=self.forward_usage,
            inherit_model=self.inherit_model,
            sub_agent_usage_limits=self.sub_agent_usage_limits,
            resource_limits=self.resource_limits,
            toolset_id=self.id,
            owning_capability=self,
        )

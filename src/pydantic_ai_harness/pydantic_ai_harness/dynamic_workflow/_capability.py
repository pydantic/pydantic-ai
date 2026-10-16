"""Dynamic workflow capability: orchestrate sub-agents from a sandboxed Python script."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.dynamic_workflow._catalog import (
    WorkflowAgent,
    index_workflow_agents,
    validate_workflow_agent,
)
from pydantic_ai_harness.dynamic_workflow._library import WorkflowLibrary
from pydantic_ai_harness.dynamic_workflow._toolset import DynamicWorkflowToolset, WorkflowResourceLimits


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

    Scripts also get helpers: `agent()` to run any sub-agent with a per-call output `schema` or
    `model`, `parallel()` and `pipeline()` for fan-out that tolerates failures, `log()` and
    `phase()` for progress events, and `budget()`.

    Scripts worth keeping can be saved as `<name>.py` files in the `workflows` directory of the
    run's workspace, each headed by a `meta` dict. They are listed to the model, run by name, and
    nest inside one another with `workflow(name, args)`, sharing the calling script's budget.

    Each sub-agent runs isolated (its own message history) with the parent's `deps` forwarded;
    by default the parent's `usage` accumulator is shared so the whole tree's spend is tallied in
    one place. Use `max_agent_calls` for a hard, host-enforced ceiling on sub-agent runs. A
    sub-agent cannot start a workflow of its own. Set `defer_loading=True` (with a stable `id`)
    to keep the tool out of the prompt until the model loads the capability.
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
    """Sandbox limits guarding the orchestration script's own memory.

    `None` applies a 256 MB backstop with no execution-time cap; `'unlimited'` removes all limits;
    a `WorkflowResourceLimits` mapping is merged onto the backstop, overriding only the caps it
    names. There is no default `max_duration_secs`: Monty's timer bounds in-sandbox execution time
    and time awaiting sub-agents does not count against it, so set one only to guard a pure-CPU
    `while True` loop.
    """

    workflows: str | Path | Sequence[str | Path] | None = 'workflows'
    """Directories of saved workflows in the run's workspace, or `None` to turn saved workflows off.

    Relative paths resolve against the workspace's working directory. Every `*.py` file directly
    inside is read at the start of each run and listed to the model in `<available_workflows>`.
    A directory that does not exist is an empty library, so having no workflow files is the
    opt-out for a run; a run without a workspace has none either. A file that fails to load is
    skipped with a warning. The model reads a saved file with a file tool such as `FileSystem`,
    not through this capability.

    The `save_workflow` tool writes to the first directory, creating it on first save, whenever
    the run's workspace is attached and writable. `None` is the off switch that stays off: no
    listing, no `run_workflow(name=...)`, no `workflow()` and no `save_workflow`.
    """

    default_agent: str | None = None
    """The sub-agent `agent(task)` runs when the script passes no `name`.

    `None` means the only sub-agent when there is one, and otherwise `name` is required.
    """

    max_concurrent_agents: int = 16
    """Most sub-agents running at once in one `run_workflow` call, nested workflows included.

    A script can pass any model name to `agent(model=...)`, so this and `max_agent_calls` are what
    bound the cost of a script the model wrote.
    """

    max_workflow_depth: int = 3
    """Most saved workflows running inside one another, counting one run by name as the first.

    A saved workflow that would run itself, directly or through another, is refused at any depth.
    """

    max_items_per_call: int = 4096
    """Most items one `parallel()` or `pipeline()` call accepts."""

    _library: WorkflowLibrary | None = field(init=False, repr=False)
    """This run's saved workflows, read in `for_run`; `None` when `workflows` is `None`."""

    @classmethod
    def get_serialization_name(cls) -> str | None:
        # Not spec-serializable: `agents` holds live Agent objects, not YAML-expressible config.
        return None

    def __post_init__(self) -> None:
        catalog = [self._normalize_workflow_agent(entry) for entry in self.agents]
        index_workflow_agents(catalog)
        self._catalog = catalog
        self._library = None if self.workflows is None else WorkflowLibrary()
        if self.workflows is not None and not self._directories:
            raise UserError(
                'DynamicWorkflow `workflows` needs at least one directory; pass `None` to turn saved workflows off.'
            )

    @property
    def _directories(self) -> tuple[str | Path, ...]:
        assert self.workflows is not None
        return (self.workflows,) if isinstance(self.workflows, (str, Path)) else tuple(self.workflows)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> DynamicWorkflow[AgentDepsT]:
        """Read this run's saved workflows from the run's workspace.

        Returns a copy holding them, so the listing is frozen for the run and the prompt cache
        stays stable. The sub-agent catalog is shared with this instance, so `reveal()` still
        reaches the run.
        """
        if self.workflows is None:
            return self
        run = copy.copy(self)
        workspace = ctx.workspace
        run._library = (
            await WorkflowLibrary.load(
                workspace,
                self._directories,
                agent_names={name for entry in self._catalog if (name := entry.resolved_name)},
            )
            if workspace.attached
            else WorkflowLibrary()
        )
        return run

    def get_instructions(self) -> str | None:
        """List this run's saved workflows in an `<available_workflows>` block, when there are any."""
        return None if self._library is None else self._library.render()

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
            default_agent=self.default_agent,
            max_concurrent_agents=self.max_concurrent_agents,
            max_workflow_depth=self.max_workflow_depth,
            max_items_per_call=self.max_items_per_call,
            library=self._library,
            save_directory=None if self.workflows is None else self._directories[0],
            toolset_id=self.id,
            owning_capability=self,
        )

"""Sub-agent toolset: a single delegate tool that runs named child agents."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Generic, cast

from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    RunCancelled,
    SkipModelRequest,
    SkipToolExecution,
    SkipToolValidation,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, ObjectJsonSchema, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

# Private import: pydantic-ai has no public way to tell capability-contributed
# toolsets apart from the agent's own in `agent.toolsets`.
from pydantic_ai.toolsets._capability_owned import CapabilityOwnedToolset
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.subagents._models import ModelOption, validate_restriction

logger = logging.getLogger(__name__)

_MODEL_ARG = 'model'
"""Name of the delegate tool's model-selection argument, shared by the function
signature and the schema rewrite that shapes it to the configured menu."""

# Signals that must always reach the parent run, even when a delegate has
# `contain_errors` on. Containing the first five would break the agent graph
# (deferred/approval/skip control-flow); a `UserError` is a setup bug that no
# retry can fix, so masking it into a retry only delays and obscures it.
# First-party cancellation (`RunContext.cancel()` in the child) raises
# `RunCancelled`, a plain `Exception`, so it must be listed here or containment
# would mislabel a deliberate stop as a crash and retry it. Once it escapes the
# delegate tool, pydantic-ai isolates it as a failed tool return in the parent
# (pydantic/pydantic-ai#7199), the same outcome as the uncontained path.
# External cancellation (`asyncio.CancelledError`, a `BaseException`) is out of
# `except Exception`'s reach already, and a shared `UsageLimitExceeded` has its
# own clause.
_ALWAYS_PROPAGATE: tuple[type[Exception], ...] = (
    CallDeferred,
    ApprovalRequired,
    SkipModelRequest,
    SkipToolValidation,
    SkipToolExecution,
    UserError,
    RunCancelled,
)


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

    models: Sequence[str] | None = None
    """Which of `SubAgents.models` this delegate may run on, as menu keys, and
    which one it runs on by default: the first key listed. Leave it unset to let
    the parent pick any configured option and to fall back to the delegate's own
    model when it picks none. Set it to pin a delegate to one option
    (`models=['fast']`) or to bound an expensive delegate to a subset. Naming a key
    the menu does not define is an error."""

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

    contain_errors: bool | None = None
    """Whether an unexpected sub-agent crash is contained instead of aborting the
    parent run. When `True`, an exception the child raises that is not an expected
    soft degradation (a provider `ModelAPIError`/`FallbackExceptionGroup`, a plain
    `ValueError` from a bad tool argument, etc.) is caught and returned to the parent
    as a bounded `ModelRetry`, so one delegate crash cannot kill the whole run. It
    stays loud: the exception rides the retry message and is logged, and
    `tool_retries` still bounds consecutive crashes into an abort. Cancellation, a
    shared usage-limit, pydantic-ai control-flow signals, and `UserError` always
    propagate regardless. Unset inherits `SubAgents.contain_errors` (default off).
    Orthogonal to `on_failure`, which only sets the message for expected soft
    degradations; a contained crash always raises the loud `ModelRetry`."""

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

    When a `models` menu is configured, the delegate tool takes an extra `model`
    argument carrying a menu key, so the parent routes each delegation to the
    model that fits the task. Without a menu the argument is not offered at all.
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
        tool_retries: int | None,
        contain_errors: bool,
        call_counts: dict[str, dict[str, int]],
        models: Mapping[str, ModelOption] | None = None,
    ) -> None:
        super().__init__()
        self._agents: dict[str, SubAgent[AgentDepsT]] = dict(agents)
        self._forward_usage = forward_usage
        self._inherit_tools = inherit_tools
        self._shared_capabilities = list(shared_capabilities)
        self._event_stream_handler = event_stream_handler
        self._tool_name = tool_name
        self._contain_errors = contain_errors
        self._models: dict[str, ModelOption] = dict(models or {})
        for name, sub_agent in self._agents.items():
            validate_restriction(name, sub_agent.models, self._models)
        # Run-scoped delegation counts, keyed by run_id then sub-agent name.
        # Shared with the capability, which clears each run's entry in wrap_run.
        self._call_counts = call_counts
        self.add_function(self.delegate_task, name=tool_name, retries=tool_retries, prepare=self._prepare_delegate)

    def _prepare_delegate(self, ctx: RunContext[AgentDepsT], tool_def: ToolDefinition) -> ToolDefinition:
        """Shape the delegate tool's `model` argument to the configured menu.

        With no menu the argument is dropped from the schema, so a capability that
        does not opt in exposes exactly the tool it did before. With a menu the
        argument becomes an enum of its keys, so the model picks from the offered
        options rather than inventing a model name. The result depends only on
        static configuration, which keeps the tool schema cache-stable.
        """
        schema: ObjectJsonSchema = {**tool_def.parameters_json_schema}
        properties: dict[str, object] = {**schema.get('properties', {})}
        if self._models:
            properties[_MODEL_ARG] = {
                'type': 'string',
                'enum': list(self._models),
                'description': (
                    'Which model to run the sub-agent on, as one of the keys listed in the instructions. '
                    "Omit it to use the sub-agent's default model."
                ),
            }
        else:
            properties.pop(_MODEL_ARG, None)
        schema['properties'] = properties
        return replace(tool_def, parameters_json_schema=schema)

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

    def _resolve_model(self, agent_name: str, sub_agent: SubAgent[AgentDepsT], key: str | None) -> ModelOption | None:
        """The menu option one delegation runs on, or `None` to leave the model as it was.

        An unset `key` falls back to the delegate's own first allowed option, and
        to no option at all when the delegate allows the whole menu.

        Raises:
            ModelRetry: the key is not on the menu, or not one this delegate allows.
        """
        allowed = sub_agent.models
        if key is None:
            return self._models[allowed[0]] if allowed else None
        option = self._models.get(key)
        if option is None:
            available = ', '.join(self._models) or '(none configured)'
            raise ModelRetry(f'Unknown model {key!r}. Available models: {available}.')
        if allowed is not None and key not in allowed:
            raise ModelRetry(
                f'Sub-agent {agent_name!r} cannot run on model {key!r}. Available models for it: {", ".join(allowed)}.'
            )
        return option

    async def delegate_task(
        self, ctx: RunContext[AgentDepsT], agent_name: str, task: str, model: str | None = None
    ) -> str:
        """Delegate a self-contained task to a named sub-agent and return its result.

        The sub-agent runs in its own fresh context and does not see this
        conversation, so `task` must contain everything it needs.

        Args:
            ctx: The run context (provides the parent's deps, usage, and tools).
            agent_name: Name of the sub-agent to run. Must be one of the agents
                listed in the instructions.
            task: The complete, self-contained instruction for the sub-agent.
            model: Which model to run the sub-agent on, as one of the model keys
                listed in the instructions. Omit it to use the sub-agent's default
                model. Only offered when a model menu is configured.
        """
        sub_agent = self._agents.get(agent_name)
        if sub_agent is None:
            available = ', '.join(sorted(self._agents))
            raise ModelRetry(f'Unknown sub-agent {agent_name!r}. Available sub-agents: {available}.')

        # Resolved before the call budget is charged, so a bad model key costs nothing.
        option = self._resolve_model(agent_name, sub_agent, model)

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

        # A selected menu option decides the model and how it runs. Without one, a
        # sub-agent with no model of its own (e.g. one loaded from disk) inherits the
        # parent run's model, and one that brought its own keeps it.
        run_model: Model | KnownModelName | str | None
        settings: ModelSettings | None
        if option is not None:
            run_model = option.model
            settings = option.settings
        else:
            # `ctx.model` is an `AbstractModel`; only a request-response `Model` can drive a
            # sub-agent run. When the parent run uses something else (a realtime model), fall
            # back to `None` so the sub-agent uses its own default rather than being handed a
            # model it cannot run with. Bind to a local, then `cast` to recover `Model[Any]`
            # from the generic `Model` (which `isinstance` narrows to `Model[Unknown]`),
            # mirroring core's own `reinject_system_prompt` idiom.
            ctx_model = ctx.model
            run_model = (
                cast('Model[Any]', ctx_model)
                if sub_agent.agent.model is None and isinstance(ctx_model, Model)
                else None
            )
            settings = None
        run = sub_agent.agent.run(
            task,
            deps=ctx.deps,
            model=run_model,
            model_settings=settings,
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
        except _ALWAYS_PROPAGATE:
            raise
        except Exception as exc:
            contain = sub_agent.contain_errors if sub_agent.contain_errors is not None else self._contain_errors
            if not contain:
                raise
            # Contain the crash so it cannot abort the parent, but keep it loud: the
            # exception rides the retry message and is logged, and `tool_retries`
            # bounds consecutive crashes into an abort.
            logger.warning('Contained crash from sub-agent %r', agent_name, exc_info=exc)
            raise ModelRetry(
                f'Sub-agent {agent_name!r} crashed: {type(exc).__name__}: {exc}. '
                f'Treat this as a recoverable failure and decide from existing evidence.'
            ) from exc
        return str(result.output)

    @staticmethod
    def _steer(on_failure: str | None, default: str) -> str:
        """A soft steering message: the delegate's `on_failure` override, else `default`."""
        if on_failure is not None:
            return on_failure
        return default

from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

from opentelemetry.trace import NoOpTracer
from typing_extensions import TypeVar

from pydantic_ai._run_context import CapabilityEventT, CustomEventT
from pydantic_ai.durable_exec._run_context import DurableRunContext
from pydantic_ai.durable_exec._utils import agent_instrumentation_settings
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent, CustomEvent
from pydantic_ai.tools import RunContext

from ._spec import TEMPORAL_ENGINE_SPEC

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

AgentDepsT = TypeVar('AgentDepsT', default=object, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""


class TemporalRunContext(DurableRunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

    By default, only the `deps`, `run_id`, `conversation_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `usage`, `usage_limits`, `deadline`, `partial_output`, `trace_include_content`, `instrumentation_version`, `loaded_capability_ids`, `discovered_tool_names`, the private dispatch-only availability supplements, and `capability_active` attributes will be available. Reading any other attribute (except `validation_context`, which is rebuilt as described below) raises a `UserError` explaining how to make it available, rather than returning its default value, so a field that didn't cross the boundary can't be mistaken for real run state.

    `agent` and `root_capability` are re-attached from the worker's agent instance, `pending_messages` holds a guard that makes [`enqueue`][pydantic_ai.tools.RunContext.enqueue] raise inside an activity, and `tool_manager` and `realtime_session` are `None`: they hold live run state that isn't serializable (for `tool_manager`, `available_tool_names` returns the resolved snapshot serialized at activity dispatch time, falling back to `discovered_tool_names` if a custom subclass doesn't carry it, and [`tool_name_for_kind`][pydantic_ai.tools.RunContext.tool_name_for_kind] answers from a kind-to-name snapshot taken at the same time; for `realtime_session`, `None` already means "not available here"). The `capabilities` registry is excluded for the same reason — it holds live capability objects (toolsets, hooks, callables) — so `active_capability_ids` likewise returns a snapshot serialized at dispatch time, which is what lets [`is_tool_available`][pydantic_ai.tools.RunContext.is_tool_available] answer for a capability-owned tool inside an activity; reading `capabilities` itself still raises. `model` is excluded as a live object too. `tracer` can't cross the boundary either, so it's resolved from the worker's agent instead: the tracer of its instrumentation settings when it's instrumented, otherwise the no-op tracer an uninstrumented run uses, so a tool that opens spans works inside an activity as it does in-process. A subclass whose `deserialize_run_context` sets `tracer` itself keeps that value. `messages` is excluded because the full history would be duplicated into every activity payload, and `prompt` is excluded because a multi-modal prompt can carry large `BinaryContent` that would likewise ride in every activity payload, risking Temporal's 2 MB limit. `model_settings` is excluded because it's only set for model requests, which receive it as their own activity parameter, and `validation_context` because it's an arbitrary user object with no serialization contract: it's rebuilt from the agent's `validation_context` setting the first time it's read. A live `workspace` cannot cross the activity boundary either: only its [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] is serialized, and the activity rebuilds `workspace` from it through the agent's capabilities (their `get_workspace`, which may read only `deps` and the fields listed here), policy wrappers included. A subclass whose `deserialize_run_context` sets `workspace` itself keeps that value.
    To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]. A subclass can use this escape hatch to opt in to carrying `prompt` if it knows its prompts are text-only.
    """

    def unavailable_field_message(self, name: str) -> str:
        return (
            f'{name!r} is not available on {self.__class__.__name__!r} inside a Temporal activity. '
            'To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to `TemporalDurability`.'
        )

    @overload
    async def emit(self, event: CustomEventT, /) -> CustomEventT: ...

    @overload
    async def emit(self, event: CapabilityEventT, /) -> CapabilityEventT: ...

    async def emit(self, event: CustomEvent | CapabilityEvent, /) -> CustomEvent | CapabilityEvent:
        """Reject `emit` from inside a Temporal activity.

        Tools and event stream handlers run inside activities where the run's event stream isn't
        reachable, so events emitted there can't currently flow back into the stream; raising beats
        silently dropping them. This covers a capability's own tools emitting a `CapabilityEvent`,
        which run in activities like any other tool. Events emitted workflow-side (e.g. from
        capability hooks) work as usual.

        Lifting this needs a transport from the activity back to the workflow and a decision on what
        a retried attempt's events mean; tracked in https://github.com/pydantic/pydantic-ai/issues/7971.
        """
        raise UserError(
            'Emitting events from a tool or event stream handler is not supported under Temporal yet, as '
            'they run inside activities that cannot reach the run event stream. This includes a capability '
            'emitting a `CapabilityEvent` from one of its own tools. Emit events from capability hooks, '
            'which run in the workflow, instead. Tracked in '
            'https://github.com/pydantic/pydantic-ai/issues/7971.'
        )

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> TemporalRunContext[Any]:
        """Deserialize the run context from a `dict[str, Any]`."""
        return cls(**ctx, deps=deps)


def deserialize_run_context(
    run_context_type: type[TemporalRunContext[Any]],
    serialized: dict[str, Any],
    *,
    deps: Any,
    agent: AbstractAgent[Any, Any] | None,
) -> RunContext[Any]:
    """Deserialize a run context inside a Temporal activity and attach the agent instance.

    See [`DurableRunContext.restore`][pydantic_ai.durable_exec.DurableRunContext.restore]; in
    addition, `tracer` is resolved from the worker's agent so custom subclasses don't need to
    know about the agent.
    """
    ctx = run_context_type.restore(serialized, deps=deps, agent=agent, engine_spec=TEMPORAL_ENGINE_SPEC)
    _restore_tracer(ctx, agent)
    return ctx


def _restore_tracer(ctx: RunContext[Any], agent: AbstractAgent[Any, Any] | None) -> None:
    """Give `ctx.tracer` the tracer the worker's agent would trace a run with.

    A run always has a tracer, a no-op one when it isn't instrumented, so tools open spans without checking
    for one. The live tracer can't cross the activity boundary, so it's resolved here the way model units
    resolve their instrumentation: from the worker's agent, which can't see an `Instrumentation` passed to a
    single `run()`. A tracer the subclass already restored is left alone.
    """
    if 'tracer' in ctx.__dict__:
        return
    settings = agent_instrumentation_settings(agent) if agent is not None else None
    ctx.__dict__['tracer'] = settings.tracer if settings is not None else NoOpTracer()
    assert isinstance(ctx, DurableRunContext)
    ctx._expose_field('tracer')  # pyright: ignore[reportPrivateUsage]

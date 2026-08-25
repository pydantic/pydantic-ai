from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter
from typing_extensions import TypeVar

from pydantic_ai._run_context import AnchoredEvidence
from pydantic_ai.durable_exec._toolset import EnqueueGuard, enqueue_not_supported_message
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

AgentDepsT = TypeVar('AgentDepsT', default=object, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""

# The serialized run context crosses the activity boundary as untyped JSON (`Any`, so
# `TemporalRunContext` subclasses can add their own fields), which means a value whose type isn't
# JSON-native arrives back in a different shape than it went in: a model or content object as a
# plain dict, a set as a list. Every carried field with such a type is listed here with the shape
# it arrives in and the adapter that turns it back into the real thing, so activity-side code gets
# the objects it would get in a non-durable run: `usage`/`usage_limits` drive the mid-chain
# continuation usage check, while the history-derived sets and dispatch-only supplements feed tool
# availability. Activity-side mutations are discarded; reveal deltas are applied to workflow state
# after the activity result returns.
_str_set_ta: TypeAdapter[set[str]] = TypeAdapter(set[str])
_REHYDRATORS: tuple[tuple[str, type[Any], TypeAdapter[Any]], ...] = (
    ('usage', dict, TypeAdapter(RunUsage)),
    ('usage_limits', dict, TypeAdapter(UsageLimits)),
    ('loaded_capability_ids', list, _str_set_ta),
    ('discovered_tool_names', list, _str_set_ta),
    ('available_tool_names', list, _str_set_ta),
    ('available_capability_ids', list, _str_set_ta),
    ('_deferred_capability_ids', list, _str_set_ta),
    ('_anchored_evidence', dict, TypeAdapter(AnchoredEvidence)),
)

# Fields that `serialize_run_context` doesn't carry but that are still readable inside an activity,
# as `None` unless the framework attaches something: `agent` and `root_capability` are re-attached
# from the worker's agent instance, `pending_messages` is replaced by an `EnqueueGuard` so
# `ctx.enqueue()` raises an explanation, and `tool_manager` is documented to be `None` inside
# activities — `available_tool_names` then reads the snapshot serialized at dispatch time (see
# the property override below), or falls back to `discovered_tool_names` without one.
# `realtime_session` is a live session object that cannot cross the boundary, and its contract
# already makes `None` mean "not available here".
_NONE_UNLESS_ATTACHED = ('agent', 'root_capability', 'pending_messages', 'tool_manager', 'realtime_session')

# Defaulted rather than guarded when a payload doesn't carry it. Unlike the guarded fields, the
# dataclass default can't be mistaken for real run state here: empty means "no anchored evidence",
# which is exactly what `is_tool_available` reads when the serving response has no provenance. A
# custom `serialize_run_context` written before this field existed therefore keeps answering — with
# the history-derived window — instead of raising for a field it never knew to carry.
_DEFAULTED_UNLESS_CARRIED: tuple[tuple[str, Any], ...] = (('_anchored_evidence', AnchoredEvidence()),)

# Reading any other omitted field raises instead of returning the `RunContext` dataclass default,
# which would silently pass for real run state (e.g. `instrumentation_version` reading as the
# default version rather than the run's, or `prompt` as `None` for a subclass that drops it).
_GUARDED_FIELDS = frozenset(RunContext.__dataclass_fields__) - {'deps', *_NONE_UNLESS_ATTACHED}


class TemporalRunContext(RunContext[AgentDepsT]):
    """The [`RunContext`][pydantic_ai.tools.RunContext] subclass to use to serialize and deserialize the run context for use inside a Temporal activity.

    By default, only the `deps`, `run_id`, `conversation_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`, `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `usage`, `usage_limits`, `partial_output`, `trace_include_content`, `instrumentation_version`, `loaded_capability_ids`, `discovered_tool_names`, the private dispatch-only availability supplements, and `capability_loaded` attributes will be available. Reading any other attribute raises a `UserError` explaining how to make it available, rather than returning its default value, so a field that didn't cross the boundary can't be mistaken for real run state.

    `agent` and `root_capability` are re-attached from the worker's agent instance, `pending_messages` holds a guard that makes [`enqueue`][pydantic_ai.tools.RunContext.enqueue] raise inside an activity, and `tool_manager` and `realtime_session` are `None`: they hold live run state that isn't serializable (for `tool_manager`, `available_tool_names` returns the resolved snapshot serialized at activity dispatch time, falling back to `discovered_tool_names` if a custom subclass doesn't carry it; for `realtime_session`, `None` already means "not available here"). The `capabilities` registry is excluded for the same reason — it holds live capability objects (toolsets, hooks, callables) — so `available_capability_ids` likewise returns a snapshot serialized at dispatch time, which is what lets [`is_tool_available`][pydantic_ai.tools.RunContext.is_tool_available] answer for a capability-owned tool inside an activity; reading `capabilities` itself still raises. `model` and `tracer` are excluded as live objects too. `messages` is excluded because the full history would be duplicated into every activity payload, and `prompt` is excluded because a multi-modal prompt can carry large `BinaryContent` that would likewise ride in every activity payload, risking Temporal's 2 MB limit. `model_settings` is excluded because it's only set for model requests, which receive it as their own activity parameter, and `validation_context` because it's an arbitrary user object with no serialization contract.
    To make another attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]. A subclass can use this escape hatch to opt in to carrying `prompt` if it knows its prompts are text-only.
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        self.__dict__ = {**kwargs, 'deps': deps}
        for name in _NONE_UNLESS_ATTACHED:
            self.__dict__.setdefault(name, None)
        for name, default in _DEFAULTED_UNLESS_CARRIED:
            self.__dict__.setdefault(name, default)
        for name, wire_type, adapter in _REHYDRATORS:
            if isinstance(value := self.__dict__.get(name), wire_type):
                self.__dict__[name] = adapter.validate_python(value)
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in self.__dict__},
        )

    def __getattribute__(self, name: str) -> Any:
        if name in _GUARDED_FIELDS and name not in object.__getattribute__(self, '__dataclass_fields__'):
            raise UserError(
                f'{name!r} is not available on {self.__class__.__name__!r} inside a Temporal activity. '
                'To make the attribute available, create a `TemporalRunContext` subclass with a custom `serialize_run_context` class method that returns a dictionary that includes the attribute and pass it as the `run_context_type` argument to `TemporalDurability`.'
            )
        return super().__getattribute__(name)

    @property
    def available_tool_names(self) -> set[str]:
        """The availability snapshot serialized at activity dispatch time.

        Live tool state doesn't cross the activity boundary, but availability was already
        resolved when the activity was dispatched — so the name form of
        [`is_tool_available`][pydantic_ai.tools.RunContext.is_tool_available] answers correctly
        for always-visible tools too, instead of degrading to the `discovered_tool_names`
        fallback. Custom subclasses whose `serialize_run_context` doesn't carry the snapshot
        keep the base fallback behavior.
        """
        if (snapshot := self.__dict__.get('available_tool_names')) is not None:
            return snapshot
        return super().available_tool_names

    @property
    def available_capability_ids(self) -> set[str]:
        """The set of active capability ids serialized at activity dispatch time.

        The `capabilities` registry itself can't cross the boundary, but the ids it resolves to
        can, so [`is_tool_available`][pydantic_ai.tools.RunContext.is_tool_available] still
        answers for a capability-owned tool instead of raising. Custom subclasses whose
        `serialize_run_context` doesn't carry the snapshot fall back to the base property, which
        reads the registry and raises inside an activity.
        """
        if (snapshot := self.__dict__.get('available_capability_ids')) is not None:
            return snapshot
        return super().available_capability_ids

    @property
    def _deferred_capability_ids(self) -> set[str]:
        """The set of on-demand capability ids serialized at activity dispatch time.

        `is_tool_available` needs the *configured* shape of a capability, not just what history says
        was loaded, and reads it from the registry — which cannot cross the boundary. Carrying the
        ids keeps a loaded capability's own tools answering as available inside an activity instead
        of falling back to a reveal marker that, for these tools, nothing can regenerate. Custom
        subclasses whose `serialize_run_context` omits the snapshot fall back to the base property,
        which reads the registry and raises inside an activity.
        """
        if (snapshot := self.__dict__.get('_deferred_capability_ids')) is not None:
            return snapshot
        return super()._deferred_capability_ids

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context to a `dict[str, Any]`."""
        return {
            'run_id': ctx.run_id,
            'conversation_id': ctx.conversation_id,
            'metadata': ctx.metadata,
            'retries': ctx.retries,
            'tool_call_id': ctx.tool_call_id,
            'tool_name': ctx.tool_name,
            'tool_call_approved': ctx.tool_call_approved,
            'tool_call_metadata': ctx.tool_call_metadata,
            'retry': ctx.retry,
            'max_retries': ctx.max_retries,
            'run_step': ctx.run_step,
            'partial_output': ctx.partial_output,
            'trace_include_content': ctx.trace_include_content,
            'instrumentation_version': ctx.instrumentation_version,
            'usage': ctx.usage,
            'usage_limits': ctx.usage_limits,
            'loaded_capability_ids': ctx.loaded_capability_ids,
            'discovered_tool_names': ctx.discovered_tool_names,
            # The dispatch-time widening of the two sets above, which `is_tool_available` reads for
            # a call the model has already made. Carried so a tool asking whether it may run gets
            # the same answer inside an activity as it would in-process.
            '_anchored_evidence': ctx._anchored_evidence,
            # A resolved snapshot: at dispatch time live tool state exists, so this carries the
            # always-visible tools that the in-activity `discovered_tool_names` fallback misses.
            'available_tool_names': ctx.available_tool_names,
            # Likewise a snapshot rather than the registry: these ids are plain strings, while the
            # capability objects they key are not serializable. `is_tool_available` consults this
            # for any capability-owned tool, so without it the definition form — the form the docs
            # send toolset authors to — raises inside an activity instead of answering.
            'available_capability_ids': ctx.available_capability_ids,
            # The configured on-demand set, which `is_tool_available` consults to tell a loaded
            # deferred capability (whose load is itself the reveal for its tools) from one that has
            # since been reconfigured always-on. Derived from the registry, so it must travel too.
            '_deferred_capability_ids': ctx._deferred_capability_ids,
            'capability_loaded': ctx.capability_loaded,
        }

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
    """Deserialize a run context and attach the agent instance.

    This is a helper used internally by the Temporal wrappers. It calls the
    (potentially user-overridden) `TemporalRunContext.deserialize_run_context`
    and then sets `agent` and `root_capability` on the result so custom subclasses
    don't need to know about either parameter. Setting `root_capability` lets the
    durability capability fire the capability chain against the live model stream
    inside the activity, which is required for capabilities like
    `ProcessEventStream` to see real (non-replayed) events.
    """
    ctx = run_context_type.deserialize_run_context(serialized, deps=deps)
    if agent is not None:
        ctx.__dict__['agent'] = agent
        ctx.__dict__['root_capability'] = agent.root_capability
    # `pending_messages` isn't serialized across the activity boundary, and any code running inside
    # an activity (a tool, a `process_tool_call` hook, an `event_stream_handler`) is in a durable
    # unit whose result is replayed without re-running it, so an enqueue would be dropped. Install
    # the same guard the in-process engines use so `ctx.enqueue()` raises the shared explanation.
    ctx.__dict__['pending_messages'] = EnqueueGuard(enqueue_not_supported_message('activity', 'workflow'))
    return ctx

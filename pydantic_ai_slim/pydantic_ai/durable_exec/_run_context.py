from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, overload

from pydantic import TypeAdapter
from typing_extensions import TypeVar

from pydantic_ai import messages as _messages
from pydantic_ai._run_context import AnchoredEvidence, CapabilityEventT, CustomEventT, ToolNamePath
from pydantic_ai.capabilities.abstract import select_workspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent, CustomEvent, ToolKindLike
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import Workspace, WorkspaceRef
from pydantic_ai.workspaces.unavailable import NO_WORKSPACE, UnavailableWorkspace

from ._toolset import DispatchSnapshots, EnqueueGuard, enqueue_not_supported_message, validation_context_from_agent

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent

    from ._spec import DurabilityEngineSpec

__all__ = ('DurableRunContext',)

AgentDepsT = TypeVar('AgentDepsT', default=object, covariant=True)
"""Type variable for the agent dependencies in `RunContext`."""


# The serialized run context crosses the durable unit boundary as an untyped mapping (`Any`, so
# subclasses can add their own fields), which means a value whose type isn't JSON-native arrives
# back in a different shape than it went in: a model or content object as a plain dict, a set as a
# list. Every carried field with such a type is listed here with the shape it arrives in and the
# adapter that turns it back into the real thing, so unit-side code gets the objects it would get
# in a non-durable run: `usage`/`usage_limits` drive the mid-chain continuation usage check, while
# the history-derived sets and dispatch-only supplements feed tool availability. Unit-side
# mutations are discarded; reveal deltas are applied to container state after the unit returns.
_str_set_ta: TypeAdapter[set[str]] = TypeAdapter(set[str])
_REHYDRATORS: tuple[tuple[str, type[Any], TypeAdapter[Any]], ...] = (
    ('usage', dict, TypeAdapter(RunUsage)),
    ('usage_limits', dict, TypeAdapter(UsageLimits)),
    ('deadline', str, TypeAdapter(datetime)),
    ('loaded_capability_ids', list, _str_set_ta),
    ('discovered_tool_names', list, _str_set_ta),
    ('available_tool_names', list, _str_set_ta),
    ('active_capability_ids', list, _str_set_ta),
    ('_deferred_capability_ids', list, _str_set_ta),
    ('_anchored_evidence', dict, TypeAdapter(AnchoredEvidence)),
    ('_tool_name_path', list, TypeAdapter(ToolNamePath)),
    # Not a `RunContext` field: the ref rides alongside so the unit can rebuild `workspace`.
    ('workspace_ref', dict, TypeAdapter(WorkspaceRef)),
)

# Fields that `serialize_run_context` doesn't carry but that are still readable inside a unit, as
# `None` unless the framework attaches something: `agent` and `root_capability` are re-attached
# from the worker's agent instance by `restore`, `pending_messages` is replaced by an `EnqueueGuard`
# so `ctx.enqueue()` raises an explanation, and `tool_manager` is documented to be `None` inside a
# unit — `available_tool_names` then reads the snapshot serialized at dispatch time (see the
# property override below), or falls back to `discovered_tool_names` without one.
# `realtime_session` is a live session object that cannot cross the boundary, and its contract
# already makes `None` mean "not available here". `_run_held_toolsets` is the same: a unit may run
# in another process, so it enters its own toolset instead of reusing the one the run holds.
_NONE_UNLESS_ATTACHED = (
    'agent',
    'root_capability',
    'pending_messages',
    'tool_manager',
    'realtime_session',
    '_durable_operations',
    '_run_capabilities_by_id',
    '_run_held_toolsets',
)

# Reading any other omitted field raises instead of returning the `RunContext` dataclass default,
# which would silently pass for real run state (e.g. `instrumentation_version` reading as the
# default version rather than the run's, or `prompt` as `None` for a subclass that drops it).
_GUARDED_FIELDS = frozenset(RunContext.__dataclass_fields__) - {'deps', *_NONE_UNLESS_ATTACHED}

# Defaulted rather than guarded when a payload doesn't carry it. Unlike the guarded fields, the
# dataclass default can't be mistaken for real run state here: empty means "no anchored evidence",
# which is exactly what `is_tool_available` reads when the serving response has no provenance. A
# custom `serialize_run_context` written before this field existed therefore keeps answering — with
# the history-derived window — instead of raising for a field it never knew to carry.
# `deadline` is defaulted for the same reason: a unit dispatched by a worker that predates it, or a
# custom `serialize_run_context` that doesn't carry it, sees no deadline, so an agent its tool runs
# just doesn't inherit one, rather than the tool failing on the read.
# `_tool_name_path` is serialized only when a name-changing wrapper set it, and empty is what a
# listing outside one carries: a payload without it lists like a run without wrappers did.
_DEFAULTED_UNLESS_CARRIED: tuple[tuple[str, Any], ...] = (
    ('_anchored_evidence', AnchoredEvidence()),
    ('deadline', None),
    ('_tool_name_path', ()),
)

# Payloads written by a worker running an older version, or by a custom `serialize_run_context` that
# still spells the old name. A unit can be dispatched by one worker version and replayed by another,
# so the old key has to keep arriving at the renamed field — otherwise the field lands in `__dict__`
# under a name nothing reads, and `capability_active` raises as "not carried".
_RENAMED_FIELDS: tuple[tuple[str, str], ...] = (
    ('capability_loaded', 'capability_active'),
    ('available_capability_ids', 'active_capability_ids'),
)


class DurableRunContext(RunContext[AgentDepsT]):
    """A [`RunContext`][pydantic_ai.tools.RunContext] rebuilt inside a durable unit that runs outside the run's process.

    Engines whose durable units (activities, tasks) can run on another worker serialize the run
    context at dispatch with [`serialize_run_context`][pydantic_ai.durable_exec.DurableRunContext.serialize_run_context]
    and rebuild it worker-side with [`restore`][pydantic_ai.durable_exec.DurableRunContext.restore].

    By default, only the `deps`, `run_id`, `conversation_id`, `metadata`, `retries`, `tool_call_id`, `tool_name`,
    `tool_call_approved`, `tool_call_metadata`, `retry`, `max_retries`, `run_step`, `usage`, `usage_limits`,
    `deadline`, `partial_output`, `trace_include_content`, `instrumentation_version`, `loaded_capability_ids`,
    `discovered_tool_names`, the private dispatch-only availability supplements, and `capability_active`
    attributes are carried. Reading any other field raises a `UserError` rather than returning its default
    value, so a field that didn't cross the boundary can't be mistaken for real run state.

    `agent` and `root_capability` are re-attached from the worker's agent instance, `pending_messages`
    holds a guard that makes [`enqueue`][pydantic_ai.tools.RunContext.enqueue] raise, and `tool_manager`
    and `realtime_session` are `None`: they hold live run state that isn't serializable. Tool availability
    (`available_tool_names`, `active_capability_ids`, [`tool_name_for_kind`][pydantic_ai.tools.RunContext.tool_name_for_kind])
    answers from snapshots taken at dispatch time. A live `workspace` cannot cross the boundary either:
    only its [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef] is serialized, and `restore` rebuilds
    `workspace` from it through the agent's capabilities. `validation_context` isn't serialized, because it's an
    arbitrary user object; it's rebuilt from the agent's `validation_context` setting the first time
    it's read.

    Engines subclass this to word errors for their own runtime. Users of an engine subclass that to carry
    more fields: override `serialize_run_context` to add keys to the mapping, which reach `__init__` as
    keyword arguments and become readable attributes.
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        kwargs.setdefault('workspace', Workspace(NO_WORKSPACE))
        self.__dict__ = {**kwargs, 'deps': deps}
        for old_name, new_name in _RENAMED_FIELDS:
            # Keyed on presence, not truthiness: `capability_active` is `None` for every unit
            # dispatched outside capability dispatch — the common case — and a value-based guard
            # would drop the key there, leaving the renamed field absent and the guard below
            # reporting it as one that never crossed the boundary.
            if old_name in self.__dict__:
                self.__dict__.setdefault(new_name, self.__dict__.pop(old_name))
        for name in _NONE_UNLESS_ATTACHED:
            self.__dict__.setdefault(name, None)
        for name, default in _DEFAULTED_UNLESS_CARRIED:
            self.__dict__.setdefault(name, default)
        for name, wire_type, adapter in _REHYDRATORS:
            if isinstance(value := self.__dict__.get(name), wire_type):
                self.__dict__[name] = adapter.validate_python(value)
        # `dataclasses.replace` rebuilds a context from its instance fields, so only the carried
        # fields (plus the dispatch snapshots) are listed: a copy made inside the unit keeps them.
        setattr(
            self,
            '__dataclass_fields__',
            {
                name: field
                for name, field in (RunContext.__dataclass_fields__ | DispatchSnapshots.__dataclass_fields__).items()
                if name in self.__dict__
            },
        )

    def __getattribute__(self, name: str) -> Any:
        if name in _GUARDED_FIELDS and name not in object.__getattribute__(self, '__dataclass_fields__'):
            if name == 'validation_context' and (agent := object.__getattribute__(self, '__dict__').get('agent')):
                # Not carried, because it's an arbitrary user object: rebuilt on first read from the
                # agent's `validation_context` setting, so a resolver that reads a field this context
                # doesn't carry fails only where it's actually used. Without an agent there is no
                # setting to rebuild from, so it stays guarded like any other uncarried field.
                value = validation_context_from_agent(agent)(self)
                self.__dict__['validation_context'] = value
                # A shallow copy shares the instance field map: give this context its own before
                # exposing the field, so a copy it was made from (or that was made from it) keeps
                # rebuilding its own value instead of reading the class default.
                self.__dict__['__dataclass_fields__'] = dict(object.__getattribute__(self, '__dataclass_fields__'))
                self._expose_field('validation_context')
                return value
            raise UserError(self.unavailable_field_message(name))
        return super().__getattribute__(name)

    def unavailable_field_message(self, name: str) -> str:
        """The error raised when user code reads a field that didn't cross the boundary.

        Engines override this to name their durable unit and how to carry more fields.
        """
        return (
            f'{name!r} is not available on {self.__class__.__name__!r} inside a durable unit. '
            f'To make the attribute available, create a `{self.__class__.__name__}` subclass with a custom '
            '`serialize_run_context` class method that returns a mapping that includes the attribute.'
        )

    def _expose_field(self, name: str) -> None:
        """Mark a framework-attached field as readable after deserialization."""
        instance_fields = object.__getattribute__(self, '__dataclass_fields__')
        instance_fields[name] = RunContext.__dataclass_fields__[name]

    @property
    def available_tool_names(self) -> set[str]:
        """The availability snapshot serialized at dispatch time.

        Live tool state doesn't cross the boundary, but availability was already resolved when the
        unit was dispatched — so the name form of
        [`is_tool_available`][pydantic_ai.tools.RunContext.is_tool_available] answers correctly
        for always-visible tools too, instead of degrading to the `discovered_tool_names`
        fallback. Custom subclasses whose `serialize_run_context` doesn't carry the snapshot
        keep the base fallback behavior.
        """
        if (snapshot := self.__dict__.get('available_tool_names')) is not None:
            return snapshot
        return super().available_tool_names

    def tool_name_for_kind(self, kind: ToolKindLike) -> str | None:
        """The name of the tool with this kind, from the snapshot serialized at dispatch time.

        The tool manager doesn't cross the boundary, but the step's tools were resolved when the unit
        was dispatched, so a tool result built in the unit can still name another tool by kind.
        Custom subclasses whose `serialize_run_context` doesn't carry the snapshot keep the base
        behavior, which returns `None` without a tool manager.
        """
        if (snapshot := self.__dict__.get('_tool_names_by_kind')) is not None:
            return snapshot.get(_messages._tool_kind_id(kind))  # pyright: ignore[reportPrivateUsage]
        return super().tool_name_for_kind(kind)

    @property
    def active_capability_ids(self) -> set[str]:
        """The set of active capability ids serialized at dispatch time.

        The `capabilities` registry itself can't cross the boundary, but the ids it resolves to
        can, so [`is_tool_available`][pydantic_ai.tools.RunContext.is_tool_available] still
        answers for a capability-owned tool instead of raising. Custom subclasses whose
        `serialize_run_context` doesn't carry the snapshot fall back to the base property, which
        reads the registry and raises inside a unit.
        """
        if (snapshot := self.__dict__.get('active_capability_ids')) is not None:
            return snapshot
        return super().active_capability_ids

    @property
    def _deferred_capability_ids(self) -> set[str]:
        """The set of on-demand capability ids serialized at dispatch time.

        `is_tool_available` needs the *configured* shape of a capability, not just what history says
        was loaded, and reads it from the registry — which cannot cross the boundary. Carrying the
        ids keeps a loaded capability's own tools answering as available inside a unit instead of
        falling back to a reveal marker that, for these tools, nothing can regenerate. Custom
        subclasses whose `serialize_run_context` omits the snapshot fall back to the base property,
        which reads the registry and raises inside a unit.
        """
        if (snapshot := self.__dict__.get('_deferred_capability_ids')) is not None:
            return snapshot
        return super()._deferred_capability_ids

    @overload
    async def emit(self, event: CustomEventT, /) -> CustomEventT: ...

    @overload
    async def emit(self, event: CapabilityEventT, /) -> CapabilityEventT: ...

    async def emit(self, event: CustomEvent | CapabilityEvent, /) -> CustomEvent | CapabilityEvent:
        """Reject `emit` from inside a durable unit, where the run's event stream isn't reachable.

        Raising beats silently dropping the event. Engines that can carry events back to the
        container override this.
        """
        raise UserError(
            'Emitting events from a tool or event stream handler is not supported inside a durable unit that runs '
            'outside the run, as it cannot reach the run event stream. Emit events from capability hooks instead.'
        )

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Serialize the run context to a `dict[str, Any]`.

        The values are Python objects (`RunUsage`, sets, `datetime`): engines encode the mapping
        with their own codec, and `__init__` turns their JSON shapes back into the real types.
        """
        serialized: dict[str, Any] = {
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
            'deadline': ctx.deadline,
            'loaded_capability_ids': ctx.loaded_capability_ids,
            'discovered_tool_names': ctx.discovered_tool_names,
            # The dispatch-time widening of the two sets above, which `is_tool_available` reads for
            # a call the model has already made. Carried so a tool asking whether it may run gets
            # the same answer inside a unit as it would in-process.
            '_anchored_evidence': ctx._anchored_evidence,
            # A resolved snapshot: at dispatch time live tool state exists, so this carries the
            # always-visible tools that the in-unit `discovered_tool_names` fallback misses.
            'available_tool_names': ctx.available_tool_names,
            # What `tool_name_for_kind` answers from inside the unit, resolved while the step's
            # tools are live: the first tool of each kind, in the order `ctx.tools` lists them.
            '_tool_names_by_kind': _tool_names_by_kind(ctx),
            # Likewise a snapshot rather than the registry: these ids are plain strings, while the
            # capability objects they key are not serializable. `is_tool_available` consults this
            # for any capability-owned tool, so without it the definition form — the form the docs
            # send toolset authors to — raises inside a unit instead of answering.
            'active_capability_ids': ctx.active_capability_ids,
            # The configured on-demand set, which `is_tool_available` consults to tell a loaded
            # deferred capability (whose load is itself the reveal for its tools) from one that has
            # since been reconfigured always-on. Derived from the registry, so it must travel too.
            '_deferred_capability_ids': ctx._deferred_capability_ids,
            'capability_active': ctx.capability_active,
        }
        # Only the reference crosses into the unit; the live handle stays in the container, and
        # `__init__` supplies the inert default when no reference was serialized.
        if (workspace_ref := ctx.workspace.ref) is not None:
            serialized['workspace_ref'] = workspace_ref
        # Set while a dynamic toolset under a name-changing wrapper is listed, so a `prepare` that
        # runs in the unit reads its tool's own retry count.
        if tool_name_path := ctx._tool_name_path:
            serialized['_tool_name_path'] = tool_name_path
        return serialized

    @classmethod
    def deserialize_run_context(cls, ctx: dict[str, Any], deps: Any) -> DurableRunContext[Any]:
        """Deserialize the run context from a `dict[str, Any]`."""
        return cls(**ctx, deps=deps)

    @classmethod
    def restore(
        cls,
        serialized: Mapping[str, Any],
        *,
        deps: Any,
        agent: AbstractAgent[Any, Any] | None,
        engine_spec: DurabilityEngineSpec,
    ) -> RunContext[Any]:
        """Rebuild a run context worker-side and attach the worker's agent.

        Calls the (potentially user-overridden) `deserialize_run_context`, then sets `agent` and
        `root_capability` so custom subclasses don't need to know about either. Setting
        `root_capability` lets the durability capability fire the capability chain against the live
        model stream inside the unit. The workspace is rebuilt from its serialized
        [`WorkspaceRef`][pydantic_ai.workspaces.WorkspaceRef], and `pending_messages` gets a guard so
        `ctx.enqueue()` raises the shared explanation worded with `engine_spec`'s nouns.
        `ctx.validation_context` is rebuilt from the agent's `validation_context` setting the first
        time it's read.
        """
        ctx = cls.deserialize_run_context(dict(serialized), deps=deps)
        if agent is not None:
            ctx.__dict__['agent'] = agent
            ctx.__dict__['root_capability'] = agent.root_capability
            _restore_workspace(ctx, agent, engine_spec)
        # `pending_messages` isn't serialized across the boundary, and any code running inside a unit
        # (a tool, a `process_tool_call` hook, an `event_stream_handler`) has its result replayed
        # without re-running it, so an enqueue would be dropped.
        ctx.__dict__['pending_messages'] = EnqueueGuard(
            enqueue_not_supported_message(engine_spec.durable_unit_noun, engine_spec.durable_container_noun)
        )
        return ctx


def _restore_workspace(ctx: RunContext[Any], agent: AbstractAgent[Any, Any], engine_spec: DurabilityEngineSpec) -> None:
    """Rebuild `ctx.workspace` from the serialized `WorkspaceRef` through the worker's capabilities.

    The same selection the container made, minus the durable wrapper: the unit is the durable
    unit, so its calls reach the backend directly. Policy the capabilities apply (a
    `ReadOnlyWorkspace`, a user wrapper) comes back with it, which is why the ref alone crosses. A
    workspace the subclass already restored is left alone, and a payload without a ref keeps the
    placeholder: the run has no workspace, or a fresh one whose `ensure` unit is the one building it.
    """
    workspace = ctx.__dict__.get('workspace')
    ref = ctx.__dict__.get('workspace_ref')
    if not isinstance(workspace, Workspace) or workspace.backend is not NO_WORKSPACE:
        return
    if not isinstance(ref, WorkspaceRef):
        return
    restored = select_workspace(agent.root_capability, ctx, ref=ref)
    if restored is None:
        restored = Workspace(
            UnavailableWorkspace(
                f'No capability on agent {agent.name!r} can supply workspace {ref.id!r} from provider '
                f'{ref.provider!r} inside this {engine_spec.engine_name} {engine_spec.durable_unit_noun}: every '
                f'`get_workspace` returned `None`. The worker must be constructed with the same workspace '
                f'capabilities as the {engine_spec.durable_container_noun}.'
            )
        )
    ctx.__dict__['workspace'] = restored


def _tool_names_by_kind(ctx: RunContext[Any]) -> dict[str, str]:
    names: dict[str, str] = {}
    for name, tool_def in ctx.tools.items():
        if tool_def.tool_kind is not None:
            names.setdefault(tool_def.tool_kind, name)
    return names

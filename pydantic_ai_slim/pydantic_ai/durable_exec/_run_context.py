"""Shared run-context projection for durable operations in another process."""

from __future__ import annotations

from typing import Any

from pydantic import TypeAdapter
from typing_extensions import TypeVar

from pydantic_ai._run_context import AnchoredEvidence
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import Workspace, WorkspaceRef
from pydantic_ai.workspaces.unavailable import NO_WORKSPACE

AgentDepsT = TypeVar('AgentDepsT', default=object, covariant=True)

_SET_ADAPTER: TypeAdapter[set[str]] = TypeAdapter(set[str])
_REHYDRATORS: tuple[tuple[str, type[Any], TypeAdapter[Any]], ...] = (
    ('usage', dict, TypeAdapter(RunUsage)),
    ('usage_limits', dict, TypeAdapter(UsageLimits)),
    ('loaded_capability_ids', list, _SET_ADAPTER),
    ('discovered_tool_names', list, _SET_ADAPTER),
    ('available_tool_names', list, _SET_ADAPTER),
    ('active_capability_ids', list, _SET_ADAPTER),
    ('_deferred_capability_ids', list, _SET_ADAPTER),
    ('_anchored_evidence', dict, TypeAdapter(AnchoredEvidence)),
    ('workspace_ref', dict, TypeAdapter(WorkspaceRef)),
)

# These live objects are attached by the engine, or read as None until attachment.
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
_GUARDED_FIELDS = frozenset(RunContext.__dataclass_fields__) - {'deps', *_NONE_UNLESS_ATTACHED}


class SerializedRunContext(RunContext[AgentDepsT]):
    """A restricted `RunContext` rebuilt from JSON-shaped durable-operation data.

    Engines can subclass this to attach worker-local state and to customize the error
    for a field that was not carried. The projection deliberately omits live models,
    history, capability objects, and arbitrary validation context. Engine-specific
    fields and legacy payload aliases belong in the engine subclass.
    """

    def __init__(self, deps: AgentDepsT, **kwargs: Any):
        kwargs.setdefault('workspace', Workspace(NO_WORKSPACE))
        self.__dict__ = {**kwargs, 'deps': deps}
        for name in _NONE_UNLESS_ATTACHED:
            self.__dict__.setdefault(name, None)
        # Older serializers may omit this field. Empty evidence is the safe fallback.
        self.__dict__.setdefault('_anchored_evidence', AnchoredEvidence())
        for name, wire_type, adapter in _REHYDRATORS:
            if isinstance(value := self.__dict__.get(name), wire_type):
                self.__dict__[name] = adapter.validate_python(value)
        # RunContext has class-level dataclass defaults. Limit the visible instance
        # fields so an omitted value cannot be mistaken for real run state.
        setattr(
            self,
            '__dataclass_fields__',
            {name: field for name, field in RunContext.__dataclass_fields__.items() if name in self.__dict__},
        )

    def __getattribute__(self, name: str) -> Any:
        if name in _GUARDED_FIELDS and name not in object.__getattribute__(self, '__dataclass_fields__'):
            raise UserError(type(self)._missing_field_message(name))
        return super().__getattribute__(name)

    @classmethod
    def _missing_field_message(cls, name: str) -> str:
        return f'{name!r} is not available on {cls.__name__!r} inside a durable operation.'

    @property
    def available_tool_names(self) -> set[str]:
        if (snapshot := self.__dict__.get('available_tool_names')) is not None:
            return snapshot
        return super().available_tool_names

    @property
    def active_capability_ids(self) -> set[str]:
        if (snapshot := self.__dict__.get('active_capability_ids')) is not None:
            return snapshot
        return super().active_capability_ids

    @property
    def _deferred_capability_ids(self) -> set[str]:
        if (snapshot := self.__dict__.get('_deferred_capability_ids')) is not None:
            return snapshot
        return super()._deferred_capability_ids

    @classmethod
    def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
        """Project run state shared by remote durable-operation engines."""
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
            '_anchored_evidence': ctx._anchored_evidence,
            'available_tool_names': ctx.available_tool_names,
            'active_capability_ids': ctx.active_capability_ids,
            '_deferred_capability_ids': ctx._deferred_capability_ids,
            'capability_active': ctx.capability_active,
        }

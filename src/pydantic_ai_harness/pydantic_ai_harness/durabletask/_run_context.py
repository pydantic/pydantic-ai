from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, TypeVar, overload

from pydantic import Field, JsonValue, TypeAdapter
from pydantic_core import to_jsonable_python

from pydantic_ai._run_context import AnchoredEvidence, ToolNamePath
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import select_workspace
from pydantic_ai.durable_exec._toolset import DispatchSnapshots, EnqueueGuard, enqueue_not_supported_message
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CapabilityEvent, CustomEvent
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import Workspace, WorkspaceRef
from pydantic_ai.workspaces.unavailable import NO_WORKSPACE

DepsT = TypeVar('DepsT')
CustomEventT = TypeVar('CustomEventT', bound=CustomEvent)
CapabilityEventT = TypeVar('CapabilityEventT', bound=CapabilityEvent)
_JSON = TypeAdapter[JsonValue](JsonValue)
__all__ = ('restore_context',)


class _DepsProxy:
    def __init__(self, deps_input: JsonValue) -> None:
        self.deps_input = deps_input

    def __getattr__(self, name: str) -> object:
        raise UserError(
            f'Dependency {name!r} is only available inside a Durable Task activity. '
            'Use deterministic `deps_input` in orchestrator-side code instead.'
        )


@dataclass(kw_only=True)
class _ContextSnapshot:
    run_id: str | None
    conversation_id: str | None
    metadata: dict[str, Any] | None
    retries: dict[str, int]
    tool_call_id: str | None
    tool_name: str | None
    tool_call_approved: bool
    tool_call_metadata: Any
    retry: int
    max_retries: int
    run_step: int
    partial_output: bool
    trace_include_content: bool
    instrumentation_version: int
    usage: RunUsage
    usage_limits: UsageLimits | None
    deadline: datetime | None
    loaded_capability_ids: set[str]
    discovered_tool_names: set[str]
    anchored_evidence: AnchoredEvidence
    available_tool_names: set[str]
    active_capability_ids: set[str]
    deferred_capability_ids: set[str]
    capability_active: bool | None
    tool_name_path: ToolNamePath
    workspace_ref: WorkspaceRef | None
    deps_input: JsonValue = Field(default=None, exclude=True)

    @classmethod
    def capture(cls, ctx: RunContext[object]) -> _ContextSnapshot:
        return cls(
            run_id=ctx.run_id,
            conversation_id=ctx.conversation_id,
            metadata=ctx.metadata,
            retries=ctx.retries,
            tool_call_id=ctx.tool_call_id,
            tool_name=ctx.tool_name,
            tool_call_approved=ctx.tool_call_approved,
            tool_call_metadata=ctx.tool_call_metadata,
            retry=ctx.retry,
            max_retries=ctx.max_retries,
            run_step=ctx.run_step,
            partial_output=ctx.partial_output,
            trace_include_content=ctx.trace_include_content,
            instrumentation_version=ctx.instrumentation_version,
            usage=ctx.usage,
            usage_limits=ctx.usage_limits,
            deadline=ctx.deadline,
            loaded_capability_ids=ctx.loaded_capability_ids,
            discovered_tool_names=ctx.discovered_tool_names,
            anchored_evidence=ctx._anchored_evidence,  # pyright: ignore[reportPrivateUsage]
            available_tool_names=ctx.available_tool_names,
            active_capability_ids=ctx.active_capability_ids,
            deferred_capability_ids=ctx._deferred_capability_ids,  # pyright: ignore[reportPrivateUsage]
            capability_active=ctx.capability_active,
            tool_name_path=ctx._tool_name_path,  # pyright: ignore[reportPrivateUsage]
            workspace_ref=ctx.workspace.ref,
            deps_input=(
                ctx.deps.deps_input
                if isinstance(ctx.deps, _DepsProxy)
                else _JSON.validate_python(to_jsonable_python(ctx.deps))
            ),
        )


class _ActivityRunContext(RunContext[DepsT], Generic[DepsT]):
    def __init__(
        self,
        *,
        deps: DepsT,
        available_tool_names: set[str],
        active_capability_ids: set[str],
        _deferred_capability_ids: set[str],
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault('workspace', Workspace(NO_WORKSPACE))
        self.__dict__ = {
            **kwargs,
            'deps': deps,
            'available_tool_names': available_tool_names,
            'active_capability_ids': active_capability_ids,
            '_deferred_capability_ids': _deferred_capability_ids,
            'pending_messages': EnqueueGuard(enqueue_not_supported_message('activity', 'orchestration')),
            'tool_manager': None,
            'realtime_session': None,
            '_durable_operations': None,
            '_run_capabilities_by_id': None,
            '_run_held_toolsets': None,
        }
        self._activity_available_tool_names = available_tool_names
        self._activity_active_capability_ids = active_capability_ids
        self._activity_deferred_capability_ids = _deferred_capability_ids
        self.__dict__['__dataclass_fields__'] = {
            name: field
            for name, field in (RunContext.__dataclass_fields__ | DispatchSnapshots.__dataclass_fields__).items()
            if name in self.__dict__
        }

    def __getattribute__(self, name: str) -> Any:
        if name in RunContext.__dataclass_fields__ and name not in object.__getattribute__(self, '__dict__'):
            raise UserError(f'{name!r} is not available inside a Durable Task activity.')
        return super().__getattribute__(name)

    def _expose_field(self, name: str) -> None:
        self.__dict__['__dataclass_fields__'][name] = RunContext.__dataclass_fields__[name]

    @property
    def available_tool_names(self) -> set[str]:
        return self._activity_available_tool_names

    @property
    def active_capability_ids(self) -> set[str]:
        return self._activity_active_capability_ids

    @property
    def _deferred_capability_ids(self) -> set[str]:
        return self._activity_deferred_capability_ids

    @overload
    async def emit(self, event: CustomEventT, /) -> CustomEventT: ...

    @overload
    async def emit(self, event: CapabilityEventT, /) -> CapabilityEventT: ...

    async def emit(self, event: CustomEvent | CapabilityEvent, /) -> CustomEvent | CapabilityEvent:
        raise UserError('Emitting events inside a Durable Task activity is not supported. Emit from capability hooks.')


def restore_context(
    snapshot: _ContextSnapshot, *, deps: DepsT, agent: AbstractAgent[DepsT, object]
) -> RunContext[DepsT]:
    ctx = _ActivityRunContext(
        run_id=snapshot.run_id,
        conversation_id=snapshot.conversation_id,
        metadata=snapshot.metadata,
        retries=snapshot.retries,
        tool_call_id=snapshot.tool_call_id,
        tool_name=snapshot.tool_name,
        tool_call_approved=snapshot.tool_call_approved,
        tool_call_metadata=snapshot.tool_call_metadata,
        retry=snapshot.retry,
        max_retries=snapshot.max_retries,
        run_step=snapshot.run_step,
        partial_output=snapshot.partial_output,
        trace_include_content=snapshot.trace_include_content,
        instrumentation_version=snapshot.instrumentation_version,
        usage=snapshot.usage,
        usage_limits=snapshot.usage_limits,
        deadline=snapshot.deadline,
        loaded_capability_ids=snapshot.loaded_capability_ids,
        discovered_tool_names=snapshot.discovered_tool_names,
        _anchored_evidence=snapshot.anchored_evidence,
        available_tool_names=snapshot.available_tool_names,
        active_capability_ids=snapshot.active_capability_ids,
        _deferred_capability_ids=snapshot.deferred_capability_ids,
        capability_active=snapshot.capability_active,
        _tool_name_path=snapshot.tool_name_path,
        workspace_ref=snapshot.workspace_ref,
        deps=deps,
        agent=agent,
        root_capability=agent.root_capability,
    )
    if snapshot.workspace_ref is not None:
        workspace = select_workspace(agent.root_capability, ctx, ref=snapshot.workspace_ref)
        if workspace is None:
            raise UserError(
                f'No capability can restore workspace {snapshot.workspace_ref.id!r} inside this Durable Task activity.'
            )
        ctx.__dict__['workspace'] = workspace
    return ctx

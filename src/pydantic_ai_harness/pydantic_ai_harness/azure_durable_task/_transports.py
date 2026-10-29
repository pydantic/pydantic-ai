from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast, runtime_checkable

from pydantic import ConfigDict, JsonValue, TypeAdapter, with_config

from pydantic_ai.durable_exec import (
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
    DurabilityEngineSpec,
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    ParameterTransport,
    ParameterTransports,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
    WorkspaceCall,
    WorkspaceCallParams,
    revalidate_function_call,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool
from pydantic_ai.workspaces import WorkspaceRef

from ._operation_backend import INPUT, ActivityInput
from ._run_context import ContextSnapshot


@runtime_checkable
class _ToolDefinitionResolver(Protocol):
    """The tool-reconstruction hook an MCP toolset publishes with no shared base.

    The capability hands MCP toolsets over as `AbstractToolset`, which does not declare
    `tool_for_tool_def`, so this structural protocol narrows them without importing
    `pydantic_ai.mcp` and the `mcp` SDK.
    """

    def tool_for_tool_def(self, tool_def: ToolDefinition, *, ctx: RunContext[Any]) -> ToolsetTool[Any]: ...


class _CompactionModelPlaceholder(Model):
    """Replaced by core's worker-side model scope before compaction."""

    @property
    def model_name(self) -> str:  # pragma: no cover
        return 'azure-durable-task-compaction-placeholder'

    @property
    def system(self) -> str:  # pragma: no cover
        return 'azure-durable-task'

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:  # pragma: no cover
        raise RuntimeError('The compaction model placeholder must be replaced before use.')


class _HasContext(Protocol):
    @property
    def context(self) -> ContextSnapshot | None: ...


ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT', bound=_HasContext)
RestoreContext = Callable[[ContextSnapshot], RunContext[Any]]


class _Transport(ParameterTransport[ParamsT, JsonValue], Generic[ParamsT, WireT]):
    def __init__(
        self,
        adapter: TypeAdapter[WireT],
        dump: Callable[[ParamsT], WireT],
        load: Callable[[WireT], ParamsT],
    ) -> None:
        self._adapter = adapter
        self._dump = dump
        self._load = load

    def dump(self, params: ParamsT) -> JsonValue:
        wire = self._dump(params)
        payload = ActivityInput(
            parameters=self._adapter.dump_python(wire, mode='json', round_trip=True),
            deps_input=wire.context.deps_input if wire.context is not None else None,
        )
        return INPUT.dump_python(payload, mode='json')

    def load(self, payload: JsonValue, *, runtime: object) -> ParamsT:
        return self._load(self._adapter.validate_python(INPUT.validate_python(payload).parameters))


@dataclass(kw_only=True)
class _Call:
    name: str
    tool_args: dict[str, Any]
    context: ContextSnapshot
    tool_def: ToolDefinition | None
    original_name: str | None = None


_CALL = TypeAdapter(_Call)


def function_transport(
    toolset: FunctionToolset[Any], restore: RestoreContext
) -> _Transport[ToolsetCallToolParams, _Call]:
    def dump(params: ToolsetCallToolParams) -> _Call:
        tool = params.tool
        assert tool is not None
        return _Call(
            name=params.name,
            tool_args=params.tool_args,
            context=ContextSnapshot.capture(params.ctx),
            tool_def=tool.tool_def,
            original_name=tool.original_name if isinstance(tool, FunctionToolsetTool) else None,
        )

    def load(wire: _Call) -> ToolsetCallToolParams:
        ctx = restore(wire.context)
        assert wire.tool_def is not None
        try:
            tool = toolset.tool_for_tool_def(wire.tool_def, ctx=ctx, original_name=wire.original_name)
        except KeyError as exc:
            raise UserError(
                f'Tool {wire.name!r} is no longer available in toolset {toolset.id!r}. '
                'Keep registered tools compatible with running orchestrations.'
            ) from exc
        return ToolsetCallToolParams(wire.name, tool_args=wire.tool_args, ctx=ctx, tool=tool)

    return _Transport(_CALL, dump, load)


@dataclass
class _Discovery:
    context: ContextSnapshot


_DISCOVERY = TypeAdapter(_Discovery)


def discovery_transport(restore: RestoreContext) -> _Transport[ToolsetGetToolsParams, _Discovery]:
    return _Transport(
        _DISCOVERY,
        lambda params: _Discovery(ContextSnapshot.capture(params.ctx)),
        lambda wire: ToolsetGetToolsParams(restore(wire.context)),
    )


def dynamic_transport(restore: RestoreContext) -> _Transport[DynamicToolsetCallToolParams, _Call]:
    return _Transport(
        _CALL,
        lambda params: _Call(
            name=params.name,
            tool_args=params.tool_args,
            context=ContextSnapshot.capture(params.ctx),
            tool_def=params.tool_def,
        ),
        lambda wire: DynamicToolsetCallToolParams(
            wire.name, tool_args=wire.tool_args, ctx=restore(wire.context), tool_def=wire.tool_def
        ),
    )


def mcp_transport(
    resolve_tool: Callable[[ToolDefinition, RunContext[Any]], ToolsetTool[Any]], restore: RestoreContext
) -> _Transport[ToolsetCallToolParams, _Call]:
    def dump(params: ToolsetCallToolParams) -> _Call:
        assert params.tool is not None
        return _Call(
            name=params.name,
            tool_args=params.tool_args,
            tool_def=params.tool.tool_def,
            context=ContextSnapshot.capture(params.ctx),
        )

    def load(wire: _Call) -> ToolsetCallToolParams:
        assert wire.tool_def is not None
        ctx = restore(wire.context)
        return ToolsetCallToolParams(
            wire.name, tool_args=wire.tool_args, ctx=ctx, tool=resolve_tool(wire.tool_def, ctx)
        )

    return _Transport(_CALL, dump, load)


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _Request:
    model_id: str | None
    messages: list[ModelMessage]
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    context: ContextSnapshot
    streaming: bool = False
    instructions: str | None = None


_REQUEST = TypeAdapter(_Request)


def request_transport(restore: RestoreContext) -> _Transport[ModelRequestParams, _Request]:
    return _Transport(
        _REQUEST,
        lambda params: _Request(
            model_id=params.model_id,
            messages=params.messages,
            model_settings=dict(params.model_settings) if params.model_settings is not None else None,
            model_request_parameters=params.model_request_parameters,
            context=ContextSnapshot.capture(params.run_context),
        ),
        lambda wire: ModelRequestParams(
            wire.model_id,
            messages=wire.messages,
            model_settings=cast(ModelSettings | None, wire.model_settings),
            model_request_parameters=wire.model_request_parameters,
            run_context=restore(wire.context),
        ),
    )


def compact_transport(restore: RestoreContext) -> _Transport[ModelCompactMessagesParams, _Request]:
    def load(wire: _Request) -> ModelCompactMessagesParams:
        # The base's compact handler installs the resolved worker-side model before using this context.
        request = ModelRequestContext(
            model=_CompactionModelPlaceholder(),
            messages=wire.messages,
            model_settings=cast(ModelSettings | None, wire.model_settings),
            model_request_parameters=wire.model_request_parameters,
        )
        request.model_id = wire.model_id
        request.streaming = wire.streaming
        return ModelCompactMessagesParams(
            wire.model_id, request_context=request, instructions=wire.instructions, run_context=restore(wire.context)
        )

    return _Transport(
        _REQUEST,
        lambda params: _Request(
            model_id=params.model_id,
            messages=params.request_context.messages,
            model_settings=(
                dict(params.request_context.model_settings)
                if params.request_context.model_settings is not None
                else None
            ),
            model_request_parameters=params.request_context.model_request_parameters,
            streaming=params.request_context.streaming,
            instructions=params.instructions,
            context=ContextSnapshot.capture(params.run_context),
        ),
        load,
    )


@dataclass(kw_only=True)
class _Cancel:
    model_id: str | None
    response: ModelResponse
    context: ContextSnapshot | None


_CANCEL = TypeAdapter(_Cancel)


def cancel_transport(restore: RestoreContext) -> _Transport[ModelCancelSuspendedResponseParams, _Cancel]:
    return _Transport(
        _CANCEL,
        lambda params: _Cancel(
            model_id=params.model_id,
            response=params.response,
            context=ContextSnapshot.capture(params.run_context) if params.run_context is not None else None,
        ),
        lambda wire: ModelCancelSuspendedResponseParams(
            wire.model_id,
            response=wire.response,
            run_context=restore(wire.context) if wire.context is not None else None,
        ),
    )


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _Event:
    event: AgentStreamEvent
    context: ContextSnapshot


_EVENT = TypeAdapter(_Event)


def event_transport(restore: RestoreContext) -> _Transport[EventStreamHandlerParams, _Event]:
    return _Transport(
        _EVENT,
        lambda params: _Event(event=params.event, context=ContextSnapshot.capture(params.run_context)),
        lambda wire: EventStreamHandlerParams(wire.event, run_context=restore(wire.context)),
    )


@dataclass(kw_only=True)
class _CapabilityCall:
    arguments: dict[str, Any]
    context: ContextSnapshot
    model_id: str | None


_CAPABILITY_CALL = TypeAdapter(_CapabilityCall)


def capability_transport(restore: RestoreContext) -> _Transport[CapabilityOperationParams, _CapabilityCall]:
    return _Transport(
        _CAPABILITY_CALL,
        lambda params: _CapabilityCall(
            arguments=params.arguments,
            context=ContextSnapshot.capture(params.run_context),
            model_id=params.model_id,
        ),
        lambda wire: CapabilityOperationParams(restore(wire.context), arguments=wire.arguments, model_id=wire.model_id),
    )


@dataclass(kw_only=True)
class _WorkspaceCall:
    call: WorkspaceCall
    ref: WorkspaceRef | None
    context: ContextSnapshot


_WORKSPACE_CALL = TypeAdapter(_WorkspaceCall)


def workspace_transport(restore: RestoreContext) -> _Transport[WorkspaceCallParams, _WorkspaceCall]:
    return _Transport(
        _WORKSPACE_CALL,
        lambda params: _WorkspaceCall(
            call=params.call, ref=params.ref, context=ContextSnapshot.capture(params.run_context)
        ),
        lambda wire: WorkspaceCallParams(run_context=restore(wire.context), ref=wire.ref, call=wire.call),
    )


class AzureDurableTaskParameterTransports(ParameterTransports):
    def __init__(
        self,
        restore: RestoreContext,
        toolsets: ContextVar[AsyncExitStack],
        *,
        engine_spec: DurabilityEngineSpec,
    ) -> None:
        self._restore = restore
        self._toolsets = toolsets
        self._engine_spec = engine_spec

    def model_request(self, result_type: object) -> ParameterTransport[ModelRequestParams, JsonValue]:
        return request_transport(self._restore)

    def compact_messages(self) -> ParameterTransport[ModelCompactMessagesParams, JsonValue]:
        return compact_transport(self._restore)

    def cancel_suspended_response(self) -> ParameterTransport[ModelCancelSuspendedResponseParams, JsonValue]:
        return cancel_transport(self._restore)

    def event_stream_handler(self) -> ParameterTransport[EventStreamHandlerParams, JsonValue]:
        return event_transport(self._restore)

    def capability_operation(
        self, declaration: CapabilityMethodDeclaration
    ) -> ParameterTransport[CapabilityOperationParams, JsonValue]:
        return capability_transport(self._restore)

    def workspace_call(self) -> ParameterTransport[WorkspaceCallParams, JsonValue]:
        return workspace_transport(self._restore)

    def function_call(self, toolset: FunctionToolset[Any]) -> ParameterTransport[ToolsetCallToolParams, JsonValue]:
        return function_transport(toolset, self._restore)

    async def prepare_function_call(
        self, toolset: FunctionToolset[Any], params: ToolsetCallToolParams
    ) -> ToolsetCallToolParams:
        await self._toolsets.get().enter_async_context(toolset)
        return await revalidate_function_call(toolset, params, engine_spec=self._engine_spec)

    def mcp_call(self, toolset: AbstractToolset[Any]) -> ParameterTransport[ToolsetCallToolParams, JsonValue]:
        assert isinstance(toolset, _ToolDefinitionResolver)
        return mcp_transport(lambda definition, ctx: toolset.tool_for_tool_def(definition, ctx=ctx), self._restore)

    def get_tools(self, toolset: AbstractToolset[Any]) -> ParameterTransport[ToolsetGetToolsParams, JsonValue]:
        return discovery_transport(self._restore)

    def get_instructions(self, toolset: AbstractToolset[Any]) -> ParameterTransport[ToolsetGetToolsParams, JsonValue]:
        return discovery_transport(self._restore)

    def dynamic_get_tools(self, toolset: DynamicToolset[Any]) -> ParameterTransport[ToolsetGetToolsParams, JsonValue]:
        return discovery_transport(self._restore)

    def dynamic_call(self, toolset: DynamicToolset[Any]) -> ParameterTransport[DynamicToolsetCallToolParams, JsonValue]:
        return dynamic_transport(self._restore)

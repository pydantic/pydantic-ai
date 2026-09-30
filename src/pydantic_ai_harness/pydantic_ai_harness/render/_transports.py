"""JSON parameter transports for Pydantic AI operations run as Render tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.workspaces import WorkspaceRef

from ._compat import (
    CallToolResult,
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
    DynamicToolsetCallToolParams,
    DynamicToolsResult,
    EventStreamHandlerParams,
    JSONObject,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    RenderJsonTransport,
    RenderRunContextCodec,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
    WorkspaceCall,
    WorkspaceCallParams,
    WorkspaceCallResult,
    capability_operation_result_type,
    dump_json_object,
    function_tool_original_name,
    load_json_object,
    load_json_type,
    make_model_request_context,
    model_settings_from_json,
    model_settings_to_json,
    resolve_function_tool_for_definition,
    resolve_mcp_tool_for_definition,
)

__all__ = (
    'RenderCancelTransport',
    'RenderCapabilityOperationTransport',
    'RenderCompactMessagesTransport',
    'RenderDynamicCallTransport',
    'RenderDynamicGetToolsTransport',
    'RenderEventStreamHandlerTransport',
    'RenderFunctionCallTransport',
    'RenderGetToolsTransport',
    'RenderMCPCallTransport',
    'RenderModelRequestTransport',
    'RenderWorkspaceCallTransport',
)

AgentDepsT = TypeVar('AgentDepsT')


@dataclass(kw_only=True)
class _ContextPayload:
    run_context: JSONObject


@dataclass(kw_only=True)
class _CallToolPayload:
    name: str
    tool_args: JSONObject
    run_context: JSONObject
    tool_def: ToolDefinition | None = None
    original_name: str | None = None


@dataclass(kw_only=True)
class _ModelRequestPayload:
    messages: list[ModelMessage]
    model_settings: JSONObject | None
    model_request_parameters: ModelRequestParameters
    run_context: JSONObject
    model_id: str | None = None


@dataclass(kw_only=True)
class _CapabilityOperationPayload:
    arguments: JSONObject
    run_context: JSONObject
    model_id: str | None = None


@dataclass(kw_only=True)
class _CompactMessagesPayload:
    messages: list[ModelMessage]
    model_settings: JSONObject | None
    model_request_parameters: ModelRequestParameters
    streaming: bool
    instructions: str | None
    run_context: JSONObject
    model_id: str | None = None


@dataclass(kw_only=True)
class _CancelPayload:
    response: ModelResponse
    model_id: str | None = None
    run_context: JSONObject | None = None


@dataclass(kw_only=True)
class _EventPayload:
    event: AgentStreamEvent
    run_context: JSONObject


class RenderFunctionCallTransport(Generic[AgentDepsT], RenderJsonTransport[ToolsetCallToolParams]):
    result_type = CallToolResult

    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT], toolset: FunctionToolset[AgentDepsT]) -> None:
        self._context_codec = context_codec
        self._toolset = toolset

    def dump(self, params: ToolsetCallToolParams) -> JSONObject:
        tool = params.tool
        if tool is None:
            raise ValueError(f'Function tool {params.name!r} has no resolved definition.')
        payload = _CallToolPayload(
            name=params.name,
            tool_args=dump_json_object(dict[str, object], params.tool_args),
            run_context=self._context_codec.dump(params.ctx),
            tool_def=tool.tool_def,
            original_name=function_tool_original_name(tool),
        )
        return dump_json_object(_CallToolPayload, payload)

    def load(self, payload: JSONObject, *, runtime: object) -> ToolsetCallToolParams:
        del runtime
        decoded = load_json_type(_CallToolPayload, payload)
        ctx = self._context_codec.load(decoded.run_context)
        try:
            tool = (
                resolve_function_tool_for_definition(
                    self._toolset,
                    decoded.tool_def,
                    ctx=ctx,
                    original_name=decoded.original_name,
                )
                if decoded.tool_def is not None
                else None
            )
        except KeyError as exc:
            raise UserError(
                f'Tool {decoded.name!r} not found in toolset {self._toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Render Workflows.'
            ) from exc
        return ToolsetCallToolParams(
            decoded.name,
            tool_args=load_json_object(decoded.tool_args),
            ctx=ctx,
            tool=tool,
        )


class RenderGetToolsTransport(Generic[AgentDepsT], RenderJsonTransport[ToolsetGetToolsParams]):
    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT], *, result_type: object) -> None:
        self._context_codec = context_codec
        self.result_type = result_type

    def dump(self, params: ToolsetGetToolsParams) -> JSONObject:
        return dump_json_object(_ContextPayload, _ContextPayload(run_context=self._context_codec.dump(params.ctx)))

    def load(self, payload: JSONObject, *, runtime: object) -> ToolsetGetToolsParams:
        del runtime
        decoded = load_json_type(_ContextPayload, payload)
        return ToolsetGetToolsParams(self._context_codec.load(decoded.run_context))


class RenderDynamicGetToolsTransport(RenderGetToolsTransport[AgentDepsT]):
    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT]) -> None:
        super().__init__(context_codec, result_type=DynamicToolsResult)


class RenderMCPCallTransport(Generic[AgentDepsT], RenderJsonTransport[ToolsetCallToolParams]):
    result_type = CallToolResult

    def __init__(
        self,
        context_codec: RenderRunContextCodec[AgentDepsT],
        toolset: AbstractToolset[AgentDepsT],
    ) -> None:
        self._context_codec = context_codec
        self._toolset = toolset

    def dump(self, params: ToolsetCallToolParams) -> JSONObject:
        tool = params.tool
        if tool is None:
            raise ValueError(f'MCP tool {params.name!r} has no resolved definition.')
        payload = _CallToolPayload(
            name=params.name,
            tool_args=dump_json_object(dict[str, object], params.tool_args),
            run_context=self._context_codec.dump(params.ctx),
            tool_def=tool.tool_def,
        )
        return dump_json_object(_CallToolPayload, payload)

    def load(self, payload: JSONObject, *, runtime: object) -> ToolsetCallToolParams:
        del runtime
        decoded = load_json_type(_CallToolPayload, payload)
        ctx = self._context_codec.load(decoded.run_context)
        if decoded.tool_def is None:
            raise ValueError(f'MCP tool {decoded.name!r} has no serialized definition.')
        tool = resolve_mcp_tool_for_definition(self._toolset, decoded.tool_def, ctx=ctx)
        return ToolsetCallToolParams(
            decoded.name,
            tool_args=load_json_object(decoded.tool_args),
            ctx=ctx,
            tool=tool,
        )


class RenderDynamicCallTransport(Generic[AgentDepsT], RenderJsonTransport[DynamicToolsetCallToolParams]):
    result_type = CallToolResult

    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT]) -> None:
        self._context_codec = context_codec

    def dump(self, params: DynamicToolsetCallToolParams) -> JSONObject:
        payload = _CallToolPayload(
            name=params.name,
            tool_args=dump_json_object(dict[str, object], params.tool_args),
            run_context=self._context_codec.dump(params.ctx),
            tool_def=params.tool_def,
        )
        return dump_json_object(_CallToolPayload, payload)

    def load(self, payload: JSONObject, *, runtime: object) -> DynamicToolsetCallToolParams:
        del runtime
        decoded = load_json_type(_CallToolPayload, payload)
        return DynamicToolsetCallToolParams(
            decoded.name,
            tool_args=load_json_object(decoded.tool_args),
            ctx=self._context_codec.load(decoded.run_context),
            tool_def=decoded.tool_def,
        )


class RenderCapabilityOperationTransport(Generic[AgentDepsT], RenderJsonTransport[CapabilityOperationParams]):
    def __init__(
        self,
        context_codec: RenderRunContextCodec[AgentDepsT],
        declaration: CapabilityMethodDeclaration,
    ) -> None:
        self._context_codec = context_codec
        self.result_type = capability_operation_result_type(declaration.result_type)

    def dump(self, params: CapabilityOperationParams) -> JSONObject:
        payload = _CapabilityOperationPayload(
            arguments=dump_json_object(dict[str, object], params.arguments),
            run_context=self._context_codec.dump(params.run_context),
            model_id=params.model_id,
        )
        return dump_json_object(_CapabilityOperationPayload, payload)

    def load(self, payload: JSONObject, *, runtime: object) -> CapabilityOperationParams:
        del runtime
        decoded = load_json_type(_CapabilityOperationPayload, payload)
        return CapabilityOperationParams(
            self._context_codec.load(decoded.run_context),
            arguments=load_json_object(decoded.arguments),
            model_id=decoded.model_id,
        )


class RenderModelRequestTransport(Generic[AgentDepsT], RenderJsonTransport[ModelRequestParams]):
    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT], *, result_type: object) -> None:
        self._context_codec = context_codec
        self.result_type = result_type

    def dump(self, params: ModelRequestParams) -> JSONObject:
        payload = _ModelRequestPayload(
            messages=params.messages,
            model_settings=model_settings_to_json(params.model_settings),
            model_request_parameters=params.model_request_parameters,
            run_context=self._context_codec.dump(params.run_context),
            model_id=params.model_id,
        )
        return dump_json_object(_ModelRequestPayload, payload)

    def load(self, payload: JSONObject, *, runtime: object) -> ModelRequestParams:
        del runtime
        decoded = load_json_type(_ModelRequestPayload, payload)
        return ModelRequestParams(
            decoded.model_id,
            messages=decoded.messages,
            model_settings=model_settings_from_json(decoded.model_settings),
            model_request_parameters=decoded.model_request_parameters,
            run_context=self._context_codec.load(decoded.run_context),
        )


class RenderCompactMessagesTransport(Generic[AgentDepsT], RenderJsonTransport[ModelCompactMessagesParams]):
    result_type = ModelResponse

    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT]) -> None:
        self._context_codec = context_codec

    def dump(self, params: ModelCompactMessagesParams) -> JSONObject:
        request_context = params.request_context
        payload = _CompactMessagesPayload(
            messages=request_context.messages,
            model_settings=model_settings_to_json(request_context.model_settings),
            model_request_parameters=request_context.model_request_parameters,
            streaming=request_context.streaming,
            instructions=params.instructions,
            run_context=self._context_codec.dump(params.run_context),
            model_id=params.model_id,
        )
        return dump_json_object(_CompactMessagesPayload, payload)

    def load(self, payload: JSONObject, *, runtime: object) -> ModelCompactMessagesParams:
        del runtime
        decoded = load_json_type(_CompactMessagesPayload, payload)
        request_context = make_model_request_context(
            messages=decoded.messages,
            model_settings=model_settings_from_json(decoded.model_settings),
            model_request_parameters=decoded.model_request_parameters,
            model_id=decoded.model_id,
            streaming=decoded.streaming,
        )
        return ModelCompactMessagesParams(
            decoded.model_id,
            request_context=request_context,
            instructions=decoded.instructions,
            run_context=self._context_codec.load(decoded.run_context),
        )


class RenderCancelTransport(Generic[AgentDepsT], RenderJsonTransport[ModelCancelSuspendedResponseParams]):
    result_type = type(None)

    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT]) -> None:
        self._context_codec = context_codec

    def dump(self, params: ModelCancelSuspendedResponseParams) -> JSONObject:
        payload = _CancelPayload(
            response=params.response,
            model_id=params.model_id,
            run_context=self._context_codec.dump(params.run_context) if params.run_context is not None else None,
        )
        return dump_json_object(_CancelPayload, payload)

    def load(self, payload: JSONObject, *, runtime: object) -> ModelCancelSuspendedResponseParams:
        del runtime
        decoded = load_json_type(_CancelPayload, payload)
        ctx = self._context_codec.load(decoded.run_context) if decoded.run_context is not None else None
        return ModelCancelSuspendedResponseParams(decoded.model_id, response=decoded.response, run_context=ctx)


class RenderEventStreamHandlerTransport(Generic[AgentDepsT], RenderJsonTransport[EventStreamHandlerParams]):
    result_type = type(None)

    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT]) -> None:
        self._context_codec = context_codec

    def dump(self, params: EventStreamHandlerParams) -> JSONObject:
        payload = _EventPayload(event=params.event, run_context=self._context_codec.dump(params.run_context))
        return dump_json_object(_EventPayload, payload)

    def load(self, payload: JSONObject, *, runtime: object) -> EventStreamHandlerParams:
        del runtime
        decoded = load_json_type(_EventPayload, payload)
        return EventStreamHandlerParams(decoded.event, run_context=self._context_codec.load(decoded.run_context))


@dataclass(kw_only=True)
class _WorkspacePayload:
    call: WorkspaceCall
    ref: WorkspaceRef | None
    run_context: JSONObject


class RenderWorkspaceCallTransport(Generic[AgentDepsT], RenderJsonTransport[WorkspaceCallParams]):
    result_type = WorkspaceCallResult

    def __init__(self, context_codec: RenderRunContextCodec[AgentDepsT]) -> None:
        self._context_codec = context_codec

    def dump(self, params: WorkspaceCallParams) -> JSONObject:
        return dump_json_object(
            _WorkspacePayload,
            _WorkspacePayload(
                call=params.call, ref=params.ref, run_context=self._context_codec.dump(params.run_context)
            ),
        )

    def load(self, payload: JSONObject, *, runtime: object) -> WorkspaceCallParams:
        del runtime
        wire = load_json_type(_WorkspacePayload, payload)
        return WorkspaceCallParams(call=wire.call, ref=wire.ref, run_context=self._context_codec.load(wire.run_context))

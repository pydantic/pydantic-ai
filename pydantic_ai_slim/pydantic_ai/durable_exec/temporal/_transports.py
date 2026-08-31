from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from pydantic import ConfigDict, with_config

from pydantic_ai import messages as _messages
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
    capability_operation_result_type,
)
from pydantic_ai.durable_exec._operation import (
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams as _SemanticEventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
)
from pydantic_ai.durable_exec._toolset import CallToolResult, DynamicToolsResult
from pydantic_ai.durable_exec._utils import StreamedActivityResult
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._operation_backend import TemporalParameterTransport
from ._toolset import CallToolParams, GetToolsParams

if TYPE_CHECKING:
    from ._durability import TemporalDurability

__all__ = (
    '_CancelParams',
    '_CapabilityOperationParams',
    '_CapabilityOperationTransport',
    '_CancelTransport',
    '_CompactMessagesTransport',
    '_DynamicCallTransport',
    '_DynamicGetToolsTransport',
    '_EventStreamHandlerParams',
    '_EventStreamHandlerTransport',
    '_FunctionCallTransport',
    '_GetToolsTransport',
    '_MCPCallTransport',
    '_ModelRequestTransport',
    '_RequestParams',
    '_StreamedActivityPayload',
)


class _FunctionCallTransport(TemporalParameterTransport[ToolsetCallToolParams, tuple[CallToolParams, Any]]):
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any], toolset: FunctionToolset[Any]) -> None:
        self._durability = durability
        self._toolset = toolset

    def dump(self, params: ToolsetCallToolParams) -> tuple[CallToolParams, Any]:
        assert params.tool is not None
        tool = params.tool
        return (
            CallToolParams(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx),
                tool_def=tool.tool_def,
                original_name=tool.original_name if isinstance(tool, FunctionToolsetTool) else None,
            ),
            params.ctx.deps,
        )

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> ToolsetCallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        try:
            tool = (
                self._toolset.tool_for_tool_def(params.tool_def, ctx=ctx, original_name=params.original_name)
                if params.tool_def is not None
                else None
            )
        except KeyError as exc:
            raise UserError(
                f'Tool {params.name!r} not found in toolset {self._toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Temporal.'
            ) from exc
        return ToolsetCallToolParams(params.name, tool_args=params.tool_args, ctx=ctx, tool=tool)


class _GetToolsTransport(TemporalParameterTransport[ToolsetGetToolsParams, tuple[GetToolsParams, Any]]):
    wire_type = GetToolsParams
    result_type = dict[str, ToolDefinition]

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: ToolsetGetToolsParams) -> tuple[GetToolsParams, Any]:
        return (
            GetToolsParams(serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx)),
            params.ctx.deps,
        )

    def load(self, payload: tuple[GetToolsParams, Any], *, runtime: object) -> ToolsetGetToolsParams:
        params, deps = payload
        return ToolsetGetToolsParams(
            self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        )


class _MCPCallTransport(TemporalParameterTransport[ToolsetCallToolParams, tuple[CallToolParams, Any]]):
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any], toolset: Any) -> None:
        self._durability = durability
        self._toolset = toolset

    def dump(self, params: ToolsetCallToolParams) -> tuple[CallToolParams, Any]:
        assert params.tool is not None
        return (
            CallToolParams(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx),
                tool_def=params.tool.tool_def,
            ),
            params.ctx.deps,
        )

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> ToolsetCallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        assert params.tool_def is not None
        return ToolsetCallToolParams(
            params.name,
            tool_args=params.tool_args,
            ctx=ctx,
            tool=self._toolset.tool_for_tool_def(params.tool_def, ctx=ctx),
        )


class _DynamicCallTransport(TemporalParameterTransport[DynamicToolsetCallToolParams, tuple[CallToolParams, Any]]):
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: DynamicToolsetCallToolParams) -> tuple[CallToolParams, Any]:
        return (
            CallToolParams(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx),
                tool_def=params.tool_def,
            ),
            params.ctx.deps,
        )

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> DynamicToolsetCallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return DynamicToolsetCallToolParams(params.name, tool_args=params.tool_args, ctx=ctx, tool_def=params.tool_def)


class _DynamicGetToolsTransport(_GetToolsTransport):
    result_type = DynamicToolsResult


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    """Serializable arguments for the model-request Temporal activity."""

    messages: list[_messages.ModelMessage]
    # `model_settings` can't be a `ModelSettings` because Temporal would end up dropping fields only defined on its subclasses.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any
    model_id: str | None = None


@dataclass(kw_only=True)
class _CapabilityOperationParams:
    arguments: dict[str, Any]
    serialized_run_context: Any
    model_id: str | None = None


class _CapabilityOperationTransport(
    TemporalParameterTransport[CapabilityOperationParams, tuple[_CapabilityOperationParams, Any]]
):
    wire_type = _CapabilityOperationParams

    def __init__(self, durability: TemporalDurability[Any], declaration: CapabilityMethodDeclaration) -> None:
        self._durability = durability
        self.result_type = capability_operation_result_type(declaration.result_type)

    def dump(self, params: CapabilityOperationParams) -> tuple[_CapabilityOperationParams, Any]:
        return (
            _CapabilityOperationParams(
                arguments=params.arguments,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.run_context),
                model_id=params.model_id,
            ),
            params.run_context.deps,
        )

    def load(self, payload: tuple[_CapabilityOperationParams, Any], *, runtime: object) -> CapabilityOperationParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return CapabilityOperationParams(ctx, arguments=params.arguments, model_id=params.model_id)


@dataclass(kw_only=True)
class _CancelParams:
    response: ModelResponse
    model_id: str | None = None
    serialized_run_context: Any = None


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CompactMessagesParams:
    messages: list[ModelMessage]
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    streaming: bool
    instructions: str | None
    serialized_run_context: Any
    model_id: str | None = None


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventStreamHandlerParams:
    event: AgentStreamEvent
    serialized_run_context: Any


# The `ModelResponse` arm decodes histories recorded by the deprecated `TemporalAgent`, whose
# stream activity returned the bare response. Remove it (and the workflow-side event synthesis
# in `request_stream_segment`) once those histories have aged out, along with `TemporalAgent`.
_StreamedActivityPayload: TypeAlias = StreamedActivityResult | ModelResponse


class _ModelRequestTransport(TemporalParameterTransport[ModelRequestParams, tuple[_RequestParams, Any]]):
    wire_type = _RequestParams

    def __init__(self, durability: TemporalDurability[Any], *, result_type: object) -> None:
        self._durability = durability
        self.result_type = result_type

    def dump(self, params: ModelRequestParams) -> tuple[_RequestParams, Any]:
        ctx = params.run_context
        return (
            _RequestParams(
                messages=params.messages,
                model_settings=cast(dict[str, Any] | None, params.model_settings),
                model_request_parameters=params.model_request_parameters,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
                model_id=params.model_id,
            ),
            ctx.deps,
        )

    def load(self, payload: tuple[_RequestParams, Any], *, runtime: object) -> ModelRequestParams:
        request, deps = payload
        ctx = self._durability.deserialize_operation_run_context(request.serialized_run_context, deps)
        return ModelRequestParams(
            request.model_id,
            messages=request.messages,
            model_settings=cast(ModelSettings | None, request.model_settings),
            model_request_parameters=request.model_request_parameters,
            run_context=ctx,
        )


class _CompactMessagesTransport(
    TemporalParameterTransport[ModelCompactMessagesParams, tuple[_CompactMessagesParams, Any]]
):
    wire_type = _CompactMessagesParams
    result_type = ModelResponse

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: ModelCompactMessagesParams) -> tuple[_CompactMessagesParams, Any]:
        ctx = params.run_context
        return (
            _CompactMessagesParams(
                messages=params.request_context.messages,
                model_settings=cast(dict[str, Any] | None, params.request_context.model_settings),
                model_request_parameters=params.request_context.model_request_parameters,
                streaming=params.request_context.streaming,
                instructions=params.instructions,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
                model_id=params.model_id,
            ),
            ctx.deps,
        )

    def load(self, payload: tuple[_CompactMessagesParams, Any], *, runtime: object) -> ModelCompactMessagesParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        request_context = ModelRequestContext(
            model=cast(Model[Any], None),
            messages=params.messages,
            model_settings=cast(ModelSettings | None, params.model_settings),
            model_request_parameters=params.model_request_parameters,
        )
        request_context.model_id = params.model_id
        request_context.streaming = params.streaming
        return ModelCompactMessagesParams(
            params.model_id,
            request_context=request_context,
            instructions=params.instructions,
            run_context=ctx,
        )


class _CancelTransport(TemporalParameterTransport[ModelCancelSuspendedResponseParams, tuple[_CancelParams, Any]]):
    wire_type = _CancelParams
    result_type = type(None)

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: ModelCancelSuspendedResponseParams) -> tuple[_CancelParams, Any]:
        ctx = params.run_context
        return (
            _CancelParams(
                response=params.response,
                model_id=params.model_id,
                serialized_run_context=(
                    self._durability.run_context_type.serialize_run_context(ctx) if ctx is not None else None
                ),
            ),
            ctx.deps if ctx is not None else None,
        )

    def load(self, payload: tuple[_CancelParams, Any], *, runtime: object) -> ModelCancelSuspendedResponseParams:
        params, deps = payload
        ctx = (
            self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
            if params.serialized_run_context is not None
            else None
        )
        return ModelCancelSuspendedResponseParams(params.model_id, response=params.response, run_context=ctx)


class _EventStreamHandlerTransport(
    TemporalParameterTransport[_SemanticEventStreamHandlerParams, tuple[_EventStreamHandlerParams, Any]]
):
    wire_type = _EventStreamHandlerParams
    result_type = type(None)

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: _SemanticEventStreamHandlerParams) -> tuple[_EventStreamHandlerParams, Any]:
        ctx = params.run_context
        return (
            _EventStreamHandlerParams(
                event=params.event,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
            ),
            ctx.deps,
        )

    def load(
        self, payload: tuple[_EventStreamHandlerParams, Any], *, runtime: object
    ) -> _SemanticEventStreamHandlerParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return _SemanticEventStreamHandlerParams(params.event, run_context=ctx)

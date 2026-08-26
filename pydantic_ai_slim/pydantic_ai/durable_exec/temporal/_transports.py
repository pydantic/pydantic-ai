from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeAlias, cast

from pydantic import ConfigDict, with_config

from pydantic_ai import messages as _messages
from pydantic_ai.durable_exec._base import (
    CancelSuspendedResponseOperationParams,
    CompactMessagesOperationParams,
    EventStreamHandlerOperationParams as _SemanticEventStreamHandlerParams,
    ModelRequestOperationParams,
    _CallToolParams,  # pyright: ignore[reportPrivateUsage]
    _DynamicCallToolParams,  # pyright: ignore[reportPrivateUsage]
    _GetToolsParams,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityMethodDeclaration,
    CapabilityOperationParams,
)
from pydantic_ai.durable_exec._toolset import CallToolResult, DynamicToolsResult
from pydantic_ai.durable_exec._utils import StreamedActivityResult
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.function import FunctionToolsetTool

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


class _FunctionCallTransport:
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any], toolset: FunctionToolset[Any]) -> None:
        self._durability = durability
        self._toolset = toolset

    def dump(self, params: _CallToolParams) -> tuple[CallToolParams, Any]:
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

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> _CallToolParams:
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
        return _CallToolParams(params.name, params.tool_args, ctx, tool)


class _GetToolsTransport:
    wire_type = GetToolsParams
    result_type = dict[str, ToolDefinition]

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: _GetToolsParams) -> tuple[GetToolsParams, Any]:
        return (
            GetToolsParams(serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx)),
            params.ctx.deps,
        )

    def load(self, payload: tuple[GetToolsParams, Any], *, runtime: object) -> _GetToolsParams:
        params, deps = payload
        return _GetToolsParams(self._durability.deserialize_operation_run_context(params.serialized_run_context, deps))


class _MCPCallTransport:
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any], toolset: Any) -> None:
        self._durability = durability
        self._toolset = toolset

    def dump(self, params: _CallToolParams) -> tuple[CallToolParams, Any]:
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

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> _CallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        assert params.tool_def is not None
        return _CallToolParams(
            params.name,
            params.tool_args,
            ctx,
            self._toolset.tool_for_tool_def(params.tool_def, ctx=ctx),
        )


class _DynamicCallTransport:
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: _DynamicCallToolParams) -> tuple[CallToolParams, Any]:
        return (
            CallToolParams(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx),
                tool_def=params.tool_def,
            ),
            params.ctx.deps,
        )

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> _DynamicCallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return _DynamicCallToolParams(params.name, params.tool_args, ctx, tool_def=params.tool_def)


class _DynamicGetToolsTransport(_GetToolsTransport):
    result_type = DynamicToolsResult


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    """Serializable arguments for the model-request Temporal activity."""

    messages: list[_messages.ModelMessage]
    # `model_settings` can't be a `ModelSettings` because Temporal would end up dropping fields only defined on its subclasses.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any
    model_id: str | None = None


@dataclass
class _CapabilityOperationParams:
    arguments: dict[str, Any]
    serialized_run_context: Any


class _CapabilityOperationTransport:
    wire_type = _CapabilityOperationParams

    def __init__(self, durability: TemporalDurability[Any], declaration: CapabilityMethodDeclaration) -> None:
        self._durability = durability
        self.result_type = declaration.result_type

    def dump(self, params: CapabilityOperationParams) -> tuple[_CapabilityOperationParams, Any]:
        return (
            _CapabilityOperationParams(
                arguments=params.arguments,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.run_context),
            ),
            params.run_context.deps,
        )

    def load(self, payload: tuple[_CapabilityOperationParams, Any], *, runtime: object) -> CapabilityOperationParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return CapabilityOperationParams(ctx, params.arguments)


@dataclass
class _CancelParams:
    response: ModelResponse
    model_id: str | None = None
    serialized_run_context: Any = None


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _CompactMessagesParams:
    request_context: ModelRequestContext
    instructions: str | None
    serialized_run_context: Any
    model_id: str | None = None


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventStreamHandlerParams:
    event: AgentStreamEvent
    serialized_run_context: Any


# The `ModelResponse` arm decodes histories recorded by the deprecated `TemporalAgent`, whose
# stream activity returned the bare response. Remove it (and the workflow-side event synthesis
# in `request_stream_segment`) once those histories have aged out, along with `TemporalAgent`.
_StreamedActivityPayload: TypeAlias = StreamedActivityResult | ModelResponse


class _ModelRequestTransport:
    wire_type = _RequestParams

    def __init__(self, durability: TemporalDurability[Any], *, result_type: object) -> None:
        self._durability = durability
        self.result_type = result_type

    def dump(self, params: ModelRequestOperationParams) -> tuple[_RequestParams, Any]:
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

    def load(self, payload: tuple[_RequestParams, Any], *, runtime: object) -> ModelRequestOperationParams:
        request, deps = payload
        ctx = self._durability.deserialize_operation_run_context(request.serialized_run_context, deps)
        return ModelRequestOperationParams(
            request.model_id,
            request.messages,
            cast(ModelSettings | None, request.model_settings),
            request.model_request_parameters,
            ctx,
        )


class _CompactMessagesTransport:
    wire_type = _CompactMessagesParams
    result_type = ModelResponse

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: CompactMessagesOperationParams) -> tuple[_CompactMessagesParams, Any]:
        ctx = params.run_context
        return (
            _CompactMessagesParams(
                request_context=params.request_context,
                instructions=params.instructions,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
                model_id=params.model_id,
            ),
            ctx.deps,
        )

    def load(self, payload: tuple[_CompactMessagesParams, Any], *, runtime: object) -> CompactMessagesOperationParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return CompactMessagesOperationParams(params.model_id, params.request_context, params.instructions, ctx)


class _CancelTransport:
    wire_type = _CancelParams
    result_type = type(None)

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: CancelSuspendedResponseOperationParams) -> tuple[_CancelParams, Any]:
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

    def load(self, payload: tuple[_CancelParams, Any], *, runtime: object) -> CancelSuspendedResponseOperationParams:
        params, deps = payload
        ctx = (
            self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
            if params.serialized_run_context is not None
            else None
        )
        return CancelSuspendedResponseOperationParams(params.model_id, params.response, ctx)


class _EventStreamHandlerTransport:
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
        return _SemanticEventStreamHandlerParams(params.event, ctx)

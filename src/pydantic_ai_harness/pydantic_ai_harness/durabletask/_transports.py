from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, cast

from pydantic import ConfigDict, JsonValue, TypeAdapter, with_config

from pydantic_ai.durable_exec._capability_operation import CapabilityOperationParams
from pydantic_ai.durable_exec._operation import (
    DynamicToolsetCallToolParams,
    EventStreamHandlerParams,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesParams,
    ModelRequestParams,
    ParameterTransport,
    ToolsetCallToolParams,
    ToolsetGetToolsParams,
)
from pydantic_ai.durable_exec._workspace import WorkspaceCall, WorkspaceCallParams
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.toolsets.abstract import ToolsetTool
from pydantic_ai.toolsets.function import FunctionToolsetTool
from pydantic_ai.workspaces import WorkspaceRef

from ._operation_backend import _INPUT, _ActivityInput  # pyright: ignore[reportPrivateUsage]
from ._run_context import _ContextSnapshot  # pyright: ignore[reportPrivateUsage]


class _HasContext(Protocol):
    @property
    def context(self) -> _ContextSnapshot | None: ...


ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT', bound=_HasContext)
RestoreContext = Callable[[_ContextSnapshot], RunContext[Any]]

_JSON = TypeAdapter[JsonValue](JsonValue)


class _Transport(ParameterTransport[ParamsT, JsonValue], Generic[ParamsT, WireT]):
    def __init__(
        self,
        wire_type: type[WireT],
        dump: Callable[[ParamsT], WireT],
        load: Callable[[WireT], ParamsT],
    ) -> None:
        self._adapter = TypeAdapter[WireT](wire_type)
        self._dump = dump
        self._load = load

    def dump(self, params: ParamsT) -> JsonValue:
        wire = self._dump(params)
        payload = _ActivityInput(
            parameters=_JSON.validate_python(self._adapter.dump_python(wire, mode='json', round_trip=True)),
            deps_input=wire.context.deps_input if wire.context is not None else None,
        )
        return _JSON.validate_python(_INPUT.dump_python(payload, mode='json'))

    def load(self, payload: JsonValue, *, runtime: object) -> ParamsT:
        return self._load(self._adapter.validate_python(_INPUT.validate_python(payload).parameters))


@dataclass(kw_only=True)
class _Call:
    name: str
    tool_args: dict[str, Any]
    context: _ContextSnapshot
    tool_def: ToolDefinition | None
    original_name: str | None = None


def function_transport(
    toolset: FunctionToolset[Any], restore: RestoreContext
) -> _Transport[ToolsetCallToolParams, _Call]:
    def dump(params: ToolsetCallToolParams) -> _Call:
        tool = params.tool
        assert tool is not None
        return _Call(
            name=params.name,
            tool_args=params.tool_args,
            context=_ContextSnapshot.capture(params.ctx),
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

    return _Transport(_Call, dump, load)


@dataclass
class _Discovery:
    context: _ContextSnapshot


def discovery_transport(restore: RestoreContext) -> _Transport[ToolsetGetToolsParams, _Discovery]:
    return _Transport(
        _Discovery,
        lambda params: _Discovery(_ContextSnapshot.capture(params.ctx)),
        lambda wire: ToolsetGetToolsParams(restore(wire.context)),
    )


def dynamic_transport(restore: RestoreContext) -> _Transport[DynamicToolsetCallToolParams, _Call]:
    return _Transport(
        _Call,
        lambda params: _Call(
            name=params.name,
            tool_args=params.tool_args,
            context=_ContextSnapshot.capture(params.ctx),
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
            context=_ContextSnapshot.capture(params.ctx),
        )

    def load(wire: _Call) -> ToolsetCallToolParams:
        assert wire.tool_def is not None
        ctx = restore(wire.context)
        return ToolsetCallToolParams(
            wire.name, tool_args=wire.tool_args, ctx=ctx, tool=resolve_tool(wire.tool_def, ctx)
        )

    return _Transport(_Call, dump, load)


@dataclass(kw_only=True)
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _Request:
    model_id: str | None
    messages: list[ModelMessage]
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    context: _ContextSnapshot
    streaming: bool = False
    instructions: str | None = None


def request_transport(restore: RestoreContext) -> _Transport[ModelRequestParams, _Request]:
    return _Transport(
        _Request,
        lambda params: _Request(
            model_id=params.model_id,
            messages=params.messages,
            model_settings=dict(params.model_settings) if params.model_settings is not None else None,
            model_request_parameters=params.model_request_parameters,
            context=_ContextSnapshot.capture(params.run_context),
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
            model=cast(Model, None),
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
        _Request,
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
            context=_ContextSnapshot.capture(params.run_context),
        ),
        load,
    )


@dataclass(kw_only=True)
class _Cancel:
    model_id: str | None
    response: ModelResponse
    context: _ContextSnapshot | None


def cancel_transport(restore: RestoreContext) -> _Transport[ModelCancelSuspendedResponseParams, _Cancel]:
    return _Transport(
        _Cancel,
        lambda params: _Cancel(
            model_id=params.model_id,
            response=params.response,
            context=_ContextSnapshot.capture(params.run_context) if params.run_context is not None else None,
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
    context: _ContextSnapshot


def event_transport(restore: RestoreContext) -> _Transport[EventStreamHandlerParams, _Event]:
    return _Transport(
        _Event,
        lambda params: _Event(event=params.event, context=_ContextSnapshot.capture(params.run_context)),
        lambda wire: EventStreamHandlerParams(wire.event, run_context=restore(wire.context)),
    )


@dataclass(kw_only=True)
class _CapabilityCall:
    arguments: dict[str, Any]
    context: _ContextSnapshot
    model_id: str | None


def capability_transport(restore: RestoreContext) -> _Transport[CapabilityOperationParams, _CapabilityCall]:
    return _Transport(
        _CapabilityCall,
        lambda params: _CapabilityCall(
            arguments=params.arguments,
            context=_ContextSnapshot.capture(params.run_context),
            model_id=params.model_id,
        ),
        lambda wire: CapabilityOperationParams(restore(wire.context), arguments=wire.arguments, model_id=wire.model_id),
    )


@dataclass(kw_only=True)
class _WorkspaceCall:
    call: WorkspaceCall
    ref: WorkspaceRef | None
    context: _ContextSnapshot


def workspace_transport(restore: RestoreContext) -> _Transport[WorkspaceCallParams, _WorkspaceCall]:
    return _Transport(
        _WorkspaceCall,
        lambda params: _WorkspaceCall(
            call=params.call, ref=params.ref, context=_ContextSnapshot.capture(params.run_context)
        ),
        lambda wire: WorkspaceCallParams(run_context=restore(wire.context), ref=wire.ref, call=wire.call),
    )

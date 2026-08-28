from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic, Literal, TypeVar, cast

from dbos import DBOS

from pydantic_ai import ToolsetTool, messages as _messages
from pydantic_ai.durable_exec._capability_operation import CapabilityOperationParams
from pydantic_ai.durable_exec._operation import (
    CapabilityOperationId,
    DurableOperation,
    DurableOperationId,
    DynamicToolsetCallToolParams,
    EventStreamHandlerId,
    EventStreamHandlerParams,
    ModelCancelSuspendedResponseId,
    ModelCancelSuspendedResponseParams,
    ModelCompactMessagesId,
    ModelCompactMessagesParams,
    ModelRequestId,
    ModelRequestParams,
    OperationConfigRole,
    ToolsetCallToolId,
    ToolsetCallToolParams,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetGetToolsParams,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation, RegisteredOperationBackend
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition

from ._operation_names import DBOSOperationNamer
from ._utils import StepConfig

P = TypeVar('P')
W = TypeVar('W')
R = TypeVar('R')


class DBOSOperationConfig:
    def __init__(self, *, model: StepConfig, event: StepConfig, tool: StepConfig) -> None:
        self._model = model
        self._event = event
        self._tool = tool

    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> StepConfig:
        if role == 'model':
            return self._model
        if role == 'event':
            return self._event
        return self._tool

    def for_tool(
        self,
        role: OperationConfigRole,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> StepConfig | Literal[False]:
        return self._tool


class DBOSBoundOperation(Generic[P, W, R]):
    def __init__(
        self,
        operation: DurableOperation[P, W, R],
        step: Callable[..., Any],
        dispatch: Callable[[Callable[..., Any], P], Any],
    ) -> None:
        self._operation = operation
        self.step = step
        self._dispatch = dispatch
        self._step_getter: Callable[[], Callable[..., Any]] = lambda: self.step

    @property
    def operation(self) -> DurableOperation[P, W, R]:
        return self._operation

    async def __call__(self, params: P, *, config: object | None = None) -> R:
        return cast(R, await self._dispatch(self._step_getter(), params))

    def use_step_getter(self, getter: Callable[[], Callable[..., Any]]) -> None:
        self._step_getter = getter


class DBOSOperationBackend(RegisteredOperationBackend[StepConfig]):
    """Register typed durable-operation handlers as DBOS steps during agent binding."""

    def __init__(self, *, agent_name: str, config: DBOSOperationConfig) -> None:
        super().__init__(namer=DBOSOperationNamer(agent_name), config=config)

    def register(
        self,
        operation: DurableOperation[P, W, R],
        *,
        name: str,
        config: StepConfig,
    ) -> tuple[BoundDurableOperation[P, W, R], Sequence[Callable[..., object]]]:
        if isinstance(operation.operation_id, CapabilityOperationId):
            step, dispatch = self._bind_capability(operation, name, config)
        elif isinstance(
            operation.operation_id,
            (ModelRequestId, ModelCompactMessagesId, ModelCancelSuspendedResponseId, EventStreamHandlerId),
        ):
            step, dispatch = self._bind_model_or_event(operation, name, config)
        else:
            step, dispatch = self._bind_toolset(operation, name, config)

        bound_operation = DBOSBoundOperation(operation, step, dispatch)
        return bound_operation, (step,)

    def _bind_capability(
        self, operation: DurableOperation[P, W, R], name: str, step_config: StepConfig
    ) -> tuple[Callable[..., Any], Callable[[Callable[..., Any], P], Any]]:
        async def capability_step(params: CapabilityOperationParams) -> object:
            return operation.result_codec.dump(await operation.handler(cast(P, params)))

        step = DBOS.step(name=name, **step_config)(capability_step)

        async def dispatch(step: Callable[..., Any], params: P) -> R:
            return operation.result_codec.load(await step(params))

        return step, dispatch

    def _bind_model_or_event(
        self, operation: DurableOperation[P, W, R], name: str, step_config: StepConfig
    ) -> tuple[Callable[..., Any], Callable[[Callable[..., Any], P], Any]]:

        match operation.operation_id:
            case ModelRequestId():

                async def model_step(
                    model_id: str | None,
                    messages: list[_messages.ModelMessage],
                    model_settings: ModelSettings | None,
                    model_request_parameters: ModelRequestParameters,
                    run_context: RunContext[Any],
                ) -> object:
                    params = ModelRequestParams(
                        model_id, messages, model_settings, model_request_parameters, run_context
                    )
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(model_step)

                async def dispatch_model(step: Callable[..., Any], params: P) -> R:
                    model_params = cast(ModelRequestParams, params)
                    payload = await step(
                        model_params.model_id,
                        model_params.messages,
                        model_params.model_settings,
                        model_params.model_request_parameters,
                        model_params.run_context,
                    )
                    return operation.result_codec.load(payload)

                dispatch: Callable[[Callable[..., Any], P], Any] = dispatch_model

            case ModelCancelSuspendedResponseId():

                async def cancel_step(
                    model_id: str | None, response: ModelResponse, run_context: RunContext[Any]
                ) -> object:
                    params = ModelCancelSuspendedResponseParams(model_id, response, run_context)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(cancel_step)

                async def dispatch_cancel(step: Callable[..., Any], params: P) -> R:
                    cancel_params = cast(ModelCancelSuspendedResponseParams, params)
                    payload = await step(cancel_params.model_id, cancel_params.response, cancel_params.run_context)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_cancel

            case ModelCompactMessagesId():

                async def compact_step(
                    model_id: str | None,
                    request_context: ModelRequestContext,
                    instructions: str | None,
                    run_context: RunContext[Any],
                ) -> object:
                    params = ModelCompactMessagesParams(model_id, request_context, instructions, run_context)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(compact_step)

                async def dispatch_compact(step: Callable[..., Any], params: P) -> R:
                    compact_params = cast(ModelCompactMessagesParams, params)
                    payload = await step(
                        compact_params.model_id,
                        compact_params.request_context,
                        compact_params.instructions,
                        compact_params.run_context,
                    )
                    return operation.result_codec.load(payload)

                dispatch = dispatch_compact

            case EventStreamHandlerId():

                async def event_step(event: _messages.AgentStreamEvent, run_context: RunContext[Any]) -> object:
                    params = EventStreamHandlerParams(event, run_context)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(event_step)

                async def dispatch_event(step: Callable[..., Any], params: P) -> R:
                    event_params = cast(EventStreamHandlerParams, params)
                    payload = await step(event_params.event, event_params.run_context)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_event

            case _ as operation_id:
                raise TypeError(f'DBOS operation {operation_id!r} is not a model or event operation')

        return step, dispatch

    def _bind_toolset(
        self, operation: DurableOperation[P, W, R], name: str, step_config: StepConfig
    ) -> tuple[Callable[..., Any], Callable[[Callable[..., Any], P], Any]]:
        match operation.operation_id:
            case ToolsetGetToolsId() | ToolsetGetInstructionsId():

                async def discovery_step(run_context: RunContext[Any]) -> object:
                    params = ToolsetGetToolsParams(run_context)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(discovery_step)

                async def dispatch_discovery(step: Callable[..., Any], params: P) -> R:
                    discovery_params = cast(ToolsetGetToolsParams, params)
                    payload = await step(discovery_params.ctx)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_discovery

            case ToolsetCallToolId(toolset_kind='mcp'):

                async def mcp_call_step(
                    tool_name: str,
                    tool_args: dict[str, Any],
                    run_context: RunContext[Any],
                    tool: ToolsetTool[Any],
                ) -> object:
                    params = ToolsetCallToolParams(tool_name, tool_args, run_context, tool)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(mcp_call_step)

                async def dispatch_mcp_call(step: Callable[..., Any], params: P) -> R:
                    call_params = cast(ToolsetCallToolParams, params)
                    payload = await step(call_params.name, call_params.tool_args, call_params.ctx, call_params.tool)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_mcp_call

            case ToolsetCallToolId(toolset_kind='dynamic') | ToolsetValidateToolArgumentsId(toolset_kind='dynamic'):

                async def dynamic_call_step(
                    tool_name: str,
                    tool_args: dict[str, Any],
                    run_context: RunContext[Any],
                    tool_def: ToolDefinition | None,
                ) -> object:
                    params = DynamicToolsetCallToolParams(tool_name, tool_args, run_context, tool_def)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(dynamic_call_step)

                async def dispatch_dynamic_call(step: Callable[..., Any], params: P) -> R:
                    call_params = cast(DynamicToolsetCallToolParams, params)
                    payload = await step(call_params.name, call_params.tool_args, call_params.ctx, call_params.tool_def)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_dynamic_call

            case _ as operation_id:
                raise TypeError(f'DBOS operation {operation_id!r} is not registered by this backend yet')

        return step, dispatch

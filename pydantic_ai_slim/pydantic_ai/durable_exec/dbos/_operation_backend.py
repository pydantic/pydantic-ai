from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Generic, Literal, TypeVar, cast

from dbos import DBOS

from pydantic_ai import ToolsetTool, messages as _messages
from pydantic_ai.durable_exec._base import (
    _CallToolParams,  # pyright: ignore[reportPrivateUsage]
    _CancelSuspendedResponseParams,  # pyright: ignore[reportPrivateUsage]
    _DynamicCallToolParams,  # pyright: ignore[reportPrivateUsage]
    _EventStreamHandlerParams,  # pyright: ignore[reportPrivateUsage]
    _GetToolsParams,  # pyright: ignore[reportPrivateUsage]
    _ModelRequestParams,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.durable_exec._operation import (
    CallToolId,
    CancelSuspendedResponseId,
    DurableOperation,
    DurableOperationId,
    EventStreamHandlerId,
    GetInstructionsId,
    GetToolsId,
    ModelRequestId,
    OperationConfigRole,
)
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation, RegisteredOperationBackend
from pydantic_ai.durable_exec._operation_names import DBOSOperationNamer
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

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
        if role is OperationConfigRole.MODEL:
            return self._model
        if role is OperationConfigRole.EVENT:
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

    def _register(
        self,
        operation: DurableOperation[P, W, R],
        *,
        name: str,
        config: StepConfig,
    ) -> tuple[BoundDurableOperation[P, W, R], Sequence[Callable[..., object]]]:
        if isinstance(operation.operation_id, (ModelRequestId, CancelSuspendedResponseId, EventStreamHandlerId)):
            step, dispatch = self._bind_model_or_event(operation, name, config)
        else:
            step, dispatch = self._bind_toolset(operation, name, config)

        bound_operation = DBOSBoundOperation(operation, step, dispatch)
        return bound_operation, (step,)

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
                    params = _ModelRequestParams(
                        model_id, messages, model_settings, model_request_parameters, run_context
                    )
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(model_step)

                async def dispatch_model(step: Callable[..., Any], params: P) -> R:
                    model_params = cast(_ModelRequestParams, params)
                    payload = await step(
                        model_params.model_id,
                        model_params.messages,
                        model_params.model_settings,
                        model_params.model_request_parameters,
                        model_params.run_context,
                    )
                    return operation.result_codec.load(payload)

                dispatch: Callable[[Callable[..., Any], P], Any] = dispatch_model

            case CancelSuspendedResponseId():

                async def cancel_step(
                    model_id: str | None, response: ModelResponse, run_context: RunContext[Any]
                ) -> object:
                    params = _CancelSuspendedResponseParams(model_id, response, run_context)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(cancel_step)

                async def dispatch_cancel(step: Callable[..., Any], params: P) -> R:
                    cancel_params = cast(_CancelSuspendedResponseParams, params)
                    payload = await step(cancel_params.model_id, cancel_params.response, cancel_params.run_context)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_cancel

            case EventStreamHandlerId():

                async def event_step(event: _messages.AgentStreamEvent, run_context: RunContext[Any]) -> object:
                    params = _EventStreamHandlerParams(event, run_context)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(event_step)

                async def dispatch_event(step: Callable[..., Any], params: P) -> R:
                    event_params = cast(_EventStreamHandlerParams, params)
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
            case GetToolsId() | GetInstructionsId():

                async def discovery_step(run_context: RunContext[Any]) -> object:
                    params = _GetToolsParams(run_context)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(discovery_step)

                async def dispatch_discovery(step: Callable[..., Any], params: P) -> R:
                    discovery_params = cast(_GetToolsParams, params)
                    payload = await step(discovery_params.ctx)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_discovery

            case CallToolId(toolset_kind='mcp'):

                async def mcp_call_step(
                    tool_name: str,
                    tool_args: dict[str, Any],
                    run_context: RunContext[Any],
                    tool: ToolsetTool[Any],
                ) -> object:
                    params = _CallToolParams(tool_name, tool_args, run_context, tool)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(mcp_call_step)

                async def dispatch_mcp_call(step: Callable[..., Any], params: P) -> R:
                    call_params = cast(_CallToolParams, params)
                    payload = await step(call_params.name, call_params.tool_args, call_params.ctx, call_params.tool)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_mcp_call

            case CallToolId(toolset_kind='dynamic'):

                async def dynamic_call_step(
                    tool_name: str, tool_args: dict[str, Any], run_context: RunContext[Any]
                ) -> object:
                    params = _DynamicCallToolParams(tool_name, tool_args, run_context)
                    return operation.result_codec.dump(await operation.handler(cast(P, params)))

                step = DBOS.step(name=name, **step_config)(dynamic_call_step)

                async def dispatch_dynamic_call(step: Callable[..., Any], params: P) -> R:
                    call_params = cast(_DynamicCallToolParams, params)
                    payload = await step(call_params.name, call_params.tool_args, call_params.ctx)
                    return operation.result_codec.load(payload)

                dispatch = dispatch_dynamic_call

            case _ as operation_id:
                raise TypeError(f'DBOS operation {operation_id!r} is not registered by this backend yet')

        return step, dispatch

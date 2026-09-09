from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

from temporalio import activity
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec._operation import (
    CapabilityOperationId,
    DurableOperation,
    DurableOperationConfig,
    DurableOperationId,
    EventStreamHandlerId,
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelRequestId,
    OperationConfigRole,
    ParameterTransport,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation, RegisteredOperationBackend

from ._activity_execution import execute_activity
from ._operation_names import TemporalOperationNamer
from ._toolset import heartbeating, model_response_payload_errors

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')


class TemporalParameterTransport(ParameterTransport[ParamsT, WireT], Protocol[ParamsT, WireT]):
    wire_type: object
    result_type: object

    @abstractmethod
    def dump(self, params: ParamsT) -> WireT: ...

    @abstractmethod
    def load(self, payload: WireT, *, runtime: object) -> ParamsT: ...


class _ModelParams(Protocol):
    model_id: str | None


class _EventParams(Protocol):
    event: Any


class TemporalOperationConfig(DurableOperationConfig[ActivityConfig]):
    def __init__(
        self,
        *,
        model: ActivityConfig,
        event: ActivityConfig,
        tool: ActivityConfig,
        resolve_tool: Callable[[DurableOperationId, object | None, str], ActivityConfig | Literal[False]],
    ) -> None:
        self._model = model
        self._event = event
        self._tool = tool
        self._resolve_tool = resolve_tool

    def base(self, role: OperationConfigRole, *, operation_id: DurableOperationId) -> ActivityConfig:
        if role == 'model':
            return self._model
        if role == 'event':
            return self._event
        return self._tool

    def for_tool(
        self,
        role: OperationConfigRole,
        *,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> ActivityConfig | Literal[False]:
        return self._resolve_tool(operation_id, tool, tool_name)


class TemporalBoundOperation(BoundDurableOperation[ParamsT, WireT, ResultT], Generic[ParamsT, WireT, ResultT]):
    def __init__(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        registration: Callable[..., Awaitable[ResultT]],
        config: ActivityConfig,
    ) -> None:
        self._operation = operation
        self.registration = registration
        self._config = config

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        payload = self._operation.parameter_transport.dump(params)
        activity_config = cast(ActivityConfig, config or self._config).copy()
        operation_id = self._operation.operation_id
        model_name = ''
        if isinstance(operation_id, ModelRequestId):
            model_name = cast(_ModelParams, params).model_id or operation_id.model_name
            suffix = ' (stream)' if operation_id.streaming else ''
            activity_config['summary'] = f'request model: {model_name}{suffix}'
        elif isinstance(operation_id, ModelCancelSuspendedResponseId):
            model_name = cast(_ModelParams, params).model_id or operation_id.model_name
            activity_config['summary'] = f'cancel suspended response: {model_name}'
        elif isinstance(operation_id, ModelCompactMessagesId):
            model_name = cast(_ModelParams, params).model_id or operation_id.model_name
            activity_config['summary'] = f'compact messages: {model_name}'
        elif isinstance(operation_id, ToolsetCallToolId):
            tool_name = cast(Any, params).name
            activity_config['summary'] = f'call tool: {operation_id.toolset_id}:{tool_name}'
        elif isinstance(operation_id, ToolsetValidateToolArgumentsId):
            tool_name = cast(Any, params).name
            activity_config['summary'] = f'validate tool args: {operation_id.toolset_id}:{tool_name}'
        elif isinstance(operation_id, ToolsetGetToolsId):
            activity_config['summary'] = f'get tools: {operation_id.toolset_id}'
        elif isinstance(operation_id, ToolsetGetInstructionsId):
            activity_config['summary'] = f'get instructions: {operation_id.toolset_id}'
        elif isinstance(operation_id, CapabilityOperationId):
            activity_config['summary'] = f'capability: {operation_id.capability_id}.{operation_id.operation}'
        elif isinstance(operation_id, EventStreamHandlerId):
            event = cast(_EventParams, params).event
            activity_config['summary'] = f'handle event: {event.event_kind}'
        else:
            # New operation ids use their stable activity name as the default summary. Their
            # parameter transport must implement `TemporalParameterTransport`, including
            # `wire_type` and `result_type`, so the payload converter can inspect the activity.
            activity_config['summary'] = self.registration.__name__

        if isinstance(operation_id, ModelRequestId | ModelCompactMessagesId):
            with model_response_payload_errors(model_name):
                return await execute_activity(
                    activity=self.registration, args=cast(Sequence[Any], payload), **activity_config
                )
        return await execute_activity(activity=self.registration, args=cast(Sequence[Any], payload), **activity_config)


class TemporalOperationBackend(RegisteredOperationBackend[ActivityConfig]):
    """Own Temporal activity definitions while preserving their existing callables."""

    def __init__(
        self,
        *,
        agent_name: str,
        deps_type: type[Any],
        model_config: ActivityConfig,
        event_config: ActivityConfig,
        tool_config: ActivityConfig,
        resolve_tool_config: Callable[[DurableOperationId, object | None, str], ActivityConfig | Literal[False]],
        runtime: object | None = None,
    ) -> None:
        super().__init__(
            namer=TemporalOperationNamer(agent_name),
            config=TemporalOperationConfig(
                model=model_config, event=event_config, tool=tool_config, resolve_tool=resolve_tool_config
            ),
        )
        self._deps_type = deps_type
        self._runtime = runtime

    def move_registration_to_end(self, registration: Callable[..., object]) -> None:
        self._registrations.remove(registration)
        self._registrations.append(registration)

    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: ActivityConfig,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]:
        transport = cast(TemporalParameterTransport[ParamsT, WireT], operation.parameter_transport)

        async def activity_handler(params: Any, deps: Any = None) -> ResultT:
            semantic_params = transport.load(cast(WireT, (params, deps)), runtime=self._runtime)
            async with heartbeating():
                return await operation.handler(semantic_params)

        # Existing operation transports retain their shipped wire dataclasses and activity
        # signatures. New operation ids ride this generic registration path and only need a
        # `TemporalParameterTransport` with `wire_type` and `result_type`.
        # Temporal's Pydantic payload converter deserializes `deps` by inspecting the
        # registered callable, so patch the exact function that the SDK will inspect.
        activity_handler.__annotations__ = {
            'params': transport.wire_type,
            'deps': self._deps_type | None,
            'return': transport.result_type,
        }
        registration = activity.defn(name=name)(activity_handler)
        bound = TemporalBoundOperation(
            operation,
            registration=cast(Callable[..., Awaitable[ResultT]], registration),
            config=config,
        )
        return bound, (registration,)

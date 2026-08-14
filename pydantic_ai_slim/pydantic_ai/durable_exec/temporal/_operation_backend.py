from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

from temporalio import activity
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec._operation import (
    CancelSuspendedResponseId,
    DurableOperation,
    DurableOperationId,
    EventStreamHandlerId,
    IdentityParameterTransport,
    ModelRequestId,
    NoCacheIdentity,
    OperationConfigRole,
    TypedResultCodec,
)
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation, RegisteredOperationBackend
from pydantic_ai.durable_exec._operation_names import TemporalOperationNamer

from ._activity_execution import execute_activity
from ._toolset import heartbeating, model_response_payload_errors

P = TypeVar('P')
W = TypeVar('W')
R = TypeVar('R')


class TemporalParameterTransport(Protocol[P, W]):
    wire_type: object
    result_type: object

    @abstractmethod
    def dump(self, params: P) -> W: ...

    @abstractmethod
    def load(self, payload: W, *, runtime: object) -> P: ...


class _ModelParams(Protocol):
    model_id: str | None


class _EventParams(Protocol):
    event: Any


class TemporalOperationConfig:
    def __init__(self, *, model: ActivityConfig, event: ActivityConfig, tool: ActivityConfig) -> None:
        self._model = model
        self._event = event
        self._tool = tool

    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> ActivityConfig:
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
    ) -> ActivityConfig | Literal[False]:
        return self._tool


class TemporalBoundOperation(Generic[P, W, R]):
    def __init__(
        self,
        operation: DurableOperation[P, W, R],
        registration: Callable[..., Awaitable[R]],
        config: ActivityConfig,
        *,
        dispatch_activity: bool,
    ) -> None:
        self._operation = operation
        self.registration = registration
        self._config = config
        self._dispatch_activity = dispatch_activity

    @property
    def operation(self) -> DurableOperation[P, W, R]:
        return self._operation

    async def __call__(self, params: P, *, config: object | None = None) -> R:
        payload = self._operation.parameter_transport.dump(params)
        if not self._dispatch_activity:
            return await self.registration(*cast(Sequence[Any], payload))
        activity_config = cast(ActivityConfig, config or self._config).copy()
        operation_id = self._operation.operation_id
        model_name = ''
        if isinstance(operation_id, ModelRequestId):
            model_name = cast(_ModelParams, params).model_id or operation_id.model_name
            suffix = ' (stream)' if operation_id.streaming else ''
            activity_config['summary'] = f'request model: {model_name}{suffix}'
        elif isinstance(operation_id, CancelSuspendedResponseId):
            model_name = cast(_ModelParams, params).model_id or operation_id.model_name
            activity_config['summary'] = f'cancel suspended response: {model_name}'
        else:
            assert isinstance(operation_id, EventStreamHandlerId)
            event = cast(_EventParams, params).event
            activity_config['summary'] = f'handle event: {event.event_kind}'

        if isinstance(operation_id, ModelRequestId):
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
        runtime: object | None = None,
    ) -> None:
        super().__init__(
            namer=TemporalOperationNamer(agent_name),
            config=TemporalOperationConfig(model=model_config, event=event_config, tool=tool_config),
        )
        self._deps_type = deps_type
        self._runtime = runtime
        self._activity_to_register: Callable[..., Awaitable[Any]] | None = None

    def register_activity(
        self,
        fn: Callable[..., Awaitable[Any]],
        *,
        operation_id: DurableOperationId,
        config_role: OperationConfigRole,
    ) -> TemporalBoundOperation[tuple[Any, ...], tuple[Any, ...], Any]:
        """Register one existing activity body through the typed backend."""

        async def handler(params: tuple[Any, ...]) -> Any:
            return await fn(*params)

        operation = DurableOperation(
            operation_id=operation_id,
            handler=handler,
            parameter_transport=IdentityParameterTransport[tuple[Any, ...]](),
            cache_identity=NoCacheIdentity[tuple[Any, ...]](),
            result_codec=TypedResultCodec(Any, mode='identity'),
            config_role=config_role,
        )
        self._activity_to_register = fn
        try:
            bound = self.bind(operation)
        finally:
            self._activity_to_register = None
        assert isinstance(bound, TemporalBoundOperation)
        return bound

    def adopt_registrations(self, registrations: Sequence[Callable[..., object]]) -> None:
        """Adopt already-decorated toolset activities without changing their identity."""
        self._registrations.extend(registrations)

    def move_registration_to_end(self, registration: Callable[..., object]) -> None:
        self._registrations.remove(registration)
        self._registrations.append(registration)

    def _register(
        self,
        operation: DurableOperation[P, W, R],
        *,
        name: str,
        config: ActivityConfig,
    ) -> tuple[BoundDurableOperation[P, W, R], Sequence[Callable[..., object]]]:
        fn = self._activity_to_register
        if fn is not None:
            # Step 8 compatibility path for Temporal toolsets, migrated in Step 10.
            fn.__annotations__['deps'] = self._deps_type | None
            registration = activity.defn(name=name)(fn)
            bound = TemporalBoundOperation(
                operation, cast(Callable[..., Awaitable[R]], registration), config, dispatch_activity=False
            )
            return bound, (registration,)

        transport = cast(TemporalParameterTransport[P, W], operation.parameter_transport)

        async def activity_handler(params: Any, deps: Any = None) -> R:
            semantic_params = transport.load(cast(W, (params, deps)), runtime=self._runtime)
            async with heartbeating():
                return await operation.handler(semantic_params)

        # Temporal's Pydantic payload converter deserializes `deps` by inspecting the
        # registered callable, so patch the exact function that the SDK will inspect.
        activity_handler.__annotations__ = {
            'params': transport.wire_type,
            'deps': self._deps_type | None,
            'return': transport.result_type,
        }
        registration = activity.defn(name=name)(activity_handler)
        bound = TemporalBoundOperation(
            operation, cast(Callable[..., Awaitable[R]], registration), config, dispatch_activity=True
        )
        return bound, (registration,)

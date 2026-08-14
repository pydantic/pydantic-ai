from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Generic, Literal, TypeVar, cast

from temporalio import activity
from temporalio.workflow import ActivityConfig

from pydantic_ai.durable_exec._operation import (
    DurableOperation,
    DurableOperationId,
    IdentityParameterTransport,
    NoCacheIdentity,
    OperationConfigRole,
    TypedResultCodec,
)
from pydantic_ai.durable_exec._operation_backend import BoundDurableOperation, RegisteredOperationBackend
from pydantic_ai.durable_exec._operation_names import TemporalOperationNamer

P = TypeVar('P')
W = TypeVar('W')
R = TypeVar('R')


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
    def __init__(self, operation: DurableOperation[P, W, R], registration: Callable[..., Awaitable[R]]) -> None:
        self._operation = operation
        self.registration = registration

    @property
    def operation(self) -> DurableOperation[P, W, R]:
        return self._operation

    async def __call__(self, params: P, *, config: object | None = None) -> R:
        payload = self._operation.parameter_transport.dump(params)
        return await self.registration(*cast(Sequence[Any], payload))


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
    ) -> None:
        super().__init__(
            namer=TemporalOperationNamer(agent_name),
            config=TemporalOperationConfig(model=model_config, event=event_config, tool=tool_config),
        )
        self._deps_type = deps_type
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

    def _register(
        self,
        operation: DurableOperation[P, W, R],
        *,
        name: str,
        config: ActivityConfig,
    ) -> tuple[BoundDurableOperation[P, W, R], Sequence[Callable[..., object]]]:
        fn = self._activity_to_register
        assert fn is not None
        # Temporal's Pydantic payload converter deserializes `deps` by inspecting the
        # registered callable, so patch the exact function that the SDK will inspect.
        fn.__annotations__['deps'] = self._deps_type | None
        registration = activity.defn(name=name)(fn)
        bound = TemporalBoundOperation(operation, cast(Callable[..., Awaitable[R]], registration))
        return bound, (registration,)

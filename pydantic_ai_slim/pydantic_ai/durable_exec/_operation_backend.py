from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, Literal, Protocol, TypeVar

from ._operation import DurableOperation, DurableOperationConfig, resolve_tool_operation_config
from ._operation_names import DurableOperationNamer

P = TypeVar('P')
W = TypeVar('W')
R = TypeVar('R')
ConfigT = TypeVar('ConfigT')
ConfigT_co = TypeVar('ConfigT_co', covariant=True)
P_bound = TypeVar('P_bound')
W_bound = TypeVar('W_bound')
R_bound = TypeVar('R_bound')


class BoundDurableOperation(Generic[P_bound, W_bound, R_bound], Protocol):
    @property
    @abstractmethod
    def operation(self) -> DurableOperation[P_bound, W_bound, R_bound]: ...

    @abstractmethod
    async def __call__(self, params: P_bound, *, config: object | None = None) -> R_bound: ...


class DurableOperationBackend(Protocol[ConfigT_co]):
    @abstractmethod
    def bind(self, operation: DurableOperation[P, W, R]) -> BoundDurableOperation[P, W, R]: ...

    @abstractmethod
    def config_for_tool(
        self,
        operation: DurableOperation[P, W, R],
        tool: object | None,
        tool_name: str,
    ) -> ConfigT_co | Literal[False]: ...

    @abstractmethod
    def registrations(self) -> Sequence[Callable[..., object]]: ...


class _CallableBoundOperation(Generic[P, W, R]):
    def __init__(
        self, operation: DurableOperation[P, W, R], dispatch: Callable[[P, object | None], Awaitable[R]]
    ) -> None:
        self._operation = operation
        self._dispatch = dispatch

    @property
    def operation(self) -> DurableOperation[P, W, R]:
        return self._operation

    async def __call__(self, params: P, *, config: object | None = None) -> R:
        return await self._dispatch(params, config)


class CallableOperationBackend(ABC, Generic[ConfigT]):
    def __init__(self, *, namer: DurableOperationNamer, config: DurableOperationConfig[ConfigT]) -> None:
        self._namer = namer
        self._config = config

    def bind(self, operation: DurableOperation[P, W, R]) -> BoundDurableOperation[P, W, R]:
        async def dispatch(params: P, explicit_config: object | None) -> R:
            invocation_name = self._namer.invocation_name(operation.operation_id, params)
            resolved_config = (
                explicit_config
                if explicit_config is not None
                else self._config.base(operation.config_role, operation.operation_id)
            )
            cache_key = operation.cache_identity.project(params)

            async def body() -> object:
                return operation.result_codec.dump(await operation.handler(params))

            payload = await self._execute(
                name=invocation_name.operation_name,
                body=body,
                cache_key=cache_key,
                config=resolved_config,
            )
            return operation.result_codec.load(payload)

        return _CallableBoundOperation(operation, dispatch)

    def config_for_tool(
        self,
        operation: DurableOperation[P, W, R],
        tool: object | None,
        tool_name: str,
    ) -> ConfigT | Literal[False]:
        return resolve_tool_operation_config(self._config, operation, tool, tool_name)

    def registrations(self) -> Sequence[Callable[..., object]]:
        return ()

    @abstractmethod
    async def _execute(
        self,
        *,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object: ...


class RegisteredOperationBackend(ABC, Generic[ConfigT]):
    """Base for backends that register named SDK handlers while binding operations."""

    def __init__(self, *, namer: DurableOperationNamer, config: DurableOperationConfig[ConfigT]) -> None:
        self._namer = namer
        self._config = config
        self._registrations: list[Callable[..., object]] = []

    def bind(self, operation: DurableOperation[P, W, R]) -> BoundDurableOperation[P, W, R]:
        name = self._namer.operation_name(operation.operation_id)
        config = self._config.base(operation.config_role, operation.operation_id)
        bound_operation, registrations = self._register(operation, name=name, config=config)
        self._registrations.extend(registrations)
        return bound_operation

    def config_for_tool(
        self,
        operation: DurableOperation[P, W, R],
        tool: object | None,
        tool_name: str,
    ) -> ConfigT | Literal[False]:
        return resolve_tool_operation_config(self._config, operation, tool, tool_name)

    def registrations(self) -> Sequence[Callable[..., object]]:
        return self._registrations

    @abstractmethod
    def _register(
        self,
        operation: DurableOperation[P, W, R],
        *,
        name: str,
        config: ConfigT,
    ) -> tuple[BoundDurableOperation[P, W, R], Sequence[Callable[..., object]]]: ...


class LegacyDurableCapability(Protocol):
    async def run_durable_unit(
        self,
        name: str,
        fn: Callable[[], Awaitable[object]],
        *,
        inputs: tuple[object, ...],
        config: object,
    ) -> object: ...


class LegacyCallableBackend(CallableOperationBackend[ConfigT]):
    def __init__(
        self,
        capability: LegacyDurableCapability,
        *,
        namer: DurableOperationNamer,
        config: DurableOperationConfig[ConfigT],
    ) -> None:
        super().__init__(namer=namer, config=config)
        self._capability = capability

    async def _execute(
        self,
        *,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object:
        return await self._capability.run_durable_unit(name, body, inputs=cache_key, config=config)

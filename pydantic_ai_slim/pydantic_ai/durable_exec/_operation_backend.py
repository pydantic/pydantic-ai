from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, Protocol, TypeVar

from ._operation import DurableOperation, DurableOperationConfig
from ._operation_names import DurableOperationNamer

P = TypeVar('P')
W = TypeVar('W')
R = TypeVar('R')
ConfigT = TypeVar('ConfigT')
P_bound = TypeVar('P_bound')
W_bound = TypeVar('W_bound')
R_bound = TypeVar('R_bound')


class BoundDurableOperation(Generic[P_bound, W_bound, R_bound], Protocol):
    @property
    @abstractmethod
    def operation(self) -> DurableOperation[P_bound, W_bound, R_bound]: ...

    @abstractmethod
    async def __call__(self, params: P_bound, *, config: object | None = None) -> R_bound: ...


class DurableOperationBackend(Protocol):
    def bind(self, operation: DurableOperation[P, W, R]) -> BoundDurableOperation[P, W, R]: ...

    def registrations(self) -> Sequence[Callable[..., object]]: ...


class _CallableBoundOperation(Generic[P, R]):
    def __init__(
        self, operation: DurableOperation[P, P, R], dispatch: Callable[[P, object | None], Awaitable[R]]
    ) -> None:
        self._operation = operation
        self._dispatch = dispatch

    @property
    def operation(self) -> DurableOperation[P, P, R]:
        return self._operation

    async def __call__(self, params: P, *, config: object | None = None) -> R:
        return await self._dispatch(params, config)


class CallableOperationBackend(ABC, Generic[ConfigT]):
    def __init__(self, *, namer: DurableOperationNamer, config: DurableOperationConfig[ConfigT]) -> None:
        self._namer = namer
        self._config = config

    def bind(self, operation: DurableOperation[P, P, R]) -> BoundDurableOperation[P, P, R]:
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

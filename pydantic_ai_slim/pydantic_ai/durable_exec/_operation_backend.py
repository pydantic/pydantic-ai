from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from typing import Generic, Literal, Protocol, TypeVar, cast

from ._operation import (
    DurableOperation,
    DurableOperationConfig,
    DurableOperationId,
    OperationConfigRole,
    resolve_tool_operation_config,
)
from ._operation_names import DurableOperationNamer, JournalOperationNamer

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
ConfigT = TypeVar('ConfigT')
ParamsT_bound = TypeVar('ParamsT_bound')
WireT_bound = TypeVar('WireT_bound')
ResultT_bound = TypeVar('ResultT_bound')


class BoundDurableOperation(Generic[ParamsT_bound, WireT_bound, ResultT_bound], Protocol):
    @property
    @abstractmethod
    def operation(self) -> DurableOperation[ParamsT_bound, WireT_bound, ResultT_bound]: ...

    @abstractmethod
    async def __call__(self, params: ParamsT_bound, *, config: object | None = None) -> ResultT_bound: ...

    # `config` remains engine-opaque because bound operations are stored across heterogeneous
    # engine implementations by `BaseDurabilityCapability`.


class DurableOperationBackend(ABC, Generic[ConfigT]):
    """Contract between `BaseDurabilityCapability` and an engine's durable primitive.

    Engine authors normally implement this by subclassing `CallableOperationBackend` or
    `RegisteredOperationBackend`. See
    [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    @abstractmethod
    def bind(
        self, operation: DurableOperation[ParamsT, WireT, ResultT]
    ) -> BoundDurableOperation[ParamsT, WireT, ResultT]: ...

    @abstractmethod
    def config_for_tool(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        tool: object | None,
        tool_name: str,
    ) -> ConfigT | Literal[False]: ...

    @abstractmethod
    def registrations(self) -> Sequence[Callable[..., object]]: ...


class _CallableBoundOperation(BoundDurableOperation[ParamsT, WireT, ResultT], Generic[ParamsT, WireT, ResultT]):
    def __init__(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        dispatch: Callable[[ParamsT, object | None], Awaitable[ResultT]],
    ) -> None:
        self._operation = operation
        self._dispatch = dispatch

    @property
    def operation(self) -> DurableOperation[ParamsT, WireT, ResultT]:
        return self._operation

    async def __call__(self, params: ParamsT, *, config: object | None = None) -> ResultT:
        return await self._dispatch(params, config)


class CallableOperationBackend(DurableOperationBackend[ConfigT]):
    """Base for engines that execute an async callback in a named durable unit.

    Subclasses implement `execute`; this base owns naming, configuration, cache identity, and
    result encoding. See the
    [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    def __init__(self, *, namer: DurableOperationNamer, config: DurableOperationConfig[ConfigT]) -> None:
        self._namer = namer
        self._config = config

    def bind(
        self, operation: DurableOperation[ParamsT, WireT, ResultT]
    ) -> BoundDurableOperation[ParamsT, WireT, ResultT]:
        async def dispatch(params: ParamsT, explicit_config: object | None) -> ResultT:
            label = operation.invocation_label(params) if operation.invocation_label is not None else None
            invocation_name = self._namer.invocation_name(operation.operation_id, label=label)
            # The bound protocol accepts engine-opaque explicit config because the capability
            # stores bound operations without carrying each backend's config parameter.
            resolved_config = (
                cast(ConfigT, explicit_config)
                if explicit_config is not None
                else self._config.base(operation.config_role, operation_id=operation.operation_id)
            )
            cache_key = operation.cache_identity.project(params)

            async def body() -> object:
                return operation.result_codec.dump(await operation.handler(params))

            payload = await self.execute(
                operation_id=operation.operation_id,
                name=invocation_name.operation_name,
                body=body,
                cache_key=cache_key,
                config=resolved_config,
            )
            return operation.result_codec.load(payload)

        return _CallableBoundOperation(operation, dispatch=dispatch)

    def config_for_tool(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        tool: object | None,
        tool_name: str,
    ) -> ConfigT | Literal[False]:
        return resolve_tool_operation_config(self._config, operation, tool=tool, tool_name=tool_name)

    def registrations(self) -> Sequence[Callable[..., object]]:
        return ()

    @abstractmethod
    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: ConfigT,
    ) -> object:
        """Execute `body` as one named durable unit.

        Args:
            operation_id: Typed identity of the semantic operation.
            name: Persisted durable unit name.
            body: Encoded semantic operation body.
            cache_key: Opaque hash inputs for engines that identify cached work by hash.
            config: Engine-specific durable unit configuration.
        """
        ...


class JournalCallableOperationBackend(CallableOperationBackend[ConfigT]):
    """Callable backend using the standard journal operation naming convention."""

    def __init__(
        self,
        *,
        agent_name: str,
        default_model_id: str | None = None,
        config: DurableOperationConfig[ConfigT],
    ) -> None:
        super().__init__(
            namer=JournalOperationNamer(agent_name, default_model_id=default_model_id or 'default'), config=config
        )


class RoleBasedOperationConfig(Generic[ConfigT]):
    """Resolve operation configuration from role defaults and an optional per-tool resolver."""

    def __init__(
        self,
        *,
        model: ConfigT,
        event: ConfigT,
        capability: ConfigT,
        tool: ConfigT,
        resolve_tool: Callable[[DurableOperationId, object | None, str], ConfigT | Literal[False]] | None = None,
    ) -> None:
        self._configs = {'model': model, 'event': event, 'capability': capability, 'tool': tool}
        self._resolve_tool = resolve_tool

    def base(self, role: OperationConfigRole, *, operation_id: DurableOperationId) -> ConfigT:
        if role == 'tool' and self._resolve_tool is not None:
            config = self._resolve_tool(operation_id, None, '')
            assert config is not False
            return config
        return self._configs[role]

    def for_tool(
        self,
        role: OperationConfigRole,
        *,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> ConfigT | Literal[False]:
        assert role == 'tool'
        if self._resolve_tool is not None:
            return self._resolve_tool(operation_id, tool, tool_name)
        return self._configs['tool']


class RegisteredOperationBackend(DurableOperationBackend[ConfigT]):
    """Base for engines that register named SDK handlers while binding operations.

    Subclasses implement `register`; this base collects the returned worker registrations and
    owns naming and configuration. See
    [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    def __init__(self, *, namer: DurableOperationNamer, config: DurableOperationConfig[ConfigT]) -> None:
        self._namer = namer
        self._config = config
        self._registrations: list[Callable[..., object]] = []

    def bind(
        self, operation: DurableOperation[ParamsT, WireT, ResultT]
    ) -> BoundDurableOperation[ParamsT, WireT, ResultT]:
        name = self._namer.operation_name(operation.operation_id)
        config = self._config.base(operation.config_role, operation_id=operation.operation_id)
        bound_operation, registrations = self.register(operation, name=name, config=config)
        self._registrations.extend(registrations)
        return bound_operation

    def config_for_tool(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        tool: object | None,
        tool_name: str,
    ) -> ConfigT | Literal[False]:
        return resolve_tool_operation_config(self._config, operation, tool=tool, tool_name=tool_name)

    def registrations(self) -> Sequence[Callable[..., object]]:
        return self._registrations

    @abstractmethod
    def register(
        self,
        operation: DurableOperation[ParamsT, WireT, ResultT],
        *,
        name: str,
        config: ConfigT,
    ) -> tuple[BoundDurableOperation[ParamsT, WireT, ResultT], Sequence[Callable[..., object]]]: ...

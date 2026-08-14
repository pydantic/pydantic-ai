from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar, cast

from ._codec import IDENTITY_CODEC, JSON_CODEC, DurabilityCodec

P = TypeVar('P')
W = TypeVar('W')
R = TypeVar('R')
ConfigT = TypeVar('ConfigT')
P_contra = TypeVar('P_contra', contravariant=True)
ConfigT_co = TypeVar('ConfigT_co', covariant=True)

ToolsetKind: TypeAlias = Literal['function', 'mcp', 'dynamic']


@dataclass(frozen=True)
class ModelRequestId:
    model_id: str | None
    streaming: bool
    model_name: str


@dataclass(frozen=True)
class CancelSuspendedResponseId:
    model_id: str | None
    model_name: str


@dataclass(frozen=True)
class EventStreamHandlerId:
    pass


@dataclass(frozen=True)
class GetToolsId:
    toolset_kind: ToolsetKind
    toolset_id: str


@dataclass(frozen=True)
class GetInstructionsId:
    toolset_id: str


@dataclass(frozen=True)
class ValidateToolArgumentsId:
    toolset_kind: ToolsetKind
    toolset_id: str


@dataclass(frozen=True)
class CallToolId:
    toolset_kind: ToolsetKind
    toolset_id: str


DurableOperationId: TypeAlias = (
    ModelRequestId
    | CancelSuspendedResponseId
    | EventStreamHandlerId
    | GetToolsId
    | GetInstructionsId
    | ValidateToolArgumentsId
    | CallToolId
)


@dataclass(frozen=True)
class OperationInvocation(Generic[P]):
    params: P
    config: object


class ParameterTransport(Generic[P, W], Protocol):
    @abstractmethod
    def dump(self, params: P) -> W: ...

    @abstractmethod
    def load(self, payload: W, *, runtime: object) -> P: ...


class CacheIdentity(Generic[P_contra], Protocol):
    @abstractmethod
    def project(self, params: P_contra) -> tuple[object, ...]: ...


class ResultCodec(Generic[R], Protocol):
    @abstractmethod
    def dump(self, value: R) -> object: ...

    @abstractmethod
    def load(self, payload: object) -> R: ...


class OperationConfigRole(str, Enum):
    MODEL = 'model'
    EVENT = 'event'
    TOOL_DISCOVERY = 'tool_discovery'
    TOOL_CALL = 'tool_call'
    TOOL_VALIDATION = 'tool_validation'


class DurableOperationConfig(Generic[ConfigT_co], Protocol):
    @abstractmethod
    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> ConfigT_co: ...

    @abstractmethod
    def for_tool(
        self,
        role: OperationConfigRole,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> ConfigT_co | Literal[False]: ...


@dataclass(frozen=True)
class DurableOperation(Generic[P, W, R]):
    operation_id: DurableOperationId
    handler: Callable[[P], Awaitable[R]]
    parameter_transport: ParameterTransport[P, W]
    cache_identity: CacheIdentity[P]
    result_codec: ResultCodec[R]
    config_role: OperationConfigRole


class IdentityParameterTransport(Generic[P]):
    def dump(self, params: P) -> P:
        return params

    def load(self, payload: P, *, runtime: object) -> P:
        return payload


class NoCacheIdentity(Generic[P]):
    def project(self, params: P) -> tuple[()]:
        return ()


class TypedResultCodec(Generic[R]):
    def __init__(self, result_type: object, *, mode: Literal['json', 'identity'] = 'json') -> None:
        self._result_type = result_type
        self._codec: DurabilityCodec = JSON_CODEC if mode == 'json' else IDENTITY_CODEC

    def dump(self, value: R) -> object:
        return self._codec.dump(self._result_type, value)

    def load(self, payload: object) -> R:
        return cast(R, self._codec.load(self._result_type, payload))

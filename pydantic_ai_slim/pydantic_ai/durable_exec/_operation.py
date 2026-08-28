from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar, cast

from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool

from ._codec import IDENTITY_CODEC, JSON_CODEC, DurabilityCodec

P = TypeVar('P')
W = TypeVar('W')
R = TypeVar('R')
ConfigT = TypeVar('ConfigT')
P_contra = TypeVar('P_contra', contravariant=True)
ConfigT_co = TypeVar('ConfigT_co', covariant=True)

ToolsetKind: TypeAlias = Literal['function', 'mcp', 'dynamic']
"""The leaf toolset categories an engine can configure and wrap.

Engine authors use this type for declarative lifecycle settings and per-tool configuration. See
[Building a durable execution backend](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
"""


@dataclass(frozen=True)
class ModelRequestId:
    """Identifies a model request operation in an engine's configuration resolver.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    model_id: str | None
    streaming: bool
    model_name: str


@dataclass(frozen=True)
class ModelRequestParams:
    model_id: str | None
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
    run_context: RunContext[Any]


@dataclass(frozen=True)
class ModelCancelSuspendedResponseId:
    """Identifies cancellation of a suspended model response for engine configuration.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    model_id: str | None
    model_name: str


@dataclass(frozen=True)
class ModelCancelSuspendedResponseParams:
    model_id: str | None
    response: ModelResponse
    run_context: RunContext[Any] | None


@dataclass(frozen=True)
class ModelCompactMessagesId:
    """Identifies a durable message-compaction operation for engine configuration.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    model_id: str | None
    model_name: str


@dataclass(frozen=True)
class ModelCompactMessagesParams:
    model_id: str | None
    request_context: ModelRequestContext
    instructions: str | None
    run_context: RunContext[Any]


@dataclass(frozen=True)
class EventStreamHandlerId:
    """Identifies a durable event-stream handler invocation for engine configuration.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    pass


@dataclass(frozen=True)
class EventStreamHandlerParams:
    event: AgentStreamEvent
    run_context: RunContext[Any]


@dataclass(frozen=True)
class CapabilityOperationId:
    """Identifies an operation contributed by a capability.

    Engine configuration receives these operations through the same backend as built-in model and
    tool operations. See the
    [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    capability_id: str
    operation: str


@dataclass(frozen=True)
class ToolsetGetToolsId:
    """Identifies durable tool discovery for a particular toolset.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    toolset_kind: ToolsetKind
    toolset_id: str


@dataclass(frozen=True)
class ToolsetGetToolsParams:
    ctx: RunContext[Any]


@dataclass(frozen=True)
class ToolsetGetInstructionsId:
    """Identifies durable instruction discovery for an MCP toolset.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    toolset_id: str


@dataclass(frozen=True)
class ToolsetValidateToolArgumentsId:
    """Identifies durable argument validation for a particular toolset.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    toolset_kind: ToolsetKind
    toolset_id: str


@dataclass(frozen=True)
class ToolsetCallToolId:
    """Identifies durable tool execution for a particular toolset.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    toolset_kind: ToolsetKind
    toolset_id: str


@dataclass(frozen=True)
class ToolsetCallToolParams:
    name: str
    tool_args: dict[str, Any]
    ctx: RunContext[Any]
    tool: ToolsetTool[Any] | None


@dataclass(frozen=True)
class DynamicToolsetCallToolParams:
    name: str
    tool_args: dict[str, Any]
    ctx: RunContext[Any]
    tool_def: ToolDefinition | None = None


DurableOperationId: TypeAlias = (
    ModelRequestId
    | ModelCompactMessagesId
    | ModelCancelSuspendedResponseId
    | CapabilityOperationId
    | EventStreamHandlerId
    | ToolsetGetToolsId
    | ToolsetGetInstructionsId
    | ToolsetValidateToolArgumentsId
    | ToolsetCallToolId
)
"""The closed union of operation identifiers passed to engine configuration.

Match on every variant when configuration depends on the operation. See
[durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
"""


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


OperationConfigRole: TypeAlias = Literal['model', 'event', 'tool', 'capability']
"""The coarse configuration bucket for an operation; its ID carries the fine-grained identity."""


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


def resolve_tool_operation_config(
    config: DurableOperationConfig[ConfigT],
    operation: DurableOperation[P, W, R],
    tool: object | None,
    tool_name: str,
) -> ConfigT | Literal[False]:
    """Resolve tool configuration shared by callable and registered backends."""
    return config.for_tool(operation.config_role, operation.operation_id, tool, tool_name)


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

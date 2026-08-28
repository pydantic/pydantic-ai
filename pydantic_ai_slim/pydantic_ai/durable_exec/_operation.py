from __future__ import annotations

from abc import abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import KW_ONLY, dataclass
from typing import Any, Generic, Literal, Protocol, TypeAlias, TypeVar, cast

from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.toolsets.abstract import ToolsetTool

from ._codec import IDENTITY_CODEC, JSON_CODEC, DurabilityCodec

ParamsT = TypeVar('ParamsT')
WireT = TypeVar('WireT')
ResultT = TypeVar('ResultT')
ConfigT = TypeVar('ConfigT')
ParamsT_contra = TypeVar('ParamsT_contra', contravariant=True)
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

    model_id: str | None = None
    _: KW_ONLY
    streaming: bool = False
    model_name: str


@dataclass(frozen=True)
class ModelRequestParams:
    model_id: str | None = None
    _: KW_ONLY
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
    run_context: RunContext[Any]


@dataclass(frozen=True)
class ModelCancelSuspendedResponseId:
    """Identifies cancellation of a suspended model response for engine configuration.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    model_id: str | None = None
    _: KW_ONLY
    model_name: str


@dataclass(frozen=True)
class ModelCancelSuspendedResponseParams:
    model_id: str | None = None
    _: KW_ONLY
    response: ModelResponse
    run_context: RunContext[Any] | None


@dataclass(frozen=True)
class ModelCompactMessagesId:
    """Identifies a durable message-compaction operation for engine configuration.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    model_id: str | None = None
    _: KW_ONLY
    model_name: str


@dataclass(frozen=True)
class ModelCompactMessagesParams:
    model_id: str | None = None
    _: KW_ONLY
    request_context: ModelRequestContext
    instructions: str | None
    run_context: RunContext[Any]


@dataclass(frozen=True, kw_only=True)
class EventStreamHandlerId:
    """Identifies a durable event-stream handler invocation for engine configuration.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    pass


@dataclass(frozen=True)
class EventStreamHandlerParams:
    event: AgentStreamEvent
    _: KW_ONLY
    run_context: RunContext[Any]


@dataclass(frozen=True)
class CapabilityOperationId:
    """Identifies an operation contributed by a capability.

    Engine configuration receives these operations through the same backend as built-in model and
    tool operations. See the
    [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    capability_id: str
    _: KW_ONLY
    operation: str


@dataclass(frozen=True)
class ToolsetGetToolsId:
    """Identifies durable tool discovery for a particular toolset.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    toolset_kind: ToolsetKind
    _: KW_ONLY
    toolset_id: str


@dataclass(frozen=True)
class ToolsetGetToolsParams:
    ctx: RunContext[Any]
    _: KW_ONLY


@dataclass(frozen=True)
class ToolsetGetInstructionsId:
    """Identifies durable instruction discovery for an MCP toolset.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    toolset_id: str
    _: KW_ONLY


@dataclass(frozen=True)
class ToolsetValidateToolArgumentsId:
    """Identifies durable argument validation for a particular toolset.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    toolset_kind: ToolsetKind
    _: KW_ONLY
    toolset_id: str


@dataclass(frozen=True)
class ToolsetCallToolId:
    """Identifies durable tool execution for a particular toolset.

    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    toolset_kind: ToolsetKind
    _: KW_ONLY
    toolset_id: str


@dataclass(frozen=True)
class ToolsetCallToolParams:
    name: str
    _: KW_ONLY
    tool_args: dict[str, Any]
    ctx: RunContext[Any]
    tool: ToolsetTool[Any] | None


@dataclass(frozen=True)
class DynamicToolsetCallToolParams:
    name: str
    _: KW_ONLY
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
"""The extensible union of operation identifiers passed to engine configuration.

The union can gain variants in minor releases, so matches need a default branch. See the
[durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
"""


class ParameterTransport(Generic[ParamsT, WireT], Protocol):
    """Serialize operation parameters for an engine boundary and rebuild them worker-side.

    The `runtime` passed to `load` is engine-side context needed while rebuilding parameters. For
    example, Temporal passes its durability capability so serialized run contexts can be restored.
    """

    @abstractmethod
    def dump(self, params: ParamsT) -> WireT: ...

    @abstractmethod
    def load(self, payload: WireT, *, runtime: object) -> ParamsT: ...


class CacheIdentity(Generic[ParamsT_contra], Protocol):
    """Project semantic parameters into opaque hash inputs for hash-keyed engines."""

    @abstractmethod
    def project(self, params: ParamsT_contra) -> tuple[object, ...]: ...


class ResultCodec(Generic[ResultT], Protocol):
    """Encode and decode an operation result across an engine boundary."""

    @abstractmethod
    def dump(self, value: ResultT) -> object: ...

    @abstractmethod
    def load(self, payload: object) -> ResultT: ...


OperationConfigRole: TypeAlias = Literal['model', 'event', 'tool', 'capability']
"""The coarse configuration bucket for an operation; its ID carries the fine-grained identity."""


class DurableOperationConfig(Generic[ConfigT_co], Protocol):
    @abstractmethod
    def base(self, role: OperationConfigRole, *, operation_id: DurableOperationId) -> ConfigT_co: ...

    @abstractmethod
    def for_tool(
        self,
        role: OperationConfigRole,
        *,
        operation_id: DurableOperationId,
        tool: object | None,
        tool_name: str,
    ) -> ConfigT_co | Literal[False]: ...


def resolve_tool_operation_config(
    config: DurableOperationConfig[ConfigT],
    operation: DurableOperation[ParamsT, WireT, ResultT],
    *,
    tool: object | None,
    tool_name: str,
) -> ConfigT | Literal[False]:
    """Resolve tool configuration shared by callable and registered backends."""
    return config.for_tool(
        operation.config_role,
        operation_id=operation.operation_id,
        tool=tool,
        tool_name=tool_name,
    )


@dataclass(frozen=True, kw_only=True)
class DurableOperation(Generic[ParamsT, WireT, ResultT]):
    """A complete semantic declaration that an engine backend can bind.

    Attributes:
        operation_id: Stable typed identity used for naming and configuration.
        handler: Semantic async operation body.
        parameter_transport: Codec between semantic parameters and engine wire parameters.
        cache_identity: Projection consulted by hash-keyed engines.
        result_codec: Codec for the handler result.
        config_role: Coarse configuration category for the operation.
        invocation_label: Optional per-call display or naming label.
    """

    operation_id: DurableOperationId
    handler: Callable[[ParamsT], Awaitable[ResultT]]
    parameter_transport: ParameterTransport[ParamsT, WireT]
    cache_identity: CacheIdentity[ParamsT]
    result_codec: ResultCodec[ResultT]
    config_role: OperationConfigRole
    invocation_label: Callable[[ParamsT], str] | None = None


class IdentityParameterTransport(ParameterTransport[ParamsT, ParamsT], Generic[ParamsT]):
    """Pass parameters through unchanged for engines that transport Python values themselves."""

    def dump(self, params: ParamsT) -> ParamsT:
        return params

    def load(self, payload: ParamsT, *, runtime: object) -> ParamsT:
        return payload


class NoCacheIdentity(CacheIdentity[ParamsT], Generic[ParamsT]):
    """Provide no semantic cache inputs for an operation."""

    def project(self, params: ParamsT) -> tuple[()]:
        return ()


class TypedResultCodec(ResultCodec[ResultT], Generic[ResultT]):
    """Encode and validate results using a declared runtime result type."""

    def __init__(self, result_type: object, *, mode: Literal['json', 'identity'] = 'json') -> None:
        self._result_type = result_type
        self._codec: DurabilityCodec = JSON_CODEC if mode == 'json' else IDENTITY_CODEC

    def dump(self, value: ResultT) -> object:
        return self._codec.dump(self._result_type, value)

    def load(self, payload: object) -> ResultT:
        return cast(ResultT, self._codec.load(self._result_type, payload))

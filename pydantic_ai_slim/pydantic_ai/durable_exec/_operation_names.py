from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from typing_extensions import assert_never

from ._operation import (
    CallToolId,
    CancelSuspendedResponseId,
    CapabilityOperationId,
    CompactMessagesId,
    DurableOperationId,
    EventStreamHandlerId,
    GetInstructionsId,
    GetToolsId,
    ModelRequestId,
    ToolsetKind,
    ValidateToolArgumentsId,
)


@dataclass(frozen=True)
class DurableInvocationName:
    operation_name: str
    display_name: str | None = None


class DurableOperationNamer(Protocol):
    """Maps typed operation IDs to stable persisted and display names.

    Engine authors can implement this protocol when `JournalOperationNamer` does not match their
    runtime's naming rules. See the
    [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    def operation_name(self, operation_id: DurableOperationId) -> str: ...

    def invocation_name(self, operation_id: DurableOperationId, params: object) -> DurableInvocationName: ...


@runtime_checkable
class _NamedToolInvocation(Protocol):
    name: str


def _toolset_prefix(kind: ToolsetKind) -> str:
    return 'mcp_server' if kind == 'mcp' else f'{kind}_toolset'


def _tool_name(params: object) -> str:
    if not isinstance(params, _NamedToolInvocation):
        raise TypeError('Tool-call invocation parameters must expose a string `name` attribute')
    return params.name


class JournalOperationNamer:
    """Stable default naming policy for sequence-based journal engines.

    Use this with either backend tier when persisted operation names can follow Pydantic AI's
    journal convention. Pin the generated names before changing agent, model, or toolset identity.
    See the [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    def __init__(self, agent_name: str, *, default_model_id: str = 'default') -> None:
        self._agent_name = agent_name
        self._default_model_id = default_model_id

    def _model_suffix(self, model_id: str | None) -> str:
        return '' if model_id is None or model_id == self._default_model_id else f'.{model_id}'

    def operation_name(self, operation_id: DurableOperationId) -> str:
        match operation_id:
            case CapabilityOperationId(capability_id=capability_id, operation=operation):
                return f'{self._agent_name}__capability__{capability_id}.{operation}'
            case ModelRequestId(model_id=model_id, streaming=streaming):
                operation = 'model.request_stream' if streaming else 'model.request'
                return f'{self._agent_name}__{operation}{self._model_suffix(model_id)}'
            case CancelSuspendedResponseId(model_id=model_id):
                return f'{self._agent_name}__model.cancel_suspended_response{self._model_suffix(model_id)}'
            case CompactMessagesId(model_id=model_id):
                return f'{self._agent_name}__model.compact_messages{self._model_suffix(model_id)}'
            case EventStreamHandlerId():
                return f'{self._agent_name}__event_stream_handler'
            case GetToolsId(toolset_kind=kind, toolset_id=toolset_id):
                return f'{self._agent_name}__{_toolset_prefix(kind)}__{toolset_id}.get_tools'
            case GetInstructionsId(toolset_id=toolset_id):
                return f'{self._agent_name}__mcp_server__{toolset_id}.get_instructions'
            case ValidateToolArgumentsId(toolset_kind=kind, toolset_id=toolset_id):
                return f'{self._agent_name}__{_toolset_prefix(kind)}__{toolset_id}.validate_args'
            case CallToolId(toolset_kind=kind, toolset_id=toolset_id):
                return f'{self._agent_name}__{_toolset_prefix(kind)}__{toolset_id}.call_tool'
        assert_never(operation_id)

    def invocation_name(self, operation_id: DurableOperationId, params: object) -> DurableInvocationName:
        name = self.operation_name(operation_id)
        if isinstance(operation_id, CallToolId) and operation_id.toolset_kind != 'mcp':
            name = f'{name}:{_tool_name(params)}'
        return DurableInvocationName(name)


class PrefectOperationNamer:
    def operation_name(self, operation_id: DurableOperationId) -> str:
        match operation_id:
            case CapabilityOperationId(capability_id=capability_id, operation=operation):
                return f'Capability: {capability_id}.{operation}'
            case ModelRequestId(streaming=True, model_name=model_name):
                return f'Model Request (Streaming): {model_name}'
            case ModelRequestId(model_name=model_name):
                return f'Model Request: {model_name}'
            case CancelSuspendedResponseId(model_name=model_name):
                return f'Cancel Suspended Response: {model_name}'
            case CompactMessagesId(model_name=model_name):
                return f'Compact Messages: {model_name}'
            case EventStreamHandlerId():
                return 'Handle Stream Event'
            case GetToolsId() | GetInstructionsId():
                raise RuntimeError(
                    'Prefect discovery operations do not have durable unit names in the current implementation'
                )
            case ValidateToolArgumentsId():
                return 'Validate Tool Args'
            case CallToolId(toolset_kind='mcp'):
                return 'Call MCP Tool'
            case CallToolId():
                return 'Call Tool'
        assert_never(operation_id)

    def invocation_name(self, operation_id: DurableOperationId, params: object) -> DurableInvocationName:
        name = self.operation_name(operation_id)
        if isinstance(operation_id, CallToolId | ValidateToolArgumentsId):
            name = f'{name}: {_tool_name(params)}'
        return DurableInvocationName(name, display_name=name)


class DBOSOperationNamer(JournalOperationNamer):
    def _model_suffix(self, model_id: str | None) -> str:
        return ''

    def invocation_name(self, operation_id: DurableOperationId, params: object) -> DurableInvocationName:
        return DurableInvocationName(self.operation_name(operation_id))


class TemporalOperationNamer:
    def __init__(self, agent_name: str) -> None:
        self._prefix = f'agent__{agent_name}'

    def operation_name(self, operation_id: DurableOperationId) -> str:
        match operation_id:
            case CapabilityOperationId(capability_id=capability_id, operation=operation):
                return f'{self._prefix}__capability__{capability_id}__{operation}'
            case ModelRequestId(streaming=True):
                return f'{self._prefix}__model_request_stream'
            case ModelRequestId():
                return f'{self._prefix}__model_request'
            case CancelSuspendedResponseId():
                return f'{self._prefix}__model_cancel_suspended_response'
            case CompactMessagesId():
                return f'{self._prefix}__model_compact_messages'
            case EventStreamHandlerId():
                return f'{self._prefix}__event_stream_handler'
            case GetToolsId(toolset_kind=kind, toolset_id=toolset_id):
                return f'{self._prefix}__{_toolset_prefix(kind)}__{toolset_id}__get_tools'
            case GetInstructionsId(toolset_id=toolset_id):
                return f'{self._prefix}__mcp_server__{toolset_id}__get_instructions'
            case ValidateToolArgumentsId(toolset_kind=kind, toolset_id=toolset_id):
                prefix = 'toolset' if kind == 'function' else _toolset_prefix(kind)
                return f'{self._prefix}__{prefix}__{toolset_id}__validate_args'
            case CallToolId(toolset_kind=kind, toolset_id=toolset_id):
                prefix = 'toolset' if kind == 'function' else _toolset_prefix(kind)
                return f'{self._prefix}__{prefix}__{toolset_id}__call_tool'
        assert_never(operation_id)

    def invocation_name(self, operation_id: DurableOperationId, params: object) -> DurableInvocationName:
        return DurableInvocationName(self.operation_name(operation_id))

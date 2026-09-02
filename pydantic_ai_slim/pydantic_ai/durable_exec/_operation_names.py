from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Protocol

from typing_extensions import assert_never

from ._operation import (
    CapabilityOperationId,
    DurableOperationId,
    EventStreamHandlerId,
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelRequestId,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetKind,
    ToolsetValidateToolArgumentsId,
)


@dataclass(frozen=True)
class DurableInvocationName:
    operation_name: str
    _: KW_ONLY
    display_name: str | None = None


class DurableOperationNamer(Protocol):
    """Maps typed operation IDs to stable persisted and display names.

    Engine authors can implement this protocol when `JournalOperationNamer` does not match their
    runtime's naming rules. See the
    [durable backend guide](https://pydantic.dev/docs/ai/capabilities/durable_execution/backends/).
    """

    def operation_name(self, operation_id: DurableOperationId) -> str: ...

    def invocation_name(self, operation_id: DurableOperationId, *, label: str | None) -> DurableInvocationName: ...


def _toolset_prefix(kind: ToolsetKind) -> str:
    # `mcp_server` preserves the naming convention already shipped on `main`.
    return 'mcp_server' if kind == 'mcp' else f'{kind}_toolset'


class JournalOperationNamer(DurableOperationNamer):
    """Stable default naming policy for sequence-based journal engines.

    Generated names are persisted compatibility data and must essentially never change. Changing
    them can strand in-flight workflows and recorded runs.

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
            case ModelCancelSuspendedResponseId(model_id=model_id):
                return f'{self._agent_name}__model.cancel_suspended_response{self._model_suffix(model_id)}'
            case ModelCompactMessagesId(model_id=model_id):
                return f'{self._agent_name}__model.compact_messages{self._model_suffix(model_id)}'
            case EventStreamHandlerId():
                return f'{self._agent_name}__event_stream_handler'
            case ToolsetGetToolsId(toolset_kind=kind, toolset_id=toolset_id):
                return f'{self._agent_name}__{_toolset_prefix(kind)}__{toolset_id}.get_tools'
            case ToolsetGetInstructionsId(toolset_id=toolset_id):
                return f'{self._agent_name}__mcp_server__{toolset_id}.get_instructions'
            case ToolsetValidateToolArgumentsId(toolset_kind=kind, toolset_id=toolset_id):
                return f'{self._agent_name}__{_toolset_prefix(kind)}__{toolset_id}.validate_args'
            case ToolsetCallToolId(toolset_kind=kind, toolset_id=toolset_id):
                return f'{self._agent_name}__{_toolset_prefix(kind)}__{toolset_id}.call_tool'
        assert_never(operation_id)

    def invocation_name(self, operation_id: DurableOperationId, *, label: str | None) -> DurableInvocationName:
        name = self.operation_name(operation_id)
        if isinstance(operation_id, ToolsetCallToolId) and operation_id.toolset_kind != 'mcp':
            assert label is not None
            name = f'{name}:{label}'
        return DurableInvocationName(name)

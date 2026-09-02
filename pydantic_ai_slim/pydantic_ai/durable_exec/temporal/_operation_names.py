from typing_extensions import assert_never

from .._operation import (
    CapabilityOperationId,
    DurableOperationId,
    EventStreamHandlerId,
    ModelCancelSuspendedResponseId,
    ModelCompactMessagesId,
    ModelRequestId,
    ToolsetCallToolId,
    ToolsetGetInstructionsId,
    ToolsetGetToolsId,
    ToolsetValidateToolArgumentsId,
)
from .._operation_names import (
    DurableInvocationName,
    DurableOperationNamer,
    _toolset_prefix as _toolset_prefix,  # pyright: ignore[reportPrivateUsage]
)


class TemporalOperationNamer(DurableOperationNamer):
    """Generate Temporal activity names that are persisted compatibility data.

    These names must essentially never change. Changing them can strand in-flight workflows and
    recorded runs.
    """

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
            case ModelCancelSuspendedResponseId():
                return f'{self._prefix}__model_cancel_suspended_response'
            case ModelCompactMessagesId():
                return f'{self._prefix}__model_compact_messages'
            case EventStreamHandlerId():
                return f'{self._prefix}__event_stream_handler'
            case ToolsetGetToolsId(toolset_kind=kind, toolset_id=toolset_id):
                return f'{self._prefix}__{_toolset_prefix(kind)}__{toolset_id}__get_tools'
            case ToolsetGetInstructionsId(toolset_id=toolset_id):
                return f'{self._prefix}__mcp_server__{toolset_id}__get_instructions'
            case ToolsetValidateToolArgumentsId(toolset_kind=kind, toolset_id=toolset_id):
                prefix = 'toolset' if kind == 'function' else _toolset_prefix(kind)
                return f'{self._prefix}__{prefix}__{toolset_id}__validate_args'
            case ToolsetCallToolId(toolset_kind=kind, toolset_id=toolset_id):
                prefix = 'toolset' if kind == 'function' else _toolset_prefix(kind)
                return f'{self._prefix}__{prefix}__{toolset_id}__call_tool'
        assert_never(operation_id)

    def invocation_name(self, operation_id: DurableOperationId, *, label: str | None) -> DurableInvocationName:
        return DurableInvocationName(self.operation_name(operation_id))

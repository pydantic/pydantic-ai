from typing_extensions import assert_never

from .._operation import (
    CallToolId,
    CancelSuspendedResponseId,
    CapabilityOperationId,
    CompactMessagesId,
    DurableOperationId,
    EventStreamHandlerId,
    GetInstructionsId,
    GetToolsId,
    ModelRequestId,
    ValidateToolArgumentsId,
)
from .._operation_names import DurableInvocationName, _toolset_prefix  # pyright: ignore[reportPrivateUsage]


class TemporalOperationNamer:
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

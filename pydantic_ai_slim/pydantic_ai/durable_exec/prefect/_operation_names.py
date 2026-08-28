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
from .._operation_names import DurableInvocationName, DurableOperationNamer


class PrefectOperationNamer(DurableOperationNamer):
    """Generate Prefect task names that are persisted compatibility data.

    These names must essentially never change. Changing them can strand in-flight flows and
    recorded runs.
    """

    def operation_name(self, operation_id: DurableOperationId) -> str:
        match operation_id:
            case CapabilityOperationId(capability_id=capability_id, operation=operation):
                return f'Capability: {capability_id}.{operation}'
            case ModelRequestId(streaming=True, model_name=model_name):
                return f'Model Request (Streaming): {model_name}'
            case ModelRequestId(model_name=model_name):
                return f'Model Request: {model_name}'
            case ModelCancelSuspendedResponseId(model_name=model_name):
                return f'Cancel Suspended Response: {model_name}'
            case ModelCompactMessagesId(model_name=model_name):
                return f'Compact Messages: {model_name}'
            case EventStreamHandlerId():
                return 'Handle Stream Event'
            case ToolsetGetToolsId(toolset_kind='mcp', toolset_id=toolset_id):
                return f'Get MCP Tools: {toolset_id}'
            case ToolsetGetToolsId(toolset_id=toolset_id):
                return f'Discover Tools: {toolset_id}'
            case ToolsetGetInstructionsId(toolset_id=toolset_id):
                return f'Get MCP Instructions: {toolset_id}'
            case ToolsetValidateToolArgumentsId():
                return 'Validate Tool Args'
            case ToolsetCallToolId(toolset_kind='mcp'):
                return 'Call MCP Tool'
            case ToolsetCallToolId():
                return 'Call Tool'
        assert_never(operation_id)

    def invocation_name(self, operation_id: DurableOperationId, *, label: str | None) -> DurableInvocationName:
        name = self.operation_name(operation_id)
        if isinstance(operation_id, (ToolsetCallToolId, ToolsetValidateToolArgumentsId)):
            assert label is not None
            name = f'{name}: {label}'
        return DurableInvocationName(name, display_name=name)

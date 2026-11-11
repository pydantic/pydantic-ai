"""Durable execution integrations for Pydantic AI.

Each subpackage adds durability for one durable-execution platform via a
capability you attach to an [`Agent`][pydantic_ai.Agent]:

- [`pydantic_ai.durable_exec.temporal`][pydantic_ai.durable_exec.temporal] —
  [`TemporalDurability`][pydantic_ai.durable_exec.temporal.TemporalDurability]
- [`pydantic_ai.durable_exec.dbos`][pydantic_ai.durable_exec.dbos] —
  [`DBOSDurability`][pydantic_ai.durable_exec.dbos.DBOSDurability]
- [`pydantic_ai.durable_exec.prefect`][pydantic_ai.durable_exec.prefect] —
  [`PrefectDurability`][pydantic_ai.durable_exec.prefect.PrefectDurability]
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._base import BaseDurabilityCapability, conversation_id_from_run_id
    from ._capability_operation import (
        CapabilityMethodDeclaration,
        CapabilityOperationParams,
        ModelRequestContextProjection,
        capability_operation_result_type,
    )
    from ._codec import IDENTITY_CODEC, JSON_CODEC, DurabilityCodec
    from ._operation import (
        CacheIdentity,
        CapabilityOperationId,
        DurableOperation,
        DurableOperationConfig,
        DurableOperationId,
        DynamicToolsetCallToolParams,
        EventStreamHandlerId,
        EventStreamHandlerParams,
        ModelCancelSuspendedResponseId,
        ModelCancelSuspendedResponseParams,
        ModelCompactMessagesId,
        ModelCompactMessagesParams,
        ModelRequestId,
        ModelRequestParams,
        OperationConfigRole,
        ParameterTransport,
        ParameterTransports,
        ResultCodec,
        ToolsetCallToolId,
        ToolsetCallToolParams,
        ToolsetGetInstructionsId,
        ToolsetGetToolsId,
        ToolsetGetToolsParams,
        ToolsetKind,
        ToolsetValidateToolArgumentsId,
    )
    from ._operation_backend import (
        BoundDurableOperation,
        CallableOperationBackend,
        DurableOperationBackend,
        JournalCallableOperationBackend,
        RegisteredOperationBackend,
        RoleBasedOperationConfig,
        durable_unit_scope,
    )
    from ._operation_names import DurableOperationNamer, JournalOperationNamer
    from ._run_context import DurableRunContext
    from ._spec import DurabilityEngineSpec
    from ._toolset import CallToolResult, DynamicToolsResult, revalidate_function_call
    from ._workspace import WorkspaceCall, WorkspaceCallParams, WorkspaceCallResult

__all__ = [
    'BaseDurabilityCapability',
    'conversation_id_from_run_id',
    'ToolsetCallToolId',
    'CallableOperationBackend',
    'ModelCancelSuspendedResponseId',
    'CapabilityOperationId',
    'ModelCompactMessagesId',
    'DurabilityCodec',
    'DurabilityEngineSpec',
    'DurableOperationBackend',
    'DurableOperationId',
    'DurableOperationNamer',
    'EventStreamHandlerId',
    'ToolsetGetInstructionsId',
    'ToolsetGetToolsId',
    'IDENTITY_CODEC',
    'JSON_CODEC',
    'JournalCallableOperationBackend',
    'JournalOperationNamer',
    'ModelRequestId',
    'OperationConfigRole',
    'RegisteredOperationBackend',
    'RoleBasedOperationConfig',
    'ToolsetKind',
    'ToolsetValidateToolArgumentsId',
    'BoundDurableOperation',
    'CacheIdentity',
    'CallToolResult',
    'capability_operation_result_type',
    'CapabilityMethodDeclaration',
    'CapabilityOperationParams',
    'durable_unit_scope',
    'DurableOperation',
    'DurableOperationConfig',
    'DurableRunContext',
    'DynamicToolsetCallToolParams',
    'DynamicToolsResult',
    'EventStreamHandlerParams',
    'ModelCancelSuspendedResponseParams',
    'ModelCompactMessagesParams',
    'ModelRequestContextProjection',
    'ModelRequestParams',
    'ParameterTransport',
    'ParameterTransports',
    'ResultCodec',
    'ToolsetCallToolParams',
    'ToolsetGetToolsParams',
    'revalidate_function_call',
    'WorkspaceCall',
    'WorkspaceCallParams',
    'WorkspaceCallResult',
]

_exports = {
    'BaseDurabilityCapability': ('._base', 'BaseDurabilityCapability'),
    'conversation_id_from_run_id': ('._base', 'conversation_id_from_run_id'),
    'DurabilityCodec': ('._codec', 'DurabilityCodec'),
    'DurabilityEngineSpec': ('._spec', 'DurabilityEngineSpec'),
    'IDENTITY_CODEC': ('._codec', 'IDENTITY_CODEC'),
    'JSON_CODEC': ('._codec', 'JSON_CODEC'),
    'ToolsetCallToolId': ('._operation', 'ToolsetCallToolId'),
    'ModelCancelSuspendedResponseId': ('._operation', 'ModelCancelSuspendedResponseId'),
    'CapabilityOperationId': ('._operation', 'CapabilityOperationId'),
    'ModelCompactMessagesId': ('._operation', 'ModelCompactMessagesId'),
    'DurableOperationId': ('._operation', 'DurableOperationId'),
    'EventStreamHandlerId': ('._operation', 'EventStreamHandlerId'),
    'ToolsetGetInstructionsId': ('._operation', 'ToolsetGetInstructionsId'),
    'ToolsetGetToolsId': ('._operation', 'ToolsetGetToolsId'),
    'ModelRequestId': ('._operation', 'ModelRequestId'),
    'OperationConfigRole': ('._operation', 'OperationConfigRole'),
    'ToolsetKind': ('._operation', 'ToolsetKind'),
    'ToolsetValidateToolArgumentsId': ('._operation', 'ToolsetValidateToolArgumentsId'),
    'CallableOperationBackend': ('._operation_backend', 'CallableOperationBackend'),
    'DurableOperationBackend': ('._operation_backend', 'DurableOperationBackend'),
    'JournalCallableOperationBackend': ('._operation_backend', 'JournalCallableOperationBackend'),
    'RegisteredOperationBackend': ('._operation_backend', 'RegisteredOperationBackend'),
    'RoleBasedOperationConfig': ('._operation_backend', 'RoleBasedOperationConfig'),
    'DurableOperationNamer': ('._operation_names', 'DurableOperationNamer'),
    'JournalOperationNamer': ('._operation_names', 'JournalOperationNamer'),
    'BoundDurableOperation': ('._operation_backend', 'BoundDurableOperation'),
    'CacheIdentity': ('._operation', 'CacheIdentity'),
    'CallToolResult': ('._toolset', 'CallToolResult'),
    'capability_operation_result_type': ('._capability_operation', 'capability_operation_result_type'),
    'CapabilityMethodDeclaration': ('._capability_operation', 'CapabilityMethodDeclaration'),
    'CapabilityOperationParams': ('._capability_operation', 'CapabilityOperationParams'),
    'durable_unit_scope': ('._operation_backend', 'durable_unit_scope'),
    'DurableOperation': ('._operation', 'DurableOperation'),
    'DurableOperationConfig': ('._operation', 'DurableOperationConfig'),
    'DurableRunContext': ('._run_context', 'DurableRunContext'),
    'DynamicToolsetCallToolParams': ('._operation', 'DynamicToolsetCallToolParams'),
    'DynamicToolsResult': ('._toolset', 'DynamicToolsResult'),
    'EventStreamHandlerParams': ('._operation', 'EventStreamHandlerParams'),
    'ModelCancelSuspendedResponseParams': ('._operation', 'ModelCancelSuspendedResponseParams'),
    'ModelCompactMessagesParams': ('._operation', 'ModelCompactMessagesParams'),
    'ModelRequestContextProjection': ('._capability_operation', 'ModelRequestContextProjection'),
    'ModelRequestParams': ('._operation', 'ModelRequestParams'),
    'ParameterTransport': ('._operation', 'ParameterTransport'),
    'ParameterTransports': ('._operation', 'ParameterTransports'),
    'ResultCodec': ('._operation', 'ResultCodec'),
    'ToolsetCallToolParams': ('._operation', 'ToolsetCallToolParams'),
    'ToolsetGetToolsParams': ('._operation', 'ToolsetGetToolsParams'),
    'revalidate_function_call': ('._toolset', 'revalidate_function_call'),
    'WorkspaceCall': ('._workspace', 'WorkspaceCall'),
    'WorkspaceCallParams': ('._workspace', 'WorkspaceCallParams'),
    'WorkspaceCallResult': ('._workspace', 'WorkspaceCallResult'),
}


def __getattr__(name: str) -> object:
    """Load engine-builder exports lazily to avoid package initialization cycles."""
    from importlib import import_module

    try:
        module_name, attribute = _exports[name]
    except KeyError:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}') from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

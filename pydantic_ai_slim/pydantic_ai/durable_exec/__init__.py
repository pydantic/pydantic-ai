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
    from ._base import BaseDurabilityCapability
    from ._codec import IDENTITY_CODEC, JSON_CODEC, DurabilityCodec
    from ._operation import (
        CapabilityOperationId,
        DurableOperationId,
        EventStreamHandlerId,
        ModelCancelSuspendedResponseId,
        ModelCompactMessagesId,
        ModelRequestId,
        OperationConfigRole,
        ToolsetCallToolId,
        ToolsetGetInstructionsId,
        ToolsetGetToolsId,
        ToolsetKind,
        ToolsetValidateToolArgumentsId,
    )
    from ._operation_backend import (
        CallableOperationBackend,
        DurableOperationBackend,
        JournalCallableOperationBackend,
        RegisteredOperationBackend,
        RoleBasedOperationConfig,
    )
    from ._operation_names import DurableOperationNamer, JournalOperationNamer
    from ._spec import DurabilityEngineSpec

__all__ = [
    'BaseDurabilityCapability',
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
]

_exports = {
    'BaseDurabilityCapability': ('._base', 'BaseDurabilityCapability'),
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

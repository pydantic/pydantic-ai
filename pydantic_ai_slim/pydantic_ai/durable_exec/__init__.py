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
        CallToolId,
        CancelSuspendedResponseId,
        CapabilityOperationId,
        CompactMessagesId,
        DurableOperationId,
        EventStreamHandlerId,
        GetInstructionsId,
        GetToolsId,
        ModelRequestId,
        OperationConfigRole,
        ToolsetKind,
        ValidateToolArgumentsId,
    )
    from ._operation_backend import CallableOperationBackend, DurableOperationBackend, RegisteredOperationBackend
    from ._operation_names import DurableOperationNamer, JournalOperationNamer

__all__ = [
    'BaseDurabilityCapability',
    'CallToolId',
    'CallableOperationBackend',
    'CancelSuspendedResponseId',
    'CapabilityOperationId',
    'CompactMessagesId',
    'DurabilityCodec',
    'DurableOperationBackend',
    'DurableOperationId',
    'DurableOperationNamer',
    'EventStreamHandlerId',
    'GetInstructionsId',
    'GetToolsId',
    'IDENTITY_CODEC',
    'JSON_CODEC',
    'JournalOperationNamer',
    'ModelRequestId',
    'OperationConfigRole',
    'RegisteredOperationBackend',
    'ToolsetKind',
    'ValidateToolArgumentsId',
]

_exports = {
    'BaseDurabilityCapability': ('._base', 'BaseDurabilityCapability'),
    'DurabilityCodec': ('._codec', 'DurabilityCodec'),
    'IDENTITY_CODEC': ('._codec', 'IDENTITY_CODEC'),
    'JSON_CODEC': ('._codec', 'JSON_CODEC'),
    'CallToolId': ('._operation', 'CallToolId'),
    'CancelSuspendedResponseId': ('._operation', 'CancelSuspendedResponseId'),
    'CapabilityOperationId': ('._operation', 'CapabilityOperationId'),
    'CompactMessagesId': ('._operation', 'CompactMessagesId'),
    'DurableOperationId': ('._operation', 'DurableOperationId'),
    'EventStreamHandlerId': ('._operation', 'EventStreamHandlerId'),
    'GetInstructionsId': ('._operation', 'GetInstructionsId'),
    'GetToolsId': ('._operation', 'GetToolsId'),
    'ModelRequestId': ('._operation', 'ModelRequestId'),
    'OperationConfigRole': ('._operation', 'OperationConfigRole'),
    'ToolsetKind': ('._operation', 'ToolsetKind'),
    'ValidateToolArgumentsId': ('._operation', 'ValidateToolArgumentsId'),
    'CallableOperationBackend': ('._operation_backend', 'CallableOperationBackend'),
    'DurableOperationBackend': ('._operation_backend', 'DurableOperationBackend'),
    'RegisteredOperationBackend': ('._operation_backend', 'RegisteredOperationBackend'),
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

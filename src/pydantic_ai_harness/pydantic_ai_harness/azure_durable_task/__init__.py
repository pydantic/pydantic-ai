"""Agent activities for Azure Durable Functions and Durable Task Scheduler orchestrations."""

try:
    import durabletask as _durabletask  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError('Install `pydantic-ai-harness[azure-durable-task]` to use Azure Durable Task.') from _import_error

from ._bridge import run_durable
from ._capability import AzureDurableTaskDurability
from ._continue_as_new import ContinueAsNew
from ._operation_backend import AzureDurableTaskOperationBackend

__all__ = (
    'AzureDurableTaskDurability',
    'AzureDurableTaskOperationBackend',
    'ContinueAsNew',
    'run_durable',
)

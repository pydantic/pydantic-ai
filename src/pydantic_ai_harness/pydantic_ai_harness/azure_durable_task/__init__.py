"""Agent activities for Azure Durable Task Scheduler orchestrations."""

try:
    import durabletask as _durabletask  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError('Install `pydantic-ai-harness[azure-durable-task]` to use Azure Durable Task.') from _import_error

from ._bridge import run_durable
from ._capability import AzureDurableTaskDurability

__all__ = (
    'AzureDurableTaskDurability',
    'run_durable',
)

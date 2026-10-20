"""Durable Task activities and orchestrations for Pydantic AI agents."""

try:
    import durabletask as _durabletask  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as _import_error:
    raise ImportError(
        'Install `pydantic-ai-harness[durabletask]` to use the Durable Task integration.'
    ) from _import_error

from ._agent import DurableTaskAgent, DurableTaskRun
from ._bridge import run_durable
from ._capability import DurableTaskDurability
from ._operation_backend import DurableTaskOperationBackend

__all__ = (
    'DurableTaskAgent',
    'DurableTaskDurability',
    'DurableTaskOperationBackend',
    'DurableTaskRun',
    'run_durable',
)

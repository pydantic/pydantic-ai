"""Version-checked adapter for the SDK's private in-process test machinery."""

from __future__ import annotations

# durabletask has no public in-process executor or registry.
# Keep private SDK access here until a supported executor is available; the in-memory
# adapter was verified against 1.13.0, and the tests using it report breakage in later versions.
# pyright: reportPrivateUsage=false
from durabletask.internal import helpers as ph, orchestrator_service_pb2 as pb
from durabletask.worker import _ActivityExecutor, _OrchestrationExecutor, _Registry

__all__ = ('ActivityExecutor', 'OrchestrationExecutor', 'Registry', 'pb', 'ph')

Registry = _Registry
ActivityExecutor = _ActivityExecutor
OrchestrationExecutor = _OrchestrationExecutor

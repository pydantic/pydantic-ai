"""Version-checked adapter for the SDK's private in-process test machinery."""

from __future__ import annotations

# durabletask 1.12.0 has no public in-process executor or registry.
# Keep private SDK access here until a supported executor is available.
# pyright: reportPrivateUsage=false
from importlib.metadata import version

from durabletask.internal import helpers as ph, orchestrator_service_pb2 as pb
from durabletask.worker import (
    _ActivityExecutor,
    _OrchestrationExecutor,
    _Registry,
    _RuntimeOrchestrationContext,
)

__all__ = ('ActivityExecutor', 'OrchestrationExecutor', 'Registry', 'pb', 'ph')

assert version('durabletask') == '1.12.0', (
    'The in-memory test adapter is verified with durabletask==1.12.0. '
    'Review SDK-private executor and generator cleanup behavior before testing another version.'
)


class Registry(_Registry):
    pass


class ActivityExecutor(_ActivityExecutor):
    pass


class OrchestrationExecutor(_OrchestrationExecutor):
    context: _RuntimeOrchestrationContext | None = None

    def process_event(self, ctx: _RuntimeOrchestrationContext, event: pb.HistoryEvent) -> None:
        self.context = ctx
        super().process_event(ctx, event)

    def close(self) -> None:
        # 1.12.0 leaves the dispatch generator open after yielding. Close its async scopes before replay.
        if self.context is not None and self.context._generator is not None:
            self.context._generator.close()

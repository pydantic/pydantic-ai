"""A compaction started ahead of its trigger, held until `FallbackCompaction` needs it."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic_ai._utils import cancel_and_drain
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import AbstractModel
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness.compaction._shared import CompactionStrategy

_SPAN_NAME = 'background_compaction'


@dataclass
class BackgroundCompaction:
    """One strategy's compaction of a snapshot of the history, running or finished.

    The snapshot is a deep copy, so a later in-place edit of a message object is seen as a
    changed history rather than silently matching. The work runs on the starting run's context,
    so a model call it makes counts toward that run's usage and limits, as a blocking one would.
    """

    snapshot: list[ModelMessage]
    model_id: str
    task: asyncio.Task[list[ModelMessage]]

    @classmethod
    def start(
        cls,
        strategy: CompactionStrategy[AgentDepsT],
        messages: Sequence[ModelMessage],
        ctx: RunContext[AgentDepsT],
    ) -> BackgroundCompaction:
        snapshot = copy.deepcopy(list(messages))
        task = asyncio.create_task(_compact_traced(strategy, list(snapshot), ctx), name=_SPAN_NAME)
        task.add_done_callback(_retrieve)
        return cls(snapshot=snapshot, model_id=ctx.model.model_id, task=task)

    def matches(self, messages: Sequence[ModelMessage], model: AbstractModel) -> bool:
        """Whether *messages* is the snapshot with only messages appended, bound for the same model."""
        size = len(self.snapshot)
        return model.model_id == self.model_id and len(messages) >= size and list(messages[:size]) == self.snapshot

    def same_start(self, messages: Sequence[ModelMessage]) -> bool:
        """Whether *messages* opens like the snapshot: the same conversation, possibly rewritten since."""
        return bool(messages) and messages[0] == self.snapshot[0]

    async def result(self) -> list[ModelMessage] | None:
        """Wait for the compaction; the compacted snapshot, or `None` when it failed or was cancelled.

        Waits with `asyncio.wait` so a cancelled caller leaves the task running for its owner to collect.
        """
        await asyncio.wait((self.task,))
        if self.task.cancelled() or self.task.exception() is not None:
            return None
        return self.task.result()

    def applied_to(self, compacted: list[ModelMessage], messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        """*compacted* followed, verbatim, by every message added to *messages* since the snapshot."""
        return [*compacted, *messages[len(self.snapshot) :]]

    async def cancel(self) -> None:
        await cancel_and_drain(self.task)


async def _compact_traced(
    strategy: CompactionStrategy[AgentDepsT], messages: list[ModelMessage], ctx: RunContext[AgentDepsT]
) -> list[ModelMessage]:
    with ctx.tracer.start_as_current_span(
        _SPAN_NAME,
        attributes={'compaction.strategy': type(strategy).__name__, 'compaction.messages_before': len(messages)},
    ):
        return await strategy.compact(messages, ctx)


def _retrieve(task: asyncio.Task[list[ModelMessage]]) -> None:
    """Mark a failure as retrieved: an unused result is dropped on purpose, not lost."""
    if not task.cancelled():
        task.exception()

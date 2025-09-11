from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

NodeId = NewType('NodeId', str)
NodeRunId = NewType('NodeRunId', str)

# The following aliases are just included for clarity; making them NewTypes is a hassle
JoinId = NodeId
ForkId = NodeId

GraphRunId = NewType('GraphRunId', str)
TaskId = NewType('TaskId', str)


@dataclass(frozen=True)
class ForkStackItem:
    """A fork stack item."""

    fork_id: ForkId
    """The ID of the node that created this fork."""
    node_run_id: NodeRunId
    """The ID associated to the specific run of the node that created this fork."""
    thread_index: int
    """The index of the execution "thread" created during the node run that created this fork.

    This is largely intended for observability/debugging; it may eventually be used to ensure idempotency."""


ForkStack = tuple[ForkStackItem, ...]

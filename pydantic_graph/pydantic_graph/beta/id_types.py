"""Type definitions for identifiers used throughout the graph execution system.

This module defines NewType wrappers and aliases for various ID types used in graph execution,
providing type safety and clarity when working with different kinds of identifiers.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import NewType

NodeID = NewType('NodeID', str)
"""Unique identifier for a node in the graph."""

NodeRunID = NewType('NodeRunID', str)
"""Unique identifier for a specific execution instance of a node."""

# The following aliases are just included for clarity; making them NewTypes is a hassle
JoinID = NodeID
"""Alias for NodeId when referring to join nodes."""

ForkID = NodeID
"""Alias for NodeId when referring to fork nodes."""

TaskID = NewType('TaskID', str)
"""Unique identifier for a task within the graph execution."""


@dataclass(frozen=True)
class ForkStackItem:
    """Represents a single fork point in the execution stack.

    When a node creates multiple parallel execution paths (forks), each fork is tracked
    using a ForkStackItem. This allows the system to maintain the execution hierarchy
    and coordinate parallel branches of execution.
    """

    fork_id: ForkID
    """The ID of the node that created this fork."""
    node_run_id: NodeRunID
    """The ID associated to the specific run of the node that created this fork."""
    thread_index: int
    """The index of the execution "thread" created during the node run that created this fork.

    This is largely intended for observability/debugging; it may eventually be used to ensure idempotency."""


ForkStack = tuple[ForkStackItem, ...]
"""A stack of fork items representing the full hierarchy of parallel execution branches.

The fork stack tracks the complete path through nested parallel executions,
allowing the system to coordinate and join parallel branches correctly.
"""


def generate_placeholder_node_id(label: str) -> str:
    """Generate a placeholder node ID, to be replaced during graph building."""
    return f'{_NODE_ID_PLACEHOLDER_PREFIX}:{label}:{uuid.uuid4()}'


def replace_placeholder_id(node_id: NodeID) -> str:
    """Returns whether a given NodeID is a placeholder node ID which should be replaced during graph building."""
    return re.sub(rf'{_NODE_ID_PLACEHOLDER_PREFIX}:([^:]+):.*', r'\1', node_id)


_NODE_ID_PLACEHOLDER_PREFIX = '__placeholder__'
"""
When Node IDs are required but not specified when building a graph, we generate placeholder values
using this prefix followed by a random string.

During graph building, we replace these with simpler and deterministically-selected values.
This ensures that the node IDs are stable when rebuilding the graph, and makes the generated mermaid diagrams etc.
easier to read.
"""

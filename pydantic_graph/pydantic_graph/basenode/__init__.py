"""The `BaseNode`-based graph runner.

This package is the permanent home for the original `pydantic_graph.Graph`
runner, where graphs are defined declaratively as a collection of
[`BaseNode`][pydantic_graph.BaseNode] subclasses. The same symbols are also
re-exported (without `.basenode`) from `pydantic_graph`, `pydantic_graph.graph`,
`pydantic_graph.persistence`, and `pydantic_graph.mermaid` for backwards
compatibility. Pin to `pydantic_graph.basenode` if you want a stable import
path that won't change meaning in v2.
"""

from .graph import Graph, GraphRun, GraphRunResult
from .persistence import (
    BaseStatePersistence,
    EndSnapshot,
    NodeSnapshot,
    Snapshot,
    SnapshotStatus,
)
from .persistence.file import FileStatePersistence
from .persistence.in_mem import FullStatePersistence, SimpleStatePersistence

__all__ = (
    'Graph',
    'GraphRun',
    'GraphRunResult',
    'BaseStatePersistence',
    'EndSnapshot',
    'NodeSnapshot',
    'Snapshot',
    'SnapshotStatus',
    'FileStatePersistence',
    'FullStatePersistence',
    'SimpleStatePersistence',
)

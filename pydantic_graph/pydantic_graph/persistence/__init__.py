"""Back-compat alias for [`pydantic_graph.basenode.persistence`][pydantic_graph.basenode.persistence].

In v1 this package was the canonical home of the state-persistence machinery
for the `BaseNode`-based `Graph` runner. The implementation has moved to
[`pydantic_graph.basenode.persistence`][pydantic_graph.basenode.persistence];
this package re-exports the same symbols so existing
`from pydantic_graph.persistence import …` imports keep working unchanged.

In v2 this name will likely be removed (or repurposed). Pin to
[`pydantic_graph.basenode.persistence`][pydantic_graph.basenode.persistence]
if you want a stable import path that won't change meaning.
"""

from ..basenode.persistence import (
    BaseStatePersistence,
    EndSnapshot,
    NodeSnapshot,
    RunEndT,
    Snapshot,
    SnapshotStatus,
    StateT,
    build_snapshot_list_type_adapter,
)

__all__ = (
    'BaseStatePersistence',
    'EndSnapshot',
    'NodeSnapshot',
    'RunEndT',
    'Snapshot',
    'SnapshotStatus',
    'StateT',
    'build_snapshot_list_type_adapter',
)

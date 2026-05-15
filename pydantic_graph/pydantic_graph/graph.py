"""Back-compat alias for [`pydantic_graph.basenode.graph`][pydantic_graph.basenode.graph].

In v1 this module was the canonical home of the `BaseNode`-based `Graph` class.
The implementation has moved to [`pydantic_graph.basenode.graph`][pydantic_graph.basenode.graph];
this module re-exports the same classes so existing
`from pydantic_graph.graph import Graph` imports keep working unchanged.

In v2, `pydantic_graph.Graph` will be repurposed to refer to the builder-based
`Graph` runner from [`pydantic_graph.graph_builder`][pydantic_graph.graph_builder].
If you want a stable import path that won't change meaning, pin to
[`pydantic_graph.basenode`][pydantic_graph.basenode].
"""

from .basenode.graph import Graph, GraphRun, GraphRunResult

__all__ = ('Graph', 'GraphRun', 'GraphRunResult')

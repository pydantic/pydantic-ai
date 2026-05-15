"""Back-compat alias for [`pydantic_graph.basenode.mermaid`][pydantic_graph.basenode.mermaid].

In v1 this module hosted the mermaid diagram generation helpers used by the
`BaseNode`-based `Graph` runner. The implementation has moved to
[`pydantic_graph.basenode.mermaid`][pydantic_graph.basenode.mermaid]; this
module re-exports the same symbols so existing
`from pydantic_graph.mermaid import generate_code` imports keep working
unchanged.

In v2, this name will likely be repurposed for the builder-based mermaid
helpers under [`pydantic_graph.graph_builder.mermaid`][pydantic_graph.graph_builder.mermaid].
Pin to [`pydantic_graph.basenode.mermaid`][pydantic_graph.basenode.mermaid] if
you want a stable import path that won't change meaning.
"""

from .basenode.mermaid import (
    DEFAULT_HIGHLIGHT_CSS,
    MermaidConfig,
    NodeIdent,
    StateDiagramDirection,
    generate_code,
    request_image,
    save_image,
)

__all__ = (
    'DEFAULT_HIGHLIGHT_CSS',
    'MermaidConfig',
    'NodeIdent',
    'StateDiagramDirection',
    'generate_code',
    'request_image',
    'save_image',
)

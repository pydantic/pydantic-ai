"""Builder-based graph execution framework.

!!! warning "Deprecated import path"
    The `pydantic_graph.beta` namespace was promoted to non-beta status in
    pydantic-ai 1.x. Import these symbols from `pydantic_graph` directly:

    ```python
    # Before
    from pydantic_graph.beta import GraphBuilder, StepContext

    # After
    from pydantic_graph import GraphBuilder, StepContext
    ```

    The `pydantic_graph.beta` namespace will be removed in v2.
"""

from __future__ import annotations as _annotations

import warnings
from typing import TYPE_CHECKING, Any

from pydantic_graph._warnings import PydanticGraphDeprecationWarning

if TYPE_CHECKING:
    from .graph import Graph
    from .graph_builder import GraphBuilder
    from .node import EndNode, StartNode
    from .step import StepContext, StepNode
    from .util import TypeExpression

__all__ = (
    'EndNode',
    'Graph',
    'GraphBuilder',
    'StartNode',
    'StepContext',
    'StepNode',
    'TypeExpression',
)


_TOP_LEVEL_TARGETS: dict[str, str | None] = {
    'EndNode': 'pydantic_graph.EndNode',
    # `Graph` from this beta module is the builder-based runner. It's not
    # re-exported at top level in 1.x because `pydantic_graph.Graph` still
    # refers to the legacy `BaseNode`-based runner; the rename happens in v2.
    'Graph': None,
    'GraphBuilder': 'pydantic_graph.GraphBuilder',
    'StartNode': 'pydantic_graph.StartNode',
    'StepContext': 'pydantic_graph.StepContext',
    'StepNode': 'pydantic_graph.StepNode',
    'TypeExpression': 'pydantic_graph.TypeExpression',
}


def __getattr__(name: str) -> Any:
    if name in _TOP_LEVEL_TARGETS:
        target = _TOP_LEVEL_TARGETS[name]
        if target is not None:
            msg = (
                f'Importing {name!r} from `pydantic_graph.beta` is deprecated, import it from `pydantic_graph` instead.'
            )
        else:
            msg = (
                f'`pydantic_graph.beta.{name}` is deprecated. The `pydantic_graph.beta` '
                f'namespace will be removed in v2.'
            )
        warnings.warn(msg, PydanticGraphDeprecationWarning, stacklevel=2)

        # Lazy submodule imports keep these named symbols routed through
        # `__getattr__` so the deprecation warning fires on access.
        if name == 'Graph':
            from .graph import Graph

            return Graph
        if name == 'GraphBuilder':
            from .graph_builder import GraphBuilder

            return GraphBuilder
        if name in ('EndNode', 'StartNode'):
            from . import node

            return getattr(node, name)
        if name in ('StepContext', 'StepNode'):
            from . import step

            return getattr(step, name)
        if name == 'TypeExpression':
            from .util import TypeExpression

            return TypeExpression

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

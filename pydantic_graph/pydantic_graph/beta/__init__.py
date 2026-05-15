"""Deprecated namespace for the builder-based graph API.

The builder-based graph API was renamed out of `pydantic_graph.beta` in v1.
Public symbols are now importable from
[`pydantic_graph.graph_builder`][pydantic_graph.graph_builder] (their
permanent home) and from `pydantic_graph` directly.

Importing from `pydantic_graph.beta` still works but emits a
[`PydanticGraphDeprecationWarning`][pydantic_graph.PydanticGraphDeprecationWarning].
The `pydantic_graph.beta` namespace will be removed in v2.
"""

from __future__ import annotations as _annotations

import warnings
from typing import TYPE_CHECKING, Any

from pydantic_graph._warnings import PydanticGraphDeprecationWarning

if TYPE_CHECKING:
    from pydantic_graph.graph_builder import (
        EndNode,
        Graph,
        GraphBuilder,
        StartNode,
        StepContext,
        StepNode,
        TypeExpression,
    )

__all__ = (
    'EndNode',
    'Graph',
    'GraphBuilder',
    'StartNode',
    'StepContext',
    'StepNode',
    'TypeExpression',
)


# Names previously importable as `from pydantic_graph.beta import X` along
# with their new canonical locations.
_FORWARDS: dict[str, str] = {
    'EndNode': 'pydantic_graph.graph_builder.EndNode',
    'Graph': 'pydantic_graph.graph_builder.Graph',
    'GraphBuilder': 'pydantic_graph.graph_builder.GraphBuilder',
    'StartNode': 'pydantic_graph.graph_builder.StartNode',
    'StepContext': 'pydantic_graph.graph_builder.StepContext',
    'StepNode': 'pydantic_graph.graph_builder.StepNode',
    'TypeExpression': 'pydantic_graph.graph_builder.TypeExpression',
}


def __getattr__(name: str) -> Any:
    if name in _FORWARDS:
        target = _FORWARDS[name]
        warnings.warn(
            f'Importing {name!r} from `pydantic_graph.beta` is deprecated, '
            f'import it from `{target.rsplit(".", 1)[0]}` (or `pydantic_graph`) instead.',
            PydanticGraphDeprecationWarning,
            stacklevel=2,
        )
        import pydantic_graph.graph_builder as _gb

        return getattr(_gb, name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

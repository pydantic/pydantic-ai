# ruff: noqa: I001  -- import order is load-bearing: legacy primitives must
# bind in the package namespace before the builder-based modules import them
# as `from pydantic_graph import BaseNode`.

"""Type-hint based graph library powering the Pydantic AI agent loop.

The package exposes two graph APIs:

- The builder-based API ([`GraphBuilder`][pydantic_graph.GraphBuilder] and
  friends), which is the recommended way to build graphs and is the future of
  this library.
- The original `BaseNode`-based [`Graph`][pydantic_graph.graph.Graph] runner,
  kept for backwards compatibility and now deprecated. Importing the runner,
  its result types, snapshot types, or persistence helpers from
  `pydantic_graph` (top level) emits a
  [`PydanticGraphDeprecationWarning`][pydantic_graph.PydanticGraphDeprecationWarning].
  [`BaseNode`][pydantic_graph.BaseNode], [`End`][pydantic_graph.End],
  [`GraphRunContext`][pydantic_graph.GraphRunContext], and
  [`Edge`][pydantic_graph.Edge] survive into v2 and are not deprecated.
"""

from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any

from ._warnings import PydanticGraphDeprecationWarning
from .exceptions import GraphRuntimeError, GraphSetupError
from .basenode import BaseNode, Edge, End, GraphRunContext

# Builder-based graph API. The implementation modules live at the top level
# (`pydantic_graph.step`, `pydantic_graph.decision`, `pydantic_graph.join`,
# `pydantic_graph.node`, etc.); `pydantic_graph.graph_builder` bundles the
# pieces (`GraphBuilder`, `Graph`/`GraphRun`, mermaid rendering) whose natural
# top-level names collide with the legacy `BaseNode`-based runner. The same
# symbols were exposed via `pydantic_graph.beta.*` in v1 — that namespace
# still works but emits a `PydanticGraphDeprecationWarning`.
from .decision import Decision
from .graph_builder import GraphBuilder
from .join import (
    Join,
    JoinNode,
    ReduceFirstValue,
    ReducerContext,
    ReducerFunction,
    reduce_dict_update,
    reduce_list_append,
    reduce_list_extend,
    reduce_null,
    reduce_sum,
)
from .node import EndNode, Fork, StartNode
from .step import Step, StepContext, StepNode
from .util import TypeExpression


if TYPE_CHECKING:
    # Re-exported lazily via `__getattr__` below with a deprecation warning.
    from .graph import Graph, GraphRun, GraphRunResult
    from .persistence import EndSnapshot, NodeSnapshot, Snapshot
    from .persistence.in_mem import FullStatePersistence, SimpleStatePersistence


# Legacy `BaseNode`-based runner symbols → canonical module they live in.
# Accessed lazily so importing `pydantic_graph` doesn't trigger the warning,
# only `from pydantic_graph import Graph` etc. does.
_DEPRECATED_LEGACY: dict[str, str] = {
    'Graph': 'pydantic_graph.graph',
    'GraphRun': 'pydantic_graph.graph',
    'GraphRunResult': 'pydantic_graph.graph',
    'EndSnapshot': 'pydantic_graph.persistence',
    'NodeSnapshot': 'pydantic_graph.persistence',
    'Snapshot': 'pydantic_graph.persistence',
    'FullStatePersistence': 'pydantic_graph.persistence.in_mem',
    'SimpleStatePersistence': 'pydantic_graph.persistence.in_mem',
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_LEGACY:
        import importlib
        import warnings

        warnings.warn(
            f'Importing `{name}` from `pydantic_graph` is deprecated. '
            f'The `BaseNode`-based `Graph` runner and its persistence machinery '
            f'are deprecated and will be removed (or repurposed) in v2; use the '
            f'builder-based `GraphBuilder` API instead, or pin to '
            f'`pydantic_graph<2` to keep using them.',
            PydanticGraphDeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(_DEPRECATED_LEGACY[name]), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


__all__ = (
    # Legacy `BaseNode`-based graph API (deprecated when imported from this
    # top-level namespace, except for `BaseNode`/`End`/`GraphRunContext`/`Edge`
    # which survive into v2)
    'BaseNode',
    'End',
    'GraphRunContext',
    'Edge',
    'Graph',
    'GraphRun',
    'GraphRunResult',
    'EndSnapshot',
    'Snapshot',
    'NodeSnapshot',
    'GraphSetupError',
    'GraphRuntimeError',
    'SimpleStatePersistence',
    'FullStatePersistence',
    # Builder-based graph API
    'GraphBuilder',
    'StepContext',
    'StepNode',
    'Step',
    'StartNode',
    'EndNode',
    'Fork',
    'Decision',
    'Join',
    'JoinNode',
    'ReducerContext',
    'ReducerFunction',
    'ReduceFirstValue',
    'reduce_dict_update',
    'reduce_list_append',
    'reduce_list_extend',
    'reduce_null',
    'reduce_sum',
    'TypeExpression',
    # Warnings
    'PydanticGraphDeprecationWarning',
)

# ruff: noqa: I001  -- import order is load-bearing: legacy primitives must
# bind in the package namespace before the builder-based `pydantic_graph.graph_builder.*`
# modules import them as `from pydantic_graph import BaseNode`.

from ._warnings import PydanticGraphDeprecationWarning
from .exceptions import GraphRuntimeError, GraphSetupError
from .nodes import BaseNode, Edge, End, GraphRunContext

# Legacy `BaseNode`-based graph API. The implementation lives at
# `pydantic_graph.basenode` (its permanent home); the top-level names below
# alias to those classes so existing imports continue to work unchanged.
# In v2, `pydantic_graph.Graph` will be repurposed to refer to the
# builder-based runner — users who want to keep the legacy class should pin
# to `pydantic_graph.basenode.Graph` etc.
from .basenode.graph import Graph, GraphRun, GraphRunResult
from .basenode.persistence import EndSnapshot, NodeSnapshot, Snapshot
from .basenode.persistence.in_mem import FullStatePersistence, SimpleStatePersistence

# Builder-based graph API. The implementation lives at
# `pydantic_graph.graph_builder` (its permanent home); the top-level names
# below alias to those classes. The same symbols were exposed via
# `pydantic_graph.beta.*` in v1 — that namespace still works but emits a
# `PydanticGraphDeprecationWarning`.
from .graph_builder.decision import Decision
from .graph_builder.graph_builder import GraphBuilder
from .graph_builder.join import (
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
from .graph_builder.node import EndNode, Fork, StartNode
from .graph_builder.step import Step, StepContext, StepNode
from .graph_builder.util import TypeExpression

__all__ = (
    # Legacy `BaseNode`-based graph API
    'Graph',
    'GraphRun',
    'GraphRunResult',
    'BaseNode',
    'End',
    'GraphRunContext',
    'Edge',
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

# ruff: noqa: I001  -- import order is load-bearing: legacy primitives must
# bind in the package namespace before `pydantic_graph.beta.*` modules import
# them as `from pydantic_graph import BaseNode`.

from ._warnings import PydanticGraphDeprecationWarning
from .exceptions import GraphRuntimeError, GraphSetupError
from .graph import Graph, GraphRun, GraphRunResult
from .nodes import BaseNode, Edge, End, GraphRunContext
from .persistence import EndSnapshot, NodeSnapshot, Snapshot
from .persistence.in_mem import FullStatePersistence, SimpleStatePersistence

# Builder-based graph API — promoted from `pydantic_graph.beta` in 1.x.
# Code physically lives in `pydantic_graph.beta.*` for the 1.x deprecation
# window; the relocation is a v2-cut concern (see #1548).
from .beta.decision import Decision
from .beta.graph_builder import GraphBuilder
from .beta.join import (
    Join,
    JoinNode,
    ReduceFirstValue,
    ReducerContext,
    reduce_dict_update,
    reduce_list_append,
    reduce_list_extend,
    reduce_null,
    reduce_sum,
)
from .beta.node import EndNode, Fork, StartNode
from .beta.step import Step, StepContext, StepNode
from .beta.util import TypeExpression

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
    # Builder-based graph API (promoted from `pydantic_graph.beta` in 1.x)
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

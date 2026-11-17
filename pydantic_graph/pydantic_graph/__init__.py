"""Type-hint based graph library powering the Pydantic AI agent loop.

Graphs are constructed with [`GraphBuilder`][pydantic_graph.GraphBuilder] from
typed step functions and (optionally) [`BaseNode`][pydantic_graph.BaseNode]
subclasses, then executed via [`Graph`][pydantic_graph.Graph] /
[`GraphRun`][pydantic_graph.GraphRun].
"""

from __future__ import annotations as _annotations

from .basenode import BaseNode, Edge, End, GraphRunContext
from .decision import Decision
from .exceptions import GraphRuntimeError, GraphSetupError
from .graph_builder import (
    EndMarker,
    ErrorMarker,
    Graph,
    GraphBuilder,
    GraphRun,
    GraphTask,
    GraphTaskRequest,
    JoinItem,
)
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

__all__ = (
    # Node primitives (declarative `BaseNode` style)
    'BaseNode',
    'End',
    'GraphRunContext',
    'Edge',
    # Builder API
    'GraphBuilder',
    'Graph',
    'GraphRun',
    'GraphTask',
    'GraphTaskRequest',
    'EndMarker',
    'ErrorMarker',
    'JoinItem',
    # Step / decision / join / topology nodes
    'Step',
    'StepContext',
    'StepNode',
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
    # Errors
    'GraphSetupError',
    'GraphRuntimeError',
)

"""The builder-based graph execution framework.

This package is the permanent home for the builder-based graph API that was
previously available under `pydantic_graph.beta`. The same symbols are also
re-exported from `pydantic_graph` directly. The `pydantic_graph.beta`
namespace still works for now but emits a
[`PydanticGraphDeprecationWarning`][pydantic_graph.PydanticGraphDeprecationWarning].

Pin to `pydantic_graph.graph_builder` if you want a stable import path that
won't change meaning when `pydantic_graph.Graph` is repurposed to refer to
this builder-based runner in v2.
"""

from .decision import Decision
from .graph import EndMarker, ErrorMarker, Graph, GraphRun, GraphTask, GraphTaskRequest, JoinItem
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
from .step import NodeStep, Step, StepContext, StepNode
from .util import TypeExpression

__all__ = (
    'Decision',
    'EndMarker',
    'EndNode',
    'ErrorMarker',
    'Fork',
    'Graph',
    'GraphBuilder',
    'GraphRun',
    'GraphTask',
    'GraphTaskRequest',
    'Join',
    'JoinItem',
    'JoinNode',
    'NodeStep',
    'ReduceFirstValue',
    'ReducerContext',
    'ReducerFunction',
    'StartNode',
    'Step',
    'StepContext',
    'StepNode',
    'TypeExpression',
    'reduce_dict_update',
    'reduce_list_append',
    'reduce_list_extend',
    'reduce_null',
    'reduce_sum',
)

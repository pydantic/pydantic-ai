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
    from pydantic_graph import (
        Decision,
        EndNode,
        Fork,
        Graph,
        GraphBuilder,
        Join,
        JoinNode,
        ReduceFirstValue,
        ReducerContext,
        ReducerFunction,
        StartNode,
        Step,
        StepContext,
        StepNode,
        TypeExpression,
        reduce_dict_update,
        reduce_list_append,
        reduce_list_extend,
        reduce_null,
        reduce_sum,
    )

__all__ = (
    'Decision',
    'EndNode',
    'Fork',
    'Graph',
    'GraphBuilder',
    'Join',
    'JoinNode',
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


# Names previously importable as `from pydantic_graph.beta import X` along
# with their new canonical locations. Most live as top-level
# `pydantic_graph.<module>.X`; the conflict-bound names (`Graph`,
# `GraphBuilder`) are bundled into `pydantic_graph.graph_builder`.
_FORWARDS: dict[str, str] = {
    'Decision': 'pydantic_graph.decision',
    'EndNode': 'pydantic_graph.node',
    'Fork': 'pydantic_graph.node',
    'Graph': 'pydantic_graph.graph_builder',
    'GraphBuilder': 'pydantic_graph.graph_builder',
    'Join': 'pydantic_graph.join',
    'JoinNode': 'pydantic_graph.join',
    'ReduceFirstValue': 'pydantic_graph.join',
    'ReducerContext': 'pydantic_graph.join',
    'ReducerFunction': 'pydantic_graph.join',
    'StartNode': 'pydantic_graph.node',
    'Step': 'pydantic_graph.step',
    'StepContext': 'pydantic_graph.step',
    'StepNode': 'pydantic_graph.step',
    'TypeExpression': 'pydantic_graph.util',
    'reduce_dict_update': 'pydantic_graph.join',
    'reduce_list_append': 'pydantic_graph.join',
    'reduce_list_extend': 'pydantic_graph.join',
    'reduce_null': 'pydantic_graph.join',
    'reduce_sum': 'pydantic_graph.join',
}


def __getattr__(name: str) -> Any:
    if name in _FORWARDS:
        import importlib

        target_module = _FORWARDS[name]
        warnings.warn(
            f'Importing {name!r} from `pydantic_graph.beta` is deprecated, '
            f'import it from `{target_module}` (or `pydantic_graph`) instead.',
            PydanticGraphDeprecationWarning,
            stacklevel=2,
        )
        return getattr(importlib.import_module(target_module), name)

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')

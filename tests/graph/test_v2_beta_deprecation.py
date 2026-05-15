"""Tests for the deprecation of the `pydantic_graph.beta` namespace.

The builder-based graph API has been renamed out of `pydantic_graph.beta`. Its
canonical home is now [`pydantic_graph.graph_builder`][pydantic_graph.graph_builder]
(and the public symbols are also re-exported from `pydantic_graph`). Imports
via `pydantic_graph.beta` still resolve, but each emits a
[`PydanticGraphDeprecationWarning`][pydantic_graph.PydanticGraphDeprecationWarning].
"""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

from pydantic_graph import (
    Decision,
    EndNode,
    Fork,
    GraphBuilder,
    Join,
    JoinNode,
    PydanticGraphDeprecationWarning,
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

# Names that were importable as `from pydantic_graph.beta import X` in v1.
_BETA_NAMES = [
    'EndNode',
    'Graph',
    'GraphBuilder',
    'StartNode',
    'StepContext',
    'StepNode',
    'TypeExpression',
]

# Submodules that were importable as `from pydantic_graph.beta.X import …` in v1.
_BETA_SUBMODULES = [
    'decision',
    'graph',
    'graph_builder',
    'id_types',
    'join',
    'mermaid',
    'node',
    'node_types',
    'parent_forks',
    'paths',
    'step',
    'util',
]


@pytest.mark.parametrize('name', _BETA_NAMES)
def test_beta_package_emits_deprecation(name: str) -> None:
    """`from pydantic_graph.beta import X` warns and forwards to `pydantic_graph.graph_builder`."""
    import pydantic_graph.beta as beta
    import pydantic_graph.graph_builder as gb

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        forwarded = getattr(beta, name)

    dep_warnings = [w for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert dep_warnings, [str(w.message) for w in caught]
    assert any(name in str(w.message) and 'graph_builder' in str(w.message) for w in dep_warnings)
    assert forwarded is getattr(gb, name)


def test_beta_unknown_attribute_raises() -> None:
    import pydantic_graph.beta as beta

    with pytest.raises(AttributeError):
        getattr(beta, 'NotARealName')


@pytest.mark.parametrize('submodule', _BETA_SUBMODULES)
def test_beta_submodule_emits_deprecation(submodule: str) -> None:
    """`import pydantic_graph.beta.<submodule>` warns and re-exports the new module's public names."""
    full_name = f'pydantic_graph.beta.{submodule}'
    sys.modules.pop(full_name, None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        beta_mod = importlib.import_module(full_name)

    dep_warnings = [w for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert dep_warnings, [str(w.message) for w in caught]
    assert any(submodule in str(w.message) and 'graph_builder' in str(w.message) for w in dep_warnings)

    new_mod = importlib.import_module(f'pydantic_graph.graph_builder.{submodule}')
    # The shim does `from <new_mod> import *`, so every public attribute defined in `new_mod`
    # should resolve to the same object via the deprecated path. (Iterate `dir(new_mod)` filtered
    # to non-underscore names whose `__module__` matches the new module — those are the symbols
    # the shim's wildcard import would pull in.)
    forwarded = [
        n
        for n in dir(new_mod)
        if not n.startswith('_') and getattr(getattr(new_mod, n), '__module__', None) == new_mod.__name__
    ]
    assert forwarded, f'{submodule!r} has no public symbols to forward'
    for name in forwarded:
        assert getattr(beta_mod, name) is getattr(new_mod, name)


def test_top_level_symbols_loadable() -> None:
    """All public builder-API names import from `pydantic_graph` without warnings."""
    assert GraphBuilder is not None
    assert StepContext is not None
    assert StepNode is not None
    assert Step is not None
    assert StartNode is not None
    assert EndNode is not None
    assert Fork is not None
    assert Decision is not None
    assert Join is not None
    assert JoinNode is not None
    assert ReducerContext is not None
    assert ReducerFunction is not None
    assert ReduceFirstValue is not None
    assert reduce_dict_update is not None
    assert reduce_list_append is not None
    assert reduce_list_extend is not None
    assert reduce_null is not None
    assert reduce_sum is not None
    assert TypeExpression is not None


def test_top_level_builder_symbols_match_graph_builder() -> None:
    """Top-level builder-API symbols are the same objects as their `graph_builder` counterparts."""
    import pydantic_graph.graph_builder as gb

    assert GraphBuilder is gb.GraphBuilder
    assert StepContext is gb.StepContext
    assert StepNode is gb.StepNode
    assert EndNode is gb.EndNode
    assert StartNode is gb.StartNode
    assert TypeExpression is gb.TypeExpression

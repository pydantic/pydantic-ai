"""Tests for the deprecation of the `pydantic_graph.beta` namespace.

The builder-based graph API has been moved out of `pydantic_graph.beta`. The
non-conflicting modules (`step`, `decision`, `join`, `node`, etc.) live at the
top level of `pydantic_graph`; the names that would collide with the legacy
`BaseNode`-based runner (`Graph`, `GraphBuilder`, builder-mermaid) are bundled
in [`pydantic_graph.graph_builder`][pydantic_graph.graph_builder]. All public
symbols are also re-exported from `pydantic_graph` directly. Imports via
`pydantic_graph.beta` still resolve, but each emits a
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

# `from pydantic_graph.beta import X` (top-level beta package) → canonical module
# the symbol now lives in.
_BETA_NAME_TO_MODULE: dict[str, str] = {
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

# `import pydantic_graph.beta.<submodule>` → canonical module the shim forwards
# to. Most submodules forward to a top-level same-named module; the names that
# collide with the legacy runner (`graph`, `graph_builder`, `mermaid`) all
# forward to the bundled `pydantic_graph.graph_builder`.
_BETA_SUBMODULE_TO_TARGET: dict[str, str] = {
    'decision': 'pydantic_graph.decision',
    'graph': 'pydantic_graph.graph_builder',
    'graph_builder': 'pydantic_graph.graph_builder',
    'id_types': 'pydantic_graph.id_types',
    'join': 'pydantic_graph.join',
    'mermaid': 'pydantic_graph.graph_builder',
    'node': 'pydantic_graph.node',
    'node_types': 'pydantic_graph.node_types',
    'parent_forks': 'pydantic_graph.parent_forks',
    'paths': 'pydantic_graph.paths',
    'step': 'pydantic_graph.step',
    'util': 'pydantic_graph.util',
}


@pytest.mark.parametrize('name', list(_BETA_NAME_TO_MODULE))
def test_beta_package_emits_deprecation(name: str) -> None:
    """`from pydantic_graph.beta import X` warns and forwards to the new canonical module."""
    import pydantic_graph.beta as beta

    target_module_name = _BETA_NAME_TO_MODULE[name]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        forwarded = getattr(beta, name)

    dep_warnings = [w for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert dep_warnings, [str(w.message) for w in caught]
    assert any(name in str(w.message) and target_module_name in str(w.message) for w in dep_warnings)

    target_module = importlib.import_module(target_module_name)
    assert forwarded is getattr(target_module, name)


def test_beta_unknown_attribute_raises() -> None:
    import pydantic_graph.beta as beta

    with pytest.raises(AttributeError):
        getattr(beta, 'NotARealName')


@pytest.mark.parametrize('submodule', list(_BETA_SUBMODULE_TO_TARGET))
def test_beta_submodule_emits_deprecation(submodule: str) -> None:
    """`import pydantic_graph.beta.<submodule>` warns and re-exports the target module's public names."""
    full_name = f'pydantic_graph.beta.{submodule}'
    sys.modules.pop(full_name, None)
    target_module_name = _BETA_SUBMODULE_TO_TARGET[submodule]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        beta_mod = importlib.import_module(full_name)

    dep_warnings = [w for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert dep_warnings, [str(w.message) for w in caught]
    assert any(submodule in str(w.message) and target_module_name in str(w.message) for w in dep_warnings)

    target_module = importlib.import_module(target_module_name)
    # The shim does `from <target_module> import *`, so every public attribute defined in
    # `target_module` should resolve to the same object via the deprecated path.
    forwarded = [
        n
        for n in dir(target_module)
        if not n.startswith('_') and getattr(getattr(target_module, n), '__module__', None) == target_module.__name__
    ]
    assert forwarded, f'{submodule!r} has no public symbols to forward'
    for name in forwarded:
        assert getattr(beta_mod, name) is getattr(target_module, name)


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


def test_top_level_builder_symbols_match_modules() -> None:
    """Top-level builder-API symbols are the same objects as their canonical-module counterparts."""
    import pydantic_graph.decision as decision
    import pydantic_graph.graph_builder as graph_builder
    import pydantic_graph.join as join
    import pydantic_graph.node as node
    import pydantic_graph.step as step
    import pydantic_graph.util as util

    assert GraphBuilder is graph_builder.GraphBuilder
    assert StepContext is step.StepContext
    assert StepNode is step.StepNode
    assert Step is step.Step
    assert StartNode is node.StartNode
    assert EndNode is node.EndNode
    assert Fork is node.Fork
    assert Decision is decision.Decision
    assert Join is join.Join
    assert JoinNode is join.JoinNode
    assert TypeExpression is util.TypeExpression

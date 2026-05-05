"""Tests for the `pydantic_graph.beta` -> top-level promotion.

Phase A: top-level imports work for the public symbols, and importing those
symbols via `from pydantic_graph.beta import X` emits a deprecation warning.
"""

from __future__ import annotations

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


@pytest.mark.parametrize(
    'name',
    ['EndNode', 'GraphBuilder', 'StartNode', 'StepContext', 'StepNode', 'TypeExpression'],
)
def test_beta_package_emits_deprecation(name: str) -> None:
    """Importing public symbols from `pydantic_graph.beta` emits a deprecation warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        import pydantic_graph.beta as beta

        getattr(beta, name)

    assert any(
        issubclass(w.category, PydanticGraphDeprecationWarning)
        and name in str(w.message)
        and 'pydantic_graph' in str(w.message)
        for w in caught
    ), [str(w.message) for w in caught]


def test_beta_graph_special_message() -> None:
    """`pydantic_graph.beta.Graph` is special-cased: no top-level alias yet."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        import pydantic_graph.beta as beta

        _ = beta.Graph

    msg_strs = [str(w.message) for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert any('removed in v2' in m for m in msg_strs), msg_strs


def test_beta_unknown_attribute_raises() -> None:
    import pydantic_graph.beta as beta

    with pytest.raises(AttributeError):
        getattr(beta, 'NotARealName')


def test_top_level_symbols_match_beta() -> None:
    """Top-level public symbols are the same objects as their beta counterparts."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', PydanticGraphDeprecationWarning)
        import pydantic_graph.beta as beta

        assert GraphBuilder is beta.GraphBuilder
        assert StepContext is beta.StepContext
        assert StepNode is beta.StepNode
        assert EndNode is beta.EndNode
        assert StartNode is beta.StartNode
        assert TypeExpression is beta.TypeExpression


def test_top_level_symbols_loadable() -> None:
    """All re-exported names import from `pydantic_graph` without warnings."""
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
    assert ReduceFirstValue is not None
    assert reduce_dict_update is not None
    assert reduce_list_append is not None
    assert reduce_list_extend is not None
    assert reduce_null is not None
    assert reduce_sum is not None
    assert TypeExpression is not None


def test_beta_submodule_imports_do_not_warn() -> None:
    """Direct submodule imports still work without a warning (Phase A scope).

    Phase A only fires the deprecation when going through the `pydantic_graph.beta`
    package namespace; the v2-cut PR will move the modules and remove the namespace.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        from pydantic_graph.beta.graph_builder import GraphBuilder as _GB

        _ = _GB

    dep_warnings = [w for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert dep_warnings == [], [str(w.message) for w in dep_warnings]

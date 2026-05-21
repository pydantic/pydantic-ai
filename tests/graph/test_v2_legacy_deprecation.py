"""Tests for the deprecation of top-level legacy `BaseNode`-runner imports.

In v2, `pydantic_graph.Graph` will be repurposed to refer to the builder-based
runner (currently at [`pydantic_graph.graph_builder.Graph`][pydantic_graph.graph_builder.Graph]),
and the legacy `BaseNode`-based runner / persistence machinery is being
phased out. To prepare for that, importing the affected names from the
`pydantic_graph` top-level emits a
[`PydanticGraphDeprecationWarning`][pydantic_graph.PydanticGraphDeprecationWarning].

Importing from the canonical submodule (e.g. `from pydantic_graph.graph import
Graph`, `from pydantic_graph.persistence import EndSnapshot`) is *not* warned —
that's the escape hatch for callers that still need the legacy class while the
deprecation runs its course. Survivors of v2 (`BaseNode`, `End`,
`GraphRunContext`, `Edge`) are not warned even at the top level.
"""

from __future__ import annotations

import importlib
import warnings

import pytest

from pydantic_graph import PydanticGraphDeprecationWarning

# Top-level legacy name → canonical submodule it forwards to (and where the
# import works without firing a deprecation warning).
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

# Top-level survivors that must NOT warn — these are part of the v2 surface.
_NOT_DEPRECATED = ('BaseNode', 'End', 'GraphRunContext', 'Edge')


@pytest.mark.parametrize('name', list(_DEPRECATED_LEGACY))
def test_top_level_legacy_import_emits_deprecation(name: str) -> None:
    """`getattr(pydantic_graph, X)` warns and forwards to the canonical submodule."""
    import pydantic_graph

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        forwarded = getattr(pydantic_graph, name)

    dep_warnings = [w for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert dep_warnings, [str(w.message) for w in caught]
    assert any(name in str(w.message) for w in dep_warnings)

    canonical_module = importlib.import_module(_DEPRECATED_LEGACY[name])
    assert forwarded is getattr(canonical_module, name)


@pytest.mark.parametrize('name', _NOT_DEPRECATED)
def test_top_level_survivor_import_does_not_warn(name: str) -> None:
    """Names that survive into v2 must not emit `PydanticGraphDeprecationWarning` at import."""
    import pydantic_graph

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        getattr(pydantic_graph, name)

    dep_warnings = [w for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert not dep_warnings, [str(w.message) for w in dep_warnings]


def test_top_level_unknown_attribute_raises() -> None:
    import pydantic_graph

    with pytest.raises(AttributeError):
        getattr(pydantic_graph, 'NotARealName')


def test_canonical_submodule_imports_do_not_warn() -> None:
    """Importing from the canonical submodule (e.g. `pydantic_graph.graph`) is the un-warned escape hatch."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for module_name, name in (
            ('pydantic_graph.graph', 'Graph'),
            ('pydantic_graph.persistence', 'EndSnapshot'),
            ('pydantic_graph.persistence.in_mem', 'FullStatePersistence'),
        ):
            assert getattr(importlib.import_module(module_name), name) is not None

    dep_warnings = [w for w in caught if issubclass(w.category, PydanticGraphDeprecationWarning)]
    assert not dep_warnings, [str(w.message) for w in dep_warnings]

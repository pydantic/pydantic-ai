"""Properties of the tool catalog that `FunctionSignature` renders.

Tools are drawn by `tests/function_signature_strategies.py`, and every property is a checker in
`tests/function_signature_invariants.py` that `test_function_signature_invariants.py` shows can fail.

Hunt for new failures with `FUNCTION_SIGNATURE_PROPERTY_EXAMPLES=2000 uv run pytest
tests/test_function_signature_properties.py`, which draws that many randomized examples per test.
"""

from __future__ import annotations as _annotations

import os
from copy import deepcopy

import pytest
from hypothesis import HealthCheck, given, settings

from pydantic_ai.function_signature import FunctionSignature

from .function_signature_invariants import Tool, catalog_violations, stability_violations
from .function_signature_strategies import exhaustive_documents, tool_sets

# The same false positive `tests/test_history_properties.py` explains: Hypothesis notes this when another thread on
# the xdist worker touches `random` during a draw. No strategy in `tests/function_signature_strategies.py` uses it.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Do not use the `random` module inside strategies:hypothesis.errors.HypothesisDeprecationWarning'
)

_EXAMPLES = os.environ.get('FUNCTION_SIGNATURE_PROPERTY_EXAMPLES')

PROPERTY_SETTINGS = settings(
    max_examples=int(_EXAMPLES) if _EXAMPLES else 50,
    derandomize=not _EXAMPLES,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def _signatures(tools: list[Tool]) -> list[FunctionSignature]:
    return [
        FunctionSignature.from_schema(name=t.name, parameters_schema=t.params, return_schema=t.returns) for t in tools
    ]


def _render(signatures: list[FunctionSignature]) -> str:
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    definitions = FunctionSignature.render_type_definitions(signatures, conflicting)
    functions = [s.render('...', conflicting_type_names=conflicting) for s in signatures]
    return '\n\n'.join([*definitions, *functions])


@given(tool_sets())
@PROPERTY_SETTINGS
def test_catalog_describes_every_tool(tools: list[Tool]):
    text = _render(_signatures(tools))
    assert catalog_violations(text, tools) == [], text


@given(tool_sets())
@PROPERTY_SETTINGS
def test_rendering_again_gives_the_same_text(tools: list[Tool]):
    signatures = _signatures(tools)
    assert stability_violations(_render(signatures), _render(signatures)) == []


@given(tool_sets())
@PROPERTY_SETTINGS
def test_subset_of_a_catalog_keeps_its_names(tools: list[Tool]):
    """Rendering part of a catalog, as dynamic workflows do, reuses the whole catalog's names."""
    signatures = _signatures(tools)
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    whole = FunctionSignature.render_type_definitions(signatures, conflicting)
    part = FunctionSignature.render_type_definitions(signatures[:1], conflicting)
    assert set(part) <= set(whole)
    whole_text = _render(signatures)
    assert signatures[0].render('...', conflicting_type_names=conflicting) in whole_text


@given(tool_sets())
@PROPERTY_SETTINGS
def test_copied_and_rebuilt_signatures_read_the_same_names(tools: list[Tool]):
    """Names survive a copy, and building the same tools again gives the same text."""
    signatures = _signatures(tools)
    conflicting = FunctionSignature.get_conflicting_type_names(signatures)
    expected = FunctionSignature.render_type_definitions(signatures, conflicting)
    assert FunctionSignature.render_type_definitions(deepcopy(signatures), conflicting) == expected
    assert _render(_signatures(tools)) == _render(signatures)


def test_every_small_pair_of_tools():
    """Every pair of the smallest documents, rather than a sample: within this bound nothing is left to chance."""
    documents = exhaustive_documents(with_optional_fields=bool(_EXAMPLES))
    for tool_a_params in documents:
        for tool_b_params in documents:
            tools = [Tool('a', tool_a_params), Tool('a_b', tool_b_params)]
            text = _render(_signatures(tools))
            assert catalog_violations(text, tools) == [], (tools, text)


def test_every_small_document_as_params_and_return():
    documents = exhaustive_documents(with_optional_fields=bool(_EXAMPLES))
    for params in documents:
        for returns in documents:
            tools = [Tool('a', params, returns)]
            text = _render(_signatures(tools))
            assert catalog_violations(text, tools) == [], (tools, text)

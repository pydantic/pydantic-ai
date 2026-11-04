"""Hypothesis strategies that draw tool sets whose schemas collide on names on purpose.

Small name pools make every tool share type names, which is where catalogs break. The pools also hold names
Python can't write plainly: keywords, dashes, dunder prefixes, and names of things the rendered text relies on.
"""

# Schemas are arbitrary JSON, so most values here are unknown to the type checker.
# pyright: reportUnknownArgumentType=false, reportUnknownLambdaType=false

from __future__ import annotations as _annotations

from itertools import product
from typing import Any

from hypothesis import strategies as st

from .function_signature_invariants import Schema, Tool

TOOL_NAMES = ['search', 'a', 'a_b', 'User', 'tool']
TYPE_NAMES = ['User', 'b_User', 'tool_User', 'Any', 'str', 'my.Type', 'class', 'a/b']
FIELD_NAMES = ['f', 'g', 'from', 'user-id', '__x', 'str', 'User']
PARAM_NAMES = ['p', 'q', 'from', 'user-id']

_PRIMITIVE = st.sampled_from([{'type': t} for t in ('string', 'integer', 'boolean', 'number')])


def _escape(name: str) -> str:
    return name.replace('~', '~0').replace('/', '~1')


def _schema_node(names: list[str], depth: int) -> st.SearchStrategy[Any]:
    leaves: list[st.SearchStrategy[Any]] = [
        _PRIMITIVE,
        st.just({}),
        st.just({'anyOf': []}),
        st.just({'enum': []}),
        st.sampled_from([{'enum': ['x']}, {'enum': [1, 'y']}, {'enum': ["it's"]}, {'enum': ['a\nb']}, {'const': 'c'}]),
        st.sampled_from([{'type': ['string', 'null']}, {'type': ['integer', 'null']}]),
        st.just({'type': 'string', 'description': 'hostile """ # \\ text\nsecond line'}),
        st.just({'type': 'integer', 'default': 5}),
        st.sampled_from([{'$ref': '#/nowhere'}, {'$ref': 'https://example.com/s.json#/$defs/User'}, {'$ref': '#'}]),
    ]
    if names:
        leaves.append(st.sampled_from(names).map(lambda n: {'$ref': f'#/$defs/{_escape(n)}'}))
        leaves.append(st.sampled_from(names).map(lambda n: {'$ref': f'#/definitions/{_escape(n)}'}))
    if depth <= 0:
        return st.one_of(leaves)
    child = _schema_node(names, depth - 1)
    return st.one_of(
        *leaves,
        child.map(lambda c: {'type': 'array', 'items': c}),
        child.map(lambda c: {'type': 'object', 'additionalProperties': c}),
        child.map(lambda c: {'anyOf': [c, {'type': 'null'}]}),
        child.map(lambda c: {'oneOf': [c, {'type': 'string'}]}),
        _object(names, depth - 1, FIELD_NAMES),
        st.builds(
            lambda title, obj: {**obj, 'title': title},
            st.sampled_from(TYPE_NAMES),
            _object(names, depth - 1, FIELD_NAMES),
        ),
    )


def _object(names: list[str], depth: int, field_names: list[str]) -> st.SearchStrategy[Schema]:
    def build(chosen: list[str], nodes: list[Any], required: list[bool]) -> Schema:
        properties = dict(zip(chosen, nodes))
        return {
            'type': 'object',
            'properties': properties,
            'required': [name for name, keep in zip(chosen, required) if keep],
        }

    return st.lists(st.sampled_from(field_names), min_size=1, max_size=3, unique=True).flatmap(
        lambda chosen: st.builds(
            build,
            st.just(chosen),
            st.lists(_schema_node(names, depth), min_size=len(chosen), max_size=len(chosen)),
            st.lists(st.booleans(), min_size=len(chosen), max_size=len(chosen)),
        )
    )


def _document(field_names: list[str], *, depth: int = 1) -> st.SearchStrategy[Schema]:
    """A schema document: a root object plus `$defs` and `definitions` its refs can point at."""

    def with_defs(names: list[str]) -> st.SearchStrategy[Schema]:
        defs = st.lists(
            st.one_of(_object(names, depth, FIELD_NAMES), _schema_node(names, 0)),
            min_size=len(names),
            max_size=len(names),
        )
        return st.builds(
            lambda root, values: {
                **root,
                '$defs': dict(zip(names, values)),
                'definitions': dict(zip(names, values)),
            },
            _object(names, depth, field_names),
            defs,
        )

    return st.lists(st.sampled_from(TYPE_NAMES), max_size=3, unique=True).flatmap(with_defs)


@st.composite
def tool_sets(draw: st.DrawFn) -> list[Tool]:
    """One to three tools with distinct names, each with its own schema documents."""
    names = draw(st.lists(st.sampled_from(TOOL_NAMES), min_size=1, max_size=3, unique=True))
    tools: list[Tool] = []
    for name in names:
        params = draw(_document(PARAM_NAMES))
        returns = draw(st.none() | _document(FIELD_NAMES))
        tools.append(Tool(name=name, params=params, returns=returns))
    return tools


def exhaustive_documents(*, with_optional_fields: bool) -> list[Schema]:
    """Every document in a small space: a parameter pointing at `A`, and `A` and `B` with one field of each form.

    Comparing all pairs is quadratic, so CI leaves out optional fields, which multiply the pairs by four.

    Documents differ only deep inside, in the form of a field, its name or whether it is required, so across a
    pair of tools they are the near twins that merging and renaming can get wrong. Comparing every pair means
    no bug in this space depends on what a run happens to draw.
    """
    forms: list[Any] = [
        {'type': 'string'},
        {'$ref': '#/$defs/A'},
        {'$ref': '#/$defs/B'},
        {'type': 'array', 'items': {'$ref': '#/$defs/B'}},
        {'anyOf': [{'$ref': '#/$defs/A'}, {'type': 'null'}]},
    ]

    def one_field(name: str, node: Any, *, required: bool = True) -> Schema:
        return {'type': 'object', 'properties': {name: node}, 'required': [name] if required else []}

    return [
        {
            **one_field('p', {'$ref': '#/$defs/A'}),
            '$defs': {'A': one_field('f', a), 'B': one_field(b_name, b, required=b_required)},
        }
        for a, b, b_name, b_required in product(
            forms, forms, ['f', 'from'], [True, False] if with_optional_fields else [True]
        )
    ]

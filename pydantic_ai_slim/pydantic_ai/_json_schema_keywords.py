"""Where a JSON Schema keeps its subschemas: each keyword whose value holds them, and the shape of that value.

`SUBSCHEMA_KEYWORDS` is the one record of those keywords and shapes, and `iter_subschemas` and `map_subschemas` read it.
`merge_json_schema_defs` reaches every keyword through it; `JsonSchemaTransformer` reads the shapes but picks which
keywords it enters, and on which schema types. `tests/test_json_schema_conformance.py` fails when a walker misses a
keyword added here.

This module imports nothing from Pydantic AI, so `_utils` and `_json_schema` can both import it.
"""

from __future__ import annotations as _annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from functools import partial
from typing import Literal, TypeAlias, TypeGuard

Subschema: TypeAlias = 'dict[str, object] | bool'
"""A JSON Schema: draft 2020-12, section 4.3 says "A JSON Schema MUST be an object or a boolean"."""

SubschemaShape: TypeAlias = Literal['schema', 'schema_map', 'schema_list', 'schema_or_list']
"""How a keyword's value holds subschemas: one schema, an object of them, a list of them, or one schema or a list."""

SUBSCHEMA_KEYWORDS: Mapping[str, SubschemaShape] = {
    # Draft 2020-12 applicator vocabulary.
    'allOf': 'schema_list',
    'anyOf': 'schema_list',
    'oneOf': 'schema_list',
    'not': 'schema',
    'if': 'schema',
    'then': 'schema',
    'else': 'schema',
    'dependentSchemas': 'schema_map',
    'prefixItems': 'schema_list',
    # Drafts before 2020-12 spell a tuple as an `items` list; 2020-12 replaced it with `prefixItems`.
    'items': 'schema_or_list',
    'contains': 'schema',
    'properties': 'schema_map',
    'patternProperties': 'schema_map',
    'additionalProperties': 'schema',
    'propertyNames': 'schema',
    # Draft 2020-12 unevaluated, content and core vocabularies.
    'unevaluatedItems': 'schema',
    'unevaluatedProperties': 'schema',
    'contentSchema': 'schema',
    '$defs': 'schema_map',
    # Draft 7, which 2019-09 replaced with `$defs`, `prefixItems` and `dependentSchemas`.
    'definitions': 'schema_map',
    'additionalItems': 'schema',
    # A property dependency's value is a list of property names instead, which isn't a subschema.
    'dependencies': 'schema_map',
}
"""Every keyword whose value holds subschemas, in draft 2020-12 and draft 7."""


def is_subschema(value: object) -> TypeGuard[Subschema]:
    """Whether `value` is a JSON Schema: an object or a boolean."""
    return isinstance(value, dict | bool)


def _is_schema_map(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, dict)


def _is_schema_list(value: object) -> TypeGuard[Sequence[object]]:
    # A tuple of schemas worked before the shape check existed: the walkers iterate it like a list.
    return isinstance(value, list | tuple)


def _members(shape: SubschemaShape, value: object) -> Iterable[object]:
    """The members of a keyword's value that can hold a subschema, given the keyword's shape."""
    if shape == 'schema_map':
        return value.values() if _is_schema_map(value) else ()
    if shape != 'schema' and _is_schema_list(value):
        return value
    return () if shape == 'schema_list' else (value,)


def iter_subschemas(node: Mapping[str, object], keywords: Iterable[str] = SUBSCHEMA_KEYWORDS) -> Iterator[Subschema]:
    """Each subschema directly under `node`, through `keywords`, skipping a member that isn't a schema."""
    for keyword in keywords:
        if (value := node.get(keyword)) is not None:
            yield from (member for member in _members(SUBSCHEMA_KEYWORDS[keyword], value) if is_subschema(member))


def map_subschemas(node: Mapping[str, object], fn: Callable[[str, Subschema], object]) -> dict[str, object]:
    """A copy of `node` with `fn(keyword, subschema)` in place of each subschema directly under it.

    Like `iter_subschemas`, it reads every keyword in `SUBSCHEMA_KEYWORDS`, and a member that isn't a schema stays.
    """
    mapped = dict(node)
    for keyword, value in node.items():
        if (shape := SUBSCHEMA_KEYWORDS.get(keyword)) is not None:
            mapped[keyword] = _map_members(shape, value, partial(fn, keyword))
    return mapped


def _map_members(shape: SubschemaShape, value: object, fn: Callable[[Subschema], object]) -> object:
    """A keyword's value with `fn` applied to each of its members that is a schema, given the keyword's shape."""
    if shape == 'schema_map':
        return {name: _map_member(member, fn) for name, member in value.items()} if _is_schema_map(value) else value
    if shape != 'schema' and _is_schema_list(value):
        return [_map_member(member, fn) for member in value]
    return value if shape == 'schema_list' else _map_member(value, fn)


def _map_member(member: object, fn: Callable[[Subschema], object]) -> object:
    return fn(member) if is_subschema(member) else member

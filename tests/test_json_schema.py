"""Tests for the _json_schema module."""

from __future__ import annotations as _annotations

from copy import deepcopy
from typing import Any

import pytest

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer


class _PassthroughTransformer(JsonSchemaTransformer):
    def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
        return schema


def test_simplify_nullable_unions():
    """Test the simplify_nullable_unions feature (deprecated, to be removed in v2)."""

    # Test with simplify_nullable_unions=True
    schema_with_null = {
        'anyOf': [
            {'type': 'string'},
            {'type': 'null'},
        ]
    }
    transformer = _PassthroughTransformer(schema_with_null, simplify_nullable_unions=True)
    result = transformer.walk()

    # Should collapse to a single nullable string
    assert result == {'type': 'string', 'nullable': True}

    # Test with simplify_nullable_unions=False (default)
    transformer2 = _PassthroughTransformer(schema_with_null, simplify_nullable_unions=False)
    result2 = transformer2.walk()

    # Should keep the anyOf structure
    assert 'anyOf' in result2
    assert len(result2['anyOf']) == 2

    # Test that non-nullable unions are unaffected
    schema_no_null = {
        'anyOf': [
            {'type': 'string'},
            {'type': 'number'},
        ]
    }
    transformer3 = _PassthroughTransformer(schema_no_null, simplify_nullable_unions=True)
    result3 = transformer3.walk()

    # Should keep anyOf since it's not nullable
    assert 'anyOf' in result3
    assert len(result3['anyOf']) == 2


def test_schema_defs_not_modified():
    """Test that the original schema $defs are not modified during transformation."""

    # Create a schema with $defs that should not be modified
    original_schema = {
        'type': 'object',
        'properties': {'value': {'$ref': '#/$defs/TestUnion'}},
        '$defs': {
            'TestUnion': {
                'anyOf': [
                    {'type': 'string'},
                    {'type': 'number'},
                ],
                'title': 'TestUnion',
            }
        },
    }

    # Keep a deepcopy to compare against later
    original_schema_copy = deepcopy(original_schema)

    # Transform the schema
    transformer = _PassthroughTransformer(original_schema)
    result = transformer.walk()

    # Verify the original schema was not modified
    assert original_schema == original_schema_copy

    # Verify the result is correct
    assert result == original_schema_copy


@pytest.mark.parametrize('value_schema', [True, False])
def test_boolean_schema_nodes_round_trip(value_schema: bool):
    """Boolean JSON Schema nodes should not crash the walker."""

    original_schema = {
        'type': 'object',
        'properties': {
            'fields': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'value': value_schema,
                    },
                },
            }
        },
    }

    transformer = _PassthroughTransformer(original_schema)

    assert transformer.walk() == original_schema


def test_boolean_schema_in_single_member_union():
    """A union that collapses to a single boolean member should be preserved."""

    schema = {'anyOf': [True]}
    result = _PassthroughTransformer(schema).walk()
    assert result == {'anyOf': [True]}


def test_simplify_nullable_union_with_boolean_member():
    """simplify_nullable_unions should not crash when a member is a boolean schema."""

    schema = {'anyOf': [True, {'type': 'null'}]}
    result = _PassthroughTransformer(schema, simplify_nullable_unions=True).walk()
    assert result == {'anyOf': [True, {'type': 'null'}]}


def test_allof_members_are_recursed():
    """allOf composition members should be recursed into by the walker, like anyOf/oneOf.

    This is a unit test because the walker is an internal helper and the regression is
    about its recursion shape, not a provider payload a VCR cassette would catch.
    """

    visited: list[dict[str, Any]] = []

    class _VisitingTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            if 'type' in schema and schema.get('type') == 'object':
                visited.append(schema)
            return schema

    schema = {
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
        ],
    }

    result = _VisitingTransformer(deepcopy(schema)).walk()

    # Both allOf members were recursed into (their object subschemas were visited).
    assert visited == [
        {'type': 'object', 'properties': {'a': {'type': 'string'}}},
        {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
    ]

    # The allOf structure is preserved with transformed members.
    assert result == {
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
        ],
    }


def test_allof_with_refs_is_inlined():
    """InlineDefsJsonSchemaTransformer should inline $ref members inside allOf.

    Before the fix, allOf members were never recursed into, so $ref resolution and
    inlining were bypassed for them. This is a unit test pinning the internal walk
    shape because the schema transformer is an internal helper used by providers.
    """

    from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer

    schema = {
        'allOf': [
            {'$ref': '#/$defs/Foo'},
            {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
        ],
        '$defs': {
            'Foo': {'type': 'object', 'properties': {'a': {'type': 'string'}}},
        },
    }

    result = InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()

    # The $ref inside allOf was inlined; no $defs should remain since there are no
    # recursive refs and inlining is preferred.
    assert '$defs' not in result
    assert 'allOf' in result
    assert result['allOf'][0] == {'type': 'object', 'properties': {'a': {'type': 'string'}}}
    assert result['allOf'][1] == {'type': 'object', 'properties': {'b': {'type': 'integer'}}}


def test_typed_schema_anyof_member_is_recursed_google():
    """GoogleJsonSchemaTransformer should strip unsupported keys from anyOf members of a typed node.

    Before the fix, composition members (allOf/anyOf/oneOf) were only recursed when the node
    had no `type`. A typed node (e.g. `type: object`) with a sibling `anyOf` left its
    members untransformed, so provider-specific cleanup (Google strips `title` and
    `exclusiveMinimum`) was never applied to them.
    """
    from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer

    schema = {
        'type': 'object',
        'properties': {'p': {'type': 'string'}},
        'anyOf': [{'type': 'integer', 'title': 'Count', 'exclusiveMinimum': 0}],
    }

    result = GoogleJsonSchemaTransformer(deepcopy(schema)).walk()

    member = result['anyOf'][0]
    assert member['type'] == 'integer'
    assert 'title' not in member
    assert 'exclusiveMinimum' not in member


def test_typed_schema_anyof_member_is_recursed_openai_strict():
    """OpenAIJsonSchemaTransformer strict should add strict fields to anyOf members of a typed node.

    Before the fix, composition members of a typed node were never walked, so OpenAI strict
    mode additions (`additionalProperties: false` and `required`) were missing from them.
    """
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

    schema = {
        'type': 'object',
        'properties': {'p': {'type': 'string'}},
        'anyOf': [{'type': 'object', 'properties': {'q': {'type': 'integer'}}}],
    }

    result = OpenAIJsonSchemaTransformer(deepcopy(schema), strict=True).walk()

    member = result['anyOf'][0]
    assert member['type'] == 'object'
    assert member['additionalProperties'] is False
    assert member['required'] == ['q']


def test_typeless_anyof_member_still_recursed():
    """Control: typeless anyOf members continue to be recursed via _handle_union."""
    from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer

    schema = {
        'anyOf': [{'type': 'integer', 'title': 'Count', 'exclusiveMinimum': 0}],
    }

    result = GoogleJsonSchemaTransformer(deepcopy(schema)).walk()

    # Single-member union collapses into the member, which is still transformed.
    assert result == {'type': 'integer'}


def test_inline_defs_preserves_ref_sibling_keywords():
    """Test internal schema walking, which has no provider request to cover with VCR."""
    schema = {
        'type': 'object',
        'properties': {
            'field': {'$ref': '#/$defs/Foo', 'description': 'field-level description', 'default': None},
        },
        '$defs': {
            'Foo': {
                'type': 'object',
                'description': 'model-level description',
                'default': 'model default',
                'properties': {'x': {'type': 'integer'}},
            }
        },
    }

    result = InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()
    field = result['properties']['field']

    # The referenced definition is inlined...
    assert field['type'] == 'object'
    assert field['properties'] == {'x': {'type': 'integer'}}
    assert '$ref' not in field
    # ...and the sibling keywords are preserved rather than dropped.
    assert field['description'] == 'field-level description'
    assert field['default'] is None

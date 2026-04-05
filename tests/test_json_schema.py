"""Tests for the _json_schema module."""

from __future__ import annotations as _annotations

from copy import deepcopy
from typing import Any

import pytest

from pydantic_ai._json_schema import JsonSchemaTransformer


def test_simplify_nullable_unions():
    """Test the simplify_nullable_unions feature (deprecated, to be removed in v2)."""

    # Create a concrete subclass for testing
    class TestTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    # Test with simplify_nullable_unions=True
    schema_with_null = {
        'anyOf': [
            {'type': 'string'},
            {'type': 'null'},
        ]
    }
    transformer = TestTransformer(schema_with_null, simplify_nullable_unions=True)
    result = transformer.walk()

    # Should collapse to a single nullable string
    assert result == {'type': 'string', 'nullable': True}

    # Test with simplify_nullable_unions=False (default)
    transformer2 = TestTransformer(schema_with_null, simplify_nullable_unions=False)
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
    transformer3 = TestTransformer(schema_no_null, simplify_nullable_unions=True)
    result3 = transformer3.walk()

    # Should keep anyOf since it's not nullable
    assert 'anyOf' in result3
    assert len(result3['anyOf']) == 2


def test_schema_defs_not_modified():
    """Test that the original schema $defs are not modified during transformation."""

    # Create a concrete subclass for testing
    class TestTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

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
    transformer = TestTransformer(original_schema)
    result = transformer.walk()

    # Verify the original schema was not modified
    assert original_schema == original_schema_copy

    # Verify the result is correct
    assert result == original_schema_copy


@pytest.mark.parametrize('value_schema', [True, False])
def test_boolean_schema_nodes_round_trip(value_schema: bool):
    """Boolean JSON Schema nodes should not crash the walker."""

    class TestTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

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

    transformer = TestTransformer(original_schema)

    assert transformer.walk() == original_schema


def test_boolean_schema_in_single_member_union():
    """A union that collapses to a single boolean member should be preserved."""

    class TestTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    schema = {'anyOf': [True]}
    result = TestTransformer(schema).walk()
    assert result == {'anyOf': [True]}


def test_simplify_nullable_union_with_boolean_member():
    """simplify_nullable_unions should not crash when a member is a boolean schema."""

    class TestTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    schema = {'anyOf': [True, {'type': 'null'}]}
    result = TestTransformer(schema, simplify_nullable_unions=True).walk()
    assert result == {'anyOf': [True, {'type': 'null'}]}

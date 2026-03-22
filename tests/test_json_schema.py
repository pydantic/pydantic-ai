"""Tests for the _json_schema module."""

from __future__ import annotations as _annotations

from copy import deepcopy
from typing import Any

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


def test_boolean_schema_nodes():
    """Test that boolean JSON Schema nodes (true/false) are handled correctly.

    In JSON Schema, `true` means 'accept anything' (equivalent to {})
    and `false` means 'accept nothing' (equivalent to {'not': {}}).
    See: https://github.com/pydantic/pydantic-ai/issues/4771
    """

    class PassthroughTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    schema = {
        'type': 'object',
        'properties': {
            'fields': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'value': True,
                        'denied': False,
                    },
                },
            },
        },
    }

    result = PassthroughTransformer(schema).walk()
    # Boolean nodes should be preserved
    assert result['properties']['fields']['items']['properties']['value'] is True
    assert result['properties']['fields']['items']['properties']['denied'] is False

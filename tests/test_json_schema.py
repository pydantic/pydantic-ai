"""Tests for the _json_schema module."""

from __future__ import annotations as _annotations

from typing import Any

from pydantic_ai._json_schema import JsonSchemaTransformer


def test_simplify_nullable_unions() -> None:
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


def test_handle_object_preserves_non_mapping_properties_sections() -> None:
    """Ensures properties/patternProperties that are present but not mappings are left untouched (cov 204-211, 217)."""

    class IdentityTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    schema: dict[str, Any] = {
        'type': 'object',
        'properties': ['unexpected-list-shape'],
        'patternProperties': ['still-not-a-dict'],
        'additionalProperties': True,
    }

    transformer = IdentityTransformer(schema)
    result = transformer.walk()

    assert result['properties'] == ['unexpected-list-shape']
    assert result['patternProperties'] == ['still-not-a-dict']
    assert result['additionalProperties'] is True

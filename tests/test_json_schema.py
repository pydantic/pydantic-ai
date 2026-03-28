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


def test_boolean_in_oneof_union():
    """Test that boolean schemas within oneOf/anyOf unions are handled correctly.

    See: https://github.com/pydantic/pydantic-ai/issues/4771
    """

    class PassthroughTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    # Case 1: oneOf with a single True collapses to True
    schema = {'oneOf': [True]}
    result = PassthroughTransformer(schema).walk()
    assert result is True

    # Case 2: anyOf with a single False collapses to False
    schema = {'anyOf': [False]}
    result = PassthroughTransformer(schema).walk()
    assert result is False

    # Case 3: oneOf with True and a typed schema — should keep union
    schema = {'oneOf': [True, {'type': 'string'}]}
    result = PassthroughTransformer(schema).walk()
    assert 'oneOf' in result
    assert len(result['oneOf']) == 2

    # Case 4: anyOf with False and a typed schema — should keep union
    schema = {'anyOf': [False, {'type': 'number'}]}
    result = PassthroughTransformer(schema).walk()
    assert 'anyOf' in result
    assert len(result['anyOf']) == 2

    # Case 5: nested bool in items inside oneOf
    schema = {
        'oneOf': [
            {'type': 'array', 'items': True},
            {'type': 'string'},
        ]
    }
    result = PassthroughTransformer(schema).walk()
    assert 'oneOf' in result
    assert result['oneOf'][0]['items'] is True


def test_boolean_schemas_in_refs():
    """Test that boolean schemas within $defs/$ref are handled correctly."""

    class PassthroughTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    # $ref pointing to a boolean schema in $defs
    schema = {
        'type': 'object',
        'properties': {
            'anything': {'$ref': '#/$defs/AnyValue'},
            'nothing': {'$ref': '#/$defs/NoValue'},
        },
        '$defs': {
            'AnyValue': True,
            'NoValue': False,
        },
    }

    result = PassthroughTransformer(schema).walk()
    # When prefer_inlined_defs is False, refs should be preserved
    assert '$defs' in result
    assert result['$defs']['AnyValue'] is True
    assert result['$defs']['NoValue'] is False


def test_deeply_nested_boolean_schemas():
    """Test boolean schemas at multiple nesting levels."""

    class PassthroughTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    schema = {
        'type': 'object',
        'properties': {
            'data': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'flexible': True,
                        'strict': {'type': 'string'},
                        'nested_array': {
                            'type': 'array',
                            'items': {
                                'type': 'object',
                                'properties': {
                                    'wildcard': True,
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    result = PassthroughTransformer(schema).walk()
    items = result['properties']['data']['items']
    assert items['properties']['flexible'] is True
    assert items['properties']['strict'] == {'type': 'string'}
    nested = items['properties']['nested_array']['items']
    assert nested['properties']['wildcard'] is True


def test_boolean_with_additional_properties():
    """Test boolean schemas as additionalProperties."""

    class PassthroughTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    # additionalProperties: true (accept any extra keys)
    schema = {
        'type': 'object',
        'properties': {'name': {'type': 'string'}},
        'additionalProperties': True,
    }
    result = PassthroughTransformer(schema).walk()
    assert result['additionalProperties'] is True

    # additionalProperties: false (no extra keys allowed)
    schema['additionalProperties'] = False
    result = PassthroughTransformer(schema).walk()
    assert result['additionalProperties'] is False


def test_boolean_in_prefix_items():
    """Test boolean schemas within prefixItems (tuple validation)."""

    class PassthroughTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            return schema

    schema = {
        'type': 'array',
        'prefixItems': [
            {'type': 'string'},
            True,
            False,
        ],
    }
    result = PassthroughTransformer(schema).walk()
    assert result['prefixItems'][0] == {'type': 'string'}
    assert result['prefixItems'][1] is True
    assert result['prefixItems'][2] is False

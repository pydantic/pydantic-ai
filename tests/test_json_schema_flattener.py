from __future__ import annotations

from typing import Any

from inline_snapshot import snapshot

from pydantic_ai._json_schema import JsonSchema, JsonSchemaTransformer


class FlattenAllofTransformer(JsonSchemaTransformer):
    """Transformer that only flattens allOf, no other transformations."""

    def __init__(self, schema: JsonSchema):
        super().__init__(schema, flatten_allof=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema


def test_flatten_allof_simple_merge() -> None:
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'required': ['a'],
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'integer'}},
                'required': ['b'],
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'a': {'type': 'string'},
                'b': {'type': 'integer'},
            },
            'required': ['a', 'b'],
            'additionalProperties': False,
        }
    )


def test_flatten_allof_nested_objects_and_pass_through_keywords() -> None:
    schema: dict[str, Any] = {
        'type': 'object',
        'title': 'Root',
        'allOf': [
            {
                'type': 'object',
                'properties': {
                    'user': {
                        'type': 'object',
                        'properties': {'id': {'type': 'string'}},
                        'required': ['id'],
                    }
                },
                'required': ['user'],
            },
            {
                'type': 'object',
                'properties': {'age': {'type': 'integer'}},
                'required': ['age'],
            },
        ],
        'description': 'test',
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'title': 'Root',
            'description': 'test',
            'properties': {
                'user': {
                    'type': 'object',
                    'properties': {'id': {'type': 'string'}},
                    'required': ['id'],
                },
                'age': {'type': 'integer'},
            },
            'required': ['age', 'user'],
        }
    )


def test_flatten_allof_does_not_touch_unrelated_unions() -> None:
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'x': {
                'anyOf': [
                    {'type': 'string'},
                    {'type': 'null'},
                ]
            }
        },
        'required': ['x'],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'x': {
                    'anyOf': [
                        {'type': 'string'},
                        {'type': 'null'},
                    ]
                }
            },
            'required': ['x'],
        }
    )


def test_flatten_allof_non_object_members_are_left_as_is() -> None:
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {'type': 'string'},
            {'type': 'integer'},
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Expect: we cannot sensibly merge non-object members; keep allOf
    assert flattened == snapshot(
        {
            'type': 'object',
            'allOf': [
                {'type': 'string'},
                {'type': 'integer'},
            ],
        }
    )


def test_flatten_allof_object_like_without_type() -> None:
    """Test that object-like schemas without explicit type are recognized."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                # No type, but has properties - should be recognized as object-like
                'properties': {'a': {'type': 'string'}},
                'required': ['a'],
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'integer'}},
                'required': ['b'],
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'a': {'type': 'string'},
                'b': {'type': 'integer'},
            },
            'required': ['a', 'b'],
        }
    )


def test_flatten_allof_with_dict_additional_properties() -> None:
    """Test merging when additionalProperties is a dict schema."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'type': 'string'},  # dict schema
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'integer'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'a': {'type': 'string'},
                'b': {'type': 'integer'},
            },
            'additionalProperties': True,  # When any member has dict additionalProperties, result should be True
        }
    )


def test_flatten_allof_with_non_dict_member() -> None:
    """Test that allOf with non-dict members is left untouched."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            'not a dict',  # Non-dict member
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Should be left untouched because one member is not a dict
    assert flattened == snapshot(
        {
            'type': 'object',
            'allOf': [
                {'type': 'object', 'properties': {'a': {'type': 'string'}}},
                'not a dict',
            ],
        }
    )


def test_flatten_allof_no_initial_properties() -> None:
    """Test flattening when root schema has no initial properties."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'required': ['a'],
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'a': {'type': 'string'}},
            'required': ['a'],
        }
    )


def test_flatten_allof_members_without_properties() -> None:
    """Test flattening when some members don't have properties/required/patternProperties."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'required': ['a'],
            },
            {
                'type': 'object',
                # No properties, required, or patternProperties
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'integer'}},
                # No required
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'a': {'type': 'string'},
                'b': {'type': 'integer'},
            },
            'required': ['a'],  # Only from first member
            'additionalProperties': False,
        }
    )


def test_flatten_allof_empty_properties_after_merge() -> None:
    """Test edge case where properties/required/patternProperties might be empty."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                # No properties, required, or patternProperties
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
        }
    )


def test_flatten_allof_with_initial_properties() -> None:
    """Test flattening when root schema has initial properties (line 229)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'root_prop': {'type': 'string'}},  # Initial properties
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'integer'}},
                'required': ['a'],
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'root_prop': {'type': 'string'},
                'a': {'type': 'integer'},
            },
            'required': ['a'],
        }
    )


def test_flatten_allof_with_pattern_properties() -> None:
    """Test flattening when members have patternProperties (lines 241, 250)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'patternProperties': {
                    '^prefix_': {'type': 'string'},
                },
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'integer'}},
                'patternProperties': {
                    '^suffix_': {'type': 'number'},
                },
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'a': {'type': 'string'},
                'b': {'type': 'integer'},
            },
            'patternProperties': {
                '^prefix_': {'type': 'string'},
                '^suffix_': {'type': 'number'},
            },
        }
    )

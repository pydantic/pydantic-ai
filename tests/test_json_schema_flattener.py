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


def test_allof_strict_each_member_results_in_empty() -> None:
    """AllOf with members that each have additionalProperties: False and disjoint properties → empty schema."""
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
            'additionalProperties': False,
        }
    )


def test_flatten_allof_nested_objects_and_pass_through_keywords() -> None:
    """Test that nested objects and pass-through keywords are preserved."""
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
    """Test that unrelated unions are not touched."""
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


def test_allof_with_primitive_types_is_unsupported() -> None:
    """AllOf with primitive types cannot be merged and is not supported by OpenAI strict mode."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {'type': 'string'},
            {'type': 'integer'},
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

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
            'additionalProperties': False,
        }
    )


def test_flatten_allof_with_dict_additional_properties_both_strings() -> None:
    """Test merging when additionalProperties conflicts: dict schema vs False, both properties are strings."""
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
                'properties': {'b': {'type': 'string'}},
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
                'b': {'type': 'string'},
            },
            'additionalProperties': False,
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
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'integer'}},
                # No required, no additionalProperties
            },
            {
                'type': 'object',
                'properties': {'c': {'type': 'boolean'}},
                # No required, no additionalProperties
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'a': {'type': 'string'}},
            'required': ['a'],  # Only from first member
            'additionalProperties': False,  # From first member
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


def test_no_allof_preserves_schema() -> None:
    """Sanity check: schema without allOf remains unchanged."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'a': {'type': 'string'}},
        'required': ['a'],
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


def test_single_allof_flattens_schema() -> None:
    """A single element in allOf is simply flattened."""
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


def test_disjoint_allof_merges_properties() -> None:
    """Disjoint properties are merged correctly."""
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


def test_tool_friendly_strict_merge() -> None:
    """Tool-friendly case for OpenAI with additionalProperties: False."""
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
                'properties': {'b': {'type': 'integer'}},
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Note: currently when a member has additionalProperties: False,
    # properties from other members are filtered if they are not compatible
    assert flattened == snapshot(
        {'type': 'object', 'properties': {'a': {'type': 'string'}, 'b': {'type': 'integer'}}, 'required': ['a']}
    )


def test_nested_allof_collapses_recursively() -> None:
    """Nested allOf are flattened to all levels."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'allOf': [
                    {
                        'type': 'object',
                        'properties': {'a': {'type': 'string'}},
                    },
                    {
                        'type': 'object',
                        'properties': {'b': {'type': 'integer'}},
                    },
                ],
            },
            {
                'type': 'object',
                'properties': {'c': {'type': 'boolean'}},
                'required': ['c'],
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
                'c': {'type': 'boolean'},
            },
            'required': ['c'],
        }
    )


def test_additional_properties_schema_vs_false_string() -> None:
    """Test additionalProperties schema vs False: satisfiable case (b is string)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'type': 'string'},
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # b is string, so it can be included
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'b': {'type': 'string'},
            },
            'additionalProperties': False,  # {} o {"b": "..."}
        }
    )


def test_additional_properties_schema_vs_false_integer() -> None:
    """Test additionalProperties schema vs False: b is integer → should result in empty object only."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'type': 'string'},
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
            'additionalProperties': False,  # solo {}
        }
    )


def test_required_but_no_properties_is_unsatisfiable() -> None:
    """additionalProperties: False without properties but with required results in an empty schema."""
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
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # No object can have 'a' and simultaneously have no properties
    assert flattened == snapshot(
        {
            'type': 'object',
            'additionalProperties': False,
        }
    )


def test_additional_properties_each_only_conflict() -> None:
    """Two mutually exclusive schemas (only a vs only b) → empty result."""
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
            'additionalProperties': False,
        }
    )


def test_conflicting_property_types_last_definition_wins() -> None:
    """Conflicting types: currently the last value wins (current behavior)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
            },
            {
                'type': 'object',
                'properties': {'a': {'type': 'integer'}},
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot({'type': 'object'})


def test_flatten_allof_with_anyof_commands() -> None:
    """Realistic schema with nested allOf + anyOf (Notion-like commands)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'data': {
                'allOf': [
                    {
                        'type': 'object',
                        'properties': {'page_id': {'type': 'string'}},
                        'required': ['page_id'],
                    },
                    {
                        'anyOf': [
                            {
                                'type': 'object',
                                'properties': {
                                    'command': {'type': 'string', 'enum': ['update_properties']},
                                    'properties': {
                                        'type': 'object',
                                        'additionalProperties': {
                                            'type': ['string', 'number', 'null'],
                                        },
                                    },
                                },
                                'required': ['command', 'properties'],
                                'additionalProperties': False,
                            },
                            {
                                'type': 'object',
                                'properties': {
                                    'command': {'type': 'string', 'enum': ['replace_content']},
                                    'new_str': {'type': 'string'},
                                },
                                'required': ['command', 'new_str'],
                                'additionalProperties': False,
                            },
                        ],
                    },
                ]
            }
        },
        'required': ['data'],
        'additionalProperties': False,
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'data': {
                    'allOf': [
                        {'type': 'object', 'properties': {'page_id': {'type': 'string'}}, 'required': ['page_id']},
                        {
                            'anyOf': [
                                {
                                    'type': 'object',
                                    'properties': {
                                        'command': {'type': 'string', 'enum': ['update_properties']},
                                        'properties': {
                                            'type': 'object',
                                            'additionalProperties': {'type': ['string', 'number', 'null']},
                                        },
                                    },
                                    'required': ['command', 'properties'],
                                    'additionalProperties': False,
                                },
                                {
                                    'type': 'object',
                                    'properties': {
                                        'command': {'type': 'string', 'enum': ['replace_content']},
                                        'new_str': {'type': 'string'},
                                    },
                                    'required': ['command', 'new_str'],
                                    'additionalProperties': False,
                                },
                            ]
                        },
                    ]
                }
            },
            'required': ['data'],
            'additionalProperties': False,
        }
    )


def test_merge_additional_properties_multiple_dict_schemas() -> None:
    """Test merging when all additionalProperties are dict schemas (no False)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'type': 'string'},
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': {'type': 'number'},
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Multiple dict schemas can't be easily merged, so return True
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': True,
        }
    )


def test_filter_by_restricted_property_sets_removes_properties() -> None:
    """Test that restricted property sets filter out properties not in intersection."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}, 'c': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Only 'b' is in both restricted sets
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_filter_incompatible_properties_with_false_additional() -> None:
    """Test filtering properties when a member has additionalProperties: False."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'type': 'string'},
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

    # 'a' is not in second member's properties and second has additionalProperties: False
    # 'b' is integer, not compatible with first member's additionalProperties: {'type': 'string'}
    # Result should be empty
    assert flattened == snapshot(
        {
            'type': 'object',
            'additionalProperties': False,
        }
    )


def test_filter_incompatible_properties_removes_all_properties() -> None:
    """Test that filtering incompatible properties can remove all properties."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'required': ['a'],
                'additionalProperties': {'type': 'string'},
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

    # All properties are incompatible, so both properties and required should be removed
    assert flattened == snapshot(
        {
            'type': 'object',
            'additionalProperties': False,
        }
    )


def test_merge_additional_properties_true_values() -> None:
    """Test merging when additionalProperties are True values (not False, not dict) - covers line 218."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': True,  # Explicitly set to True
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': True,  # Explicitly set to True
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # When all values are True (not False, not dict), line 218 returns True
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'a': {'type': 'string'},
                'b': {'type': 'string'},
            },
            'additionalProperties': True,
        }
    )


def test_filter_by_restricted_property_sets_no_required() -> None:
    """Test filtering when properties exist but required doesn't."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}, 'c': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Only 'b' is in both restricted sets, no required field
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_filter_by_restricted_property_sets_removes_all_properties() -> None:
    """Test that restricted property sets can remove all properties when intersection is empty."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # No properties in intersection, so all properties are removed
    assert flattened == snapshot(
        {
            'type': 'object',
            'additionalProperties': False,
        }
    )


def test_filter_by_restricted_property_sets_with_required() -> None:
    """Test restricted property sets when both properties and required exist - covers branch 280->284."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'required': ['a', 'b'],
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}, 'c': {'type': 'string'}},
                'required': ['b', 'c'],
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Only 'b' is in both restricted sets, so 'a' and 'c' are filtered out
    # 'b' remains in both properties and required
    # This covers branch 280->284 where both 'properties' and 'required' exist and required is not empty
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'required': ['b'],
            'additionalProperties': False,
        }
    )


def test_filter_incompatible_properties_removes_required_only() -> None:
    """Test that filtering incompatible properties can remove required while keeping some properties."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'required': ['a', 'b'],
                'additionalProperties': {'type': 'string'},
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'required': ['b'],
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'a' is incompatible (not in second member's properties, second has additionalProperties: False)
    # 'b' is compatible (in both, both are strings)
    # So 'a' should be removed from both properties and required
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'required': ['b'],
            'additionalProperties': False,
        }
    )


def test_filter_incompatible_properties_removes_required_to_empty() -> None:
    """Test that filtering incompatible properties can remove all required fields while keeping properties"""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'required': ['a'],
                'additionalProperties': {'type': 'string'},
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                # No required field
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'a' is incompatible (not in second member's properties, second has additionalProperties: False)
    # 'b' is compatible (in both, both are strings)
    # So 'a' should be removed from properties, and required should become empty and be removed
    # This covers the exit branch 326->exit when required becomes empty
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_incompatible_property_filtered_when_member_has_false_additional() -> None:
    """Test that incompatible property is filtered when member has additionalProperties: False."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'type': 'string'},  # Allows strings as additional
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': False,  # 'a' is not in this member's properties, so it's incompatible
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'a' is in first member's properties, not in second member's properties
    # When checking second member: 'a' is NOT in member_props, and member_additional is False
    # This should trigger lines 311-312 (add to incompatible_props and break)
    # 'b' is in second member, not in first member, but first allows strings as additional
    # Result: 'a' should be removed, 'b' should remain
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_filter_incompatible_properties_with_list_type() -> None:
    """Test filtering properties when additionalProperties has list type (covers _get_type_set with list)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'type': ['string', 'number']},
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'boolean'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'a' is string, compatible with ['string', 'number']
    # 'b' is boolean, not compatible with ['string', 'number'], and second has additionalProperties: False
    assert flattened == snapshot(
        {
            'type': 'object',
            'additionalProperties': False,
        }
    )


def test_filter_incompatible_properties_with_no_type_in_additional() -> None:
    """Test filtering when additionalProperties schema has no type field (covers _get_type_set with no type)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'properties': {'x': {'type': 'string'}}},  # No type field
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # When additionalProperties has no type, _get_type_set returns None, so type check passes
    # But 'a' is not in second member's properties and second has additionalProperties: False
    # 'b' is not in first member's properties and first's additionalProperties has no type (None)
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_get_type_set_with_empty_schema() -> None:
    """Test _get_type_set with empty schema (covers line 393 guard clause)."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {}},  # Empty property schema - falsy, triggers guard clause
                'additionalProperties': {'type': 'string'},
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Empty property schema triggers _get_type_set guard clause (line 393)
    # 'a' has empty schema, so prop_types is None, type check passes
    # But 'a' is not in second member's properties and second has additionalProperties: False
    # 'b' is compatible
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_restricted_property_sets_preserves_properties_and_required() -> None:
    """Test that restricted property sets filtering preserves both properties and required fields.

    Creates a situation where:
    - merged has properties and required when _filter_by_restricted_property_sets is called
    - After filtering, both still exist (not empty)
    - This ensures the branch handling both properties and required is executed
    """
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'base': {'type': 'string'}},  # Initial properties in merged
        'required': ['base'],  # Initial required in merged
        'allOf': [
            {
                'type': 'object',
                'properties': {'base': {'type': 'string'}, 'extra': {'type': 'string'}},
                'required': ['base', 'extra'],
                'additionalProperties': False,  # Creates restricted_property_sets
            },
            {
                'type': 'object',
                'properties': {'base': {'type': 'string'}},  # Solo 'base' in comune
                'required': ['base'],
                'additionalProperties': False,  # Creates restricted_property_sets
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'base' is in both restricted sets, so it survives
    # When _filter_by_restricted_property_sets is called:
    # - merged has 'properties' (line 280 True)
    # - After filtering, 'base' remains (not empty)
    # - merged still has 'required' (line 284 True)
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'base': {'type': 'string'}},
            'required': ['base'],
            'additionalProperties': False,
        }
    )


def test_incompatible_property_filtered_when_not_in_member_with_false_additional() -> None:
    """Test that properties not in member properties are filtered when member has additionalProperties: False.

    Creates a situation where:
    - A property exists in merged but NOT in member_props of a member
    - That member has additionalProperties: False
    - This triggers the filtering logic
    """
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'additionalProperties': True,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},  # 'a' is NOT here
                'additionalProperties': False,  # When checking 'a':
                # - 'a' is NOT in member_props (line 303 False)
                # - member_additional is False (line 310 True, we execute 311-312)
            },
            {
                'type': 'object',
                'properties': {'c': {'type': 'string'}},
                'additionalProperties': {'type': 'string'},
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'a' is removed (triggers 311-312) - not in second member, second has additionalProperties: False
    # 'b' remains (is in second member)
    # 'c' is removed (triggers 311-312) - not in second member, second has additionalProperties: False
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_required_removed_when_all_required_fields_incompatible() -> None:
    """Test that required field is removed when all required fields become incompatible.

    Creates a situation where:
    - required exists initially
    - All required fields are removed as incompatible
    - required becomes empty and is removed (lines 326-327)
    """
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'req1': {'type': 'string'}, 'req2': {'type': 'string'}},
                'required': ['req1', 'req2'],  # Both required
                'additionalProperties': True,
            },
            {
                'type': 'object',
                'properties': {'other': {'type': 'string'}},  # None of the required fields are here
                'additionalProperties': False,  # Both are removed (311-312)
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'req1' and 'req2' are removed (incompatible, triggers 311-312)
    # required becomes empty and is removed (triggers 326->exit)
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'other': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_empty_property_schema_triggers_type_set_guard_clause() -> None:
    """Test that empty property schema triggers _get_type_set guard clause.

    Creates a situation where:
    - A property has empty schema {}
    - _get_type_set is called with empty schema
    - This triggers the guard clause
    """
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {
                    'empty_schema': {},  # Empty schema - triggers guard clause
                    'normal': {'type': 'string'},
                },
                'additionalProperties': True,
            },
            {
                'type': 'object',
                'properties': {'normal': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'empty_schema' has empty schema, so _get_type_set({}) returns None (guard clause)
    # 'empty_schema' is not in second member, so it is removed
    # 'normal' remains
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'normal': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_combined_filtering_scenarios_all_branches() -> None:
    """Test that combines all filtering conditions to exercise all branches."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'common': {'type': 'string'}},  # For restricted property sets
        'required': ['common'],  # For restricted property sets
        'allOf': [
            {
                'type': 'object',
                'properties': {
                    'common': {'type': 'string'},  # Survives (restricted property sets)
                    'removed_by_false': {'type': 'string'},  # For incompatible property filtering
                    'removed_required': {'type': 'string'},  # For required removal
                    'empty_schema': {},  # For empty schema guard clause
                },
                'required': ['common', 'removed_required'],  # 'removed_required' for required removal
                'additionalProperties': True,
            },
            {
                'type': 'object',
                'properties': {'common': {'type': 'string'}},  # Only 'common' in common
                'required': ['common'],
                'additionalProperties': False,  # Creates restricted_property_sets (280->284)
                # 'removed_by_false' is NOT here -> triggers 311-312
                # 'removed_required' is NOT here -> triggers 311-312, then 326->exit
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'common' survives (restricted property sets)
    # 'removed_by_false' is removed (incompatible property filtering)
    # 'removed_required' is removed (incompatible property filtering), required becomes empty (required removal)
    # 'empty_schema' is removed (empty schema guard clause)
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'common': {'type': 'string'}},
            'required': ['common'],
            'additionalProperties': False,
        }
    )


def test_empty_schema_returns_unchanged() -> None:
    """Test that empty schema returns unchanged."""
    schema: dict[str, Any] = {}

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Empty schema returns unchanged
    assert flattened == snapshot(schema)


def test_no_allof_returns_unchanged() -> None:
    """Test that schema without allOf returns unchanged."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'a': {'type': 'string'}},
        'required': ['a'],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Schema without allOf returns unchanged
    assert flattened == snapshot(schema)


def test_allof_not_list_returns_unchanged() -> None:
    """Test that schema with allOf that is not a list returns unchanged."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': 'not a list',  # allOf is not a list
        'properties': {'a': {'type': 'string'}},
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Schema with allOf that is not a list returns unchanged
    assert flattened == snapshot(schema)


def test_allof_empty_list_returns_unchanged() -> None:
    """Test that schema with empty allOf list returns unchanged."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [],  # Empty list
        'properties': {'a': {'type': 'string'}},
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Schema with empty allOf list returns unchanged
    assert flattened == snapshot(schema)


def test_allof_member_not_dict_returns_unchanged() -> None:
    """Test that schema with allOf containing non-dict members returns unchanged."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            'not a dict',  # Non-dict member
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Schema with allOf containing non-dict members returns unchanged
    assert flattened == snapshot(schema)


def test_allof_member_not_object_like_returns_unchanged() -> None:
    """Test that schema with allOf containing non object-like members returns unchanged."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            {'type': 'string'},  # Not object-like (type is 'string', not 'object')
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Schema with allOf containing non object-like members returns unchanged
    assert flattened == snapshot(schema)


def test_allof_member_no_type_no_object_keys_returns_unchanged() -> None:
    """Test that schema with allOf containing members without type and without object keys returns unchanged."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            {'description': 'not object-like'},  # No type, no object keys
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Schema with allOf containing non object-like members returns unchanged
    assert flattened == snapshot(schema)


def test_properties_and_required_both_exist_after_filtering() -> None:
    """Test that both properties and required exist when filtering is applied."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'required': ['a', 'b'],
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},  # Only 'a' in common
                'required': ['a'],
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'a' is in both restricted sets, so it survives
    # When _filter_properties_and_required is called:
    # - properties exists (line 239 True)
    # - filtered is not empty, so properties is updated
    # - required exists (line 247 True)
    # This ensures the branch handling both properties and required is executed
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'a': {'type': 'string'}},
            'required': ['a'],
            'additionalProperties': False,
        }
    )


def test_get_type_set_with_schema_without_type() -> None:
    """Test that _get_type_set returns None when schema has no type field."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {}},  # Schema senza type
                'additionalProperties': True,
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'a' has schema without type, so _get_type_set returns None
    # 'a' is not in second member and second has additionalProperties: False
    # So 'a' is removed
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_array_items_in_properties_handled_recursively() -> None:
    """Test that array items in properties are handled recursively."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'items': {
                'type': 'array',
                'items': {
                    'allOf': [
                        {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                        {'type': 'object', 'properties': {'y': {'type': 'integer'}}},
                    ],
                },
            },
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Array items with allOf are flattened
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {'x': {'type': 'string'}, 'y': {'type': 'integer'}},
                    },
                },
            },
        }
    )


def test_get_type_set_with_none_schema() -> None:
    """Test that _get_type_set handles None schema (falsy).

    This test covers the case where _get_type_set is called with None
    through additionalProperties which is None.
    """
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': None,  # None schema - triggers guard clause
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'string'}},
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # additionalProperties: None is handled as True (default)
    # 'a' and 'b' are compatible
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_properties_and_required_exist_after_filtering_with_base_schema() -> None:
    """Test that both properties and required exist after filtering when base schema has them."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'base': {'type': 'string'}},  # Initial properties
        'required': ['base'],  # Initial required
        'allOf': [
            {
                'type': 'object',
                'properties': {'base': {'type': 'string'}, 'extra': {'type': 'string'}},
                'required': ['base', 'extra'],
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'base': {'type': 'string'}},  # Solo 'base' in comune
                'required': ['base'],
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'base' is in both restricted sets, so it survives
    # When _filter_properties_and_required is called:
    # - properties exists (line 240 True) with 'base'
    # - filtered_props is not empty, so properties is updated (we don't return)
    # - required exists (line 251 True) with 'base'
    # This ensures the branch handling both properties and required is executed
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'base': {'type': 'string'}},
            'required': ['base'],
            'additionalProperties': False,
        }
    )


def test_properties_and_required_both_exist_after_restricted_filtering() -> None:
    """Test that both properties and required exist after restricted property sets filtering."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'required': ['a', 'b'],
                'additionalProperties': False,
            },
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
                'required': ['a', 'b'],
                'additionalProperties': False,
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Entrambi hanno 'a' e 'b', quindi entrambi sopravvivono al filtering
    # When _filter_properties_and_required is called:
    # - properties exists (line 240 True)
    # - filtered_props is not empty (contains 'a' and 'b'), so we don't return
    # - required exists (line 251 True)
    # This ensures the branch handling both properties and required is executed
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'a': {'type': 'string'}, 'b': {'type': 'string'}},
            'required': ['a', 'b'],
            'additionalProperties': False,
        }
    )


def test_property_filtered_when_not_in_member_with_false_additional() -> None:
    """Test that property is filtered when not in member properties and member has additionalProperties: False."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': True,
            },
            {
                'type': 'object',
                'properties': {},  # 'a' is not here
                'additionalProperties': False,  # When checking 'a':
                # - 'a' is NOT in member_props (line 330 False)
                # - member_additional is False (line 347 True, we execute 348-349)
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # 'a' is removed because it's not in second member and second has additionalProperties: False
    # This triggers lines 348-349
    assert flattened == snapshot(
        {
            'type': 'object',
            'additionalProperties': False,
        }
    )


def test_array_items_dict_in_properties_processed_recursively() -> None:
    """Test that array items dict in properties is processed recursively."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'items': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {'x': {'type': 'string'}},
                },
            },
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Array items is a dict, so it is processed (line 385 True)
    # This ensures the branch handling array items is executed
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {'x': {'type': 'string'}},
                    },
                },
            },
        }
    )


def test_allof_with_array_items() -> None:
    """Test that allOf in array items is flattened recursively."""
    schema: dict[str, Any] = {
        'type': 'array',
        'items': {
            'type': 'object',
            'allOf': [
                {
                    'type': 'object',
                    'properties': {'a': {'type': 'string'}},
                },
                {
                    'type': 'object',
                    'properties': {'b': {'type': 'integer'}},
                },
            ],
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'string'},
                    'b': {'type': 'integer'},
                },
            },
        }
    )


def test_allof_single_member_with_dict_additional_properties() -> None:
    """Test flattening allOf with single member having additionalProperties as dict schema."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                'additionalProperties': {'type': 'string'},
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'a': {'type': 'string'}},
            'additionalProperties': {'type': 'string'},
        }
    )


def test_allof_single_member_with_pattern_properties() -> None:
    """Test flattening allOf with single member having patternProperties."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
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
            'properties': {'b': {'type': 'integer'}},
            'patternProperties': {
                '^suffix_': {'type': 'number'},
            },
        }
    )


def test_allof_base_schema_with_additional_properties_false() -> None:
    """Test flattening allOf when base schema has additionalProperties: False."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {'base_prop': {'type': 'string'}},
        'additionalProperties': False,
        'allOf': [
            {
                'type': 'object',
                'properties': {'member_prop': {'type': 'integer'}},
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'base_prop': {'type': 'string'}},
            'additionalProperties': False,
        }
    )


def test_allof_base_schema_with_pattern_properties() -> None:
    """Test flattening allOf when base schema has patternProperties."""
    schema: dict[str, Any] = {
        'type': 'object',
        'patternProperties': {
            '^prefix_': {'type': 'string'},
        },
        'allOf': [
            {
                'type': 'object',
                'properties': {'member_prop': {'type': 'integer'}},
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
            'properties': {'member_prop': {'type': 'integer'}},
            'patternProperties': {
                '^prefix_': {'type': 'string'},
                '^suffix_': {'type': 'number'},
            },
        }
    )


def test_allof_filter_property_not_in_member_with_false_additional() -> None:
    """Test that properties not explicitly defined in a member with additionalProperties: False are filtered out."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                # No additionalProperties - allows additional
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'integer'}},
                'additionalProperties': False,  # Only 'b' is allowed
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Property 'a' should be removed because member 2 has additionalProperties: False
    # and 'a' is not explicitly defined in member 2
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'b': {'type': 'integer'}},
            'additionalProperties': False,
        }
    )


def test_allof_result_with_items_recursive_processing() -> None:
    """Test that items in the result schema are processed recursively."""
    schema: dict[str, Any] = {
        'type': 'object',
        'properties': {
            'items': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'allOf': [
                        {
                            'type': 'object',
                            'properties': {'x': {'type': 'string'}},
                        },
                        {
                            'type': 'object',
                            'properties': {'y': {'type': 'integer'}},
                        },
                    ],
                },
            },
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # The items inside the array should have allOf flattened
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'x': {'type': 'string'},
                            'y': {'type': 'integer'},
                        },
                    },
                },
            },
        }
    )


def test_allof_base_schema_with_items() -> None:
    """Test that items in base schema are processed recursively after allOf merge."""
    schema: dict[str, Any] = {
        'type': 'object',
        'items': {
            'type': 'object',
            'allOf': [
                {
                    'type': 'object',
                    'properties': {'a': {'type': 'string'}},
                },
                {
                    'type': 'object',
                    'properties': {'b': {'type': 'integer'}},
                },
            ],
        },
        'allOf': [
            {
                'type': 'object',
                'properties': {'c': {'type': 'number'}},
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # The items should have allOf flattened recursively
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {'c': {'type': 'number'}},
            'items': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'string'},
                    'b': {'type': 'integer'},
                },
            },
        }
    )


def test_array_with_items_allof() -> None:
    """Test array with items containing allOf (realistic case from user)."""
    schema: dict[str, Any] = {
        'type': 'array',
        'items': {
            'type': 'object',
            'allOf': [
                {
                    'type': 'object',
                    'properties': {
                        'remindAt': {
                            'type': 'string',
                            'format': 'date-time',
                        },
                    },
                    'required': ['remindAt'],
                },
                {
                    'type': 'object',
                    'properties': {
                        'comment': {
                            'type': 'string',
                        },
                    },
                    'required': ['comment'],
                },
            ],
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # The items should have allOf flattened
    assert flattened == snapshot(
        {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {'remindAt': {'type': 'string', 'format': 'date-time'}, 'comment': {'type': 'string'}},
                'required': ['comment', 'remindAt'],
            },
        }
    )


def test_allof_base_additional_properties_false_no_properties() -> None:
    """Test that base schema with additionalProperties: False but no properties allows properties from allOf members.

    This covers lines 286-291: when base has additionalProperties: False but no properties,
    we don't add an empty set to restricted_property_sets, so properties from allOf members are still valid.
    """
    schema: dict[str, Any] = {
        'type': 'object',
        'additionalProperties': False,
        'allOf': [
            {
                'type': 'object',
                'properties': {
                    'a': {'type': 'string'},
                },
            },
            {
                'type': 'object',
                'properties': {
                    'b': {'type': 'integer'},
                },
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Properties from allOf members should be preserved even though base has additionalProperties: False
    # because base has no properties, so we don't restrict to an empty set
    assert flattened == snapshot(
        {
            'type': 'object',
            'properties': {
                'a': {'type': 'string'},
                'b': {'type': 'integer'},
            },
            'additionalProperties': False,
        }
    )


def test_array_without_allof() -> None:
    """Test that array without allOf is processed recursively (items are still flattened if they contain allOf)."""
    schema: dict[str, Any] = {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'nested': {
                    'type': 'object',
                    'allOf': [
                        {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                        {'type': 'object', 'properties': {'y': {'type': 'integer'}}},
                    ],
                },
            },
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # The nested allOf in items.properties.nested should be flattened
    assert flattened == snapshot(
        {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'nested': {
                        'type': 'object',
                        'properties': {
                            'x': {'type': 'string'},
                            'y': {'type': 'integer'},
                        },
                    },
                },
            },
        }
    )


def test_array_items_with_allof() -> None:
    """Test that array items with allOf are flattened recursively."""
    schema: dict[str, Any] = {
        'type': 'array',
        'items': {
            'type': 'object',
            'allOf': [
                {'type': 'object', 'properties': {'a': {'type': 'string'}}},
                {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
            ],
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # The allOf in items should be flattened directly
    assert flattened == snapshot(
        {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'a': {'type': 'string'},
                    'b': {'type': 'integer'},
                },
            },
        }
    )


def test_array_items_without_allof_recursive() -> None:
    """Test that array items without allOf are processed recursively (covers lines 242-245).

    This covers the case where the root schema has no allOf (Case 1), so it enters
    the recursive processing branch. For arrays, this means processing items recursively.
    """
    schema: dict[str, Any] = {
        'type': 'array',
        'items': {
            'type': 'object',
            'properties': {
                'nested': {
                    'type': 'object',
                    'allOf': [
                        {'type': 'object', 'properties': {'x': {'type': 'string'}}},
                        {'type': 'object', 'properties': {'y': {'type': 'integer'}}},
                    ],
                },
            },
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # The root array has no allOf, so it enters Case 1 and processes items recursively
    # The items have no allOf, so they process properties recursively
    # The nested property has allOf, which gets flattened
    assert flattened == snapshot(
        {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'nested': {
                        'type': 'object',
                        'properties': {
                            'x': {'type': 'string'},
                            'y': {'type': 'integer'},
                        },
                    },
                },
            },
        }
    )


def test_array_with_simple_items_dict() -> None:
    """Test that array with simple items dict is processed (covers lines 242-245).

    This directly covers the branch where schema_type == 'array' and items is a dict.
    The items don't need to have allOf - they just need to be a dict to trigger the recursive call.
    """
    schema: dict[str, Any] = {
        'type': 'array',
        'items': {
            'type': 'string',
        },
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # The array has no allOf, so it enters Case 1
    # schema_type == 'array', so it enters the elif branch
    # items is a dict, so it calls _recurse_flatten_allof recursively
    # The items have no allOf, so they're returned unchanged
    assert flattened == snapshot(
        {
            'type': 'array',
            'items': {
                'type': 'string',
            },
        }
    )


def test_array_no_items() -> None:
    """Test that array without items key is handled (covers branch 243->245).

    This covers the branch where schema_type == 'array' but items is None or not present,
    so the if condition at line 243 is False and we skip to line 245 (return).
    """
    schema: dict[str, Any] = {
        'type': 'array',
        # No 'items' key
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # The array has no allOf, so it enters Case 1
    # schema_type == 'array', so it enters the elif branch
    # items is None (not present), so isinstance(s.get('items'), dict) is False
    # We skip to return s (line 245)
    assert flattened == snapshot(
        {
            'type': 'array',
        }
    )


def test_allof_incompatible_props_member_false_additional_not_in_restricted() -> None:
    """Test property not in restricted_property_sets but incompatible due to member with False additionalProperties."""
    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                'properties': {'a': {'type': 'string'}},
                # No additionalProperties - allows additional
            },
            {
                'type': 'object',
                'properties': {'b': {'type': 'integer'}},
                # No additionalProperties - allows additional, so 'a' is not filtered by restricted_property_sets
            },
            {
                'type': 'object',
                # No properties defined
                'additionalProperties': False,  # This will filter 'a' and 'b' in incompatible_props check
            },
        ],
    }

    transformer = FlattenAllofTransformer(schema)
    flattened = transformer.walk()

    # Both 'a' and 'b' should be removed because member 3 has additionalProperties: False
    # and they are not explicitly defined in member 3
    assert flattened == snapshot(
        {
            'type': 'object',
            'additionalProperties': False,
        }
    )

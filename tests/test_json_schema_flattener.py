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

    # Nota: attualmente quando un membro ha additionalProperties: False,
    # le proprietà degli altri membri vengono filtrate se non sono compatibili
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
    """additionalProperties schema vs False: caso soddisfacibile (b è string)."""
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

    # b è string, quindi può essere incluso
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
    """additionalProperties schema vs False: b integer → dovrebbe restare solo {}."""
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

    # Nessun oggetto può avere 'a' e contemporaneamente nessuna proprietà
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
    """Test that filtering incompatible properties can remove all required fields while keeping properties."""
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

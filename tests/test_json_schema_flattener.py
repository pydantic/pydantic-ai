from __future__ import annotations

import copy
from typing import Any


def test_flatten_allof_simple_merge() -> None:
    # Import inline to avoid import errors before implementation exists in editors
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))

    assert 'allOf' not in flattened
    assert flattened['type'] == 'object'
    assert flattened['properties']['a']['type'] == 'string'
    assert flattened['properties']['b']['type'] == 'integer'
    # union of required keys
    assert set(flattened['required']) == {'a', 'b'}
    # boolean AP should remain False when all are False
    assert flattened.get('additionalProperties') is False


def test_flatten_allof_nested_objects_and_pass_through_keywords() -> None:
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))
    assert flattened.get('title') == 'Root'
    assert flattened.get('description') == 'test'
    assert 'allOf' not in flattened
    assert set(flattened['required']) == {'user', 'age'}
    assert flattened['properties']['user']['type'] == 'object'
    assert set(flattened['properties']['user']['required']) == {'id'}


def test_flatten_allof_does_not_touch_unrelated_unions() -> None:
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))
    assert flattened['properties']['x'].get('anyOf') is not None


def test_flatten_allof_non_object_members_are_left_as_is() -> None:
    from pydantic_ai._json_schema import flatten_allof

    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {'type': 'string'},
            {'type': 'integer'},
        ],
    }

    # Expect: we cannot sensibly merge non-object members; keep allOf
    flattened = flatten_allof(copy.deepcopy(schema))
    assert 'allOf' in flattened


def test_flatten_allof_object_like_without_type() -> None:
    """Test that object-like schemas without explicit type are recognized."""
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))
    assert 'allOf' not in flattened
    assert set(flattened['required']) == {'a', 'b'}


def test_flatten_allof_with_dict_additional_properties() -> None:
    """Test merging when additionalProperties is a dict schema."""
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))
    assert 'allOf' not in flattened
    # When any member has dict additionalProperties, result should be True
    assert flattened.get('additionalProperties') is True


def test_flatten_allof_with_non_dict_member() -> None:
    """Test that allOf with non-dict members is left untouched."""
    from pydantic_ai._json_schema import flatten_allof

    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            'not a dict',  # Non-dict member
        ],
    }

    flattened = flatten_allof(copy.deepcopy(schema))
    # Should be left untouched because one member is not a dict
    assert 'allOf' in flattened


def test_flatten_allof_no_initial_properties() -> None:
    """Test flattening when root schema has no initial properties."""
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))
    assert 'allOf' not in flattened
    assert flattened['properties']['a']['type'] == 'string'
    assert flattened['required'] == ['a']


def test_flatten_allof_members_without_properties() -> None:
    """Test flattening when some members don't have properties/required/patternProperties."""
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))
    assert 'allOf' not in flattened
    assert set(flattened['properties'].keys()) == {'a', 'b'}
    assert flattened['required'] == ['a']  # Only from first member
    assert flattened.get('additionalProperties') is False


def test_flatten_allof_empty_properties_after_merge() -> None:
    """Test edge case where properties/required/patternProperties might be empty."""
    from pydantic_ai._json_schema import flatten_allof

    schema: dict[str, Any] = {
        'type': 'object',
        'allOf': [
            {
                'type': 'object',
                # No properties, required, or patternProperties
            },
        ],
    }

    flattened = flatten_allof(copy.deepcopy(schema))
    assert 'allOf' not in flattened
    # Should not have properties/required if they're empty
    assert 'properties' not in flattened or not flattened.get('properties')
    assert 'required' not in flattened or not flattened.get('required')


def test_flatten_allof_with_initial_properties() -> None:
    """Test flattening when root schema has initial properties (line 229)."""
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))
    assert 'allOf' not in flattened
    assert 'root_prop' in flattened['properties']
    assert 'a' in flattened['properties']
    assert flattened['required'] == ['a']


def test_flatten_allof_with_pattern_properties() -> None:
    """Test flattening when members have patternProperties (lines 241, 250)."""
    from pydantic_ai._json_schema import flatten_allof

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

    flattened = flatten_allof(copy.deepcopy(schema))
    assert 'allOf' not in flattened
    assert 'patternProperties' in flattened
    assert '^prefix_' in flattened['patternProperties']
    assert '^suffix_' in flattened['patternProperties']
    assert flattened['patternProperties']['^prefix_']['type'] == 'string'
    assert flattened['patternProperties']['^suffix_']['type'] == 'number'

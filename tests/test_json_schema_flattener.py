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

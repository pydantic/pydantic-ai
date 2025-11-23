"""Tests for Anthropic JSON schema transformer and strict compatibility detection.

The AnthropicJsonSchemaTransformer checks whether schemas are 'lossless' - meaning
Anthropic's SDK won't drop validation constraints during transformation to their
structured output format.

When constraints would be dropped (making the schema 'lossy'), `is_strict_compatible`
is set to False. This prevents automatic use of strict mode, which would make
server-side validation impossible since the constraints wouldn't be enforced.

Key concepts:
- **Lossless**: Schema constraints are fully preserved by Anthropic's transformer
- **Lossy**: SDK drops constraints (e.g., minLength, pattern, minItems > 1)
- **Strict compatible**: Schema can safely use strict=True for guaranteed validation

See: https://docs.claude.com/en/docs/build-with-claude/structured-outputs
"""

from __future__ import annotations as _annotations

from typing import Annotated

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.anthropic import AnthropicJsonSchemaTransformer

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


def test_show_lossless_transform():
    """Shows that a simple model without constraints is lossless."""

    class Person(BaseModel):
        name: str
        age: int

    strict = None
    transformer = AnthropicJsonSchemaTransformer(Person.model_json_schema(), strict=strict)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'additionalProperties': False,
            'required': ['name', 'age'],
        }
    )


def test_show_lossy_transform():
    """Shows that a model with validation constraints is detected as lossy."""

    class Person(BaseModel):
        name: str = Field(min_length=3)
        age: int

    original_schema = Person.model_json_schema()
    strict = True
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=strict)
    transformed = transformer.walk()
    assert original_schema == snapshot(
        {
            'properties': {
                'name': {'minLength': 3, 'title': 'Name', 'type': 'string'},
                'age': {'title': 'Age', 'type': 'integer'},
            },
            'required': ['name', 'age'],
            'title': 'Person',
            'type': 'object',
        }
    )
    # it's not strict compatible but we forced strict=True
    assert transformer.is_strict_compatible is False
    # anthropic's transform_schema shoves constraints into description
    assert transformed == snapshot(
        {
            'properties': {'name': {'type': 'string', 'description': '{minLength: 3}'}, 'age': {'type': 'integer'}},
            'required': ['name', 'age'],
            'additionalProperties': False,
            'type': 'object',
        }
    )


def test_lossless_nested_model():
    """Nested models without constraints are lossless."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    original_schema = Person.model_json_schema()
    assert original_schema == snapshot(
        {
            '$defs': {
                'Address': {
                    'type': 'object',
                    'properties': {
                        'street': {'title': 'Street', 'type': 'string'},
                        'city': {'title': 'City', 'type': 'string'},
                    },
                    'required': ['street', 'city'],
                    'title': 'Address',
                }
            },
            'properties': {'name': {'title': 'Name', 'type': 'string'}, 'address': {'$ref': '#/$defs/Address'}},
            'required': ['name', 'address'],
            'title': 'Person',
            'type': 'object',
        }
    )
    # strict=True forces transformation
    strict = True
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=strict)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            '$defs': {
                'Address': {
                    'type': 'object',
                    'properties': {'street': {'type': 'string'}, 'city': {'type': 'string'}},
                    'additionalProperties': False,
                    'required': ['street', 'city'],
                }
            },
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'address': {'$ref': '#/$defs/Address'}},
            'additionalProperties': False,
            'required': ['name', 'address'],
        }
    )


def test_lossy_string_constraints():
    """String with min_length constraint are lossy."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]

    original_schema = User.model_json_schema()
    strict = None
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=strict)
    transformed = transformer.walk()

    # SDK drops minLength, making it lossy
    assert transformer.is_strict_compatible is False

    # Original schema has minLength constraint
    assert original_schema == snapshot(
        {
            'properties': {'username': {'minLength': 3, 'title': 'Username', 'type': 'string'}},
            'required': ['username'],
            'title': 'User',
            'type': 'object',
        }
    )

    # Transformed schema has constraint dropped and moved to description
    assert transformed == snapshot(
        {'type': 'object', 'properties': {'username': {'minLength': 3, 'type': 'string'}}, 'required': ['username']}
    )


def test_lossy_number_constraints():
    """Number with minimum constraint should be lossy (constraint gets dropped)."""

    class Product(BaseModel):
        price: Annotated[float, Field(ge=0)]

    strict = None
    transformer = AnthropicJsonSchemaTransformer(Product.model_json_schema(), strict=strict)
    transformed = transformer.walk()

    # SDK drops minimum, making it lossy
    assert transformer.is_strict_compatible is False
    # Transformed schema has constraint dropped and moved to description
    assert transformed == snapshot(
        {'type': 'object', 'properties': {'price': {'minimum': 0, 'type': 'number'}}, 'required': ['price']}
    )


def test_lossy_pattern_constraint():
    """String with pattern constraint should be lossy (constraint gets dropped)."""

    class Email(BaseModel):
        address: Annotated[str, Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')]

    original_schema = Email.model_json_schema()
    strict = None
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=strict)
    transformed = transformer.walk()

    # SDK drops pattern, making it lossy
    assert transformer.is_strict_compatible is False

    # Original schema has pattern constraint
    assert original_schema == snapshot(
        {
            'properties': {
                'address': {'pattern': '^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', 'title': 'Address', 'type': 'string'}
            },
            'required': ['address'],
            'title': 'Email',
            'type': 'object',
        }
    )

    # Transformed schema has constraint dropped and moved to description
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {'address': {'pattern': '^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', 'type': 'string'}},
            'required': ['address'],
        }
    )


def test_strict_false_no_transformation():
    """When strict=False, no transformation is applied."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]

    strict = False
    transformer = AnthropicJsonSchemaTransformer(User.model_json_schema(), strict=strict)
    transformed = transformer.walk()

    # `'minLength': 3` proves no transformation occurred
    assert transformed == snapshot(
        {'type': 'object', 'properties': {'username': {'minLength': 3, 'type': 'string'}}, 'required': ['username']}
    )


def test_lossy_array_items_with_constraints():
    """Detect lossy changes in array items with inline validation constraints."""

    class StringList(BaseModel):
        items: list[Annotated[str, Field(min_length=1)]]

    original_schema = StringList.model_json_schema()
    strict = None
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=strict)
    transformed = transformer.walk()

    # Array items have inline minLength constraint which gets dropped
    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            # `'minLength': 1` proves no transformation occurred
            'properties': {'items': {'type': 'array', 'items': {'minLength': 1, 'type': 'string'}}},
            'required': ['items'],
        }
    )


def test_lossy_schema_with_defs():
    """Detect lossy changes in schemas using $defs with validation constraints."""

    class UserProfile(BaseModel):
        name: Annotated[str, Field(min_length=3)]
        age: int

    class Account(BaseModel):
        profile: UserProfile
        backup_profile: UserProfile | None = None

    original_schema = Account.model_json_schema()
    strict = None
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=strict)
    transformer.walk()

    # UserProfile is in $defs with minLength constraint which gets dropped
    assert transformer.is_strict_compatible is False

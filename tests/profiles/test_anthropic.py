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


def test_lossless_simple_model():
    """Simple BaseModel with basic types should be lossless."""

    class Person(BaseModel):
        name: str
        age: int

    transformer = AnthropicJsonSchemaTransformer(Person.model_json_schema(), strict=True)
    transformer.walk()

    assert transformer.is_strict_compatible is True


def test_lossless_nested_model():
    """Nested BaseModels should be lossless."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    transformer = AnthropicJsonSchemaTransformer(Person.model_json_schema(), strict=True)
    transformer.walk()

    assert transformer.is_strict_compatible is True


def test_lossy_string_constraints():
    """String with min_length constraint should be lossy (constraint gets dropped)."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]

    original_schema = User.model_json_schema()
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=True)
    result = transformer.walk()

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
    assert result == snapshot(
        {
            'type': 'object',
            'properties': {'username': {'type': 'string', 'description': '{minLength: 3}'}},
            'required': ['username'],
            'additionalProperties': False,
        }
    )


def test_lossy_number_constraints():
    """Number with minimum constraint should be lossy (constraint gets dropped)."""

    class Product(BaseModel):
        price: Annotated[float, Field(ge=0)]

    transformer = AnthropicJsonSchemaTransformer(Product.model_json_schema(), strict=True)
    result = transformer.walk()

    # SDK drops minimum, making it lossy
    assert transformer.is_strict_compatible is False
    # Transformed schema has constraint dropped and moved to description
    assert result == snapshot(
        {
            'type': 'object',
            'properties': {'price': {'type': 'number', 'description': '{minimum: 0}'}},
            'required': ['price'],
            'additionalProperties': False,
        }
    )


def test_lossy_pattern_constraint():
    """String with pattern constraint should be lossy (constraint gets dropped)."""

    class Email(BaseModel):
        address: Annotated[str, Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')]

    original_schema = Email.model_json_schema()
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=True)
    result = transformer.walk()

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
    assert result == snapshot(
        {
            'type': 'object',
            'properties': {'address': {'type': 'string', 'description': '{pattern: ^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$}'}},
            'required': ['address'],
            'additionalProperties': False,
        }
    )


def test_transformer_output():
    """Verify transformer produces expected output for a simple model."""

    class SimpleModel(BaseModel):
        name: str
        count: int

    transformer = AnthropicJsonSchemaTransformer(SimpleModel.model_json_schema(), strict=True)
    result = transformer.walk()

    assert result == snapshot(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'count': {'type': 'integer'}},
            'required': ['name', 'count'],
            'additionalProperties': False,
        }
    )


def test_lossy_array_with_constrained_items():
    """Array with lossy item schema should be lossy."""

    class Container(BaseModel):
        items: list[Annotated[str, Field(min_length=5)]]

    transformer = AnthropicJsonSchemaTransformer(Container.model_json_schema(), strict=True)
    transformer.walk()

    # Array items with constraints are lossy
    assert transformer.is_strict_compatible is False


def test_lossy_array_min_items():
    """Array with minItems > 1 should be lossy (constraint gets dropped)."""

    class ItemList(BaseModel):
        items: Annotated[list[str], Field(min_length=2)]

    transformer = AnthropicJsonSchemaTransformer(ItemList.model_json_schema(), strict=True)
    transformer.walk()

    # SDK drops minItems > 1, making it lossy
    assert transformer.is_strict_compatible is False


def test_lossy_unsupported_string_format():
    """String with unsupported format should be lossy (format gets dropped)."""
    # Note: Using raw schema because Pydantic doesn't expose custom format generation in normal API
    schema = {
        'type': 'object',
        'properties': {
            'value': {
                'type': 'string',
                'format': 'regex',  # Unsupported format (not in SupportedStringFormats)
            }
        },
        'required': ['value'],
    }

    transformer = AnthropicJsonSchemaTransformer(schema, strict=True)
    transformer.walk()

    # SDK drops unsupported formats, making it lossy
    assert transformer.is_strict_compatible is False


def test_lossy_nested_defs():
    """Schema with $defs containing nested schemas with constraints should be lossy."""

    class ConstrainedString(BaseModel):
        value: Annotated[str, Field(min_length=5)]

    class Container(BaseModel):
        item: ConstrainedString

    original = Container.model_json_schema()
    transformer = AnthropicJsonSchemaTransformer(original, strict=True)
    result = transformer.walk()

    # Nested schema in $defs has constraints, making it lossy
    assert transformer.is_strict_compatible is False

    assert original == snapshot(
        {
            '$defs': {
                'ConstrainedString': {
                    'properties': {'value': {'minLength': 5, 'type': 'string'}},
                    'required': ['value'],
                    'type': 'object',
                }
            },
            'properties': {'item': {'$ref': '#/$defs/ConstrainedString'}},
            'required': ['item'],
            'title': 'Container',
            'type': 'object',
        }
    )
    assert result == snapshot(
        {
            '$defs': {
                'ConstrainedString': {
                    'type': 'object',
                    'properties': {'value': {'type': 'string', 'description': '{minLength: 5}'}},
                    'additionalProperties': False,
                    'required': ['value'],
                }
            },
            'type': 'object',
            'properties': {'item': {'$ref': '#/$defs/ConstrainedString'}},
            'additionalProperties': False,
            'required': ['item'],
        }
    )

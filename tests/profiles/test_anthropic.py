"""Tests for Anthropic JSON schema transformer.

The AnthropicJsonSchemaTransformer handles schema transformation based on the strict parameter:
- strict=True: Calls Anthropic's transform_schema() which adds additionalProperties
  and moves unsupported constraints to descriptions
- strict=False/None: Does not call transform_schema()

In all cases, title and $schema fields are removed by the base transformer.

The is_strict_compatible flag is set based on the strict parameter:
- strict=True → is_strict_compatible=True
- strict=False/None → is_strict_compatible=False

See: https://docs.claude.com/en/docs/build-with-claude/structured-outputs
"""

from __future__ import annotations as _annotations

from typing import Annotated

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.profiles.anthropic import AnthropicJsonSchemaTransformer, anthropic_model_profile

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]


# =============================================================================
# Transformer Tests - strict=True (transformation enabled)
# =============================================================================


def test_strict_true_simple_schema():
    """With strict=True, simple schemas are transformed (additionalProperties added, title removed)."""

    class Person(BaseModel):
        name: str
        age: int

    transformer = AnthropicJsonSchemaTransformer(Person.model_json_schema(), strict=True)
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


def test_strict_true_schema_with_constraints():
    """With strict=True, schemas with constraints are transformed (constraints moved to description)."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        email: Annotated[str, Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')]

    original_schema = User.model_json_schema()
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert original_schema == snapshot(
        {
            'properties': {
                'username': {'minLength': 3, 'title': 'Username', 'type': 'string'},
                'email': {'pattern': '^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', 'title': 'Email', 'type': 'string'},
            },
            'required': ['username', 'email'],
            'title': 'User',
            'type': 'object',
        }
    )
    # Anthropic's transform_schema() moves unsupported constraints to description
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'type': 'string', 'description': '{minLength: 3}'},
                'email': {'type': 'string', 'description': '{pattern: ^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$}'},
            },
            'additionalProperties': False,
            'required': ['username', 'email'],
        }
    )


def test_strict_true_nested_model():
    """With strict=True, nested models are transformed."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    transformer = AnthropicJsonSchemaTransformer(Person.model_json_schema(), strict=True)
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


# =============================================================================
# Transformer Tests - strict=False (transformation disabled)
# =============================================================================


def test_strict_false_preserves_schema():
    """With strict=False, schemas are not transformed (only title/$schema removed)."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        age: int

    original_schema = User.model_json_schema()
    transformer = AnthropicJsonSchemaTransformer(original_schema, strict=False)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    # Constraints preserved, title removed
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['username', 'age'],
        }
    )


# =============================================================================
# Transformer Tests - strict=None (transformation disabled, default case)
# =============================================================================


def test_strict_none_preserves_schema():
    """With strict=None (default), schemas are not transformed (only title/$schema removed)."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        age: int

    transformer = AnthropicJsonSchemaTransformer(User.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    # Constraints preserved, title removed
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['username', 'age'],
        }
    )


def test_strict_none_simple_schema():
    """With strict=None, simple schemas are not transformed (only title/$schema removed)."""

    class Person(BaseModel):
        name: str
        age: int

    transformer = AnthropicJsonSchemaTransformer(Person.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    # No additionalProperties added, title removed
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'required': ['name', 'age'],
        }
    )


# =============================================================================
# Model Profile Tests
# =============================================================================


def test_model_profile_supported_model():
    """Models that support structured outputs have supports_json_schema_output=True."""
    profile = anthropic_model_profile('claude-sonnet-4-5')
    assert profile is not None
    assert profile.supports_json_schema_output is True


def test_model_profile_unsupported_model():
    """Models that don't support structured outputs have supports_json_schema_output=False."""
    profile = anthropic_model_profile('claude-sonnet-4-0')
    assert profile is not None
    assert profile.supports_json_schema_output is False


def test_model_profile_opus():
    """Opus 4.1 supports structured outputs."""
    profile = anthropic_model_profile('claude-opus-4-1')
    assert profile is not None
    assert profile.supports_json_schema_output is True

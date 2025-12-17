"""Tests for Google JSON schema transformer.

The GoogleJsonSchemaTransformer transforms JSON schemas for compatibility with Gemini API:
- Converts `const` to `enum` with inferred `type` field
- Removes unsupported fields like $schema, title, discriminator, examples
- Handles format fields by moving them to description
"""

from __future__ import annotations as _annotations

from typing import Literal

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile


# =============================================================================
# Transformer Tests - const to enum conversion with type inference
# =============================================================================


def test_const_string_infers_type():
    """When converting const to enum, type should be inferred for string values."""
    schema = {'const': 'hello'}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert transformed == snapshot({'enum': ['hello'], 'type': 'string'})


def test_const_integer_infers_type():
    """When converting const to enum, type should be inferred for integer values."""
    schema = {'const': 42}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert transformed == snapshot({'enum': [42], 'type': 'integer'})


def test_const_float_infers_type():
    """When converting const to enum, type should be inferred for float values."""
    schema = {'const': 3.14}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert transformed == snapshot({'enum': [3.14], 'type': 'number'})


def test_const_boolean_infers_type():
    """When converting const to enum, type should be inferred for boolean values."""
    schema = {'const': True}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert transformed == snapshot({'enum': [True], 'type': 'boolean'})


def test_const_false_boolean_infers_type():
    """When converting const to enum, type should be inferred for False boolean."""
    schema = {'const': False}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert transformed == snapshot({'enum': [False], 'type': 'boolean'})


def test_const_preserves_existing_type():
    """When const has an existing type field, it should be preserved."""
    schema = {'const': 'hello', 'type': 'string'}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert transformed == snapshot({'enum': ['hello'], 'type': 'string'})


def test_const_in_nested_object():
    """const should be properly converted in nested object properties."""

    class TaggedModel(BaseModel):
        tag: Literal['hello']
        value: str

    schema = TaggedModel.model_json_schema()
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    # The tag property should have both enum and type
    assert transformed['properties']['tag'] == snapshot({'enum': ['hello'], 'type': 'string'})


# =============================================================================
# Transformer Tests - field removal
# =============================================================================


def test_removes_schema_field():
    """$schema field should be removed."""
    schema = {'$schema': 'http://json-schema.org/draft-07/schema#', 'type': 'string'}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert '$schema' not in transformed
    assert transformed == snapshot({'type': 'string'})


def test_removes_title_field():
    """title field should be removed."""
    schema = {'title': 'MyString', 'type': 'string'}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert 'title' not in transformed
    assert transformed == snapshot({'type': 'string'})


def test_removes_discriminator_field():
    """discriminator field should be removed."""
    schema = {'discriminator': {'propertyName': 'type'}, 'type': 'object'}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert 'discriminator' not in transformed
    assert transformed == snapshot({'type': 'object'})


def test_removes_examples_field():
    """examples field should be removed."""
    schema = {'examples': ['foo', 'bar'], 'type': 'string'}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert 'examples' not in transformed
    assert transformed == snapshot({'type': 'string'})


def test_removes_exclusive_min_max():
    """exclusiveMinimum and exclusiveMaximum should be removed."""
    schema = {'type': 'integer', 'exclusiveMinimum': 0, 'exclusiveMaximum': 100}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert 'exclusiveMinimum' not in transformed
    assert 'exclusiveMaximum' not in transformed
    assert transformed == snapshot({'type': 'integer'})


# =============================================================================
# Transformer Tests - format handling
# =============================================================================


def test_format_moved_to_description():
    """format should be moved to description for string types."""
    schema = {'type': 'string', 'format': 'date-time'}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert 'format' not in transformed
    assert transformed == snapshot({'type': 'string', 'description': 'Format: date-time'})


def test_format_appended_to_existing_description():
    """format should be appended to existing description."""
    schema = {'type': 'string', 'format': 'email', 'description': 'User email address'}
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    assert 'format' not in transformed
    assert transformed == snapshot({'type': 'string', 'description': 'User email address (format: email)'})


# =============================================================================
# Model Profile Tests
# =============================================================================


def test_model_profile_gemini_2():
    """Gemini 2.x models should have proper profile settings."""
    profile = google_model_profile('gemini-2.0-flash')
    assert profile is not None
    assert profile.json_schema_transformer == GoogleJsonSchemaTransformer
    assert profile.supports_json_schema_output is True


def test_model_profile_gemini_3():
    """Gemini 3.x models should support native output with builtin tools."""
    profile = google_model_profile('gemini-3.0-pro')
    assert profile is not None
    assert profile.google_supports_native_output_with_builtin_tools is True  # type: ignore


def test_model_profile_image_model():
    """Image models should have limited capabilities."""
    profile = google_model_profile('gemini-2.0-flash-image')
    assert profile is not None
    assert profile.supports_image_output is True
    assert profile.supports_json_schema_output is False
    assert profile.supports_tools is False

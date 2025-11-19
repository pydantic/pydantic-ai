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

    transformer = AnthropicJsonSchemaTransformer(User.model_json_schema(), strict=True)
    transformer.walk()

    # SDK drops minLength, making it lossy
    assert transformer.is_strict_compatible is False


def test_lossy_number_constraints():
    """Number with minimum constraint should be lossy (constraint gets dropped)."""

    class Product(BaseModel):
        price: Annotated[float, Field(ge=0)]

    transformer = AnthropicJsonSchemaTransformer(Product.model_json_schema(), strict=True)
    transformer.walk()

    # SDK drops minimum, making it lossy
    assert transformer.is_strict_compatible is False


def test_lossy_pattern_constraint():
    """String with pattern constraint should be lossy (constraint gets dropped)."""

    class Email(BaseModel):
        address: Annotated[str, Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')]

    transformer = AnthropicJsonSchemaTransformer(Email.model_json_schema(), strict=True)
    transformer.walk()

    # SDK drops pattern, making it lossy
    assert transformer.is_strict_compatible is False


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

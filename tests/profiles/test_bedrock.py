"""Tests for Bedrock JSON schema transformer.

The BedrockJsonSchemaTransformer strips Bedrock-incompatible constraints when
strict=True (native output or explicit strict tools) while preserving constraints
that Bedrock accepts (string constraints, enum, const, $defs/$ref, etc.).

The is_strict_compatible flag is set based on the strict parameter:
- strict=True  → is_strict_compatible=True  (constraints stripped, schema is clean)
- strict=False → is_strict_compatible=False
- strict=None  → is_strict_compatible=False  (no auto-promotion to strict)
"""

from __future__ import annotations as _annotations

from typing import Annotated, Literal

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.providers.bedrock import BedrockJsonSchemaTransformer

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='bedrock not installed'),
]


# =============================================================================
# Transformer Tests - strict=True
# =============================================================================


def test_strict_true_simple_schema():
    """With strict=True, simple schemas are returned (no-op transform), is_strict_compatible=True."""

    class Person(BaseModel):
        name: str
        age: int

    transformer = BedrockJsonSchemaTransformer(Person.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'required': ['name', 'age'],
            'additionalProperties': False,
        }
    )


def test_strict_true_schema_with_constraints():
    """With strict=True, string constraints (minLength, pattern) are preserved — Bedrock accepts these."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        email: Annotated[str, Field(pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')]

    original_schema = User.model_json_schema()
    transformer = BedrockJsonSchemaTransformer(original_schema, strict=True)
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
    # String constraints preserved (Bedrock accepts minLength/pattern), title removed
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'email': {'pattern': '^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', 'type': 'string'},
            },
            'required': ['username', 'email'],
            'additionalProperties': False,
        }
    )


def test_strict_true_nested_model():
    """With strict=True, nested models with $defs are preserved."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    transformer = BedrockJsonSchemaTransformer(Person.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            '$defs': {
                'Address': {
                    'type': 'object',
                    'properties': {
                        'street': {'type': 'string'},
                        'city': {'type': 'string'},
                    },
                    'required': ['street', 'city'],
                    'additionalProperties': False,
                }
            },
            'type': 'object',
            'additionalProperties': False,
            'properties': {'name': {'type': 'string'}, 'address': {'$ref': '#/$defs/Address'}},
            'required': ['name', 'address'],
        }
    )


# =============================================================================
# Transformer Tests - strict=False
# =============================================================================


def test_strict_false_preserves_schema():
    """With strict=False, schema is preserved, is_strict_compatible=False."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        age: int

    transformer = BedrockJsonSchemaTransformer(User.model_json_schema(), strict=False)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['username', 'age'],
            'additionalProperties': False,
        }
    )


# =============================================================================
# Transformer Tests - strict=None (default case)
# =============================================================================


def test_strict_none_preserves_schema():
    """With strict=None (default), schema is preserved, is_strict_compatible=False."""

    class User(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        age: int

    transformer = BedrockJsonSchemaTransformer(User.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'age': {'type': 'integer'},
            },
            'required': ['username', 'age'],
            'additionalProperties': False,
        }
    )


def test_strict_none_simple_schema():
    """With strict=None, simple schemas are preserved, is_strict_compatible=False."""

    class Person(BaseModel):
        name: str
        age: int

    transformer = BedrockJsonSchemaTransformer(Person.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}},
            'required': ['name', 'age'],
            'additionalProperties': False,
        }
    )


def test_strict_none_incompatible_schema_disables_auto_strict():
    """With strict=None and constrained fields, is_strict_compatible=False.

    This ensures strict is NOT auto-enabled for tools with constrained schemas,
    mirroring the Anthropic test_strict_tools_incompatible_schema_not_auto_enabled.
    """

    class ConstrainedInput(BaseModel):
        username: Annotated[str, Field(min_length=3)]
        count: Annotated[int, Field(ge=0)]

    transformer = BedrockJsonSchemaTransformer(ConstrainedInput.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    # Constraints are preserved (no stripping when strict is not True)
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'username': {'minLength': 3, 'type': 'string'},
                'count': {'minimum': 0, 'type': 'integer'},
            },
            'required': ['username', 'count'],
            'additionalProperties': False,
        }
    )


# =============================================================================
# Transformer Tests - strict=True constraint stripping
# =============================================================================


def test_strict_true_strips_numeric_constraints():
    """With strict=True, numeric constraints (minimum, maximum, multipleOf) are stripped and noted in description."""

    class Task(BaseModel):
        score: Annotated[float, Field(ge=0.0, le=100.0)]
        rating: Annotated[int, Field(multiple_of=5)]

    transformer = BedrockJsonSchemaTransformer(Task.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'score': {'type': 'number', 'description': 'minimum=0.0, maximum=100.0'},
                'rating': {'type': 'integer', 'description': 'multipleOf=5'},
            },
            'required': ['score', 'rating'],
            'additionalProperties': False,
        }
    )


def test_strict_true_strips_exclusive_bounds():
    """With strict=True, exclusive bounds (gt, lt) are stripped and noted in description."""

    class Range(BaseModel):
        value: Annotated[int, Field(gt=0, lt=100)]

    transformer = BedrockJsonSchemaTransformer(Range.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'value': {'type': 'integer', 'description': 'exclusiveMinimum=0, exclusiveMaximum=100'},
            },
            'required': ['value'],
            'additionalProperties': False,
        }
    )


def test_strict_true_strips_array_max_items():
    """With strict=True, maxItems is stripped and noted in description."""

    class Config(BaseModel):
        tags: Annotated[list[str], Field(max_length=5)]

    transformer = BedrockJsonSchemaTransformer(Config.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'tags': {'type': 'array', 'items': {'type': 'string'}, 'description': 'maxItems=5'},
            },
            'required': ['tags'],
            'additionalProperties': False,
        }
    )


def test_strict_true_strips_array_min_items_gt1():
    """With strict=True, minItems > 1 is stripped and noted in description."""

    class Config(BaseModel):
        tags: Annotated[list[str], Field(min_length=3)]

    transformer = BedrockJsonSchemaTransformer(Config.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'tags': {'type': 'array', 'items': {'type': 'string'}, 'description': 'minItems=3'},
            },
            'required': ['tags'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_array_min_items_0_and_1():
    """With strict=True, minItems=0 and minItems=1 are preserved — Bedrock accepts these."""

    class Config(BaseModel):
        optional_tags: Annotated[list[str], Field(min_length=0)]
        required_tags: Annotated[list[str], Field(min_length=1)]

    transformer = BedrockJsonSchemaTransformer(Config.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'optional_tags': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 0},
                'required_tags': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1},
            },
            'required': ['optional_tags', 'required_tags'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_string_constraints():
    """With strict=True, string constraints (minLength, maxLength, pattern) are preserved."""

    class Input(BaseModel):
        name: Annotated[str, Field(min_length=1, max_length=100)]
        code: Annotated[str, Field(pattern=r'^[A-Z]{3}$')]

    transformer = BedrockJsonSchemaTransformer(Input.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string', 'minLength': 1, 'maxLength': 100},
                'code': {'type': 'string', 'pattern': '^[A-Z]{3}$'},
            },
            'required': ['name', 'code'],
            'additionalProperties': False,
        }
    )


def test_strict_true_mixed_constraints():
    """With strict=True, numeric constraints are stripped while string constraints on the same model are kept."""

    class MixedModel(BaseModel):
        name: Annotated[str, Field(min_length=1)]
        score: Annotated[float, Field(ge=0.0, le=100.0)]

    transformer = BedrockJsonSchemaTransformer(MixedModel.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string', 'minLength': 1},
                'score': {'type': 'number', 'description': 'minimum=0.0, maximum=100.0'},
            },
            'required': ['name', 'score'],
            'additionalProperties': False,
        }
    )


def test_strict_true_description_appended():
    """With strict=True, stripped constraint info is appended to existing description, not replacing it."""

    class Task(BaseModel):
        score: Annotated[float, Field(ge=0.0, le=100.0, description='The task score')]

    transformer = BedrockJsonSchemaTransformer(Task.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'score': {
                    'type': 'number',
                    'description': 'The task score (minimum=0.0, maximum=100.0)',
                },
            },
            'required': ['score'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_default_values():
    """With strict=True, default values are preserved — Bedrock accepts these."""

    class CityWithDefaults(BaseModel):
        city: str
        country: str = 'Unknown'
        population: int = 0

    transformer = BedrockJsonSchemaTransformer(CityWithDefaults.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'city': {'type': 'string'},
                'country': {'default': 'Unknown', 'type': 'string'},
                'population': {'default': 0, 'type': 'integer'},
            },
            'required': ['city'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_any_of_with_null():
    """With strict=True, anyOf with null type (optional fields) is preserved — Bedrock accepts these."""

    class PersonOptional(BaseModel):
        name: str
        nickname: str | None = None

    transformer = BedrockJsonSchemaTransformer(PersonOptional.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'nickname': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None},
            },
            'required': ['name'],
            'additionalProperties': False,
        }
    )


def test_strict_true_preserves_literal_unions():
    """With strict=True, Literal union types are preserved via anyOf — Bedrock accepts these."""

    class StatusModel(BaseModel):
        status: Literal['active', 'inactive'] | int

    transformer = BedrockJsonSchemaTransformer(StatusModel.model_json_schema(), strict=True)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is True
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'status': {
                    'anyOf': [{'enum': ['active', 'inactive'], 'type': 'string'}, {'type': 'integer'}],
                },
            },
            'required': ['status'],
            'additionalProperties': False,
        }
    )


def test_strict_false_preserves_numeric_constraints():
    """With strict=False, numeric constraints are preserved — no stripping occurs."""

    class Task(BaseModel):
        score: Annotated[float, Field(ge=0.0, le=100.0)]
        rating: Annotated[int, Field(multiple_of=5)]

    transformer = BedrockJsonSchemaTransformer(Task.model_json_schema(), strict=False)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'score': {'type': 'number', 'minimum': 0.0, 'maximum': 100.0},
                'rating': {'type': 'integer', 'multipleOf': 5},
            },
            'required': ['score', 'rating'],
            'additionalProperties': False,
        }
    )


def test_strict_none_preserves_numeric_constraints():
    """With strict=None, numeric constraints are preserved — no stripping, no auto-promotion."""

    class Task(BaseModel):
        score: Annotated[float, Field(ge=0.0, le=100.0)]

    transformer = BedrockJsonSchemaTransformer(Task.model_json_schema(), strict=None)
    transformed = transformer.walk()

    assert transformer.is_strict_compatible is False
    assert transformed == snapshot(
        {
            'type': 'object',
            'properties': {
                'score': {'type': 'number', 'minimum': 0.0, 'maximum': 100.0},
            },
            'required': ['score'],
            'additionalProperties': False,
        }
    )

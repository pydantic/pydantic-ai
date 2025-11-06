"""Test to verify that discriminator field is not supported by Google Gemini API.

This test empirically demonstrates that Pydantic discriminated unions (which generate
oneOf schemas with discriminator mappings) cause validation errors with Google's SDK.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Cat(BaseModel):
    """A cat."""

    pet_type: Literal['cat']
    meows: int


class Dog(BaseModel):
    """A dog."""

    pet_type: Literal['dog']
    barks: float


class Owner(BaseModel):
    """An owner with a pet."""

    name: str
    pet: Cat | Dog = Field(..., discriminator='pet_type')


async def test_discriminated_union_schema_stripping():
    """Verify that discriminator field is stripped from schemas.

    This test documents that while oneOf is supported, the discriminator field
    used by Pydantic discriminated unions must be stripped because it causes
    validation errors with Google's SDK.

    Without stripping, we would get:
        properties.pet.oneOf: Extra inputs are not permitted
    """
    from typing import Any

    from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer

    # Generate schema for discriminated union
    schema = Owner.model_json_schema()

    # The schema should have discriminator before transformation
    assert 'discriminator' in schema['$defs']['Owner']['properties']['pet']

    # Transform the schema
    transformer = GoogleJsonSchemaTransformer(schema)
    transformed = transformer.walk()

    # Verify discriminator is stripped from all nested schemas
    def check_no_discriminator(obj: dict[str, Any]) -> None:
        if isinstance(obj, dict):
            assert 'discriminator' not in obj, 'discriminator should be stripped'
            for value in obj.values():
                if isinstance(value, dict):
                    check_no_discriminator(value)  # type: ignore[arg-type]
                elif isinstance(value, list):
                    for item in value:  # type: ignore[reportUnknownVariableType]
                        if isinstance(item, dict):
                            check_no_discriminator(item)  # type: ignore[arg-type]

    check_no_discriminator(transformed)

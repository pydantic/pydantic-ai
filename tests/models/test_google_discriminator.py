"""Test to verify that discriminator field is not supported by Google Gemini API.

This test empirically demonstrates that Pydantic discriminated unions (which generate
oneOf schemas with discriminator mappings) cause validation errors with Google's SDK.
"""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.providers.google import GoogleProvider


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


@pytest.mark.skip(
    reason='Discriminated unions (oneOf with discriminator) are not supported by Google Gemini API'
)
async def test_discriminated_union_not_supported():
    """Verify that discriminated unions cause validation errors.

    This test documents that while oneOf is supported, the discriminator field
    used by Pydantic discriminated unions is not supported and causes validation errors.

    Expected error:
        properties.pet.oneOf: Extra inputs are not permitted
    """
    provider = GoogleProvider(vertexai=True, project='ck-nest-dev', location='europe-west1')
    model = GoogleModel('gemini-2.5-flash', provider=provider)
    agent = Agent(model, output_type=NativeOutput(Owner))

    # This would fail with validation error if discriminator was included
    result = await agent.run('Create an owner named John with a cat that meows 5 times')
    assert result.output.name == 'John'
    assert result.output.pet.pet_type == 'cat'

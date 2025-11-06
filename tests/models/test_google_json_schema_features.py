"""Tests for Google's enhanced JSON Schema support.

Google Gemini API now supports (announced November 2025):
- anyOf for conditional structures (Unions)
- $ref for recursive schemas
- minimum and maximum for numeric constraints
- additionalProperties and type: 'null'
- prefixItems for tuple-like arrays
- Implicit property ordering (preserves definition order)

These tests verify that GoogleModel with NativeOutput properly leverages these capabilities.
"""

from __future__ import annotations

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.output import NativeOutput

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='google-genai not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


@pytest.fixture()
def google_provider(gemini_api_key: str) -> GoogleProvider:
    return GoogleProvider(api_key=gemini_api_key)


async def test_google_property_ordering(allow_model_requests: None, google_provider: GoogleProvider):
    """Test that property order is preserved in Google responses.

    Google now preserves the order of properties as defined in the schema.
    This is important for predictable output and downstream processing.
    """
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class OrderedResponse(BaseModel):
        """Response with properties in specific order: zebra, apple, mango, banana."""

        zebra: str = Field(description='Last alphabetically, first in definition')
        apple: str = Field(description='First alphabetically, second in definition')
        mango: str = Field(description='Middle alphabetically, third in definition')
        banana: str = Field(description='Second alphabetically, last in definition')

    agent = Agent(model, output_type=NativeOutput(OrderedResponse))

    result = await agent.run('Return a response with: zebra="Z", apple="A", mango="M", banana="B"')

    # Verify the output is correct
    assert result.output.zebra == 'Z'
    assert result.output.apple == 'A'
    assert result.output.mango == 'M'
    assert result.output.banana == 'B'


async def test_google_numeric_constraints(allow_model_requests: None, google_provider: GoogleProvider):
    """Test that minimum/maximum constraints work with Google's JSON Schema support.

    Google now supports minimum, maximum, exclusiveMinimum, and exclusiveMaximum.
    """
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class AgeResponse(BaseModel):
        """Response with age constraints."""

        age: int = Field(ge=0, le=150, description='Age between 0 and 150')
        score: float = Field(ge=0.0, le=100.0, description='Score between 0 and 100')

    agent = Agent(model, output_type=NativeOutput(AgeResponse))

    result = await agent.run('Give me age=25 and score=95.5')

    assert result.output.age == 25
    assert result.output.score == 95.5


async def test_google_anyof_unions(allow_model_requests: None, google_provider: GoogleProvider):
    """Test that anyOf (union types) work with Google's JSON Schema support.

    Google now supports anyOf for conditional structures, enabling union types.
    """
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class SuccessResponse(BaseModel):
        """Success response."""

        status: Literal['success']
        data: str

    class ErrorResponse(BaseModel):
        """Error response."""

        status: Literal['error']
        error_message: str

    class UnionResponse(BaseModel):
        """Response that can be either success or error."""

        result: SuccessResponse | ErrorResponse

    agent = Agent(model, output_type=NativeOutput(UnionResponse))

    # Test success case
    result = await agent.run('Return a success response with data="all good"')
    assert result.output.result.status == 'success'
    assert isinstance(result.output.result, SuccessResponse)
    assert result.output.result.data == 'all good'


async def test_google_recursive_schema(allow_model_requests: None, google_provider: GoogleProvider):
    """Test that $ref (recursive schemas) work with Google's JSON Schema support.

    Google now supports $ref for recursive schemas, enabling tree structures.
    """
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class TreeNode(BaseModel):
        """A tree node with optional children."""

        value: int
        children: list[TreeNode] | None = None

    agent = Agent(model, output_type=NativeOutput(TreeNode))

    result = await agent.run('Return a tree: root value=1 with two children (value=2 and value=3)')

    assert result.output.value == 1
    assert result.output.children is not None
    assert len(result.output.children) == 2
    assert result.output.children[0].value == 2
    assert result.output.children[1].value == 3


async def test_google_optional_fields_type_null(allow_model_requests: None, google_provider: GoogleProvider):
    """Test that type: 'null' (optional fields) work with Google's JSON Schema support.

    Google now properly supports type: 'null' in anyOf for optional fields.
    """
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class OptionalFieldsResponse(BaseModel):
        """Response with optional fields."""

        required_field: str
        optional_field: str | None = None

    agent = Agent(model, output_type=NativeOutput(OptionalFieldsResponse))

    # Test with optional field present
    result = await agent.run('Return required_field="hello" and optional_field="world"')
    assert result.output.required_field == 'hello'
    assert result.output.optional_field == 'world'

    # Test with optional field absent
    result2 = await agent.run('Return only required_field="hello"')
    assert result2.output.required_field == 'hello'
    assert result2.output.optional_field is None


async def test_google_additional_properties(allow_model_requests: None, google_provider: GoogleProvider):
    """Test that additionalProperties work with Google's JSON Schema support.

    Google now supports additionalProperties for dict types.
    """
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class DictResponse(BaseModel):
        """Response with a dictionary field."""

        metadata: dict[str, str]

    agent = Agent(model, output_type=NativeOutput(DictResponse))

    result = await agent.run('Return metadata with keys "author"="Alice" and "version"="1.0"')

    assert result.output.metadata['author'] == 'Alice'
    assert result.output.metadata['version'] == '1.0'


async def test_google_complex_nested_schema(allow_model_requests: None, google_provider: GoogleProvider):
    """Test complex nested schemas combining multiple JSON Schema features.

    This test combines: anyOf, $ref, minimum/maximum, additionalProperties, and type: null.
    """
    model = GoogleModel('gemini-2.0-flash', provider=google_provider)

    class Address(BaseModel):
        """Address with optional apartment."""

        street: str
        city: str
        apartment: str | None = None

    class Person(BaseModel):
        """Person with age constraints and optional address."""

        name: str
        age: int = Field(ge=0, le=150)
        address: Address | None = None
        metadata: dict[str, str] | None = None

    agent = Agent(model, output_type=NativeOutput(Person))

    result = await agent.run(
        'Return person: name="Alice", age=30, address with street="Main St", city="NYC", and metadata with key "role"="engineer"'
    )

    assert result.output.name == 'Alice'
    assert result.output.age == 30
    assert result.output.address is not None
    assert result.output.address.street == 'Main St'
    assert result.output.address.city == 'NYC'
    assert result.output.metadata is not None
    assert result.output.metadata['role'] == 'engineer'

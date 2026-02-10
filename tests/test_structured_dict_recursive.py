"""Tests for StructuredDict with recursive JSON schemas (issue #4018)."""

import pytest
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import TypeAliasType

from pydantic_ai import StructuredDict

pytestmark = pytest.mark.anyio


class Entry(BaseModel):
    """Entry in a map with a key and recursive value."""

    key: str
    value: 'JSONValue'
    model_config = ConfigDict(extra='forbid')


class Map(BaseModel):
    """Map containing entries."""

    entries: list[Entry]
    model_config = ConfigDict(extra='forbid')


# Recursive type alias - this creates $defs with recursive $refs
JSONValue = TypeAliasType(
    'JSONValue',
    str | int | float | bool | None | list['JSONValue'] | Map,  # pyright: ignore[reportInvalidTypeForm]
)


class OutputWithJSONValue(BaseModel):
    """Output model that includes JSONValue - generates $defs in schema."""

    name: str = Field(..., description='Name of the output')
    data: JSONValue = Field(..., description='Arbitrary JSON data')
    model_config = ConfigDict(extra='forbid')


def test_structured_dict_with_recursive_defs():
    """Test that StructuredDict can be created with a schema containing recursive $defs."""
    schema = OutputWithJSONValue.model_json_schema()

    # Verify the schema has $defs (sanity check)
    assert '$defs' in schema
    assert 'JSONValue' in schema['$defs']

    # This should not raise an error anymore
    MyStructuredDict = StructuredDict(json_schema=schema)

    # Verify it's a valid type
    assert MyStructuredDict is not None
    assert hasattr(MyStructuredDict, '__get_pydantic_json_schema__')


def test_structured_dict_recursive_validation():
    """Test that TypeAdapter can validate data against StructuredDict with recursive schema."""
    schema = OutputWithJSONValue.model_json_schema()
    MyStructuredDict = StructuredDict(json_schema=schema)

    # Create a TypeAdapter (this is what Pydantic AI uses internally)
    adapter = TypeAdapter(MyStructuredDict)

    # Test with simple data
    simple_data = {'name': 'test', 'data': 'simple_string'}
    result = adapter.validate_python(simple_data)
    assert result['name'] == 'test'
    assert result['data'] == 'simple_string'

    # Test with nested map data
    nested_data = {
        'name': 'nested_test',
        'data': {'entries': [{'key': 'a', 'value': 'hello'}]},
    }
    result = adapter.validate_python(nested_data)
    assert result['name'] == 'nested_test'
    assert result['data']['entries'][0]['key'] == 'a'

    # Test with deeply recursive data
    recursive_data = {
        'name': 'recursive_test',
        'data': {
            'entries': [
                {'key': 'level1', 'value': {'entries': [{'key': 'level2', 'value': 123}]}},
                {'key': 'another', 'value': [1, 2, 3]},
            ]
        },
    }
    result = adapter.validate_python(recursive_data)
    assert result['name'] == 'recursive_test'
    assert result['data']['entries'][0]['key'] == 'level1'


def test_structured_dict_with_simple_schema():
    """Test that StructuredDict still works with simple schemas (no $defs)."""
    simple_schema = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string'},
            'age': {'type': 'integer'},
        },
        'required': ['name', 'age'],
    }

    MyStructuredDict = StructuredDict(json_schema=simple_schema)
    adapter = TypeAdapter(MyStructuredDict)

    data = {'name': 'John', 'age': 30}
    result = adapter.validate_python(data)
    assert result['name'] == 'John'
    assert result['age'] == 30


def test_structured_dict_with_non_recursive_defs():
    """Test StructuredDict with $defs that are not recursive."""

    class Address(BaseModel):
        street: str
        city: str

    class Person(BaseModel):
        name: str
        address: Address

    schema = Person.model_json_schema()

    # This schema has $defs but they're not recursive
    assert '$defs' in schema

    MyStructuredDict = StructuredDict(json_schema=schema)
    adapter = TypeAdapter(MyStructuredDict)

    data = {'name': 'Alice', 'address': {'street': '123 Main St', 'city': 'NYC'}}
    result = adapter.validate_python(data)
    assert result['name'] == 'Alice'
    assert result['address']['city'] == 'NYC'


def test_recursive_schema_collision_coverage():
    """Test recursion where root key collides with def key (for coverage)."""
    schema = {
        'title': 'Node',
        'type': 'object',
        'properties': {'child': {'$ref': '#/$defs/Node'}},
        '$defs': {'Node': {'type': 'string'}},
    }
    sd = StructuredDict(schema)
    json_schema = sd.__get_pydantic_json_schema__(None, None)  # pyright: ignore

    # The transformer handling collision creates Node_root
    # Fix should unwrap it.
    assert json_schema['type'] == 'object'

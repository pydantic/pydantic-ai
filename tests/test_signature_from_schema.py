"""Tests for JSON schema to Python signature conversion.

Uses roundtrip testing: define functions with various signatures, generate JSON schemas,
convert back to Python signatures, and verify the output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel
from typing_extensions import NotRequired, TypedDict

from pydantic_ai._function_schema import function_schema
from pydantic_ai._signature_from_schema import (
    signature_from_function,
    signature_from_schema,
    validate_signature,
)
from pydantic_ai.tools import GenerateToolJsonSchema, RunContext


class UserInput(BaseModel):
    """User input model for roundtrip testing."""

    name: str
    email: str


class Config(TypedDict):
    """Config TypedDict for roundtrip testing."""

    timeout: int
    retries: NotRequired[int]


@dataclass
class Point:
    """Point dataclass for roundtrip testing."""

    x: float
    y: float


# Models for recursive $defs and union tests


class Address(BaseModel):
    street: str
    city: str


class User(BaseModel):
    name: str
    address: Address


class TreeNode(BaseModel):
    value: str
    children: list[TreeNode] = []


class Circle(BaseModel):
    radius: float


class Rectangle(BaseModel):
    width: float
    height: float


class Country(BaseModel):
    name: str


class CompanyAddress(BaseModel):
    street: str
    country: Country


class Company(BaseModel):
    name: str
    headquarters: CompanyAddress


class ConfigModel(BaseModel):
    key: str


class TestSignatureFromFunction:
    """Tests for signature_from_function with actual functions."""

    def test_simple_function(self):
        def greet(name: str) -> str:
            """Say hello."""
            return f'Hello, {name}!'

        result = signature_from_function(greet)
        assert result == snapshot('def greet(name: str) -> str: ...')

    def test_function_with_description(self):
        def greet(name: str) -> str:
            return f'Hello, {name}!'

        result = signature_from_function(greet, description='Say hello to someone')
        assert result == snapshot("""\
def greet(name: str) -> str:
    \"\"\"Say hello to someone\"\"\"""")

    def test_function_with_defaults(self):
        def greet(name: str, greeting: str = 'Hello') -> str:
            return f'{greeting}, {name}!'

        result = signature_from_function(greet)
        assert result == snapshot("def greet(name: str, greeting: str = 'Hello') -> str: ...")

    def test_function_with_optional(self):
        def process(data: str, limit: int | None = None) -> list[str]:
            return [data]

        result = signature_from_function(process)
        assert result == snapshot('def process(data: str, limit: int | None = None) -> list[str]: ...')

    def test_function_with_run_context(self):
        def tool_func(ctx: RunContext[int], query: str) -> str:
            return query

        result = signature_from_function(tool_func)
        assert result == snapshot('def tool_func(query: str) -> str: ...')

    def test_async_function(self):
        async def fetch(url: str) -> str:
            return url

        result = signature_from_function(fetch)
        assert result == snapshot('async def fetch(url: str) -> str: ...')

    def test_function_with_complex_types(self):
        def process(items: list[dict[str, Any]]) -> dict[str, int]:
            return {}

        result = signature_from_function(process)
        assert result == snapshot('def process(items: list[dict[str, Any]]) -> dict[str, int]: ...')

    def test_function_with_union(self):
        def process(value: int | str) -> bool:
            return True

        result = signature_from_function(process)
        assert result == snapshot('def process(value: int | str) -> bool: ...')

    def test_custom_name(self):
        def original_name(x: int) -> int:
            return x

        result = signature_from_function(original_name, name='custom_name')
        assert result == snapshot('def custom_name(x: int) -> int: ...')


class TestSignatureFromSchema:
    """Tests for signature_from_schema with JSON schemas."""

    def test_basic_types(self):
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'integer'},
                'score': {'type': 'number'},
                'active': {'type': 'boolean'},
            },
            'required': ['name', 'age', 'score', 'active'],
        }
        result = signature_from_schema('process', schema)
        assert result.signature == snapshot('def process(name: str, age: int, score: float, active: bool) -> Any: ...')
        assert result.typeddict_defs == []

    def test_optional_parameters(self):
        schema = {
            'type': 'object',
            'properties': {
                'required_param': {'type': 'string'},
                'optional_param': {'type': 'integer'},
            },
            'required': ['required_param'],
        }
        result = signature_from_schema('func', schema)
        assert result.signature == snapshot(
            'def func(required_param: str, optional_param: int | None = None) -> Any: ...'
        )

    def test_with_defaults(self):
        schema = {
            'type': 'object',
            'properties': {
                'name': {'type': 'string'},
                'count': {'type': 'integer', 'default': 10},
                'flag': {'type': 'boolean', 'default': False},
            },
            'required': ['name'],
        }
        result = signature_from_schema('func', schema)
        assert result.signature == snapshot('def func(name: str, count: int = 10, flag: bool = False) -> Any: ...')

    def test_array_types(self):
        schema = {
            'type': 'object',
            'properties': {
                'items': {'type': 'array', 'items': {'type': 'string'}},
                'numbers': {'type': 'array', 'items': {'type': 'integer'}},
            },
            'required': ['items', 'numbers'],
        }
        result = signature_from_schema('func', schema)
        assert result.signature == snapshot('def func(items: list[str], numbers: list[int]) -> Any: ...')

    def test_nested_object_generates_typeddict(self):
        schema = {
            'type': 'object',
            'properties': {
                'user': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'email': {'type': 'string'},
                    },
                    'required': ['name', 'email'],
                }
            },
            'required': ['user'],
        }
        result = signature_from_schema('get_user', schema)
        assert result.typeddict_defs == snapshot(
            [
                """\
class GetUserUser(TypedDict):
    name: str
    email: str"""
            ]
        )
        assert result.signature == snapshot('def get_user(user: GetUserUser) -> Any: ...')

    def test_anyof_with_null(self):
        schema = {
            'type': 'object',
            'properties': {'value': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}},
            'required': ['value'],
        }
        result = signature_from_schema('func', schema)
        assert result.signature == snapshot('def func(value: str | None) -> Any: ...')

    def test_anyof_union(self):
        schema = {
            'type': 'object',
            'properties': {'value': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}},
            'required': ['value'],
        }
        result = signature_from_schema('func', schema)
        assert result.signature == snapshot('def func(value: str | int) -> Any: ...')

    def test_enum(self):
        schema = {
            'type': 'object',
            'properties': {
                'status': {
                    'type': 'string',
                    'enum': ['pending', 'active', 'completed'],
                }
            },
            'required': ['status'],
        }
        result = signature_from_schema('func', schema)
        assert result.signature == snapshot("def func(status: Literal['pending', 'active', 'completed']) -> Any: ...")

    def test_with_description(self):
        schema = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
        }
        result = signature_from_schema('greet', schema, description='Greet a person')
        assert result.signature == snapshot(
            """\
def greet(name: str) -> Any:
    \"\"\"Greet a person\"\"\""""
        )

    def test_custom_return_type(self):
        schema = {
            'type': 'object',
            'properties': {'x': {'type': 'integer'}},
            'required': ['x'],
        }
        result = signature_from_schema('func', schema, return_type='str')
        assert result.signature == snapshot('def func(x: int) -> str: ...')

    def test_refs_and_defs(self):
        schema = {
            'type': 'object',
            'properties': {
                'user': {'$ref': '#/$defs/User'},
            },
            'required': ['user'],
            '$defs': {
                'User': {
                    'type': 'object',
                    'properties': {
                        'name': {'type': 'string'},
                        'age': {'type': 'integer'},
                    },
                    'required': ['name', 'age'],
                }
            },
        }
        result = signature_from_schema('process', schema)
        assert result.typeddict_defs == snapshot(
            [
                """\
class User(TypedDict):
    name: str
    age: int"""
            ]
        )
        assert result.signature == snapshot('def process(user: User) -> Any: ...')

    def test_dict_with_additional_properties(self):
        schema = {
            'type': 'object',
            'properties': {
                'metadata': {
                    'type': 'object',
                    'additionalProperties': {'type': 'string'},
                }
            },
            'required': ['metadata'],
        }
        result = signature_from_schema('func', schema)
        assert result.signature == snapshot('def func(metadata: dict[str, str]) -> Any: ...')


class TestRoundtripWithFunctionSchema:
    """Roundtrip tests: function -> JSON schema -> signature string."""

    def test_simple_function_roundtrip(self):
        def get_weather(city: str, units: str = 'celsius') -> str:
            """Get weather for a city."""
            return ''

        fs = function_schema(get_weather, GenerateToolJsonSchema)
        result = signature_from_schema('get_weather', fs.json_schema, fs.description)

        assert result.signature == snapshot(
            """\
def get_weather(city: str, units: str = 'celsius') -> Any:
    \"\"\"Get weather for a city.\"\"\""""
        )

    def test_pydantic_model_param_roundtrip(self):
        def create_user(user: UserInput) -> str:
            """Create a new user."""
            return ''

        fs = function_schema(create_user, GenerateToolJsonSchema)
        result = signature_from_schema('create_user', fs.json_schema, fs.description)

        assert 'name: str' in result.signature
        assert 'email: str' in result.signature

    def test_typeddict_param_roundtrip(self):
        def configure(config: Config) -> bool:
            """Apply configuration."""
            return True

        fs = function_schema(configure, GenerateToolJsonSchema)
        result = signature_from_schema('configure', fs.json_schema, fs.description)

        assert 'timeout: int' in result.signature

    def test_dataclass_param_roundtrip(self):
        def plot(point: Point) -> None:
            """Plot a point."""
            pass

        fs = function_schema(plot, GenerateToolJsonSchema)
        result = signature_from_schema('plot', fs.json_schema, fs.description)

        assert 'x: float' in result.signature
        assert 'y: float' in result.signature

    def test_optional_union_roundtrip(self):
        def search(query: str, limit: int | None = None) -> list[str]:
            """Search for items."""
            return []

        fs = function_schema(search, GenerateToolJsonSchema)
        result = signature_from_schema('search', fs.json_schema, fs.description)

        assert 'query: str' in result.signature
        assert 'limit:' in result.signature

    def test_list_param_roundtrip(self):
        def batch_process(items: list[str]) -> int:
            """Process items in batch."""
            return 0

        fs = function_schema(batch_process, GenerateToolJsonSchema)
        result = signature_from_schema('batch_process', fs.json_schema, fs.description)

        assert 'items: list[str]' in result.signature


class TestValidateSignature:
    """Tests for signature validation."""

    def test_valid_signature(self):
        assert validate_signature('def foo(x: int) -> str: ...') is True

    def test_valid_async_signature(self):
        assert validate_signature('async def bar(y: str) -> int: ...') is True

    def test_valid_with_docstring(self):
        sig = '''\
def foo(x: int) -> str:
    """A description."""'''
        assert validate_signature(sig) is True

    def test_invalid_signature_raises(self):
        with pytest.raises(SyntaxError):
            validate_signature('def foo(x: int -> str: ...')

    def test_valid_typeddict_and_signature(self):
        code = """\
class MyData(TypedDict):
    name: str

def func(data: MyData) -> Any: ..."""
        assert validate_signature(code) is True


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_schema(self):
        schema: dict[str, Any] = {'type': 'object', 'properties': {}}
        result = signature_from_schema('empty_func', schema)
        assert result.signature == snapshot('def empty_func() -> Any: ...')

    def test_no_properties(self):
        schema: dict[str, Any] = {'type': 'object'}
        result = signature_from_schema('no_props', schema)
        assert result.signature == snapshot('def no_props() -> Any: ...')

    def test_const_value(self):
        schema = {
            'type': 'object',
            'properties': {'action': {'const': 'delete'}},
            'required': ['action'],
        }
        result = signature_from_schema('func', schema)
        assert result.signature == snapshot("def func(action: Literal['delete']) -> Any: ...")

    def test_multiline_description(self):
        description = """Process data.

This function does important things.
Multiple lines here."""
        schema = {
            'type': 'object',
            'properties': {'x': {'type': 'integer'}},
            'required': ['x'],
        }
        result = signature_from_schema('process', schema, description=description)
        assert result.signature == snapshot(
            """\
def process(x: int) -> Any:
    \"\"\"
    Process data.

    This function does important things.
    Multiple lines here.
    \"\"\""""
        )

    def test_generated_signatures_are_valid_python(self):
        """Verify all generated signatures pass Python's parser."""
        schemas = [
            {'type': 'object', 'properties': {'x': {'type': 'string'}}, 'required': ['x']},
            {
                'type': 'object',
                'properties': {'data': {'type': 'object', 'properties': {'a': {'type': 'integer'}}}},
                'required': ['data'],
            },
            {
                'type': 'object',
                'properties': {'items': {'type': 'array', 'items': {'type': 'string'}}},
                'required': ['items'],
            },
        ]

        for schema in schemas:
            result = signature_from_schema('test_func', schema)
            full_code = '\n\n'.join(result.typeddict_defs + [result.signature])
            validate_signature(full_code)


class TestMCPSchemas:
    """Tests for MCP tool JSON schema patterns.

    MCP tools only expose JSON schemas (no function access). These tests verify
    conversion works with real MCP schema patterns found in the wild.
    """

    def test_mcp_simple_tool(self):
        """Real MCP tool schema pattern from GitHub wiki tool."""
        schema = {
            'type': 'object',
            'properties': {
                'repoName': {
                    'type': 'string',
                    'description': 'GitHub repository: owner/repo',
                }
            },
            'required': ['repoName'],
            'additionalProperties': False,
        }
        result = signature_from_schema('read_wiki_structure', schema, 'Get wiki docs')
        assert result.signature == snapshot(
            """\
def read_wiki_structure(repoName: str) -> Any:
    \"\"\"Get wiki docs\"\"\""""
        )
        assert result.typeddict_defs == snapshot([])
        validate_signature(result.signature)

    def test_mcp_with_json_schema_draft(self):
        """MCP tools may include $schema field - should be ignored."""
        schema = {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'type': 'object',
            'properties': {'query': {'type': 'string'}},
            'required': ['query'],
        }
        result = signature_from_schema('search', schema)
        assert result.signature == snapshot('def search(query: str) -> Any: ...')
        assert result.typeddict_defs == snapshot([])
        validate_signature(result.signature)

    def test_mcp_additional_properties_false(self):
        """MCP tools often have additionalProperties: false."""
        schema = {
            'type': 'object',
            'properties': {'path': {'type': 'string'}},
            'required': ['path'],
            'additionalProperties': False,
        }
        result = signature_from_schema('read_file', schema)
        assert result.signature == snapshot('def read_file(path: str) -> Any: ...')
        assert result.typeddict_defs == snapshot([])
        validate_signature(result.signature)

    def test_mcp_nested_object_with_additional_properties_false(self):
        """MCP tools with nested objects and additionalProperties: false."""
        schema = {
            'type': 'object',
            'properties': {
                'config': {
                    'type': 'object',
                    'properties': {
                        'timeout': {'type': 'integer'},
                        'retries': {'type': 'integer'},
                    },
                    'required': ['timeout'],
                    'additionalProperties': False,
                }
            },
            'required': ['config'],
            'additionalProperties': False,
        }
        result = signature_from_schema('configure', schema)
        assert result.signature == snapshot('def configure(config: ConfigureConfig) -> Any: ...')
        assert result.typeddict_defs == snapshot(
            [
                """\
class ConfigureConfig(TypedDict):
    timeout: int
    retries: NotRequired[int]"""
            ]
        )
        validate_signature('\n\n'.join(result.typeddict_defs + [result.signature]))

    def test_mcp_multiple_params(self):
        """MCP tool with multiple parameters of different types."""
        schema = {
            'type': 'object',
            'properties': {
                'owner': {'type': 'string', 'description': 'Repository owner'},
                'repo': {'type': 'string', 'description': 'Repository name'},
                'branch': {'type': 'string', 'description': 'Branch name'},
                'recursive': {'type': 'boolean', 'description': 'Include subdirectories'},
            },
            'required': ['owner', 'repo'],
            'additionalProperties': False,
        }
        result = signature_from_schema('list_files', schema, 'List files in a GitHub repository')
        assert result.signature == snapshot(
            """\
def list_files(owner: str, repo: str, branch: str | None = None, recursive: bool | None = None) -> Any:
    \"\"\"List files in a GitHub repository\"\"\""""
        )
        assert result.typeddict_defs == snapshot([])
        validate_signature(result.signature)


# Roundtrip tests for recursive $defs and union types


def test_nested_basemodel_roundtrip():
    """Nested BaseModel generates $defs with references."""

    def process(user: User) -> str:
        return ''

    fs = function_schema(process, GenerateToolJsonSchema)
    result = signature_from_schema('process', fs.json_schema, fs.description)
    # Top-level model fields are expanded; nested Address becomes a TypedDict
    assert 'Address' in '\n'.join(result.typeddict_defs)
    assert 'address: Address' in result.signature
    validate_signature('\n\n'.join(result.typeddict_defs + [result.signature]))


def test_self_referential_model_roundtrip():
    """Self-referential BaseModel (tree pattern)."""

    def traverse(node: TreeNode) -> None:
        pass

    fs = function_schema(traverse, GenerateToolJsonSchema)
    result = signature_from_schema('traverse', fs.json_schema, fs.description)
    assert 'TreeNode' in '\n'.join(result.typeddict_defs)
    validate_signature('\n\n'.join(result.typeddict_defs + [result.signature]))


def test_union_of_basemodels_roundtrip():
    """Union of BaseModels generates multiple TypedDicts."""

    def draw(shape: Circle | Rectangle) -> None:
        pass

    fs = function_schema(draw, GenerateToolJsonSchema)
    result = signature_from_schema('draw', fs.json_schema, fs.description)
    defs = '\n'.join(result.typeddict_defs)
    assert 'Circle' in defs
    assert 'Rectangle' in defs
    validate_signature('\n\n'.join(result.typeddict_defs + [result.signature]))


def test_union_basemodel_and_primitive_roundtrip():
    """Union of BaseModel and str."""

    def process(data: ConfigModel | str) -> None:
        pass

    fs = function_schema(process, GenerateToolJsonSchema)
    result = signature_from_schema('process', fs.json_schema, fs.description)
    validate_signature('\n\n'.join(result.typeddict_defs + [result.signature]))


def test_deeply_nested_models_roundtrip():
    """Three levels of nested BaseModels."""

    def info(company: Company) -> str:
        return ''

    fs = function_schema(info, GenerateToolJsonSchema)
    result = signature_from_schema('info', fs.json_schema, fs.description)
    defs = '\n'.join(result.typeddict_defs)
    assert 'Country' in defs
    assert 'CompanyAddress' in defs
    assert 'Company' in defs
    validate_signature('\n\n'.join(result.typeddict_defs + [result.signature]))

"""Pin schema constraints before requests; provider recordings do not validate arbitrary schemas."""

from copy import deepcopy

import pytest

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer, JsonSchema, JsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, GoogleOpenAPISchemaTransformer
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.tools import ToolDefinition


@pytest.mark.parametrize('transformer', [InlineDefsJsonSchemaTransformer, GoogleOpenAPISchemaTransformer])
@pytest.mark.parametrize('definition', [True, False])
@pytest.mark.parametrize('root', [False, True])
@pytest.mark.parametrize('siblings', [{}, {'description': 'value'}, {'not': {'type': 'string'}}, {'type': 'integer'}])
def test_inline_boolean_defs(
    transformer: type[JsonSchemaTransformer], definition: bool, root: bool, siblings: JsonSchema
):
    reference: JsonSchema = {'$ref': '#/$defs/X', **deepcopy(siblings)}
    schema: JsonSchema = reference if root else {'type': 'object', 'properties': {'x': reference}, 'required': ['x']}
    schema['$defs'] = {'X': definition}
    original = deepcopy(schema)
    expected = deepcopy(siblings)
    if not definition:
        # `not: {}` rejects every value, including 7 when the sibling was `not: {type: string}`.
        expected['not'] = {}

    walker = transformer(schema)
    if transformer is GoogleOpenAPISchemaTransformer and not definition:
        with pytest.raises(UserError, match='cannot represent a boolean `false` definition'):
            walker.walk()
        assert schema == original
        return
    result = walker.walk()
    assert result == (expected if root else {'type': 'object', 'properties': {'x': expected}, 'required': ['x']})
    assert walker.walk() == result
    assert schema == original


@pytest.mark.parametrize('transformer', [InlineDefsJsonSchemaTransformer, GoogleOpenAPISchemaTransformer])
@pytest.mark.parametrize('definition', [True, False])
@pytest.mark.parametrize('root', [False, True])
@pytest.mark.parametrize(
    'name,reference,encoded_name',
    [
        ('X/Y', '#/$defs/X~1Y', 'X~1Y'),
        ('X~Y', '#/$defs/X~0Y', 'X~0Y'),
        ('X~1Y', '#/$defs/X~01Y', 'X~01Y'),
    ],
)
def test_inline_boolean_defs_escaped_names(
    transformer: type[JsonSchemaTransformer], definition: bool, root: bool, name: str, reference: str, encoded_name: str
):
    field: JsonSchema = {'$ref': reference, 'not': {'type': 'string'}}
    schema: JsonSchema = field if root else {'type': 'object', 'properties': {'x': field}, 'required': ['x']}
    schema['$defs'] = {name: definition, encoded_name: not definition}
    original = deepcopy(schema)
    walker = transformer(schema)
    if transformer is GoogleOpenAPISchemaTransformer and not definition:
        with pytest.raises(UserError, match='cannot represent a boolean `false` definition'):
            walker.walk()
        assert schema == original
        return
    expected: JsonSchema = {'not': {'type': 'string'} if definition else {}}
    result = walker.walk()
    assert result == (expected if root else {'type': 'object', 'properties': {'x': expected}, 'required': ['x']})
    assert walker.walk() == result
    assert schema == original


@pytest.mark.parametrize('union_kind', ['allOf', 'anyOf', 'oneOf'])
@pytest.mark.parametrize('definition', [True, False])
def test_inline_boolean_defs_single_member_union(union_kind: str, definition: bool):
    schema: JsonSchema = {
        union_kind: [{'$ref': '#/$defs/X'}],
        'not': {'type': 'string'},
        '$defs': {'X': definition},
    }
    original = deepcopy(schema)
    expected: JsonSchema = {'not': {'type': 'string'}}
    if not definition:
        expected[union_kind] = [{'not': {}}]
    walker = InlineDefsJsonSchemaTransformer(schema)
    assert walker.walk() == expected
    assert walker.walk() == expected
    assert schema == original


@pytest.mark.parametrize('recursive', [False, True])
@pytest.mark.parametrize(
    'name,reference,encoded_name',
    [
        ('X/Y', '#/$defs/X~1Y', 'X~1Y'),
        ('X~Y', '#/$defs/X~0Y', 'X~0Y'),
        ('X~1Y', '#/$defs/X~01Y', 'X~01Y'),
    ],
)
def test_inline_dictionary_defs_escaped_names(name: str, reference: str, encoded_name: str, recursive: bool):
    definition: JsonSchema = (
        {'type': 'object', 'properties': {'child': {'$ref': reference}}} if recursive else {'type': 'integer'}
    )
    schema: JsonSchema = {'$ref': reference, '$defs': {name: definition, encoded_name: False}}
    original = deepcopy(schema)
    expected: JsonSchema = {'$ref': reference, '$defs': {name: deepcopy(definition)}} if recursive else definition
    walker = InlineDefsJsonSchemaTransformer(schema)
    assert walker.walk() == expected
    assert walker.walk() == expected
    assert schema == original


@pytest.mark.parametrize('definition', [True, False])
@pytest.mark.parametrize('strict', [None, False, True])
@pytest.mark.parametrize(
    'siblings',
    [{}, {'description': 'value'}, {'not': {'type': 'string'}}, {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}],
)
def test_openai_boolean_root_defs(definition: bool, strict: bool | None, siblings: JsonSchema):
    schema: JsonSchema = {'$ref': '#/$defs/X', '$defs': {'X': definition}, **deepcopy(siblings)}
    original = deepcopy(schema)
    expected: JsonSchema = {'$defs': {'X': definition}, **deepcopy(siblings)}
    if not definition:
        with pytest.raises(UserError, match='does not support boolean `false` root schemas'):
            OpenAIJsonSchemaTransformer(schema, strict=strict)
        assert schema == original
        return

    transformer = OpenAIJsonSchemaTransformer(schema, strict=strict)
    assert transformer.walk() == expected
    assert transformer.walk() == expected
    assert schema == original


@pytest.mark.parametrize('definition', [True, False])
@pytest.mark.parametrize(
    'name,reference',
    [('X/Y', '#/$defs/X~1Y'), ('X~Y', '#/$defs/X~0Y'), ('X~1Y', '#/$defs/X~01Y')],
)
def test_openai_boolean_root_defs_escaped_names(name: str, reference: str, definition: bool):
    schema: JsonSchema = {'$ref': reference, '$defs': {name: definition}, 'not': {'type': 'string'}}
    original = deepcopy(schema)
    if not definition:
        with pytest.raises(UserError, match='does not support boolean `false` root schemas'):
            OpenAIJsonSchemaTransformer(schema, strict=False)
        assert schema == original
        return
    expected: JsonSchema = {'$defs': {name: definition}, 'not': {'type': 'string'} if definition else {}}
    assert OpenAIJsonSchemaTransformer(schema, strict=False).walk() == expected
    assert schema == original


@pytest.mark.parametrize('decoy_definition', [True, False])
@pytest.mark.parametrize(
    'name,reference,encoded_name',
    [
        ('X/Y', '#/$defs/X~1Y', 'X~1Y'),
        ('X~Y', '#/$defs/X~0Y', 'X~0Y'),
        ('X~1Y', '#/$defs/X~01Y', 'X~01Y'),
    ],
)
def test_openai_dictionary_root_defs_escaped_names(
    name: str, reference: str, encoded_name: str, decoy_definition: bool
):
    definition: JsonSchema = {
        'type': 'object',
        'properties': {'value': {'type': 'integer'}},
        'required': ['value'],
        'additionalProperties': False,
    }
    schema: JsonSchema = {'$ref': reference, '$defs': {name: definition, encoded_name: decoy_definition}}
    original = deepcopy(schema)
    expected: JsonSchema = {'$defs': deepcopy(schema['$defs']), **deepcopy(definition)}
    assert OpenAIJsonSchemaTransformer(schema, strict=False).walk() == expected
    assert schema == original


@pytest.mark.parametrize(
    'profile,inlined',
    [
        pytest.param(amazon_model_profile('nova-pro'), True, id='amazon'),
        pytest.param(meta_model_profile('llama-3.3-70b'), True, id='meta'),
        pytest.param(qwen_model_profile('qwen3'), True, id='qwen'),
        pytest.param(ModelProfile(json_schema_transformer=GoogleOpenAPISchemaTransformer), True, id='google-openapi'),
        pytest.param(ModelProfile(json_schema_transformer=GoogleJsonSchemaTransformer), False, id='google'),
        pytest.param(ModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer), False, id='openai'),
    ],
)
@pytest.mark.parametrize('definition', [True, False])
def test_boolean_defs_tool_preparation(profile: ModelProfile | None, inlined: bool, definition: bool):
    schema: JsonSchema = {
        'type': 'object',
        'properties': {'x': {'$ref': '#/$defs/X', 'not': {'type': 'string'}}},
        'required': ['x'],
        '$defs': {'X': definition},
    }
    parameters = ModelRequestParameters(
        function_tools=[ToolDefinition(name='accept', parameters_json_schema=schema, strict=False)]
    )
    original = deepcopy(parameters)
    model = TestModel(profile=profile)
    if (
        profile is not None
        and profile.get('json_schema_transformer') is GoogleOpenAPISchemaTransformer
        and not definition
    ):
        with pytest.raises(UserError, match='cannot represent a boolean `false` definition'):
            model.prepare_request(None, parameters)
        assert parameters == original
        return
    _, prepared = model.prepare_request(None, parameters)
    expected = deepcopy(schema)
    if inlined:
        expected.pop('$defs')
        expected['properties']['x'] = {'not': {'type': 'string'} if definition else {}}
    elif profile is not None and profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer:
        expected['properties']['x'] = {'not': {'type': 'string'}, 'anyOf': [{'$ref': '#/$defs/X'}]}

    assert prepared.function_tools[0].parameters_json_schema == expected
    assert parameters == original

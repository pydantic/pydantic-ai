"""Tests for the _json_schema module."""

from __future__ import annotations as _annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, create_model

from pydantic_ai import Agent, ModelMessage, ModelResponse, PromptedOutput, TextPart, Tool, UserError
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer, JsonSchema, JsonSchemaTransformer
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles.meta import meta_model_profile


class _PassthroughTransformer(JsonSchemaTransformer):
    def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
        return schema


def test_simplify_nullable_unions():
    """Test the simplify_nullable_unions feature (deprecated, to be removed in v2)."""

    # Test with simplify_nullable_unions=True
    schema_with_null = {
        'anyOf': [
            {'type': 'string'},
            {'type': 'null'},
        ]
    }
    transformer = _PassthroughTransformer(schema_with_null, simplify_nullable_unions=True)
    result = transformer.walk()

    # Should collapse to a single nullable string
    assert result == {'type': 'string', 'nullable': True}

    # Test with simplify_nullable_unions=False (default)
    transformer2 = _PassthroughTransformer(schema_with_null, simplify_nullable_unions=False)
    result2 = transformer2.walk()

    # Should keep the anyOf structure
    assert 'anyOf' in result2
    assert len(result2['anyOf']) == 2

    # Test that non-nullable unions are unaffected
    schema_no_null = {
        'anyOf': [
            {'type': 'string'},
            {'type': 'number'},
        ]
    }
    transformer3 = _PassthroughTransformer(schema_no_null, simplify_nullable_unions=True)
    result3 = transformer3.walk()

    # Should keep anyOf since it's not nullable
    assert 'anyOf' in result3
    assert len(result3['anyOf']) == 2


def test_schema_defs_not_modified():
    """Test that the original schema $defs are not modified during transformation."""

    # Create a schema with $defs that should not be modified
    original_schema = {
        'type': 'object',
        'properties': {'value': {'$ref': '#/$defs/TestUnion'}},
        '$defs': {
            'TestUnion': {
                'anyOf': [
                    {'type': 'string'},
                    {'type': 'number'},
                ],
                'title': 'TestUnion',
            }
        },
    }

    # Keep a deepcopy to compare against later
    original_schema_copy = deepcopy(original_schema)

    # Transform the schema
    transformer = _PassthroughTransformer(original_schema)
    result = transformer.walk()

    # Verify the original schema was not modified
    assert original_schema == original_schema_copy

    # Verify the result is correct
    assert result == original_schema_copy


@pytest.mark.parametrize('value_schema', [True, False])
def test_boolean_schema_nodes_round_trip(value_schema: bool):
    """Boolean JSON Schema nodes should not crash the walker."""

    original_schema = {
        'type': 'object',
        'properties': {
            'fields': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'value': value_schema,
                    },
                },
            }
        },
    }

    transformer = _PassthroughTransformer(original_schema)

    assert transformer.walk() == original_schema


def test_boolean_schema_in_single_member_union():
    """A union that collapses to a single boolean member should be preserved."""

    schema = {'anyOf': [True]}
    result = _PassthroughTransformer(schema).walk()
    assert result == {'anyOf': [True]}


def test_simplify_nullable_union_with_boolean_member():
    """simplify_nullable_unions should not crash when a member is a boolean schema."""

    schema = {'anyOf': [True, {'type': 'null'}]}
    result = _PassthroughTransformer(schema, simplify_nullable_unions=True).walk()
    assert result == {'anyOf': [True, {'type': 'null'}]}


_VALUE_LIMIT = 1_000
"""The limit these tests patch in, so their schemas stay small."""


def _doubling_chain(levels: int) -> JsonSchema:
    """A schema whose definitions each refer to the one below twice, so inlining it doubles per level."""
    defs: dict[str, Any] = {'D0': {'type': 'string'}}
    for i in range(1, levels + 1):
        ref = {'$ref': f'#/$defs/D{i - 1}'}
        defs[f'D{i}'] = {'type': 'object', 'properties': {'a': ref, 'b': ref}}
    return {'type': 'object', 'properties': {'x': {'$ref': f'#/$defs/D{levels}'}}, '$defs': defs}


def test_inlining_past_the_value_limit_is_refused_before_the_request(monkeypatch: pytest.MonkeyPatch):
    """A tool schema that would only grow past the limit once inlined is refused while the request is prepared.

    `FunctionModel` rather than VCR: the profiles that inline `$defs` are served by many providers, and the limit is
    reached while preparing the request, before any provider is involved.
    """

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('done')])  # pragma: no cover

    def look_up(**kwargs: Any) -> str:
        return 'found'  # pragma: no cover

    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', _VALUE_LIMIT, raising=False)
    # Over 2 ** levels JSON values once inlined, from a schema of a few dozen.
    schema = _doubling_chain(_VALUE_LIMIT.bit_length())
    tool = Tool.from_schema(look_up, name='look_up', description=None, json_schema=schema)
    agent = Agent(FunctionModel(model_function, profile=meta_model_profile('llama-3.3-70b')), tools=[tool])

    with pytest.raises(
        UserError, match=f"^The JSON schema of tool 'look_up' is too large: .* more than {_VALUE_LIMIT:,} JSON values"
    ):
        agent.run_sync('look it up')


def test_an_output_schema_past_the_value_limit_is_refused_before_the_request(monkeypatch: pytest.MonkeyPatch):
    """The schema a prompted output describes is limited the same way, and the error says it is the output's."""

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('{}')])  # pragma: no cover

    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', _VALUE_LIMIT, raising=False)
    # Each level refers to the one below twice, so inlining doubles per level.
    level: type[BaseModel] = create_model('Level0', a=(bool, ...), b=(bool, ...))
    for i in range(1, _VALUE_LIMIT.bit_length() + 1):
        level = create_model(f'Level{i}', a=(level, ...), b=(level, ...))
    agent = Agent(
        FunctionModel(model_function, profile=meta_model_profile('llama-3.3-70b')), output_type=PromptedOutput(level)
    )

    with pytest.raises(
        UserError, match=f'^The output JSON schema is too large: .* more than {_VALUE_LIMIT:,} JSON values'
    ):
        agent.run_sync('fill it in')


def test_inlining_counts_every_value_of_every_copy(monkeypatch: pytest.MonkeyPatch):
    """The count is what inlining adds: every value in a definition, at each site it is copied to, walked or not."""
    pet = {'type': 'object', 'properties': {'kind': {'enum': ['cat', 'dog']}}}
    schema = {
        'type': 'object',
        'properties': {name: {'$ref': '#/$defs/Pet'} for name in ('a', 'b', 'c')},
        '$defs': {'Pet': pet},
    }
    # Three copies of `Pet`, of 7 JSON values each: the object, its `type` and `properties`, and `kind` with its
    # `enum` list of two.
    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', 21, raising=False)

    assert InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()['properties'] == {'a': pet, 'b': pet, 'c': pet}

    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', 20, raising=False)

    with pytest.raises(UserError, match='expand to more than 20 JSON values'):
        InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()


def test_a_schema_walked_without_inlining_is_not_limited(monkeypatch: pytest.MonkeyPatch):
    """Without inlining, each definition is walked once and a `$ref` stays a `$ref`, so nothing expands."""
    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', _VALUE_LIMIT, raising=False)
    schema = _doubling_chain(_VALUE_LIMIT.bit_length())

    assert _PassthroughTransformer(deepcopy(schema)).walk() == schema


def test_inlining_counts_array_items_and_union_members(monkeypatch: pytest.MonkeyPatch):
    """Array items and union members in an inlined definition count once per copy, like its other values."""
    pets = {
        'type': 'array',
        'prefixItems': [{'type': 'string'}],
        'items': {'anyOf': [{'type': 'integer'}, {'type': 'null'}]},
    }
    schema = {
        'type': 'object',
        'properties': {name: {'$ref': '#/$defs/Pets'} for name in ('a', 'b')},
        '$defs': {'Pets': pets},
    }
    # Two copies of `Pets`, of 11 JSON values each: the array, its `type`, `prefixItems` with its one schema of two
    # values, and `items` with its `anyOf` list of two schemas of two values each.
    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', 22, raising=False)

    assert InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()['properties'] == {'a': pets, 'b': pets}

    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', 21, raising=False)

    with pytest.raises(UserError, match='expand to more than 21 JSON values'):
        InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()


def test_inlining_counts_a_shared_object_at_each_place_it_appears(monkeypatch: pytest.MonkeyPatch):
    """An object a definition repeats outside its walked subschemas, here in `examples`, counts at each place."""
    kind = {'enum': ['cat', 'dog']}
    pet = {'type': 'object', 'properties': {'kind': kind}, 'examples': [kind]}
    schema = {'type': 'object', 'properties': {'a': {'$ref': '#/$defs/Pet'}}, '$defs': {'Pet': pet}}
    # One copy of `Pet`, of 12 JSON values: the object, its `type`, `properties`, and `examples`, and `kind` with its
    # `enum` list of two in both places.
    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', 12, raising=False)

    assert InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()['properties'] == {'a': pet}

    monkeypatch.setattr('pydantic_ai._json_schema.MAX_INLINED_SCHEMA_VALUES', 11, raising=False)

    with pytest.raises(UserError, match='expand to more than 11 JSON values'):
        InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()

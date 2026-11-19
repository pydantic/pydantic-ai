"""Tests for the _json_schema module."""

from __future__ import annotations as _annotations

import json
import re
from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, create_model
from typing_extensions import TypeAliasType

from pydantic_ai import Agent, ModelMessage, ModelResponse, PromptedOutput, TextPart, Tool, UserError
from pydantic_ai._json_schema import (
    InlineDefsJsonSchemaTransformer,
    JsonSchema,
    JsonSchemaTransformer,
    resolve_json_pointer,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, GoogleOpenAPISchemaTransformer
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as provider_imports_successful:
    from pydantic_ai.providers.anthropic import AnthropicJsonSchemaTransformer
    from pydantic_ai.providers.bedrock import BedrockJsonSchemaTransformer
    from pydantic_ai.providers.openrouter import (
        _OpenRouterGoogleJsonSchemaTransformer,  # pyright: ignore[reportPrivateUsage]
    )

Key = TypeAliasType('Key', str)


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


_MALFORMED_SCHEMAS: dict[str, tuple[Any, str]] = {
    # id: (schema, expected error message)
    'propertyNames-is-a-string': (
        {'type': 'object', 'propertyNames': 'x'},
        "Invalid JSON Schema at `propertyNames`: a schema must be an object or a boolean, got 'x'",
    ),
    'contains-is-a-string': (
        {'type': 'array', 'contains': 'x'},
        "Invalid JSON Schema at `contains`: a schema must be an object or a boolean, got 'x'",
    ),
    'root-is-a-string': (
        'string',
        "Invalid JSON Schema at `<root>`: a schema must be an object or a boolean, got 'string'",
    ),
    'subschema-is-a-string': (
        {'type': 'object', 'properties': {'x': 'string'}},
        "Invalid JSON Schema at `properties.x`: a schema must be an object or a boolean, got 'string'",
    ),
    'subschema-is-a-list': (
        {'type': 'object', 'properties': {'x': ['string']}},
        "Invalid JSON Schema at `properties.x`: a schema must be an object or a boolean, got ['string']",
    ),
    'nested': (
        {'type': 'object', 'properties': {'x': {'type': 'object', 'properties': {'y': 'string'}}}},
        "Invalid JSON Schema at `properties.x.properties.y`: a schema must be an object or a boolean, got 'string'",
    ),
    'properties-is-a-string': (
        {'type': 'object', 'properties': 'nope'},
        "Invalid JSON Schema at `properties`: `properties` must be an object of schemas, got 'nope'",
    ),
    'properties-is-empty-string': (
        {'type': 'object', 'properties': ''},
        "Invalid JSON Schema at `properties`: `properties` must be an object of schemas, got ''",
    ),
    'patternProperties-is-a-string': (
        {'type': 'object', 'patternProperties': 'nope'},
        "Invalid JSON Schema at `patternProperties`: `patternProperties` must be an object of schemas, got 'nope'",
    ),
    'additionalProperties-is-a-string': (
        {'type': 'object', 'additionalProperties': 'nope'},
        "Invalid JSON Schema at `additionalProperties`: a schema must be an object or a boolean, got 'nope'",
    ),
    'items-is-zero': (
        {'type': 'array', 'items': 0},
        'Invalid JSON Schema at `items`: a schema must be an object or a boolean, got 0',
    ),
    # A list `items` is draft 4 tuple validation, so it's checked member by member.
    'items-list-member-is-a-string': (
        {'type': 'array', 'items': ['string']},
        "Invalid JSON Schema at `items.0`: a schema must be an object or a boolean, got 'string'",
    ),
    'prefixItems-is-a-string': (
        {'type': 'array', 'prefixItems': 'nope'},
        "Invalid JSON Schema at `prefixItems`: `prefixItems` must be a list of schemas, got 'nope'",
    ),
    'anyOf-is-a-string': (
        {'anyOf': 'nope'},
        "Invalid JSON Schema at `anyOf`: `anyOf` must be a list of schemas, got 'nope'",
    ),
    'anyOf-member-is-a-string': (
        {'anyOf': [{'type': 'string'}, 'nope']},
        "Invalid JSON Schema at `anyOf.1`: a schema must be an object or a boolean, got 'nope'",
    ),
    'allOf-is-a-string-typed': (
        {'type': 'object', 'allOf': 'nope'},
        "Invalid JSON Schema at `allOf`: `allOf` must be a list of schemas, got 'nope'",
    ),
    'defs-is-a-string': (
        {'type': 'object', '$defs': 'nope'},
        "Invalid JSON Schema at `$defs`: `$defs` must be an object of schemas, got 'nope'",
    ),
    'defs-entry-is-a-string': (
        {'type': 'object', 'properties': {'x': {'$ref': '#/$defs/X'}}, '$defs': {'X': 'nope'}},
        "Invalid JSON Schema at `$defs.X`: a schema must be an object or a boolean, got 'nope'",
    ),
    'ref-is-an-int': (
        {'type': 'object', 'properties': {'x': {'$ref': 5}}},
        'Invalid JSON Schema at `properties.x.$ref`: `$ref` must be a string, got 5',
    ),
    'type-is-an-int': (
        {'type': 5},
        'Invalid JSON Schema at `type`: `type` must be a string or a list of strings, got 5',
    ),
    'required-is-an-int': (
        {'type': 'object', 'required': 5},
        'Invalid JSON Schema at `required`: `required` must be a list, got 5',
    ),
    'enum-is-an-int': (
        {'type': 'object', 'properties': {'x': {'enum': 5}}},
        'Invalid JSON Schema at `properties.x.enum`: `enum` must be a list, got 5',
    ),
    'type-list-has-an-int': (
        {'type': ['string', 5]},
        "Invalid JSON Schema at `type`: `type` must be a string or a list of strings, got ['string', 5]",
    ),
    'minItems-is-a-string': (
        {'type': 'array', 'items': {'type': 'string'}, 'minItems': 'x'},
        "Invalid JSON Schema at `minItems`: `minItems` must be an integer, got 'x'",
    ),
    'minItems-is-a-bool': (
        {'type': 'array', 'items': {'type': 'string'}, 'minItems': True},
        'Invalid JSON Schema at `minItems`: `minItems` must be an integer, got True',
    ),
    'minItems-is-a-fraction': (
        {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1.5},
        'Invalid JSON Schema at `minItems`: `minItems` must be an integer, got 1.5',
    ),
    'defs-entry-union-member-is-a-string': (
        {'type': 'object', '$defs': {'X': {'anyOf': [{'type': 'string'}, 'nope']}}},
        "Invalid JSON Schema at `$defs.X.anyOf.1`: a schema must be an object or a boolean, got 'nope'",
    ),
}


@pytest.mark.parametrize('transformer_cls', [_PassthroughTransformer, InlineDefsJsonSchemaTransformer])
@pytest.mark.parametrize('case', list(_MALFORMED_SCHEMAS), ids=list(_MALFORMED_SCHEMAS))
def test_malformed_schema_raises_user_error(transformer_cls: type[JsonSchemaTransformer], case: str):
    """A schema whose shape is not that of a JSON Schema is rejected when the transformer is constructed.

    Draft 2020-12, section 4.3: a schema is an object or a boolean. Such shapes only come from hand-written
    or third-party (e.g. MCP) tool definitions, which are passed through as plain dicts. The transformers used
    to crash on them with an `AttributeError`/`TypeError` from wherever the keyword happened to be read, and
    falsy values slipped through to the provider; instead the keywords the transformers read are declared
    once and checked once, and the error names the offending node.

    Unit test rather than VCR: the check fires before a request exists, so there is nothing to record.
    """
    schema, message = _MALFORMED_SCHEMAS[case]

    with pytest.raises(UserError) as exc_info:
        transformer_cls(deepcopy(schema))

    assert str(exc_info.value) == message


def test_malformed_schema_reached_through_a_json_pointer_raises_user_error():
    """An inlining walker follows a `$ref` into draft-07 `definitions`, which the up-front check doesn't visit."""
    schema = {
        'type': 'object',
        'properties': {'a': {'$ref': '#/definitions/A'}},
        'definitions': {'A': {'type': 'object', 'properties': {'b': 'string'}}},
    }

    with pytest.raises(UserError) as exc_info:
        InlineDefsJsonSchemaTransformer(schema).walk()

    assert str(exc_info.value) == (
        "Invalid JSON Schema at `definitions.A.properties.b`: a schema must be an object or a boolean, got 'string'"
    )


def test_schema_shape_ignores_keywords_the_transformers_do_not_read():
    """Provider extensions and keywords outside the declared shape pass through untouched, whatever their values."""
    schema = {
        'type': 'object',
        'properties': {'x': {'type': 'string', 'format': 5, 'nullable': 'yes', 'x-custom': {'a': 1}}},
        'discriminator': 'not-an-object',
        'not': 'not-checked',
    }

    assert _PassthroughTransformer(deepcopy(schema)).walk() == schema


@pytest.mark.skipif(not provider_imports_successful(), reason='provider packages not installed')
@pytest.mark.parametrize('strict', [None, True, False])
def test_malformed_schema_error_is_the_same_for_every_provider(strict: bool | None):
    """The check runs before any provider `transform()`, so the error does not depend on the model in use.

    On `main` the same input crashed at different points, or not at all, depending on the transformer.
    """
    schema = {'type': 'object', 'properties': {'x': 'string'}}
    message = "Invalid JSON Schema at `properties.x`: a schema must be an object or a boolean, got 'string'"

    for transformer_cls in (
        OpenAIJsonSchemaTransformer,
        GoogleJsonSchemaTransformer,
        GoogleOpenAPISchemaTransformer,
        AnthropicJsonSchemaTransformer,
        BedrockJsonSchemaTransformer,
        _OpenRouterGoogleJsonSchemaTransformer,
    ):
        with pytest.raises(UserError) as exc_info:
            transformer_cls(deepcopy(schema), strict=strict)
        assert str(exc_info.value) == message, transformer_cls.__name__


_VALID_SCHEMAS: dict[str, Any] = {
    'type-is-a-list': {'type': ['string', 'null']},
    # Tuples serialize as JSON arrays, and worked before the shape check existed.
    'type-is-a-tuple': {'type': ('string', 'null')},
    'required-is-a-tuple': {'type': 'object', 'properties': {'a': {'type': 'string'}}, 'required': ('a',)},
    'enum-is-a-tuple': {'enum': ('a', 'b')},
    'anyOf-is-a-tuple': {'anyOf': ({'type': 'string'}, {'type': 'null'})},
    'prefixItems-is-a-tuple': {'type': 'array', 'prefixItems': ({'type': 'string'},)},
    # Since draft 6, a number with a zero fractional part is an integer.
    'minItems-is-an-integral-float': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 2.0},
    # `None` is treated as an absent key, as tooling that dumps with `exclude_none=False` produces.
    'none-subschemas': {'additionalProperties': None, 'items': None, 'propertyNames': None, 'contains': None},
    'boolean-property': {'type': 'object', 'properties': {'x': True, 'y': False}},
    'boolean-pattern-property': {'type': 'object', 'patternProperties': {'^x': True}},
    'boolean-additional-properties': {'type': 'object', 'additionalProperties': False},
    'boolean-items': {'type': 'array', 'items': True},
    'boolean-property-names': {'type': 'object', 'propertyNames': True},
    'boolean-contains': {'type': 'array', 'contains': True},
    'boolean-prefix-item': {'type': 'array', 'prefixItems': [True, {'type': 'string'}]},
    'boolean-union-member': {'anyOf': [{'type': 'string'}, False]},
    'empty-containers': {'type': 'object', 'properties': {}, 'required': [], 'allOf': [], 'enum': []},
    'min-items-zero': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 0},
}


@pytest.mark.parametrize('case', list(_VALID_SCHEMAS), ids=list(_VALID_SCHEMAS))
def test_schema_shape_accepts_valid_schemas(case: str):
    """Shapes JSON Schema allows and the walker already handles must not be rejected by the check."""
    schema = _VALID_SCHEMAS[case]

    # Compared as JSON, since the walker rebuilds a tuple of schemas as a list.
    assert json.loads(json.dumps(_PassthroughTransformer(deepcopy(schema)).walk())) == json.loads(json.dumps(schema))


def test_schema_shape_accepts_a_boolean_defs_entry():
    """A boolean `$defs` entry is a valid schema, so the check passes it.

    The inlining walker still crashes on it (issue #8621); that is a library bug on a valid schema, and the
    shape check must not turn it into a "your schema is invalid" error.
    """
    _PassthroughTransformer({'type': 'object', '$defs': {'X': True}})


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


def test_allof_members_are_recursed():
    """allOf composition members should be recursed into by the walker, like anyOf/oneOf.

    This is a unit test because the walker is an internal helper and the regression is
    about its recursion shape, not a provider payload a VCR cassette would catch.
    """

    visited: list[dict[str, Any]] = []

    class _VisitingTransformer(JsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            if 'type' in schema and schema.get('type') == 'object':
                visited.append(schema)
            return schema

    schema = {
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
        ],
    }

    result = _VisitingTransformer(deepcopy(schema)).walk()

    # Both allOf members were recursed into (their object subschemas were visited).
    assert visited == [
        {'type': 'object', 'properties': {'a': {'type': 'string'}}},
        {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
    ]

    # The allOf structure is preserved with transformed members.
    assert result == {
        'allOf': [
            {'type': 'object', 'properties': {'a': {'type': 'string'}}},
            {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
        ],
    }


def test_allof_with_refs_is_inlined():
    """InlineDefsJsonSchemaTransformer should inline $ref members inside allOf.

    Before the fix, allOf members were never recursed into, so $ref resolution and
    inlining were bypassed for them. This is a unit test pinning the internal walk
    shape because the schema transformer is an internal helper used by providers.
    """

    from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer

    schema = {
        'allOf': [
            {'$ref': '#/$defs/Foo'},
            {'type': 'object', 'properties': {'b': {'type': 'integer'}}},
        ],
        '$defs': {
            'Foo': {'type': 'object', 'properties': {'a': {'type': 'string'}}},
        },
    }

    result = InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()

    # The $ref inside allOf was inlined; no $defs should remain since there are no
    # recursive refs and inlining is preferred.
    assert '$defs' not in result
    assert 'allOf' in result
    assert result['allOf'][0] == {'type': 'object', 'properties': {'a': {'type': 'string'}}}
    assert result['allOf'][1] == {'type': 'object', 'properties': {'b': {'type': 'integer'}}}


def test_typed_schema_anyof_member_is_recursed_google():
    """GoogleJsonSchemaTransformer should strip unsupported keys from anyOf members of a typed node.

    Before the fix, composition members (allOf/anyOf/oneOf) were only recursed when the node
    had no `type`. A typed node (e.g. `type: object`) with a sibling `anyOf` left its
    members untransformed, so provider-specific cleanup (Google strips `title` and
    `exclusiveMinimum`) was never applied to them.
    """
    from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer

    schema = {
        'type': 'object',
        'properties': {'p': {'type': 'string'}},
        'anyOf': [{'type': 'integer', 'title': 'Count', 'exclusiveMinimum': 0}],
    }

    result = GoogleJsonSchemaTransformer(deepcopy(schema)).walk()

    member = result['anyOf'][0]
    assert member['type'] == 'integer'
    assert 'title' not in member
    assert 'exclusiveMinimum' not in member


def test_typed_schema_anyof_member_is_recursed_openai_strict():
    """OpenAIJsonSchemaTransformer strict should add strict fields to anyOf members of a typed node.

    Before the fix, composition members of a typed node were never walked, so OpenAI strict
    mode additions (`additionalProperties: false` and `required`) were missing from them.
    """
    schema = {
        'type': 'object',
        'properties': {'p': {'type': 'string'}},
        'anyOf': [{'type': 'object', 'properties': {'q': {'type': 'integer'}}}],
    }

    result = OpenAIJsonSchemaTransformer(deepcopy(schema), strict=True).walk()

    member = result['anyOf'][0]
    assert member['type'] == 'object'
    assert member['additionalProperties'] is False
    assert member['required'] == ['q']


def test_described_options_fold_into_one_enum_google():
    """An `Enum` with member docstrings renders as `anyOf` of described `const`s; Gemini gets one `enum` again.

    Gemini does not hold the model to an `anyOf` of one-value `enum`s the way it holds it to a plain `enum`,
    so the options fold back together and each description moves into the parent's.
    """
    from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer

    schema = {
        'type': 'string',
        'description': 'How urgent the ticket is.',
        'anyOf': [
            {'const': 'low', 'description': 'Can wait a week.'},
            {'const': 'high', 'description': 'Needs attention today.'},
            {'const': 'unknown'},
        ],
    }

    assert GoogleJsonSchemaTransformer(deepcopy(schema)).walk() == snapshot(
        {
            'type': 'string',
            'enum': ['low', 'high', 'unknown'],
            'description': 'How urgent the ticket is.\nlow: Can wait a week.\nhigh: Needs attention today.',
        }
    )
    # A union of real alternatives is left alone, and so is one of single values of different types.
    mixed = {'anyOf': [{'const': 'low'}, {'type': 'integer'}]}
    assert GoogleJsonSchemaTransformer(deepcopy(mixed)).walk() == snapshot(
        {'anyOf': [{'enum': ['low'], 'type': 'string'}, {'type': 'integer'}]}
    )
    typed_apart = {'anyOf': [{'const': 'low'}, {'const': 1}]}
    assert GoogleJsonSchemaTransformer(deepcopy(typed_apart)).walk() == snapshot(
        {'anyOf': [{'enum': ['low'], 'type': 'string'}, {'enum': [1], 'type': 'integer'}]}
    )
    # A parent with its own `enum`, or typed differently from its options, is left alone: folding would widen it.
    own_enum = {'type': 'string', 'enum': ['low'], 'anyOf': [{'const': 'low'}, {'const': 'high'}]}
    assert GoogleJsonSchemaTransformer(deepcopy(own_enum)).walk() == snapshot(
        {
            'type': 'string',
            'enum': ['low'],
            'anyOf': [{'enum': ['low'], 'type': 'string'}, {'enum': ['high'], 'type': 'string'}],
        }
    )
    typed_parent = {'type': 'string', 'anyOf': [{'const': 1}, {'const': 2}]}
    assert GoogleJsonSchemaTransformer(deepcopy(typed_parent)).walk() == snapshot(
        {'type': 'string', 'anyOf': [{'enum': [1], 'type': 'integer'}, {'enum': [2], 'type': 'integer'}]}
    )
    # Options with no descriptions and no type fold into a bare `enum`, with nothing added to the description.
    bare = {'anyOf': [{'enum': ['a']}, {'enum': ['b']}]}
    assert GoogleJsonSchemaTransformer(deepcopy(bare)).walk() == snapshot({'enum': ['a', 'b']})


def test_typeless_anyof_member_still_recursed():
    """Control: typeless anyOf members continue to be recursed via _handle_union."""
    from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer

    schema = {
        'anyOf': [{'type': 'integer', 'title': 'Count', 'exclusiveMinimum': 0}],
    }

    result = GoogleJsonSchemaTransformer(deepcopy(schema)).walk()

    # Single-member union collapses into the member, which is still transformed.
    assert result == {'type': 'integer'}


def test_list_form_items_are_walked():
    """A draft-7 tuple, spelled as an `items` list rather than `prefixItems`, has each of its schemas walked.

    `zod-to-json-schema`, which the MCP TypeScript SDK uses for zod v3 tool schemas, emits this shape.
    The walk runs before any request is built, so the transformer output is asserted directly.
    """
    schema = {
        '$defs': {'Point': {'title': 'Point', 'type': 'object', 'properties': {'x': {'type': 'integer'}}}},
        'type': 'object',
        'properties': {
            'pair': {'type': 'array', 'items': [{'type': 'string', 'title': 'A'}, {'$ref': '#/$defs/Point'}]}
        },
    }

    # OpenAI keeps `$defs` and strips `title` from each element.
    assert OpenAIJsonSchemaTransformer(deepcopy(schema), strict=False).walk() == snapshot(
        {
            'type': 'object',
            'properties': {'pair': {'type': 'array', 'items': [{'type': 'string'}, {'$ref': '#/$defs/Point'}]}},
            '$defs': {'Point': {'type': 'object', 'properties': {'x': {'type': 'integer'}}}},
        }
    )
    # Inlining resolves the `$ref` inside the list.
    assert InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk() == snapshot(
        {
            'type': 'object',
            'properties': {
                'pair': {
                    'type': 'array',
                    'items': [
                        {'type': 'string', 'title': 'A'},
                        {'title': 'Point', 'type': 'object', 'properties': {'x': {'type': 'integer'}}},
                    ],
                }
            },
        }
    )


def test_inline_defs_preserves_ref_sibling_keywords():
    """Test internal schema walking, which has no provider request to cover with VCR."""
    schema = {
        'type': 'object',
        'properties': {
            'field': {'$ref': '#/$defs/Foo', 'description': 'field-level description', 'default': None},
        },
        '$defs': {
            'Foo': {
                'type': 'object',
                'description': 'model-level description',
                'default': 'model default',
                'properties': {'x': {'type': 'integer'}},
            }
        },
    }

    result = InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()
    field = result['properties']['field']

    # The referenced definition is inlined...
    assert field['type'] == 'object'
    assert field['properties'] == {'x': {'type': 'integer'}}
    assert '$ref' not in field
    # ...and the sibling keywords are preserved rather than dropped.
    assert field['description'] == 'field-level description'
    assert field['default'] is None


# The schema pydantic emits for `Pet = TypeAliasType('Pet', Union[Cat, Dog])` used as two fields of one
# model: a union-typed `$def` referenced more than once.
SHARED_UNION_DEF_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'title': 'Args',
    'properties': {
        'first': {'$ref': '#/$defs/Pet'},
        'second': {'$ref': '#/$defs/Pet'},
    },
    'required': ['first', 'second'],
    '$defs': {
        'Cat': {
            'type': 'object',
            'title': 'Cat',
            'properties': {'meow': {'type': 'string', 'title': 'Meow'}},
            'required': ['meow'],
        },
        'Dog': {
            'type': 'object',
            'title': 'Dog',
            'properties': {'woof': {'type': 'string', 'title': 'Woof'}},
            'required': ['woof'],
        },
        'Pet': {'anyOf': [{'$ref': '#/$defs/Cat'}, {'$ref': '#/$defs/Dog'}]},
    },
}

INLINED_PET: dict[str, Any] = {
    'anyOf': [
        {
            'type': 'object',
            'title': 'Cat',
            'properties': {'meow': {'type': 'string', 'title': 'Meow'}},
            'required': ['meow'],
        },
        {
            'type': 'object',
            'title': 'Dog',
            'properties': {'woof': {'type': 'string', 'title': 'Woof'}},
            'required': ['woof'],
        },
    ]
}


def test_inline_defs_repeated_union_ref():
    """Every `$ref` to a union `$def` inlines the whole definition, not just the first one.

    The first inline site used to walk the stored definition in place and pop its `anyOf` off it, so
    every later `$ref` to the same definition inlined `{}` — a schema meaning "anything", silently
    sent to the model. Unit test: the corruption is in the walker itself, and a cassette would only
    pin one provider's copy of the resulting payload.
    """
    result = InlineDefsJsonSchemaTransformer(deepcopy(SHARED_UNION_DEF_SCHEMA)).walk()

    assert result['properties']['first'] == INLINED_PET
    assert result['properties']['second'] == INLINED_PET


def test_inlined_defs_are_independent_objects():
    """Each inline site gets its own objects, so mutating one doesn't reach through to the others.

    The missing copy that emptied union definitions also left object subtrees aliased across sites.
    """
    result = InlineDefsJsonSchemaTransformer(deepcopy(SHARED_UNION_DEF_SCHEMA)).walk()
    first = result['properties']['first']
    second = result['properties']['second']

    assert first is not second
    first['anyOf'][0]['properties']['meow']['type'] = 'integer'
    assert second['anyOf'][0]['properties']['meow']['type'] == 'string'


def test_inline_defs_does_not_mutate_defs():
    """Inlining reads the stored definitions without walking or transforming them in place."""
    schema = deepcopy(SHARED_UNION_DEF_SCHEMA)
    transformer = InlineDefsJsonSchemaTransformer(schema)
    transformer.walk()

    assert schema == SHARED_UNION_DEF_SCHEMA
    assert transformer.defs == SHARED_UNION_DEF_SCHEMA['$defs']


def test_inline_defs_walks_each_def_once():
    """A `$def` is walked once per `walk()`, however many times it's referenced.

    Inlining correctly means expanding the definition's whole subtree at every reference site, which
    without this would repeat the walk once per site (and, transitively, once per nested `$ref`).
    """
    transformed: list[str] = []

    class _TitleRecordingTransformer(InlineDefsJsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            if title := schema.get('title'):
                transformed.append(title)
            return schema

    _TitleRecordingTransformer(deepcopy(SHARED_UNION_DEF_SCHEMA)).walk()

    # `Cat` and `Dog` are each walked once even though `Pet` — itself walked once for both fields —
    # references them, and each field inlines a copy of the result.
    assert transformed == ['Meow', 'Cat', 'Woof', 'Dog', 'Args']


def test_inline_defs_rewalks_defs_on_each_walk():
    """Each `walk()` transforms definitions using the transformer's current state."""

    class _StrictTitleTransformer(InlineDefsJsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            if self.strict and (title := schema.get('title')):
                schema['title'] = title.upper()
            return schema

    schema = {
        'type': 'object',
        'properties': {'value': {'$ref': '#/$defs/Value'}},
        '$defs': {'Value': {'type': 'string', 'title': 'Value'}},
    }
    transformer = _StrictTitleTransformer(schema, strict=False)

    assert transformer.walk()['properties']['value']['title'] == 'Value'

    transformer.strict = True

    assert transformer.walk()['properties']['value']['title'] == 'VALUE'


def test_inline_defs_repeated_ref_with_siblings():
    """`$ref` sibling keywords apply to their own site only, never to the shared definition."""
    schema = {
        'type': 'object',
        'properties': {
            'described': {'$ref': '#/$defs/Pet', 'description': 'field-level description'},
            'plain': {'$ref': '#/$defs/Pet'},
            'defaulted': {'$ref': '#/$defs/Pet', 'default': None},
        },
        '$defs': {'Pet': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}},
    }

    result = InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()

    pet = {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}
    assert result['properties']['described'] == {**pet, 'description': 'field-level description'}
    assert result['properties']['plain'] == pet
    assert result['properties']['defaulted'] == {**pet, 'default': None}


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


def test_inline_defs_skips_keywords_without_matching_type():
    """Object and array keywords are deliberately left as written, `$ref`s included, without a matching `type`.

    They're walked only when `type` is exactly `'object'` or `'array'` respectively. The dangling `$ref` is the
    documented output, not a bug: see the `InlineDefsJsonSchemaTransformer` docstring, and the comment in
    `JsonSchemaTransformer._handle` for why these keywords aren't walked. Setting `type` to `'object'` gets the
    reference inlined. Unit test: the behavior is the walker's own, and a cassette would only pin one provider's
    copy of the payload.
    """
    schema = {
        '$defs': {'Payload': {'type': 'object', 'properties': {'value': {'type': 'string'}}}},
        'properties': {'payload': {'$ref': '#/$defs/Payload'}},
    }

    assert InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk() == snapshot(
        {'properties': {'payload': {'$ref': '#/$defs/Payload'}}}
    )
    assert InlineDefsJsonSchemaTransformer({**deepcopy(schema), 'type': ['object', 'null']}).walk() == snapshot(
        {'properties': {'payload': {'$ref': '#/$defs/Payload'}}, 'type': ['object', 'null']}
    )
    assert InlineDefsJsonSchemaTransformer(
        {'$defs': deepcopy(schema['$defs']), 'items': {'$ref': '#/$defs/Payload'}}
    ).walk() == snapshot({'items': {'$ref': '#/$defs/Payload'}})
    assert InlineDefsJsonSchemaTransformer({**deepcopy(schema), 'type': 'object'}).walk() == snapshot(
        {'properties': {'payload': {'type': 'object', 'properties': {'value': {'type': 'string'}}}}, 'type': 'object'}
    )


def test_inline_defs_recursive_ref():
    """A recursive `$def` is emitted as `$defs` + `$ref`, walked and transformed like the rest.

    The definition emitted alongside the `$ref` used to be the object the walk had transformed in
    place; now that inlining copies instead, it comes from the same walked-once definition the inline
    sites are copied from. The transformer uppercases titles so a raw, unwalked definition would show.
    """

    class _TitleUpperTransformer(InlineDefsJsonSchemaTransformer):
        def transform(self, schema: dict[str, Any]) -> dict[str, Any]:
            if title := schema.get('title'):
                schema['title'] = title.upper()
            return schema

    schema = {
        'type': 'object',
        'title': 'Wrapper',
        'properties': {'a': {'$ref': '#/$defs/Node'}, 'b': {'$ref': '#/$defs/Node'}},
        '$defs': {
            'Node': {
                'type': 'object',
                'title': 'Node',
                'properties': {'children': {'type': 'array', 'title': 'Children', 'items': {'$ref': '#/$defs/Node'}}},
            }
        },
    }

    transformer = _TitleUpperTransformer(deepcopy(schema))
    result = transformer.walk()

    assert transformer.recursive_refs == {'Node'}
    walked_node = {
        'type': 'object',
        'title': 'NODE',
        'properties': {'children': {'type': 'array', 'title': 'CHILDREN', 'items': {'$ref': '#/$defs/Node'}}},
    }
    assert result == {
        '$defs': {
            'Node': walked_node,
            'Wrapper': {
                'type': 'object',
                'title': 'WRAPPER',
                # The first site unpacks one level of the recursion; from then on `Node` is known to
                # be recursive, so the second site keeps its `$ref`.
                'properties': {'a': walked_node, 'b': {'$ref': '#/$defs/Node'}},
            },
        },
        '$ref': '#/$defs/Wrapper',
    }


def test_inline_defs_recursive_ref_root_key_collides_with_a_def():
    """A root whose title already names a recursive `$def` gets a distinct key, not an overwrite.

    With recursive refs the output has to be `$defs` + `$ref`, and the root's key is derived from
    its `title` when it has no `$ref` of its own. If that title happens to match a definition, the
    root would otherwise clobber the definition it points at.
    """
    schema = {
        'type': 'object',
        'title': 'Node',
        'properties': {'child': {'$ref': '#/$defs/Node'}},
        '$defs': {
            'Node': {
                'type': 'object',
                'title': 'Node',
                'properties': {'child': {'$ref': '#/$defs/Node'}},
            }
        },
    }

    result = InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()

    assert result['$ref'] == '#/$defs/Node_root'
    assert set(result['$defs']) == {'Node', 'Node_root'}
    # The definition the root points at is intact, not overwritten by the root.
    assert result['$defs']['Node']['properties']['child'] == {'$ref': '#/$defs/Node'}


@pytest.mark.parametrize(
    'json_schema, location',
    [
        pytest.param({'type': 'object', 'properties': {'x': 'string'}}, 'properties.x', id='direct'),
        pytest.param(
            {
                'type': 'object',
                'properties': {'a': {'$ref': '#/definitions/A'}},
                'definitions': {'A': {'type': 'object', 'properties': {'x': 'string'}}},
            },
            'definitions.A.properties.x',
            id='through-ref',
        ),
    ],
)
def test_agent_run_rejects_a_tool_schema_that_is_not_a_json_schema(json_schema: dict[str, Any], location: str):
    """A malformed hand-written tool schema fails the run with a `UserError` before any request is made.

    A node only reachable through a `$ref` (draft-07 `definitions`) is found while walking, not up front, and
    still gets the tool name.

    The error names the tool, which is known at the `prepare_request` call site but not to the transformer.
    Provider-agnostic: `FunctionModel` with a profile that installs the inlining transformer stands in for
    Google/Bedrock, which are the providers that inline `$defs`. Not a VCR test because the rejection happens
    in `prepare_request`, before there is a request to record; the model function asserts it is never reached.
    """

    def never_called(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        raise AssertionError('the model must not be called for a tool with an invalid schema')  # pragma: no cover

    def lookup(**_kwargs: Any) -> str:
        raise AssertionError('the tool must not be called for a tool with an invalid schema')  # pragma: no cover

    model = FunctionModel(never_called, profile=ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer))
    tool = Tool.from_schema(
        lookup,
        name='lookup',
        description='A tool whose schema puts the type name where a subschema belongs.',
        json_schema=json_schema,
    )
    agent = Agent(model, tools=[tool])

    with pytest.raises(UserError) as exc_info:
        agent.run_sync('hi')

    assert str(exc_info.value) == (
        f"Tool 'lookup': Invalid JSON Schema at `{location}`: a schema must be an object or a boolean, got 'string'"
    )


ADDRESS_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {'street': {'type': 'string'}, 'city': {'type': 'string'}},
    'required': ['street', 'city'],
    'additionalProperties': False,
}

# The input schema the MCP TypeScript SDK sends for a zod v3 tool `{ from: Address, to: Address }`: its
# `zod-to-json-schema` defaults point the reused subschema at its first occurrence instead of `$defs`.
MCP_POINTER_REF_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'properties': {'from': ADDRESS_SCHEMA, 'to': {'$ref': '#/properties/from'}},
    'required': ['from', 'to'],
    'additionalProperties': False,
    '$schema': 'http://json-schema.org/draft-07/schema#',
}


@pytest.mark.parametrize(
    'schema,expected',
    [
        pytest.param(
            MCP_POINTER_REF_SCHEMA,
            {**MCP_POINTER_REF_SCHEMA, 'properties': {'from': ADDRESS_SCHEMA, 'to': ADDRESS_SCHEMA}},
            id='sibling-property',
        ),
        pytest.param(
            {
                'type': 'object',
                'properties': {
                    'list': {'type': 'array', 'items': ADDRESS_SCHEMA},
                    'one': {'$ref': '#/properties/list/items', 'description': 'One address'},
                },
            },
            {
                'type': 'object',
                'properties': {
                    'list': {'type': 'array', 'items': ADDRESS_SCHEMA},
                    'one': {**ADDRESS_SCHEMA, 'description': 'One address'},
                },
            },
            id='array-items-with-sibling-keyword',
        ),
        pytest.param(
            {
                'type': 'object',
                'properties': {'from': {'$ref': '#/definitions/Address'}},
                'definitions': {'Address': ADDRESS_SCHEMA},
            },
            {
                'type': 'object',
                'properties': {'from': ADDRESS_SCHEMA},
                'definitions': {'Address': ADDRESS_SCHEMA},
            },
            id='draft-07-definitions',
        ),
        pytest.param(
            {
                'type': 'object',
                'properties': {
                    'a/b~c': {'type': 'array', 'prefixItems': [{'type': 'integer'}, ADDRESS_SCHEMA]},
                    'to': {'$ref': '#/properties/a~1b~0c/prefixItems/1'},
                },
            },
            {
                'type': 'object',
                'properties': {
                    'a/b~c': {'type': 'array', 'prefixItems': [{'type': 'integer'}, ADDRESS_SCHEMA]},
                    'to': ADDRESS_SCHEMA,
                },
            },
            id='escaped-tokens-and-list-index',
        ),
    ],
)
def test_inline_defs_resolves_json_pointer_refs(schema: dict[str, Any], expected: dict[str, Any]):
    """A local JSON-pointer `$ref` (RFC 6901) is inlined like a `$defs` one.

    Unit test: the pointer is resolved by the walker itself; a cassette would only pin one provider's
    copy of the resulting payload.
    """
    assert InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk() == expected


@pytest.mark.parametrize(
    'schema,ref',
    [
        pytest.param(
            {
                'type': 'object',
                'properties': {'name': {'type': 'string'}, 'children': {'type': 'array', 'items': {'$ref': '#'}}},
            },
            '#',
            id='root',
        ),
        pytest.param(
            {
                'type': 'object',
                'properties': {
                    'tree': {
                        'type': 'object',
                        'properties': {'children': {'type': 'array', 'items': {'$ref': '#/properties/tree'}}},
                    }
                },
            },
            '#/properties/tree',
            id='ancestor',
        ),
        pytest.param(
            {
                'type': 'object',
                'properties': {'tree': {'$ref': '#/definitions/Tree'}},
                'definitions': {
                    'Tree': {
                        'type': 'object',
                        'properties': {'children': {'type': 'array', 'items': {'$ref': '#/definitions/Tree'}}},
                    }
                },
            },
            '#/definitions/Tree',
            id='draft-07-definitions',
        ),
    ],
)
def test_inline_defs_recursive_json_pointer_ref_raises(schema: dict[str, Any], ref: str):
    """A recursive JSON pointer has no `$defs` name to keep it as a `$ref` under, so inlining refuses it.

    Unit test: the walker raises before any request is built.
    """
    with pytest.raises(UserError, match=f'^Recursive JSON pointer `\\$ref` {re.escape(repr(ref))} '):
        InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk()


@pytest.mark.parametrize(
    'ref',
    [
        '#/properties/missing',
        '#/required/9',
        '#/properties/t/prefixItems/01',
        '#/properties/t/prefixItems/\u0661',
        '#/required/\u00b2',
        pytest.param('#/properties/t/prefixItems/' + '1' * 4301, id='index-past-int-digit-limit'),
        '#/required/0',
        '#/$defs/Missing',
    ],
)
def test_inline_defs_dangling_ref_raises(ref: str):
    """A `$ref` that resolves to no schema object raises rather than inlining something else.

    RFC 6901 array indexes are ASCII digits without a leading zero. Under a looser digit check, `01` and an
    Arabic-Indic `1` would reach `prefixItems[1]`, and a superscript `2` passes `str.isdigit()` but crashes `int()`,
    as does an index longer than Python's integer string conversion limit.
    Unit test: the walker raises before any request is built.
    """
    schema = {
        'type': 'object',
        'properties': {
            'a': {'type': 'string'},
            't': {'type': 'array', 'prefixItems': [{'type': 'integer'}, {'type': 'string'}]},
            'b': {'$ref': ref},
        },
        'required': ['a'],
    }

    with pytest.raises(
        UserError, match=f'^Could not find \\$ref definition for {re.escape(ref.removeprefix("#/$defs/"))}$'
    ):
        InlineDefsJsonSchemaTransformer(schema).walk()


def test_non_inlining_transformer_keeps_json_pointer_refs():
    """Only the inlining path resolves JSON pointers; every other transformer passes the `$ref` through.

    Unit test: pins that the walker leaves the schema as it was, which no single provider request shows.
    """
    assert _PassthroughTransformer(deepcopy(MCP_POINTER_REF_SCHEMA)).walk() == MCP_POINTER_REF_SCHEMA


def test_json_pointer_ref_tool_with_inlining_profile():
    """A tool defined from an MCP-shaped schema with a JSON-pointer `$ref` reaches an inlining model.

    `FunctionModel` rather than VCR: the profiles that inline `$defs` (Meta, Qwen, Amazon) are served
    by many providers, and the failure is in preparing the request, before any provider is involved.
    """

    def ship_order(**kwargs: Any) -> str:
        return 'shipped'  # pragma: no cover

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools[0].parameters_json_schema == snapshot(
            {
                'type': 'object',
                'properties': {
                    'from': {
                        'type': 'object',
                        'properties': {'street': {'type': 'string'}, 'city': {'type': 'string'}},
                        'required': ['street', 'city'],
                        'additionalProperties': False,
                    },
                    'to': {
                        'type': 'object',
                        'properties': {'street': {'type': 'string'}, 'city': {'type': 'string'}},
                        'required': ['street', 'city'],
                        'additionalProperties': False,
                    },
                },
                'required': ['from', 'to'],
                'additionalProperties': False,
                '$schema': 'http://json-schema.org/draft-07/schema#',
            }
        )
        return ModelResponse(parts=[TextPart('done')])

    tool = Tool.from_schema(ship_order, name='ship_order', description=None, json_schema=MCP_POINTER_REF_SCHEMA)
    agent = Agent(FunctionModel(model_function, profile=meta_model_profile('llama-3.3-70b')), tools=[tool])

    assert agent.run_sync('ship it').output == 'done'


def test_inline_defs_resolves_property_names_ref_through_tool_preparation():
    """A named dict key type reaches the model with its property-name definition inlined."""

    def accept_values(values: dict[Key, int]) -> str:
        return 'accepted'  # pragma: no cover

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools[0].parameters_json_schema == snapshot(
            {
                'type': 'object',
                'properties': {
                    'values': {
                        'type': 'object',
                        'additionalProperties': {'type': 'integer'},
                        'propertyNames': {'type': 'string'},
                    }
                },
                'required': ['values'],
                'additionalProperties': False,
            }
        )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(
        FunctionModel(model_function, profile=meta_model_profile('llama-3.3-70b')),
        tools=[accept_values],
    )
    assert agent.run_sync('accept').output == 'done'


@pytest.mark.parametrize(
    ('prefer_inlined_defs', 'expected'),
    [
        (
            True,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {
                        'values': {
                            'additionalProperties': {'type': 'integer'},
                            'propertyNames': {'type': 'string'},
                            'type': 'object',
                            'properties': {},
                        }
                    },
                    'required': ['values'],
                    'type': 'object',
                }
            ),
        ),
        (
            False,
            snapshot(
                {
                    'additionalProperties': False,
                    'properties': {
                        'values': {
                            'additionalProperties': {'type': 'integer'},
                            'propertyNames': {'$ref': '#/$defs/Key'},
                            'type': 'object',
                            'properties': {},
                        }
                    },
                    'required': ['values'],
                    'type': 'object',
                    '$defs': {'Key': {'type': 'string'}},
                }
            ),
        ),
    ],
)
def test_property_names_ref_through_tool_preparation_follows_prefer_inlined_defs(
    prefer_inlined_defs: bool, expected: JsonSchema
):
    """A transformer that drops `$defs` inlines the `$ref` under `propertyNames`; one that keeps them leaves both alone.

    A recording can't check this: cassettes match on method and URI, not the request body.
    """

    class InliningOpenAIJsonSchemaTransformer(OpenAIJsonSchemaTransformer):
        def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
            super().__init__(schema, strict=strict)
            self.prefer_inlined_defs = True

    def accept_values(values: dict[Key, int]) -> str:
        return 'accepted'  # pragma: no cover

    def model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.function_tools[0].parameters_json_schema == expected
        return ModelResponse(parts=[TextPart('done')])

    transformer = InliningOpenAIJsonSchemaTransformer if prefer_inlined_defs else OpenAIJsonSchemaTransformer
    profile = ModelProfile(json_schema_transformer=transformer)
    agent = Agent(FunctionModel(model_function, profile=profile), tools=[accept_values])
    assert agent.run_sync('accept').output == 'done'


def test_inline_defs_resolves_a_contains_ref():
    """Pydantic does not emit `contains`, so this walker-only branch needs direct coverage."""
    schema: JsonSchema = {
        'type': 'array',
        'contains': {'$ref': '#/$defs/Positive'},
        '$defs': {'Positive': {'type': 'integer', 'minimum': 1}},
    }
    assert InlineDefsJsonSchemaTransformer(schema).walk() == snapshot(
        {'type': 'array', 'contains': {'type': 'integer', 'minimum': 1}}
    )


def test_inline_defs_subclass_transforms_a_keyword_ref_once():
    """A `$ref` under `propertyNames` or `contains` is walked once, so a subclass's `transform()` runs on it once.

    A recording can't check this: cassettes match on method and URI, not the request body.
    """

    class MarkingTransformer(InlineDefsJsonSchemaTransformer):
        def transform(self, schema: JsonSchema) -> JsonSchema:
            if 'description' in schema:
                schema['description'] += '!'
            return schema

    schema: JsonSchema = {
        'type': 'object',
        'properties': {
            'keyed': {'type': 'object', 'propertyNames': {'$ref': '#/$defs/Key'}},
            'listed': {'type': 'array', 'contains': {'$ref': '#/$defs/Key'}},
            'key': {'$ref': '#/$defs/Key'},
        },
        '$defs': {'Key': {'type': 'string', 'description': 'key'}},
    }
    assert MarkingTransformer(schema).walk() == snapshot(
        {
            'type': 'object',
            'properties': {
                'keyed': {'type': 'object', 'propertyNames': {'type': 'string', 'description': 'key!'}},
                'listed': {'type': 'array', 'contains': {'type': 'string', 'description': 'key!'}},
                'key': {'type': 'string', 'description': 'key!'},
            },
        }
    )


@pytest.mark.parametrize(('type_', 'keyword'), [('object', 'propertyNames'), ('array', 'contains')])
@pytest.mark.parametrize('value', [True, False])
def test_boolean_extra_subschemas_round_trip(type_: str, keyword: str, value: bool):
    """Pydantic does not emit these boolean schemas, so exercise the walker directly."""
    schema: JsonSchema = {'type': type_, keyword: value}
    assert InlineDefsJsonSchemaTransformer(schema).walk() == schema


@pytest.mark.parametrize('node', [True, False])
def test_resolve_json_pointer_returns_boolean_schema(node: bool):
    """Boolean schemas are valid JSON Schema nodes, so a pointer to one must resolve to it."""
    schema: JsonSchema = {'properties': {'flag': node, 'obj': {'type': 'string'}}}
    assert resolve_json_pointer(schema, '#/properties/flag') is node
    assert resolve_json_pointer(schema, '#/properties/obj') == {'type': 'string'}
    assert resolve_json_pointer(schema, '#/properties/missing') is None
    assert resolve_json_pointer(schema, '#') == schema


@pytest.mark.parametrize(('flag', 'inlined'), [(True, {}), (False, {'not': {}})])
def test_inline_defs_resolves_boolean_schema_via_json_pointer(flag: bool, inlined: dict[str, Any]):
    """A `$ref` to a boolean schema outside `$defs` must inline rather than raise `UserError`.

    `_walk_def` already handles boolean definitions, but `resolve_json_pointer` used to return `None`
    for every non-dict node, so the reference looked unresolved and inlining failed.
    """
    schema: JsonSchema = {
        'type': 'object',
        'properties': {'flag': {'$ref': '#/defs/allow'}},
        'defs': {'allow': flag},
    }
    assert InlineDefsJsonSchemaTransformer(deepcopy(schema)).walk() == {
        'type': 'object',
        'properties': {'flag': inlined},
        'defs': {'allow': flag},
    }

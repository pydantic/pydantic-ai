"""Tests for the _json_schema module."""

from __future__ import annotations as _annotations

from copy import deepcopy
from typing import Any

import pytest

from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer


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
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

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


def test_typeless_anyof_member_still_recursed():
    """Control: typeless anyOf members continue to be recursed via _handle_union."""
    from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer

    schema = {
        'anyOf': [{'type': 'integer', 'title': 'Count', 'exclusiveMinimum': 0}],
    }

    result = GoogleJsonSchemaTransformer(deepcopy(schema)).walk()

    # Single-member union collapses into the member, which is still transformed.
    assert result == {'type': 'integer'}


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

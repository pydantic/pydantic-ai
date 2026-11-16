"""Dump golden fixtures from the real `pydantic-ai`, for the Rust and TypeScript conformance tests.

Run it from the repository root: `uv run python experiments/pydantic-ai-core/fixtures/generate.py`.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, get_args
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter

from pydantic_ai import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import KnownModelName
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, openai_model_profile

# The gpt-oss transformer is private, so it is taken from the profile that selects it.
GPT_OSS_TRANSFORMER = openai_model_profile('gpt-oss-120b').get('json_schema_transformer')
assert GPT_OSS_TRANSFORMER is not None

TRANSFORMER_IDS: dict[type[JsonSchemaTransformer], str] = {
    InlineDefsJsonSchemaTransformer: 'inline_defs',
    OpenAIJsonSchemaTransformer: 'openai',
    GPT_OSS_TRANSFORMER: 'gpt_oss',
}

EXTRA_MODEL_NAMES = {
    '',
    'gpt-oss-120b',
    'o1-mini',
    'gpt-4o-search-preview',
    'gpt-5.4-nano-2026-03-17',
    'gpt-5.3-codex',
    'gpt-5-chat-latest',
    'gpt-5.1-codex-max',
    'gpt-5.2-pro',
    'gpt-6-astra',
    'gpt-6.1-terra',
    'openrouter/o3',
}


class Color(StrEnum):
    """A color."""

    red = 'red'
    green = 'green'


class Address(BaseModel):
    """A postal address."""

    street: str
    zip: Annotated[str, Field(pattern=r'^\d{5}$', min_length=5, max_length=5)]


class Person(BaseModel):
    """A person."""

    name: str = Field(description='Full name')
    age: Annotated[int, Field(ge=0)] | None = None
    tags: list[str] = []
    address: Address
    home: Address | None = Field(default=None, description='Home address')
    work: Address = Field(description='Where they work')
    color: Color = Color.red
    kind: Literal['a', 'b']
    born: date
    seen: datetime
    id: UUID
    scores: dict[str, float]
    pair: tuple[int, str]
    anything: Any
    bare: list  # pyright: ignore[reportMissingTypeArgument]
    unique: set[int]


class Tree(BaseModel):
    """A tree that refers to itself."""

    value: int
    children: list[Tree] = []


class Ping(BaseModel):
    """One half of a mutually recursive pair."""

    pong: Pong | None = None


class Pong(BaseModel):
    """The other half of a mutually recursive pair."""

    ping: Ping | None = None


class Cat(BaseModel):
    """A cat."""

    type: Literal['cat']
    meow: bool


class Dog(BaseModel):
    """A dog."""

    type: Literal['dog']
    bark: bool


class Owner(BaseModel):
    """The owner of a pet, which is a discriminated union."""

    pet: Annotated[Cat | Dog, Field(discriminator='type')]


STRING = {'type': 'string'}

SCHEMAS: dict[str, dict[str, Any]] = {
    'person': Person.model_json_schema(),
    'tree': Tree.model_json_schema(),
    'ping_pong': Ping.model_json_schema(),
    'owner': Owner.model_json_schema(),
    'list_of_person': TypeAdapter(list[Person]).json_schema(),
    'optional_tree': TypeAdapter(Tree | None).json_schema(),
    'empty': {},
    'object_without_properties': {'type': 'object'},
    'boolean_subschemas': {
        'type': 'object',
        'properties': {'any': True, 'none': False, 'items_true': {'type': 'array', 'items': True}},
        'additionalProperties': True,
    },
    'additional_properties_schema': {
        'type': 'object',
        'additionalProperties': {'$ref': '#/$defs/S'},
        '$defs': {'S': STRING},
    },
    'untyped_arrays': {
        'type': 'object',
        'properties': {
            'empty_items': {'type': 'array', 'items': {}},
            'described_items': {'type': 'array', 'items': {'description': 'x'}},
            'missing_items': {'type': 'array'},
        },
        'required': ['empty_items', 'described_items', 'missing_items'],
    },
    'prefix_items': {'type': 'array', 'prefixItems': [STRING, {'$ref': '#/$defs/S'}], '$defs': {'S': STRING}},
    'draft_04_tuple': {'type': 'array', 'items': [STRING, {'type': 'integer'}]},
    'one_of': {'oneOf': [STRING, {'type': 'integer'}], 'title': 'Either'},
    'one_of_and_any_of': {'type': 'object', 'oneOf': [STRING], 'anyOf': [{'type': 'null'}]},
    'single_member_union': {'anyOf': [{'type': 'string', 'title': 'Inner'}], 'description': 'outer', 'title': 'Outer'},
    'single_member_union_rejecting': {'anyOf': [{'not': {}}], 'not': STRING},
    'nested_unions': {'allOf': [{'anyOf': [{'oneOf': [STRING]}]}]},
    'json_pointer_ref': {
        'type': 'object',
        'properties': {'from': {'type': 'string', 'minLength': 1}, 'to': {'$ref': '#/properties/from'}},
        'required': ['from', 'to'],
    },
    'json_pointer_ref_into_array': {
        'type': 'object',
        'properties': {
            'a': {'anyOf': [STRING, {'type': 'object', 'properties': {}}]},
            'b': {'$ref': '#/properties/a/anyOf/1'},
        },
    },
    'json_pointer_recursive': {
        'type': 'object',
        'properties': {'child': {'$ref': '#'}},
    },
    'draft_07_definitions': {
        'type': 'object',
        'properties': {'node': {'$ref': '#/definitions/Node'}},
        'definitions': {
            'Node': {
                'type': 'object',
                'title': 'Node',
                'properties': {'next': {'$ref': '#/definitions/Node'}, 'label': {'type': 'string', 'default': 'x'}},
            }
        },
    },
    'ref_with_siblings': {
        'type': 'object',
        'properties': {'a': {'$ref': '#/$defs/S', 'description': 'An S', 'default': 'x'}, 'b': {'$ref': '#/$defs/S'}},
        '$defs': {'S': {'type': 'string', 'title': 'S', 'description': 'shared'}},
    },
    'missing_definition': {'type': 'object', 'properties': {'a': {'$ref': '#/$defs/Missing'}}},
    'boolean_definitions': {
        'type': 'object',
        'properties': {
            'yes': {'$ref': '#/$defs/Yes'},
            'no': {'$ref': '#/$defs/No'},
            'no_with_not': {'$ref': '#/$defs/No', 'not': STRING, 'description': 'never'},
        },
        '$defs': {'Yes': True, 'No': False},
    },
    'root_ref_to_true': {'$ref': '#/$defs/Yes', '$defs': {'Yes': True}, 'description': 'anything'},
    'root_ref_to_false': {'$ref': '#/$defs/No', '$defs': {'No': False}},
    'root_ref_to_typeless_union': {
        '$ref': '#/$defs/U',
        '$defs': {'U': {'anyOf': [STRING, {'type': 'array', 'items': {'$ref': '#/$defs/U'}}], 'title': 'U'}},
    },
    'root_ref_to_single_member_union': {
        '$ref': '#/$defs/U',
        '$defs': {'U': {'allOf': [{'type': 'object', 'properties': {'u': {'$ref': '#/$defs/U'}}}], 'title': 'U'}},
    },
    'escaped_definition_name': {
        'type': 'object',
        'properties': {'a': {'$ref': '#/$defs/a~1b~0c'}},
        '$defs': {'a/b~c': {'type': 'object', 'properties': {'again': {'$ref': '#/$defs/a~1b~0c'}}}},
    },
    'type_list': {
        'type': ['object', 'null'],
        'properties': {'a': {'$ref': '#/$defs/S'}},
        '$defs': {'S': STRING},
    },
    'type_list_nested': {
        'type': 'object',
        'properties': {'maybe': {'type': ['object', 'null'], 'properties': {'a': {'$ref': '#/$defs/S'}}}},
        '$defs': {'S': STRING},
    },
    'strict_incompatible_keywords': {
        'type': 'object',
        'properties': {
            'short': {'type': 'string', 'minLength': 2, 'maxLength': 4, 'description': 'A code'},
            'mapped': {
                'type': 'object',
                'patternProperties': {'^a': {'type': 'string', 'title': 'A'}},
                'minProperties': 1,
            },
            'named': {'type': 'object', 'propertyNames': {'$ref': '#/$defs/S'}, 'unevaluatedProperties': False},
            'set': {
                'type': 'array',
                'items': STRING,
                'uniqueItems': True,
                'contains': {'const': "it's"},
                'minContains': 1,
            },
            'color': {'type': 'string', 'format': 'color'},
            'when': {'type': 'string', 'format': 'date-time'},
            'lookahead': {'type': 'string', 'pattern': '^(?=it\'s "quoted")\\w+\n$'},
            'escaped_lookahead': {'type': 'string', 'pattern': '^\\(?=x'},
            'float': {'type': 'array', 'items': STRING, 'minContains': 1.5, 'maxContains': None},
        },
        '$defs': {'S': STRING},
    },
    'required_null': {'type': 'object', 'required': None},
    'required_string_in_definitions': {
        'type': 'object',
        'properties': {},
        'required': [],
        'definitions': {'N': {'type': 'object', 'required': 'x'}},
    },
    'required_subset': {'type': 'object', 'properties': {'a': STRING, 'b': STRING}, 'required': ['a']},
    'discriminator_and_schema': {
        '$schema': 'https://json-schema.org/draft/2020-12/schema',
        'type': 'object',
        'discriminator': {},
    },
    'union_under_typed_node': {
        'type': 'object',
        'anyOf': [{'$ref': '#/$defs/S'}, {'title': 'T'}],
        '$defs': {'S': STRING},
    },
    'too_large_to_inline': {
        'type': 'object',
        'properties': {'root': {'$ref': '#/$defs/D17'}},
        '$defs': {
            'D0': STRING,
            **{
                f'D{level}': {
                    'type': 'object',
                    'properties': {side: {'$ref': f'#/$defs/D{level - 1}'} for side in ('left', 'right')},
                }
                for level in range(1, 18)
            },
        },
    },
    'malformed_properties': {'type': 'object', 'properties': []},
    'malformed_type': {'type': 5},
    'malformed_required': {'type': 'object', 'properties': {'a': {'type': 'object', 'required': 'a'}}},
    'malformed_items': {'type': 'array', 'items': "it's a string"},
    'malformed_any_of': {'anyOf': {'type': 'string'}},
    'malformed_nested': {'type': 'object', 'properties': {'a': {'type': 'array', 'items': [STRING, 3]}}},
    'malformed_definition': {'$defs': {'S': 'string'}},
    'malformed_min_items': {'type': 'array', 'minItems': 1.5},
    'malformed_pointer_target': {
        'type': 'object',
        'properties': {'a': {'$ref': '#/definitions/Bad'}},
        'definitions': {'Bad': {'properties': {'x': None, 'y': 'nope'}}},
    },
}


def encode_profile(profile: Any) -> dict[str, Any]:
    """The profile as JSON, with ids where it holds classes and sorted lists where it holds sets."""
    encoded = dict(profile)
    encoded['json_schema_transformer'] = TRANSFORMER_IDS[encoded['json_schema_transformer']]
    encoded['supported_native_tools'] = sorted(tool.kind for tool in encoded['supported_native_tools'])
    if 'openai_reasoning_efforts' in encoded:
        encoded['openai_reasoning_efforts'] = sorted(encoded['openai_reasoning_efforts'])
    return encoded


def main() -> None:
    """Write `golden.json` next to this file."""
    model_names = {name.split(':', 1)[-1] for name in get_args(KnownModelName.__value__)} | EXTRA_MODEL_NAMES
    profiles = [{'model': name, 'expected': encode_profile(openai_model_profile(name))} for name in sorted(model_names)]

    schemas: list[dict[str, Any]] = []
    crashes: list[str] = []
    for name, schema in SCHEMAS.items():
        for transformer, transformer_id in TRANSFORMER_IDS.items():
            for strict in (None, True, False):
                case = {'name': name, 'transformer': transformer_id, 'strict': strict, 'schema': schema}
                try:
                    instance = transformer(deepcopy(schema), strict=strict)
                    expected = {'schema': instance.walk(), 'is_strict_compatible': instance.is_strict_compatible}
                except UserError as e:
                    expected = {'error': str(e)}
                except (AttributeError, TypeError, KeyError, AssertionError) as e:
                    # Not a `UserError`, so not behaviour to conform to.
                    crashes.append(f'{name} {transformer_id} strict={strict}: {type(e).__name__}: {e}')
                    continue
                schemas.append({**case, 'expected': expected})

    golden = Path(__file__).parent / 'golden.json'
    golden.write_text(json.dumps({'profiles': profiles, 'schemas': schemas}, indent=1) + '\n')
    errors = sum('error' in case['expected'] for case in schemas)
    print(f'{len(profiles)} profiles, {len(schemas)} schema cases ({errors} expect an error) -> {golden}')
    for crash in crashes:
        print('skipped, Python crashed:', crash)


if __name__ == '__main__':
    main()

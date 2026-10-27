from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue, StrictInt

from pydantic_ai import Agent, ModelProfile, UserError
from pydantic_ai._json_schema import resolve_json_pointer
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.output import NativeOutput, ToolOutput
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openai_chatgpt import OpenAIChatGPTProvider
from pydantic_ai.providers.openai_codex import OpenAICodexProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture

pytestmark = pytest.mark.vcr


class LengthOutput(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=5)]


class MixedOutput(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=5)]
    values: tuple[int, int, int, int]


class SingleTupleOutput(BaseModel):
    values: tuple[StrictInt]


@dataclass(frozen=True)
class WireCase:
    id: str
    schema_kind: Literal['length', 'mixed', 'single_tuple']
    output_mode: Literal['tool', 'native']
    api: Literal['chat', 'responses']
    expected_wire: dict[str, JsonValue]


WIRE_CASES: list[WireCase] = [
    WireCase(
        'length-tool-chat',
        'length',
        'tool',
        'chat',
        snapshot(
            {
                'schema': {
                    'properties': {'name': {'maxLength': 5, 'minLength': 2, 'type': 'string'}},
                    'required': ['name'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': True,
            }
        ),
    ),
    WireCase(
        'length-tool-responses',
        'length',
        'tool',
        'responses',
        snapshot(
            {
                'schema': {
                    'properties': {'name': {'maxLength': 5, 'minLength': 2, 'type': 'string'}},
                    'required': ['name'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': True,
            }
        ),
    ),
    WireCase(
        'length-native-chat',
        'length',
        'native',
        'chat',
        snapshot(
            {
                'schema': {
                    'properties': {'name': {'maxLength': 5, 'minLength': 2, 'type': 'string'}},
                    'required': ['name'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': True,
            }
        ),
    ),
    WireCase(
        'length-native-responses',
        'length',
        'native',
        'responses',
        snapshot(
            {
                'schema': {
                    'properties': {'name': {'maxLength': 5, 'minLength': 2, 'type': 'string'}},
                    'required': ['name'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': True,
            }
        ),
    ),
    WireCase(
        'single-tuple-tool-chat',
        'single_tuple',
        'tool',
        'chat',
        snapshot(
            {
                'schema': {
                    'properties': {
                        'values': {'maxItems': 1, 'minItems': 1, 'type': 'array', 'items': {'type': 'integer'}}
                    },
                    'required': ['values'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': True,
            }
        ),
    ),
    WireCase(
        'single-tuple-tool-responses',
        'single_tuple',
        'tool',
        'responses',
        snapshot(
            {
                'schema': {
                    'properties': {
                        'values': {'maxItems': 1, 'minItems': 1, 'type': 'array', 'items': {'type': 'integer'}}
                    },
                    'required': ['values'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': True,
            }
        ),
    ),
    WireCase(
        'single-tuple-native-chat',
        'single_tuple',
        'native',
        'chat',
        snapshot(
            {
                'schema': {
                    'properties': {
                        'values': {'maxItems': 1, 'minItems': 1, 'type': 'array', 'items': {'type': 'integer'}}
                    },
                    'required': ['values'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': True,
            }
        ),
    ),
    WireCase(
        'single-tuple-native-responses',
        'single_tuple',
        'native',
        'responses',
        snapshot(
            {
                'schema': {
                    'properties': {
                        'values': {'maxItems': 1, 'minItems': 1, 'type': 'array', 'items': {'type': 'integer'}}
                    },
                    'required': ['values'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': True,
            }
        ),
    ),
    WireCase(
        'mixed-tool-chat',
        'mixed',
        'tool',
        'chat',
        snapshot(
            {
                'schema': {
                    'properties': {
                        'name': {'maxLength': 5, 'minLength': 2, 'type': 'string'},
                        'values': {
                            'maxItems': 4,
                            'minItems': 4,
                            'prefixItems': [
                                {'type': 'integer'},
                                {'type': 'integer'},
                                {'type': 'integer'},
                                {'type': 'integer'},
                            ],
                            'type': 'array',
                        },
                    },
                    'required': ['name', 'values'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': None,
            }
        ),
    ),
    WireCase(
        'mixed-tool-responses',
        'mixed',
        'tool',
        'responses',
        snapshot(
            {
                'schema': {
                    'properties': {
                        'name': {'maxLength': 5, 'minLength': 2, 'type': 'string'},
                        'values': {
                            'maxItems': 4,
                            'minItems': 4,
                            'prefixItems': [
                                {'type': 'integer'},
                                {'type': 'integer'},
                                {'type': 'integer'},
                                {'type': 'integer'},
                            ],
                            'type': 'array',
                        },
                    },
                    'required': ['name', 'values'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': False,
            }
        ),
    ),
    WireCase(
        'mixed-native-chat',
        'mixed',
        'native',
        'chat',
        snapshot(
            {
                'schema': {
                    'properties': {
                        'name': {'maxLength': 5, 'minLength': 2, 'type': 'string'},
                        'values': {
                            'maxItems': 4,
                            'minItems': 4,
                            'prefixItems': [
                                {'type': 'integer'},
                                {'type': 'integer'},
                                {'type': 'integer'},
                                {'type': 'integer'},
                            ],
                            'type': 'array',
                        },
                    },
                    'required': ['name', 'values'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': False,
            }
        ),
    ),
    WireCase(
        'mixed-native-responses',
        'mixed',
        'native',
        'responses',
        snapshot(
            {
                'schema': {
                    'properties': {
                        'name': {'maxLength': 5, 'minLength': 2, 'type': 'string'},
                        'values': {
                            'maxItems': 4,
                            'minItems': 4,
                            'prefixItems': [
                                {'type': 'integer'},
                                {'type': 'integer'},
                                {'type': 'integer'},
                                {'type': 'integer'},
                            ],
                            'type': 'array',
                        },
                    },
                    'required': ['name', 'values'],
                    'type': 'object',
                    'additionalProperties': False,
                },
                'strict': False,
            }
        ),
    ),
]


@pytest.mark.parametrize('case', WIRE_CASES, ids=lambda case: case.id)
async def test_openai_strict_schema_wire(
    case: WireCase,
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
):
    """Pin the schema and strict flag the OpenAI adapters actually send.

    Length constraints and bounded one-item tuples are compatible with strict mode on first-party non-fine-tuned
    models. Other tuple schemas remain available in non-strict mode. Capturing the live request keeps this assertion
    sensitive to payload drift during cassette playback.
    """
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    model = (
        OpenAIChatModel('gpt-6-luna', provider=provider)
        if case.api == 'chat'
        else OpenAIResponsesModel('gpt-6-luna', provider=provider)
    )

    if case.schema_kind == 'length':
        output_type = LengthOutput
        prompt = "Return a name with the value 'x'."
    elif case.schema_kind == 'single_tuple':
        output_type = SingleTupleOutput
        prompt = "Return values as the one-element array containing the string '7'."
    else:
        output_type = MixedOutput
        prompt = "Return the name 'okay' and values [1, 2, 3, 4]."

    output = ToolOutput(output_type) if case.output_mode == 'tool' else NativeOutput(output_type)
    result = await Agent(model, output_type=output).run(prompt, model_settings={'thinking': False})
    if case.schema_kind == 'length':
        assert isinstance(result.output, LengthOutput)
        assert 2 <= len(result.output.name) <= 5
    elif case.schema_kind == 'single_tuple':
        assert isinstance(result.output, SingleTupleOutput)
        assert result.output.values == (7,)
        assert type(result.output.values[0]) is int
    else:
        assert isinstance(result.output, MixedOutput)
        assert result.output == MixedOutput(name='okay', values=(1, 2, 3, 4))

    path = '/chat/completions' if case.api == 'chat' else '/responses'
    body = request_capture.body(path)
    if case.output_mode == 'tool':
        tools = body['tools']
        assert isinstance(tools, list) and tools
        tool = tools[0]
        assert isinstance(tool, dict)
        if case.api == 'chat':
            function = tool['function']
            assert isinstance(function, dict)
            schema = function['parameters']
            strict = function.get('strict')
        else:
            schema = tool['parameters']
            strict = tool.get('strict')
    elif case.api == 'chat':
        response_format = body['response_format']
        assert isinstance(response_format, dict)
        json_schema = response_format['json_schema']
        assert isinstance(json_schema, dict)
        schema = json_schema['schema']
        strict = json_schema.get('strict')
    else:
        text = body['text']
        assert isinstance(text, dict)
        response_format = text['format']
        assert isinstance(response_format, dict)
        schema = response_format['schema']
        strict = response_format.get('strict')

    assert isinstance(schema, dict)
    if case.schema_kind in ('length', 'single_tuple'):
        assert strict is True
    wire_projection: dict[str, JsonValue] = {'schema': schema, 'strict': strict}
    assert wire_projection == case.expected_wire


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize('output_mode', ['tool', 'native'])
async def test_openai_strict_tuple_guard_runs_before_request(
    api: Literal['chat', 'responses'],
    output_mode: Literal['tool', 'native'],
    allow_model_requests: None,
    openai_api_key: str,
    request_capture: RequestCapture,
):
    """An explicit strict request with tuple output fails before any provider request is sent."""
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    model = (
        OpenAIChatModel('gpt-6-luna', provider=provider)
        if api == 'chat'
        else OpenAIResponsesModel('gpt-6-luna', provider=provider)
    )
    output = ToolOutput(MixedOutput, strict=True) if output_mode == 'tool' else NativeOutput(MixedOutput, strict=True)

    with pytest.raises(UserError) as exc_info:
        await Agent(model, output_type=output).run('Return a tuple.')

    assert str(exc_info.value) == (
        'Pydantic AI cannot use this tuple schema with `prefixItems` in OpenAI strict mode. '
        'Use a bounded list such as `Annotated[list[int], Field(min_length=4, max_length=4)]`, '
        'or set `strict=False`.'
    )
    assert request_capture.paths == []


@dataclass(frozen=True)
class ProfileCase:
    id: str
    model_name: str
    profile_selector: Callable[[str], ModelProfile | None]
    keeps_length_constraints_in_strict_mode: bool
    normalizes_single_item_tuples: bool


PROFILE_CASES: list[ProfileCase] = [
    ProfileCase('openai', 'gpt-4o', OpenAIProvider.model_profile, True, True),
    ProfileCase('openai-finetuned', 'ft:gpt-4o:org::id', OpenAIProvider.model_profile, False, False),
    ProfileCase('chatgpt', 'gpt-5.6-luna', OpenAIChatGPTProvider.model_profile, True, True),
    ProfileCase('shared-profile', 'gpt-4o', openai_model_profile, False, False),
    ProfileCase('azure', 'gpt-4o', AzureProvider.model_profile, False, False),
    ProfileCase('openrouter', 'openai/gpt-4o', OpenRouterProvider.model_profile, False, False),
    ProfileCase('codex', 'gpt-5.6-luna', OpenAICodexProvider.model_profile, False, False),
]


@pytest.mark.parametrize('case', PROFILE_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize('strict', [None, True, False], ids=['inferred', 'explicit-true', 'explicit-false'])
def test_openai_provider_selects_length_schema_compatibility(case: ProfileCase, strict: bool | None):
    """First-party OpenAI infers strict mode with length bounds; shared profiles infer non-strict."""
    profile = case.profile_selector(case.model_name)
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    schema: dict[str, JsonValue] = {
        'type': 'object',
        'properties': {'name': {'type': 'string', 'minLength': 2, 'maxLength': 5}},
        'required': ['name'],
    }
    transformer = transformer_type(schema, strict=strict)
    result = transformer.walk()
    name_schema = result['properties']['name']
    keeps_length_constraints = 'minLength' in name_schema and 'maxLength' in name_schema

    assert keeps_length_constraints is (case.keeps_length_constraints_in_strict_mode or strict is not True)
    if strict is None:
        assert transformer.is_strict_compatible is case.keeps_length_constraints_in_strict_mode


@pytest.mark.parametrize('case', PROFILE_CASES, ids=lambda case: case.id)
@pytest.mark.parametrize('strict', [None, True, False], ids=['inferred', 'explicit-true', 'explicit-false'])
def test_openai_provider_scopes_single_item_tuple_normalization(case: ProfileCase, strict: bool | None):
    """Only first-party non-fine-tuned OpenAI rewrites the bounded one-item tuple schema."""
    profile = case.profile_selector(case.model_name)
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    prefix_item: dict[str, JsonValue] = {'type': 'integer', 'description': 'the sole value'}
    array_schema: dict[str, JsonValue] = {
        'type': 'array',
        'prefixItems': [prefix_item],
        'items': {'type': 'string', 'description': 'unreachable tail metadata'},
        'minItems': 1,
        'maxItems': 1,
    }
    schema: dict[str, JsonValue] = {
        'type': 'object',
        'properties': {'values': array_schema},
        'required': ['values'],
    }
    transformer = transformer_type(schema, strict=strict)

    if case.normalizes_single_item_tuples and strict is not False:
        result = transformer.walk()
        values_schema = result['properties']['values']
        assert values_schema == {
            'type': 'array',
            'items': prefix_item,
            'minItems': 1,
            'maxItems': 1,
        }
        if strict is None:
            assert transformer.is_strict_compatible
    else:
        result = transformer.walk()
        values_schema = result['properties']['values']
        assert values_schema == array_schema
        if strict is not False:
            assert transformer.is_strict_compatible


@pytest.mark.parametrize('strict', [None, True], ids=['inferred', 'explicit-true'])
def test_openai_single_item_tuple_normalization_preserves_prefix_metadata_and_definition_ref(strict: bool | None):
    """The conversion carries the prefix metadata and definition reference into `items`."""
    profile = OpenAIProvider.model_profile('gpt-4o')
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    prefix_item: dict[str, JsonValue] = {
        '$ref': '#/$defs/strict_integer',
        'description': 'the sole value',
    }
    schema: dict[str, JsonValue] = {
        'type': 'array',
        'prefixItems': [prefix_item],
        'items': {'type': 'string'},
        'minItems': 1,
        'maxItems': 1,
        '$defs': {'strict_integer': {'type': 'integer'}},
    }
    transformer = transformer_type(schema, strict=strict)
    normalized = transformer.walk()

    assert normalized == {
        'type': 'array',
        'items': {
            'description': 'the sole value',
            'anyOf': [{'$ref': '#/$defs/strict_integer'}],
        },
        'minItems': 1,
        'maxItems': 1,
        '$defs': {'strict_integer': {'type': 'integer'}},
    }


@pytest.mark.parametrize(
    'tail_schema',
    [
        pytest.param({'type': 'array'}, id='untyped-array'),
        pytest.param(
            {
                'type': 'array',
                'prefixItems': [{'type': 'integer'}, {'type': 'string'}],
                'maxItems': 2,
            },
            id='nested-two-item-tuple',
        ),
        pytest.param({'type': 'string', 'pattern': '(?=x)'}, id='strict-incompatible-pattern'),
        pytest.param({'type': 'string', 'default': 'x'}, id='defaulted-tail'),
    ],
)
@pytest.mark.parametrize('strict', [None, True, False], ids=['inferred', 'explicit-true', 'explicit-false'])
def test_openai_single_item_tuple_normalizes_before_visiting_dead_tail(
    tail_schema: dict[str, JsonValue], strict: bool | None
):
    """A dead tail cannot affect the eligible tuple's strict compatibility or error behavior."""
    profile = OpenAIProvider.model_profile('gpt-4o')
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    array_schema: dict[str, JsonValue] = {
        'type': 'array',
        'prefixItems': [{'type': 'integer'}],
        'items': tail_schema,
        'minItems': 1,
        'maxItems': 1,
    }
    schema: dict[str, JsonValue] = {
        'type': 'object',
        'properties': {'values': array_schema},
        'required': ['values'],
    }
    transformer = transformer_type(schema, strict=strict)

    result = transformer.walk()
    values_schema = result['properties']['values']
    if strict is False:
        assert values_schema == array_schema
    else:
        assert values_schema == {
            'type': 'array',
            'items': {'type': 'integer'},
            'minItems': 1,
            'maxItems': 1,
        }
        if strict is None:
            assert transformer.is_strict_compatible


@pytest.mark.parametrize(
    'array_schema',
    [
        pytest.param({'type': 'array', 'prefixItems': [{'type': 'integer'}]}, id='missing-max-items'),
        pytest.param({'type': 'array', 'prefixItems': [{'type': 'integer'}], 'maxItems': True}, id='boolean-max-items'),
        pytest.param({'type': 'array', 'prefixItems': [{'type': 'integer'}], 'maxItems': 1.0}, id='float-max-items'),
        pytest.param(
            {'type': 'array', 'prefixItems': [{'type': 'integer'}], 'maxItems': 0}, id='unmatched-zero-max-items'
        ),
        pytest.param({'type': 'array', 'prefixItems': [{'type': 'integer'}], 'maxItems': 2}, id='unmatched-max-items'),
        pytest.param(
            {'type': 'array', 'prefixItems': [{'type': 'integer'}, {'type': 'integer'}], 'maxItems': 1},
            id='multiple-prefix-items',
        ),
    ],
)
@pytest.mark.parametrize('strict', [None, True], ids=['inferred', 'explicit-true'])
def test_openai_single_item_tuple_normalization_requires_exact_integer_bound(
    array_schema: dict[str, JsonValue], strict: bool | None
):
    """Other tuple shapes retain the conservative prefixItems guard."""
    profile = OpenAIProvider.model_profile('gpt-4o')
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    schema: dict[str, JsonValue] = {
        'type': 'object',
        'properties': {'values': array_schema},
        'required': ['values'],
    }
    transformer = transformer_type(schema, strict=strict)
    if strict is True:
        with pytest.raises(UserError, match='cannot use this tuple schema with `prefixItems` in OpenAI strict mode'):
            transformer.walk()
    else:
        result = transformer.walk()
        assert result['properties']['values'] == array_schema
        assert not transformer.is_strict_compatible


@pytest.mark.parametrize(
    ('ref', 'ref_target'),
    [
        pytest.param('#/properties/values/prefixItems/0', 'prefixItems', id='removed-prefix-node'),
        pytest.param('#/properties/values/items', 'items', id='overwritten-items-node'),
    ],
)
@pytest.mark.parametrize('field_order', ['ref-first', 'candidate-first'])
@pytest.mark.parametrize('strict', [None, True], ids=['inferred', 'explicit-true'])
def test_openai_single_item_tuple_normalization_keeps_referenced_nodes(
    ref: str, ref_target: str, field_order: str, strict: bool | None
):
    """References into removed or overwritten nodes fall back to the conservative tuple guard."""
    profile = OpenAIProvider.model_profile('gpt-4o')
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    array_schema: dict[str, JsonValue] = {
        'type': 'array',
        'prefixItems': [{'type': 'integer'}],
        'items': {'type': 'string'},
        'minItems': 1,
        'maxItems': 1,
    }
    ref_schema: dict[str, JsonValue] = {'$ref': ref}
    properties: dict[str, JsonValue] = (
        {'head': ref_schema, 'values': array_schema}
        if field_order == 'ref-first'
        else {'values': array_schema, 'head': ref_schema}
    )
    schema: dict[str, JsonValue] = {'type': 'object', 'properties': properties, 'required': list(properties)}
    transformer = transformer_type(schema, strict=strict)

    if strict is True:
        with pytest.raises(UserError, match='cannot use this tuple schema with `prefixItems` in OpenAI strict mode'):
            transformer.walk()
    else:
        result = transformer.walk()
        assert result['properties']['values'] == array_schema
        assert result['properties']['head'] == ref_schema
        expected_target = {'type': 'integer'} if ref_target == 'prefixItems' else {'type': 'string'}
        assert resolve_json_pointer(result, ref) == expected_target
        assert not transformer.is_strict_compatible


@pytest.mark.parametrize(
    ('anchor_keyword', 'reference_keyword', 'ref', 'root_id'),
    [
        pytest.param('$anchor', '$ref', '#tail', None, id='anchor-fragment'),
        pytest.param(
            '$anchor', '$ref', 'https://example.com/schema#tail', 'https://example.com/schema', id='anchor-absolute-uri'
        ),
        pytest.param('$dynamicAnchor', '$dynamicRef', '#tail', None, id='dynamic-anchor-fragment'),
        pytest.param(
            '$dynamicAnchor',
            '$dynamicRef',
            'https://example.com/schema#tail',
            'https://example.com/schema',
            id='dynamic-anchor-absolute-uri',
        ),
    ],
)
@pytest.mark.parametrize('field_order', ['ref-first', 'candidate-first'])
@pytest.mark.parametrize('strict', [None, True], ids=['inferred', 'explicit-true'])
def test_openai_single_item_tuple_normalization_preserves_anchor_references(
    anchor_keyword: str, reference_keyword: str, ref: str, root_id: str | None, field_order: str, strict: bool | None
):
    """Static and dynamic anchor refs into an overwritten tail keep the conservative tuple behavior."""
    profile = OpenAIProvider.model_profile('gpt-4o')
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    array_schema: dict[str, JsonValue] = {
        'type': 'array',
        'prefixItems': [{'type': 'integer'}],
        'items': {'type': 'string', anchor_keyword: 'tail'},
        'minItems': 1,
        'maxItems': 1,
    }
    ref_schema: dict[str, JsonValue] = {reference_keyword: ref}
    properties: dict[str, JsonValue] = (
        {'alias': ref_schema, 'values': array_schema}
        if field_order == 'ref-first'
        else {'values': array_schema, 'alias': ref_schema}
    )
    schema: dict[str, JsonValue] = {'type': 'object', 'properties': properties, 'required': list(properties)}
    if root_id is not None:
        schema['$id'] = root_id
    transformer = transformer_type(schema, strict=strict)

    if strict is True:
        with pytest.raises(UserError, match='cannot use this tuple schema with `prefixItems` in OpenAI strict mode'):
            transformer.walk()
    else:
        result = transformer.walk()
        assert result['properties']['values'] == array_schema
        assert result['properties']['alias'] == ref_schema
        assert result.get('$id') == root_id
        assert not transformer.is_strict_compatible


def test_openai_single_item_tuple_without_conversion_keeps_length_support_with_local_ref():
    """A non-candidate local ref does not disable the private length-constraint profile."""
    profile = OpenAIProvider.model_profile('gpt-4o')
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    schema: dict[str, JsonValue] = {
        'type': 'object',
        'properties': {
            'name': {'type': 'string', 'minLength': 2},
            'alias': {'$ref': '#/properties/name'},
        },
        'required': ['name', 'alias'],
    }
    transformer = transformer_type(schema, strict=True)

    result = transformer.walk()
    assert result['properties']['name'] == {'type': 'string', 'minLength': 2}
    assert result['properties']['alias'] == {'$ref': '#/properties/name'}


def test_openai_single_item_tuple_normalization_is_skipped_for_strict_false():
    """The bounded tuple rewrite only enables inferred or requested strict mode."""
    profile = OpenAIProvider.model_profile('gpt-4o')
    assert profile is not None
    transformer_type = profile.get('json_schema_transformer')
    assert transformer_type is not None

    array_schema: dict[str, JsonValue] = {
        'type': 'array',
        'prefixItems': [{'type': 'integer'}],
        'items': {'type': 'string'},
        'minItems': 1,
        'maxItems': 1,
    }
    schema: dict[str, JsonValue] = {'type': 'object', 'properties': {'values': array_schema}}
    transformer = transformer_type(schema, strict=False)

    result = transformer.walk()
    assert result['properties']['values'] == array_schema

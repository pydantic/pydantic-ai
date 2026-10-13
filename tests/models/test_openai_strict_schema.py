from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field, JsonValue

from pydantic_ai import Agent, ModelProfile, UserError
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.output import NativeOutput, ToolOutput
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.providers.azure import AzureProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider

from .._inline_snapshot import snapshot
from ..conftest import RequestCapture

pytestmark = pytest.mark.vcr


class LengthOutput(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=5)]


class MixedOutput(BaseModel):
    name: Annotated[str, Field(min_length=2, max_length=5)]
    values: tuple[int, int, int, int]


@dataclass(frozen=True)
class WireCase:
    id: str
    schema_kind: Literal['length', 'mixed']
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

    Length constraints are compatible with strict mode on first-party non-fine-tuned models. Tuple schemas remain
    available in non-strict mode, including when combined with length constraints. Capturing the live request keeps
    this assertion sensitive to payload drift during cassette playback.
    """
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    model = (
        OpenAIChatModel('gpt-4o', provider=provider)
        if case.api == 'chat'
        else OpenAIResponsesModel('gpt-4o', provider=provider)
    )

    if case.schema_kind == 'length':
        output_type = LengthOutput
        prompt = "Return a name with the value 'x'."
    else:
        output_type = MixedOutput
        prompt = "Return the name 'okay' and values [1, 2, 3, 4]."

    output = ToolOutput(output_type) if case.output_mode == 'tool' else NativeOutput(output_type)
    result = await Agent(model, output_type=output).run(prompt)
    if case.schema_kind == 'length':
        assert isinstance(result.output, LengthOutput)
        assert 2 <= len(result.output.name) <= 5
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
        OpenAIChatModel('gpt-4o', provider=provider)
        if api == 'chat'
        else OpenAIResponsesModel('gpt-4o', provider=provider)
    )
    output = ToolOutput(MixedOutput, strict=True) if output_mode == 'tool' else NativeOutput(MixedOutput, strict=True)

    with pytest.raises(UserError, match='OpenAI strict mode does not support tuple types'):
        await Agent(model, output_type=output).run('Return a tuple.')

    assert request_capture.paths == []


@dataclass(frozen=True)
class ProfileCase:
    id: str
    model_name: str
    profile_selector: Callable[[str], ModelProfile | None]
    keeps_length_constraints_in_strict_mode: bool


PROFILE_CASES: list[ProfileCase] = [
    ProfileCase('openai', 'gpt-4o', OpenAIProvider.model_profile, True),
    ProfileCase('openai-finetuned', 'ft:gpt-4o:org::id', OpenAIProvider.model_profile, False),
    ProfileCase('shared-profile', 'gpt-4o', openai_model_profile, False),
    ProfileCase('azure', 'gpt-4o', AzureProvider.model_profile, False),
    ProfileCase('openrouter', 'openai/gpt-4o', OpenRouterProvider.model_profile, False),
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

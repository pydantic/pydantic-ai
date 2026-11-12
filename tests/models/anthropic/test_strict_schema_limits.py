"""Tests for Anthropic's explicit complexity limits on the strict schemas in one request.

Anthropic compiles the strict schemas of a request (tools with `strict=True`, and the Native Output schema) into a
grammar, and rejects a request whose strict schemas together exceed its explicit limits:
https://platform.claude.com/docs/en/build-with-claude/structured-outputs#schema-complexity-limits
"""

from __future__ import annotations as _annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, create_model

from pydantic_ai import Agent, Tool
from pydantic_ai.exceptions import ModelHTTPError, ModelRequestRejectedError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models._anthropic_schema_limits import exceeded_limit, is_schema_too_complex_error
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import NativeOutput, OutputObjectDefinition, ToolOutput
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.anthropic import AnthropicModelProfile
from pydantic_ai.settings import ModelSettings, ToolOrOutput

from ..._inline_snapshot import snapshot
from ...conftest import RequestCapture, try_import
from ..test_anthropic import MockAnthropic, completion_message

with try_import() as imports_successful:
    from anthropic import (
        AsyncAnthropic,
        AsyncAnthropicBedrock,
        AsyncAnthropicBedrockMantle,
        AsyncAnthropicFoundry,
        AsyncAnthropicVertex,
    )
    from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaToolUseBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

if TYPE_CHECKING:
    ANTHROPIC_MODEL_FIXTURE = Callable[..., AnthropicModel]

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.vcr,
]

_PROMPT = 'Fill in a survey answer with `answer_0` set to 1 and every other field left empty.'


def _model(name: str, fields: dict[str, Any]) -> type[BaseModel]:
    return create_model(name, **fields)


# 17 fields of type `int | None`, one more union-typed parameter than Anthropic allows in strict schemas.
Survey = _model('Survey', {f'answer_{i}': (int | None, None) for i in range(17)})


class Node(BaseModel):
    """A tree, whose schema is recursive."""

    name: str
    children: list[Node]


def fill(a: str = '', b: str = '', c: str = '', d: str = '') -> str:
    """Fill in up to four fields."""
    return a + b + c + d  # pragma: no cover


def _strict_tool(index: int) -> Tool[Any]:
    def lookup(key: str) -> str:
        return key  # pragma: no cover

    return Tool(lookup, name=f'lookup_{index}', strict=True)


def _request_shape(body: dict[str, Any]) -> dict[str, Any]:
    output_config = body.get('output_config')
    return {
        'tool_choice': body.get('tool_choice'),
        'output_format': isinstance(output_config, dict) and 'format' in output_config,
        'tools': [(tool['name'], tool.get('strict')) for tool in body.get('tools', [])],
    }


@dataclass
class StepDownCase:
    model_name: str
    model_settings: ModelSettings
    client: str = 'anthropic'


STEP_DOWN_CASES = {
    # Extended thinking rejects a forced tool choice.
    'haiku-4-5-extended-thinking': StepDownCase('claude-haiku-4-5', {'thinking': True}),
    # Thinks by default, adaptively; forcing the output tool would stop it from thinking.
    'sonnet-5-default': StepDownCase('claude-sonnet-5', {}),
    'bedrock-invoke-haiku-4-5': StepDownCase(
        'us.anthropic.claude-haiku-4-5-20251001-v1:0', {'thinking': True}, client='bedrock'
    ),
}


@pytest.mark.parametrize('case', STEP_DOWN_CASES.values(), ids=STEP_DOWN_CASES.keys())
async def test_default_output_over_limits_uses_non_strict_tool_output(
    allow_model_requests: None,
    anthropic_model: ANTHROPIC_MODEL_FIXTURE,
    request: pytest.FixtureRequest,
    request_capture: RequestCapture,
    case: StepDownCase,
):
    """A bare `output_type` on a request that thinks uses Native Output, unless its schema is over Anthropic's limits.

    Native Output is always strict, and Anthropic rejects these schemas in strict mode, so the output tool is sent
    instead, without `strict` (which doesn't count toward the limits) and without forcing the call, as the request
    thinks.
    """
    if case.client == 'bedrock':
        client = request.getfixturevalue('anthropic_bedrock_client')
        model = AnthropicModel(case.model_name, provider=AnthropicProvider(anthropic_client=client))
    else:
        model = anthropic_model(case.model_name, capture=True)

    result = await Agent(model, output_type=Survey).run(_PROMPT, model_settings=case.model_settings)

    assert isinstance(result.output, Survey)
    assert result.output.model_dump(exclude_none=True) == {'answer_0': 1}
    assert _request_shape(request_capture.bodies()[0]) == snapshot(
        {'tool_choice': {'type': 'auto'}, 'output_format': False, 'tools': [('final_result', None)]}
    )
    assert [
        part.tool_name
        for message in result.all_messages()
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ] == ['final_result']


@dataclass
class ExplicitStrictCase:
    output_type: Any
    tools: list[Tool[Any]]
    error: str


EXPLICIT_STRICT_CASES = {
    'native-output-unions': ExplicitStrictCase(NativeOutput(Survey), [], '17 parameters with union types'),
    'strict-tool-output-unions': ExplicitStrictCase(
        ToolOutput(Survey, strict=True), [], '17 parameters with union types'
    ),
    'native-output-recursive': ExplicitStrictCase(
        NativeOutput(Node), [], 'a recursive schema, which strict mode does not support'
    ),
    # Counted across all strict schemas: 21 optional parameters in the output and 4 in the strict tool's.
    'native-output-and-strict-tool-optional': ExplicitStrictCase(
        NativeOutput(_model('Form', {f'field_{i}': (str, '') for i in range(21)})),
        [Tool(fill, strict=True)],
        '25 optional parameters',
    ),
    # Stepping the default output mode down doesn't help when the strict function tools alone are over a limit.
    'default-output-strict-tools': ExplicitStrictCase(
        Survey, [_strict_tool(i) for i in range(21)], '21 strict tools, over the limit of 20'
    ),
    'text-output-strict-tools': ExplicitStrictCase(
        str, [_strict_tool(i) for i in range(21)], '21 strict tools, over the limit of 20'
    ),
}


@pytest.mark.parametrize('case', EXPLICIT_STRICT_CASES.values(), ids=EXPLICIT_STRICT_CASES.keys())
async def test_explicit_strict_over_limits_raises_before_sending(
    allow_model_requests: None, anthropic_model: ANTHROPIC_MODEL_FIXTURE, case: ExplicitStrictCase
):
    """Strict schemas the user asked for that Anthropic would reject raise a `ModelRequestRejectedError` rather than a 400.

    The error is raised before a request is sent, which is why there's no cassette.
    """
    agent = Agent(anthropic_model('claude-sonnet-4-5'), output_type=case.output_type, tools=case.tools)

    with pytest.raises(ModelRequestRejectedError, match=case.error) as exc_info:
        await agent.run(_PROMPT, model_settings={'thinking': True})

    assert 'https://pydantic.dev/docs/ai/models/anthropic/#schema-complexity-limits' in str(exc_info.value)


async def test_explicit_strict_over_limits_falls_back(
    allow_model_requests: None, anthropic_model: ANTHROPIC_MODEL_FIXTURE
):
    """`FallbackModel` falls back on the pre-send error, as it does on the 400 Anthropic would send.

    A unit test: the Anthropic request is refused before it's sent, so there's nothing to record.
    """

    def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart('{}')])

    other = FunctionModel(answer, profile=ModelProfile(supports_json_schema_output=True))
    fallback = FallbackModel(anthropic_model('claude-sonnet-4-5'), other)

    result = await Agent(fallback, output_type=NativeOutput(Survey)).run(_PROMPT)

    assert isinstance(result.output, Survey)
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert response.model_name == other.model_name


def _mock_model(*responses: BetaMessage) -> AnthropicModel:
    return AnthropicModel(
        'claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=MockAnthropic.create_mock(list(responses)))
    )


async def test_strict_tools_a_tool_choice_leaves_out_dont_count(allow_model_requests: None):
    """Strict tools a `tool_choice` keeps out of the request don't count.

    A unit test, since what's under test is that the request goes out at all.
    """
    model = _mock_model(
        completion_message([BetaTextBlock(text='Done', type='text')], BetaUsage(input_tokens=5, output_tokens=1))
    )
    agent = Agent(model, tools=[_strict_tool(i) for i in range(21)])

    result = await agent.run('Say done.', model_settings={'tool_choice': ToolOrOutput(function_tools=['lookup_0'])})

    assert result.output == 'Done'


async def test_strict_tools_count_under_tool_choice_none(allow_model_requests: None):
    """`tool_choice='none'` still sends the tools, and Anthropic counts them: 21 strict tools got a 400 live.

    A unit test: the error is raised before a request is sent, so there's nothing to record.
    """
    agent = Agent(_mock_model(), tools=[_strict_tool(i) for i in range(21)])

    with pytest.raises(ModelRequestRejectedError, match='21 strict tools, over the limit of 20'):
        await agent.run('Say done.', model_settings={'tool_choice': 'none'})


async def test_default_output_over_limits_doesnt_warn_about_strict_mode(allow_model_requests: None):
    """Deciding against Native Output doesn't warn about the `dict` field, which only strict mode can't represent.

    A unit test, as the warning would come from transforming the schema before any request is sent.
    """
    model = _mock_model(
        completion_message(
            [BetaToolUseBlock(id='call_1', name='final_result', input={'counts': {'a': 1}}, type='tool_use')],
            BetaUsage(input_tokens=5, output_tokens=10),
        )
    )
    inventory = _model(
        'Inventory', {'counts': (dict[str, int], ...), **{f'answer_{i}': (int | None, None) for i in range(17)}}
    )

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = await Agent(model, output_type=inventory).run('Count.', model_settings={'thinking': True})

    assert result.output.model_dump(exclude_none=True) == {'counts': {'a': 1}}


@pytest.mark.parametrize(
    ('client', 'enforced'),
    [
        pytest.param(lambda: AsyncAnthropic(api_key='test'), True, id='anthropic'),
        pytest.param(lambda: AsyncAnthropicBedrock(api_key='test', aws_region='us-east-1'), True, id='bedrock-invoke'),
        pytest.param(
            lambda: AsyncAnthropicBedrockMantle(api_key='test', aws_region='us-east-1'), True, id='bedrock-mantle'
        ),
        pytest.param(
            lambda: AsyncAnthropicVertex(project_id='test', region='us-east5', access_token='test'),
            False,
            id='vertex',
        ),
        pytest.param(lambda: AsyncAnthropicFoundry(api_key='test', resource='test'), False, id='foundry'),
    ],
)
def test_strict_schema_limits_per_client(client: Callable[[], Any], enforced: bool):
    """The limits are checked only on the transports they were verified on; elsewhere the request is left to the API.

    A unit test, since what's under test is whether preparing an over-limit `NativeOutput` request raises, and our
    credentials don't reach Claude on Vertex or Foundry.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client()))
    params = ModelRequestParameters(
        output_mode='native', output_object=OutputObjectDefinition(json_schema=Survey.model_json_schema())
    )

    assert model.profile.get('anthropic_enforces_strict_schema_limits') is enforced
    if enforced:
        with pytest.raises(ModelRequestRejectedError, match='17 parameters with union types'):
            model.prepare_request(None, params)
    else:
        _, prepared = model.prepare_request(None, params)
        assert prepared.output_mode == 'native'


def test_unverified_client_can_opt_in_to_strict_schema_limits():
    """A Vertex or Foundry user whose deployment enforces the limits can opt in through the model's `profile`.

    A unit test, as it checks profile resolution, which sends no request.
    """
    client = AsyncAnthropicVertex(project_id='test', region='us-east5', access_token='test')
    model = AnthropicModel(
        'claude-sonnet-4-5',
        provider=AnthropicProvider(anthropic_client=client),
        profile=AnthropicModelProfile(anthropic_enforces_strict_schema_limits=True),
    )

    assert model.profile.get('anthropic_enforces_strict_schema_limits') is True


async def test_schema_too_complex_error_has_hint(allow_model_requests: None, anthropic_model: ANTHROPIC_MODEL_FIXTURE):
    """A schema within the explicit limits can still be too complex for Anthropic to compile.

    Anthropic answers 13 optional parameters in one output object this way, although its explicit limit is 24.
    The 400 stays a `ModelHTTPError` and gains a hint pointing at the docs.
    """
    form = _model('Form', {f'field_{i}': (str, '') for i in range(13)})
    agent = Agent(anthropic_model('claude-haiku-4-5'), output_type=NativeOutput(form))

    with pytest.raises(ModelHTTPError) as exc_info:
        await agent.run('Fill in the form.')

    error = exc_info.value
    assert isinstance(error, ModelRequestRejectedError)
    assert error.status_code == 400
    assert error.hint == snapshot(
        "Anthropic couldn't compile the request's strict schemas into a grammar. Simplify them, set `strict=True` "
        'only on the tools that need it, or use `ToolOutput` without `strict=True` or `PromptedOutput` for a large '
        'output type. See https://pydantic.dev/docs/ai/models/anthropic/#schema-complexity-limits'
    )


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {'type': 'object', 'properties': properties, 'required': required, 'additionalProperties': False}


def _optional(count: int, prefix: str = 'o') -> dict[str, Any]:
    return _object({f'{prefix}{i}': {'type': 'string'} for i in range(count)}, [])


def _unions(count: int, prefix: str = 'u') -> dict[str, Any]:
    names = [f'{prefix}{i}' for i in range(count)]
    return _object({name: {'anyOf': [{'type': 'string'}, {'type': 'integer'}]} for name in names}, names)


_LOOKUP = _object({'key': {'type': 'string'}}, ['key'])


@dataclass
class CountCase:
    tools: list[dict[str, Any]]
    output: dict[str, Any] | None
    exceeded: str | None


# Each schema was sent to the Messages API in strict mode on 2026-10-26, and the API's own count decided `exceeded`.
COUNT_CASES = {
    'optional-at-limit': CountCase([], _optional(24), None),
    'optional-over-limit': CountCase([], _optional(25), '25 optional parameters, over the limit of 24'),
    'unions-at-limit': CountCase([], _unions(16), None),
    'unions-over-limit': CountCase([], _unions(17), '17 parameters with union types, over the limit of 16'),
    'type-arrays-count-as-unions': CountCase(
        [],
        _object({f'u{i}': {'type': ['string', 'null']} for i in range(17)}, [f'u{i}' for i in range(17)]),
        '17 parameters with union types, over the limit of 16',
    ),
    'nullable-optional-counts-as-both': CountCase(
        [],
        _object(
            {
                **{f'n{i}': {'anyOf': [{'type': 'string'}, {'type': 'null'}]} for i in range(16)},
                **{f'o{i}': {'type': 'string'} for i in range(9)},
            },
            [],
        ),
        '25 optional parameters, over the limit of 24',
    ),
    'nested-objects-count': CountCase(
        [],
        _object({**{f'o{i}': {'type': 'string'} for i in range(13)}, 'nested': _optional(13, 'p')}, ['nested']),
        '26 optional parameters, over the limit of 24',
    ),
    'array-items-objects-count': CountCase(
        [],
        _object({'items': {'type': 'array', 'items': _unions(17)}}, ['items']),
        '17 parameters with union types, over the limit of 16',
    ),
    'union-branches-count': CountCase(
        [],
        _object({'x': {'anyOf': [_unions(16), {'type': 'null'}]}}, ['x']),
        '17 parameters with union types, over the limit of 16',
    ),
    'union-array-items-dont-count': CountCase(
        [],
        _object(
            {
                f'a{i}': {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}
                for i in range(17)
            },
            [f'a{i}' for i in range(17)],
        ),
        None,
    ),
    'enums-dont-count': CountCase(
        [],
        _object({f'e{i}': {'type': 'string', 'enum': ['a', 'b']} for i in range(20)}, [f'e{i}' for i in range(20)]),
        None,
    ),
    'all-of-doesnt-count': CountCase(
        [],
        _object({f'u{i}': {'allOf': [{'type': 'string'}]} for i in range(17)}, [f'u{i}' for i in range(17)]),
        None,
    ),
    'definitions-dont-count': CountCase(
        [], {**_object({'a': {'$ref': '#/$defs/D'}}, ['a']), '$defs': {'D': _unions(17, 'x')}}, None
    ),
    'reference-to-union-doesnt-count': CountCase(
        [],
        {
            **_object({f'r{i}': {'$ref': '#/$defs/U'} for i in range(17)}, [f'r{i}' for i in range(17)]),
            '$defs': {'U': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}},
        },
        None,
    ),
    'strict-tools-at-limit': CountCase([_LOOKUP] * 20, None, None),
    'strict-tools-over-limit': CountCase([_LOOKUP] * 21, None, '21 strict tools, over the limit of 20'),
    'counted-across-schemas': CountCase([_optional(13)], _optional(12), '25 optional parameters, over the limit of 24'),
    'recursive-definition': CountCase(
        [],
        {
            **_object({'root': {'$ref': '#/$defs/N'}}, ['root']),
            '$defs': {'N': _object({'children': {'type': 'array', 'items': {'$ref': '#/$defs/N'}}}, ['children'])},
        },
        'a recursive schema, which strict mode does not support',
    ),
    # Not sent live: the other ways a schema can refer back to itself.
    'recursive-root': CountCase(
        [_object({'children': {'type': 'array', 'items': {'$ref': '#'}}}, ['children'])],
        None,
        'a recursive schema, which strict mode does not support',
    ),
    'recursive-draft-07-definitions': CountCase(
        [],
        {
            **_object({'a': {'$ref': '#/definitions/A'}, 'b': {'$ref': '#/definitions/A'}}, ['a', 'b']),
            'definitions': {
                'A': _object({'b': {'$ref': '#/definitions/B'}}, ['b']),
                'B': _object({'a': {'anyOf': [{'$ref': '#/definitions/A'}, {'type': 'null'}]}}, ['a']),
            },
        },
        'a recursive schema, which strict mode does not support',
    ),
    'boolean-subschemas': CountCase([], _object({'a': True, 'b': {'anyOf': [True, {'type': 'string'}]}}, []), None),
    'pointer-reference-isnt-followed': CountCase(
        [], _object({'a': _optional(1), 'b': {'$ref': '#/properties/a'}}, ['a', 'b']), None
    ),
    'shared-definition-isnt-recursive': CountCase(
        [],
        {
            **_object({'a': {'$ref': '#/$defs/D'}, 'b': {'$ref': '#/$defs/D'}, 'c': {'$ref': '#/$defs/E'}}, []),
            '$defs': {'D': _optional(1), 'E': _object({'d': {'$ref': '#/$defs/D'}}, ['d'])},
        },
        None,
    ),
}


@pytest.mark.parametrize('case', COUNT_CASES.values(), ids=COUNT_CASES.keys())
def test_exceeded_limit(case: CountCase):
    """Count what Anthropic counts, which was established by sending each schema to the API.

    A unit test, as the cases pin the counting rules themselves.
    """
    assert exceeded_limit(case.tools, case.output) == case.exceeded


@pytest.mark.parametrize(
    ('message', 'expected'),
    [
        ('Schema is too complex.', True),
        ('The compiled grammar is too large, which would cause performance issues.', True),
        ('The model returned the following errors: Schema is too complex.', True),
        ('Schemas contains too many optional parameters (25)', False),
        (None, False),
    ],
)
def test_is_schema_too_complex_error(message: object, expected: bool):
    """Recognize both of Anthropic's messages for schemas too complex to compile, also as Bedrock wraps them.

    A unit test, as it pins message matching; the hinted 400s themselves are recorded in the cassettes above.
    """
    assert is_schema_too_complex_error(message) is expected

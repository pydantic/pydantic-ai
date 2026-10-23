from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Literal

import httpx2
import pytest
from pydantic import Field, JsonValue
from typing_extensions import TypeAliasType

from pydantic_ai import Agent, NativeOutput

from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@dataclass
class Replacement:
    old_text: str
    new_text: str


@dataclass
class Edits:
    replacements: list[Replacement]


@dataclass
class Node:
    value: str
    children: list[Node] | None = None


Key = TypeAliasType('Key', str)
Positive = TypeAliasType('Positive', Annotated[int, Field(gt=0)])


def edit_file(path: str, replacements: list[Replacement] | None = None) -> str:
    """Edit a file by exact string replacement."""
    return 'ok'  # pragma: no cover


def accept_keys(values: dict[Key, int]) -> str:
    return 'ok'  # pragma: no cover


def add_contains(schema: dict[str, JsonValue]) -> None:
    items = schema['items']
    assert isinstance(items, dict)
    options = items['anyOf']
    assert isinstance(options, list)
    schema['contains'] = deepcopy(options[0])


def accept_contains(values: Annotated[list[Positive | str], Field(json_schema_extra=add_contains)]) -> str:
    return 'ok'  # pragma: no cover


def accept_node(node: Node) -> str:
    return 'ok'  # pragma: no cover


_REPLACEMENT_SCHEMA: dict[str, JsonValue] = {
    'type': 'object',
    'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
    'required': ['old_text', 'new_text'],
    'additionalProperties': False,
}
_TOOL_SCHEMAS: dict[tuple[bool, bool | None], dict[str, JsonValue]] = {}
for _inlined in (False, True):
    for _strict in (None, True, False):
        _replacement = deepcopy(_REPLACEMENT_SCHEMA)
        if _strict is False:
            _replacement.pop('additionalProperties')
        _items: dict[str, JsonValue] = _replacement if _inlined else {'$ref': '#/$defs/Replacement'}
        _replacements: dict[str, JsonValue] = {
            'anyOf': [{'type': 'array', 'items': _items}, {'type': 'null'}],
        }
        if _strict is not True:
            _replacements['default'] = None
        _parameters: dict[str, JsonValue] = {
            'type': 'object',
            'properties': {'path': {'type': 'string'}, 'replacements': _replacements},
            'required': ['path', 'replacements'] if _strict else ['path'],
            'additionalProperties': False,
        }
        if not _inlined:
            _parameters['$defs'] = {'Replacement': _replacement}
        _TOOL_SCHEMAS[(_inlined, _strict)] = _parameters

_INLINED_SCHEMAS: dict[bool | None, dict[str, JsonValue]] = {
    None: _TOOL_SCHEMAS[(True, None)],
    True: _TOOL_SCHEMAS[(True, True)],
    False: _TOOL_SCHEMAS[(True, False)],
}
_REFERENCED_SCHEMAS: dict[bool | None, dict[str, JsonValue]] = {
    None: _TOOL_SCHEMAS[(False, None)],
    True: _TOOL_SCHEMAS[(False, True)],
    False: _TOOL_SCHEMAS[(False, False)],
}
_NATIVE_SCHEMA: dict[str, JsonValue] = {
    'type': 'object',
    'properties': {'replacements': {'type': 'array', 'items': {'$ref': '#/$defs/Replacement'}}},
    'required': ['replacements'],
    'additionalProperties': False,
    '$defs': {'Replacement': _REPLACEMENT_SCHEMA},
}
_NATIVE_SCHEMAS: dict[bool | None, dict[str, JsonValue]] = {
    None: _NATIVE_SCHEMA,
    True: _NATIVE_SCHEMA,
    False: {
        'type': 'object',
        'properties': {'replacements': {'type': 'array', 'items': {'$ref': '#/$defs/Replacement'}}},
        'required': ['replacements'],
        '$defs': {
            'Replacement': {key: value for key, value in _REPLACEMENT_SCHEMA.items() if key != 'additionalProperties'}
        },
    },
}
_TOOL_STRICT_FIELDS: dict[tuple[str, bool | None], tuple[bool, bool | None]] = {
    ('chat', None): (False, None),
    ('chat', True): (True, True),
    ('chat', False): (False, None),
    ('responses', None): (True, False),
    ('responses', True): (True, True),
    ('responses', False): (True, False),
}
_NATIVE_STRICT_FLAGS: dict[bool | None, bool] = {None: True, True: True, False: False}

_KEY_SCHEMAS: dict[bool, dict[bool | None, dict[str, JsonValue]]] = {False: {}, True: {}}
_CONTAINS_SCHEMAS: dict[bool, dict[bool | None, dict[str, JsonValue]]] = {False: {}, True: {}}
for _inlined in (False, True):
    for _strict in (None, True, False):
        _key: dict[str, JsonValue] = {'type': 'string'} if _inlined else {'$ref': '#/$defs/Key'}
        _values: dict[str, JsonValue] = {
            'type': 'object',
            'additionalProperties': {'type': 'integer'},
            'propertyNames': _key,
            'properties': {},
        }
        if _strict is True:
            _values.pop('propertyNames')
            _values['description'] = f'propertyNames={_key}'
            _values['additionalProperties'] = False
            _values['required'] = []
        _keys: dict[str, JsonValue] = {
            'type': 'object',
            'properties': {'values': _values},
            'required': ['values'],
            'additionalProperties': False,
        }
        if not _inlined:
            _keys['$defs'] = {'Key': {'type': 'string'}}
        _KEY_SCHEMAS[_inlined][_strict] = _keys

        _positive: dict[str, JsonValue] = {'exclusiveMinimum': 0, 'type': 'integer'}
        _positive_ref: dict[str, JsonValue] = _positive if _inlined else {'$ref': '#/$defs/Positive'}
        _array: dict[str, JsonValue] = {
            'type': 'array',
            'items': {'anyOf': [_positive_ref, {'type': 'string'}]},
            'contains': _positive_ref,
        }
        if _strict is True:
            _array.pop('contains')
            _array['description'] = f'contains={_positive_ref}'
        _contains: dict[str, JsonValue] = {
            'type': 'object',
            'properties': {'values': _array},
            'required': ['values'],
            'additionalProperties': False,
        }
        if not _inlined:
            _contains['$defs'] = {'Positive': _positive}
        _CONTAINS_SCHEMAS[_inlined][_strict] = _contains

_NODE_SCHEMAS: dict[bool | None, dict[str, JsonValue]] = {}
for _strict in (None, True, False):
    _children: dict[str, JsonValue] = {
        'anyOf': [{'type': 'array', 'items': {'$ref': '#'}}, {'type': 'null'}],
    }
    if _strict is not True:
        _children['default'] = None
    _node: dict[str, JsonValue] = {
        'type': 'object',
        'properties': {'value': {'type': 'string'}, 'children': _children},
        'required': ['value', 'children'] if _strict else ['value'],
    }
    if _strict is not False:
        _node['additionalProperties'] = False
    _node_root = deepcopy(_node)
    _node_root['$defs'] = {'Node': _node}
    _NODE_SCHEMAS[_strict] = _node_root
_EDITS_OUTPUT = Edits([Replacement('old', 'new')])
_EDITS_CONTENT = '{"replacements": [{"old_text": "old", "new_text": "new"}]}'


@dataclass(frozen=True)
class Case:
    id: str
    tool_schemas: dict[bool | None, dict[str, JsonValue]]
    provider_name: Literal['azure', 'azure-custom', 'openai'] = 'azure'
    model_name: str = 'grok-4-1-fast-reasoning'
    function: Callable[..., str] = edit_file
    native_schema: dict[bool | None, dict[str, JsonValue]] | None = None
    expected_output: str | Edits = 'hello'
    response_content: str = 'hello'


CASES: list[Case] = [
    Case('azure-grok', _INLINED_SCHEMAS),
    Case('azure-grok-mixed-case', _INLINED_SCHEMAS, model_name='Grok-4-1-fast-reasoning'),
    Case('azure-openai', _REFERENCED_SCHEMAS, model_name='gpt-4o'),
    Case('azure-deepseek', _REFERENCED_SCHEMAS, model_name='DeepSeek-R1'),
    Case('generic-openai', _REFERENCED_SCHEMAS, provider_name='openai'),
    Case(
        'azure-grok-native',
        _INLINED_SCHEMAS,
        native_schema=_NATIVE_SCHEMAS,
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case(
        'azure-openai-native',
        _REFERENCED_SCHEMAS,
        model_name='gpt-4o',
        native_schema=_NATIVE_SCHEMAS,
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case(
        'generic-openai-native',
        _REFERENCED_SCHEMAS,
        provider_name='openai',
        native_schema=_NATIVE_SCHEMAS,
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case(
        'azure-custom-profile-native',
        _REFERENCED_SCHEMAS,
        provider_name='azure-custom',
        native_schema=_NATIVE_SCHEMAS,
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case('azure-grok-property-names', _KEY_SCHEMAS[True], function=accept_keys),
    Case('generic-openai-property-names', _KEY_SCHEMAS[False], provider_name='openai', function=accept_keys),
    Case('azure-grok-contains', _CONTAINS_SCHEMAS[True], function=accept_contains),
    Case('generic-openai-contains', _CONTAINS_SCHEMAS[False], provider_name='openai', function=accept_contains),
    Case('azure-grok-recursive', _NODE_SCHEMAS, function=accept_node),
    Case('generic-openai-recursive', _NODE_SCHEMAS, provider_name='openai', function=accept_node),
]


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize('strict', [None, True, False])
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_azure_grok_tool_schema(
    allow_model_requests: None,
    request_capture: RequestCapture,
    api: Literal['chat', 'responses'],
    strict: bool | None,
    case: Case,
):
    """Capture the SDK's actual request; Azure acceptance is reporter-verified, not simulated here."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if api == 'chat':
            return httpx2.Response(
                200,
                json={
                    'id': 'chatcmpl-test',
                    'object': 'chat.completion',
                    'created': 0,
                    'model': case.model_name,
                    'choices': [
                        {
                            'index': 0,
                            'message': {'role': 'assistant', 'content': case.response_content},
                            'finish_reason': 'stop',
                        }
                    ],
                },
            )
        return httpx2.Response(
            200,
            json={
                'id': 'resp-test',
                'object': 'response',
                'created_at': 0,
                'model': case.model_name,
                'status': 'completed',
                'output': [
                    {
                        'id': 'msg-test',
                        'type': 'message',
                        'role': 'assistant',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': case.response_content, 'annotations': []}],
                    }
                ],
            },
        )

    async with httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        event_hooks={'request': list(request_capture.client.event_hooks['request'])},
    ) as client:
        endpoint = 'https://example.services.ai.azure.com/openai/v1'
        provider = (
            AzureProvider(azure_endpoint=endpoint, api_key='test', http_client=client)
            if case.provider_name != 'openai'
            else OpenAIProvider(base_url=endpoint, api_key='test', http_client=client)
        )
        profile = (
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)
            if case.provider_name == 'azure-custom'
            else None
        )
        model = (OpenAIChatModel if api == 'chat' else OpenAIResponsesModel)(
            case.model_name, provider=provider, profile=profile
        )
        agent = Agent(model, output_type=NativeOutput(Edits, strict=strict)) if case.native_schema else Agent(model)

        agent.tool_plain(strict=strict)(case.function)

        assert (await agent.run('Say hi.')).output == case.expected_output

    body = request_capture.body('/chat/completions' if api == 'chat' else '/responses')
    tools = body['tools']
    assert isinstance(tools, list)
    tool = tools[0]
    assert isinstance(tool, dict)
    if api == 'chat':
        tool = tool['function']
        assert isinstance(tool, dict)
    assert tool['parameters'] == case.tool_schemas[strict]
    assert ('strict' in tool, tool.get('strict')) == _TOOL_STRICT_FIELDS[(api, strict)]

    if case.native_schema is not None:
        output_format = body['response_format'] if api == 'chat' else body['text']
        assert isinstance(output_format, dict)
        output_format = output_format['json_schema'] if api == 'chat' else output_format['format']
        assert isinstance(output_format, dict)
        assert output_format['schema'] == case.native_schema[strict]
        assert output_format['strict'] is _NATIVE_STRICT_FLAGS[strict]

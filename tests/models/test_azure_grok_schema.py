from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx2
import pytest
from pydantic import JsonValue

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


_REPLACEMENT_SCHEMA: dict[str, JsonValue] = {
    'type': 'object',
    'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
    'required': ['old_text', 'new_text'],
    'additionalProperties': False,
}
_TOOL_SCHEMAS: dict[tuple[bool, bool | None], dict[str, JsonValue]] = {}
for _inlined in (False, True):
    for _strict in (None, True):
        _items: dict[str, JsonValue] = _REPLACEMENT_SCHEMA if _inlined else {'$ref': '#/$defs/Replacement'}
        _replacements: dict[str, JsonValue] = {
            'anyOf': [{'type': 'array', 'items': _items}, {'type': 'null'}],
        }
        if _strict is None:
            _replacements['default'] = None
        _parameters: dict[str, JsonValue] = {
            'type': 'object',
            'properties': {'path': {'type': 'string'}, 'replacements': _replacements},
            'required': ['path', 'replacements'] if _strict else ['path'],
            'additionalProperties': False,
        }
        if not _inlined:
            _parameters['$defs'] = {'Replacement': _REPLACEMENT_SCHEMA}
        _TOOL_SCHEMAS[(_inlined, _strict)] = _parameters

_INLINED_SCHEMAS: dict[bool | None, dict[str, JsonValue]] = {
    None: _TOOL_SCHEMAS[(True, None)],
    True: _TOOL_SCHEMAS[(True, True)],
}
_REFERENCED_SCHEMAS: dict[bool | None, dict[str, JsonValue]] = {
    None: _TOOL_SCHEMAS[(False, None)],
    True: _TOOL_SCHEMAS[(False, True)],
}
_NATIVE_SCHEMA: dict[str, JsonValue] = {
    'type': 'object',
    'properties': {'replacements': {'type': 'array', 'items': {'$ref': '#/$defs/Replacement'}}},
    'required': ['replacements'],
    'additionalProperties': False,
    '$defs': {'Replacement': _REPLACEMENT_SCHEMA},
}
_STRICT_FLAGS: dict[bool | None, bool] = {None: False, True: True}
_EDITS_OUTPUT = Edits([Replacement('old', 'new')])
_EDITS_CONTENT = '{"replacements": [{"old_text": "old", "new_text": "new"}]}'


@dataclass(frozen=True)
class Case:
    id: str
    tool_schemas: dict[bool | None, dict[str, JsonValue]]
    provider_name: Literal['azure', 'azure-custom', 'openai'] = 'azure'
    model_name: str = 'grok-4-1-fast-reasoning'
    native_schema: dict[str, JsonValue] | None = None
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
        native_schema=_NATIVE_SCHEMA,
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case(
        'azure-openai-native',
        _REFERENCED_SCHEMAS,
        model_name='gpt-4o',
        native_schema=_NATIVE_SCHEMA,
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case(
        'generic-openai-native',
        _REFERENCED_SCHEMAS,
        provider_name='openai',
        native_schema=_NATIVE_SCHEMA,
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
    Case(
        'azure-custom-profile-native',
        _REFERENCED_SCHEMAS,
        provider_name='azure-custom',
        native_schema=_NATIVE_SCHEMA,
        expected_output=_EDITS_OUTPUT,
        response_content=_EDITS_CONTENT,
    ),
]


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize('strict', [None, True])
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

        @agent.tool_plain(strict=strict)
        def edit_file(path: str, replacements: list[Replacement] | None = None) -> str:
            """Edit a file by exact string replacement."""
            return 'ok'  # pragma: no cover

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
    assert tool.get('strict', False) == _STRICT_FLAGS[strict]

    if case.native_schema is not None:
        output_format = body['response_format'] if api == 'chat' else body['text']
        assert isinstance(output_format, dict)
        output_format = output_format['json_schema'] if api == 'chat' else output_format['format']
        assert isinstance(output_format, dict)
        assert output_format['schema'] == case.native_schema
        assert output_format['strict'] is True

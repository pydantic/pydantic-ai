from dataclasses import dataclass
from typing import Literal

import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import Agent, NativeOutput

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
    from pydantic_ai.providers.azure import AzureProvider
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


@pytest.mark.parametrize('api', ['chat', 'responses'])
@pytest.mark.parametrize('strict', [None, True])
@pytest.mark.parametrize(
    'provider_name,model_name,inlined,native_output',
    [
        ('azure', 'grok-4-1-fast-reasoning', True, False),
        ('azure', 'Grok-4-1-fast-reasoning', True, False),
        ('azure', 'gpt-4o', False, False),
        ('azure', 'DeepSeek-R1', False, False),
        ('openai', 'grok-4-1-fast-reasoning', False, False),
        ('azure', 'grok-4-1-fast-reasoning', True, True),
        ('azure', 'gpt-4o', False, True),
        ('openai', 'grok-4-1-fast-reasoning', False, True),
        ('azure-custom', 'grok-4-1-fast-reasoning', False, True),
    ],
)
async def test_azure_grok_tool_schema(
    allow_model_requests: None,
    api: Literal['chat', 'responses'],
    strict: bool | None,
    provider_name: Literal['azure', 'azure-custom', 'openai'],
    model_name: str,
    inlined: bool,
    native_output: bool,
):
    """Capture the SDK's actual request; Azure acceptance is reporter-verified, not simulated here."""
    bodies: list[dict[str, JsonValue]] = []
    content = '{"replacements": [{"old_text": "old", "new_text": "new"}]}' if native_output else 'hello'

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(TypeAdapter(dict[str, JsonValue]).validate_json(request.content))
        if api == 'chat':
            return httpx2.Response(
                200,
                json={
                    'id': 'chatcmpl-test',
                    'object': 'chat.completion',
                    'created': 0,
                    'model': model_name,
                    'choices': [
                        {'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}
                    ],
                },
            )
        return httpx2.Response(
            200,
            json={
                'id': 'resp-test',
                'object': 'response',
                'created_at': 0,
                'model': model_name,
                'status': 'completed',
                'output': [
                    {
                        'id': 'msg-test',
                        'type': 'message',
                        'role': 'assistant',
                        'status': 'completed',
                        'content': [{'type': 'output_text', 'text': content, 'annotations': []}],
                    }
                ],
            },
        )

    @dataclass
    class Replacement:
        old_text: str
        new_text: str

    @dataclass
    class Edits:
        replacements: list[Replacement]

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        endpoint = 'https://example.services.ai.azure.com/openai/v1'
        provider = (
            AzureProvider(azure_endpoint=endpoint, api_key='test', http_client=client)
            if provider_name != 'openai'
            else OpenAIProvider(base_url=endpoint, api_key='test', http_client=client)
        )
        profile = (
            OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)
            if provider_name == 'azure-custom'
            else None
        )
        model = (OpenAIChatModel if api == 'chat' else OpenAIResponsesModel)(
            model_name, provider=provider, profile=profile
        )
        agent = Agent(model, output_type=NativeOutput(Edits, strict=strict)) if native_output else Agent(model)

        @agent.tool_plain(strict=strict)
        def edit_file(path: str, replacements: list[Replacement] | None = None) -> str:
            """Edit a file by exact string replacement."""
            return 'ok'  # pragma: no cover

        assert (await agent.run('Say hi.')).output == (Edits([Replacement('old', 'new')]) if native_output else 'hello')

    assert len(bodies) == 1
    tools = bodies[0]['tools']
    assert isinstance(tools, list)
    tool = tools[0]
    assert isinstance(tool, dict)
    if api == 'chat':
        tool = tool['function']
        assert isinstance(tool, dict)
    parameters = tool['parameters']
    replacement: dict[str, JsonValue] = {
        'type': 'object',
        'properties': {'old_text': {'type': 'string'}, 'new_text': {'type': 'string'}},
        'required': ['old_text', 'new_text'],
        'additionalProperties': False,
    }
    replacements: dict[str, JsonValue] = {
        'anyOf': [
            {'type': 'array', 'items': replacement if inlined else {'$ref': '#/$defs/Replacement'}},
            {'type': 'null'},
        ],
    }
    if strict is None:
        replacements['default'] = None
    expected: dict[str, JsonValue] = {
        'type': 'object',
        'properties': {'path': {'type': 'string'}, 'replacements': replacements},
        'required': ['path', 'replacements'] if strict else ['path'],
        'additionalProperties': False,
    }
    if not inlined:
        expected['$defs'] = {'Replacement': replacement}
    assert parameters == expected
    assert tool.get('strict', False) == (strict is True)

    if native_output:
        output_format = bodies[0]['response_format'] if api == 'chat' else bodies[0]['text']
        assert isinstance(output_format, dict)
        output_format = output_format['json_schema'] if api == 'chat' else output_format['format']
        assert isinstance(output_format, dict)
        assert output_format['schema'] == {
            'type': 'object',
            'properties': {'replacements': {'type': 'array', 'items': {'$ref': '#/$defs/Replacement'}}},
            'required': ['replacements'],
            'additionalProperties': False,
            '$defs': {'Replacement': replacement},
        }
        assert output_format['strict'] is True

"""Synthetic SSE contract tests. These are not recordings or live-provider verification."""

from __future__ import annotations

import json
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, UserError
from pydantic_ai.messages import ModelMessage, ModelRequest, SystemPromptPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import CodeExecutionTool, WebSearchTool
from pydantic_ai.tools import ToolDefinition

from .._inline_snapshot import snapshot
from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModelSettings
    from pydantic_ai.models.openai_chatgpt import (
        OpenAIChatGPTModel,
        _CompletedStream,  # pyright: ignore[reportPrivateUsage]
    )
    from pydantic_ai.providers.openai_chatgpt import OpenAIChatGPTProvider

    from ..providers.test_openai_chatgpt import credentials

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai/pyjwt not installed')


def events(*, tool: bool = False, terminal: str | None = 'response.completed', failed_error: bool = False) -> str:
    response: dict[str, Any] = {
        'id': 'resp-tool' if tool else 'resp-text',
        'object': 'response',
        'created_at': 1,
        'model': 'gpt-6.1-sol',
        'status': 'completed',
        'output': [],
        'usage': {'input_tokens': 3, 'output_tokens': 2, 'total_tokens': 5},
    }
    values: list[dict[str, Any]] = [
        {
            'type': 'response.created',
            'sequence_number': 0,
            'response': {**response, 'status': 'in_progress', 'usage': None},
        }
    ]
    if tool:
        values.extend(
            [
                {
                    'type': 'response.output_item.added',
                    'sequence_number': 1,
                    'output_index': 0,
                    'item': {
                        'id': 'fc-moo',
                        'type': 'function_call',
                        'name': 'moo',
                        'arguments': '',
                        'call_id': 'call-moo',
                        'status': 'in_progress',
                    },
                },
                {
                    'type': 'response.function_call_arguments.delta',
                    'sequence_number': 2,
                    'output_index': 0,
                    'item_id': 'fc-moo',
                    'delta': '{}',
                },
            ]
        )
    else:
        values.append(
            {
                'type': 'response.output_text.delta',
                'sequence_number': 1,
                'item_id': 'msg-test',
                'output_index': 0,
                'content_index': 0,
                'delta': 'Moo!',
            }
        )
    if terminal == 'error':
        values.append(
            {'type': 'error', 'sequence_number': 3, 'code': 'usage_limit', 'message': 'Quota exceeded', 'param': None}
        )
    elif terminal:
        if failed_error:
            response['error'] = {'code': 'subscription_sharing_usage_limit_exceeded', 'message': 'Quota exceeded'}
        values.append({'type': terminal, 'sequence_number': 3, 'response': response})
    return ''.join('data: ' + json.dumps(event) + '\n\n' for event in values) + 'data: [DONE]\n\n'


@pytest.mark.parametrize('stream', [False, True])
async def test_authenticated_tool_roundtrip(allow_model_requests: None, stream: bool):
    bodies: list[dict[str, Any]] = []
    headers: list[httpx2.Headers] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == 'https://api.openai.com/v1/responses'
        bodies.append(json.loads(request.content))
        headers.append(request.headers)
        return httpx2.Response(
            200, headers={'content-type': 'text/event-stream'}, content=events(tool=len(bodies) == 1)
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        provider = OpenAIChatGPTProvider(credentials(), http_client=client)
        agent = Agent(OpenAIChatGPTModel('gpt-6.1-sol', provider=provider), instructions='Call moo once.')
        calls: list[str] = []

        @agent.tool_plain
        def moo() -> str:
            """Return a cow sound."""
            calls.append('moo')
            return 'Moo!'

        if stream:
            async with agent.run_stream('Make a sound.') as result:
                output = await result.get_output()
        else:
            output = (await agent.run('Make a sound.')).output
        assert output == 'Moo!'
        assert calls == ['moo']
        assert len(bodies) == 2
        assert bodies[0]['input'][0] == bodies[1]['input'][0]
        assert bodies[1]['input'][-1]['type'] == 'function_call_output'
        assert bodies[1]['input'][-1]['call_id'] == 'call-moo'
        assert bodies[1]['input'][-1]['output'] == 'Moo!'
        assert bodies[0]['input'][0] == snapshot(
            {
                'type': 'additional_tools',
                'role': 'developer',
                'tools': [
                    {
                        'name': 'moo',
                        'parameters': {'additionalProperties': False, 'properties': {}, 'type': 'object'},
                        'type': 'function',
                        'description': 'Return a cow sound.',
                        'strict': False,
                    }
                ],
            }
        )
        assert all(b['stream'] is True and b['store'] is False and 'tools' not in b for b in bodies)
        assert all(h['authorization'] == 'Bearer synthetic-access' for h in headers)
        codex_headers = {'chatgpt-account-id', 'originator', 'session-id', 'thread-id', 'x-client-request-id'}
        assert all(not codex_headers.intersection(h) for h in headers)


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('local_tools', [False, True])
@pytest.mark.filterwarnings('ignore:Sampling parameters.*:UserWarning')
async def test_preview_dialect_and_usage(allow_model_requests: None, stream: bool, local_tools: bool):
    bodies: list[dict[str, Any]] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=events())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = OpenAIChatGPTModel('gpt-6.1-sol', provider=OpenAIChatGPTProvider(credentials(), http_client=client))
        messages: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart('Be helpful.'), UserPromptPart('Hi')])]
        parameters = ModelRequestParameters(
            function_tools=[ToolDefinition(name='test', parameters_json_schema={'type': 'object'})]
            if local_tools
            else [],
            native_tools=[WebSearchTool()],
        )
        settings: OpenAIResponsesModelSettings = {
            'openai_store': True,
            'max_tokens': 10,
            'temperature': 0.4,
            'top_p': 0.9,
        }
        if stream:
            async with model.request_stream(messages, settings, parameters) as result:
                async for _ in result:
                    pass
                response = result.get()
        else:
            response = await model.request(messages, settings, parameters)
        assert response.usage.input_tokens == 3
        assert response.usage.output_tokens == 2
        body = bodies[0]
        assert {'max_output_tokens', 'temperature', 'top_p', 'previous_response_id'}.isdisjoint(body)
        assert body['stream'] is True and body['store'] is False
        assert body['tools'][0]['type'] in ('web_search', 'web_search_preview')
        assert (body['input'][0].get('type') == 'additional_tools') is local_tools
        assert all(item.get('role') != 'system' for item in body['input'])
        with pytest.raises(UserError, match='CodeExecutionTool'):
            await model.request(messages, None, ModelRequestParameters(native_tools=[CodeExecutionTool()]))


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('terminal', [None, 'response.failed', 'response.incomplete', 'error'])
async def test_unsuccessful_terminal_event(allow_model_requests: None, stream: bool, terminal: str | None):
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            headers={'content-type': 'text/event-stream'},
            content=events(terminal=terminal, failed_error=terminal == 'response.failed'),
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = OpenAIChatGPTModel('gpt-6.1-sol', provider=OpenAIChatGPTProvider(credentials(), http_client=client))
        with pytest.raises(ModelAPIError):
            if stream:
                async with model.request_stream([], None, ModelRequestParameters()) as result:
                    async for _ in result:
                        pass
            else:
                await model.request([], None, ModelRequestParameters())


async def test_failed_stream_without_error(allow_model_requests: None):
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, headers={'content-type': 'text/event-stream'}, content=events(terminal='response.failed')
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = OpenAIChatGPTModel('gpt-6.1-sol', provider=OpenAIChatGPTProvider(credentials(), http_client=client))
        with pytest.raises(ModelAPIError, match='ChatGPT inference failed'):
            await model.request([], None, ModelRequestParameters())


async def test_completed_stream_delegates_close():
    """Pin the internal SDK stream interface; request contexts normally close the original stream."""

    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=events())

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        provider = OpenAIChatGPTProvider(credentials(), http_client=client)
        source = await provider.client.responses.create(model='gpt-6.1-sol', input=[], store=False, stream=True)
        await _CompletedStream(source, 'gpt-6.1-sol').close()
        assert source.response.is_closed

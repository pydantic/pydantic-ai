"""Provider request mapping for the live duplicate-ID failures documented in #11120.

These tests inspect the real adapters' outgoing message structures without making provider calls.
The issue records the API rejection; the tests make ID uniqueness and result pairing repeatable.
"""

from __future__ import annotations

import json
from typing import Literal, cast

import httpx2
import pydantic_core
import pytest
from pydantic import JsonValue

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestParameters


async def _mapped_messages(adapter: str, messages: list[ModelMessage]) -> JsonValue:
    """Exercise each mapper that the issue's API probes confirmed rejects repeated IDs."""
    params = ModelRequestParameters()
    if adapter == 'openai':
        pytest.importorskip('openai')
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
        mapped = await model._map_messages(messages, params)  # pyright: ignore[reportPrivateUsage]
    elif adapter == 'anthropic':
        pytest.importorskip('anthropic')
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        model = AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(api_key='test'))
        _, mapped = await model._map_message(messages, params, {})  # pyright: ignore[reportPrivateUsage]
    elif adapter == 'google':
        pytest.importorskip('google.genai')
        from pydantic_ai.models.google import GoogleModel
        from pydantic_ai.providers.google import GoogleProvider

        model = GoogleModel('gemini-2.5-flash', provider=GoogleProvider(api_key='test'))
        _, mapped = await model._map_messages(messages, params)  # pyright: ignore[reportPrivateUsage]
    elif adapter == 'bedrock':
        pytest.importorskip('boto3')
        from pydantic_ai.models.bedrock import BedrockConverseModel
        from pydantic_ai.providers.bedrock import BedrockProvider

        model = BedrockConverseModel(
            'us.anthropic.claude-sonnet-4-6',
            provider=BedrockProvider(region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test'),
        )
        try:
            _, mapped = await model._map_messages(messages, params, None)  # pyright: ignore[reportPrivateUsage]
        finally:
            model.client.close()
    elif adapter == 'mistral':
        pytest.importorskip('mistralai')
        from pydantic_ai.models.mistral import MistralModel
        from pydantic_ai.providers.mistral import MistralProvider

        model = MistralModel('mistral-large-latest', provider=MistralProvider(api_key='test'))
        _, params = model.prepare_request(None, params)
        mapped = await model._map_messages(model.prepare_messages(messages), params)  # pyright: ignore[reportPrivateUsage]
    else:
        assert adapter == 'cohere'
        pytest.importorskip('cohere')
        from pydantic_ai.models.cohere import CohereModel
        from pydantic_ai.providers.cohere import CohereProvider

        model = CohereModel('command-r7b-12-2024', provider=CohereProvider(api_key='test'))
        mapped = model._map_messages(messages, params)  # pyright: ignore[reportPrivateUsage]
    return cast(JsonValue, pydantic_core.to_jsonable_python(mapped, exclude_none=True))


def _wire_ids(value: JsonValue) -> tuple[list[str], list[str]]:
    calls: list[str] = []
    results: list[str] = []
    if isinstance(value, list):
        for item in value:
            child_calls, child_results = _wire_ids(item)
            calls.extend(child_calls)
            results.extend(child_results)
    elif isinstance(value, dict):
        if value.get('type') in ('function', 'tool_use') and isinstance(call_id := value.get('id'), str):
            calls.append(call_id)
        for key, item in value.items():
            if key in ('tool_use_id', 'tool_call_id') and isinstance(item, str):
                results.append(item)
            elif key in ('function_call', 'toolUse') and isinstance(item, dict):
                call_id = item.get('id', item.get('toolUseId'))
                assert isinstance(call_id, str)
                calls.append(call_id)
            elif key in ('function_response', 'toolResult') and isinstance(item, dict):
                result_id = item.get('id', item.get('toolUseId'))
                assert isinstance(result_id, str)
                results.append(result_id)
            else:
                child_calls, child_results = _wire_ids(item)
                calls.extend(child_calls)
                results.extend(child_results)
    return calls, results


@pytest.mark.parametrize('adapter', ['openai', 'anthropic', 'google', 'bedrock', 'mistral', 'cohere'])
async def test_duplicate_ids_within_response_are_unique_and_paired(adapter: str) -> None:
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart('lookup', {'q': 'a'}, 'call_0'), ToolCallPart('lookup', {'q': 'b'}, 'call_0')]
        ),
        ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0'), ToolReturnPart('lookup', 'b', 'call_0')]),
    ]
    stored = ModelMessagesTypeAdapter.dump_json(messages)
    calls, results = _wire_ids(await _mapped_messages(adapter, messages))
    assert len(calls) == len(set(calls)) == 2
    assert results == calls
    assert ModelMessagesTypeAdapter.dump_json(messages) == stored


@pytest.mark.parametrize('adapter', ['openai', 'anthropic', 'google', 'bedrock', 'mistral', 'cohere'])
async def test_reuse_across_responses_is_only_rewritten_for_cohere(adapter: str) -> None:
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart('lookup', {'q': 'a'}, 'call_0')]),
        ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0')]),
        ModelResponse(parts=[ToolCallPart('lookup', {'q': 'b'}, 'call_0')]),
        ModelRequest(parts=[ToolReturnPart('lookup', 'b', 'call_0')]),
    ]
    calls, results = _wire_ids(await _mapped_messages(adapter, messages))
    assert len(calls) == 2
    assert len(set(calls)) == (2 if adapter == 'cohere' else 1)
    assert results == calls


@pytest.mark.parametrize('adapter', ['openai', 'anthropic', 'google', 'bedrock', 'mistral', 'cohere'])
async def test_unique_ids_keep_their_existing_wire_values(adapter: str) -> None:
    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart('lookup', {'q': 'a'}, 'aB3dE5gH9'), ToolCallPart('lookup', {'q': 'b'}, 'bC4eF6hI0')]
        ),
        ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'aB3dE5gH9'), ToolReturnPart('lookup', 'b', 'bC4eF6hI0')]),
    ]
    calls, results = _wire_ids(await _mapped_messages(adapter, messages))
    assert calls == results == ['aB3dE5gH9', 'bC4eF6hI0']


@pytest.mark.parametrize('mode', ['run', 'run_stream', 'request'])
async def test_public_requests_send_paired_ids_without_mutating_history(
    mode: Literal['run', 'run_stream', 'request'], allow_model_requests: None
) -> None:
    """The real agent/SDK boundary sends unique IDs for both normal and streamed requests."""
    pytest.importorskip('openai')
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    captured: list[JsonValue] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        captured.append(body['messages'])
        if body.get('stream'):
            chunks: list[JsonValue] = [
                {
                    'id': 'reply',
                    'created': 1,
                    'model': 'gpt-4o',
                    'object': 'chat.completion.chunk',
                    'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'done'}, 'finish_reason': None}],
                },
                {
                    'id': 'reply',
                    'created': 1,
                    'model': 'gpt-4o',
                    'object': 'chat.completion.chunk',
                    'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                },
            ]
            return httpx2.Response(
                200,
                headers={'content-type': 'text/event-stream'},
                text=''.join(f'data: {json.dumps(chunk)}\n\n' for chunk in chunks) + 'data: [DONE]\n\n',
            )
        return httpx2.Response(
            200,
            json={
                'id': 'reply',
                'created': 1,
                'model': 'gpt-4o',
                'object': 'chat.completion',
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'done'}, 'finish_reason': 'stop'}],
            },
        )

    messages: list[ModelMessage] = [
        ModelResponse(
            parts=[ToolCallPart('lookup', {'q': 'a'}, 'call_0'), ToolCallPart('lookup', {'q': 'b'}, 'call_0')]
        ),
        ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0'), ToolReturnPart('lookup', 'b', 'call_0')]),
    ]
    stored = ModelMessagesTypeAdapter.dump_json(messages)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='test', http_client=client))
        agent = Agent(model)
        if mode == 'run':
            assert (await agent.run('continue', message_history=messages)).output == 'done'
        elif mode == 'run_stream':
            async with agent.run_stream('continue', message_history=messages) as result:
                assert await result.get_output() == 'done'
        else:
            response = await model.request(messages, None, ModelRequestParameters())
            assert response.parts == [TextPart('done')]
    assert len(captured) == 1
    calls, results = _wire_ids(captured[0])
    assert len(calls) == len(set(calls)) == 2
    assert results == calls
    assert ModelMessagesTypeAdapter.dump_json(messages) == stored


async def test_openai_responses_keeps_ids_it_already_accepts() -> None:
    pytest.importorskip('openai')
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

    model = OpenAIResponsesModel('gpt-4o', provider=OpenAIProvider(api_key='test'))
    messages: list[ModelMessage] = [
        ModelResponse(parts=[ToolCallPart('lookup', {}, 'call_0'), ToolCallPart('lookup', {}, 'call_0')]),
        ModelRequest(parts=[ToolReturnPart('lookup', 'a', 'call_0'), ToolReturnPart('lookup', 'b', 'call_0')]),
    ]
    _, mapped = await model._map_messages(messages, {}, ModelRequestParameters())  # pyright: ignore[reportPrivateUsage]
    wire = cast(list[dict[str, JsonValue]], pydantic_core.to_jsonable_python(mapped))
    ids = [item['call_id'] for item in wire if item.get('type') in ('function_call', 'function_call_output')]
    assert ids == ['call_0'] * 4

"""Round-trip recorded DeepSeek web tool responses with synthetic page signatures."""

from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from pydantic_ai import Agent, ModelResponse, NativeOutput, PromptedOutput
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai.exceptions import ModelServerError, UserError
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
)
from pydantic_ai.models import Model
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.deepseek_messages import DeepSeekMessagesProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI and Anthropic SDKs required')

_FIXTURES = Path(__file__).parent / 'fixtures' / 'deepseek-web'
_BODY = TypeAdapter(dict[str, JsonValue])


def deepseek_model(api: str, client: httpx2.AsyncClient) -> Model:
    if api == 'messages':
        return AnthropicModel('deepseek-flash', provider=DeepSeekMessagesProvider(api_key='test', http_client=client))
    model_type = OpenAIChatModel if api == 'chat' else OpenAIResponsesModel
    return model_type('deepseek-flash', provider=DeepSeekProvider(api_key='test', http_client=client))


@pytest.fixture
def recordings(api: str) -> tuple[bytes, bytes]:
    response = (_FIXTURES / f'{api}.json').read_bytes()
    if api == 'chat':
        body = json.loads(response)
        message = body['choices'][0]['message']
        message['server_tool_calls'][0]['reasoning_content'] = 'Thinking before reading'
        message['reasoning_content'] = 'Thinking after reading'
        response = json.dumps(body).encode()
    return response, (_FIXTURES / f'{api}.sse').read_bytes() + b'\n'


@pytest.mark.parametrize('api', ['chat', 'responses', 'messages'])
@pytest.mark.parametrize('stream', [False, True])
async def test_deepseek_web_fetch_history(
    api: str, stream: bool, recordings: tuple[bytes, bytes], allow_model_requests: None
) -> None:
    requests: list[dict[str, JsonValue]] = []
    first_response, streamed_response = recordings

    async def handle(request: httpx2.Request) -> httpx2.Response:
        body = _BODY.validate_json(await request.aread())
        requests.append(body)
        if len(requests) == 1 and stream:
            return httpx2.Response(
                200,
                content=streamed_response,
                headers={'content-type': 'text/event-stream'},
            )
        return httpx2.Response(200, content=first_response, headers={'content-type': 'application/json'})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = deepseek_model(api, client)
        agent = Agent(model, capabilities=[WebFetch()], model_settings={'thinking': False, 'max_tokens': 256})
        prompt = 'Read https://httpbin.org/uuid and reply READY.'
        if stream:
            async with agent.run_stream(prompt) as result:
                assert await result.get_output() == 'READY'
                messages_json = result.all_messages_json()
        else:
            result = await agent.run(prompt)
            assert result.output == 'READY'
            messages_json = result.all_messages_json()

        history = ModelMessagesTypeAdapter.validate_json(messages_json)
        response = history[-1]
        assert isinstance(response, ModelResponse)
        (call,) = [p for p in response.parts if isinstance(p, NativeToolCallPart)]
        (result_part,) = [p for p in response.parts if isinstance(p, NativeToolReturnPart)]
        assert call.tool_name == result_part.tool_name == 'web_fetch'
        assert call.tool_call_id == result_part.tool_call_id
        text_index = next(i for i, part in enumerate(response.parts) if isinstance(part, TextPart))
        assert response.parts.index(call) < response.parts.index(result_part) < text_index
        assert response.usage.details['tool_fetch'] == 1
        assert b'recorded-page-signature' in messages_json

        await Agent(model, model_settings={'thinking': False}).run('What was the UUID?', message_history=history)
        if api == 'chat':
            if not stream:
                assert [part.content for part in response.parts if isinstance(part, ThinkingPart)] == [
                    'Thinking before reading',
                    'Thinking after reading',
                ]
                full_history = json.dumps(requests[1]['messages'])
                assert 'Thinking before reading' in full_history
                assert 'Thinking after reading' in full_history
                response.finish_reason = 'content_filter'
                await Agent(model, model_settings={'thinking': False}).run(
                    'What was the UUID?', message_history=history
                )
                content_filtered_history = json.dumps(requests[-1]['messages'])
                assert 'Thinking before reading' not in content_filtered_history
                assert 'Thinking after reading' not in content_filtered_history
                assert 'recorded-page-signature' in content_filtered_history
                response.finish_reason = 'stop'
                response.parts = [part for part in response.parts if not isinstance(part, ThinkingPart)]
                await Agent(model, model_settings={'thinking': False}).run(
                    'What was the UUID?', message_history=history
                )
                thinking_filtered_history = json.dumps(requests[-1]['messages'])
                assert 'Thinking before reading' not in thinking_filtered_history
                assert 'Thinking after reading' not in thinking_filtered_history
                assert 'recorded-page-signature' in thinking_filtered_history
            response.parts = [part for part in response.parts if part is not call and part is not result_part]
            await Agent(model, model_settings={'thinking': False}).run('What was the UUID?', message_history=history)
            filtered_history = json.dumps(requests[-1]['messages'])
            assert 'server_tool_calls' not in filtered_history
            assert 'recorded-page-signature' not in filtered_history

    tool = {'type': 'deepseek_read_url'}
    if api == 'messages':
        tool['name'] = 'deepseek_read_url'
    assert requests[0]['tools'] == [tool]
    assert 'web_search_options' not in requests[0]
    assert not requests[1].get('tools')
    wire_history = json.dumps(requests[1]['input' if api == 'responses' else 'messages'])
    assert wire_history.count('recorded-page-signature') == 1
    assert call.tool_call_id in wire_history


@pytest.fixture
def search_recording(api: str) -> bytes:
    return (_FIXTURES / f'{api}-search.json').read_bytes()


@pytest.mark.parametrize(
    ('api', 'queries', 'calls', 'citations'),
    [('chat', 2, 1, 7), ('responses', 3, 1, 5), ('messages', 1, 3, 0)],
)
async def test_deepseek_web_search_usage(
    api: str, queries: int, calls: int, citations: int, search_recording: bytes, allow_model_requests: None
) -> None:
    async def handle(request: httpx2.Request) -> httpx2.Response:
        body = _BODY.validate_json(await request.aread())
        tool = {'type': 'deepseek_web_search'}
        if api == 'messages':
            tool['name'] = 'deepseek_web_search'
        assert body['tools'] == [tool]
        return httpx2.Response(200, content=search_recording, headers={'content-type': 'application/json'})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        agent = Agent(deepseek_model(api, client), capabilities=[WebSearch()])
        response = (await agent.run('Search the web and cite your sources.')).response
    tool_calls = {part.tool_call_id: part.tool_name for part in response.parts if isinstance(part, NativeToolCallPart)}
    tool_returns = {
        part.tool_call_id: part.tool_name for part in response.parts if isinstance(part, NativeToolReturnPart)
    }
    assert tool_calls == tool_returns
    assert len(tool_calls) == calls
    assert set(tool_calls.values()) == {'web_search'}
    assert response.usage.details['tool_search'] == queries
    assert sum(len(part.citations or ()) for part in response.parts if isinstance(part, TextPart)) == citations


async def test_deepseek_messages_http_200_generation_error(allow_model_requests: None) -> None:
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(
            200, json={'type': 'error', 'error': {'type': 'api_error', 'message': 'Generation interrupted'}}
        )
    )
    async with httpx2.AsyncClient(transport=transport) as client:
        model = deepseek_model('messages', client)
        with pytest.raises(ModelServerError, match='Generation interrupted') as exc_info:
            await Agent(model, capabilities=[WebSearch()]).run('Search the web.')
    assert exc_info.value.provider_error_type == 'api_error'
    assert not exc_info.value.in_stream


def test_deepseek_web_tool_profile_scope() -> None:
    provider = DeepSeekProvider(api_key='test')
    pro = OpenAIResponsesModel('deepseek-v4-pro', provider=provider)
    flash = OpenAIResponsesModel('deepseek-flash', provider=provider)
    native_tool = WebSearchTool(allowed_domains=['example.com'])
    assert WebFetchTool not in pro.profile.get('supported_native_tools', frozenset())
    assert WebFetchTool in flash.profile.get('supported_native_tools', frozenset())
    assert pro.unsupported_native_tool_settings(native_tool) == frozenset()
    assert flash.unsupported_native_tool_settings(native_tool) == frozenset({'allowed_domains'})


class SearchAnswer(BaseModel):
    summary: str


@pytest.mark.parametrize(('api', 'native'), [('chat', False), ('responses', False), ('responses', True)])
async def test_deepseek_web_tools_reject_non_text_format(api: str, native: bool, allow_model_requests: None) -> None:
    def handle(_: httpx2.Request) -> httpx2.Response:
        pytest.fail('Structured format must be rejected before making a provider request.')

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        agent = Agent(
            deepseek_model(api, client),
            capabilities=[WebSearch()],
            output_type=NativeOutput(SearchAnswer) if native else PromptedOutput(SearchAnswer),
        )
        with pytest.raises(UserError, match='DeepSeek web tools require text output'):
            await agent.run('Search the web.')

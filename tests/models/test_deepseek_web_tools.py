"""Round-trip recorded DeepSeek web tool responses with synthetic page signatures."""

from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import Agent, ModelResponse
from pydantic_ai.capabilities import WebFetch
from pydantic_ai.messages import ModelMessagesTypeAdapter, NativeToolCallPart, NativeToolReturnPart, TextPart
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


@pytest.fixture
def recordings(api: str) -> tuple[bytes, bytes]:
    return (_FIXTURES / f'{api}.json').read_bytes(), (_FIXTURES / f'{api}.sse').read_bytes() + b'\n'


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
        model: Model
        if api == 'messages':
            model = AnthropicModel(
                'deepseek-flash', provider=DeepSeekMessagesProvider(api_key='test', http_client=client)
            )
        else:
            model_type = OpenAIChatModel if api == 'chat' else OpenAIResponsesModel
            model = model_type('deepseek-flash', provider=DeepSeekProvider(api_key='test', http_client=client))
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

    tool = {'type': 'deepseek_read_url'}
    if api == 'messages':
        tool['name'] = 'deepseek_read_url'
    assert requests[0]['tools'] == [tool]
    assert 'web_search_options' not in requests[0]
    assert not requests[1].get('tools')
    wire_history = json.dumps(requests[1]['input' if api == 'responses' else 'messages'])
    assert wire_history.count('recorded-page-signature') == 1
    assert call.tool_call_id in wire_history


def test_deepseek_web_tool_profile_scope() -> None:
    provider = DeepSeekProvider(api_key='test')
    pro = OpenAIResponsesModel('deepseek-v4-pro', provider=provider)
    flash = OpenAIResponsesModel('deepseek-flash', provider=provider)
    native_tool = WebSearchTool(allowed_domains=['example.com'])
    assert WebFetchTool not in pro.profile.get('supported_native_tools', frozenset())
    assert WebFetchTool in flash.profile.get('supported_native_tools', frozenset())
    assert pro.unsupported_native_tool_settings(native_tool) == frozenset()
    assert flash.unsupported_native_tool_settings(native_tool) == frozenset({'allowed_domains'})

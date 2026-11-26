"""Round-trip recorded DeepSeek web tool responses with synthetic page signatures."""

from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from pydantic_ai import Agent, ModelResponse, NativeOutput, PromptedOutput
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai.exceptions import ModelAPIError, ModelServerError, UserError
from pydantic_ai.messages import (
    ContentCitationAnchor,
    ModelMessagesTypeAdapter,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ThinkingPart,
    WebCitationSource,
)
from pydantic_ai.models import Model
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
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
        settings = OpenAIResponsesModelSettings(thinking=False, max_tokens=256)
        if api == 'chat':
            settings['openai_logprobs'] = False
        agent = Agent(model, capabilities=[WebFetch()], model_settings=settings)
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
            response.parts.clear()
            await Agent(model).run('Continue.', message_history=history)
            assert '"assistant"' not in json.dumps(requests[-1]['messages'])

            # A summarizer can replace the whole turn with plain text and discard provider metadata.
            response.parts = [TextPart('Page summary')]
            response.provider_details = None
            await Agent(model).run('Continue.', message_history=history)
            summary_history = json.dumps(requests[-1]['messages'])
            assert 'Page summary' in summary_history
            assert 'server_tool_calls' not in summary_history

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
def stream_without_call_start(api: str) -> bytes:
    """Retain the final result when the provider omits a tool's initial progress event."""
    frames: list[str] = []
    for frame in (_FIXTURES / f'{api}.sse').read_text().strip().split('\n\n'):
        if api == 'responses':
            if frame.startswith('event: response.output_item.added') and '"type":"web_search_call"' in frame:
                continue
        elif frame.startswith('data: {'):
            chunk = json.loads(frame.removeprefix('data: '))
            choice = chunk['choices'][0]
            event = choice.get('server_tool_event', {})
            if event.get('type') == 'call_started':
                continue
            if event.get('type') == 'call_completed':
                # An incomplete notification has no call to match; the authoritative summary still follows.
                incomplete_choice = {**choice, 'server_tool_event': {**event, 'call': None}}
                incomplete_chunk = {**chunk, 'choices': [incomplete_choice]}
                frames.append(f'data: {json.dumps(incomplete_chunk)}')
            if choice['delta'].get('content') == 'READ':
                reasoning_chunk = {**chunk, 'choices': [{**choice, 'delta': {'reasoning_content': 'Read the page.'}}]}
                frames.append(f'data: {json.dumps(reasoning_chunk)}')
        frames.append(frame)
    return ('\n\n'.join(frames) + '\n\n').encode()


@pytest.mark.parametrize('api', ['chat', 'responses'])
async def test_deepseek_final_result_restores_missing_call(
    api: str, stream_without_call_start: bytes, allow_model_requests: None
) -> None:
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(200, content=stream_without_call_start, headers={'content-type': 'text/event-stream'})
    )
    async with httpx2.AsyncClient(transport=transport) as client:
        agent = Agent(deepseek_model(api, client), capabilities=[WebFetch()])
        async with agent.run_stream('Read the page.') as result:
            await result.get_output()
        parts = result.response.parts
    (call,) = [p for p in parts if isinstance(p, NativeToolCallPart)]
    (returned,) = [p for p in parts if isinstance(p, NativeToolReturnPart)]
    assert call.tool_call_id == returned.tool_call_id
    assert call.tool_name == returned.tool_name == 'web_fetch'
    if api == 'chat':
        (thinking,) = [p for p in parts if isinstance(p, ThinkingPart)]
        assert thinking.content == 'Read the page.'
        assert (result.response.provider_details or {})['deepseek_final_reasoning_step'] == 1


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


_RFC_TITLE = 'The title of RFC 9110 on rfc-editor.org is "HTTP Semantics"'
_RFC_URL = 'https://www.rfc-editor.org/rfc/rfc9110.html'
_RFC_INFO = 'https://www.rfc-editor.org/info/rfc9110/'
_RFC_ERRATA = 'https://www.rfc-editor.org/errata/eid7419'
_RFC_ERRATA_SEARCH = 'https://www.rfc-editor.org/errata_search.php?rfc=9110&rec_status=15&presentation=records'


@pytest.mark.parametrize(
    ('api', 'stream', 'cited_text', 'urls'),
    [
        ('chat', True, _RFC_TITLE, [_RFC_URL, _RFC_INFO, _RFC_ERRATA]),
        (
            'responses',
            True,
            _RFC_TITLE,
            [_RFC_INFO, 'https://www.rfc-editor.org/authors/rfc10049.pdf#4#4#4', _RFC_ERRATA],
        ),
        ('messages', True, _RFC_TITLE + ' ', [_RFC_URL, _RFC_INFO, _RFC_ERRATA_SEARCH]),
        (
            'messages',
            False,
            'The title of RFC 9110 is "HTTP Semantics"',
            [_RFC_URL, _RFC_ERRATA_SEARCH, 'https://www.rfc-editor.org/errata/eid7870'],
        ),
    ],
)
async def test_deepseek_search_citation_sources_and_anchors(
    api: str, stream: bool, cited_text: str, urls: list[str], allow_model_requests: None
) -> None:
    # These recorded answers cite the sentence before its final punctuation. Messages puts the period in a separate block.
    suffix = 'sse' if stream else 'json'
    content = (_FIXTURES / f'{api}-citations.{suffix}').read_bytes()
    media_type = 'text/event-stream' if stream else 'application/json'
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(200, content=content + b'\n', headers={'content-type': media_type})
    )
    async with httpx2.AsyncClient(transport=transport) as client:
        agent = Agent(deepseek_model(api, client), capabilities=[WebSearch()], model_settings={'thinking': False})
        prompt = 'Search for the title of RFC 9110 on rfc-editor.org. Answer in one short sentence with a citation.'
        if stream:
            async with agent.run_stream(prompt) as result:
                await result.get_output()
                response = result.response
        else:
            response = (await agent.run(prompt)).response

    source_urls: list[str] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            for citation in part.citations or ():
                assert isinstance(citation.anchor, ContentCitationAnchor)
                assert part.content[citation.anchor.start : citation.anchor.end] == cited_text
                for source in citation.sources:
                    assert isinstance(source, WebCitationSource)
                    source_urls.append(source.url)
    assert source_urls == urls


async def test_deepseek_messages_uses_only_deepseek_credentials(
    monkeypatch: pytest.MonkeyPatch, allow_model_requests: None
) -> None:
    monkeypatch.setenv('ANTHROPIC_CUSTOM_HEADERS', 'x-proxy-id: unrelated\nx-api-key: unrelated')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'deepseek-key')

    def handle(request: httpx2.Request) -> httpx2.Response:
        assert request.headers['x-api-key'] == 'deepseek-key'
        assert 'x-proxy-id' not in request.headers
        return httpx2.Response(
            200,
            json={
                'type': 'message',
                'model': 'deepseek-flash',
                'content': [{'type': 'text', 'text': 'Hello'}],
                'usage': {'input_tokens': 1, 'output_tokens': 1},
                'stop_reason': 'end_turn',
            },
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        model = AnthropicModel('deepseek-flash', provider=DeepSeekMessagesProvider(http_client=client))
        # Setting max_retries exercises the SDK's per-request client copy.
        await Agent(model).run('Hello', model_settings={'max_retries': 0})


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


def test_deepseek_messages_requires_its_own_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('DEEPSEEK_API_KEY', raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'unrelated')
    with pytest.raises(UserError, match='DEEPSEEK_API_KEY'):
        DeepSeekMessagesProvider()


async def test_deepseek_messages_default_client_lifecycle() -> None:
    provider = DeepSeekMessagesProvider(api_key='test')
    async with provider:
        assert not provider.client.is_closed()
    assert provider.client.is_closed()


@pytest.mark.parametrize('error_type', [None, 'api_error'])
async def test_deepseek_chat_http_200_generation_error(error_type: str | None, allow_model_requests: None) -> None:
    error = {'message': 'Generation interrupted'}
    if error_type:
        error['type'] = error_type
    transport = httpx2.MockTransport(lambda _: httpx2.Response(200, json={'error': error}))
    async with httpx2.AsyncClient(transport=transport) as client:
        with pytest.raises(ModelAPIError, match='Generation interrupted') as exc:
            await Agent(deepseek_model('chat', client), capabilities=[WebSearch()]).run('Search the web.')
    assert exc.value.provider_error_type == error_type
    assert not exc.value.in_stream


@pytest.mark.parametrize('api', ['messages', 'responses'])
async def test_deepseek_replays_standard_search_history(api: str, allow_model_requests: None) -> None:
    """A standard search call from the same provider still uses the protocol's ordinary replay format."""
    source = {
        'type': 'web_search_result',
        'url': 'https://example.org',
        'title': 'Example',
        'encrypted_content': 'opaque',
    }
    reply: dict[str, object]
    if api == 'messages':
        reply = {
            'id': 'msg_1',
            'model': 'deepseek-flash',
            'type': 'message',
            'role': 'assistant',
            'content': [
                {'type': 'server_tool_use', 'name': 'web_search', 'id': 'search_1', 'input': {'query': 'example'}},
                {'type': 'web_search_tool_result', 'tool_use_id': 'search_1', 'content': [source]},
                {'type': 'text', 'text': 'Found it.'},
            ],
            'stop_reason': 'end_turn',
            'usage': {'input_tokens': 1, 'output_tokens': 1},
        }
    else:
        reply = {
            'id': 'resp_1',
            'model': 'deepseek-flash',
            'status': 'completed',
            'created_at': 1,
            'output': [
                {
                    'type': 'web_search_call',
                    'id': 'search_1',
                    'status': 'completed',
                    'action': {'type': 'search', 'query': 'example'},
                },
                {
                    'type': 'message',
                    'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'Found it.', 'annotations': []}],
                },
            ],
        }
    requests: list[bytes] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request.content)
        return httpx2.Response(200, json=reply)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        agent = Agent(deepseek_model(api, client), capabilities=[WebSearch()])
        result = await agent.run('Search for an example.')
        assert any(isinstance(p, NativeToolReturnPart) and p.tool_call_id == 'search_1' for p in result.response.parts)
        await agent.run('What did you find?', message_history=result.all_messages())
    assert b'Found it.' in requests[1]
    if api == 'messages':
        assert b'search_1' in requests[1]
    else:
        # DeepSeek Responses is stateless; an unsigned search ID cannot restore server-side state.
        assert b'search_1' not in requests[1]


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
        assert False, 'Structured format must be rejected before making a provider request.'

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        agent = Agent(
            deepseek_model(api, client),
            capabilities=[WebSearch()],
            output_type=NativeOutput(SearchAnswer) if native else PromptedOutput(SearchAnswer),
        )
        with pytest.raises(UserError, match='DeepSeek web tools require text output'):
            await agent.run('Search the web.')


@pytest.mark.parametrize(
    ('api', 'settings', 'setting'),
    [
        ('chat', {'openai_logprobs': True}, 'openai_logprobs'),
        ('chat', {'openai_top_logprobs': 0}, 'openai_top_logprobs'),
        ('responses', {'openai_top_logprobs': 0}, 'openai_top_logprobs'),
        ('responses', {'openai_previous_response_id': 'resp_123'}, 'openai_previous_response_id'),
        ('responses', {'openai_conversation_id': 'conv_123'}, 'openai_conversation_id'),
    ],
)
async def test_deepseek_web_tools_reject_incompatible_settings(
    api: str, settings: OpenAIResponsesModelSettings, setting: str, allow_model_requests: None
) -> None:
    def handle(_: httpx2.Request) -> httpx2.Response:
        assert False, f'{setting} should be rejected before sending the request.'

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        agent = Agent(deepseek_model(api, client), capabilities=[WebFetch()])
        with pytest.raises(UserError, match=setting):
            await agent.run('Read the page.', model_settings=settings)

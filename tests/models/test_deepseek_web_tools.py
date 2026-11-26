"""Round-trip recorded DeepSeek web tool responses with synthetic page signatures."""

from __future__ import annotations

import json
from pathlib import Path

import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import Agent, ModelResponse
from pydantic_ai._utils import is_str_dict
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
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
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel, OpenAIResponsesModelSettings
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.deepseek import DeepSeekProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK required')

_FIXTURES = Path(__file__).parent / 'fixtures' / 'deepseek-web'
_BODY = TypeAdapter(dict[str, JsonValue])


def deepseek_model(api: str, client: httpx2.AsyncClient) -> Model:
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


@pytest.mark.parametrize('api', ['chat', 'responses'])
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
        if api == 'responses':
            # Client annotations on persisted results are not Responses protocol fields.
            assert is_str_dict(result_part.content) and result_part.provider_details
            result_part.content['client_note'] = 'local annotation'
            result_part.provider_details['deepseek_responses_web_tool']['type'] = 'client_annotation'

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
                no_thinking_replay = OpenAIChatModel(
                    'deepseek-flash',
                    provider=DeepSeekProvider(api_key='test', http_client=client),
                    profile=OpenAIModelProfile(openai_chat_send_back_thinking_parts=False),
                )
                await Agent(no_thinking_replay).run('What was the UUID?', message_history=history)
                redacted = json.dumps(requests[-1]['messages'])
                assert 'Thinking before reading' not in redacted
                assert 'Thinking after reading' not in redacted
                assert 'recorded-page-signature' in redacted
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
            assert is_str_dict(call.args) and is_str_dict(result_part.content)
            url = call.args.pop('url')
            await Agent(model).run('Continue.', message_history=history)
            assert 'server_tool_calls' not in json.dumps(requests[-1]['messages'])
            call.args['url'] = url
            del result_part.content['page']
            await Agent(model).run('Continue.', message_history=history)
            assert 'recorded-page-signature' not in json.dumps(requests[-1]['messages'])
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
    assert requests[0]['tools'] == [tool]
    assert 'web_search_options' not in requests[0]
    assert not requests[1].get('tools')
    wire_history = json.dumps(requests[1]['input' if api == 'responses' else 'messages'])
    assert 'local annotation' not in wire_history
    assert 'client_annotation' not in wire_history
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
            if event.get('type') == 'step_reasoning_done':
                reasoning = {**chunk, 'choices': [{**choice, 'delta': {'reasoning_content': 'Before reading.'}}]}
                reasoning['choices'][0].pop('server_tool_event')
                frames.append(f'data: {json.dumps(reasoning)}')
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
        before, after = [p for p in parts if isinstance(p, ThinkingPart)]
        assert before.content == 'Before reading.'
        assert (before.provider_details or {})['deepseek_reasoning_step'] == 0
        assert after.content == 'Read the page.'
        assert after.provider_details is None


@pytest.fixture
def search_recording(api: str) -> bytes:
    return (_FIXTURES / f'{api}-search.json').read_bytes()


@pytest.mark.parametrize(
    ('api', 'queries', 'calls', 'citations'),
    [('chat', 2, 1, 7), ('responses', 3, 1, 5)],
)
async def test_deepseek_web_search_usage(
    api: str, queries: int, calls: int, citations: int, search_recording: bytes, allow_model_requests: None
) -> None:
    async def handle(request: httpx2.Request) -> httpx2.Response:
        body = _BODY.validate_json(await request.aread())
        tool = {'type': 'deepseek_web_search'}
        if body.get('tools'):
            assert body['tools'] == [tool]
        else:
            original = json.loads(search_recording)['output']
            expected = next(item for item in original if item['type'] == 'web_search_call')
            assert expected in json.loads(request.content)['input']
        return httpx2.Response(200, content=search_recording, headers={'content-type': 'application/json'})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        agent = Agent(deepseek_model(api, client), capabilities=[WebSearch()])
        result = await agent.run('Search the web and cite your sources.')
        response = result.response
        if api == 'responses':
            await Agent(agent.model).run('What did you find?', message_history=result.all_messages())
    tool_calls = {part.tool_call_id: part.tool_name for part in response.parts if isinstance(part, NativeToolCallPart)}
    tool_returns = {
        part.tool_call_id: part.tool_name for part in response.parts if isinstance(part, NativeToolReturnPart)
    }
    assert tool_calls == tool_returns
    assert len(tool_calls) == calls
    assert set(tool_calls.values()) == {'web_search'}
    assert response.usage.details['web_search_requests'] == queries
    assert 'tool_fetch' not in response.usage.details
    assert sum(len(part.citations or ()) for part in response.parts if isinstance(part, TextPart)) == citations


_RFC_TITLE = 'The title of RFC 9110 on rfc-editor.org is "HTTP Semantics"'
_RFC_URL = 'https://www.rfc-editor.org/rfc/rfc9110.html'
_RFC_INFO = 'https://www.rfc-editor.org/info/rfc9110/'
_RFC_ERRATA = 'https://www.rfc-editor.org/errata/eid7419'


@pytest.mark.parametrize(
    ('api', 'cited_text', 'urls'),
    [
        ('chat', _RFC_TITLE, [_RFC_URL, _RFC_INFO, _RFC_ERRATA]),
        (
            'responses',
            _RFC_TITLE,
            [_RFC_INFO, 'https://www.rfc-editor.org/authors/rfc10049.pdf#4#4#4', _RFC_ERRATA],
        ),
    ],
)
async def test_deepseek_search_citation_sources_and_anchors(
    api: str, cited_text: str, urls: list[str], allow_model_requests: None
) -> None:
    content = (_FIXTURES / f'{api}-citations.sse').read_bytes()
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(200, content=content + b'\n', headers={'content-type': 'text/event-stream'})
    )
    async with httpx2.AsyncClient(transport=transport) as client:
        agent = Agent(deepseek_model(api, client), capabilities=[WebSearch()], model_settings={'thinking': False})
        prompt = 'Search for the title of RFC 9110 on rfc-editor.org. Answer in one short sentence with a citation.'
        async with agent.run_stream(prompt) as result:
            await result.get_output()
            response = result.response

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


async def test_deepseek_replays_standard_search_history(allow_model_requests: None) -> None:
    """An unsigned search item follows the ordinary stateless Responses replay path."""
    reply: dict[str, object] = {
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
        agent = Agent(deepseek_model('responses', client), capabilities=[WebSearch()])
        result = await agent.run('Search for an example.')
        assert any(isinstance(p, NativeToolReturnPart) and p.tool_call_id == 'search_1' for p in result.response.parts)
        await agent.run('What did you find?', message_history=result.all_messages())
    assert b'Found it.' in requests[1]
    assert b'search_1' not in requests[1]


@pytest.mark.parametrize('flash_name', ['deepseek-flash', 'deepseek-v4-flash'])
def test_deepseek_web_tool_profile_scope(flash_name: str) -> None:
    provider = DeepSeekProvider(api_key='test')
    pro = OpenAIResponsesModel('deepseek-v4-pro', provider=provider)
    flash = OpenAIResponsesModel(flash_name, provider=provider)
    native_tool = WebSearchTool(allowed_domains=['example.com'])
    assert WebFetchTool not in pro.profile.get('supported_native_tools', frozenset())
    assert WebFetchTool in flash.profile.get('supported_native_tools', frozenset())
    assert pro.unsupported_native_tool_settings(native_tool) == frozenset()
    assert flash.unsupported_native_tool_settings(native_tool) == frozenset({'allowed_domains'})


@pytest.mark.parametrize(
    ('api', 'settings', 'wire_field', 'value'),
    [
        ('chat', {'openai_logprobs': True}, 'logprobs', True),
        ('chat', {'openai_top_logprobs': 0}, 'top_logprobs', 0),
        ('responses', {'openai_top_logprobs': 0}, 'top_logprobs', 0),
        ('responses', {'openai_previous_response_id': 'resp_123'}, 'previous_response_id', 'resp_123'),
        ('responses', {'openai_conversation_id': 'conv_123'}, 'conversation', 'conv_123'),
    ],
)
async def test_web_tool_provider_settings_reach_deepseek(
    api: str, settings: OpenAIResponsesModelSettings, wire_field: str, value: JsonValue, allow_model_requests: None
) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        assert _BODY.validate_json(request.content)[wire_field] == value
        return httpx2.Response(400, json={'error': {'message': 'Unsupported web tool setting'}})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as client:
        agent = Agent(deepseek_model(api, client), capabilities=[WebFetch()])
        with pytest.raises(ModelHTTPError, match='Unsupported web tool setting') as exc:
            await agent.run('Read the page.', model_settings=settings)
    assert exc.value.status_code == 400

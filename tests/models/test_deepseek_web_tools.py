"""DeepSeek Flash hosted web tools on the Responses API."""

import httpx2
import pytest

from pydantic_ai import Agent, ModelRequest, ModelResponse
from pydantic_ai._utils import is_str_dict
from pydantic_ai.capabilities import WebFetch, WebSearch
from pydantic_ai.messages import (
    ContentCitationAnchor,
    ModelMessagesTypeAdapter,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    WebCitationSource,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import WebFetchTool, WebSearchTool

from ..conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from openai.types import responses

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider

    from .mock_openai import MockOpenAIResponses, response_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK required')


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['non_streamed', 'streamed'])
async def test_web_fetch_signed_history(
    stream: bool, request_capture: RequestCapture, deepseek_api_key: str, allow_model_requests: None
) -> None:
    model = OpenAIResponsesModel(
        'deepseek-flash', provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=request_capture.client)
    )
    agent = Agent(model, capabilities=[WebFetch()], model_settings={'thinking': False})
    prompt = 'Read https://httpbin.org/json using the URL reading tool. Reply only READY, without quoting the page.'
    if stream:
        async with agent.run_stream(prompt) as streamed:
            output = await streamed.get_output()
            history_json = streamed.all_messages_json()
            response = streamed.response
    else:
        result = await agent.run(prompt)
        output = result.output
        history_json = result.all_messages_json()
        response = result.response
    assert 'wonderwidgets' not in output.lower()
    (call,) = [p for p in response.parts if isinstance(p, NativeToolCallPart)]
    (returned,) = [p for p in response.parts if isinstance(p, NativeToolReturnPart)]
    assert call.tool_name == returned.tool_name == 'web_fetch'
    assert call.tool_call_id == returned.tool_call_id
    assert response.usage.details['tool_fetch'] == 1
    assert returned.provider_details
    signature = returned.provider_details['deepseek_web_tool']['signature']
    assert signature

    history = ModelMessagesTypeAdapter.validate_json(history_json)
    # Client annotations on reloaded history must not become fields of the Responses input item.
    assert isinstance(history[-1], ModelResponse)
    (reloaded,) = [p for p in history[-1].parts if isinstance(p, NativeToolReturnPart)]
    assert is_str_dict(reloaded.content) and reloaded.provider_details
    reloaded.content['client_note'] = 'local annotation'
    reloaded.provider_details['deepseek_web_tool']['type'] = 'client_annotation'
    follow_up = await Agent(model, model_settings={'thinking': False}).run(
        'What is the exact title of the first slide in the slideshow from that JSON page?', message_history=history
    )
    assert 'wake up to wonderwidgets' in follow_up.output.lower()
    first, second = request_capture.bodies('/responses')
    assert first['tools'] == [{'type': 'deepseek_read_url'}]
    assert not second.get('tools')
    items = second['input']
    assert isinstance(items, list)
    (replay,) = [item for item in items if isinstance(item, dict) and item.get('type') == 'web_search_call']
    assert replay['id'] == call.tool_call_id
    assert replay['signature'] == signature
    assert 'client_note' not in replay


@pytest.mark.vcr
async def test_web_search_citations(
    request_capture: RequestCapture, deepseek_api_key: str, allow_model_requests: None
) -> None:
    model = OpenAIResponsesModel(
        'deepseek-flash', provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=request_capture.client)
    )
    agent = Agent(model, capabilities=[WebSearch()], model_settings={'thinking': False})
    prompt = 'Search for the title of RFC 9110 on rfc-editor.org. Answer in one short sentence with a citation.'
    async with agent.run_stream(prompt) as result:
        await result.get_output()
        response = result.response
        history = result.all_messages()
    assert len(response.native_tool_calls) == 1
    assert response.usage.details['web_search_requests'] == 2
    assert 'tool_fetch' not in response.usage.details
    citations = [
        (part.content, citation)
        for part in response.parts
        if isinstance(part, TextPart)
        for citation in part.citations or ()
    ]
    source_urls: list[str] = []
    for text, citation in citations:
        assert isinstance(citation.anchor, ContentCitationAnchor)
        assert (
            text[citation.anchor.start : citation.anchor.end]
            == 'The title of RFC 9110 on rfc-editor.org is "HTTP Semantics"'
        )
        for source in citation.sources:
            assert isinstance(source, WebCitationSource)
            source_urls.append(source.url)
    assert 'https://www.rfc-editor.org/rfc/rfc9110.html' in source_urls

    await Agent(model, model_settings={'thinking': False}).run('What title did you find?', message_history=history)
    first, second = request_capture.bodies('/responses')
    assert first['tools'] == [{'type': 'deepseek_web_search'}]
    items = second['input']
    assert isinstance(items, list)
    assert any(isinstance(item, dict) and item.get('kind') == 'search' for item in items)


@pytest.mark.parametrize('with_added', [True, False])
async def test_web_tool_final_args(with_added: bool, allow_model_requests: None) -> None:
    """The completed item supplies the final arguments even when the initial event is partial or absent."""
    item = responses.ResponseFunctionWebSearch.model_validate(
        {
            'type': 'web_search_call',
            'id': 'search_1',
            'kind': 'search',
            'status': 'completed',
            'action': {'type': 'search', 'query': 'RFC 9110'},
        }
    )
    reply = response_message([item])
    added = responses.ResponseOutputItemAddedEvent(
        item=item.model_copy(update={'action': None, 'status': 'in_progress'}),
        output_index=0,
        sequence_number=1,
        type='response.output_item.added',
    )
    client = MockOpenAIResponses.create_mock_stream(
        [
            responses.ResponseCreatedEvent(response=reply, type='response.created', sequence_number=0),
            *([added] if with_added else []),
            responses.ResponseOutputItemDoneEvent(
                item=item, output_index=0, sequence_number=2, type='response.output_item.done'
            ),
            responses.ResponseCompletedEvent(response=reply, type='response.completed', sequence_number=3),
        ]
    )
    model = OpenAIResponsesModel('deepseek-flash', provider=DeepSeekProvider(openai_client=client))
    async with model.request_stream(
        [ModelRequest.user_text_prompt('Search.')], None, ModelRequestParameters()
    ) as stream:
        async for _ in stream:
            pass
        call, result = stream.get().parts
    assert isinstance(call, NativeToolCallPart) and isinstance(result, NativeToolReturnPart)
    assert call.tool_call_id == result.tool_call_id == 'search_1'
    assert call.args_as_dict() == {'type': 'search', 'query': 'RFC 9110'}


@pytest.mark.parametrize(
    'provider_details',
    [None, {'deepseek_web_tool': 'edited'}, {'deepseek_web_tool': {}}],
    ids=['standard', 'invalid-metadata', 'missing-kind'],
)
async def test_deepseek_replays_standard_search_history(
    provider_details: dict[str, object] | None, allow_model_requests: None
) -> None:
    """A standard OpenAI web_search_call without kind follows the ordinary stateless Responses replay path."""
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
        agent = Agent(
            OpenAIResponsesModel('deepseek-flash', provider=DeepSeekProvider(api_key='test', http_client=client)),
            capabilities=[WebSearch()],
        )
        result = await agent.run('Search for an example.')
        (returned,) = [p for p in result.response.parts if isinstance(p, NativeToolReturnPart)]
        assert returned.tool_call_id == 'search_1'
        returned.provider_details = provider_details
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

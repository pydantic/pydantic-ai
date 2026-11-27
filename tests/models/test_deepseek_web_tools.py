"""DeepSeek Flash hosted web tools on the Responses API, through `DeepSeekResponsesModel`."""

from __future__ import annotations

import json

import httpx2
import pytest
from inline_snapshot import snapshot
from pydantic import JsonValue

from pydantic_ai import Agent, ModelRequest, ModelResponse
from pydantic_ai._utils import is_str_dict
from pydantic_ai.capabilities import AbstractCapability, WebFetch, WebSearch
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

    from pydantic_ai.models.deepseek import DeepSeekResponsesModel
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAIResponses, response_message

pytestmark = pytest.mark.skipif(not imports_successful(), reason='OpenAI SDK required')


def _reply(*output: JsonValue) -> dict[str, JsonValue]:
    return {'id': 'resp_1', 'model': 'deepseek-flash', 'status': 'completed', 'created_at': 1, 'output': list(output)}


_FOUND_IT: dict[str, JsonValue] = {
    'type': 'message',
    'role': 'assistant',
    'content': [{'type': 'output_text', 'text': 'Found it.', 'annotations': []}],
}


def _mock_client(reply: dict[str, JsonValue], requests: list[dict[str, JsonValue]]) -> httpx2.AsyncClient:
    """A client that answers every request with `reply` and records each request body in `requests`."""

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=reply)

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handle))


@pytest.mark.vcr
@pytest.mark.parametrize('stream', [False, True], ids=['non_streamed', 'streamed'])
async def test_web_fetch_signed_history(
    stream: bool, request_capture: RequestCapture, deepseek_api_key: str, allow_model_requests: None
) -> None:
    model = DeepSeekResponsesModel(
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
    """A streamed search maps DeepSeek's citations, usage and search item, and replays the item.

    DeepSeek's `url_citation` offsets span the claim the source supports: in the recording they are 0 and 59, and the
    answer's first 59 characters are the sentence the RFC supports, with no citation marker in them. So they map to a
    `ContentCitationAnchor`, not the `MarkerCitationAnchor` OpenAI's link offsets map to.
    """
    model = DeepSeekResponsesModel(
        'deepseek-flash', provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=request_capture.client)
    )
    agent = Agent(model, capabilities=[WebSearch()], model_settings={'thinking': False})
    prompt = 'Search for the title of RFC 9110 on rfc-editor.org. Answer in one short sentence with a citation.'
    async with agent.run_stream(prompt) as result:
        await result.get_output()
        response = result.response
        history = result.all_messages()
    assert len(response.native_tool_calls) == 1
    # DeepSeek bills no search, so the count stays in the details and isn't billed as `web_searches`.
    assert response.usage.details['tool_search'] == 2
    assert 'tool_fetch' not in response.usage.details
    assert 'web_search_requests' not in response.usage.details
    assert not hasattr(response.usage, 'web_searches')
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
    """The completed item supplies the final arguments even when the initial event is partial or absent.

    The recorded streams send the full arguments on `output_item.added`, so only a built stream can leave them out.
    """
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
    model = DeepSeekResponsesModel('deepseek-flash', provider=DeepSeekProvider(openai_client=client))
    async with model.request_stream(
        [ModelRequest.user_text_prompt('Search.')], None, ModelRequestParameters()
    ) as stream:
        async for _ in stream:
            pass
        call, result = stream.get().parts
    assert isinstance(call, NativeToolCallPart) and isinstance(result, NativeToolReturnPart)
    assert call.tool_call_id == result.tool_call_id == 'search_1'
    assert call.args_as_dict() == {'type': 'search', 'query': 'RFC 9110'}


@pytest.mark.parametrize(('model_name', 'billed'), [('deepseek-v4-pro', 1), ('deepseek-flash', 0)])
async def test_streamed_web_search_billing(model_name: str, billed: int, allow_model_requests: None) -> None:
    """A streamed search goes unbilled where DeepSeek's web tools run, and counts as on `OpenAIResponsesModel` elsewhere.

    No recording holds a `deepseek-v4-pro` search, so a built stream carries a standard search item.
    """
    item = responses.ResponseFunctionWebSearch.model_validate(
        {
            'type': 'web_search_call',
            'id': 'search_1',
            'status': 'completed',
            'action': {'type': 'search', 'query': 'RFC 9110'},
        }
    )
    reply = response_message([item])
    client = MockOpenAIResponses.create_mock_stream(
        [
            responses.ResponseCreatedEvent(response=reply, type='response.created', sequence_number=0),
            responses.ResponseOutputItemAddedEvent(
                item=item, output_index=0, sequence_number=1, type='response.output_item.added'
            ),
            responses.ResponseOutputItemDoneEvent(
                item=item, output_index=0, sequence_number=2, type='response.output_item.done'
            ),
            responses.ResponseCompletedEvent(response=reply, type='response.completed', sequence_number=3),
        ]
    )
    model = DeepSeekResponsesModel(model_name, provider=DeepSeekProvider(openai_client=client))
    async with model.request_stream(
        [ModelRequest.user_text_prompt('Search.')], None, ModelRequestParameters()
    ) as stream:
        async for _ in stream:
            pass
    assert getattr(stream.get().usage, 'web_searches', 0) == billed


async def test_web_tool_usage_counts(allow_model_requests: None) -> None:
    """Each non-zero web tool count lands in the details, and none counts as a billed web search.

    No recording holds an image search, so a mock transport reports the counts.
    """
    reply: dict[str, JsonValue] = {
        **_reply(_FOUND_IT),
        'usage': {
            'input_tokens': 10,
            'input_tokens_details': {'cached_tokens': 0},
            'output_tokens': 2,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': 12,
            'server_tool_use': {'tool_search': 1, 'tool_fetch': 0, 'tool_image_search': 2},
        },
    }
    requests: list[dict[str, JsonValue]] = []
    async with _mock_client(reply, requests) as client:
        model = DeepSeekResponsesModel('deepseek-flash', provider=DeepSeekProvider(api_key='test', http_client=client))
        result = await Agent(model, capabilities=[WebSearch()]).run('Find a picture of a kiwi.')
    details = result.response.usage.details
    assert {key: details[key] for key in ('tool_search', 'tool_fetch', 'tool_image_search') if key in details} == {
        'tool_search': 1,
        'tool_image_search': 2,
    }
    assert 'web_search_requests' not in details
    assert not hasattr(result.response.usage, 'web_searches')


@pytest.mark.parametrize(
    'provider_details',
    [None, {'deepseek_web_tool': 'edited'}, {'deepseek_web_tool': {}}],
    ids=['standard', 'invalid-metadata', 'missing-kind'],
)
async def test_deepseek_replays_standard_search_history(
    provider_details: dict[str, object] | None, allow_model_requests: None
) -> None:
    """A standard OpenAI `web_search_call` without `kind`, or one with edited metadata, isn't sent back as a DeepSeek item.

    The edits are made to history no recording holds, so a mock transport answers both requests.
    """
    standard_search: dict[str, JsonValue] = {
        'type': 'web_search_call',
        'id': 'search_1',
        'status': 'completed',
        'action': {'type': 'search', 'query': 'example'},
    }
    requests: list[dict[str, JsonValue]] = []
    async with _mock_client(_reply(standard_search, _FOUND_IT), requests) as client:
        agent = Agent(
            DeepSeekResponsesModel('deepseek-flash', provider=DeepSeekProvider(api_key='test', http_client=client)),
            capabilities=[WebSearch()],
        )
        result = await agent.run('Search for an example.')
        (returned,) = [p for p in result.response.parts if isinstance(p, NativeToolReturnPart)]
        assert returned.tool_call_id == 'search_1'
        returned.provider_details = provider_details
        await agent.run('What did you find?', message_history=result.all_messages())
    assert 'Found it.' in json.dumps(requests[1])
    assert 'search_1' not in json.dumps(requests[1])


# The local fallbacks are only declared: the mocked model answers without calling a tool.
def _search_example(query: str) -> str:
    """Search the web."""
    raise NotImplementedError


def _fetch_example(url: str) -> str:
    """Fetch a URL."""
    raise NotImplementedError


@pytest.mark.parametrize(
    'capability',
    [
        pytest.param(
            WebSearch(native=WebSearchTool(allowed_domains=['example.com']), local=_search_example),
            id='restricted-search',
        ),
        pytest.param(WebFetch(native=False, local=_fetch_example), id='fetch-native-off'),
    ],
)
async def test_web_tool_local_fallback(capability: AbstractCapability[object], allow_model_requests: None) -> None:
    """A web tool DeepSeek's server can't run as asked is declared as the local function tool instead.

    The choice is made before the request goes out, so the test reads the request a mock transport receives.
    """
    requests: list[dict[str, JsonValue]] = []
    async with _mock_client(_reply(_FOUND_IT), requests) as client:
        model = DeepSeekResponsesModel('deepseek-flash', provider=DeepSeekProvider(api_key='test', http_client=client))
        await Agent(model, capabilities=[capability]).run('Look it up.')
    (request,) = requests
    tools = request['tools']
    assert isinstance(tools, list)
    assert [tool['type'] for tool in tools if isinstance(tool, dict)] == ['function']


async def test_web_fetch_history_to_another_provider(allow_model_requests: None) -> None:
    """A signed DeepSeek read in history stays out of a request to another provider.

    The claim is about what the second request carries, so a mock transport stands in for both providers.
    """
    read_link: dict[str, JsonValue] = {
        'type': 'web_search_call',
        'id': 'read_1',
        'status': 'completed',
        'kind': 'read_link',
        'action': {'type': 'open_page', 'url': 'https://example.com'},
        'page': {'url': 'https://example.com', 'title': 'Example'},
        'signature': 'signed-page',
    }
    deepseek_requests: list[dict[str, JsonValue]] = []
    async with _mock_client(_reply(read_link, _FOUND_IT), deepseek_requests) as client:
        model = DeepSeekResponsesModel('deepseek-flash', provider=DeepSeekProvider(api_key='test', http_client=client))
        result = await Agent(model, capabilities=[WebFetch()]).run('Read https://example.com.')
    openai_requests: list[dict[str, JsonValue]] = []
    async with _mock_client(_reply(_FOUND_IT), openai_requests) as client:
        model = OpenAIResponsesModel('gpt-5.4', provider=OpenAIProvider(api_key='test', http_client=client))
        await Agent(model).run('What did the page say?', message_history=result.all_messages())
    (request,) = openai_requests
    assert request['input'] == snapshot(
        [
            {'role': 'user', 'content': 'Read https://example.com.'},
            {'role': 'assistant', 'content': 'Found it.'},
            {'role': 'user', 'content': 'What did the page say?'},
        ]
    )


@pytest.mark.parametrize(
    ('model', 'model_name', 'tool_type'),
    [
        pytest.param(OpenAIResponsesModel, 'deepseek-flash', 'web_search', id='openai-responses-model'),
        pytest.param(DeepSeekResponsesModel, 'deepseek-v4-pro', 'web_search', id='deepseek-v4-pro'),
        pytest.param(DeepSeekResponsesModel, 'deepseek-flash', 'deepseek_web_search', id='deepseek-flash'),
    ],
)
async def test_web_search_declaration(
    model: type[OpenAIResponsesModel], model_name: str, tool_type: str, allow_model_requests: None
) -> None:
    """Only `DeepSeekResponsesModel` with `deepseek-flash` declares DeepSeek's web search; the others keep the standard tool.

    The declaration is what the request carries, so the test reads the request a mock transport receives.
    """
    requests: list[dict[str, JsonValue]] = []
    async with _mock_client(_reply(_FOUND_IT), requests) as client:
        await Agent(
            model(model_name, provider=DeepSeekProvider(api_key='test', http_client=client)), capabilities=[WebSearch()]
        ).run('Search.')
    (request,) = requests
    tools = request['tools']
    assert isinstance(tools, list)
    assert [tool['type'] for tool in tools if isinstance(tool, dict)] == [tool_type]


def test_deepseek_web_tool_profile_scope() -> None:
    """Only `DeepSeekResponsesModel` with `deepseek-flash` gets DeepSeek's web tools and their restrictions.

    These are the profile and the restricting settings, which no request shows.
    """
    provider = DeepSeekProvider(api_key='test')
    pro = DeepSeekResponsesModel('deepseek-v4-pro', provider=provider)
    flash = DeepSeekResponsesModel('deepseek-flash', provider=provider)
    openai_flash = OpenAIResponsesModel('deepseek-flash', provider=provider)
    restricted = WebSearchTool(allowed_domains=['example.com'])
    assert WebFetchTool not in OpenAIResponsesModel.supported_native_tools()
    assert WebFetchTool not in openai_flash.profile.get('supported_native_tools', frozenset())
    assert WebFetchTool not in pro.profile.get('supported_native_tools', frozenset())
    assert WebFetchTool in flash.profile.get('supported_native_tools', frozenset())
    assert openai_flash.unsupported_native_tool_settings(restricted) == frozenset()
    assert pro.unsupported_native_tool_settings(restricted) == frozenset()
    assert flash.unsupported_native_tool_settings(restricted) == frozenset({'allowed_domains'})

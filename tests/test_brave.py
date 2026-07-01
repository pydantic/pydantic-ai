from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from pydantic_ai._run_context import RunContext
from pydantic_ai.common_tools.brave import (
    BraveSearchToolset,
    brave_image_search_tool,
    brave_llm_context_tool,
    brave_local_descriptions_tool,
    brave_local_pois_tool,
    brave_news_search_tool,
    brave_place_search_tool,
    brave_rich_search_tool,
    brave_video_search_tool,
    brave_web_search_tool,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

pytestmark = pytest.mark.anyio


@pytest.fixture
def run_context() -> RunContext[None]:
    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


def test_factory_requires_api_key():
    with pytest.raises(ValueError, match='api_key must be provided'):
        brave_web_search_tool()


async def test_web_search_get_request_headers_params_and_location(run_context: RunContext[None]):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'type': 'search',
                'query': {'original': 'pydantic ai'},
                'web': {
                    'type': 'search',
                    'results': [
                        {
                            'type': 'search_result',
                            'title': 'Pydantic AI',
                            'url': 'https://ai.pydantic.dev/',
                            'description': 'Agent framework',
                        }
                    ],
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = brave_web_search_tool(
            'brave-key',
            client=client,
            count=2,
            goggles=['goggle-a', 'goggle-b'],
            loc_lat=37.7749,
            loc_long=-122.4194,
        )
        result = await tool.function_schema.call({'query': 'pydantic ai', 'freshness': 'pd'}, run_context)

    request = requests[0]
    assert request.method == 'GET'
    assert request.url.path == '/res/v1/web/search'
    assert request.headers['accept'] == 'application/json'
    assert request.headers['accept-encoding'] == 'gzip'
    assert request.headers['x-subscription-token'] == 'brave-key'
    assert request.headers['x-loc-lat'] == '37.7749'
    assert request.headers['x-loc-long'] == '-122.4194'
    assert request.url.params.multi_items() == [
        ('q', 'pydantic ai'),
        ('country', 'US'),
        ('search_lang', 'en'),
        ('ui_lang', 'en-US'),
        ('count', '2'),
        ('offset', '0'),
        ('safesearch', 'moderate'),
        ('freshness', 'pd'),
        ('text_decorations', 'true'),
        ('spellcheck', 'true'),
        ('goggles', 'goggle-a'),
        ('goggles', 'goggle-b'),
        ('enable_rich_callback', 'false'),
        ('include_fetch_metadata', 'false'),
        ('operators', 'true'),
    ]
    assert result['web']['results'][0]['title'] == 'Pydantic AI'


async def test_supported_post_endpoints_send_json_body(run_context: RunContext[None]):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'type': 'news',
                'query': {'original': 'ai'},
                'results': [{'type': 'news_result', 'title': 'AI News', 'url': 'https://example.com/news'}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = brave_news_search_tool('brave-key', client=client, method='POST', count=1, goggles=['trusted'])
        result = await tool.function_schema.call({'query': 'ai'}, run_context)

    request = requests[0]
    assert request.method == 'POST'
    assert request.url.path == '/res/v1/news/search'
    assert json.loads(request.content) == {
        'q': 'ai',
        'country': 'US',
        'search_lang': 'en',
        'ui_lang': 'en-US',
        'count': 1,
        'offset': 0,
        'safesearch': 'strict',
        'spellcheck': True,
        'goggles': ['trusted'],
        'operators': True,
        'include_fetch_metadata': False,
    }
    assert result['results'][0]['title'] == 'AI News'


@pytest.mark.parametrize(
    'factory,args,path,response,expected_type',
    [
        (
            brave_image_search_tool,
            {'query': 'mountains'},
            '/res/v1/images/search',
            {'type': 'images', 'results': [{'type': 'image_result', 'title': 'Mountain'}]},
            'images',
        ),
        (
            brave_video_search_tool,
            {'query': 'python tutorial'},
            '/res/v1/videos/search',
            {'type': 'videos', 'results': [{'type': 'video_result', 'title': 'Python'}]},
            'videos',
        ),
        (
            brave_llm_context_tool,
            {'query': 'pydantic ai'},
            '/res/v1/llm/context',
            {'grounding': {'generic': [{'url': 'https://ai.pydantic.dev/', 'snippets': ['context']}]}, 'sources': {}},
            None,
        ),
        (
            brave_place_search_tool,
            {'query': 'coffee', 'latitude': 37.7749, 'longitude': -122.4194},
            '/res/v1/local/place_search',
            {'type': 'locations', 'results': [{'type': 'location_result', 'title': 'Coffee'}]},
            'locations',
        ),
        (
            brave_local_descriptions_tool,
            {'ids': ['loc1']},
            '/res/v1/local/descriptions',
            {
                'type': 'local_descriptions',
                'results': [{'type': 'local_description', 'id': 'loc1', 'description': 'Nice.'}],
            },
            'local_descriptions',
        ),
        (
            brave_rich_search_tool,
            {'callback_key': 'callback'},
            '/res/v1/web/rich',
            {'type': 'rich', 'results': [{'weather': {'temperature': 20}}]},
            'rich',
        ),
    ],
)
async def test_endpoint_paths_and_typed_responses(
    factory: Callable[..., object],
    args: dict[str, object],
    path: str,
    response: dict[str, object],
    expected_type: str | None,
    run_context: RunContext[None],
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = factory('brave-key', client=client)
        result = await tool.function_schema.call(args, run_context)  # type: ignore[attr-defined]

    assert requests[0].method == 'GET'
    assert requests[0].url.path == path
    if expected_type is not None:
        assert result['type'] == expected_type


async def test_local_pois_repeats_ids_and_sends_location_headers(run_context: RunContext[None]):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                'type': 'local_pois',
                'results': [
                    {'type': 'location_result', 'id': 'loc1', 'title': 'One'},
                    {'type': 'location_result', 'id': 'loc2', 'title': 'Two'},
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = brave_local_pois_tool('brave-key', client=client, loc_lat=1.25, loc_long=2.5)
        result = await tool.function_schema.call({'ids': ['loc1', 'loc2'], 'units': 'metric'}, run_context)

    assert requests[0].url.path == '/res/v1/local/pois'
    assert requests[0].url.params.multi_items() == [
        ('ids', 'loc1'),
        ('ids', 'loc2'),
        ('search_lang', 'en'),
        ('ui_lang', 'en-US'),
        ('units', 'metric'),
    ]
    assert requests[0].headers['x-loc-lat'] == '1.25'
    assert requests[0].headers['x-loc-long'] == '2.5'
    assert [item['id'] for item in result['results']] == ['loc1', 'loc2']


def test_bound_params_hidden_from_schema():
    tool = brave_web_search_tool(
        'brave-key',
        count=5,
        country='GB',
        goggles=['trusted'],
        loc_lat=51.5072,
        loc_long=-0.1276,
    )
    properties = tool.function_schema.json_schema['properties']

    assert 'query' in properties
    assert 'freshness' in properties
    assert 'count' not in properties
    assert 'country' not in properties
    assert 'goggles' not in properties
    assert 'loc_lat' not in properties
    assert 'loc_long' not in properties


def test_unbound_params_visible_in_schema():
    tool = brave_web_search_tool('brave-key')
    properties = tool.function_schema.json_schema['properties']

    assert 'query' in properties
    assert 'count' in properties
    assert 'goggles' in properties
    assert 'loc_lat' in properties
    assert 'loc_long' in properties


def test_toolset_include_flags():
    toolset = BraveSearchToolset(
        'brave-key',
        include_web_search=True,
        include_news_search=False,
        include_image_search=False,
        include_video_search=False,
        include_llm_context=False,
        include_place_search=False,
        include_local_pois=False,
        include_local_descriptions=False,
        include_rich_search=True,
    )

    assert set(toolset.tools) == {'brave_web_search', 'brave_rich_search'}


def test_toolset_all_disabled():
    toolset = BraveSearchToolset(
        'brave-key',
        include_web_search=False,
        include_news_search=False,
        include_image_search=False,
        include_video_search=False,
        include_llm_context=False,
        include_place_search=False,
        include_local_pois=False,
        include_local_descriptions=False,
        include_rich_search=False,
    )

    assert set(toolset.tools) == set()

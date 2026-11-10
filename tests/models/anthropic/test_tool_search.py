"""Tests for tool search on the legacy `AsyncAnthropicBedrock` (InvokeModel) client."""

from __future__ import annotations as _annotations

import os
from typing import Literal

import httpx2
import pytest
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import Agent, ModelHTTPError, ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.capabilities import ToolSearch
from pydantic_ai.messages import NativeToolSearchCallPart, NativeToolSearchReturnPart

from ..._inline_snapshot import snapshot
from ...conftest import RequestCapture, try_import
from ..conftest import content_blocks, json_objects

with try_import() as imports_successful:
    from anthropic import AsyncAnthropicBedrock

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='anthropic not installed')


@pytest.mark.vcr()
async def test_anthropic_tool_search_bm25_on_legacy_bedrock(
    allow_model_requests: None, request_capture: RequestCapture
):
    """An explicit `bm25` strategy on the legacy Bedrock InvokeModel API sends the `bm25` variant and replays it as recorded.

    InvokeModel accepts `tool_search_tool_bm25_20251119` without a beta header on Sonnet 4.6:
    https://github.com/pydantic/pydantic-ai/issues/9911. The second run replays the first run's search.
    """
    # `AsyncAnthropicBedrock`'s SigV4 signer imports `botocore`, which only ships under the `bedrock` extra.
    pytest.importorskip('botocore')

    client = AsyncAnthropicBedrock(
        aws_access_key=os.environ.get('AWS_ACCESS_KEY_ID', 'test-access-key'),
        aws_secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY', 'test-secret-key'),
        aws_session_token=os.environ.get('AWS_SESSION_TOKEN'),
        aws_region='us-east-1',
        http_client=request_capture.client,
    )
    agent = Agent(
        AnthropicModel('us.anthropic.claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client)),
        capabilities=[ToolSearch(strategy='bm25')],
    )

    @agent.tool_plain(defer_loading=True)
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f'Sunny, 21C in {city}'

    @agent.tool_plain(defer_loading=True)
    def search_files(query: str) -> str:
        """Search through files in the workspace."""
        return 'No files found.'

    result = await agent.run('What is the weather in Paris?')
    await agent.run('Now search my files for the quarterly report.', message_history=result.all_messages())

    bodies = request_capture.bodies()
    assert [body.get('anthropic_beta') for body in bodies] == [None, None, None, None]
    assert [
        [tool for tool in json_objects(body['tools']) if str(tool['name']).startswith('tool_search_tool_')]
        for body in bodies
    ] == snapshot(
        [
            [{'type': 'tool_search_tool_bm25_20251119', 'name': 'tool_search_tool_bm25'}],
            [{'type': 'tool_search_tool_bm25_20251119', 'name': 'tool_search_tool_bm25'}],
            [{'type': 'tool_search_tool_bm25_20251119', 'name': 'tool_search_tool_bm25'}],
            [{'type': 'tool_search_tool_bm25_20251119', 'name': 'tool_search_tool_bm25'}],
        ]
    )
    assert content_blocks(bodies[-1], 'server_tool_use') == snapshot(
        [
            {
                'id': 'srvtoolu_bdrk_01Tg8SnCY2u9tPguiJz7RsNS',
                'type': 'server_tool_use',
                'name': 'tool_search_tool_bm25',
                'input': {'query': 'weather'},
            },
            {
                'id': 'srvtoolu_bdrk_01Cb5xF4nXEPyjntbiTBBNed',
                'type': 'server_tool_use',
                'name': 'tool_search_tool_bm25',
                'input': {'query': 'search files'},
            },
        ]
    )


@pytest.mark.parametrize(
    'recorded,strategy,expected',
    [
        pytest.param(
            'bm25',
            None,
            {'name': 'tool_search_tool_regex', 'input': {'pattern': 'weather forecast'}},
            id='unset',
        ),
        pytest.param(
            'bm25',
            'regex',
            {'name': 'tool_search_tool_regex', 'input': {'pattern': 'weather forecast'}},
            id='regex',
        ),
        pytest.param(
            'bm25',
            'bm25',
            {'name': 'tool_search_tool_bm25', 'input': {'query': 'weather forecast'}},
            id='bm25',
        ),
        pytest.param(
            'regex',
            'bm25',
            {'name': 'tool_search_tool_regex', 'input': {'pattern': 'weather forecast'}},
            id='recorded-regex-bm25',
        ),
    ],
)
async def test_anthropic_legacy_bedrock_bm25_history_replay(
    allow_model_requests: None,
    recorded: Literal['bm25', 'regex'],
    strategy: Literal['bm25', 'regex'] | None,
    expected: dict[str, JsonValue],
):
    """On `AsyncAnthropicBedrock`, a recorded `bm25` search replays as `regex` unless the request sends `bm25` explicitly.

    A recorded `regex` search stays `regex`, even when the request sends `bm25`.

    The history stands in for one recorded on another Anthropic transport. A mock transport stands in for a
    cassette because the claim is the replayed request, not the response.
    """
    bodies: list[dict[str, JsonValue]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(TypeAdapter(dict[str, JsonValue]).validate_json(request.content))
        return httpx2.Response(400, json={'message': 'captured'})

    # A bearer token skips SigV4 signing, which would need `botocore` from the `bedrock` extra.
    async with AsyncAnthropicBedrock(
        api_key='x',
        aws_region='us-east-1',
        base_url='http://localhost',
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    ) as client:
        agent = Agent(
            AnthropicModel('claude-sonnet-4-6', provider=AnthropicProvider(anthropic_client=client)),
            capabilities=[ToolSearch(strategy=strategy)],
        )

        @agent.tool_plain(defer_loading=True)
        def get_weather(city: str) -> str:
            raise NotImplementedError

        history: list[ModelMessage] = [
            ModelRequest.user_text_prompt('What is the weather in Paris?'),
            ModelResponse(
                parts=[
                    NativeToolSearchCallPart(
                        args={'queries': ['weather forecast']},
                        tool_call_id='srvtoolu_1',
                        provider_name='anthropic',
                        provider_details={'strategy': recorded},
                    ),
                    NativeToolSearchReturnPart(
                        tool_call_id='srvtoolu_1',
                        provider_name='anthropic',
                        content={'discovered_tools': [{'name': 'get_weather'}]},
                    ),
                ],
                provider_name='anthropic',
            ),
        ]
        with pytest.raises(ModelHTTPError):
            await agent.run('And in Rome?', message_history=history)

    [body] = bodies
    assert [{key: block[key] for key in ('name', 'input')} for block in content_blocks(body, 'server_tool_use')] == [
        expected
    ]

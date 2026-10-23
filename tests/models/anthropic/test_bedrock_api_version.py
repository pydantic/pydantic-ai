"""The `anthropic_version` the legacy Bedrock client sends in the request body.

`anthropic>=2` defaults it to the Messages API version `2026-10-21`, which Bedrock rejects with
`Invalid API version` for Claude Sonnet 4.5 and Opus 4.5, so `AnthropicModel` pins the version
`anthropic<2` sent. Mocked at the HTTP transport rather than recorded, because Bedrock can't be
recorded here; going through the real SDK client checks the pin survives its request rewriting.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic, AsyncAnthropicBedrock

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]

_MESSAGE_JSON = {
    'id': 'msg_0123',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-4-5',
    'content': [{'type': 'text', 'text': 'OK'}],
    'stop_reason': 'end_turn',
    'usage': {'input_tokens': 5, 'output_tokens': 1},
}


async def _request_body(settings: ModelSettings, *, bedrock: bool) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_MESSAGE_JSON)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        client = (
            AsyncAnthropicBedrock(api_key='x', aws_region='us-east-1', http_client=http_client)
            if bedrock
            else AsyncAnthropic(api_key='x', http_client=http_client)
        )
        model = AnthropicModel(
            'us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=AnthropicProvider(anthropic_client=client)
        )
        # A small `max_tokens` keeps the request non-streaming, matching the mocked JSON response.
        await Agent(model, model_settings={'max_tokens': 256, **settings}).run('Hello')

    [body] = bodies
    return body


@pytest.mark.parametrize(
    'settings,expected',
    [
        pytest.param({}, 'bedrock-2023-05-31', id='default'),
        # Alongside the sampling settings, which ride in `extra_body` too.
        pytest.param({'temperature': 0.5}, 'bedrock-2023-05-31', id='with-sampling-setting'),
        pytest.param({'extra_body': {'metadata': {'keep': True}}}, 'bedrock-2023-05-31', id='with-other-extra-body'),
        # An explicit `extra_body` entry still wins, as it does over every setting merged there.
        pytest.param({'extra_body': {'anthropic_version': '2026-10-21'}}, '2026-10-21', id='explicit-extra-body'),
    ],
)
async def test_legacy_bedrock_pins_anthropic_version(
    allow_model_requests: None, settings: ModelSettings, expected: str
):
    body = await _request_body(settings, bedrock=True)
    assert body['anthropic_version'] == expected


async def test_direct_api_sends_no_body_anthropic_version(allow_model_requests: None):
    """The direct API takes its version from the `anthropic-version` header, so the pin stays off its body."""
    body = await _request_body({}, bedrock=False)
    assert 'anthropic_version' not in body

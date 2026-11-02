from __future__ import annotations as _annotations

import json

import httpx2
import pytest
from pydantic import BaseModel, JsonValue

from pydantic_ai import Agent, ModelRequest

from ..conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.direct import model_request
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.output import NativeOutput
    from pydantic_ai.providers.together import TogetherProvider
    from pydantic_ai.settings import ModelSettings

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
]


class CityLocation(BaseModel):
    city: str
    country: str


def _completion_response(content: str) -> dict[str, JsonValue]:
    return {
        'id': 'chatcmpl-together-test',
        'object': 'chat.completion',
        'created': 0,
        'model': 'openai/gpt-oss-120b',
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
    }


def _capture_transport(bodies: list[dict[str, JsonValue]], content: str) -> httpx2.MockTransport:
    """Transport recording every request body and answering with a canned completion.

    MockTransport stands in for a cassette: the assertions pin the outgoing body, and recording a
    real cassette would need a live Together API key.
    """

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.read()))
        return httpx2.Response(200, json=_completion_response(content))

    return httpx2.MockTransport(handler)


async def test_together_native_output_builds_json_schema_response_format(allow_model_requests: None):
    """`NativeOutput` builds a `json_schema` response_format for a Together model whose family profile claims it.

    The claim is the OpenAI family's, not a Together-wide one: the provider-layer
    `supports_json_schema_output`/`supports_json_object_output` flags stay gated on a live Together probe.
    """
    bodies: list[dict[str, JsonValue]] = []

    async with httpx2.AsyncClient(
        transport=_capture_transport(bodies, '{"city": "Mexico City", "country": "Mexico"}')
    ) as http_client:
        provider = TogetherProvider(api_key='test', http_client=http_client)
        model = OpenAIChatModel('openai/gpt-oss-120b', provider=provider)
        agent = Agent(model, output_type=NativeOutput(CityLocation))

        result = await agent.run('What is the capital of Mexico?')

    assert result.output == CityLocation(city='Mexico City', country='Mexico')
    response_format = bodies[0]['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['strict'] is True
    assert response_format['json_schema']['schema']['properties'] == {
        'city': {'type': 'string'},
        'country': {'type': 'string'},
    }


async def test_together_max_tokens_and_thinking_reach_request_body(allow_model_requests: None):
    """`max_tokens` and the unified `thinking` setting reach the Together request body.

    `max_tokens` maps to `max_completion_tokens` under the default profile flag; thinking maps to graded
    `reasoning_effort` for `gpt-oss-*`, with `False` clamping to the lowest documented level.
    """
    bodies: list[dict[str, JsonValue]] = []

    async with httpx2.AsyncClient(transport=_capture_transport(bodies, 'ok')) as http_client:
        provider = TogetherProvider(api_key='test', http_client=http_client)
        model = OpenAIChatModel('openai/gpt-oss-120b', provider=provider)

        await model_request(
            model,
            [ModelRequest.user_text_prompt('hello')],
            model_settings=ModelSettings(max_tokens=100, thinking='medium'),
            model_request_parameters=ModelRequestParameters(),
        )
        await model_request(
            model,
            [ModelRequest.user_text_prompt('hello')],
            model_settings=ModelSettings(max_tokens=100, thinking=False),
            model_request_parameters=ModelRequestParameters(),
        )

    assert bodies[0]['max_completion_tokens'] == 100
    assert bodies[0]['reasoning_effort'] == 'medium'
    assert bodies[1]['reasoning_effort'] == 'low'

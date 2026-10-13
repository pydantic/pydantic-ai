from __future__ import annotations as _annotations

import json

import httpx2
import pytest
from pydantic import BaseModel

from pydantic_ai import Agent, NativeOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.fireworks import FireworksProvider


class City(BaseModel):
    name: str
    country: str


def test_native_output_sends_json_schema_response_format(allow_model_requests: None) -> None:
    """`NativeOutput` must reach Fireworks as a json_schema response_format, including unprofiled model names."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        content = '{"name": "Paris", "country": "France"}'
        return httpx2.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': 'accounts/fireworks/models/glm-5p3',
                'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}
                ],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
            },
        )

    provider = FireworksProvider(
        api_key='dummy',
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )
    model = OpenAIChatModel('accounts/fireworks/models/glm-5p3', provider=provider)
    agent = Agent(model, output_type=NativeOutput(City))

    result = agent.run_sync('What is the capital of France?')

    assert result.output == City(name='Paris', country='France')
    assert [r.method for r in requests] == ['POST']
    body = json.loads(requests[0].content)
    assert body['model'] == 'accounts/fireworks/models/glm-5p3'
    response_format = body['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['name'] == 'City'
    assert set(response_format['json_schema']['schema']['properties']) == {'name', 'country'}


def test_fireworks_native_output_reaches_endpoint(allow_model_requests: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The `fireworks:` string entry point must resolve to a profile that lets `NativeOutput` reach the endpoint."""
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        content = '{"name": "Paris", "country": "France"}'
        return httpx2.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': 'accounts/fireworks/models/kimi-k3',
                'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}
                ],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 10, 'total_tokens': 20},
            },
        )

    def recording_client() -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))

    monkeypatch.setattr('pydantic_ai.providers._openai_compatible.create_async_httpx2_client', recording_client)
    monkeypatch.setenv('FIREWORKS_API_KEY', 'dummy')

    agent = Agent('fireworks:accounts/fireworks/models/kimi-k3', output_type=NativeOutput(City))
    result = agent.run_sync('What is the capital of France?')

    assert result.output == City(name='Paris', country='France')
    assert [r.method for r in requests] == ['POST']
    assert requests[0].url.path == '/inference/v1/chat/completions'
    body = json.loads(requests[0].content)
    assert body['model'] == 'accounts/fireworks/models/kimi-k3'
    response_format = body['response_format']
    assert response_format['type'] == 'json_schema'
    assert response_format['json_schema']['name'] == 'City'
    assert set(response_format['json_schema']['schema']['properties']) == {'name', 'country'}

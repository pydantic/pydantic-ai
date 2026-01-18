from __future__ import annotations as _annotations

import json
from typing import Any

import httpx
import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.models.databricks import DatabricksModel
from pydantic_ai.providers.databricks import DatabricksProvider
from pydantic_ai.result import RunUsage

from ..conftest import try_import

with try_import() as imports_successful:
    pass

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.anyio,
]


class MockDatabricksTransport(httpx.AsyncBaseTransport):
    def __init__(self, responses: list[dict[str, Any] | list[str]] | None = None, stream: bool = False):
        self.responses = responses or []
        self.request_count = 0
        self.stream = stream

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.request_count += 1
        if self.responses:
            content = self.responses.pop(0)
        else:
            # Default response
            if self.stream:
                common_def = '"created": 123, "model": "default", "object": "chat.completion.chunk"'
                content = [
                    f'data: {{"id":"default", {common_def}, "choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":null}}]}}\n\n',
                    f'data: {{"id":"default", {common_def}, "choices":[{{"index":0,"delta":{{"content":"Default"}}}}]}}\n\n',
                    'data: [DONE]\n\n',
                ]
            else:
                content = {
                    'id': 'chatcmpl-default',
                    'object': 'chat.completion',
                    'created': 1677652288,
                    'model': 'databricks-dbrx-instruct',
                    'choices': [
                        {
                            'index': 0,
                            'message': {
                                'role': 'assistant',
                                'content': [{'type': 'text', 'text': 'Hello from Databricks!'}],
                            },
                            'finish_reason': 'stop',
                        }
                    ],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30},
                }

        if self.stream:
            if not isinstance(content, list):
                raise ValueError(f'Stream content must be list of strings, got {type(content)}')

            async def stream_content():
                for chunk in content:
                    if isinstance(chunk, str):
                        yield chunk.encode('utf-8')
                    else:
                        yield json.dumps(chunk).encode('utf-8')

            return httpx.Response(200, content=stream_content())

        return httpx.Response(200, json=content)


async def test_databricks_model_list_content(allow_model_requests: None):
    client = httpx.AsyncClient(transport=MockDatabricksTransport())
    provider = DatabricksProvider(api_key='foo', base_url='https://example.com', http_client=client)
    model = DatabricksModel('databricks-dbrx-instruct', provider=provider)
    agent = Agent(model)

    result = await agent.run('Hello')
    assert result.output == 'Hello from Databricks!'
    assert result.usage() == snapshot(RunUsage(requests=1, input_tokens=10, output_tokens=20))


async def test_databricks_model_string_content(allow_model_requests: None):
    # Test standard string content just in case
    response = {
        'id': 'chatcmpl-124',
        'object': 'chat.completion',
        'created': 1677652299,
        'model': 'databricks-dbrx-instruct',
        'choices': [
            {'index': 0, 'message': {'role': 'assistant', 'content': 'Hello string world!'}, 'finish_reason': 'stop'}
        ],
        'usage': {'prompt_tokens': 5, 'completion_tokens': 5, 'total_tokens': 10},
    }

    client = httpx.AsyncClient(transport=MockDatabricksTransport([response]))
    provider = DatabricksProvider(api_key='foo', base_url='https://example.com', http_client=client)
    model = DatabricksModel('databricks-dbrx-instruct', provider=provider)
    agent = Agent(model)

    result = await agent.run('Hello')
    assert result.output == 'Hello string world!'


def test_infer_model_databricks():
    from pydantic_ai.models import infer_model
    from pydantic_ai.models.databricks import DatabricksModel

    model = infer_model('databricks:my-model')
    assert isinstance(model, DatabricksModel)
    assert model.model_name == 'my-model'


async def test_databricks_model_streaming_list_content(allow_model_requests: None):
    # Test streaming where delta content is a list
    common = '"created": 1677652288, "model": "databricks-dbrx-instruct", "object": "chat.completion.chunk"'
    chunks = [
        f'data: {{"id":"chatcmpl-1", {common}, "choices":[{{"index":0,"delta":{{"role":"assistant","content":""}},"finish_reason":null}}]}}\n\n',
        f'data: {{"id":"chatcmpl-1", {common}, "choices":[{{"index":0,"delta":{{"content":[{{"type":"text","text":"Stream"}}]}}}}]}}\n\n',
        f'data: {{"id":"chatcmpl-1", {common}, "choices":[{{"index":0,"delta":{{"content":[{{"type":"text","text":"ing"}}]}}}}]}}\n\n',
        'data: [DONE]\n\n',
    ]

    client = httpx.AsyncClient(transport=MockDatabricksTransport(responses=[chunks], stream=True))
    provider = DatabricksProvider(api_key='foo', base_url='https://example.com', http_client=client)
    model = DatabricksModel('databricks-dbrx-instruct', provider=provider)
    agent = Agent(model)

    async with agent.run_stream('Hello') as result:
        output = ''
        async for chunk in result.stream_text():
            output += chunk
        assert output == 'Streaming'

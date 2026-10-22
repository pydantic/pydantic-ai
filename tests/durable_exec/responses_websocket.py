"""An OpenAI Responses model asking for the WebSocket transport, for checking that durable model units use HTTP."""

from __future__ import annotations

from collections.abc import AsyncIterable

import httpx2
from pydantic import JsonValue, TypeAdapter

from pydantic_ai import AgentStreamEvent, RunContext
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

Frame = dict[str, JsonValue]
_FRAME_ADAPTER = TypeAdapter(Frame)
_COMPLETED: Frame = {
    'id': 'resp_http',
    'model': 'gpt-4o',
    'object': 'response',
    'created_at': 1704067200,
    'status': 'completed',
    'output': [
        {
            'type': 'message',
            'id': 'msg_http',
            'role': 'assistant',
            'status': 'completed',
            'content': [{'type': 'output_text', 'text': 'from HTTP', 'annotations': [], 'logprobs': []}],
        }
    ],
    'parallel_tool_calls': True,
    'tool_choice': 'auto',
    'tools': [],
    'usage': {'input_tokens': 5, 'output_tokens': 1, 'total_tokens': 6},
}
_STREAM: list[Frame] = [
    {
        'type': 'response.created',
        'sequence_number': 0,
        'response': {**_COMPLETED, 'status': 'in_progress', 'output': []},
    },
    {
        'type': 'response.output_item.added',
        'sequence_number': 1,
        'output_index': 0,
        'item': {'type': 'message', 'id': 'msg_http', 'role': 'assistant', 'status': 'in_progress', 'content': []},
    },
    {
        'type': 'response.output_text.delta',
        'sequence_number': 2,
        'output_index': 0,
        'content_index': 0,
        'item_id': 'msg_http',
        'delta': 'from HTTP',
        'logprobs': [],
    },
    {'type': 'response.completed', 'sequence_number': 3, 'response': _COMPLETED},
]


def responses_reply(request: httpx2.Request) -> httpx2.Response:
    """Answer an HTTP Responses API request with `from HTTP`, streamed when it asks for a stream."""
    if not _FRAME_ADAPTER.validate_json(request.content).get('stream'):
        return httpx2.Response(200, json=_COMPLETED)
    body = ''.join(f'event: {event["type"]}\ndata: {_FRAME_ADAPTER.dump_json(event).decode()}\n\n' for event in _STREAM)
    return httpx2.Response(200, content=body.encode(), headers={'content-type': 'text/event-stream'})


def websocket_model(http_client: httpx2.AsyncClient) -> OpenAIResponsesModel:
    # The cassette fixture lets requests to localhost through to the mock transport.
    return OpenAIResponsesModel(
        'gpt-4o',
        provider=OpenAIProvider(base_url='http://localhost/v1', api_key='test', http_client=http_client),
        settings=OpenAIResponsesModelSettings(openai_transport='websocket'),
    )


async def drain_events(ctx: RunContext[None], stream: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in stream:
        pass

"""Tests for serving an agent as an OpenAI Responses API endpoint (`Agent.to_responses()`)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI
    from openai.types.responses import Response, ResponseOutputMessage, ResponseStreamEvent
    from starlette.applications import Starlette

    from pydantic_ai._responses._events import encode_sse, response_event_stream
    from pydantic_ai._responses._messages import load_messages
    from pydantic_ai._responses.request_types import ResponsesRequest

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='starlette/openai not installed'),
]


async def post(app: Starlette, body: dict[str, Any]) -> httpx.Response:
    transport = httpx.ASGITransport(app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        return await client.post('/v1/responses', json=body)


async def stream_events(app: Starlette, body: dict[str, Any]) -> list[dict[str, Any]]:
    transport = httpx.ASGITransport(app)
    events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        async with client.stream('POST', '/v1/responses', json={**body, 'stream': True}) as response:
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/event-stream')
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    events.append(json.loads(line[len('data: ') :]))
    return events


async def test_non_streaming_string_input():
    """A bare-string `input` runs the agent and returns a valid `Response` with the assistant text."""
    agent = Agent(TestModel(custom_output_text='The sun rises in the east.'))
    response = await post(agent.to_responses(), {'model': 'gpt-x', 'input': 'where does the sun rise?'})

    assert response.status_code == 200
    body = response.json()
    parsed = Response.model_validate(body)
    assert parsed.output_text == 'The sun rises in the east.'
    assert parsed.status == 'completed'
    assert parsed.model == 'gpt-x'
    assert parsed.id.startswith('resp_')
    message = parsed.output[0]
    assert isinstance(message, ResponseOutputMessage) and message.id.startswith('msg_')
    assert parsed.usage is not None and parsed.usage.output_tokens > 0
    assert parsed.error is None


async def test_non_streaming_message_list_input():
    """A list `input` of role messages is replayed as history; the last user turn drives the run."""
    captured: list[ModelMessage] = []

    async def capture(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        captured.extend(messages)
        yield 'ack'

    agent = Agent(FunctionModel(stream_function=capture))
    response = await post(
        agent.to_responses(),
        {
            'model': 'gpt-x',
            'input': [
                {'role': 'user', 'content': 'first question'},
                {'role': 'assistant', 'content': 'first answer'},
                {'role': 'user', 'content': [{'type': 'input_text', 'text': 'second question'}]},
            ],
        },
    )

    assert response.status_code == 200
    assert Response.model_validate(response.json()).output_text == 'ack'
    # The replayed `input` history reaches the model as alternating request/response messages.
    replayed = [(type(p).__name__, getattr(p, 'content', None)) for m in captured for p in m.parts]
    assert replayed == snapshot(
        [
            ('UserPromptPart', 'first question'),
            ('TextPart', 'first answer'),
            ('UserPromptPart', 'second question'),
        ]
    )


async def test_streaming_event_sequence():
    """Streaming emits the canonical Responses event sequence and the deltas concatenate to the text."""
    agent = Agent(TestModel(custom_output_text='Hello world'))
    events = await stream_events(agent.to_responses(), {'model': 'gpt-x', 'input': 'hi'})

    assert [e['type'] for e in events] == snapshot(
        [
            'response.created',
            'response.in_progress',
            'response.output_item.added',
            'response.content_part.added',
            'response.output_text.delta',
            'response.output_text.delta',
            'response.output_text.done',
            'response.content_part.done',
            'response.output_item.done',
            'response.completed',
        ]
    )

    assert [e['sequence_number'] for e in events] == list(range(len(events)))

    text = ''.join(e['delta'] for e in events if e['type'] == 'response.output_text.delta')
    assert text == 'Hello world'

    completed = events[-1]
    assert completed['type'] == 'response.completed'
    assert completed['response']['status'] == 'completed'
    assert completed['response']['output'][0]['content'][0]['text'] == 'Hello world'


async def collect(events: AsyncIterator[ResponseStreamEvent]) -> list[dict[str, Any]]:
    return [json.loads(encode_sse(event).removeprefix('data: ')) async for event in events]


async def test_event_stream_text_part_with_content():
    """Text that arrives as a `PartStartEvent` with content (not just deltas) is streamed as a delta."""

    async def event_generator():
        yield PartStartEvent(index=0, part=TextPart(content='Hello'))
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta=' world'))
        yield PartEndEvent(index=0, part=TextPart(content='Hello world'))

    events = await collect(
        response_event_stream(
            event_generator(), model='gpt-x', response_id='resp_1', message_id='msg_1', created_at=0.0
        )
    )

    deltas = [e['delta'] for e in events if e['type'] == 'response.output_text.delta']
    assert deltas == snapshot(['Hello', ' world'])
    assert events[-1]['response']['output'][0]['content'][0]['text'] == 'Hello world'


async def test_event_stream_no_text_skips_item():
    """A run that produces no text emits no output item, going straight to `response.completed`."""

    async def event_generator():
        return
        yield  # pragma: no cover

    events = await collect(
        response_event_stream(
            event_generator(), model='gpt-x', response_id='resp_1', message_id='msg_1', created_at=0.0
        )
    )

    assert [e['type'] for e in events] == snapshot(['response.created', 'response.in_progress', 'response.completed'])
    assert events[-1]['response']['output'] == []


async def test_options_preflight():
    """An `OPTIONS` preflight request is answered for CORS."""
    transport = httpx.ASGITransport(Agent(TestModel()).to_responses())
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.request('OPTIONS', '/v1/responses')
    assert response.status_code == 200


async def test_tools_run_server_side():
    """Internal tool calls execute server-side and are not surfaced; only the final text is returned."""
    agent = Agent(TestModel())

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    response = await post(agent.to_responses(), {'model': 'gpt-x', 'input': 'weather?'})
    parsed = Response.model_validate(response.json())
    # The model called the tool, but the endpoint exposes only the assistant message output.
    assert all(item.type == 'message' for item in parsed.output)


async def boom(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise RuntimeError('kaboom')
    yield ''  # pragma: no cover


async def test_streaming_error_emits_failed_event():
    """An error during a streaming run is reported as a terminal `response.failed` event."""
    agent = Agent(FunctionModel(stream_function=boom))
    events = await stream_events(agent.to_responses(), {'model': 'gpt-x', 'input': 'hi'})

    assert [e['type'] for e in events] == snapshot(['response.created', 'response.in_progress', 'response.failed'])
    failed = events[-1]['response']
    assert failed['status'] == 'failed'
    assert failed['error'] == {'code': 'server_error', 'message': 'kaboom'}


async def test_non_streaming_error_returns_failed_response():
    """An error during a non-streaming run yields a well-formed failed `Response`, not a raw 500."""
    agent = Agent(FunctionModel(stream_function=boom))
    response = await post(agent.to_responses(), {'model': 'gpt-x', 'input': 'hi'})

    assert response.status_code == 200
    parsed = Response.model_validate(response.json())
    assert parsed.status == 'failed'
    assert parsed.error is not None and parsed.error.message == 'kaboom'
    assert parsed.output == []


async def test_invalid_request_body_returns_422():
    """A request body missing the required `input` field is rejected with 422."""
    agent = Agent(TestModel())
    response = await post(agent.to_responses(), {'model': 'gpt-x'})
    assert response.status_code == 422


async def test_load_messages_round_trips_tool_calls():
    """`load_messages` reconstructs tool call/output pairs from replayed `input` items."""
    request = ResponsesRequest.model_validate(
        {
            'input': [
                {'role': 'system', 'content': 'be nice'},
                {'role': 'user', 'content': 'weather in Paris?'},
                {'type': 'function_call', 'call_id': 'c1', 'name': 'get_weather', 'arguments': '{"city":"Paris"}'},
                {'type': 'function_call_output', 'call_id': 'c1', 'output': 'sunny'},
                {'role': 'assistant', 'content': 'It is sunny in Paris.'},
                {'type': 'function_call', 'call_id': 'c2', 'name': 'log', 'arguments': '{}'},
            ]
        }
    )
    messages = load_messages(request)

    assert messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='be nice', timestamp=IsDatetime()),
                    UserPromptPart(content='weather in Paris?', timestamp=IsDatetime()),
                ]
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='get_weather', args='{"city":"Paris"}', tool_call_id='c1')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name='get_weather', content='sunny', tool_call_id='c1', timestamp=IsDatetime())
                ]
            ),
            # The trailing assistant text and tool call collapse into one response message.
            ModelResponse(
                parts=[
                    TextPart(content='It is sunny in Paris.'),
                    ToolCallPart(tool_name='log', args='{}', tool_call_id='c2'),
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_openai_sdk_client_round_trip():
    """The real `openai` SDK can call the endpoint for both non-streaming and streaming requests."""
    agent = Agent(TestModel(custom_output_text='Hello from Pydantic AI.'))
    transport = httpx.ASGITransport(agent.to_responses())
    http_client = httpx.AsyncClient(transport=transport, base_url='http://test')
    client = AsyncOpenAI(base_url='http://test/v1', api_key='unused', http_client=http_client)

    response = await client.responses.create(model='gpt-x', input='hi')
    assert response.output_text == 'Hello from Pydantic AI.'

    text = ''
    async with client.responses.stream(model='gpt-x', input='hi') as stream:
        async for event in stream:
            if event.type == 'response.output_text.delta':
                text += event.delta
        final = await stream.get_final_response()
    assert text == 'Hello from Pydantic AI.'
    assert final.output_text == 'Hello from Pydantic AI.'

    await http_client.aclose()

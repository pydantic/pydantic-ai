"""Tests for serving an agent as an OpenAI Responses API endpoint (`Agent.to_openai_responses()`)."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from pydantic_ai import Agent, RunContext
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
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage, UsageLimits

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, try_import

with try_import() as imports_successful:
    from openai import AsyncOpenAI, BadRequestError, InternalServerError
    from openai.types.responses import Response, ResponseOutputMessage, ResponseStreamEvent
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import Response as StarletteResponse
    from starlette.routing import Route

    from pydantic_ai.openai_responses import handle_openai_responses_request
    from pydantic_ai.openai_responses._events import build_usage, encode_sse, response_event_stream
    from pydantic_ai.openai_responses._messages import load_messages
    from pydantic_ai.openai_responses.types import ResponsesRequest

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='starlette/openai not installed'),
]


async def post(
    app: Starlette, body: dict[str, Any], path: str = '/v1/responses', headers: dict[str, str] | None = None
) -> httpx.Response:
    transport = httpx.ASGITransport(app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        return await client.post(path, json=body, headers=headers)


async def stream_events(app: Starlette, body: dict[str, Any]) -> list[dict[str, Any]]:
    transport = httpx.ASGITransport(app)
    events: list[dict[str, Any]] = []
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        async with client.stream('POST', '/v1/responses', json={**body, 'stream': True}) as response:
            assert response.status_code == 200
            assert response.headers['content-type'].startswith('text/event-stream')
            event_type: str | None = None
            async for line in response.aiter_lines():
                if line.startswith('event: '):
                    event_type = line[len('event: ') :]
                elif line.startswith('data: '):
                    data = json.loads(line[len('data: ') :])
                    assert event_type == data['type']
                    events.append(data)
                    event_type = None
    return events


def parse_sse_frame(frame: str) -> dict[str, Any]:
    event_type: str | None = None
    data: dict[str, Any] | None = None
    for line in frame.splitlines():
        if line.startswith('event: '):
            event_type = line[len('event: ') :]
        elif line.startswith('data: '):
            data = json.loads(line[len('data: ') :])
    assert data is not None
    assert event_type == data['type']
    return data


async def collect(events: AsyncIterator[ResponseStreamEvent]) -> list[dict[str, Any]]:
    return [parse_sse_frame(encode_sse(event)) async for event in events]


async def test_non_streaming_string_input():
    """A bare-string `input` runs the agent and returns a valid `Response` with the assistant text."""
    agent = Agent(TestModel(custom_output_text='The sun rises in the east.'))
    response = await post(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'where does the sun rise?'})

    assert response.status_code == 200
    parsed = Response.model_validate(response.json())
    assert parsed.output_text == 'The sun rises in the east.'
    assert parsed.status == 'completed'
    assert parsed.model == 'test'
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
        agent.to_openai_responses(),
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
    replayed = [(type(p).__name__, getattr(p, 'content', None)) for m in captured for p in m.parts]
    assert replayed == snapshot(
        [
            ('UserPromptPart', 'first question'),
            ('TextPart', 'first answer'),
            ('UserPromptPart', 'second question'),
        ]
    )


async def test_agent_system_prompt_is_reinjected():
    """The agent's configured `system_prompt` reaches the model even though history is rebuilt from the request."""
    captured: list[list[ModelMessage]] = []

    async def capture(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        captured.append(messages)
        yield 'ok'

    agent = Agent(FunctionModel(stream_function=capture), system_prompt='You are a pirate.')

    await post(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'hi'})
    assert [(type(p).__name__, getattr(p, 'content', None)) for m in captured[0] for p in m.parts] == snapshot(
        [('SystemPromptPart', 'You are a pirate.'), ('UserPromptPart', 'hi')]
    )

    captured.clear()
    await post(
        agent.to_openai_responses(),
        {'model': 'gpt-x', 'input': [{'role': 'system', 'content': 'Be terse.'}, {'role': 'user', 'content': 'hi'}]},
    )
    assert [(type(p).__name__, getattr(p, 'content', None)) for m in captured[0] for p in m.parts] == snapshot(
        [('SystemPromptPart', 'Be terse.'), ('UserPromptPart', 'hi')]
    )


async def test_instructions_pipeline():
    """Client and server instructions are passed to the model request in order."""
    captured: list[str | None] = []

    async def capture(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        captured.append(info.instructions)
        yield 'ok'

    agent = Agent(FunctionModel(stream_function=capture))

    await post(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'hi', 'instructions': 'client'})
    assert captured == ['client']

    captured.clear()
    await post(
        agent.to_openai_responses(instructions='server'),
        {'model': 'gpt-x', 'input': 'hi', 'instructions': 'client'},
    )
    assert captured == ['server\n\nclient']


async def test_streaming_event_sequence():
    """Streaming emits the canonical Responses event sequence and the deltas concatenate to the text."""
    agent = Agent(TestModel(custom_output_text='Hello world'))
    events = await stream_events(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'hi'})

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


async def test_event_stream_text_part_with_content():
    """Text that arrives as a `PartStartEvent` with content is streamed as a delta."""

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


async def test_event_stream_delta_without_start_opens_part():
    """A text delta without a preceding start event still opens one Responses content part."""

    async def event_generator():
        yield PartDeltaEvent(index=0, delta=TextPartDelta(content_delta='Hello'))

    events = await collect(
        response_event_stream(
            event_generator(), model='gpt-x', response_id='resp_1', message_id='msg_1', created_at=0.0
        )
    )

    assert [e['type'] for e in events] == snapshot(
        [
            'response.created',
            'response.in_progress',
            'response.output_item.added',
            'response.content_part.added',
            'response.output_text.delta',
            'response.output_text.done',
            'response.content_part.done',
            'response.output_item.done',
            'response.completed',
        ]
    )


async def test_event_stream_empty_text_part_is_not_surfaced():
    """A text part that never receives content (e.g. alongside a tool call) opens no content part."""

    async def event_generator():
        yield PartStartEvent(index=0, part=TextPart(content=''))
        yield PartEndEvent(index=0, part=TextPart(content=''))
        yield PartStartEvent(index=1, part=TextPart(content='real answer'))
        yield PartEndEvent(index=1, part=TextPart(content='real answer'))

    events = await collect(
        response_event_stream(
            event_generator(), model='gpt-x', response_id='resp_1', message_id='msg_1', created_at=0.0
        )
    )

    added = [e['content_index'] for e in events if e['type'] == 'response.content_part.added']
    assert added == [0]
    assert events[-1]['response']['output'][0]['content'] == snapshot(
        [{'annotations': [], 'text': 'real answer', 'type': 'output_text', 'logprobs': None}]
    )


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


async def test_options_route_exists():
    """An `OPTIONS` request returns 200, matching the web UI route pattern."""
    transport = httpx.ASGITransport(Agent(TestModel()).to_openai_responses())
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        response = await client.request('OPTIONS', '/v1/responses')
    assert response.status_code == 200


async def test_tools_run_server_side():
    """Internal tool calls execute server-side and are not surfaced; only the final text is returned."""
    tool_calls: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'The weather is sunny.'
        else:
            yield {0: DeltaToolCall(name='get_weather', json_args='{"city":"Paris"}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        tool_calls.append(city)
        return f'sunny in {city}'

    response = await post(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'weather?'})
    parsed = Response.model_validate(response.json())

    assert tool_calls == ['Paris']
    assert parsed.output_text == 'The weather is sunny.'
    assert len(parsed.output) == 1
    assert parsed.output[0].type == 'message'


async def boom(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
    raise RuntimeError('kaboom')
    yield ''  # pragma: no cover


async def test_streaming_error_emits_failed_event():
    """An error during a streaming run is reported as a terminal `response.failed` event."""
    agent = Agent(FunctionModel(stream_function=boom))
    events = await stream_events(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'hi'})

    assert [e['type'] for e in events] == snapshot(['response.created', 'response.in_progress', 'response.failed'])
    failed = events[-1]['response']
    assert failed['status'] == 'failed'
    assert failed['error'] == {'code': 'server_error', 'message': 'kaboom'}


async def test_non_streaming_error_returns_500():
    """An error during a non-streaming run is reported as an OpenAI HTTP error."""
    agent = Agent(FunctionModel(stream_function=boom))
    app = agent.to_openai_responses()

    response = await post(app, {'model': 'gpt-x', 'input': 'hi'})
    assert response.status_code == 500
    assert response.json()['error'] == {
        'message': 'kaboom',
        'type': 'server_error',
        'param': None,
        'code': None,
    }

    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test')
    client = AsyncOpenAI(base_url='http://test/v1', api_key='unused', http_client=http_client, max_retries=0)
    with pytest.raises(InternalServerError, match='kaboom'):
        await client.responses.create(model='gpt-x', input='hi')
    await http_client.aclose()


async def test_invalid_request_body_returns_400():
    """A request body missing the required `input` field is rejected with an OpenAI error body."""
    agent = Agent(TestModel())
    app = agent.to_openai_responses()
    response = await post(app, {'model': 'gpt-x'})

    assert response.status_code == 400
    body = response.json()
    assert body['error']['type'] == 'invalid_request_error'
    assert body['error']['param'] == 'input'

    http_client = httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test')
    client = AsyncOpenAI(base_url='http://test/v1', api_key='unused', http_client=http_client, max_retries=0)
    with pytest.raises(BadRequestError):
        await client.responses.create(model='gpt-x')
    await http_client.aclose()


async def test_non_text_input_is_rejected():
    """Non-text input items like `input_image` are rejected: the endpoint is a text projection."""
    agent = Agent(TestModel())
    response = await post(
        agent.to_openai_responses(),
        {
            'model': 'gpt-x',
            'input': [
                {'role': 'user', 'content': [{'type': 'input_image', 'image_url': 'https://example.com/kiwi.jpg'}]}
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()['error']['type'] == 'invalid_request_error'


async def test_previous_response_id_is_rejected():
    """Server-side Responses state is not stored, so `previous_response_id` must fail explicitly."""
    agent = Agent(TestModel())
    response = await post(
        agent.to_openai_responses(),
        {'model': 'gpt-x', 'input': 'hi', 'previous_response_id': 'resp_previous'},
    )

    assert response.status_code == 400
    assert response.json()['error']['param'] == 'previous_response_id'


async def test_orphaned_function_call_output_is_rejected():
    """A `function_call_output` without a preceding `function_call` is invalid Responses history."""
    agent = Agent(TestModel())
    response = await post(
        agent.to_openai_responses(),
        {'model': 'gpt-x', 'input': [{'type': 'function_call_output', 'call_id': 'call_missing', 'output': 'sunny'}]},
    )

    assert response.status_code == 400
    body = response.json()
    assert body['error']['type'] == 'invalid_request_error'
    assert body['error']['param'] == 'input'
    assert 'call_missing' in body['error']['message']


async def test_empty_input_returns_400():
    """An empty `input` list reaches the agent as no prompt/history and is reported as a bad request."""
    agent = Agent(TestModel())
    response = await post(agent.to_openai_responses(), {'model': 'gpt-x', 'input': []})

    assert response.status_code == 400
    body = response.json()
    assert body['error']['type'] == 'invalid_request_error'
    assert 'No message history' in body['error']['message']


async def test_load_messages_round_trips_tool_calls():
    """`load_messages` reconstructs tool call/output pairs from replayed `input` items."""
    request = ResponsesRequest.model_validate(
        {
            'input': [
                {'role': 'system', 'content': 'be nice'},
                {'role': 'user', 'content': 'weather in Paris?'},
                {'type': 'function_call', 'call_id': 'c1', 'name': 'get_weather', 'arguments': '{"city":"Paris"}'},
                {
                    'type': 'function_call_output',
                    'call_id': 'c1',
                    'output': [
                        {'type': 'output_text', 'text': 'sunny'},
                        {'type': 'input_text', 'text': 'dry'},
                    ],
                },
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
                    ToolReturnPart(
                        tool_name='get_weather', content='sunny\ndry', tool_call_id='c1', timestamp=IsDatetime()
                    )
                ]
            ),
            ModelResponse(
                parts=[
                    TextPart(content='It is sunny in Paris.'),
                    ToolCallPart(tool_name='log', args='{}', tool_call_id='c2'),
                ],
                timestamp=IsDatetime(),
            ),
        ]
    )


async def test_multi_part_text_streaming_and_non_streaming():
    """Separate Pydantic AI text parts are surfaced as separate Responses content parts."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'It is sunny.'
        else:
            yield 'Let me check.'
            yield {0: DeltaToolCall(name='get_weather', json_args='{"city":"Paris"}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    events = await stream_events(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'weather?'})

    assert [
        (event['type'], event.get('content_index'))
        for event in events
        if event['type'] in {'response.content_part.added', 'response.output_text.done', 'response.content_part.done'}
    ] == snapshot(
        [
            ('response.content_part.added', 0),
            ('response.output_text.done', 0),
            ('response.content_part.done', 0),
            ('response.content_part.added', 1),
            ('response.output_text.done', 1),
            ('response.content_part.done', 1),
        ]
    )
    completed = Response.model_validate(events[-1]['response'])
    assert completed.output_text == 'Let me check.It is sunny.'
    message = completed.output[0]
    assert isinstance(message, ResponseOutputMessage)
    assert [part.text for part in message.content if part.type == 'output_text'] == snapshot(
        ['Let me check.', 'It is sunny.']
    )

    response = await post(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'weather?'})
    parsed = Response.model_validate(response.json())
    assert parsed.output_text == 'Let me check.It is sunny.'
    parsed_message = parsed.output[0]
    assert isinstance(parsed_message, ResponseOutputMessage)
    assert [part.text for part in parsed_message.content if part.type == 'output_text'] == snapshot(
        ['Let me check.', 'It is sunny.']
    )


async def test_trailing_assistant_message_falls_back_to_result_output():
    """A run that short-circuits from a trailing assistant message still returns that text."""
    agent = Agent(TestModel())
    response = await post(
        agent.to_openai_responses(),
        {'model': 'gpt-x', 'input': [{'role': 'assistant', 'content': 'previous answer'}]},
    )

    parsed = Response.model_validate(response.json())
    assert parsed.output_text == 'previous answer'


async def test_structured_output_agent_returns_empty_output():
    """Agents with a non-text `output_type` produce an empty `output`: the endpoint is a text projection."""
    agent = Agent(TestModel(), output_type=int)
    response = await post(agent.to_openai_responses(), {'model': 'gpt-x', 'input': 'how many?'})

    parsed = Response.model_validate(response.json())
    assert parsed.status == 'completed'
    assert parsed.output == []
    assert parsed.output_text == ''


async def test_model_parameter_supplies_agent_model():
    """The `model=` app parameter can serve an agent with no configured model."""
    agent = Agent()
    response = await post(
        agent.to_openai_responses(model=TestModel(model_name='served-test')),
        {'model': 'client-model', 'input': 'hi'},
    )

    parsed = Response.model_validate(response.json())
    assert parsed.output_text
    assert parsed.model == 'served-test'


async def test_model_echo_uses_served_model_name():
    """The response echoes the served model rather than the client-requested model alias."""
    agent = Agent(TestModel(model_name='served-model'))
    response = await post(agent.to_openai_responses(), {'model': 'client-model', 'input': 'hi'})

    assert Response.model_validate(response.json()).model == 'served-model'


async def test_model_echo_uses_served_model_string():
    """A string `model=` app parameter is echoed as the served model name."""
    agent = Agent()
    response = await post(agent.to_openai_responses(model='test'), {'model': 'client-model', 'input': 'hi'})

    assert Response.model_validate(response.json()).model == 'test'


async def test_model_echo_falls_back_to_requested_model():
    """With no served model to report, the response echoes the client's `model` (or empty string)."""
    agent = Agent()

    # The run itself fails (no model anywhere), but the request's model is still echoed in
    # the streaming preamble events emitted before the failure.
    events = await stream_events(agent.to_openai_responses(), {'model': 'client-model', 'input': 'hi'})
    assert events[0]['response']['model'] == 'client-model'
    assert events[-1]['type'] == 'response.failed'

    events = await stream_events(agent.to_openai_responses(), {'input': 'hi'})
    assert events[0]['response']['model'] == ''


async def test_handle_request_supports_per_request_deps():
    """A custom route can pass request-specific `deps` to the handler."""
    captured_deps: list[str] = []

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'done'
        else:
            yield {0: DeltaToolCall(name='capture_dep', json_args='{}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=stream), deps_type=str)

    @agent.tool
    def capture_dep(ctx: RunContext[str]) -> str:
        captured_deps.append(ctx.deps)
        return ctx.deps

    async def route(request: Request) -> StarletteResponse:
        return await handle_openai_responses_request(request, agent, deps=request.headers['x-deps'])

    app = Starlette(routes=[Route('/v1/responses', route, methods=['POST'])])
    response = await post(
        app, {'model': 'gpt-x', 'input': 'hi'}, path='/v1/responses', headers={'x-deps': 'per-request'}
    )

    assert response.status_code == 200
    assert captured_deps == ['per-request']


async def test_usage_limits_on_to_openai_responses():
    """Fixed-config apps can enforce usage limits across the server-side tool loop."""

    async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            yield 'done'
        else:
            yield {0: DeltaToolCall(name='get_weather', json_args='{"city":"Paris"}', tool_call_id='call_1')}

    agent = Agent(FunctionModel(stream_function=stream))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    response = await post(
        agent.to_openai_responses(usage_limits=UsageLimits(request_limit=1)),
        {'model': 'gpt-x', 'input': 'weather?'},
    )

    assert response.status_code == 500
    body = response.json()
    assert body['error']['type'] == 'server_error'
    assert 'request_limit' in body['error']['message']


async def test_custom_path():
    """The app can mount the Responses route at a custom path."""
    agent = Agent(TestModel(custom_output_text='custom path'))
    response = await post(
        agent.to_openai_responses(path='/custom/responses'),
        {'model': 'gpt-x', 'input': 'hi'},
        path='/custom/responses',
    )

    assert response.status_code == 200
    assert Response.model_validate(response.json()).output_text == 'custom path'


def test_usage_maps_reasoning_tokens():
    """Responses usage details preserve reasoning token counts."""
    usage = build_usage(RunUsage(input_tokens=1, output_tokens=2, details={'reasoning_tokens': 3}))

    assert usage.output_tokens_details is not None
    assert usage.output_tokens_details.reasoning_tokens == 3


async def test_openai_sdk_client_round_trip():
    """The real `openai` SDK can call the endpoint for both non-streaming and streaming requests."""
    agent = Agent(TestModel(custom_output_text='Hello from Pydantic AI.'))
    transport = httpx.ASGITransport(agent.to_openai_responses())
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

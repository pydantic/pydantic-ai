"""A streamed response that ends before the provider's terminal event is recorded as `'incomplete'`.

A provider can't be made to end a stream early on demand, so each case serves its stream through a mocked
transport: once cut off cleanly after the first text delta, and once with the events that end it.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, FinishReason, ModelResponse, ModelResponseState
from pydantic_ai.models import Model
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import ExternalToolset

from ..conftest import try_import

with try_import() as anthropic_available:
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


def _sse(event: str, data: dict[str, Any]) -> str:
    return f'event: {event}\ndata: {json.dumps(data)}\n\n'


def _anthropic(http_client: httpx2.AsyncClient) -> Model:
    return AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='test', http_client=http_client))


def _google(http_client: httpx2.AsyncClient) -> Model:
    return GoogleModel(
        'gemini-2.5-flash',
        provider=GoogleProvider(api_key='test', http_client=http_client, base_url='http://localhost'),
    )


def _openai_responses(http_client: httpx2.AsyncClient) -> Model:
    return OpenAIResponsesModel('gpt-5', provider=OpenAIProvider(api_key='test', http_client=http_client))


def _openai_response(status: str, output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'id': 'resp_1',
        'object': 'response',
        'created_at': 1,
        'status': status,
        'model': 'gpt-5',
        'output': output,
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
    }


_ANTHROPIC_EVENTS = (
    _sse(
        'message_start',
        {
            'type': 'message_start',
            'message': {
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'model': 'claude-sonnet-4-5',
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 9, 'output_tokens': 1},
            },
        },
    ),
    _sse(
        'content_block_start',
        {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
    ),
    _sse(
        'content_block_delta',
        {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'Hello, th'}},
    ),
)
_ANTHROPIC_CONTENT_BLOCK_STOP = _sse('content_block_stop', {'type': 'content_block_stop', 'index': 0})
_ANTHROPIC_MESSAGE_DELTA = _sse(
    'message_delta',
    {
        'type': 'message_delta',
        'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
        'usage': {'output_tokens': 4},
    },
)
_ANTHROPIC_MESSAGE_STOP = _sse('message_stop', {'type': 'message_stop'})

_GOOGLE_TEXT_CHUNK: dict[str, Any] = {
    'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Hello, th'}]}, 'index': 0}],
    'modelVersion': 'gemini-2.5-flash',
    'responseId': 'resp_1',
}

_OPENAI_MESSAGE: dict[str, Any] = {
    'id': 'msg_1',
    'type': 'message',
    'role': 'assistant',
    'status': 'completed',
    'content': [{'type': 'output_text', 'text': 'Hello, th', 'annotations': []}],
}


@dataclass(frozen=True)
class Case:
    id: str
    model: Callable[[httpx2.AsyncClient], Model]
    events: tuple[str, ...]
    """The stream up to and including its first text delta."""
    terminal_events: tuple[str, ...]
    """The events that end the stream."""
    marks: tuple[pytest.MarkDecorator, ...] = ()


CASES = [
    Case(
        id='anthropic',
        model=_anthropic,
        events=_ANTHROPIC_EVENTS,
        terminal_events=(_ANTHROPIC_CONTENT_BLOCK_STOP, _ANTHROPIC_MESSAGE_DELTA, _ANTHROPIC_MESSAGE_STOP),
        marks=(pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),),
    ),
    Case(
        id='google',
        model=_google,
        events=(f'data: {json.dumps(_GOOGLE_TEXT_CHUNK)}\r\n\r\n',),
        terminal_events=(
            'data: '
            + json.dumps(
                {
                    'candidates': [{'finishReason': 'STOP', 'index': 0}],
                    'usageMetadata': {'promptTokenCount': 9, 'candidatesTokenCount': 4, 'totalTokenCount': 13},
                    'modelVersion': 'gemini-2.5-flash',
                    'responseId': 'resp_1',
                }
            )
            + '\r\n\r\n',
        ),
        marks=(pytest.mark.skipif(not google_available(), reason='google-genai not installed'),),
    ),
    Case(
        id='openai-responses',
        model=_openai_responses,
        events=(
            _sse(
                'response.created',
                {
                    'type': 'response.created',
                    'sequence_number': 0,
                    'response': _openai_response('in_progress', []),
                },
            ),
            _sse(
                'response.output_item.added',
                {
                    'type': 'response.output_item.added',
                    'sequence_number': 1,
                    'output_index': 0,
                    'item': {**_OPENAI_MESSAGE, 'status': 'in_progress', 'content': []},
                },
            ),
            _sse(
                'response.content_part.added',
                {
                    'type': 'response.content_part.added',
                    'sequence_number': 2,
                    'item_id': 'msg_1',
                    'output_index': 0,
                    'content_index': 0,
                    'part': {'type': 'output_text', 'text': '', 'annotations': []},
                },
            ),
            _sse(
                'response.output_text.delta',
                {
                    'type': 'response.output_text.delta',
                    'sequence_number': 3,
                    'item_id': 'msg_1',
                    'output_index': 0,
                    'content_index': 0,
                    'delta': 'Hello, th',
                    'logprobs': [],
                },
            ),
        ),
        terminal_events=(
            _sse(
                'response.output_item.done',
                {'type': 'response.output_item.done', 'sequence_number': 4, 'output_index': 0, 'item': _OPENAI_MESSAGE},
            ),
            _sse(
                'response.completed',
                {
                    'type': 'response.completed',
                    'sequence_number': 5,
                    'response': _openai_response('completed', [_OPENAI_MESSAGE]),
                },
            ),
        ),
        marks=(pytest.mark.skipif(not openai_available(), reason='openai not installed'),),
    ),
]


async def _stream(
    model: Callable[[httpx2.AsyncClient], Model], events: Sequence[str]
) -> tuple[ModelResponse, ModelResponse]:
    """Stream `events` through `Agent.run_stream`; return the last `stream_response()` and recorded responses."""
    body = ''.join(events).encode()
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=body)
    )
    async with httpx2.AsyncClient(transport=transport) as http_client:
        agent = Agent(model(http_client))
        async with agent.run_stream('Say hello.') as result:
            snapshots = [response async for response in result.stream_response(debounce_by=None)]
        recorded = result.all_messages()[-1]
    assert isinstance(recorded, ModelResponse)
    return snapshots[-1], recorded


@pytest.mark.parametrize(
    ('ends', 'expected_state', 'expected_finish_reason'),
    [('cut-off', 'incomplete', None), ('terminal-event', 'complete', 'stop')],
    ids=['cut-off', 'terminal-event'],
)
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id, marks=case.marks) for case in CASES])
async def test_stream_state(
    allow_model_requests: None,
    case: Case,
    ends: str,
    expected_state: ModelResponseState,
    expected_finish_reason: FinishReason | None,
):
    """The final `stream_response()` snapshot and the recorded response report whether the stream ended properly."""
    events = case.events if ends == 'cut-off' else case.events + case.terminal_events
    final, recorded = await _stream(case.model, events)

    assert (final.text, final.state, final.finish_reason) == ('Hello, th', expected_state, expected_finish_reason)
    assert (recorded.state, recorded.finish_reason) == (expected_state, expected_finish_reason)


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_message_stop_without_stop_reason(allow_model_requests: None):
    """`message_stop` ends an Anthropic stream even when no `message_delta` reported a stop reason."""
    final, recorded = await _stream(
        _anthropic, (*_ANTHROPIC_EVENTS, _ANTHROPIC_CONTENT_BLOCK_STOP, _ANTHROPIC_MESSAGE_STOP)
    )

    assert (final.state, final.finish_reason) == ('complete', None)
    assert (recorded.state, recorded.finish_reason) == ('complete', None)


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_stream_response_again_after_cut_off_stream(allow_model_requests: None):
    """Calling `stream_response()` again after a cut-off stream ended yields its final snapshot once."""
    body = ''.join(_ANTHROPIC_EVENTS).encode()
    transport = httpx2.MockTransport(
        lambda _: httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=body)
    )
    async with httpx2.AsyncClient(transport=transport) as http_client:
        async with Agent(_anthropic(http_client)).run_stream('Say hello.') as result:
            [response async for response in result.stream_response(debounce_by=None)]
            again = [response async for response in result.stream_response(debounce_by=None)]

    assert [(response.text, response.state) for response in again] == [('Hello, th', 'incomplete')]


def _anthropic_tool_use(index: int, tool_call_id: str, tool_name: str, args: str) -> tuple[str, ...]:
    return (
        _sse(
            'content_block_start',
            {
                'type': 'content_block_start',
                'index': index,
                'content_block': {'type': 'tool_use', 'id': tool_call_id, 'name': tool_name, 'input': {}},
            },
        ),
        _sse(
            'content_block_delta',
            {
                'type': 'content_block_delta',
                'index': index,
                'delta': {'type': 'input_json_delta', 'partial_json': args},
            },
        ),
        _sse('content_block_stop', {'type': 'content_block_stop', 'index': index}),
    )


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_deferred_tool_pause_on_cut_off_stream_resumes_as_a_pause(allow_model_requests: None):
    """A run that paused for a deferred call of a cut-off response resumes that pause, as it does a complete one.

    Run again without a prompt, the deferred call pauses the run again; with a new prompt, the unprocessed call is
    refused. Neither sends another request.
    """
    body = ''.join(
        (
            _ANTHROPIC_EVENTS[0],
            *_anthropic_tool_use(0, 'call_create', 'create_file', '{"path": "a.py"}'),
            *_anthropic_tool_use(1, 'call_delete', 'delete_file', '{"path": "b.py"}'),
        )
    ).encode()
    requests: list[httpx2.Request] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=body)

    async def drain(_: RunContext[object], events: AsyncIterable[AgentStreamEvent]) -> None:
        async for _event in events:
            pass

    delete_file = ToolDefinition(
        name='delete_file',
        parameters_json_schema={'type': 'object', 'properties': {'path': {'type': 'string'}}, 'required': ['path']},
    )
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handle)) as http_client:
        agent = Agent(
            _anthropic(http_client),
            output_type=[str, DeferredToolRequests],
            toolsets=[ExternalToolset([delete_file])],
        )

        @agent.tool_plain
        def create_file(path: str) -> str:
            return f'Created {path}'

        paused = await agent.run('Create a.py and delete b.py.', event_stream_handler=drain)
        history = paused.all_messages()
        resumed = await agent.run(message_history=history)
        with pytest.raises(UserError, match='unprocessed tool calls'):
            await agent.run('Next', message_history=history)

    response = history[1]
    assert isinstance(response, ModelResponse)
    assert response.state == 'incomplete'
    assert isinstance(paused.output, DeferredToolRequests)
    assert [call.tool_name for call in paused.output.calls] == ['delete_file']
    assert resumed.output == paused.output
    assert len(requests) == 1

"""Tests for Anthropic message threads (`anthropic_thread`).

These run against a mock HTTP transport rather than cassettes: recording needs an Anthropic API key, and the
failures that make a request replay (`thread_not_found` after a thread expires, `thread_already_continued`) can't be
triggered reliably against the live API. Each mock response carries the request shape Anthropic documents at
https://platform.claude.com/docs/en/build-with-claude/message-threads, and the tests assert the request bodies
the model sent, which a cassette matcher wouldn't check.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent, ModelMessage, ModelRequest, ModelResponse, NativeToolReturnPart, TextPart, ToolReturn
from pydantic_ai.exceptions import ModelHTTPError

from ...conftest import try_import

with try_import() as imports_successful:
    from anthropic import AsyncAnthropic

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
]

_BETA = 'message-threads-2026-08-12'
_SETTINGS: AnthropicModelSettings = {'max_tokens': 256, 'anthropic_thread': True}


@dataclass
class _Api:
    """A mock Messages API that records each request and answers from `replies`, then with text."""

    replies: list[httpx2.Response | dict[str, Any]] = field(default_factory=list[httpx2.Response | dict[str, Any]])
    bodies: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    betas: list[str] = field(default_factory=list[str])
    stream: bool = False

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        self.bodies.append(json.loads(request.content))
        self.betas.append(request.headers.get('anthropic-beta', ''))
        reply = self.replies.pop(0) if self.replies else {'type': 'text', 'text': 'OK'}
        if isinstance(reply, httpx2.Response):
            return reply
        message = {
            'id': f'msg_{len(self.bodies)}',
            'type': 'message',
            'role': 'assistant',
            'model': 'claude-sonnet-4-5',
            'usage': {'input_tokens': 5, 'output_tokens': 1},
        }
        stop_reason = 'tool_use' if reply['type'] == 'tool_use' else 'end_turn'
        if not self.stream:
            return httpx2.Response(200, json={**message, 'content': [reply], 'stop_reason': stop_reason})
        if reply['type'] == 'text':
            start: dict[str, Any] = {'type': 'text', 'text': ''}
            delta: dict[str, Any] = {'type': 'text_delta', 'text': reply['text']}
        else:
            start = {**reply, 'input': {}}
            delta = {'type': 'input_json_delta', 'partial_json': json.dumps(reply['input'])}
        events: list[dict[str, Any]] = [
            {'type': 'message_start', 'message': {**message, 'content': [], 'stop_reason': None}},
            {'type': 'content_block_start', 'index': 0, 'content_block': start},
            {'type': 'content_block_delta', 'index': 0, 'delta': delta},
            {'type': 'content_block_stop', 'index': 0},
            {'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {}},
            {'type': 'message_stop'},
        ]
        body = ''.join(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n' for event in events)
        return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, content=body.encode())


@asynccontextmanager
async def _model(api: _Api) -> AsyncGenerator[AnthropicModel]:
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle)) as http_client:
        client = AsyncAnthropic(api_key='x', http_client=http_client)
        yield AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))


def _thread_error(status: int, error_code: str) -> httpx2.Response:
    error_type = 'not_found_error' if status == 404 else 'invalid_request_error'
    error = {'type': error_type, 'message': 'thread', 'details': {'error_code': error_code}}
    return httpx2.Response(status, json={'type': 'error', 'error': error})


def _user_texts(body: dict[str, Any]) -> list[str]:
    """The text of every message in a request body, in order."""
    texts: list[str] = []
    for message in body['messages']:
        content: str | list[dict[str, Any]] = message['content']
        blocks = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
        texts += [block['text'] for block in blocks if block['type'] == 'text']
    return texts


def _instructions(body: dict[str, Any]) -> str:
    """The instructions in a request's `system`, after the framework's own leading blocks."""
    system = body['system']
    return system if isinstance(system, str) else system[-1]['text']


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f'Sunny in {city}'


async def _run(
    agent: Agent[None, str], prompt: str, history: Sequence[ModelMessage], stream: bool
) -> list[ModelMessage]:
    if not stream:
        return (await agent.run(prompt, message_history=history)).all_messages()
    async with agent.run_stream(prompt, message_history=history) as result:
        await result.get_output()
    return result.all_messages()


@pytest.mark.parametrize('stream', [False, True])
async def test_create_then_continue(allow_model_requests: None, stream: bool):
    """The first request creates a thread; the next run sends only its new message, without `system`."""
    api = _Api(stream=stream)
    async with _model(api) as model:
        agent = Agent(model, instructions='Be brief.', model_settings=_SETTINGS)
        first = await _run(agent, 'Hi', [], stream)
        await _run(agent, 'Again', first, stream)

    create, cont = api.bodies
    assert create['thread'] == {'type': 'create'}
    assert _instructions(create) == 'Be brief.'
    assert cont['thread'] == {'type': 'continue', 'previous_message_id': 'msg_1'}
    assert 'system' not in cont and 'tools' not in cont
    assert _user_texts(cont) == ['Again']
    assert all(_BETA in betas for betas in api.betas)


async def test_continue_within_a_tool_call_loop(allow_model_requests: None):
    """Inside one run, the request answering a tool call sends only the tool result."""
    api = _Api(replies=[{'type': 'tool_use', 'id': 'toolu_1', 'name': 'get_weather', 'input': {'city': 'Paris'}}])
    async with _model(api) as model:
        await Agent(model, tools=[get_weather], model_settings=_SETTINGS).run('Weather in Paris?')

    create, cont = api.bodies
    assert create['thread'] == {'type': 'create'} and create['tools']
    assert cont['thread'] == {'type': 'continue', 'previous_message_id': 'msg_1'}
    assert 'tools' not in cont
    [message] = cont['messages']
    assert [block['type'] for block in message['content']] == ['tool_result']


@pytest.mark.parametrize(
    ('status', 'error_code'),
    [(404, 'thread_not_found'), (400, 'thread_already_continued'), (400, 'thread_continue_in_flight')],
)
@pytest.mark.parametrize('stream', [False, True])
async def test_replay_when_thread_cannot_be_continued(
    allow_model_requests: None, status: int, error_code: str, stream: bool
):
    """A continue request the API can't serve is sent again as a create request with the full conversation."""
    api = _Api(replies=[{'type': 'text', 'text': 'Hello'}, _thread_error(status, error_code)], stream=stream)
    async with _model(api) as model:
        agent = Agent(model, instructions='Be brief.', model_settings=_SETTINGS)
        first = await _run(agent, 'Hi', [], stream)
        second = await _run(agent, 'Again', first, stream)

    _, failed, replay = api.bodies
    assert failed['thread']['type'] == 'continue'
    assert replay['thread'] == {'type': 'create'}
    assert _instructions(replay) == 'Be brief.'
    assert _user_texts(replay) == ['Hi', 'Hello', 'Again']
    # The replayed response starts the new thread, so the run after it continues from it.
    response = second[-1]
    assert isinstance(response, ModelResponse)
    assert response.provider_response_id == 'msg_3'
    assert response.provider_details and 'thread_fingerprint' in response.provider_details


async def test_unrelated_404_is_not_replayed(allow_model_requests: None):
    """A 404 without a thread error code, like an unknown model, reaches the caller."""
    not_found = httpx2.Response(404, json={'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model'}})
    api = _Api(replies=[{'type': 'text', 'text': 'Hello'}, not_found])
    async with _model(api) as model:
        agent = Agent(model, model_settings=_SETTINGS)
        first = await agent.run('Hi')
        with pytest.raises(ModelHTTPError):
            await agent.run('Again', message_history=first.all_messages())
    assert len(api.bodies) == 2


async def test_changed_instructions_start_a_new_thread(allow_model_requests: None):
    """A thread keeps its `system`, so a run with different instructions creates a new thread."""
    api = _Api()
    async with _model(api) as model:
        first = await Agent(model, instructions='Be brief.', model_settings=_SETTINGS).run('Hi')
        await Agent(model, instructions='Be verbose.', model_settings=_SETTINGS).run(
            'Again', message_history=first.all_messages()
        )

    assert api.bodies[1]['thread'] == {'type': 'create'}
    assert _instructions(api.bodies[1]) == 'Be verbose.'


async def test_response_from_another_model_breaks_the_chain(allow_model_requests: None):
    """A later response that wasn't stored in the thread, say from `FallbackModel`, means a create request."""
    api = _Api()
    async with _model(api) as model:
        agent = Agent(model, model_settings=_SETTINGS)
        first = await agent.run('Hi')
        history = [
            *first.all_messages(),
            ModelRequest.user_text_prompt('Next'),
            ModelResponse(parts=[TextPart('From another model')], provider_name='openai', provider_response_id='r'),
        ]
        await agent.run('Again', message_history=history)

    assert api.bodies[1]['thread'] == {'type': 'create'}
    assert _user_texts(api.bodies[1]) == ['Hi', 'OK', 'Next', 'From another model', 'Again']


async def test_off_by_default(allow_model_requests: None):
    """Without the setting, no thread is sent and responses record no fingerprint."""
    api = _Api()
    async with _model(api) as model:
        result = await Agent(model, model_settings={'max_tokens': 256}).run('Hi')

    assert 'thread' not in api.bodies[0]
    assert _BETA not in api.betas[0]
    response = result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert 'thread_fingerprint' not in (response.provider_details or {})


async def test_edited_history_starts_a_new_thread(allow_model_requests: None):
    """The thread holds the history as it was sent, so a request after an edit to it creates a new thread."""
    api = _Api(replies=[{'type': 'text', 'text': 'Hello'}])
    async with _model(api) as model:
        agent = Agent(model, model_settings=_SETTINGS)
        first = await agent.run('My card number is 1234')
        redacted = [ModelRequest.user_text_prompt('My card number is [redacted]'), *first.all_messages()[1:]]
        await agent.run('Again', message_history=redacted)

    assert api.bodies[1]['thread'] == {'type': 'create'}
    assert _user_texts(api.bodies[1]) == ['My card number is [redacted]', 'Hello', 'Again']


async def test_refusal_starts_a_new_thread(allow_model_requests: None):
    """Anthropic may already hold a refused turn, so the next request doesn't continue from it."""
    refusal = {
        'id': 'msg_refused',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-5',
        'content': [{'type': 'text', 'text': 'No.'}],
        'stop_reason': 'refusal',
        'usage': {'input_tokens': 5, 'output_tokens': 1},
    }
    api = _Api(replies=[httpx2.Response(200, json=refusal)])
    async with _model(api) as model:
        agent = Agent(model, model_settings=_SETTINGS)
        first = await agent.run('Hi')
        await agent.run('Rephrased', message_history=first.all_messages())

    assert api.bodies[1]['thread'] == {'type': 'create'}


async def test_changed_skills_start_a_new_thread(allow_model_requests: None):
    """`container.skills` is fixed for a thread, so changing it creates a new thread."""
    api = _Api()

    def skills(skill_id: str) -> AnthropicModelSettings:
        return {**_SETTINGS, 'anthropic_container': {'skills': [{'type': 'anthropic', 'skill_id': skill_id}]}}

    async with _model(api) as model:
        agent = Agent(model)
        first = await agent.run('Hi', model_settings=skills('pdf'))
        await agent.run('Again', message_history=first.all_messages(), model_settings=skills('xlsx'))

    assert api.bodies[1]['thread'] == {'type': 'create'}


async def test_unsupported_request_is_sent_without_a_thread(allow_model_requests: None):
    """A request using something threads don't support is sent again without `thread`, and isn't continued from."""
    api = _Api(replies=[_thread_error(400, 'thread_unsupported_request')])
    async with _model(api) as model:
        agent = Agent(model, model_settings=_SETTINGS)
        first = await agent.run('Hi')
        await agent.run('Again', message_history=first.all_messages())

    rejected, unthreaded, second = api.bodies
    assert rejected['thread'] == {'type': 'create'}
    assert 'thread' not in unthreaded
    assert second['thread'] == {'type': 'create'}
    response = first.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert 'thread_fingerprint' not in (response.provider_details or {})


def weather_with_metadata(city: str) -> ToolReturn:
    """Get the weather for a city."""
    return ToolReturn(return_value=f'Sunny in {city}', metadata={'handle': object()})


async def test_application_metadata_is_not_digested(allow_model_requests: None):
    """`metadata` is never sent, so a value that isn't JSON, or a change to it, doesn't stop the thread continuing."""
    api = _Api(
        replies=[{'type': 'tool_use', 'id': 'toolu_1', 'name': 'weather_with_metadata', 'input': {'city': 'Paris'}}]
    )
    async with _model(api) as model:
        agent = Agent(model, tools=[weather_with_metadata], model_settings=_SETTINGS)
        first = await agent.run('Weather in Paris?')
        history = first.all_messages()
        first_request = history[0]
        assert isinstance(first_request, ModelRequest)
        history[0] = replace(first_request, metadata={'tag': 'added later'})
        await agent.run('Thanks', message_history=history)

    assert [body['thread']['type'] for body in api.bodies] == ['create', 'continue', 'continue']


def test_response_part_metadata_is_not_digested():
    """Response parts' `metadata` is left out of the digest too, even when it isn't JSON.

    A unit test, as a native tool return with application metadata needs a server tool turn to reach the history.
    """
    from pydantic_ai.models.anthropic import _thread_key  # pyright: ignore[reportPrivateUsage]

    part = NativeToolReturnPart(tool_name='web_search', content='ok', provider_name='anthropic')
    request = ModelRequest.user_text_prompt('Hi')
    response = ModelResponse(parts=[part], provider_name='anthropic')

    def history(metadata: dict[str, Any] | None) -> list[ModelMessage]:
        return [request, replace(response, parts=[replace(part, metadata=metadata)])]

    assert _thread_key('f', history({'handle': object()})) == _thread_key('f', history(None))
